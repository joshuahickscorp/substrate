# SCALING

What to flip when the Mac Studio (Apple Silicon) or a rented CUDA box arrives. Nothing here is
a code change: the system is built cached-latent-first and device-flag-driven on purpose, so
scale is config plus two scripts. The frozen substrate never trains, so scaling means a bigger
frozen encoder, real latent caches, larger batches/datasets, and unlocking the tiers that need
more compute or an environment. Read APPLE_SILICON.md (the MPS-first story), ARCHITECTURE.md
(device boundary, frozen invariant), and EXPERIMENTS.md (tier tags) alongside this.

## The flips, in order

### 1. Device: STAY on mps (the Studio is Apple Silicon)
The Mac Studio is an M-series Apple Silicon machine, NOT a CUDA box: more GPU cores and far
more unified memory than the laptop, same Metal backend. So the headline scale-up is NOT a
device flip at all: keep `device=mps` and the same code runs, just bigger. Everything
device-touching goes through `devices.resolve(cfg.device.kind)`; `auto` already prefers mps on
Apple Silicon. `configs/device/mps.yaml` carries the Apple-Silicon knobs (`amp: true` fp16,
`pin_memory: false` unified memory, `allow_cpu_fallback: true`). On the Studio, raise batch and
cache sizes to use the larger unified memory, and re-test the 64-frame encoder forward on Metal
(the laptop hits an MPS-compiler limit on it; more GPU cores should lift it; see APPLE_SILICON.md).

`device=cuda` is the rented-box path used ONLY for Tier R environment rollouts (E5/E10), which
Metal cannot cover cheaply. `configs/device/cuda.yaml` carries `amp: true`, `pin_memory: true`,
`num_workers: 8`, `allow_cpu_fallback: false`. Select it only for those legs.

### 2. Larger encoder: vjepa2_vitl -> vjepa2_vitg
Default substrate is `vjepa2_vitl_fpc64_256` (ViT-L, embed_dim 1024, 256px). On the Studio,
go bigger:

```
encoder=vjepa2_vitg     # ViT-g, embed_dim 1408, 384px, 64 frames per clip
```

The shell reads `latent_dim` off the selected encoder config, so the predictor and heads
resize automatically. No shell code changes. The dense 2.1 encoders
(`vjepa21_vitb`, `vjepa21_vitl`, both `dense: true`, per-patch tokens) exist for the
`2.1-only` experiments (E6); see the tier table below.

### 3. Real latent caching: synthetic -> real V-JEPA
Today the substrate defaults to frozen-random + the synthetic latent generator (the grids run
on these, tagged provisional). Real V-JEPA 2 ViT-L weights load and a real-encoder cache has
been validated (see STATUS.md and the retrospective ledger). The remaining step for REAL natural-video science is
just dropping real clips and running the video cache:

```
uv pip install -e ".[encoder,video]"    # transformers + huggingface-hub + torchvision (decode)
# drop class-foldered clips under <dir>/<class>/*.mp4, then:
.venv/bin/python scripts/cache_video.py +source=<dir> encoder=vjepa2_vitl_fpc64_256 device=mps +total=N
```

`substrate/video.py` decodes + preprocesses (frame-sample, resize, ImageNet-normalize) to
V-JEPA's `[B,64,3,256,256]`; `cache_video.py` runs the frozen encoder once and writes the
memmap store (`substrate/cache.py`, `substrate/latent_store.py`). Because the encoder is frozen
the cache never goes stale. Every experiment then reads the real cache with no change; the
`backend` field flips `frozen_random` -> `vjepa_hf`, your proof the latents are real. The
structured-synthetic real-encoder cache (`scripts/cache_real_encoder.py`, no video files
needed) is the bridge until natural clips are on disk.

Verified encoder ids (HF, probed 2026-06): real and present are `vjepa2-vitl-fpc64-256` (1024),
`vjepa2-vith-fpc64-256` (1280), `vjepa2-vitg-fpc64-384` (1408). V-JEPA 2.1 dense is NOT yet on
HF under any verified id, so the `vjepa21_*` configs are placeholders (`available: false`) and
the 2.1-only experiments (E6 dense) stay deferred until 2.1 ships.

### 4. Batch / dataset scale-up knobs
All overridable as CLI dotlist (`group.key=value`), no edits to source:

- `shell.buffer.capacity=<N>`: bigger latent hippocampus (laptop runs small; Studio can
  hold the real stream).
- `cfg.device.num_workers` (via `device=cuda`): real dataloader parallelism.
- experiment params under `cfg.experiment.*`: samples_per_task, n_tasks, classes_per_task,
  epochs_per_task, batch_size, replay_batch (the toy scale baked into scaffolds is seconds
  on mps; scale these up on cuda).
- seeds: 5 minimum for the real run (3 to 5 for E10); run the determinism sanity loop first
  (`diagnostics/determinism`) and trust no cross-condition delta inside the Metal spread.
  On cuda determinism is tighter, so the tolerances loosen back toward exact.

### 5. AMP / precision
`device=cuda` turns on `amp: true` and tf32. Nothing else to do. The frozen encoder is
already `no_grad`, so AMP only touches the trainable shell.

## What unlocks per tier

Tier tags come straight from EXPERIMENTS.md. Each experiment carries one.

| Tier | Means | Needs | Unlocks |
|---|---|---|---|
| `cpu-now` | runs now, laptop, cached latents | nothing extra | E1, E2, E3, E4, E7, E8, E9, I4 (and E5 data-selection variant) |
| `gpu-later` | needs the Studio for scale/speed | device=cuda (+ vitg) | E6 (gpu side), E7 speedup claim, E10 (gpu side) |
| `env-later` | needs an environment + rollouts | an env adapter + rented CUDA | E5 rollout variant, E10 capstone |
| `2.1-only` | needs V-JEPA 2.1 dense weights | encoder=vjepa21_vitl + real cache | E6 relational-map dense-vs-pooled |

Reading: the whole `cpu-now` column runs today and is what the laptop session built and
tested. `gpu-later` is the same experiments at real scale (separate the representational
claim from the compute claim: an experiment that ties at toy scale and wins at full scale is
a compute result, taxonomy entry 9). `env-later` is blocked on an environment, not just a
GPU: E5's curiosity-as-self-curriculum rollout variant and the E10 open-ended capstone both
need an interactive env to act in. `2.1-only` is E6, which needs the dense per-patch tokens
the 2.1 encoders emit (`dense: true`); on the pooled 2 encoders E6 has nothing to factorize.

## Rented-CUDA path (E5/E10 rollouts, Tier R legs)

The rollout-heavy work and the synthesized campaign's Tier R legs are queued disabled and
need a rented CUDA box plus a real environment, not just the Studio:

- E5 env-rollout variant and E10 capstone (env-later): provide the env adapter, then run on
  rented CUDA.
- Tier R campaign legs (rollout-heavy E5, E10, POET env-generation, cultural accumulation):
  they sit in the run queue with `enabled: false`. On the rented box, provide the env
  adapter and flip `enabled: true`, then run the queue.

Tier mapping (from DECISIONS.md, synthesized from Vol I Section 9 tractability): Tier C =
laptop-feasible cached-latent legs, Tier E = environment-needed, Tier R = rented-GPU /
lab-scale. C runs now; E and R wait for env + rented CUDA.

## First commands on the new machine

Mac Studio (Apple Silicon -- stay on mps, scale up the cached-latent bank). The ONE pipeline
surface drives the whole acquisition + run path; the raw scripts below it still work for hand
control:

```
uv venv --python 3.12 .venv
uv pip install -e ".[dev,ann,encoder,video,apple]"   # video=torchvision decode, apple=mlx (optional)
make test                                            # confirm green on the new box first
make doctor                                          # readiness (python/torch/mps/disk/video/hf/encoders/cache/config)
# --- the one Studio pipeline (plan/acquire/validate/cache/run/optimize/report) ---
python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900   # writes runs/studio_pipeline/latest/
# REVIEW runs/studio_pipeline/latest/license_ledger.md (resolve manual/blocked sources first), then:
python scripts/studio_pipeline.py acquire  --plan runs/studio_pipeline/latest/plan.json \
    --execute --budget-gb 900 --accept-license       # REAL downloads, gated + budgeted
python scripts/studio_pipeline.py validate --plan runs/studio_pipeline/latest/plan.json
python scripts/studio_pipeline.py cache    --plan runs/studio_pipeline/latest/plan.json --execute
python scripts/studio_pipeline.py run --gated --tiers C --full   # gated conveyor (gates are kill switches)
python scripts/studio_pipeline.py optimize --cache <cache_id>    # throughput lane (not science)
python scripts/studio_pipeline.py report
# --- raw scripts (hand control, still supported) ---
make diag                                            # determinism + diagnostics; set tolerances
# verify the 64-frame ViT-L forward runs on Metal here (it hangs the M3; more GPU cores should lift it):
.venv/bin/python scripts/cache_real_encoder.py device=mps +classes=2 +per_class=1   # smoke
.venv/bin/python scripts/cache_video.py +source=<dir> encoder=vjepa2_vitl_fpc64_256 device=mps +total=N
.venv/bin/python scripts/run_queue.py --tiers C --full   # full factorials (217 run-units; see cost projection)
# E6 (dense 2.1) waits for V-JEPA 2.1 to ship on HF (vjepa21_* configs are placeholders today).
# Tier R env rollouts (E5/E10) are the only cuda path: rent a GPU and `--tiers C,E,R --full`.
```

Current M3 Pro before the Studio arrives: `make local-max` (profile m3pro-local-max) does the
most real work that is SAFE here (plan, dry-run acquire, generate control corpora, validate,
tiny real cache, queue/cost audit, microbench, one gated leg, report) under hard kill switches
(10 GB download default / 25 GB cap, 2 GB fixtures, 128 clips, 60 GB free-disk floor, 90 min,
Tier C only). It never downloads heavy assets or starts a long sweep.

Rented CUDA (env rollouts, Tier R):

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev,ann,encoder]"
make test
# provide the env adapter, then enable the Tier R legs in the run queue and launch:
make queue-dry                                    # confirm the plan with no compute spent
.venv/bin/python scripts/run_queue.py             # after flipping enabled:true on Tier R legs
make accept                                        # end-to-end acceptance
```

Sanity order on any new box: `make test` green, then determinism to set tolerances, then
cache real latents once, then run. Never trust a cross-condition delta before the
determinism loop, never trust a downstream result before E1 the gate passes.

## Form

No em dashes or en dashes anywhere (commas, colons, parentheses only), per BLACKHOLE.md.
