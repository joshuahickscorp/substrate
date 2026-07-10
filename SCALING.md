# SCALING

What to flip when the Mac Studio (Apple Silicon) or a rented CUDA box arrives. Nothing here is
a code change: the system is built cached-latent-first and device-flag-driven on purpose, so
scale is config plus two scripts. The frozen substrate never trains, so scaling means a bigger
frozen encoder, real latent caches, larger batches/datasets, and unlocking the tiers that need
more compute or an environment. Read APPLE_SILICON.md (the MPS-first story), ARCHITECTURE.md
(device boundary, frozen invariant), and EXPERIMENTS.md (tier tags) alongside this.

> CURRENT LOCALIZATION (2026-07-10): encoder scale is no longer an off-device boundary. Pinned
> ViT-L, ViT-H, and ViT-g weights all load offline and complete supervised local CPU forwards;
> eight shared referents have been cached serially through every scale. Use the 180-minute
> `m3pro-local-max` envelope and one heavy model at a time. A larger machine buys throughput and
> corpus capacity only after a measured local rung, while natural-video rights, unpublished dense
> V-JEPA 2.1 weights, and interactive environments remain separate input blockers. This note
> supersedes lower historical wording that treats an encoder name or planning tier as hardware proof.

## The flips, in order

### 1. Device: measure CPU vs MPS per workload
The Mac Studio is an M-series Apple Silicon machine, NOT a CUDA box: more GPU cores and far
more unified memory than the laptop, same Metal backend. The headline scale-up is therefore
not an automatic device flip. Everything
device-touching goes through `devices.resolve(cfg.device.kind)`; `auto` already prefers mps on
Apple Silicon. `configs/device/mps.yaml` carries the Apple-Silicon knobs (`amp: true` fp16,
`pin_memory: false` unified memory, `allow_cpu_fallback: true`). On this M3 Pro, serial CPU is the
verified path for the large V-JEPA forwards. Re-benchmark MPS on any future host before selecting it.

`device=cuda` remains an optional rented-box throughput path. `configs/device/cuda.yaml` carries
`amp: true`, `pin_memory: true`, `num_workers: 8`, `allow_cpu_fallback: false`. An environment
dependency does not itself require CUDA: first supply the adapter and run a bounded local rung.

### 2. Larger encoder: vjepa2_vitl -> vjepa2_vitg
Default substrate is `vjepa2_vitl_fpc64_256` (ViT-L, embed_dim 1024, 256px). To run the largest
verified local scale serially:

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
| `cpu-now` | runs now, laptop, cached latents or bounded local trajectories | nothing extra | E1-E5 where registered, E7-E9, I4, and local action mechanics |
| `gpu-later` | legacy runnable enum, no current registry row uses it | no implied device | preserve only for backward compatibility |
| `env-later` | needs evidence or ecology beyond the bounded local adapter | rendered/substrate trajectories or a generated population ecology | E10 capstone, substrate-grounded CM10 |
| `2.1-only` | needs V-JEPA 2.1 dense weights | encoder=vjepa21_vitl + real cache | E6 relational-map dense-vs-pooled |

Reading: the whole `cpu-now` column runs today and is what the laptop session built and
tested. The retained `gpu-later` value is schema compatibility, not an active hardware label;
the current registry contains zero rows with it. `env-later` now means the bounded local adapter is
scientifically insufficient: E10 needs a generated population ecology and substrate-grounded CM10
needs rendered/citable action referents and exact controls. `2.1-only` is E6, which needs the dense per-patch tokens
the 2.1 encoders emit (`dense: true`); on the pooled 2 encoders E6 has nothing to factorize.

## Optional rented-CUDA throughput path (only after local evidence gates)

The synthesized campaign's historical Tier R legs remain queued disabled. The local adapter exists;
a rented CUDA box is only an optional higher-throughput rung after an exact row exceeds local bounds:

- E10 capstone (env-later): extend the existing bounded adapter with population/environment generation, run a bounded
  local rung, then move only a measured throughput remainder.
- Tier R campaign legs (E10, POET env-generation, cultural accumulation) sit in the run queue with
  `enabled: false`. First implement and measure a bounded local rung; only then consider enabling one.

Tier mapping (from DECISIONS.md, synthesized from Vol I Section 9 tractability): Tier C =
laptop-feasible cached-latent legs, Tier E = environment-needed, Tier R = rented-GPU /
lab-scale. C runs now; E and R wait for scientific gates and local measurements, not merely a rental.

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
(10 GB download default / 25 GB cap, 2 GB fixtures, 128 clips, a derived 40 GB free-disk floor
equal to 10 GB OS reserve + 25 GB maximum pending download + 5 GB working headroom, 180 min,
Tier C only). It never downloads heavy assets or starts a long sweep.

Rented CUDA (env rollouts, Tier R):

```
uv venv --python 3.12 .venv && uv pip install -e ".[dev,ann,encoder]"
make test
# after the local adapter, exact evidence, and scientific gates are satisfied, enable the Tier R legs:
make queue-dry                                    # confirm the plan with no compute spent
.venv/bin/python scripts/run_queue.py             # after flipping enabled:true on Tier R legs
make accept                                        # end-to-end acceptance
```

Sanity order on any new box: `make test` green, then determinism to set tolerances, then
cache real latents once, then run. Never trust a cross-condition delta before the
determinism loop, never trust a downstream result before E1 the gate passes.

## Form

No em dashes or en dashes anywhere (commas, colons, parentheses only), per BLACKHOLE.md.
