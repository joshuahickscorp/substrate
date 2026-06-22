# devsys

A developmental continual-learning system built around a frozen V-JEPA perceptual
substrate. The headline question: how far can a trainable shell go on top of inherited,
fixed perception, learning continually without forgetting, on a laptop today and a GPU
later, with every experiment forced to declare its null.

## What this is (and is not)

This is NOT a JEPA. We do not train a JEPA, we do not pretrain an encoder, we do not touch
encoder weights. V-JEPA is ONE module here: a frozen perceptual substrate, loaded once,
called only under `no_grad`, never receiving a gradient. It lives under `substrate/` and is
inference-only. A unit test asserts no encoder parameter ever gets a gradient
(`Frozen-substrate invariant`, see ARCHITECTURE.md).

Everything else is the trainable shell plus the research surface around it: the latent
predictor and heads, the latent hippocampus (replay buffer), staged plasticity,
consolidation, neuromodulation, the learning rules, the metrics, the diagnostics, the
experiment bank (E1..E10 + I4), and the campaign harness. That is the part we build, train,
and study. The encoder is a fixed boundary condition the shell learns against.

You cannot recover information the frozen encoder discarded. Any mechanism that needs a
variable X must first prove X is linearly decodable from the frozen latent
(`diagnostics/linear_probe`). That gate is the spine of the whole program.

## Cached-latent-first design

The encoder is the one large, slow, fixed object. So it runs once. Video clips go through
`substrate.encoder` a single time, the latents land in a memmap cache
(`substrate/cache.py`, `substrate/latent_store.py`), and all learning iterates on the cache,
never on pixels. Because the encoder is frozen, cached latents never go stale: there is no
retraining that would invalidate them. This is what makes the laptop feasible and the GPU
optional for most of the bank.

When real weights are not present (this session), the substrate falls back to a frozen
deterministic random projection and a synthetic latent generator (`substrate/datasets.py`),
and the store records `backend` so synthetic-substrate latents are never mistaken for real
V-JEPA latents. Every downstream experiment is built and tested on this path. Swapping in
real latents is a config + script flip, not a code change (see SCALING.md).

## Device flag (Apple-Silicon-native)

This project targets Metal (MPS) first. Everything device-touching goes through
`devices.resolve(cfg.device.kind)`, which prefers mps on Apple Silicon. The same code runs on
the M3 laptop today and a bigger M-series Mac Studio later (more GPU cores, more unified
memory) with NO device change: the Studio is Apple Silicon, not CUDA. fp16 autocast
(`devices.autocast`), unified memory (no pinning), and graceful MPS->CPU op fallback are built
in. `device=cuda` is the rented-box path used ONLY for Tier R environment rollouts.

```
.venv/bin/python scripts/run_experiment.py experiment=e1_baseline device=mps
```

See APPLE_SILICON.md (the MPS-first story, the 64-frame Metal limit, MLX option) and SCALING.md.

Note on data: V-JEPA ships pretrained WEIGHTS (which load fine, see CPU_RUN_REPORT.md), not a
training dataset. Its benchmarks (Something-Something-v2, Ego4D, EPIC-KITCHENS) are separate
datasets to obtain when expanding to natural-video latents. The system runs today on cached
real-encoder latents (structured-synthetic video) and the synthetic latent generator.

## Quickstart

Python 3.12 via `uv`. From the repo root:

```
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"      # add ann for hnswlib; encoder for real V-JEPA weights
make test                       # full suite (mps/cpu, seconds); count grows, see STATUS.md
make e1                         # E1 the gate: naive forgets, protected retains, both learn
make i4                         # I4 backprop-alternatives comparison (FA/DFA/FF/...)
make queue-dry                  # dry-run the campaign queue (no compute spent)
make accept                     # end-to-end acceptance check
```

`make install` does the venv + install in one step (it pulls `[dev,ann]`). Use
`make lint` / `make types` / `make fmt` for ruff + mypy. `make diag` runs the diagnostics.
E1 must pass (the gate) before any downstream result is trusted: see EXPERIMENTS.md for the
build-order DAG (E1 gates everything; E2,E3 feed E4; E2+E3+E4 = the Level-5 headline).

## Rehearse the Mac Studio workflow (one command)

```
make rehearse        # python scripts/studio_rehearsal.py
```

This is the Mac-Studio REHEARSAL CAPSULE: it walks the entire future Studio workflow end to end
on tiny generated fixtures, with NO downloads and NO long runs, and writes
`runs/studio_rehearsal/{report.md,summary.json}`. The path it proves:

  tiny video corpus (generated .npy clips) -> source validation -> decode + preprocess ->
  cache creation -> cache integrity -> full-grid dry-run + cost agreement -> one tiny Tier C
  run -> provenance manifests -> microbenchmarks -> Markdown + JSON report.

It is a rehearsal, not a science result, and the report tags every stage real / mocked /
provisional. On THIS machine the video DECODE is mocked (no codec): the corpus is `.npy` clips
that flow through the exact same validate/decode/preprocess/cache contract; on the Studio the
same path runs over real `.mp4` with a video backend (`uv pip install -e ".[video]"`). Other
operator tools: `make doctor` (readiness), `make cache-list`, `make storage`, `make bench`,
`make report`, `make docs` (drift gate).

## Repo map

```
substrate/      the frozen module + its access path (NOT trained)
  encoder.py        V-JEPA wrapper: requires_grad=False, no_grad only; lazy real weights,
                    frozen-random fallback
  cache.py          run the encoder once, write latents to a memmap
  latent_store.py   memmap-backed latent store (read path for all learning)
  datasets.py       task/class/domain-incremental streams + synthetic latent generator
shell/          the trainable shell (everything that learns)
  predictor.py      latent->latent predictor (+ action-conditioned variant)
  heads.py          task heads + probabilistic gaussian head (calibration, epistemic split)
  ensemble.py       ensemble disagreement uncertainty
  buffer.py         latent hippocampus: prioritized replay, KV faiss/brute index, eviction
  plasticity.py     staged plasticity (hard/soft/learned) + PNN rigidity + reopening
  consolidation.py  EWC (Fisher proxy) + SI (path integral), selectable + composable
  neuromod.py       DA=RPE, ACh=expected unc, NE=unexpected unc; scalar gates on lr/replay
  modulation.py     context-gating, working memory, chunking
learning/       the learning rules
  backprop.py       standard backprop trainer (accuracy ceiling); the Learner wiring
  alternatives/     FA, DFA, FF, target-prop, eq-prop, predictive-coding, local rules
metrics/        BWT/FWT/adaptation speed/avg acc (continual.py); adaptation-retention
                frontier + AUC (frontier.py, the program's central metric)
diagnostics/    linear_probe (the gate), noisy_tv, calibration, fisher_trace, determinism
experiments/    base.py (the doctrine contract), e1 harness, i4 harness, E2..E10 scaffolds
harness/        runner.py + cli.py (run, compose, campaign queue)
campaign/       synthesized training campaign (legs/tracks/tiers, run queue); see DECISIONS.md
configs/        OmegaConf group composition: device/, encoder/, shell/, experiment/
scripts/        run_experiment.py, cache_latents.py, run_queue.py, acceptance.py
```

## The doctrine contract

`experiments/base.Experiment` is abstract and refuses to instantiate unless the subclass
declares `metric`, `baseline`, `ablation`, and `null_hypothesis`. An experiment that does
not state its null cannot run. This is enforced in code, not convention. Every null in the
bank maps to one entry in the negative-result taxonomy (EXPERIMENTS.md), so a failed
experiment is a result, not a dead end.

## Where to read next

- ARCHITECTURE.md: the layering, the module -> corpus-lever map, the frozen-substrate
  invariant, the device boundary.
- EXPERIMENTS.md: the E1..E10 + I4 registry (metric, null, baseline, ablation, tier), the
  build-order DAG, the negative-result taxonomy, the diagnostic gates.
- SCALING.md: exactly what to flip when the Mac Studio or rented CUDA arrives, per tier,
  with first-commands-on-the-new-machine.
- STATUS.md: live build log (done / scaffolded / not started).
- DECISIONS.md, ISSUES.md: rationale log and deferred-item ledger.

## Form

Code FORM follows BLACKHOLE.md (density, flat structure, few load-bearing files, surface
every failure). No em dashes or en dashes anywhere (commas, colons, parentheses only).
