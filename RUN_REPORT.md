# RUN_REPORT

Autonomous build of `devsys`: a developmental continual-learning system around a FROZEN
V-JEPA perceptual substrate. Built front to back in one session on an Apple M3 Pro (18 GB),
device-agnostic so the Mac Studio / rented CUDA transition is a config flip. Acceptance is
green 10/10.

## What was built (all phases complete)
- Foundation: OmegaConf group-composition config, global seeding + a determinism utility
  that MEASURES Metal spread (not bit-exactness), a device layer (mps/cpu/cuda + graceful
  mps->cpu fallback), logging + per-run manifests.
- Substrate (frozen, inference-only): encoder loader (lazy real V-JEPA weights, frozen-random
  fallback), latent caching pipeline, memmap latent store, synthetic + real stream loaders.
  A unit test asserts no encoder parameter ever takes a gradient.
- Trainable shell: predictor (+ action-conditioned), task heads (+ probabilistic Gaussian),
  ensemble (disagreement), prioritized replay buffer (PER + faiss/brute KV index + reservoir/
  fifo/priority eviction), plasticity controller (hard/soft/learned + perineuronal-net
  rigidity + signal-triggered reopening), consolidation (EWC Fisher + SI path integral),
  neuromodulation (DA/ACh/NE gates), modulation (context-gating/working-memory/chunking).
- E1 (the gate): a continual-learning harness that DEMONSTRABLY forgets (naive, BWT -0.24..
  -0.40) then retains (replay+EWC, BWT -0.07..-0.14), both arms learn the last task, under
  fixed seeds, with saved plots. Metrics: BWT, FWT, adaptation speed, adaptation-retention
  frontier + AUC.
- Diagnostics: linear-probe distinctiveness, noisy-TV (epistemic disagreement collapses on
  irreducible noise while raw error stays high and learning progress separates), calibration
  (reliability + ECE), Fisher-trace critical-period signature, determinism sanity loop.
- E2-E10 scaffolds: composable, metric + explicit null baked in, each toy-runnable with its
  own integration test. I4 backprop-alternatives FULLY implemented (backprop, feedback
  alignment, DFA, forward-forward, target propagation, equilibrium propagation, predictive
  coding) with a comparison table (accuracy, gap, locality, weight-transport, separate-
  backward, activation-memory, cost).
- Campaign (Volume IV, SYNTHESIZED): resource sets (encoders/streams/seeds/budget_controls),
  14 legs across 11 tracks, a tier-tagged dependency-aware `run_queue.yaml`, a queue runner
  that validates + topo-orders + tier-gates + runs legs. Dry-run resolves; all 10 Tier C
  legs run end to end at toy scale.
- Docs: README, ARCHITECTURE, EXPERIMENTS, SCALING, STATUS, DECISIONS, ISSUES, this report.

## Test results
- 65 tests pass (unit: foundation, substrate, shell, diagnostics; integration: E1 gate, I4,
  per-scaffold, campaign queue). ruff lint + format clean. mypy clean (51 source files).
- `scripts/acceptance.py`: 10/10 PASS (full suite, ruff lint, ruff format, mypy, E1 gate,
  diagnostics, I4 table, queue dry-run, one toy Tier C leg, registry has 11 experiments).

## Deferred (environment-bound, scaffolded + unblock recorded)
- Real V-JEPA latent caching: weights not downloaded this session. The frozen-random
  substrate + synthetic latent path is fully operational and everything is built/tested on
  it. UNBLOCK: `uv pip install -e ".[encoder]"` (+ network), then
  `python scripts/cache_latents.py encoder=vjepa2_vitl_fpc64_256 device=mps +total=N`.
- Tier E leg (track10 curiosity env-rollout variant): data-selection variant is toy-runnable
  now; the env variant needs an environment. Queued disabled. UNBLOCK: provide an env adapter,
  set enabled=true, run `--tiers C,E`.
- Tier R legs (track11: E10 autotelic capstone, POET env-generation, cultural accumulation):
  need rented CUDA + a procedural environment + (POET/cultural) population-scale infra. Queued
  disabled. UNBLOCK on the Studio: provide env + `--tiers C,E,R --full --run-disabled`.

## Degraded
- None. No hard failures, no expected-failures in the suite (ISSUES.md is clean of defects).

## DECISIONS digest (full in DECISIONS.md)
- BLACKHOLE governs code form; this prompt governs scope. No em dashes anywhere.
- Python 3.12 via uv (3.14 system Python lacks stable ML wheels). torch 2.12.1, MPS verified.
- Config = OmegaConf + a ~40-line Hydra-style group composer (not full Hydra): fewer deps.
- NN index = faiss-cpu primary with a pure-torch brute-force fallback in the buffer.
- THE BIG ONE: there is NO Volume IV. The corpus is exactly 3 volumes; the "extended training
  campaign" (tracks, C/E/R tiers, Section 4 DAG, resource sets) exists nowhere. It was
  SYNTHESIZED faithfully from Vol I Section 8 build-order DAG + Section 9 compute map + the
  E1-E10/I4 experiment bank. Recorded in DECISIONS.md.
- `null` renamed to `null_hypothesis` everywhere (yaml parses bare `null` as a None key).
- E1 gate uses domain-incremental low-separation streams: separable clusters under-forget
  (exactly the corpus's predicted E1 outcome); overlap forces the interference the gate needs.

## ISSUES digest (full in ISSUES.md)
- Only environment-bound deferrals (real weights, Tier E/R). No code defects, no xfails.

## Campaign manifest summary (campaign/run_queue.yaml)
- 14 legs / 11 tracks. Tier C (cached-latent, enabled, 10 legs): track01 E1 gate (no deps,
  gates all) -> track02 E2 replay, track03 E3 plasticity, track04 E4 neuromod (each dep
  track01) -> track05 Level-5 integration (deps track02+03+04) ; track06 E7 sparse (deps
  track01+03), track07 E8 dendritic, track08 E9 local + I4, track09 E6 relational (dep
  track01). Tier E (disabled): track10 E5 curiosity (dep track04). Tier R (disabled):
  track11 E10 autotelic + POET + cultural (deps track05/track10). Dependency DAG validated,
  topo-ordered, no cycles.

## First commands on the new machine (Mac Studio / rented CUDA)
```
cd <repo>
uv venv --python 3.12 .venv && uv pip install -e ".[dev,ann]"
.venv/bin/python -m pytest -q            # 65 tests
.venv/bin/python scripts/acceptance.py   # 10/10
# scale up:
make e1                                   # or: python scripts/run_experiment.py experiment=e1_baseline device=cuda
python scripts/run_queue.py --dry-run --tiers C,E,R
python scripts/run_queue.py --tiers C --full          # full-scale Tier C campaign on the Studio
# real latents (after weights are fetchable):
uv pip install -e ".[encoder]"
python scripts/cache_latents.py encoder=vjepa2_vitl_fpc64_256 device=cuda +total=10000
```
See SCALING.md for the full flip-list (device flag, larger encoder, dense 2.1 unlocks, the
rented-CUDA rollout path for E5/E10 and the Tier R legs).
