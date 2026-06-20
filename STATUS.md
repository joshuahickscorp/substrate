# STATUS (live log)

Legend: [x] done+tested, [~] scaffolded/deferred, [ ] not started, [!] degraded.

## Phase 0 read+plan
- [x] read all corpus volumes (I, II, III); confirmed NO Volume IV exists
- [x] ARCHITECTURE.md (module -> lever map)
- [x] EXPERIMENTS.md (E1-E10 + I4 registry: metric, null, tier)

## Phase 1 foundation
- [x] pyproject, ruff, mypy, pytest, Makefile
- [x] config (OmegaConf group composition + dotlist), seeding+determinism, devices (mps/cpu/cuda + fallback), logging+manifests
- [x] tests: foundation (7) green; lint+types clean

## Phase 2 substrate
- [x] encoder (frozen, lazy real weights + frozen-random fallback); grad-free invariant tested
- [x] latent cache pipeline + memmap store + datasets + synthetic generator (9 tests)
- [~] real latent caching DEFERRED (no weights this session); synthetic path operational

## Phase 3 trainable shell
- [x] predictor (+action-conditioned), heads (+probabilistic gaussian), ensemble (disagreement)
- [x] buffer (prioritized PER, KV faiss/brute index, reservoir/fifo/priority eviction)
- [x] plasticity (hard/soft/learned + PNN rigidity + triggered reopening)
- [x] consolidation (EWC fisher + SI path-integral, hand-case math tested), neuromod (DA/ACh/NE), modulation (context/WM/chunking)
- [x] 21 shell tests green; lint+types clean

## Phase 4 E1 (the gate)
- [x] metrics: continual (BWT/FWT/adaptation), frontier (Pareto + AUC)
- [x] learning.Learner (backprop trainer wiring buffer+EWC/SI+plasticity+neuromod)
- [x] experiments.base doctrine contract (no null -> cannot define), E1 harness, runner, CLI
- [x] E1 integration gate GREEN: naive BWT -0.40 forgets, protected BWT -0.14 retains, both learn last task; plots saved

## Phase 5 diagnostics
- [x] linear_probe (decodability), noisy_tv (epistemic collapses on noise, error stays high, LP separates)
- [x] calibration (reliability+ECE), fisher_trace (critical-period signature), determinism (Metal spread)
- [x] 9 diagnostics tests green

## Phase 6 experiment scaffolds + local-learning comparison
- [x] E2-E10 scaffolds (all 9; metric+null baked, toy-runnable, each with its own integration test)
- [x] I4 backprop-alternatives (full): backprop, FA, DFA, FF, target-prop, eq-prop, predictive-coding + table
- [x] all 11 experiments registered; full suite green; lint+types clean

## Phase 7 polish
- [x] README, SCALING written (wave 1); ARCHITECTURE/EXPERIMENTS finalized
- [x] harness sweep + queue, scripts (cache_latents, run_queue, acceptance)

## Phase 8 campaign (synthesized; no Vol IV)
- [x] resources (encoders, streams, seeds, budget_controls)
- [x] 14 legs across 11 tracks (track01..track11), run_queue.yaml (tier-tagged, dep-aware DAG)
- [x] harness/queue.py + sweep.py + scripts/run_queue.py; dry-run resolves, all 10 Tier C legs run toy
- [~] Tier E (track10 curiosity env) + Tier R (track11 capstone/POET/cultural) DEFERRED (disabled): need env + rented CUDA

## CPU-Now campaign (execution session)
- [x] unlock attempted: real V-JEPA 2 ViT-L weights LOAD from HF (network up); real forward fixed
      (pixel_values_videos=) and validated [B,64,3,256,256]->[B,8192,1024]
- [~] real-encoder caching: 64-frame ViT-L forward is very slow on M3 (MPS hung; CPU ~minutes/clip),
      pipeline proven, useful sample DEFERRED to Studio; campaign science runs PROVISIONAL (synthetic latents)
- [x] parallel CPU harness (cpu_pool: process pool, thread caps, mem-aware, checkpointed, resumable)
- [x] studies legs: 11A determinism, 11B seed-variance, 11D negative-registry, 8B assoc-memory, 8C pc-depth, cost-projection
- [x] T0/T1/8B/8C real results: CPU bit-identical (det), seeds {headline 5, sanity 3}, Hopfield cap 6x vs ff 1x, PC gap widens with depth
- [ ] full campaign driver run (T0-T2 complete, T3 budget-drained) + cost projection + CPU_RUN_REPORT.md

## Acceptance
- [x] scripts/acceptance.py GREEN 10/10 (suite, lint, types, E1 gate, diagnostics, I4, queue dry-run, toy Tier C leg, registry)
- [x] 68 tests pass; ruff + mypy clean; em-dash-free across code/config/docs
- [x] adversarial review wave applied: fixed SI path-integral wiring, reservoir bias, E9 measured-memory + non-vacuous test, determinism absolute tolerance, Level-5 leg now genuinely combines E2+E3+E4, make diag entry point
