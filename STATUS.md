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
- [ ] predictor (+action-conditioned), heads (+probabilistic), ensemble
- [ ] buffer (prioritized, KV index, eviction), plasticity (hard/soft/learned + PNN rigidity)
- [ ] consolidation (EWC + SI), neuromod (DA/ACh/NE gates), modulation (context/WM/chunking)

## Phase 4 E1 (the gate)
- [ ] continual harness + metrics (BWT/FWT/adaptation/frontier) + forget-then-retain integration test

## Phase 5 diagnostics
- [ ] linear_probe, noisy_tv, calibration, fisher_trace, determinism

## Phase 6 experiment scaffolds + local-learning comparison
- [ ] E2-E10 scaffolds (metric+null baked, toy-runnable)
- [ ] I4 backprop-alternatives (full): backprop, FA, DFA, FF, target-prop, eq-prop, predictive-coding

## Phase 7 polish
- [ ] README, SCALING, finalize ARCHITECTURE/EXPERIMENTS

## Phase 8 campaign (synthesized; no Vol IV)
- [ ] resources (encoders, streams, seeds, budget), legs (track01..track11), run_queue.yaml
- [ ] harness/queue.py + scripts/run_queue.py; dry-run + one toy Tier C leg

## Acceptance
- [ ] scripts/acceptance.py green end to end
