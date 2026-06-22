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
- [x] real-encoder caching: 96 real V-JEPA latents cached on CPU (24.3s/clip); REAL-ENCODER results:
      linear-probe acc 1.000 (chance 0.167, decodable), and forget-then-retain on real latents
      (naive BWT -1.0 collapses to chance, replay+EWC BWT 0.0 perfect retention). Natural-video content
      deferred to Studio (structured-synthetic content here); grid science remains PROVISIONAL
- [x] parallel CPU harness (cpu_pool: process pool, thread caps, mem-aware, checkpointed, resumable)
- [x] studies legs: 11A determinism, 11B seed-variance, 11D negative-registry, 8B assoc-memory, 8C pc-depth, cost-projection
- [x] T0/T1/8B/8C real results: CPU bit-identical (det), seeds {headline 5, sanity 3}, Hopfield cap 6x vs ff 1x, PC gap widens with depth
- [x] full campaign run: T0-T2 complete (E1 5-seed x 3 streams, I4/E9 full, 8B/8C), T3 grids 10/10 in budget, 0 degraded
- [x] Studio cost projection (Tier C ~0.11h, E ~1.5h, R ~20h full-scale, laptop-throttled) + CPU_RUN_REPORT.md
- [x] 98 tests green; lint+types clean; all campaign numbers tagged real-encoder/provisional

## Apple Silicon native (this session)
- [x] MPS-first device layer: apple_silicon_info (M3 Pro 6P+6E ~19GB), fp16 autocast, unified-memory defaults
- [x] Studio reframed as Apple Silicon (not CUDA): SCALING.md + configs/device/* + APPLE_SILICON.md; cuda = Tier-R-only
- [x] mlx optional `apple` extra; V-JEPA = weights not dataset (clarified in README/DECISIONS)
- [x] real-encoder E2 replay-scheme comparison on the 96-latent store (no new video): replay beats naive, prioritized ties random (corpus null confirmed on REAL latents)

## Acceptance
- [x] scripts/acceptance.py GREEN 10/10 (suite, lint, types, E1 gate, diagnostics, I4, queue dry-run, toy Tier C leg, registry)
- [x] re-verified 10/10 after the Apple Silicon reframe; 103 tests green

## Plug-and-play hardening (this session)
- [x] encoder ids verified on HF (vitl/vith/vitg real; added vjepa2_vith); 2.1 dense not on HF -> placeholder+deferred
- [x] real-video ingestion: substrate/video.py (decode + tested preprocessing core) + scripts/cache_video.py + [video] extra
- [x] campaign legs carry full_axes/full_seeds (217 full-scale run-units); run_queue --full runs genuine factorials
- [x] 108 tests green; acceptance re-verified 10/10; em-dash clean

## Pre-Studio hardening sprint (this session)
- [x] F1 full-grid accounting: cost_projection == run_queue --full == manifest (single source of truth, tested)
- [x] F4 provenance: git/packages/device/seed/encoder/cache-id/result-tag in every run manifest + cache provenance.json
- [x] F7 fail-fast validation (validate.check_all clean) wired into the runner
- [x] F2 studio_doctor, F9 cache_tool, F14 storage_tool, F5 bench, F11 build_report, F6 check_docs (drift gate) + Makefile targets
- [x] F3 video hardening (validate source, label-map persist, clip hashes, dup detect, corrupt-skip, mocked decode fixture)
- [x] F8 failure rehearsal, F13 MPS routing tests, F15 encoder-registry honesty, F12 queue dry-run UX
- [x] FAISS Apple-Silicon segfault found + mitigated: buffer default index=brute, KVIndex subprocess-probes faiss, exact fallback
- [x] full suite green; ruff + mypy clean; acceptance 10/10; docs-drift gate clean

## Mac-Studio rehearsal capsule (this session)
- [x] one command `make rehearse` (scripts/studio_rehearsal.py) walks the WHOLE Studio workflow on tiny fixtures
- [x] tiny video-corpus generator (substrate/fixtures.py): deterministic .npy clips + injected duplicate + short clip
- [x] codec-free path: video.read_video decodes .npy (mocked decode); same validate/preprocess/cache contract
- [x] 9 stages all pass (~11s): doctor, corpus, validation, decode+preprocess, cache, integrity, full-grid dry-run + cost agreement, miniature Tier C run, microbench
- [x] report.md + summary.json under runs/studio_rehearsal/, every stage tagged real/mocked/provisional, with Studio day-one commands
- [x] tests: fixtures (4), capsule (5), failure rehearsal +2 (empty class folder, bad cache metadata)
- [x] 68 tests pass; ruff + mypy clean; em-dash-free across code/config/docs
- [x] adversarial review wave applied: fixed SI path-integral wiring, reservoir bias, E9 measured-memory + non-vacuous test, determinism absolute tolerance, Level-5 leg now genuinely combines E2+E3+E4, make diag entry point
