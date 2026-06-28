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
- [x] Studio cost projection (Tier C ~0.11h, E ~1.5h, R ~20h full-scale, laptop-throttled), historical CPU campaign values consolidated in /Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md
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

## Studio acquisition layer (this session)
- [x] ONE pipeline surface scripts/studio_pipeline.py: plan -> acquire -> validate -> cache -> run -> optimize -> report, plus local-max + profiles
- [x] device profiles + kill switches (src/devsys/studio/profiles.py): studio-1tb (900 GB usable) and m3pro-local-max (10/25 GB download, 2 GB fixtures, 128 clips, 60 GB free floor, 90 min, Tier C); hard caps enforced, dry-run by default
- [x] dataset registry registry/datasets.yaml (11 sources: ssv2, kinetics700, epic, ego4d subset+full, ego-exo4d, howto100m, audioset, laion, synthetic, local) + model registry registry/models.yaml (aux/distilled/quantized, never replace canonical); schema + honesty validated
- [x] 1 TB knapsack planner (studio/planner.py): breadth-first, budget + per-source + source-count caps, subset scaling, license gating; full Ego4D NEVER planned by default
- [x] dry-run downloader orchestrator (studio/downloader.py): execute+budget+license gated, resume manifest, hash/dedup, unsafe-archive (path-traversal) refusal, clean remote-block without credentials
- [x] data cards + license ledger (studio/datacards.py); synthetic control expansion (studio/controls.py): 9 families (moving/permanence/occlusion/relation/containment/noisy-TV/navigation/class-inc/domain-inc)
- [x] gated conveyor (cmd_run): gates are kill switches (registry valid, free disk, tier allowed, run-count cap) that STOP the run, not warnings
- [x] local-max ran REAL on this M3 Pro (12 stages pass): generated 5 control families, built+validated a 12-clip real cache (frozen-random -> provisional), queue/cost audit agreed, one gated leg ran; report under runs/studio_pipeline/
- [x] 6-dimension adversarial subagent audit (acquisition/cache-queue/device-safety/downloader/conveyor/docs-tests): 41 findings, 31 confirmed, ALL integrated. Key fixes: stratified clip-cap + honest class coverage (capped cache never claims dropped classes; validate_cache enforces it); CLI run/cache/optimize default to the SAFE m3pro profile (laptop full-run fails the run-count kill switch, Studio passes --profile studio-1tb); post-fetch + cumulative-resume budget enforcement; effective_budget rejects negative; partial-cleanup + phantom-resume + corrupt-manifest-warn + manifest provenance; registry honesty (aux/distilled/quantized never replace canonical, canonical real-encoder must be a verified HF id, broader signed-terms heuristic); planner budget clamps to usable disk; unsafe-archive Windows paths + safe_extract; _set_latest pointer reconciliation; docs-drift gate now checks studio_pipeline subcommands
- [x] 339 tests pass (93 new across profiles/registry/planner/downloader/controls/datacards/pipeline + audit regressions); ruff + mypy clean; docs-drift gate clean; acceptance 10/10

## Developmental capacities layer (this session)
- [x] sentience-ADJACENT north star + safety rails (src/devsys/devel/north_star.py): the developmental loop, the allowed engineering vocabulary, and a claim scanner that flags affirmative sentience/consciousness claims while passing disclaimers; gates every rendered report. NO sentience/consciousness/agency claims anywhere
- [x] paradigm frontier registry (registry/paradigms.yaml, 17 candidates from Frontiers 21-30: plasticity/fast-weights, abstraction, memory, alt-learning, modularity, active-curriculum, world-model, multimodal) with schema + honesty validation (a candidate can never claim a canonical result tag)
- [x] developmental capacity ladder (registry/capacities.yaml, 14 rungs: sensory grounding -> permanence -> episodic memory -> plastic adaptation -> consolidation -> curiosity -> abstraction -> causal sketch -> self-monitoring -> language-mediated -> teacher lane -> skill library -> meta-learning -> provenance continuity), each with baseline/ablation/metric/null/local+studio tests/failure interpretation/promotion rule
- [x] curriculum engine (src/devsys/devel/curriculum.py): REAL on-device learning-progress data selection over generated controls; permutation-test noise detection picks the learnable-but-not-mastered family and REJECTS the aleatoric noisy-TV (verified: chooses hard_motion, rejects aleatoric_tv)
- [x] automated ablation/hypothesis engine (ablation.py): ranks paradigm candidates by expected info gain per compute hour, gates on scope, respects an hour budget, flags redundant candidates, names the next-best experiment (local next-best = learning_progress_sampling)
- [x] paper-watch (offline, registry/paperwatch.yaml, 9 topics) + metacognition self-monitoring report (distinguishes measured capacities from claims; gated by the safety rail)
- [x] markdown consolidation (Frontier 36): removed dead scripts/_scaffold_api.md (0 refs); consolidated old run reports and maximal-goal prompt into /Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md; added a markdown LEDGER to check_docs so stale docs cannot regrow; canonical doctrine = corpus vols + BLACKHOLE.md + docs/STUDIO_MAXIMIZATION_2026_06_27.md
- [x] new commands: scripts/devel.py (paradigms/capacities/ablation/curriculum/metacognition/paperwatch/validate) + make devel/ladder/curriculum
- [x] 6-dimension adversarial subagent audit of the developmental layer (safety-rails/registries/curriculum/ablation/markdown/integration): 36 agents, 19 confirmed, ALL integrated. Key fixes: HARDENED the sentience scanner (was bypassable) to clause-scoped negation + bidirectional noun/verb + mentalistic predicates + decoy-stripping for rhetorical double-negatives (now flags all audited bypasses, 0 false positives on our own text); curriculum permutation baseline now averaged over the same fold count as acc_full (killed a variance asymmetry that false-rejected a learnable family) + default eval_clips 48; stable sha256 family seeding in controls (Python hash() is PYTHONHASHSEED-salted); registry honesty (result_tag .strip(), capacity tag/provenance closed-vocab, free-text sentience scan at the registry boundary, tags-must-be-list); ablation "redundant" relabelled competing_groups + per-row basis=assumption
- [x] 378 tests pass (39 new: devel registries/north-star/ablation/metacognition/curriculum + markdown consolidation + audit regressions); ruff + mypy clean; docs-drift + markdown-ledger gate clean; acceptance 10/10
