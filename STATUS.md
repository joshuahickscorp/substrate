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
- [x] conducted bank registered (E1-E10, I4, + EX3/EX8/EX12/EX16/EX17); full suite green; lint+types clean

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

## Experiment-bank expansion + scaffold (this session)
- [x] registry/experiments.yaml: the machine-readable, PREREGISTERED bank (39 catalogued: E1-E10, I4,
  EX1-EX18, D1-D6, A1-A4). Every row carries null + headline metric + falsifier + controls + gates +
  resource_tier + exp_tier + capacity/paradigm map + proof linkage (atlas factor / null card / R-level)
  + failure-taxonomy slot. Validated by devsys.devel.registries.validate_experiment.
- [x] loader/validator in devel/registries.py: closed-vocab tiers, implemented rows must map to a real
  REGISTRY id or an existing module, REGISTRY and bank file cannot diverge, free-text passes the
  sentience rail. EXPERIMENTS.md is now GENERATED from the registry (no drift); `scripts/devel.py
  experiments [--render]` lists/validates/regenerates it.
- [x] new diagnostics: diagnostics/geometry.py (D1: CKA linear+kernel, RSA, effective rank, anisotropy,
  intrinsic dim, neighbourhood overlap), diagnostics/compute.py (D5: param/FLOP accounting + matched
  compute), diagnostics/substrate_ablation.py (D2: real / frozen-random / shuffled / compressed arms).
- [x] new shell primitive: shell/refine.py IterativeRefiner (residual latent refinement, fixed-N +
  adaptive halting), the latent-reasoning primitive.
- [x] 5 new runnable cpu-now experiments (configs + campaign legs on track12, gated on the E1 gate):
  EX12 atlas (self-checking decodability + geometry), EX17 latent iterative reasoning (tied refiner vs
  compute-matched untied depth, FLOP-matched), EX8 intrinsic-motivation bake-off (LP/disagreement/RND/
  prediction-error on the noisy-TV), EX16 codebook/VQ abstraction, EX3 test-time adaptation (label-free
  overlay, base restored on revert). REGISTRY now 16 (was 11).
- [x] EX1/EX2/EX4/EX5/EX6/EX7/EX9/EX10/EX11/EX13/EX14/EX15/EX18 + D3/D4/D6 + A1-A4 catalogued
  registry-only/deferred with the full contract (the precise next tranche, not vague).
- [x] 421 tests pass (46 new: registry validation, geometry math, compute accounting, substrate
  ablation, refiner determinism, the 5 EX experiments); ruff + mypy clean; docs-drift + markdown-ledger
  gate clean (GO.md ledgered); acceptance 10/10.

## Cross-disciplinary roots expansion (this session): full-overkill cpu-now build
- [x] 6 new diagnostics (src/devsys/diagnostics/): convergence.py (contraction factor, fixed-point
  residual, basin stability; Y1/Y2/N9), nonlinear_probe.py (capacity-capped MLP probe + readout-
  contribution index, nonlinear-minus-linear real-minus-frozen-random; P10/P1), held_out_combo.py
  (factorized latents + held-out-combination decoding + systematicity sweep; C1/C9/S6), seed_consistency.py
  (cross-seed CKA + Hungarian code agreement + code stability; Y3/P5/S5), bottleneck.py (capability-per-bit
  + quantization robustness; I1/I8), sysid.py (least-squares latent system id, one/k-step R2, controllability
  Gramian rank, action-shuffle delta, planning_licensed gate for EX2; Y7). 17 known-answer tests green.
- [x] shell extensions: refine.py IterativeRefiner +mode ("residual"|"predictive_coding") +pc_rate, +unroll()
  returning per-step update norms (feeds the convergence diagnostics), +Verifier head (N3/N9/N11/Y9).
- [x] 77 new Experiment subclasses across 9 series files (n_neuro_replay_reasoning, d_developmental,
  b_biology, p_philosophy, c_cogsci, i_infotheory, y_dynamics, s_semiotics, a_perception), each with a
  config, a preregistered registry/experiments.yaml row (null + headline metric + falsifier + taxonomy slot
  + tier), and an integration test; mechanisms implemented INLINE per the established EX pattern. Honest
  nulls only (the bound IS the result; pooled-latent object/spatial tests record taxonomy-3 and ship the
  bound). Generated by a parallel-agent workflow, self-verified against the live editable install; the shared
  files (scaffolds.py, registry, EXPERIMENTS.md) integrated centrally.
- [x] standing controls wired at the harness level and reused by every relevant experiment: beat frozen-
  random (substrate_ablation), match compute (diagnostics.compute), beat a tuned baseline, the noisy-TV guard
  for curiosity signals, the seed-stability harness.
- [x] registry/experiments.yaml now 116 catalogued (94 implemented, 17 registry-only, 5 deferred); REGISTRY
  93 registered (was 16); by series A12/B9/C9/D14/E10/EX18/I10/N10/P8/S8/Y8. EXPERIMENTS.md regenerated.
  Tranche-2 (N2/D6/D10/B8/Y4/Y10/I10, weights-needed P7/S2/S8 + dense-2.1 retargets, environment-needed
  P8/A9/EX2-live) stays registry-only/deferred with full contracts.
- [x] 512 tests pass (new: foundation-diagnostics known-answer suite + one integration case per roots
  experiment, each asserting the doctrine contract + an explicit null check); ruff + mypy clean (109 source
  files); docs-drift gate clean; acceptance 10/10.

## Pre-Studio maximal push (this session): run everything real, close the remaining scaffolding, stage transfer
- [x] all 93 cpu-now roots+EX+E experiments RUN FOR REAL (not just scaffolded) on the M3 Pro, 47s wall,
  0 errors; results persisted per-id at runs/pre_studio/<id>.json + a machine-readable _summary.json.
- [x] adversarial verification of every null_supported=False candidate positive (2 independent workflow
  passes, ~28 distinct experiments checked against the standing controls each actually measured): EVERY
  toy-scale rejected null failed at least one control (frozen-random, matched-compute, tuned-baseline,
  seed-stability, or the noisy-TV guard). Zero confirmed positives at toy scale; this is the expected,
  honest result (toy configs are underpowered by design) and is itself the pre-Studio finding: nothing
  here should be reported as a real effect without a Studio-scale rerun.
- [x] closed the remaining registry-only cpu-now scaffolding left from the earlier EX-expansion session
  (14 rows, all resource_tier=cpu-now, previously deferred only for session-scope discipline):
  - 3 rows pointed at EXISTING standing-control infrastructure already wired into every roots experiment
    (a1_frozen_random_arm -> diagnostics/substrate_ablation.py, a2_matched_compute_arm ->
    diagnostics/compute.py, d6_rollout_gate -> diagnostics/sysid.py, metrics corrected to the module's
    real return keys) and flipped to implemented with no new code, since the mechanism was already real.
  - 4 new small diagnostic modules (built directly, foundation-first, with known-answer tests):
    diagnostics/difficulty_calibration.py (D3, certifies a regime via a known-separable reference before
    trusting a tie), diagnostics/transfer_matrix.py (D4, T-by-T linear-probe transfer + asymmetry index),
    diagnostics/buffer_compression.py (A3, retention-per-byte, generalizes I5's inline quantize-on-store
    loop into reusable infra), diagnostics/latent_robustness.py (A4, shuffle/noise/dropout/low-bit
    degradation curves). tests/unit/test_scaffolding_diagnostics.py, 6 known-answer tests green.
  - 7 new Experiment subclasses (EX1 generative vs stored-buffer replay, EX4 hypernet fast-weights vs
    gradient-TTA, EX6 active-inference free-energy objective, EX7 Reptile meta-learning, EX11 causal/
    interventional probing, EX14 memory bake-off FIFO/Hopfield/uncertainty-indexed, EX18 latent self-
    verification built on EX17's refiner+Verifier), each with a config + integration test, generated by a
    parallel-agent workflow (self-verified against the live editable install, one file per agent, no
    shared-file writes), centrally integrated into scaffolds.py/registry/EXPERIMENTS.md. Honest toy
    results: EX1/EX7/EX11/EX14/EX18 nulls held; EX4's hypernet beat static-head but did not match
    gradient-TTA; EX6's free-energy objective beat the baseline selectors and stayed calibrated.
  - registry/experiments.yaml now 116 catalogued, 108 implemented, 3 registry-only (weights/env-needed:
    ex2_latent_planning, ex9_slot_attention, ex10_cross_modal), 5 deferred (studio-scale: ex13_long_stream,
    ex15_rejuvenation, ex5_local_rules_scale, e10_openended's full population form, ex2's live-env form).
    REGISTRY 100 registered (was 93). tests/integration/test_scaffolding_experiments.py, 8 tests green.
- [x] real V-JEPA 2 weights confirmed already staged in the local HF cache (ViT-L, ViT-H; neither repo is
  gated, no HF token needed) and a real-encoder latent cache built from them: scripts/cache_real_encoder.py
  device=mps overflowed the M3 Pro's ~16GB MPS attention buffer ceiling at 64-frame/256px clips even at
  batch=1 (a genuine, reproducible Studio-necessary boundary, not a bug); device=cpu succeeded (16 clips,
  333s, ~21s/clip, backend=vjepa_hf confirming real weights not the frozen-random fallback, linear-probe
  acc=1.0 vs chance=0.25). Two aux encoders staged and fully transferable (~/.cache/huggingface): DINOv2-
  large (image, unblocks S2 cross-domain alignment) and VideoMAEv2-Base (video, unblocks S8 multimodal).
- [x] 526 tests pass; ruff + mypy clean (120 source files); registry validates; docs-drift gate clean
  (526 tests collected, REGISTRY size 100 at this point in the build, 10 acceptance checks); acceptance
  10/10 (31 campaign legs plan clean).
- [x] full interpretation and adversarial verification of all 100 results: runs/pre_studio/RESULTS_PRE_STUDIO.md
  (per-series tables, 63 nulls held, 9 taxonomy-3 bounds, 15 ambiguous, and every one of the 25 candidate
  positives across the first 93 individually re-checked against the SPECIFIC standing control it declares,
  not just whatever it happened to report; the adversarial-verify workflow ran twice independently after two
  API rate-limit interruptions, both passes agreeing: zero survive). Failure-mode breakdown: 17 never ran or
  failed frozen-random, 2 matched-compute, 2 seed-stability, 1 tuned-baseline, 1 the mechanism itself failed
  on inspection. An addendum covers the 7 later experiments: EX4 fails the same tuned-baseline pattern; EX6
  looked like a clean positive on first read and was later refuted by a proper three-pass adversarial check
  (see below), so both of the addendum's candidates are now honest negatives.
- [x] STUDIO_HANDOFF.md (root, ledgered in OPERATIONAL_MD): what transfers (HF weight cache incl. two fully-
  downloaded aux encoders, the verified non-degenerate 16-clip real-latent cache, the registry, the 4 new
  diagnostics, the fresh local-max rehearsal), the honest 9-row Studio-gated table (only 3 of 9 are truly
  gated; the other 6 already run on the laptop or are blocked on unwritten code, not hardware), and a
  prioritized first-things-to-run list built from the adversarial-review failure modes.
- [x] scripts/studio_pipeline.py local-max rehearsal re-run against current HEAD (112053b): all 12 stages pass
  (free_disk_killswitch, doctor, registry_validate, plan, acquire_dryrun, generate_controls, validate_source,
  build_cache, queue_cost_audit, microbench, gated_run, datacards_ledger). runs/studio_pipeline/latest updated.

## Post-handoff pass (this session): a real corpus-changing finding, disk hygiene, closing more gaps
- [x] disk hygiene: found and removed 3.5GB of orphaned `.incomplete` HF-cache retry blobs left over from
  the earlier interrupted vjepa2-vitl-fpc64-256 download attempt (before the working device=cpu recipe was
  found); verified the real encoder still loads correctly (backend=vjepa_hf) after cleanup.
- [x] found and fixed a real methodology gap: e4_neuromod.py and e7_sparse.py read a single `cfg.seed`, not
  `experiment.seeds`, so the earlier "seed_stability REFUTED" verdict for both was based on config overrides
  that were silent no-ops (still one seed). Re-ran both through devsys.harness.sweep.run_sweep's REAL
  per-seed/per-axis grid (30 genuine runs each, 6 axis combos x 5 seeds): e4_neuromod's negative finding came
  back robustly confirmed (all 30 runs show both gates amplify error on noise, the wrong direction, disagreement
  mean=20.07 std=1.25); e7_sparse's disqualifying seed-stability objection did NOT hold up (sparse beats
  dense in direction in 27/30 and 26/30 runs, mean gains 0.14-0.18). **e7_sparse is now a provisionally
  confirmed positive, the one candidate in the full 100-experiment corpus that has cleared its specific
  standing control.** Wired diagnostics/difficulty_calibration.py (D3) directly into e7_sparse.py as a
  regime_calibration field (reference_score=1.0 vs chance=0.167, confirming the stream carries real
  structure, closing the "is this tie/gap meaningful" gap the registry's own relation field named).
- [x] gave EX6 (active-inference free-energy objective) the proper adversarial pass flagged as outstanding:
  three independent checks (two code re-analyses, one numerical resimulation) unanimously REFUTED it. The
  apparent effect (free_energy beats learning_progress on noisy-TV rejection) is a variance-magnitude
  artifact of the hardcoded noise_scale=5.0 hyperparameter: a trivial zero-learning "avoid the higher-
  variance region" heuristic reproduces the exact result, and an inversion test (lowering the noise
  region's variance below the learnable region's own residual scale) flips the "free-energy" selector to
  pick the irreducible noise 100% of the time. Risk (~=plain NLL) drives ~99% of the effect; the KL-
  complexity term is negligible or slightly opposing. RESULTS_PRE_STUDIO.md and STUDIO_HANDOFF.md corrected
  (both had cited EX6 as a plausible/top-priority positive from an earlier, insufficiently adversarial read).
- [x] RESULTS_PRE_STUDIO.md and STUDIO_HANDOFF.md updated throughout to reflect both findings; corrected a
  stale disabled-legs example in the handoff's transfer checklist (track04_e4_neuromod/track06_e7_sparse are
  actually enabled:true Tier C legs that had simply never been executed, not disabled Tier E/R legs).
- [x] real-encoder latent cache rebuilt larger in the background (64 clips vs the earlier 16, same verified
  device=cpu/batch=1 recipe) for a stronger real-weight evidence base to hand off.
- [x] found the studio pipeline's disk kill-switch (min_free_disk_gb=60) now genuinely trips on this
  machine (free disk drifted from ~63GB to ~53-58GB over the session from system-level activity unrelated
  to this repo, confirmed by checking pytest tmp usage (~23MB) and runs/ growth (~176MB), both negligible).
  3 studio_pipeline integration tests that exercise the REAL kill switch against REAL disk_usage() are
  consequently red at low-disk moments; this is the kill switch correctly refusing to proceed, not a code
  regression, and is itself a useful live confirmation the safety floor works as designed. At low-disk
  moments acceptance.py's "full test suite" step (and only that step) goes 9 of 10, purely from
  test_local_max_smoke; ruff, mypy, the E1 gate, diagnostics, I4, queue dry-run, the toy leg, and the
  registry check all still pass cleanly regardless of disk state.
- [x] real-encoder cache quadrupled to 64 real V-JEPA 2 ViT-L latents (8 classes, ~21s/clip, ~23 minutes
  wall-clock on CPU, backend=vjepa_hf confirmed, linear-probe acc=1.0 vs chance=0.125), same verified
  device=cpu/batch=1 recipe as the earlier 16-clip build (MPS still overflows its attention buffer
  regardless of batch size). STUDIO_HANDOFF.md/RESULTS_PRE_STUDIO.md updated with the larger n.

## Genuine studio-scale work landed on this laptop (user request: keep making progress while waiting)
Of the 9 Studio-gated rows, 2 (ex13_long_stream, ex5_local_rules_scale) were blocked on UNWRITTEN CODE, not
literal hardware impossibility: their own registry descriptions say the compute-heavy or unavailable-
encoder parts are what's blocked, while the core mechanism is real and CPU-runnable, just much slower than
the Studio would give. Implemented both for real via a parallel-agent workflow (same pattern as the earlier
builds), each shipped with a SCALED (not toy) default config designed to run for real minutes on this
laptop, then run again at an even larger "grind" scale via overrides to give visible, genuine background
work. Registry rows flipped to implemented, resource_tier cpu-now (from studio-scale); only ex2_latent_
planning remains genuinely Studio-gated of the original 9.
- [x] ex13_long_stream.py (class EX13): naive-sequential vs replay+EWC-protected vs a frozen-random-
  substrate control arm, over a domain-incremental stream, anchor-task retention curve (not O(T^2)) plus
  periodic effective-rank sampling. Shipped scale (240 tasks): naive collapses (retention_threshold_length
  25), protected never collapses, divergence +0.25. Grind scale (3000 tasks, dim 96, hidden 128, ran in
  50.7s): divergence GREW to +0.53 (a durable, not-shrinking effect). At both scales the advantage does
  NOT clearly survive the frozen-random control (0.667 vs protected's 0.722 at shipped scale) — an honest
  null with one caveat found this session: the control arm's stream (n_tasks_control) is shorter than the
  main arms', not a length-matched comparison, flagged as a Studio follow-up.
- [x] ex5_local_rules_scale.py (class EX5): three PERSISTENT (weights carried across tasks) rule
  implementations (backprop ceiling, feedback_alignment with a fixed random backward matrix, predictive_
  coding via local energy descent) extending I4's from-scratch comparison to real backward-transfer on a
  continual stream, with a depth-sweep control. Genuine unforced positive at two scales (80 tasks/112s and
  300 tasks x 4 widths x 3 seeds): feedback_alignment and predictive_coding beat backprop on BOTH accuracy
  and BWT, stable across the depth sweep; backprop's own ceiling accuracy is barely above chance (0.336 vs
  0.25) with strongly negative BWT (-0.72). Adversarially checked: PLAUSIBLE-BUT-UNVERIFIED, not confirmed
  and not refuted, the specific gap being backprop's Adam optimizer vs the local rules' plain delta updates
  (same nominal lr, not a matched effective step size) — a real, named, actionable Studio follow-up, not a
  vague caveat.
- [x] tests/integration/test_studio_gated_implementable.py (3 tests, toy-scale overrides so the suite stays
  fast); scaffolds.py + registry/experiments.yaml + EXPERIMENTS.md integrated centrally; ruff + mypy clean;
  REGISTRY 100 -> 102. STUDIO_HANDOFF.md and RESULTS_PRE_STUDIO.md updated with both experiments' results
  at both scales and the adversarial-review verdicts, including a new priority-0 Studio follow-up item.
- [x] ex5_local_rules_scale grind (300 tasks x 150 epochs, 24.8 min) FLIPPED the retention finding: the
  local-rules-beat-backprop-on-BWT effect vanished at larger scale (FA/PC BWT -0.640/-0.649 now slightly
  WORSE than backprop -0.604; rules_with_bwt_advantage empty). The accuracy advantage persisted. This
  confirms the adversarial verifier's prediction: the shipped-scale "retention advantage" was a fragile,
  scale-dependent optimizer artifact, not a durable property. RESULTS doc updated to REFUTE the retention
  half while keeping the narrower accuracy finding. A cautionary, load-bearing example of why scaling a
  toy positive up is itself a standing control.

## Migrated 3 more Studio-gated rows to the laptop and ran them for real (user request)
User asked to migrate every genuinely-implementable Studio-gated experiment to the MacBook and run them
in sequence, given even hour-long runs fit a 12h ceiling. Audited the 9 originally-gated rows: only
ex2_latent_planning's LIVE arm, e6_relational, ex10_cross_modal (both need nonexistent real weights), and
e10_openended (needs multi-agent infra) are truly blocked; the rest were unwritten code. Built the 3
implementable ones via a parallel-agent workflow:
- [x] ex15_rejuvenation.py (EX15): shrink-and-perturb rejuvenation on ex13's long-stream harness, 3 arms
  tracking effective_rank/dead_unit_count/retained_accuracy. Honest null at shipped scale (a small frozen-
  latent shell shows NO loss of plasticity until pushed to dim256/hidden256/thousands-of-tasks, where the
  mechanism does activate and rejuvenation measurably restores effective rank). Big-run grind launched.
- [x] ex9_slot_attention.py (EX9): K learned slots vs a parameter-matched flat-pooled head on synthetic
  relation-change windows where the pooled latent is the NOISY SUM of both entities (faithful to real
  pooled V-JEPA output). null_supported=True: slots TIE flat, the expected taxonomy-3 substrate bound
  (pooling already summed the entities before the shell sees them, so there is no per-slot structure to
  recover). A clean, citable bound a dense 2.1 retarget of E6 must beat. Slot-count sweep launched.
- [x] ex2_latent_planning.py (EX2, synthetic arm only, live arm stays deferred): learned latent dynamics
  + shooting/CEM MPC vs flat-reactive-head + action-shuffle, rollout-predictability gated. Candidate
  positive (null_supported=False, planner beats both controls). Adversarially checked PLAUSIBLE-BUT-
  UNVERIFIED leaning weak: real flaw found (planner scored in-belief against its own model, flat head
  scored in the true env, so not a matched yardstick; flat_head_success=0.0 is a strawman; single-seed);
  the one clean signal is the action-shuffle gap (1.16, actions carry non-permutable info). Named Studio
  follow-ups to promote it. Big-run grind launched.
- [x] All 3 run through the standard REGISTRY path; tests/integration/test_migrated_experiments.py (4
  tests); scaffolds + registry + EXPERIMENTS.md integrated; ruff + mypy clean (125 source files); registry
  validates; docs gate clean (533 tests, 105 experiments). REGISTRY 102 -> 105; 113 of 116 rows now
  implemented, only 3 genuinely deferred. A sequential big-run grind of all 3 (ex15 ~92min, ex9 slot-sweep,
  ex2 ~95min, ~5h total) is running in the background, comfortably inside the 12h ceiling.

## DOCTRINE_SYNTHESIS.md: the corpus read through the two central doctrinal questions
- [x] DOCTRINE_SYNTHESIS.md (root, ledgered): a 4-agent workflow (two theme readers, a gap-analysis agent,
  a synthesizer) interpreted all ~105 real results against the project's two doctrinal questions,
  developmental plasticity/moldability and abstraction independent of language/symbols, using the original
  12-agent research map (the joyful-squid plan) as the frame. Not a new run and NOT an implementation plan
  (explicitly deferred per the user's sequencing); purely an interpretation to ground the eventual
  architectural decision in real evidence instead of literature-only hypotheses.
- [x] Honest headline it records: of ~25-27 candidate positives, exactly ONE survives adversarially
  (e7_sparse, provisional), one is plausible-but-unverified (ex5_local_rules_scale), everything else was
  refuted or is underpowered/taxonomy-3. Neither doctrinal question has a demonstrated substrate-level
  positive yet. Plasticity: every biological-signature mechanism (Fisher-triggered windows, ACh/NE
  gating, tag-and-capture, homeostatic scaling, replay ordering) ties or loses its non-biological baseline;
  the U-shaped-overgeneralization developmental signature is flat zero; the one plasticity positive
  (e7_sparse) is structural, not developmental-scheduling. Abstraction: five independent series converge
  on the same artifact (a frozen-random projection reproduces the effect), with s10_anti_self_deception the
  load-bearing meta-test showing the toy eval regime currently cannot distinguish real V-JEPA-specific
  abstraction from generic linear geometry; latent iterative computation behaves like plain unrolled depth;
  emergent codes are idiolects (below the frozen-random floor across seeds), not shared languages.
- [x] Proposes 7 concrete NEXT experimentation lanes (not an architecture): real-latent replication of the
  doctrine-load-bearing P/S/C/D series on the existing 64-clip real cache, a systematic difficulty-
  recalibration sweep over ceiling-saturated experiments, a never-built developmental-sensitive-window +
  structural-plasticity lane (D6/B8/Y4, D6 flagged as the single most direct untested test of the
  moldability thesis), cross-domain/cross-modal alignment (S2/S8/ex10), an object-binding-before-pooling
  probe via coarse spatial-token pooling, and the two Studio-priority closures (e7_sparse frozen-random
  arm, ex5 optimizer-matching control). README updated to surface RESULTS/DOCTRINE/HANDOFF docs.

## Started lane 1: real-latent replication (the doctrine synthesis's #1 next lane)
The whole corpus ran on synthetic Gaussian-cluster latents; a clean result there says nothing about the
real frozen V-JEPA 2 substrate. Scaffolded the infrastructure to re-ask the doctrine-load-bearing
questions on REAL encoder geometry:
- [x] src/devsys/substrate/real_latent.py: real_task_stream (class/task/domain-incremental slicing of a
  real LatentStore, drop-in for make_task_stream) + factorized_arrays (decode a two-factor cache stored as
  a composite label y=a*n_b+b via a factors.json sidecar) + open_real_store/factors_meta. tests/unit/
  test_real_latent.py, 6 known-answer tests green. Exported from devsys.substrate.
- [x] scripts/cache_factorized_encoder.py: builds a FACTORIZED real-encoder cache where TWO independent
  visual factors vary (hue = A, orientation/drift = B) at fixed spatial frequency, so held-out-combination
  and compositionality probes have real geometry to test (the single-factor cache_real_encoder.py entangles
  freq/angle/motion/color into one class index, useless for compositionality). Composite-label + sidecar,
  no LatentStore schema change.
- [x] scripts/real_latent_replication.py: the lane driver. Runs the doctrine-load-bearing probes on a real
  cache vs its frozen-random-projection control, LEADING with the nonlinear (readout_contribution/P10) and
  compositional (held_out_combination/C1-S6) tests, because a bare linear probe of separable classes is
  projection-invariant by construction. Reuses the existing diagnostics that were only ever run on
  synthetic data. Validated on the real 64-clip cache: everything ceiling-saturated (effective rank 3.4 of
  1024, all probes 1.0), so real is projection-invariant there, exactly the doctrine's predicted result and
  direct motivation for the harder factorized cache.
- [x] Queued (chained after the migrated grind, no CPU contention): build a 432-clip factorized real cache
  (6 hue x 6 orientation x 12, ~2.5h) then run the replication driver, writing runs/pre_studio/
  real_repl_factorized.json. This is the first time the compositional-abstraction question gets asked on
  real V-JEPA geometry rather than a Gaussian toy. 539 tests; ruff + mypy clean (126 source files); docs
  gate clean; registry validates.
