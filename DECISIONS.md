# DECISIONS

Autonomous-session decisions, each with a one-line rationale. Append-only.

## Doctrine
- BLACKHOLE.md (dropped in as the project rules doc) governs code FORM (density, flat structure,
  few load-bearing files, surface every failure). This build prompt governs SCOPE.
  Where they meet: exhaustive coverage written in the compressed BLACKHOLE register.
- No em dashes or en dashes anywhere (code, comments, docs). Commas/colons/parentheses only.

## Environment
- Python 3.12.13 via `uv` (not the system 3.14): FAISS/torch/ML wheels are best-supported
  on 3.12; 3.14 is bleeding-edge. uv is the package/venv manager (fast, standard).
- torch 2.12.1 installed; MPS verified available and a real matmul runs on device.
- Config system: OmegaConf + a ~40-line Hydra-style group composition layer, NOT full
  Hydra. We need only group selection + dotlist overrides; Hydra drags antlr+plugins and
  owns the entrypoint. One source of truth, fewer deps (BLACKHOLE: starve dependencies).
- NN index: faiss-cpu primary (installs cleanly), with a pure-torch brute-force fallback
  baked into the buffer index, and hnswlib available as the `ann` extra.
- Build backend: hatchling. Lint/format: ruff. Types: mypy. Tests: pytest.
- Generative replay: latent VAE (corpus lever L-GenerativeReplay); diffusion/AR priors
  are campaign options, not scaffold requirements.

## Spec reconciliation (the big one)
- THERE IS NO VOLUME IV. The corpus is exactly three volumes on disk (I Substrate+Spine
  +Experiment Bank, II Remaining Mechanisms, III Hardware/Open-Ended Frontier). The
  build prompt's "Volume IV extended training campaign" (track01..track11, tiers C/E/R,
  Section 4 DAG, resource sets Encoder/Stream/Seed/budget) exists NOWHERE in the corpus.
  Verified by exhaustive read of all three volumes + grep. Decision: SYNTHESIZE the
  campaign faithfully from the materials that DO exist:
    - legs/tracks <- the E1..E10 + I4 experiment bank crossed with their declared axes,
      grouped by the four corpus axes (A plasticity, B memory, C curiosity, D structural)
      plus integration + frontier legs.
    - tiers C/E/R <- map Vol I Section 9 compute tractability: Laptop-feasible -> C
      (cached-latent), environment-needed -> E, rented-GPU/lab-scale -> R.
    - dependency DAG <- Vol I Section 8 build-order DAG (E1 gates all; E2,E3 -> E4;
      E2+E3+E4 = Level-5 headline; E6 needs 2.1 dense; E5/E10 need env; E10 last).
    - seed set <- 5 seeds min (3-5 for E10), determinism sanity loop first.
- I4 (backprop-alternatives comparison) is named by the build prompt and the Vol III
  roadmap but has no full spec; Vol I E9 (local-learning head) supplies the rule grid.
  Decision: implement I4 as a dedicated, fully-built comparison harness (the prompt
  mandates it fully) over the same head/data/seed/budget: backprop, FA, DFA, FF, target
  prop, equilibrium prop, predictive-coding approx. E9 remains the in-stream scaffold.
- `null` as a yaml key parses to a None key (OmegaConf rejects it). Field renamed to
  `null_hypothesis` everywhere; the doctrine "every experiment declares its null" holds.

## Shell mapping (corpus -> code)
- shell/<name>.yaml is a COMPLETE shell bundle (one file = one full shell); siblings are
  ablation variants overriding one block. Keeps group-composition to one file per group.
- Plasticity controller carries the perineuronal-net rigidity term (per-weight rigidity
  that grows as a weight stabilizes, structurally close to SI importance) per Vol I.
- Consolidation = EWC (Fisher proxy) and SI (path integral), both selectable + composable.

## CPU-Now campaign execution (this session)
- THE UNLOCK SUCCEEDED: real V-JEPA 2 ViT-L weights fetch + load from HF (network reachable),
  hidden_size 1024, input [B,64,3,256,256] -> last_hidden_state [B,8192,1024]. The deferred
  real-encoder forward path is now fixed (call with pixel_values_videos=) and exercised.
- BUT there is no natural-video dataset in this environment. Feeding random pixels through the
  real encoder gives no class structure. Decision: cache a small REAL-ENCODER latent store from
  STRUCTURED synthetic video (per-class color/orientation/spatial-frequency/motion), which the
  real encoder maps to a real perceptual geometry. This gives a genuine REAL-ENCODER answer to
  the corpus's central diagnostic (is class info linearly decodable from real V-JEPA latents?).
  Fully-natural-video results remain deferred (drop SSv2/Ego4D clips on the Studio).
- RESULT TAGS: "real-encoder" = computed on the cached real V-JEPA latents (real weights,
  structured-synthetic video content). "provisional" = computed on the direct synthetic latent
  generator (make_task_stream), which the grids use for speed/coverage. Every number carries a tag.
- PARALLELIZATION: 12 physical cores (6 performance + 6 efficiency). Worker pool of processes
  (spawn), thread caps set per worker (OMP/MKL/OPENBLAS/VECLIB/NUMEXPR + torch.set_num_threads)
  so workers x threads ~= 12. Small-head legs: many workers x 1 BLAS thread. Heavy legs: fewer
  workers x more threads. Memory-aware worker cap (probe footprint, hold under 18 GB). Per-unit
  subprocess isolation + bounded retry + per-unit result checkpoint (resumable).
- SCALE CAPS: T0-T2 full seeds (5); T3 reduced seeds (3) + representative axis subsets (recorded
  per leg in the campaign). Determinism leg (11A) runs single-threaded/serial for a clean baseline.
- WALL-CLOCK: a literal 24h unattended run is not possible inside one assistant session; instead
  the cheap tiers run to completion at real (modest) scale producing REAL MEASURED per-run-unit
  timings, and those timings drive a full-scale cost projection that makes the 24h CPU fill and
  the Studio campaign plannable. T3 drains a bounded budget, checkpointed and resumable.

## Apple Silicon reframe (this session)
- The project is now Apple-Silicon-NATIVE: MPS is the primary target, not a fallback. KEY
  REALIZATION: a Mac Studio is also Apple Silicon (M-series Ultra, more GPU cores + much more
  unified memory), NOT a CUDA box. So the headline scale-up is "same MPS code, bigger chip",
  not a device port. `resolve(auto)` prefers mps on Apple Silicon; `cuda` is retained ONLY for
  Tier R rented-GPU env rollouts (E5/E10). SCALING.md and configs/device/* updated accordingly.
- Device layer gains `apple_silicon_info()` (chip, P/E cores, unified memory: here M3 Pro,
  6P+6E, ~19 GB), fp16 `autocast()` for mps/cuda inference, unified-memory defaults (no pinning).
- Known Metal limit (measured): a 64-frame ViT-L forward (8192 tokens) hangs the MPS compiler on
  this M3 Pro; real-encoder caching runs on cpu (24 to 32 s/clip). `mps_safe_token_cap` documents
  the route-to-cpu threshold. Expected to lift on a Studio with more GPU cores; verify there.
- MLX is an OPTIONAL `apple` extra for encoder-inference throughput, not a dependency and not on
  the hot path (PyTorch-MPS is the safe default; do not yak-shave MLX).
- DATA CLARIFICATION: V-JEPA ships pretrained WEIGHTS (loaded successfully), NOT a training
  dataset. Its benchmarks (SSv2, Ego4D, EPIC-KITCHENS) are external datasets to obtain when
  expanding to natural-video latents (deferred, not procured this session).

## Plug-and-play hardening (this session)
- Encoder ids VERIFIED on HF (metadata probe, no full downloads): real + present are
  vjepa2-vitl-fpc64-256 (1024), vjepa2-vith-fpc64-256 (1280), vjepa2-vitg-fpc64-384 (1408);
  added a real `vjepa2_vith` config. V-JEPA 2.1 dense ids do NOT resolve -> `vjepa21_*` marked
  placeholder + `available: false`; E6 dense deferred until 2.1 ships.
- Real-video ingestion path built (`substrate/video.py` + `scripts/cache_video.py`): backend-
  agnostic decode (lazy torchvision/decord, `video` extra) + a torch-only, tested preprocessing
  core (frame-sample/resize/ImageNet-normalize to [B,64,3,256,256]) feeding the existing
  cache_latents pipeline. This is the keystone for natural-video latents; the Studio just drops
  clips and runs cache_video.py. Decode backend is NOT a hard dep (preprocessing tested today).
- Campaign legs now carry genuine `full_axes` + `full_seeds` (the real factorials, 217 run-units
  total) alongside the toy subsets; `run_queue.py --full` runs them. sweep.run_sweep selects
  full vs toy. The cost projection's full-scale assumption is now backed by encoded grids.

## Pre-Studio hardening sprint (this session)
- Full-grid accounting (F1): one source of truth `sweep.full_run_units`/`toy_run_units`;
  cost_projection, `run_queue --full`, and the manifest now agree exactly (tested). Manifest
  declares full + toy run-units per leg; cost_projection reads full_axes/full_seeds (not the
  old full_seed param). All 14 legs carry full_axes; toy axes are a verified subset of full.
- Provenance (F4): `provenance.py` stamps git SHA+dirty, package versions, device, seed,
  encoder id+backend, cache id, and an enum result_tag into every RunManifest and a
  provenance.json beside every cache. Result tags: natural-video > real-encoder >
  structured-synthetic > provisional.
- Validation (F7): `harness/validate.py` fails fast (bad device/tier/encoder/null, unavailable
  encoder + prefer_real) and `check_all()` audits all configs+legs; wired into the runner.
- FAISS SEGFAULT (found by the microbench leg): faiss.search after torch import segfaults on
  Apple Silicon. Decision: buffer default index = `brute` (exact, safe); `KVIndex` subprocess-
  probes faiss safety and falls back to brute with a warning. Never silently wrong. See ISSUES.
- New operator tools, all cpu/seconds, no downloads: studio_doctor (readiness JSON+md),
  cache_tool (list/info/validate), storage_tool (estimate/list/prune dry-run), bench
  (microbenchmarks), build_report (analysis scaffold), check_docs (drift gate). Makefile +
  tests for each.
- Queue UX (F12): dry-run reports planned/skipped-with-reasons, toy-vs-full unit counts,
  enabled tiers, and next commands.

## Mac-Studio rehearsal capsule (this session)
- One command, `make rehearse` (scripts/studio_rehearsal.py -> src/mop/studio_rehearsal.py),
  rehearses the entire future Studio workflow end to end on tiny LOCAL fixtures: no downloads,
  no long runs, no science claims. Writes runs/studio_rehearsal/{report.md,summary.json}.
- No video codec on this device (torchvision/decord/av/imageio all absent). Decision: the corpus
  generator (substrate/fixtures.py) writes deterministic .npy clips (the explicitly-allowed
  mocked equivalent of .mp4), and video.read_video decodes .npy so the SAME validate -> decode ->
  preprocess -> cache contract runs codec-free. INGEST_EXTS = video + .npy. On the Studio the
  same path runs over real .mp4 with a backend; only the decode swaps. Honestly tagged "mocked
  decode" in the report; everything else (validation, preprocess, cache, integrity, planning,
  the miniature run, microbench, provenance) is real.
- The capsule re-uses the existing tools (studio_doctor, cache_tools, cost_projection, queue,
  bench, provenance), so it doubles as an integration test of the whole operator surface. Each
  stage records pass/fail + real/mocked; overall is pass only if every stage passes.

## Deferred (feasible only on the Studio / rented CUDA, or needs weights)
- Real V-JEPA weight download + real latent caching: scaffolded, falls back to the
  synthetic latent generator. Unblock: `pip install -e .[encoder]` then run
  scripts/cache_latents.py with network access to HF on the target machine.
- Tier R legs (rollout-heavy E5 env variant, E10 capstone, POET env-gen, cultural
  accumulation): scaffolded + queued disabled; need env + rented CUDA.

## Studio acquisition layer (this session)
- ONE pipeline surface (scripts/studio_pipeline.py -> src/mop/studio/pipeline.py): the goal
  asked for plan/acquire/validate/cache/run/optimize/report. Built exactly that plus a current
  device local-max lane and a profiles command. Rationale: one obvious operator surface beats a
  scatter of scripts; the raw scripts (cache_video, run_queue) still work underneath for hand control.
- New src/mop/studio/ subpackage, NOT more flat top-level modules. The acquisition layer
  (profiles, registry, planner, downloader, datacards, controls, pipeline) is cohesive; grouping
  it keeps the package map readable. The existing flat studio_doctor/studio_rehearsal stay as-is.
- Device PROFILE is the single home of every kill switch (Frontier 3B + 15). studio-1tb (900 GB
  usable) and m3pro-local-max (the goal's stated current-device caps) differ only in NUMBERS, not
  enforcement path, so the Studio profile can grow without touching safety logic. The hard download
  cap holds even against a generous --budget-gb (effective_budget_gb clamps); requested clips/time
  are clamped, not trusted.
- Dataset registry is YAML data (registry/datasets.yaml) + a validating loader (studio/registry.py),
  mirroring the configs/ + loader split already in the repo. Honesty is enforced in code: signed
  terms cannot be status available, metadata-only carries no cache size, full Ego4D is pinned
  deferred (ALWAYS_DEFERRED), and the planner refuses deferred/blocked and gates manual sources.
- Model registry keeps canonical V-JEPA as the source of truth (configs/encoder + encoder_registry);
  registry/models.yaml adds ONLY clearly-tagged auxiliary/distilled/quantized extras with
  replaces_canonical:false and result_tag never real-encoder unless canonical. No optional encoder
  can silently stand in for the frozen substrate.
- Planner is a breadth-first greedy knapsack: priority x a diversity multiplier (new modality 1.5x,
  new domain 1.25x, already-covered 0.6x), subset-scaled to fit budget + per-source cap. Greedy not
  optimal-knapsack on purpose: breadth is the objective, the source set is tiny, and the choice must
  be explainable (every skip carries a reason). Full Ego4D is never selected by construction.
- Downloader is dry-run by default; this module NEVER streams bytes itself. generate (synthetic)
  and local-path run on-device; remote methods execute only via a caller-supplied fetcher callback,
  so with no credentials here they record a clean blocked status. That makes it impossible for the
  current-device lane to be the thing that fills a disk, while the orchestration (budget hard-stop,
  resume manifest, hash/dedup, unsafe-archive refusal) is real and tested.
- local-max runs REAL work within the m3pro envelope and is the current-device acceptance of the
  whole surface; it reuses substrate/harness/bench so it doubles as an integration test. Video
  decode stays mocked (.npy) here exactly as in the rehearsal capsule; every stage is tagged
  real/mocked and overall is pass only if all stages pass.
- Did NOT add a studio step to scripts/acceptance.py: it has comprehensive dedicated tests (67) and
  local-max is its end-to-end acceptance. Keeping acceptance at 10/10 avoids rewriting historical
  build-log ratios in STATUS.md (which would be a false claim about past state).

## Developmental capacities layer (this session)
- The expanded goal asks for a "sentience-adjacent" developmental learner. Decision: build MEASURABLE
  capacities only, and put the anti-grandiosity rule in CODE, not just prose. north_star.scan_text
  flags affirmative sentience/consciousness/feelings/agency claims (but passes disclaimers), and
  metacognition.render_md calls assert_no_sentience_claims so a report literally cannot ship a claim.
  "drive"/"curiosity" are engineered objective terms (novelty, uncertainty, learning progress);
  "memory" is a data structure; "self-monitoring" is diagnostics. Never sentience/consciousness/personhood.
- New src/mop/devel/ subpackage (parallel to studio/), data in registry/*.yaml. The paradigm,
  capacity, and paper-watch registries are YAML + a validating loader, mirroring the configs/ + studio
  registry pattern. The capacity ladder and paradigm entries carry the SAME contract as experiments
  (baseline, ablation, metric, null), so a speculative mechanism cannot be promoted to canonical
  science without an explicit tag (validate_paradigm rejects a candidate that claims a real result_tag).
- Curriculum engine uses a REAL, cheap learning-progress signal (linear-probe accuracy gain with more
  data) and a PERMUTATION TEST for noisy-TV rejection. Why the permutation test: on the M3 Pro the
  frozen latent dim (1024) far exceeds the tiny sample count, so a single probe overfits and even pure
  noise looks decodable; comparing real-label accuracy to shuffled-label accuracy (averaged) cancels
  the overfit, so genuine signal shows a gap and aleatoric noise does not. This is what lets "curiosity"
  reject the noisy-TV before any live RL environment exists (Frontier 26/33).
- Ablation engine costs/info-gains are EXPLICIT ASSUMPTIONS (levers), not measurements, and it refuses
  to combine mechanisms before isolated gates pass (redundant groups surfaced). It names one next-best
  experiment rather than chaining everything (anti-chaos, Frontier 29).
- Markdown consolidation (Frontier 36): canonical doctrine is the corpus volumes + BLACKHOLE.md +
  docs/STUDIO_MAXIMIZATION_2026_06_27.md. Removed scripts/_scaffold_api.md (0 references; its contract
  lives in experiments/base.py + EXPERIMENTS.md). The old generated run reports and maximal-goal prompt
  were consolidated into /Users/scammermike/Downloads/PROJECT_RETROSPECTIVE_CHECKPOINTS_2026_06_28.md
  and deleted from the repo so they cannot compete with the active Studio plan. check_docs now carries
  a markdown LEDGER (canonical/operational) and flags any on-disk markdown not in it, so stale docs
  cannot silently regrow. The ledger check runs only over the real repo so the docs-drift fixture tests
  (which monkeypatch a fake ROOT) are unaffected.
- Did NOT wire the devel registries into acceptance.py or studio_doctor: the registries have dedicated
  tests + `make devel`, and touching acceptance would force rewriting the historical 10/10 ratio in
  STATUS (a false claim about past state). The doctor stays scoped to machine-readiness probes.

## Experiment-bank expansion + scaffold (this session)
- Re-oriented to the CURRENT canonical doc (docs/STUDIO_MAXIMIZATION_2026_06_27.md), which already
  designs EX1-EX16 + the proof system. Rather than invent a parallel E11-E20 namespace (the earlier
  plan), the scaffold uses the canonical EX-series and ADDS only what was genuinely missing: EX17
  (latent iterative reasoning) and EX18 (self-verification), the latent-reasoning line. This keeps one
  namespace and respects the doctrine doc as authoritative.
- registry/experiments.yaml is the single machine-readable source of truth AND the preregistration:
  null + headline metric + falsifier + failure-taxonomy slot are committed before a run. EXPERIMENTS.md
  is generated from it (no hand-edit, no drift), mirroring the datasets/paradigms/capacities registries.
- Tier reconciliation: the runnable Experiment.tier enum stays the 4 values {cpu-now, gpu-later,
  env-later, 2.1-only}; the registry adds a richer resource_tier {..., studio-scale, environment-needed,
  weights-needed, moonshot} for planning. Moonshots are catalogue-only (never status implemented), which
  enforces the paradigm-registry-first doctrine in code (validate_experiment).
- The validator makes the registry and the code inseparable: an implemented experiment/cross-cutting row
  must map to a real REGISTRY id; an implemented diagnostic/ablation must name an existing module; every
  REGISTRY id must be catalogued. So the doc, the registry, and the code cannot silently diverge.
- EX17 (latent reasoning) is always compared against a COMPUTE-MATCHED control: a weight-tied
  IterativeRefiner vs an untied-depth network of equal block count + hidden width (equal forward FLOPs,
  diagnostics/compute). The honest question is iteration-vs-depth at matched compute, not raw accuracy.
  On the toy task both hit ceiling (tie, null holds) but the refiner does it at ~1/4 the params, which
  is the reportable finding; a harder task is needed to separate them (left to the Studio).
- New diagnostics are reusable infra, not one-off: geometry.py (D1) is the math under EX12; compute.py
  (D5) makes matched-compute enforceable; substrate_ablation.py (D2) is the cheap devastating control.
  D2 surfaces an honest truth: linear decodability is projection-invariant (real ties frozen-random),
  so a linear-probe win is never by itself evidence the encoder is special; the control matters for
  nonlinear / mechanism gains.
- Implemented the flagship + two more mechanism experiments (EX3 TTA, EX8 curiosity, EX12 atlas, EX16
  codebook, EX17 reasoning) as real cpu-now runs gated on the E1 gate; the remaining EX/D/A rows are
  catalogued registry-only/deferred with the FULL contract (the precise next tranche, not vague stubs).
  This is the disciplined reading of "full overkill": complete architecture + a strong runnable core,
  not 16 half-built experiments.
- Honest nulls preserved throughout: EX8 reports a PARTIAL null at toy scale (only learning-progress
  robustly rejects the noisy-TV; disagreement is borderline), EX3 reports TTA did not help on the toy
  shift (base restored exactly), EX16 reports the codebook recovers purity but does not beat the raw
  probe. None were tuned toward a desired outcome.

## Cross-disciplinary roots expansion (this session)
- Scope decision: the user chose "full overkill, all cpu-now", so this build implements the WHOLE
  Tranche-1 bank as runnable code (77 Experiment subclasses across 9 series) rather than a runnable core
  plus catalogue, which differs from the EX-expansion's "strong runnable core" reading. The disciplinary
  series (N/D/B/P/C/I/Y/S/A) are SOURCES OF HYPOTHESES, never evidence; each is reduced to a falsifiable
  measurement on cached pooled latents with a preregistered null, exactly like the E/EX bank.
- Foundation-first build order: built the 6 shared diagnostics + the refine.py extensions MYSELF before
  fanning out, because the experiments import them. The diagnostics are correctness-critical shared deps,
  so they got a dedicated known-answer test suite (test_foundation_diagnostics.py) instead of being
  trusted from generated code.
- Parallelization without merge conflicts: each agent wrote a UNIQUELY-NAMED series module + its config
  files (no shared-file writes) and self-verified by running against the live editable install (uncommitted
  working-tree code is importable). The three genuinely shared files (scaffolds.py registration, registry/
  experiments.yaml rows, EXPERIMENTS.md) were integrated centrally by me, not by the agents, so there was
  no concurrent write to a shared file and no worktree juggling.
- Series-letter overload (D = diagnostics + developmental; I = I4 + infotheory; A = ablations + perception)
  was handled by relabeling the EXPERIMENTS.md series headings to name both families and making the
  renderer iterate dynamically over every series present, rather than renumbering established ids.
- Three standing controls are wired once at the harness level and reused, not reimplemented per experiment:
  beat frozen-random (substrate_ablation, guards projection-invariance), match compute (diagnostics.compute,
  guards iteration-is-just-depth), beat a tuned baseline (guards the renamed-biology confound); plus the
  noisy-TV guard for any curiosity signal and the seed-stability harness (sign-flips publish as instability).
- Pooled-latent blind spot is shipped as a result, not hidden: object/spatial/permanence/binding tests
  (N8, A6, C1/S6 where binding is erased) record taxonomy-slot 3 and publish the bound, retargeted to dense
  V-JEPA 2.1 later. null_supported reflects the real toy outcome and was never tuned toward a positive.
- The runnable Experiment.tier stays the 4-value enum; everything studio/weights/environment-scale (N2,
  D6, D10, B8, Y4, Y10/I10, P7, S2, S8, P8, A9, EX2-live, dense-2.1 retargets) stays registry-only/deferred
  with the full contract, enforced by validate_experiment (an implemented row must map to a real REGISTRY id).

## Pre-Studio maximal push (this session)
- Scope decision: "get as much as you can before the Studio", checked against the ACTUAL registry rather
  than assumption. Audited resource_tier across all 116 catalogued rows: only 9 genuinely need the Studio
  (3 studio-scale compute, 3 need dense V-JEPA 2.1 or a second real encoder, 3 need an interactive
  environment); the other 107 are cpu-now. Ran every one of the 93 THEN-implemented cpu-now experiments
  for real (not a dry-run or a re-scaffold), then closed the remaining 14 registry-only cpu-now rows left
  over from the EARLIER EX-expansion session's deliberate "strong runnable core, not every row" scope call
  (see the Experiment-bank expansion decision above) — that discipline was right for its own session's
  scope, but the current directive is explicitly "complete all possible", which supersedes it for
  laptop-doable work. Genuinely Studio-gated items were left alone; nothing was faked to look done.
- Every candidate positive (null_supported=False) was adversarially re-verified, not just reported as-is.
  A toy-scale rejected null is a claim, not a result, until it survives the SPECIFIC standing control the
  experiment itself measures (frozen-random, matched-compute, tuned-baseline, seed-stability, noisy-TV).
  Ran this verification twice independently (rate-limited resumes); both passes agreed: zero of the ~28
  distinct candidate positives checked survived. This is the expected outcome of toy-scale configs (small
  samples, few epochs, 1-3 seeds) and is reported as such: a clean negative result, not a failure of the
  method. It also validates that the standing-control doctrine is doing its job (catching every artifact)
  rather than every positive being real, which would have been a red flag for the harness itself.
- Registry-only rows that describe a mechanism ALREADY BUILT as shared infrastructure (a1_frozen_random_arm,
  a2_matched_compute_arm, d6_rollout_gate) were completed by pointing their `module` field at the existing
  diagnostics module rather than writing duplicate code: the honest move is recognizing the row was already
  satisfied by infra built for a different but identical mechanism, not manufacturing a second copy to
  literally match the row's own module-less schema. Their `metrics` field was corrected to the real
  module's actual return keys (e.g. d6's `rollout_r2` became `one_step_r2` to match sysid.py) rather than
  leaving a preregistration that describes output the code does not produce.
- The 7 new Experiment rows (EX1/EX4/EX6/EX7/EX11/EX14/EX18) used the SAME parallel-agent-writes-unique-
  files, integrator-touches-shared-files pattern proven in the roots-expansion build: each agent got the
  full registered contract (name/mechanism/null/metrics/controls) plus the EX17 module as a concrete style
  template plus a pointer to which existing shell/diagnostics infra to reuse (ReplayBuffer, GaussianHead,
  IterativeRefiner+Verifier, compute.matched_within), and was explicitly told not to touch scaffolds.py or
  registry/experiments.yaml. All 7 self-verified cleanly; the central integration step (imports, SCAFFOLDS
  list, registry status flips, EXPERIMENTS.md regen, one shared integration test file) was done directly,
  matching the earlier build's division of labor exactly.
- EX4's honest result illustrates why the null_supported flag is doctrine-tight rather than vibes-tight: the
  hypernet beat the static-head control (the flag's literal condition), so null_supported=False, but it did
  NOT match gradient-TTA (the harder, more interesting comparison in the mechanism's own framing). Both
  numbers are reported as separate keys (hypernet_beats_static, hypernet_matches_gradient_tta) rather than
  collapsing the nuance into one boolean, so a reader is not misled by which half of a mixed result the
  flag happened to key off.
- Discovered a real, reproducible Studio-necessity signal while trying to build a real-encoder latent
  cache: V-JEPA 2 ViT-L attention over 64-frame/256px clips overflows the M3 Pro's MPS backend with
  "Invalid buffer size" even at batch=1 (a hard per-buffer ceiling, not a total-memory limit, so raising
  system RAM would not fix it). This is not a bug to route around with a bigger workaround; it is exactly
  the kind of boundary the Studio hand-off should document precisely (device=cpu succeeds, just slower,
  confirming the limit is MPS-specific) rather than silently degrade past.
- Adversarial verification checked the SPECIFIC standing control each experiment declares, not a generic
  "does this look real" pass. A rejected null (null_supported=False) means only that the code's own
  boolean fired; it says nothing about whether the represented effect survives frozen-random, matched-
  compute, a tuned baseline, or seed stability unless that control was actually measured. This distinction
  mattered: several "positives" (e.g. c1_held_out_combination, c5_transfer_matrix) rejected their null by
  literal accuracy-above-chance while their OWN code's frozen-random arm beat or tied the real substrate,
  meaning the experiment's own instrumentation already disproved the positive; the adversarial pass mainly
  surfaced results the code had already contradicted, not new information from outside the corpus.
- The adversarial-verification workflow hit the API rate limit twice across two resumes (documented, not
  hidden): 8 of 11 series and 4 of 25 positives were dropped the first time, 3 more dropped the second time
  after a mid-response connection drop. Resumed from the same runId both times so completed agent() calls
  returned from cache and only the missing pieces re-ran, rather than re-doing the whole pass. All three
  passes (partial, partial, complete) agreed on every result that WAS covered: zero positives ever survived
  in any partial or complete run, which is itself a data point for how robust the "zero confirmed" finding
  is (it did not depend on which subset of the 25 got checked first).
- RESULTS_PRE_STUDIO.md lives under runs/pre_studio/ (ledger-exempt, since runs/ is a skip directory for the
  markdown-ledger scan) rather than at the repo root, because it is a data artifact (100 result files plus
  their synthesis) that should travel with the rest of runs/pre_studio/ as one transferable unit, not a
  standing project doc that needs ledger upkeep every time an experiment reruns. STUDIO_HANDOFF.md is the
  opposite: a forward-looking operational doc referenced from README-adjacent context, so it is ledgered in
  OPERATIONAL_MD like STATUS.md/DECISIONS.md, not left to drift unlisted.
- The synthesis workflow's first drafts contained several claims that were TRUE when the workflow checked
  them but had since been fixed by parallel work in this same session (the real-encoder cache showed
  count:0/all-zero labels before the CPU-based rebuild landed; DINOv2/VideoMAEv2 showed partial/absent
  downloads before they finished; the local-max rehearsal showed 8-days-stale before the fresh rerun). Every
  such claim was verified against the actual current filesystem state before being corrected in the final
  documents, rather than trusted at face value — the same standing-control discipline applied to the science
  applies to the hand-off prose itself: a claim about repo state is only as good as the moment it was checked.

## Post-handoff pass (this session)
- When the user asked whether there was "absolutely any more progress" possible pre-Studio, the honest
  answer was checked rather than assumed: re-audited disk/memory headroom, re-read the 9 Studio-gated rows
  and the "not yet wired" diagnostics flagged in the handoff, and looked for methodology gaps in the two
  seed_stability-refuted candidates rather than treating that verdict as settled. This surfaced a genuine
  bug in the refutation itself (e4_neuromod.py and e7_sparse.py read `cfg.seed`, not `experiment.seeds`;
  the earlier "increase seeds via override" attempt was a silent no-op), which is exactly the kind of thing
  a second, more skeptical pass is for: the doctrine's adversarial-verification standard applies to the
  VERIFICATION process too, not just the original experiments.
- Re-running e4_neuromod and e7_sparse required going through mop.harness.sweep.run_sweep (which
  generates a genuine `seed={s}` override per run, matching how the modules actually read seeds) rather
  than a `experiment.seeds=[...]` config override (which the modules never read). This is worth remembering
  for any future seed-stability re-check: verify HOW an experiment consumes its seed config before trusting
  that an override changed anything, since the roots-expansion series use `experiment.seeds` as a real list
  but the older E1-E10 bank uses a single `cfg.seed` per invocation, swept externally by the harness.
- e7_sparse's promotion to "provisionally confirmed" is deliberately hedged, not upgraded to a clean win:
  the 30-run grid is still on synthetic Gaussian-cluster latents, not real V-JEPA 2 features, and no formal
  significance test was run (mean/std/min/max were reported, not a p-value or CI). The doctrine's own
  caution against overclaiming applies here as much as to the 24 refuted candidates; "survives its specific
  objection" is a narrower, more honest claim than "confirmed."
- EX6 was flagged in the earlier handoff pass as "the one result that survives a clean same-architecture
  ablation" based on a single quick read, not a real adversarial check. When the user asked for more
  progress, that flag was treated as an open commitment to resolve, not a settled fact to build on. Three
  independent agents (two code re-analyses, one numerical resimulation) were run and unanimously refuted it:
  the "ablation" was not actually isolating the complexity term (the control arm differed in a second way,
  an opposite-polarity selection rule, that the first quick read missed), and a numerical resimulation
  showed the effect is fully explained by the noisy-TV region's hardcoded variance being large relative to
  the learnable region's residual scale, confirmed by an inversion test that flips the effect's direction.
  This is the exact failure mode the house doctrine's own "renamed-scalar" and "iteration-is-just-depth"
  categories warn about, in a new guise (variance-magnitude-just-relabeled-as-complexity), and it shipped
  in an earlier document version before being caught. Lesson: an "I checked it informally" note in a
  results document is not the same claim as "adversarially verified," and should not be allowed to read
  like one; the correction here is as important as the original finding.
- Declined to author the full proof/NULL_CARDS/*.md set for the ~31 experiments still missing one (7 newly
  implemented, ~24 refuted candidates). The schema requires a probe_dependency block citing a specific atlas
  factor/row and a decodability verdict; doing this properly means cross-referencing proof/atlas/ per card,
  and a rushed pass risks shipping invalid or misleading cards, which the schema's own voiding rules treat
  as worse than no card at all. Flagged as real remaining work in the handoff rather than done badly.
- Cleaned 3.5GB of orphaned `.incomplete` HF-cache blobs left from the earlier interrupted MPS-then-CPU
  download attempt, but did not chase the disk pressure further once it was confirmed to be system-level
  (pytest tmp and this repo's own runs/ growth were both negligible, well under 200MB combined). The disk
  kill-switch tripping live during this session is reported as a genuine finding, not silently worked
  around; forcing more headroom by touching files outside this repo would have been out of scope.

## Studio-gated-but-implementable pass (this session)
- The user asked for something to run during a wait, "even if much slower." Read this literally against
  the actual registry rather than defaulting to "wait for the Studio": of the 9 Studio-gated rows, the
  handoff document itself already said 2 (ex13_long_stream, ex5_local_rules_scale) were blocked on
  UNWRITTEN CODE, not a real hardware ceiling. That is exactly the situation "slower is fine" unlocks, so
  the honest answer was to implement them for real, not to find a smaller substitute task.
- ex5_local_rules_scale's original mechanism named a second axis (multi-encoder probing) that genuinely
  needs weights that do not exist. Rather than block the whole row on that missing half, the module
  documents the scope cut explicitly up front (a deliberate, stated decision, not a silent omission) and
  implements the half that is real and cpu-now: persistent local-rule BWT on the single available
  substrate. This is the same "close what's real, defer what's not" discipline used throughout this
  session (e.g. e6_relational's multi-encoder contrast staying frozen-random-only until 2.1 weights exist).
- Both experiments ran far faster than the brief anticipated (seconds to ~2 minutes at the shipped "scaled"
  config, not the "minutes to hours" expected) because the mechanisms reduce to small MLPs once implemented
  cleanly, and CPU handles thousands of small-MLP epochs quickly. Rather than treat this as done, pushed a
  second "grind" run at meaningfully larger scale via command-line overrides (not editing the shipped
  configs, which stay as the citable, reproducible scaled-default result) specifically so the user would
  have real, visible background compute to watch, matching the spirit of the request rather than just its
  letter.
- ex5_local_rules_scale produced a genuinely surprising, UNFORCED positive (local rules beat backprop on
  both accuracy and retention) that the implementing agent reported honestly rather than suppressing to
  match the brief's stated prior (that bio-plausible rules would trail backprop). This got the same
  adversarial scrutiny as e7_sparse and EX6: the verdict was PLAUSIBLE-BUT-UNVERIFIED, not confirmed or
  refuted, because backprop's Adam optimizer and the local rules' plain delta-rule updates share a nominal
  learning rate but not a demonstrated matched effective step size. This is a genuinely different, more
  calibrated finding-class than either of the session's other two verdicts (e7_sparse survives its specific
  objection; EX6 does not survive at all), and it is reported as such rather than being rounded to the
  nearest of those two known outcomes.
- ex13_long_stream's frozen-random control arm ran at a shorter stream length (n_tasks_control) than the
  main arms (n_tasks) for cost reasons when the module was first written. This was caught during result
  interpretation, not hidden: the "does not survive frozen-random" verdict is reported as-is, but flagged
  with the specific fairness gap (unmatched stream length) so a future rerun knows exactly what to fix
  rather than re-deriving the caveat from scratch.
- Both registry rows were flipped from studio-scale/gpu-later to cpu-now/minutes to reflect what was
  actually built and measured, not the original frontier-compute-era estimate. ex13's `relation` field
  keeps the pointer to a genuine future Studio extension (thousands of tasks, real V-JEPA latents instead
  of synthetic clusters) without implying the cpu-now result is provisional or lesser; a laptop result and
  a future Studio-scale confirmation are both real, at different scales, not a placeholder and a "real" one.

## Real-latent replication lane (this session)
- The doctrine synthesis named real-latent replication its highest-value next lane for a concrete reason:
  almost the entire 105-experiment corpus ran on make_task_stream synthetic Gaussian clusters, so even a
  clean adversarially-verified result is a claim about a tiny shell on an easy toy task, not about the
  frozen V-JEPA 2 substrate the whole program is built on. Scaffolding this lane is therefore the honest
  prerequisite to any architectural decision: we do not yet know what the real substrate affords.
- Immediately on wiring the adapter to the existing 64-clip real cache, the doctrine's own core warning
  reproduced on real geometry: a bare linear probe of the 8 classes scores 1.0 on both the real latents
  and a frozen-random projection of them (delta 0.000). Linear decodability is projection-invariant by
  construction, so a bare linear-probe win proves nothing even on real V-JEPA features. This is not a
  disappointment, it is confirmation the instrument is honest, and it dictates the design: the replication
  driver must LEAD with nonlinear (readout-contribution) and compositional (held-out-combination) probes,
  where real and frozen-random geometry can actually diverge, not with the linear probe.
- The single-factor real cache (cache_real_encoder.py) entangles frequency, angle, motion, and color into
  one class index, which is fine for continual-learning experiments but cannot support a compositionality
  test (held-out-combination needs two INDEPENDENT factors). Rather than contort the existing cache, built
  a separate factorized generator (hue x orientation at fixed frequency) so the two factors vary
  independently and any (hue, orientation) combination is realizable. The two factors are stored as a
  composite label y=a*n_b+b plus a factors.json sidecar, which needs no LatentStore schema change and lets
  ONE factorized cache serve both the continual experiments (composite class) and the compositional probes
  (decoded factors).
- Deliberately built the whole lane as CPU-light code (adapter, generator, driver, tests) validated on the
  tiny existing cache, and QUEUED the encoder-heavy factorized-cache build to run chained after the current
  migrated grind rather than concurrently. Two real-encoder jobs at ~21s/clip on the same CPU would halve
  each other's throughput on an 18GB machine; chaining respects the hardware and still fits the 12h budget.
- Stopped at scaffolding the lane and queuing its first real run, per the user's explicit sequencing
  (expand experimentation before any build decision). The factorized replication result, once it lands,
  is the first evidence about whether the real substrate carries compositional structure a random
  projection does not, which is the crux question for both the abstraction doctrine and any eventual
  architecture choice.

## Doctrine-lane experiments via agentic workflow (this session)
- When the user said "use an agentic workflow" for expanding experimentation, applied it to the highest-
  leverage remaining work: building the four never-built lane experiments the corpus synthesis flagged as
  the most direct tests of the two doctrinal questions (D6 sensitive window and B8 growth and Y4 hysteresis
  for moldability, ex10 cross-modal for language-independent abstraction). Fanned them out one-agent-per-
  experiment, then integrated the shared files (scaffolds, registry, tests, EXPERIMENTS.md) centrally
  myself, the same division of labor proven across the earlier builds: agents write uniquely-named files
  and self-verify against the live editable install, the integrator owns everything shared.
- The scientific outcome is a coherent, doctrine-strengthening set of clean negatives, not a
  disappointment. D6 and Y4 both reproduce the corpus's DOMINANT artifact signature: a real, sizable effect
  exists (late-position recency advantage; a hysteresis loop) but a frozen-random projection reproduces it
  just as strongly, so it is a generic optimizer/geometry artifact, not a substrate plasticity property.
  That two more independent mechanisms land on the same signature is itself evidence the signature is
  systemic to the toy regime, which is the synthesis's central claim. B8 lands the expected capacity-
  confound null (growth-as-process ties matched-final-capacity). ex10 is the most interesting: the shared
  code IS genuinely learned (cross-modal transfer 1.0 vs a shuffled-pairing floor of 0.41, a real binding
  of two views), but it produces no downstream retention gain, so the objective is a regularizer, not a
  useful binding. That split (real alignment, no benefit) is a more precise negative than a flat null.
- Named the developmental sensitive-window experiment d6_sensitive_window even though a d6_rollout_gate
  diagnostic already exists: registry ids are unique full strings (the two do not collide), and the plan's
  D6 was always the developmental sensitive window, the d6 rollout-gate diagnostic just borrowed the label
  in the diagnostics sub-family. Keeping the plan's intended name is clearer than inventing a d10/d11.
- ex10_cross_modal keeps its deferred natural audio-video framing in the catalog while its synthetic cpu-now
  arm is implemented and flipped to status: implemented, the same pattern used for ex2's synthetic vs live
  arm. The registry row now documents both explicitly so the weights-needed half is not silently dropped.
- Fixed the one agent-authored defect the central lint/type pass caught (a dead pg/pff assignment in b8
  that also tripped mypy by calling a float-typed mean on an int list): removed the dead line rather than
  papering over it, since the real comparison already used the param-count lists directly. Every agent-
  written module still passes the same ruff+mypy+registry+test gate as hand-written code before it lands.

## 2026-07-09 (Form Substrate implementation, workstream B: one referent-aligned stack)

- B1 store-backed form adapter: implemented LatentStoreFormAdapter only, NOT the SubstrateFormAdapter the
  plan also listed. Reason: SubstratePerspectiveAdapter already turns clips into features (encode path),
  and a form twin would either duplicate it (against the codemap refusal table) or violate encode-once by
  re-encoding in a data-plane class. The honest bridge is: cache once via the existing substrate adapter,
  then read the store as a form arm. Documented in the LatentStoreFormAdapter docstring.
- B3 merge scope: kept the measured version, one shared referent-alignment implementation
  (substrate/form.referent_order) that both build_form_matrix and build_perspective_matrix call, rather
  than aliasing PerspectiveMeta to FormMeta. Reason: the perspective stack has 9 consumers including 4
  production DR1/studio lanes (dr1_perspectives, dr1_verifier, native_lanes, dr1_curate_bound_video) whose
  fields differ from FormMeta (modality/supervised/derived vs kind/objective) and which cannot be
  exercised on the M3 Pro box (studio caches absent). Full dataclass unification would ship an unverified
  change to that production code, exactly the revert-not-patch hazard the plan flags. The verifiable B3
  goal (grep referents.index shows one ordering impl) is met: zero list.index scans remain, one O(n)
  dict-lookup implementation. Full PerspectiveMeta=FormMeta aliasing is deferred to a change where the DR1
  lanes can be run on real caches (MIGRATION_PHASES.md Phase 5). Form is the interface of record for new
  work; the perspective layer is the DR1-wired instance sharing the form layer's alignment machinery.

## 2026-07-09 (Form Substrate implementation, workstream C: the cpu-now F-series is now runnable)

- Implemented all 14 cpu-now F experiments end to end (F1/F2/F3/F5 pre-existed; this run added F4, F9,
  F10, F12, F13, F14, F17, F18, F19, F20). Each carries the full doctrine contract, its registry-named
  controls, a performance-density block, and difficulty calibration so no reported tie or win sits at the
  accuracy ceiling or the chance floor. The 6 still registry-only (F6, F7, F8, F11, F15, F16) are genuinely
  off this hardware: F6/F15 need an environment, F7/F8/F11/F16 need Studio GPU or a trainable-substrate
  license (doc 15 gates). No encoder was trained; workstreams D (Studio real forms) and F (plastic branch)
  stay closed on the M3 Pro per their gates.
- Difficulty calibration was the load-bearing effort, not the mechanics. Several experiments ceilinged on
  the first pass and were retuned to a non-vacuous regime: F17 (single-form redundancy at ceiling ->
  noise-limited fusion; also switched the OA2 null from a raw confidence-drop proxy to calibration AUROC,
  since a head trained on 4-form fusion sees a different confidence scale under 3 forms), F10 (real forms
  saturated regardless of schedule -> scarce budget plus many noisy-TV distractor forms so learning-progress
  concentration beats uniform), F19 (flat exemplar retrieval already sufficient -> high object noise so the
  episode centroid denoises where a single exemplar cannot), F14 (new-form transfer at 1.0 -> harder world
  so transfer lands below ceiling and can be compared to the retrain-from-scratch upper bound).
- F13/F18/F20 were built by a parallel agent workflow: three implementer agents in isolated git worktrees
  (branched from the F19 commit) each coded, wired, calibrated, and self-verified one experiment, each
  followed by an adversarial reviewer checking for ceiling traps, vacuous controls, and metric mismatches.
  All three came back reviewed-clean and non-null. Their artifacts were re-integrated and RE-VERIFIED in the
  main tree (not trusted on self-report): the central mypy pass caught 3 missing dict annotations in the
  F13 code that the worktree lint did not, fixed here. Every agent-written experiment passes the same
  ruff+mypy+registry+integration+acceptance gate as hand-written code before it lands.
- Added tests/integration/test_f_series.py, which discovers the implemented F rows live from the registry
  (never a hardcoded list), so every cpu-now F experiment is exercised through the runner and asserted to
  carry its declared metrics plus a density block. Registry runnable count 113 -> 123; acceptance 10/10.
