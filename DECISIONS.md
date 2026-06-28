# DECISIONS

Autonomous-session decisions, each with a one-line rationale. Append-only.

## Doctrine
- BLACKHOLE.md (dropped in as CLAUDE.md) governs code FORM (density, flat structure,
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
- One command, `make rehearse` (scripts/studio_rehearsal.py -> src/devsys/studio_rehearsal.py),
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
- ONE pipeline surface (scripts/studio_pipeline.py -> src/devsys/studio/pipeline.py): the goal
  asked for plan/acquire/validate/cache/run/optimize/report. Built exactly that plus a current
  device local-max lane and a profiles command. Rationale: one obvious operator surface beats a
  scatter of scripts; the raw scripts (cache_video, run_queue) still work underneath for hand control.
- New src/devsys/studio/ subpackage, NOT more flat top-level modules. The acquisition layer
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
- New src/devsys/devel/ subpackage (parallel to studio/), data in registry/*.yaml. The paradigm,
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
