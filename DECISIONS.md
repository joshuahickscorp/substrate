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

## Deferred (feasible only on the Studio / rented CUDA, or needs weights)
- Real V-JEPA weight download + real latent caching: scaffolded, falls back to the
  synthetic latent generator. Unblock: `pip install -e .[encoder]` then run
  scripts/cache_latents.py with network access to HF on the target machine.
- Tier R legs (rollout-heavy E5 env variant, E10 capstone, POET env-gen, cultural
  accumulation): scaffolded + queued disabled; need env + rented CUDA.
