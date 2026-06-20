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

## Deferred (feasible only on the Studio / rented CUDA, or needs weights)
- Real V-JEPA weight download + real latent caching: scaffolded, falls back to the
  synthetic latent generator. Unblock: `pip install -e .[encoder]` then run
  scripts/cache_latents.py with network access to HF on the target machine.
- Tier R legs (rollout-heavy E5 env variant, E10 capstone, POET env-gen, cultural
  accumulation): scaffolded + queued disabled; need env + rented CUDA.
