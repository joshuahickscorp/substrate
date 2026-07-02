# M3 Pro Run Report: the Mixture-of-Perspectives maximization run

This is the empirical result of the full laptop (M3 Pro, 18GB) maximization run for the Mixture of
Perspectives (MoP) program, executed 2026-07-02. It is the artifact the Studio run is planned against.
Every result is graded by evidence strength so nothing is over-claimed. House style: no em or en dashes.

## 0. How to read this (evidence grades)

- **REAL-DECISIVE**: real frozen V-JEPA encoder, non-vacuous control (random-init same-arch ViT at matched
  resolution, never a square latent projection), statistically significant. These are the science.
- **PILOT-HELD**: a cached-latent or synthetic-regime experiment whose effect beat its non-vacuous baseline
  AND survived a 10-seed rerun without a sign flip. Real evidence the effect EXISTS, but the registered /
  at-scale claim is deferred to the Studio.
- **NULL**: the mechanism tied its baseline at matched compute. Recorded as an asset (a citable bound), not
  a loss. Many of these replicate prior corpus nulls exactly.
- **UNSTABLE**: rejected its null at 5 seeds but reverted (or sign-flipped) at 10 seeds. Demoted to null,
  honestly.
- **NOT-EVALUABLE**: the task self-reported a calibration failure (no hardness gradient, out-of-band
  control) and refused to issue a verdict rather than fake one. Needs recalibration, not a rerun.

Tally of the ~30 distinct MoP experiments: 24 null-supported, 5 pilot-held survivors, 3 unstable, 5
not-evaluable, on top of the 3 real-decisive substrate results and the PR1 router-licensing gate.

## 1. The real-decisive results (the science)

These used the real frozen encoder and non-vacuous controls, and they are the load-bearing findings.

| Finding | Result | Statistic | Grade |
|---------|--------|-----------|-------|
| The frozen substrate is SPECIAL because of pretraining | real V-JEPA 0.517 vs random-init same-arch ViT-L at matched 256px 0.241 (not above chance) vs random-pixel 0.103, decoding shape under heavy nuisance | V-JEPA vs chance p=1.6e-5; V-JEPA vs random-init p=0.029 | REAL-DECISIVE (single split; Studio multi-seed owns the headline) |
| The substrate FACTORS bound attributes compositionally, off ceiling | held-out (shape,color) 0.725 = seen 0.708 (gap -0.017); random-pixel collapses to 0.05 | held-out vs chance p=1.2e-12; vs random-pixel p=1.3e-10 | REAL-DECISIVE for the within-arm factorization; substrate-specific attribution needs the matched-resolution rerun (Studio) |
| Reasoning modes make decorrelated errors (a router is licensed) | heterogeneous oracle gain 0.155 vs homogeneous seed-copy oracle 0.118 + seed SD 0.014 | PR1 GREEN | REAL-DECISIVE as an existence result; an oracle is an upper bound, a learned router still has to reach it |

The through-line: the frozen pooled V-JEPA latent is a genuinely structured, compositional perceptual
code (not a blank projection), and its heterogeneous readings are complementary enough that a coordinator
is worth building. Those are the two prerequisites MoP needs, and both held.

## 2. The pilot-held survivors (5 of ~30, held at 10 seeds)

These beat their non-vacuous baseline and survived the 10-seed rerun. They are cached-latent or
synthetic-regime pilots: the effect is real, the at-scale claim is a Studio job.

- **at3_time_axis (temporal currency exists).** Full-clip latents decode motion direction and speed that a
  token-matched static frame cannot (needs_time = motion_dir4, speed2). The temporal axis is a genuine
  perspective carrying decodable structure a static view lacks. Directly supports the multi-perspective
  thesis.
- **ws2_fusion_tournament (structure beats capacity).** At matched parameters, a structured fusion beats a
  concat-MLP: the workspace adds value BEYOND raw capacity. This is the cleanest pilot-scale positive for
  the core MoP coordinator, because it clears the "just more parameters" confound.
- **al2_alignment_pilot (perspectives share an alignable code).** A learned rank-32 cross-substrate map
  predicts a target substrate above the random-map-of-equal-rank floor. This is MoP's empirical
  precondition: different perspectives can be aligned and translated, above the vacuous floor. Conceptually
  the most important survivor.
- **at1_grid_pilot (pretraining-invariance is cross-substrate).** The substrate-is-special effect
  generalizes across the laptop columns (image and single-frame video), each beating its OWN random-init
  control. Not a V-JEPA-only artifact.
- **pr7_fast_slow (two-timescale plasticity helps).** A fast (Hebbian) plus slow (SGD) store beats the
  slow-only baseline. The deep research flagged the Hebbian version as a FLOOR that a delta-rule update
  provably dominates, so this is a real lead to upgrade, not a ceiling.

## 3. The nulls (24), and why the biggest cluster is the correct result

The entire test-time-compute lane is null at matched compute, and this is the CORRECT general result for
our regime, not a failure (confirmed against Snell 2024, Saunshi 2025, Geiping 2025, Kamoi 2024 in the
round-2 deep research):

- verify-revise ties single-shot at matched FLOPs (replicates ex18).
- beam search ties deeper greedy at matched FLOPs (the oracle-beam gap proves headroom the learned scorer
  cannot reach: the generation-verification asymmetry).
- latent debate ties max(single, ensemble) at matched FLOPs (replicates ex17).
- confidence-stop ties the free update-norm rule; DR8 fixed-point has no attractor on either the V-JEPA or
  the random-init cache (it is unrolled depth, the n9/y1 result).

Iterative compute pays only on difficulty-graded, depth-bottlenecked, verifiable tasks; our additive
1024-d regime is none of those. These nulls are rescuable ONLY after a certified hardness gradient exists,
with an honest kill-switch if they still tie on a genuinely hard, verifiable task. The plasticity nulls
(uncertainty gating, etc.) largely replicate the e3/e4 negatives (biological schedules tie tuned
baselines).

## 4. The honest deflations

- **dr2_sparse_real is UNSTABLE.** The e7_sparse architectural forgetting positive rejected its null at 5
  seeds but reverted at 10 (sign flip). So the sparse-heads win is weaker on the real cache than on
  synthetic data, and does not robustly replicate at 10 seeds here. The 30-run Studio protocol must settle
  it before it is claimed.
- **pr5_content_gated_cp and ws5_router_slot both sign-flipped** and are demoted to null. The critical-
  period schedule and the router-slot ablation showed no stable effect.
- **ex2 planning (a standing positive) is a synthetic toy** that touches no V-JEPA latent (flagged in the
  semantic-positions audit): it is a positive about a hand-written 8-d MPC, not about the substrate.

## 5. Not-evaluable (recalibration, not rerun): mt5 halting, al1 uncertainty router, dr12 disagreement,
ws3 arbitration, at4 programmatic-ceiling (the last is a guard, not a null). Each self-reported an
out-of-band control or absent hardness gradient and refused a verdict. The honesty machinery worked: these
go on the recalibrate list.

## 6. What this means for MoP, and the Studio handoff priorities

The laptop run establishes the FRAME and clears the two prerequisites (structured substrate, complementary
modes), plus three pilot-scale positives that are genuinely MoP-relevant: time is a real perspective (at3),
a structured coordinator beats capacity (ws2), and perspectives share an alignable code (al2). It also
delivers a large, honest null map (the reasoning lane is regime-correct-null) and one honest deflation
(sparse heads do not robustly replicate on real latents).

The binding constraint remains what it has been: almost every cross-modal SEMANTIC question (code, math,
physics, paired-language) has no runnable test surface yet, because the cached substrate exposes only shape
and color and we have no DSL/executor, physics, numerosity, or paired-language-encoder caches (per
SEMANTIC_POSITIONS.md). The Studio priorities, in order:

1. Multi-seed the substrate headline numbers (the +0.276 and the compositional -0.017) with the random-init
   arm at matched resolution: turn REAL-DECISIVE-single-split into REAL-DECISIVE-multi-seed.
2. Promote the 3 pilot-held MoP survivors (al2 alignment, ws2 fusion-beats-capacity, at3 temporal currency)
   to their registered at-scale claims with the full control stack.
3. Settle dr2 sparse-real with the 30-run protocol.
4. Build the modality caches (DR1 bound-attribute video, then a DSL/executor, physics, numerosity, and a
   paired-language cache) so the semantic layer stops being theory without an instrument.
5. PR9 continual-backprop reinit (the one frontier-certified baseline-beater never run) on a stream long
   enough to induce plasticity loss.

## 7. Bottom line

The frozen V-JEPA substrate is real, structured, compositional, and its perspectives are alignable and
complementary: MoP is a live hypothesis, not a refuted one. The mechanism lane is mostly honest nulls (the
reasoning nulls are the correct regime result), with five real pilot-scale survivors, three of them
squarely MoP-relevant. Nothing here licenses custom-model training. The deliverable is exactly what the
doctrine promised: a rigorous map of what a frozen-substrate cognition can and cannot do, with the sharp
negatives that a scale-first lab has no incentive to publish, and a short list of real positives to scale.
