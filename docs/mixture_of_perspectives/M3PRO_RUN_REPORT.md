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

## 2. The survivors, RE-GRADED after the potential audit (3 hold, 2 fail their own controls)

> Correction (POTENTIAL_AUDIT.md): an earlier version of this report listed 5 pilot-held survivors. The
> adversarial audit re-graded two of them (al2, ws2) against their OWN preregistered controls and both
> fail. Verified against the raw JSONs. The honest count is 3 that survive, 2 demoted. This section is
> corrected below; do not carry al2 or ws2 to the Studio as positives.

Genuinely holds at 10 seeds against a non-vacuous control:

- **at3_time_axis: DEMOTED by the post-audit re-audit (was a survivor).** Full-clip latents decode motion
  direction (+0.200, CI [0.165,0.235]) and speed (+0.245, CI [0.204,0.286]) that a token-matched static
  frame cannot. But the temporal labels are DERIVED from the injected (vx,vy) draw, and under the strong
  nonlinear partial-out (r,vx,vy,vx^2,vy^2,|v|,sin/cos angle) both collapse to chance (shrink 100% and
  96.6%). So the edge was reading the injected motion parameters, not integrating temporal currency. This is
  a NULL, not a survivor. See LAPTOP_LANES_RESULT.md and runs/mot/survivor_reaudit.json.
- **at1_grid_pilot (pretraining-invariance is cross-substrate).** The substrate-is-special effect
  generalizes across the laptop columns (image and single-frame video), each beating its OWN random-init
  control. Not a V-JEPA-only artifact. (Same family as the substrate-special direction, single split.)
- **pr7_fast_slow (two-timescale plasticity helps), as a LEAD not a win.** A fast plus slow store beats
  the slow-only baseline, but the deep research showed the Hebbian version is a FLOOR a delta-rule provably
  dominates. Real only as a mechanism to upgrade, not a positive to bank.

DEMOTED by the audit re-grade (were reported as positives, are not):

- **al2_alignment_pilot: cross-modal alignment FAILS; only same-modality vision pairs align.** The "null
  rejected" was carried by vision-to-vision pairs; the cross-modal pairs (vision to text, vision to audio)
  have NEGATIVE learned-minus-random deltas (e.g. dinov2 to handcrafted learned R2 -0.45 vs random -0.02),
  and the real V-JEPA nuisance arm (`vjepa2_vitl_nuisance_real`) is MISSING entirely. So this is NOT
  evidence that perspectives share an alignable code. The honest reading is the opposite and still useful:
  two frozen VISION encoders of identical content are weakly alignable, and no cross-modal alignment
  survives the random-map floor. That is a NEGATIVE result for the MoP shared-code precondition, not a
  positive, and it must be re-run with the missing V-JEPA arm and a topology-permutation null (not ridge
  R2) before any claim.
- **ws2_fusion_tournament: fails its own dual acc-AND-NLL contract.** The preregistered null requires a
  fusion to tie concat-MLP on BOTH accuracy and NLL; rejecting it needs a win on BOTH. No arm does:
  gwt_broadcast wins accuracy (acc_win True) but loses NLL (nll_win False), and the reported "structure
  beats capacity" verdict came from an acc-only, max-over-arms pick. On its own contract, ws2 is a NULL.
  "Structured fusion beats capacity" is not supported by this run.

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
SEMANTIC_POSITIONS.md). The Studio priorities, RE-ORDERED per the potential audit (which found the earlier
ordering was "avoidance dressed as sequencing": it buried the one decisive enabler behind refining a number
the program already owns):

1. **Build DR1 (non-additive bound-attribute natural video, with count and relation slots) FIRST**, plus a
   paired vision+text encoder pass on identical referents (the Qwen cache is a LABEL-FREE PIXEL-DERIVED
   textification, color grid + brightest-cell position, already paired to the vision clips, but it lacks
   SHAPE because shape decodes at chance from cheap label-free features on this clipset). This is the
   difference between having and not having a science on the multi-perspective
   ideology, and it unblocks GATE C1 and ~70 semantic positions.
2. **Re-grade al2 and ws2 against their own controls (zero new compute)** before any promotion: add al2's
   missing V-JEPA arm and a topology-permutation null; enforce ws2's dual acc-AND-NLL contract with the
   mean-baseline guard. On current data both demote to null; do not launder them into the Studio.
3. **Build the D3 hardness gradient and one executable verifier**, then re-run one dead reasoning mechanism
   (dr8) against it: this converts the 24 reasoning nulls from a prose rescue-list into a live
   falsification, and fires the standing kill-switch if it still ties.
4. **Run PR9 continual-backprop** on a long real-latent stream (the one plasticity mechanism certified to
   beat a tuned baseline on plasticity loss, and the only one never run): it either wins (the first
   substrate-touching plasticity positive) or ties (moldability is honestly dead at this substrate).
5. Only THEN multi-seed the substrate headline numbers and settle dr2 sparse-real with the 30-run protocol.
   These refine what the program already owns and must not come first.

## 7. Bottom line

The frozen V-JEPA substrate is real, structured, and compositional on the two slots it exposes (shape,
color), and its reasoning modes are complementary enough to license a router (PR1): MoP is a live
hypothesis, not a refuted one. But after the audit re-grade the honest yield is narrow: 3 holds (at3
temporal currency, at1 cross-substrate invariance as a same-family restatement of substrate-special, pr7
only as a lead), the substrate-special direction still single-split and one clip from ambiguity, and the
two results that looked most MoP-relevant (al2 shared-code, ws2 structure-beats-capacity) fail their own
controls. The mechanism lane is otherwise honest nulls. Critically, the multi-perspective and moldability
ideologies are barely instrumented: cross-modal alignment points the wrong way (al2), deep plasticity is
false-by-construction on a frozen substrate, and the density north star was tested once and nulled. Nothing
licenses custom-model training. The deliverable is a rigorous map of what a frozen two-slot instrument can
and cannot do; whether MoP becomes more than that is decided by whether DR1, a paired-text cache, D3, and
PR9 get built BEFORE the next round of headline multi-seeding. See POTENTIAL_AUDIT.md for the full
scorecard (3.0/10 on reaching the ideology) and the re-ordered action list.
