# Laptop Lanes Result: one axis per ideology, run in parallel, adversarially verified

This is the consolidated result of four laptop lanes run in parallel (one per audit ideology axis) alongside
the A6 workflow, then integrated. Every lane had a build agent and an independent adversarial verifier, and
every headline below is one that SURVIVED its verifier. House style: no em or en dashes. Companion:
`A6_RESULT.md` (the abstraction axis), `POTENTIAL_AUDIT.md` (the scorecard this updates).

## 0. What moved, in one paragraph

The falsification engine got sharper by turning on the positives, and it cost a headline: at3 temporal
currency is DEMOTED (its motion and speed decode were reading the injected velocity draw, not integrating
time). Density got its first real instrument: two honest, gaming-guarded Pareto frontiers (capability per
FLOP, capability per parameter) plotted from existing data, moving density from an unmeasured 3 to about
4.5, though the frontiers show the mixture arms DOMINATED by single specialized modes. The density mechanism
bet came back a clean null: a trained router on the real cache loses to both a tuned best-single reader and
a compute-matched homogeneous bank. Moldability did not move (it cannot on the laptop), but its Studio
instrument is now validated and turnkey. The through-line with A6 is exact: on this synthetic clipset,
apparent positives keep resolving to the injected generative nuisance (velocity, position, size), not to
abstract structure. That is the single most important thing the laptop can teach before the Studio.

## 1. Falsification lane: positive-survivor re-audit (axis 6/10 -> 7/10)

Applied A6-grade adversarial controls to the three Studio-bound positives and the substrate headline
(`runs/mot/survivor_reaudit.json`, `scripts/mop_survivor_reaudit.py`). Verdicts:

- **at3 temporal currency: DEMOTE.** The temporal labels (motion_dir4, speed2) are derived from the injected
  (vx, vy) draw. Under the honest strong control (project out r, vx, vy, vx^2, vy^2, |v|, sin/cos of the
  motion angle, train-fit), the full-vs-single-frame edge collapses: motion_dir4 shrinks 100 percent,
  speed2 shrinks 96.6 percent, both to chance. at3 was reading the injected motion parameters, not
  integrating temporal currency. This removes one of the three positives before it reached the Studio.
- **at1 cross-substrate invariance: HARDEN.** The two survivors' per-clip shape-probe correctness is only
  modestly correlated (phi 0.329), so they contribute genuinely independent invariance evidence, not one
  signal double-counted.
- **pr7 plasticity: HOLD.** The delta-rule null holds (CI entirely below the Hebbian floor). The Hebbian
  fast-store gain is real but tiny (+0.029, CI lo 0.027): a single-mechanism flicker, not a headline.
- **substrate-special: HOLD, fragile.** Bootstrapping the single 29-clip split, 63.7 percent of resamples
  keep the Fisher p below 0.05, and a one-clip adverse swing crosses it (p 0.0877). The direction is
  corroborated by the 200-clip on-disk caches (delta CI lo 0.504), but the single-split p is fragile as the
  file admits. Multi-seeding it is Studio B5, still LAST.

Why this raises the axis: the audit's 6-not-8 gap was that rigor was applied to nulls and not to the
positives. This lane applied it to the positives and demoted one on the merits.

## 2. Density lane: the first capability-density frontiers (axis 3/10 -> ~4.5/10)

`runs/mot/density_frontier.json`, `scripts/mop_density_frontier.py`. Before this lane the word "density"
appeared zero times in the run report and nothing was scored as a ratio, though density is the north star
and `mop/diagnostics/compute.py` already did matched-FLOP accounting. Now two of the four density axes have
real, non-trivial, gaming-guarded Pareto frontiers computed from existing data:

- **capability per FLOP** (source `mt123_router_pilots`): 5 arms, frontier = {reactive, sparse}; planner,
  routed, and blend_full are dominated. The routed mixture is strictly dominated by its own best single mode
  (sparse) at matched compute, consistent with the source mt1 null. FLOP counts are the real per-sample
  convention from the source config.
- **capability per parameter** (substrate shape-decode, recomputed via `linear_probe`): DINOv2 Pareto-
  dominates all: 0.861 shape accuracy at a 384d readout, about 3x V-JEPA per readout-parameter. The ranking
  survives a common-128d-readout control, so it is substrate quality, not readout capacity.
- **retention per byte** and **adaptation per update**: no existing run exposes bytes-per-exemplar or a
  matched updates/adaptation ratio, so these are marked Studio-only and left unfaked.

Honest reading: this closes a MEASUREMENT gap, not a capability gap. The density story the frontiers tell is
"single specialized modes win, the mixture is dominated," which is why the axis moves to about 4.5 and not
higher, and half the density north star (retention, adaptation) stays a real Studio gap.

## 3. Density mechanism lane: trained router on the real cache (null, an asset)

`runs/mot/router_mechanism.json`, `scripts/mop_router_mechanism.py`. Attempted to convert PR1's
oracle-existence result into a trained-mechanism win. Result: NULL, matched-compute honest. A trained
heterogeneous router (shape under nuisance, 0.860) loses to both a tuned best-single DINOv2 reader (0.870,
best on all 10 seeds) and a compute-matched homogeneous 40-copy bank (0.876). Both preregistered rejection
gates fail (router_vs_best CI lo -0.034 with a sign flip; router_vs_homo CI lo -0.043 with a sign flip). The
density-mechanism null the audit flagged is confirmed on the REAL cache, not just the synthetic mt123
regime. This kills the "maybe a trained router beats the baseline on the real cache" hope cleanly.

## 4. Moldability lane: the plasticity-loss certificate, validated (axis stays 2/10)

`runs/mot/plasticity_certificate.json`, `scripts/mop_plasticity_certificate.py`. The laptop cannot induce
Studio-scale plasticity loss, so the score does not move, and this lane does not pretend otherwise. What it
delivers is the down-payment: the instrument Studio PR9 needs, validated to fire and not false-fire. Under a
fixed plain-SGD baseline over a 150-task stream, the certificate measures early-minus-late learning
accuracy. On a concept-drift stream it FIRES (gap +0.513, CI [0.498, 0.528], dead ReLU units rise 0 to
0.75, the mechanistic signature PR9 targets). On a matched stationary stream it stays QUIET (gap ~0, CI
contains 0). The verifier added the decisive position-vs-identity control (shuffled task order preserves the
gap, fresh-net difficulty is flat across stream position), upgrading the interpretation from asserted to
demonstrated. Studio PR9 is now turnkey and de-risked.

## 5. The through-line, and what it hands the Studio

A6 found the cross-modal shared code is nuisance geometry, not semantic abstraction, and its shape-axis bet
was a bounding null even with a caption that provably carried shape. This re-audit found at3 was reading the
injected velocity, not time. Same lesson twice: on a synthetic clipset where position, size, orientation,
and motion are injected as nuisance, apparent higher-order positives keep resolving to that injected
nuisance. This is not a defect of the method, it is the method working: the partial-out controls are doing
exactly their job. It hands DR1 a precise mandate: real video is what dissociates an abstraction from the
injected nuisance, so DR1 must (a) enforce the A6 acceptance gate (the target attribute is label-free
recoverable in the caption, probe-verified) and (b) keep the nuisance-residualized alignment as the success
criterion, never raw alignment.
