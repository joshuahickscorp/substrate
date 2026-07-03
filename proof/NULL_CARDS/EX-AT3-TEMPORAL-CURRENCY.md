# Null card: at3 full-clip temporal edge is carried by injected velocity, not temporal integration

Registry-completion entry (axis-ceiling falsification lane). Form per BLACKHOLE.md:
no em or en dashes, no agency language. Encoder frozen; only linear reader heads and a train-fit
residualizer are computed.

## Claim under test

at3 (time axis): full-clip V-JEPA latents decode motion factors (motion_dir4, speed2) better than
single-frame latents of the same clips. Read as a temporal-currency claim, this says the full-clip
representation integrates motion over time in a way a single frame cannot, i.e. it carries genuine
temporal structure.

## Control (strong motion residualization, train-fit)

The temporal labels are DERIVED from the injected (vx, vy) draw in nuisance.npy: motion_dir4 is the
quadrant of atan2(vy, vx), speed2 is a median split of sqrt(vx^2 + vy^2). So both labels are exact
functions of the generative velocity, not observed dynamics. A LINEAR (r, vx, vy) partial-out cannot
remove a magnitude (speed) or a quadrant (needs sign structure), so a linear-only control lets the
factor survive spuriously. The strong control projects out the ground-truth motion FULLY from BOTH
the full-clip and the single-frame latents (train-fit): intercept, r, vx, vy, vx^2, vy^2, |v|,
sin(ang), cos(ang). Then recompute the full-minus-single delta over 10 seeds.
Preregistered rule (fixed in code before any number is read; a tie is a null): HARDEN iff the strong-
residualized delta CI lo > 0 with no per-seed sign flip; DEMOTE iff the CI includes 0 or the sign
flips; HOLD iff CI lo > 0 but the effect shrinks more than 50 percent.

## Result (10 seeds)

  - Baseline full-minus-single delta (no control) is large and stable: motion_dir4 mean +0.200 CI
    [0.1645, 0.2355]; speed2 mean +0.245 CI [0.2037, 0.2863]. This is the at3 edge as reported.
  - Linear (r, vx, vy) partial-out is NOT enough: it kills motion_dir4 (mean -0.012) but leaves
    speed2 almost untouched (mean +0.227), exactly the magnitude/quadrant blind spot predicted. A
    linear-only control would have falsely "hardened" speed2.
  - STRONG motion residualization collapses BOTH factors. motion_dir4: strong delta mean -0.000, CI
    [-0.0361, +0.0361], sign flips (n_pos 4 / n_neg 6, consistent_sign 0), shrink 1.00. speed2:
    strong delta mean +0.008, CI [-0.031, +0.0477], sign flips (n_pos 5 / n_neg 5, consistent_sign
    0), shrink 0.966. Independent re-verify (axis_falsification/verify_nulls.py) reproduces both:
    motion_dir4 CI [-0.0361, +0.0361], speed2 CI [-0.031, +0.0477], both flip sign.
Both CIs straddle zero and both flip sign, so by the preregistered rule the verdict is DEMOTE. Once
the injected velocity is fully removed, the full-clip latents decode motion no better than the
single-frame latents.

## Why it is an asset

This demotes at3 from a temporal-integration claim to what it actually is: the full-clip encoder is
reading the velocity that was injected into the render, and the "temporal edge" is that same velocity
still present in the label. The card is doubly honest. First, it shows the weaker LINEAR control that
a less careful audit would have used, and shows it would have falsely kept speed2 (linear leaves
+0.227), which is exactly why the strong control is the right one. Second, it keeps the large
baseline delta on the record rather than hiding it. A registry that reported only the raw
full-minus-single edge would have banked a spurious temporal-currency positive; demoting it raises
the falsification-engine axis. The strong design is nonlinear-capable (magnitude and direction), so
the collapse cannot be explained by a control that was too weak to reach the factor.

```yaml
exp_id:            EX-AT3-TEMPORAL-CURRENCY
title:             full-clip vs single-frame motion edge collapses when injected velocity is fully removed
hypothesis:        full-clip V-JEPA latents integrate motion over time beyond what a single frame carries
null_hypothesis:   after projecting out the ground-truth motion fully (r, vx, vy, vx^2, vy^2, |v|, sin(ang), cos(ang)), the full-minus-single decodability delta is within noise (velocity-carried, not temporal integration)
baseline:          same linear reader head; baseline delta reported (motion_dir4 +0.200, speed2 +0.245); weaker linear (r, vx, vy) control shown to falsely retain speed2 (+0.227)
ablation:          strong nonlinear-capable motion design on BOTH full and single latents, train-fit; 10 seeds; sign-flip and shrink-fraction reported
metric:            probe_acc
probe_dependency:
  factor:          motion
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/motion.json
  decodable:       no
  acc_above_chance: 0.0
encoder_scale:     L
seeds:
  n:               10
  sem:             0.0184
  sign_stability:  unstable (motion_dir4 consistent_sign 0; speed2 consistent_sign 0 after strong residualization)
provenance_tag:    structured-synthetic
result:            baseline delta motion_dir4 +0.200 / speed2 +0.245; strong-residualized motion_dir4 mean -0.000 CI [-0.0361, +0.0361] flips shrink 1.00; speed2 mean +0.008 CI [-0.031, +0.0477] flips shrink 0.966; verdict DEMOTE
taxonomy_category: 3
verdict:           SUBSTRATE-BOUND
badges:            [substrate-blindspot, seed-instability]
raw_run_id:        runs/mot/at3_time_axis_seeds10.json (baseline edge); reaudit T1 (strong-residualization DEMOTE); independent re-verify axis_falsification/verify_nulls.py + verify_nulls.json
repro_level:       R2
```
