# Null card: degenerate-ensemble retention advantage is seed-unstable (last frozen-random gap)

Registry-completion entry (series B, matched-compute continual-learning lane). Form per BLACKHOLE.md:
no em or en dashes, no agency/understanding/intelligence language. The encoder is frozen and never
trained; only the small predictors and their agreement heads are fit on the domain-incremental stream.

## Claim under test

The degeneracy bet: building the shell as K structurally-distinct sub-predictors (widths 32, 24, 40)
trained to agree on outputs retains more of past tasks (higher backward transfer) than a matched-param
single predictor and than K identical copies, and does so because the real domain-incremental geometry
supplies complementary error structure. If true, the retention gain is substrate-specific: it should
shrink or vanish when the inputs are replaced by a fixed generic projection.

## Control (the two frozen-projection substrates the effect must survive)

Two preregistered controls applied to every task's x before training, both isolating substrate:
  - frozen_random_projection: an invertible fixed random map (e7-comparable). If the retention gain is
    a generic architectural fact it holds here; if it is substrate-specific it shrinks.
  - rank_reduced_projection: a lossy fixed projection (a stronger, information-removing control).
Metric is backward-transfer retention, degenerate minus the best of the two baselines (single, copies)
per seed, at matched compute. Preregistered null H0: degeneracy ties the matched-param single and ties
pure redundancy on retention; any apparent win is capacity removed by the matched-compute control. The
decision rule pre-declares a seed-spread margin the real gain must clear before the frozen-projection
controls can be read at all. A tie is a null. No threshold was tuned toward a win.

## Result (5 seeds)

  - Real degenerate-minus-best-baseline retention gain: mean +0.093. Per-seed gains are +0.390, 0.000,
    +0.075, +0.125, -0.125: the sign flips and the range spans 0.515. The seed-spread margin is 0.2575,
    and the real gain (+0.093) does NOT clear it (real_gain_clears_margin: false). On a fresh 5-seed set
    the same gain lands at -0.035, confirming the mean sits inside the noise band rather than above it.
  - Because the real effect never clears its own margin, the substrate question cannot be settled: the
    frozen_random arm returns mean gain +0.066 (ratio 0.710 of the unstable real gain, resolution
    inconclusive), and the rank_reduced arm returns mean gain +0.002 (ratio 0.022, resolution
    inconclusive). Both controls are moot because there is no robust effect for them to attribute.
  - Gates: real_gain_clears_margin false; frozen_random_resolution inconclusive; rank_reduced_resolution
    inconclusive. The null is not rejected. Verdict SEED-UNSTABLE.

Mechanism (why the null holds, not just that it holds): the per-seed retention is dominated by whether
the stream happens to be near-fully retained anyway. Seed 1 saturates (degenerate, single, and copies
all at or near 1.0, gain 0.0) and seed 3 nearly does (degenerate 1.0, copies 0.875); the two large
positive gains come only from the seeds where the single predictor collapses (seed 0 single bwt 0.070).
So the "advantage" is a small number of low-retention seeds where any second predictor helps, not a
stable property of the degenerate geometry. There is no headroom the controls could localize.

## Why it is an asset

This closes the last open frozen-random gap in series B: the one continual-learning claim whose
substrate-specificity was still unresolved is now bounded as underpowered at this scale, not left
dangling. The honest outcome is that the retention gain does not survive its own pre-declared seed-spread
margin, so both frozen-projection controls are correctly reported as inconclusive rather than spun as
confirming substrate-specificity. A registry that reported the +0.093 mean as a win while suppressing
the sign flip, the 0.515 per-seed range, and the -0.035 fresh-seed replication would be a whitewash;
showing it is the rigor working. The concrete follow-up is a seed and sample-size increase: the effect
must first clear its margin before the frozen_random vs rank_reduced question is even askable.

```yaml
exp_id:            b5_degeneracy_robustness
title:             degenerate K-predictor retention gain does not clear its seed-spread margin at 5 seeds
hypothesis:        K structurally-distinct sub-predictors retain more of past tasks than a matched-param single and than K identical copies, because the real domain-incremental geometry supplies complementary error structure
null_hypothesis:   "H0: degeneracy ties the matched-param single predictor and ties pure redundancy on retention (any win was capacity removed by the matched-compute control)"
baseline:          matched-param single predictor and K identical copies (best-of-two taken per seed), matched-FLOPs enforced not just matched-params
ablation:          two frozen-projection controls applied to every task's x, frozen_random_projection (invertible, e7-comparable) and rank_reduced_projection (lossy, stronger); both returned inconclusive because the real effect does not clear the margin
metric:            bwt
probe_dependency:
  factor:          identity
  encoder:         vjepa2_vitl_fpc64_256
  atlas_row:       atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable:       yes
  acc_above_chance: null
encoder_scale:     L
seeds:
  n:               5
  sem:             null
  sign_stability:  unstable (per-seed real gains +0.390, 0.000, +0.075, +0.125, -0.125; sign flips; -0.035 on a fresh 5-seed set)
provenance_tag:    structured-synthetic
result:            "real degenerate-minus-best-baseline retention gain mean +0.093 does not clear seed-spread margin 0.2575 (real_gain_clears_margin false); frozen_random gain +0.066 (ratio 0.710, inconclusive); rank_reduced gain +0.002 (ratio 0.022, inconclusive); fresh-seed replication -0.035"
taxonomy_category: 5
verdict:           SEED-UNSTABLE
badges:            [seed-instability]
raw_run_id:        runs/pre_studio/close_b5_degeneracy.json (md5 bb49165b169fa231f5234110bda5327f); 5 seeds [0,1,2,3,4]; elapsed 3.4s
repro_level:       R1
```

## What this closes

This bounds the degenerate-ensemble retention advantage as an underpowered null at 5 seeds: the real
gain (+0.093) sits inside its own seed-spread margin (0.2575), flips sign across seeds, and reads -0.035
on a fresh seed set, so the frozen_random and rank_reduced substrate controls are correctly moot. It is
the last open frozen-random gap in series B, now filled with a SEED-UNSTABLE verdict rather than an
overclaimed win.
