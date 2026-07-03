# Survival card: short-horizon latent planning beats flat and shuffle on true dynamics

Registry-completion entry (EX series, synthetic planning arm). Form per BLACKHOLE.md: no em or
en dashes, no agency/understanding/intelligence language. The encoder is frozen and never trained;
only the latent dynamics head g(z,a)->z' and the reader heads are fit, and search is MPC over that
learned model. This is a survival card: the pre-declared null is rejected.

## Claim under test

The planning bet: a learned latent dynamics model plus shooting/CEM MPC rollout reaches goals a flat
reactive head cannot, on a synthetic action-conditioned dynamical family. The honest-scoring worry
was that the planner had been graded IN-BELIEF (against its own learned model) while the flat
baseline was graded on the TRUE synthetic environment, so they were not on one yardstick. This
closure re-scores the planner AND the action-shuffle control on the TRUE dynamics: the selected
action sequence is executed through _step_true (noise 0.0) and the real terminal distance measured,
exactly as the flat baseline was already scored.

## Control (what the planner must beat, graded identically)

Two preregistered controls, all three arms now graded on TRUE terminal distance to goal, 120 trials
per seed, 3 seeds (0, 1, 2), latent dim 32, action dim 8, horizon 8, 400 shooting samples, 3 CEM
iters. Flat-reactive-head: a one-step reactive head, no rollout. Action-shuffle: the planner's own
selected action sequence permuted before execution, which isolates the value of the ORDERING the
planner found from the value of the action marginals. Gate: rollout-predictability (k-step R2 floor
0.5) must hold or the task is degenerate and no planning claim is admissible.

Preregistered null H0: the learned dynamics does not enable planning the flat shell cannot do, or
rollout error is too high to plan against. Reject only if the planner beats BOTH the flat head and
the shuffle control on true terminal distance on every seed, with the rollout-predictability gate
passing. A tie is a null.

## Result (3 seeds, true dynamics)

  - planner minus flat, true terminal distance: planner mean 5.667 (stdev 0.299) vs flat mean 6.982
    (stdev 0.168); goal distance reduction mean 1.315 (stdev 0.438), per seed [1.436, 0.728, 1.780].
    Planner beats flat on 3 / 3 seeds.
  - planner minus shuffle: shuffle mean 6.972 (stdev 0.224); shuffle-minus-planner gap mean 1.305
    (stdev 0.0755), per seed [1.349, 1.199, 1.367]. Planner beats shuffle on 3 / 3 seeds. The gap is
    tight across seeds, so the win is the action ORDERING, not the action marginals.
  - Gate: rollout predictable on 3 / 3 seeds. One-step R2 about 0.967 to 0.968; k-step R2 0.872 to
    0.886, all above the 0.5 floor. The task is non-degenerate.
  - Honesty check: in-belief scoring was optimistic (planner belief minus true mean 0.830, per seed
    [0.661, 1.192, 0.636]), so belief was NOT the whole story, but the win survives once the optimism
    is removed by grading on true dynamics.

## Why it is an asset

The null is rejected on the yardstick that could have killed it. The in-belief scores were optimistic
by about 0.83 latent units, exactly the failure mode the closure was built to catch, yet the planner
still beats both the flat head and the ordering-scrambled shuffle on the true environment on every
seed. The shuffle control is the decisive one: it strips out the action marginals, so the remaining
1.305-unit gap is attributable to the temporal ORDERING the MPC search found, not to which actions it
used. The provenance is honest about its ceiling: the dynamics are structured-synthetic, so this is a
cpu-now precursor. Studio should re-test on real action-conditioned rollouts before the claim is
promoted past provisional.

```yaml
exp_id:            ex2_latent_planning
title:             short-horizon latent MPC reaches goals a flat reactive head and an action-shuffle control do not, on true synthetic dynamics
hypothesis:        a learned latent dynamics model g(z,a)->z' plus shooting/CEM MPC rollout reduces true goal-latent distance beyond a flat reactive head at acceptable rollout error
null_hypothesis:   H0: the learned dynamics does not enable planning the flat shell cannot do, or rollout error is too high to plan against; reject only if the planner beats BOTH the flat head and the action-shuffle control on true terminal distance on every seed with the rollout-predictability gate passing
baseline:          flat-reactive-head (one-step reactive head, no rollout, graded on true terminal distance identically to the planner)
ablation:          action-shuffle control (planner's selected action sequence permuted before execution) isolates temporal ordering from action marginals; both arms re-scored through _step_true (noise 0.0)
metric:            adaptation_steps_to_threshold
probe_dependency:
  factor:          controllability
  encoder:         structured-synthetic-dynamics (action-conditioned family, dim 32, action_dim 8)
  atlas_row:       in-run rollout-predictability gate (k-step R2), no atlas cache row for the synthetic arm
  decodable:       yes
  acc_above_chance: 0.372
encoder_scale:     synthetic
seeds:
  n:               3
  sem:             0.253
  sign_stability:  stable at S>=3 (planner beats flat 3/3 and shuffle 3/3, no per-seed sign flip)
provenance_tag:    structured-synthetic
result:            planner true terminal dist mean 5.667 vs flat 6.982 (reduction 1.315, stdev 0.438, 3/3 seeds); planner beats shuffle 3/3 (shuffle-minus-planner gap mean 1.305, stdev 0.0755); rollout predictable 3/3 (k-step R2 0.872 to 0.886); in-belief optimism mean 0.830
taxonomy_category: null rejected
verdict:           PUBLISH-POSITIVE
badges:            []
raw_run_id:        runs/pre_studio/close_ex2_planning.json (id ex2_latent_planning, survives=true, 3 seeds, 120 trials/seed)
repro_level:       R1
```

## What this closes

This closes ex2_latent_planning on the cpu-now synthetic arm: short-horizon latent MPC reaches goals
a flat reactive head does not, and the win is the temporal action ordering (it survives the shuffle
control) rather than a scoring artifact (it survives re-grading on true dynamics after removing the
0.830-unit in-belief optimism). It is provisional and structured-synthetic; the live
interactive-environment arm stays deferred (Tier R), and Studio should re-test on real
action-conditioned rollouts before promotion past R1.
