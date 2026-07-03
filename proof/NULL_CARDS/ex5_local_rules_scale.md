# Null card: local learning rules do not beat backprop once the optimizer is matched (REFUTED)

Registry entry ex5_local_rules_scale (extends I4 to persistent-weight retention). Form per BLACKHOLE.md:
no em or en dashes, no agency/understanding/intelligence language. The encoder is frozen and never
trained; this is a small MLP continual-learning stream, and only the learning rule and optimizer vary.

## Claim under test

The local-rules bet: a biologically local credit-assignment rule (feedback alignment or predictive coding)
gives a better continual-learning tradeoff than backprop on a 40-task domain-incremental stream, showing
up as higher final accuracy and less forgetting (better BWT). Feedback alignment did lead the default
backprop arm (acc 0.498 vs 0.366, BWT -0.546 vs -0.696), which looked like a local-rule retention win.

## Control (the tuned baseline the rule must beat)

The default backprop arm used Adam. Adam carries per-parameter second-moment state that inflates the
effective step and drives catastrophic forgetting on this stream, so it is a strawman ceiling. The added
fourth arm backprop_sgd is plain torch.optim.SGD (momentum=0), whose applied update is exactly lr*grad and
thus linear in lr. Per seed, on task 0 only, the mean per-step applied-update L2 of feedback_alignment and
predictive_coding is measured, their mean taken as the target, and the SGD lr calibrated so its effective
step magnitude matches the local rules (calibrated_lr mean 0.0731 vs nominal 0.05). This isolates credit
assignment (exact backprop gradient vs local surrogate) from optimizer state and scale. Preregistered null
H0: no local rule comes within the accuracy margin of backprop AND none offers a retention advantage that
justifies the gap. Falsifier: a local rule ties backprop accuracy or shows a retention advantage at matched
budget. A tie is a null.

## Result (5 seeds)

  - Effective-step-matched backprop_sgd: acc 0.509 +/- 0.068, BWT -0.539 +/- 0.077.
  - Best local rule (feedback_alignment): acc 0.498 +/- 0.049, BWT -0.546 +/- 0.041.
  - predictive_coding: acc 0.403 +/- 0.031, BWT -0.636 +/- 0.041. Default backprop_adam: acc 0.366, BWT -0.696.
  - Matched-SGD backprop now ties or beats every local rule on BOTH accuracy and BWT. It recovers +0.157
    BWT over Adam (adam_vs_sgd_bwt_delta 0.15714), closing the whole apparent local-rule advantage: acc gap
    closed fraction 1.08, BWT gap closed fraction 1.05 (both >= 1.0, i.e. fully closed).

The apparent local-rule win was an Adam optimizer artifact, not a property of local credit assignment. Once
backprop uses a plain-SGD update at the local rules' own effective step size, it forgets less and matches or
leads them. Non-degeneracy check: chance is 0.25 (4 classes per task) and every arm sits above it (0.366 to
0.509), so the stream is learnable and BWT is a real retention signal, not a floor artifact.

## Why it is an asset

This falsifies the local-rules retention claim on its own testbed by adding the control it was missing: an
optimizer-matched backprop arm. The lift came from Adam's state, so the honest reading is that the earlier
result measured optimizer choice, not credit-assignment biology. Showing this converts a promising local-rule
lead into a bounded negative and sharpens the live claim: a local rule must beat backprop at a MATCHED
optimizer, which none does here.

```yaml
exp_id:            ex5_local_rules_scale
title:             local learning rules do not beat effective-step-matched backprop on accuracy or BWT
hypothesis:        a local rule (feedback alignment / predictive coding) gives a better continual-learning accuracy-and-retention tradeoff than backprop at matched budget
null_hypothesis:   "H0: no local rule comes within the accuracy margin of backprop AND none offers a continual-learning or memory advantage that justifies the gap"
baseline:          backprop_sgd, plain torch.optim.SGD momentum=0, lr calibrated per seed to the local rules' effective per-step update L2 (calibrated_lr mean 0.0731 vs nominal 0.05); default backprop_adam kept as the strawman ceiling
ablation:          matched-effective-step SGD isolates credit assignment from optimizer state; Adam-vs-SGD BWT delta +0.157 attributes the apparent local-rule lead to Adam
metric:            bwt
probe_dependency:
  factor:          identity
  encoder:         n/a-mlp-continual-stream
  atlas_row:       registry difficulty-calibration; chance 0.25 (4 classes/task), all arms 0.366 to 0.509 above chance
  decodable:       yes
  acc_above_chance: 0.259
encoder_scale:     n/a
seeds:
  n:               5
  sem:             0.0305 (backprop_sgd BWT std 0.0771 / sqrt(5))
  sign_stability:  stable at S>=3 (matched-SGD ties or beats every local rule on both acc and BWT across 5 seeds)
provenance_tag:    structured-synthetic
result:            "backprop_sgd acc 0.509+/-0.068 BWT -0.539+/-0.077 ties/beats feedback_alignment acc 0.498+/-0.049 BWT -0.546+/-0.041; matched-SGD recovers +0.157 BWT over Adam; acc gap closed 1.08, BWT gap closed 1.05"
taxonomy_category: 5
verdict:           CAPACITY-ARTIFACT
badges:            [capacity-artifact, tuned-baseline-tie]
raw_run_id:        runs/pre_studio/close_ex5_local_rules.json (resolution refuted; 5 seeds [0-4])
repro_level:       R1
```

## What this closes

The local-rules retention lead on ex5 was an Adam artifact. At a matched effective step size, plain-SGD
backprop closes the whole gap (+0.157 BWT, acc and BWT gap-closed fractions >= 1.0) and ties or beats every
local rule, so the null is not rejected. Refuted.
