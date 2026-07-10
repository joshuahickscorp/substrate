# Canonical null card: f10_intrinsic_form_curriculum

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

intrinsic_form_curriculum

## Machine-Readable Card

```yaml
exp_id: f10_intrinsic_form_curriculum
title: intrinsic_form_curriculum
hypothesis: candidate lessons span observation forms; scheduler chooses form and item
  by learning progress with noisy-TV and nuisance controls
null_hypothesis: learning-progress selection fails to improve untouched held-out-form
  transfer over every control or spends as much time on noisy forms as uniform, so
  the curriculum is not form-aware
baseline: uniform-form-sampling, prediction-error-sampling, novelty-sampling, noisy-tv-form
ablation: learning-progress improves untouched held-out-form transfer and rejects
  refreshed noisy forms better than every control
metric: coverage_per_update
probe_dependency:
  factor: active_curriculum
  encoder: structured-form-fixture
  atlas_row: 'not-applicable: factor labels are explicit fixture state'
  decodable: 'yes'
  acc_above_chance: 1.0
encoder_scale: not-applicable
seeds:
  'n': 5
  sem: not-estimated-at-contract-lock
  sign_stability: not-estimated-at-contract-lock
provenance_tag: provisional
result: contract locked for the next canonical run; this card is not evidence of temporal
  preregistration for runs that predate the audit
taxonomy_category: 8
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f10_intrinsic_form_curriculum.json
repro_level: R0
```
