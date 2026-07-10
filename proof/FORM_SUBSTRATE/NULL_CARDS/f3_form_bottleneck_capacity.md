# Canonical null card: f3_form_bottleneck_capacity

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

form_bottleneck_capacity

## Machine-Readable Card

```yaml
exp_id: f3_form_bottleneck_capacity
title: form_bottleneck_capacity
hypothesis: project aligned forms through one nested orthonormal basis, zero-pad every
  arm into the same head dimension, and compare identical initialized heads at matched
  rows and updates
null_hypothesis: the nested wide bottleneck fails to beat the zero-padded small bottleneck
  by the preregistered margin, or wide performance remains near the chance or shuffled-label
  floor
baseline: nested-small-bottleneck, shuffled-label-floor, no-bottleneck, all-form-concatenation-upper-bound,
  identical-zero-padded-head, matched-data-updates
ablation: the nested wide prefix beats the small prefix across five seeds while remaining
  below no-bottleneck and concatenation upper bounds
metric: wide_form_acc
probe_dependency:
  factor: bottleneck
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
taxonomy_category: 4
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f3_form_bottleneck_capacity.json
repro_level: R0
```
