# Canonical null card: f17_missing_form_recovery

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

missing_form_recovery

## Machine-Readable Card

```yaml
exp_id: f17_missing_form_recovery
title: missing_form_recovery
hypothesis: train on a multi-form matrix, drop one arm at evaluation, compare recovery
  against impute-by-mean and best-remaining-form controls while scoring whether confidence
  predicts correctness under absence (OA2 calibration AUROC)
null_hypothesis: recovery fails to beat the strongest tuned single-form, impute-by-mean,
  or zero-filled-concat control, or confidence does not predict correctness under
  a missing form, so recovery or monitoring is uninformative
baseline: impute-by-mean, best-remaining-form, zero-filled-concat, matched-head
ablation: recovery beats every declared control while confidence still predicts correctness
  under absence above the chance floor
metric: recovery_acc
probe_dependency:
  factor: form_recovery
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
taxonomy_category: 2
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f17_missing_form_recovery.json
repro_level: R0
```
