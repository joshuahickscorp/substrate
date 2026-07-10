# Canonical null card: f7_developmental_form_growth

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

developmental_form_growth

## Machine-Readable Card

```yaml
exp_id: f7_developmental_form_growth
title: developmental_form_growth
hypothesis: grow hidden Form capacity when measured learning progress stalls, then
  compare with fixed-final capacity and an event-count-matched random growth schedule
null_hypothesis: learning-progress growth fails to beat the stronger fixed-final or
  random-growth control on the adaptation-retention frontier at matched final parameters
baseline: fixed-final-capacity, event-count-matched-random-growth, matched-final-params,
  matched-updates
ablation: learning-progress growth beats both fixed-final and random-timing controls
  on the adaptation-retention frontier across five seeds with exact final-parameter
  and update equality
metric: frontier_auc_gain
probe_dependency:
  factor: plasticity
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
taxonomy_category: 1
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f7_developmental_form_growth.json
repro_level: R0
```
