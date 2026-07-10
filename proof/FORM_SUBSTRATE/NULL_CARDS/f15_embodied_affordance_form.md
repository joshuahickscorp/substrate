# Canonical null card: f15_embodied_affordance_form

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

embodied_affordance_form

## Machine-Readable Card

```yaml
exp_id: f15_embodied_affordance_form
title: embodied_affordance_form
hypothesis: paired object-action-outcome episodes under a held-out appearance shift,
  comparing consequence-conditioned Forms with zero-consequence passive and shuffled-action
  controls
null_hypothesis: consequence-conditioned Form tokens fail to beat both passive observation
  and action-shuffled controls on held-out affordance decoding and action selection
baseline: zero-consequence-passive-form, shuffled-action-outcomes, identical-head,
  matched-updates
ablation: consequence-conditioned Forms improve both held-out affordance decoding
  and selected-action success beyond passive and shuffled-action controls across five
  seeds
metric: affordance_decode_acc
probe_dependency:
  factor: affordance
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
taxonomy_category: 7
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f15_embodied_affordance_form.json
repro_level: R0
```
