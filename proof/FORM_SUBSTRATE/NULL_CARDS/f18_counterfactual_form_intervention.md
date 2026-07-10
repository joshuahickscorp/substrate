# Canonical null card: f18_counterfactual_form_intervention

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

counterfactual_form_intervention

## Machine-Readable Card

```yaml
exp_id: f18_counterfactual_form_intervention
title: counterfactual_form_intervention
hypothesis: programmatic randomized interventions rendered as paired Form-A before
  and Form-B after states; a vector-valued state transport is compared to an identical
  correlational map on unseen intervention values with rows, updates, parameters,
  and FLOPs matched
null_hypothesis: the intervention predictor leaks (predicts only seen intervention
  values) or ties the correlational predictor, so the matrix binds appearances rather
  than intervention structure
baseline: correlational-predictor, random-intervention-direction, shuffled-counterfactual-pairs,
  matched-compute
ablation: decoded and geometric Form-B predictions on unseen intervention values beat
  correlational, random-direction, and shuffled-counterfactual controls across five
  seeds at exactly matched training compute
metric: counterfactual_match_acc
probe_dependency:
  factor: intervention_effect
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
taxonomy_category: 10
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f18_counterfactual_form_intervention.json
repro_level: R0
```
