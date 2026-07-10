# Canonical null card: f6_sensorimotor_form_closure

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

sensorimotor_form_closure

## Machine-Readable Card

```yaml
exp_id: f6_sensorimotor_form_closure
title: sensorimotor_form_closure
hypothesis: deterministic gridworld observation-action Form stream with identical-architecture
  action-shuffle and zero-action controls, evaluated by autoregressive rollout prediction
  and executed goal reachability
null_hypothesis: true action-conditioned Form closure fails to beat both action-blind
  and action-shuffled controls on held-out rollout prediction and deterministic goal
  reachability
baseline: zero-action-form, shuffled-action-referents, identical-architecture, matched-updates
ablation: true action Form improves both autoregressive rollout R2 and executed goal
  success beyond the strongest zero-action or shuffled-action control across five
  seeds
metric: rollout_r2
probe_dependency:
  factor: controllability
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f6_sensorimotor_form_closure.json
repro_level: R0
```
