# Canonical null card: f9_cross_form_compositional_binding

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

cross_form_compositional_binding

## Machine-Readable Card

```yaml
exp_id: f9_cross_form_compositional_binding
title: cross_form_compositional_binding
hypothesis: hold out cross-form factor pairs, train on diagonal complements, and test
  held-out combinations after referent alignment
null_hypothesis: held-out cross-form combinations collapse toward the strongest single-form,
  shuffled-label, or shuffled-referent floor while seen pairs stay high, so the system
  did not bind factors across forms
baseline: shuffle-label-floor, single-form-conjunction-baseline, shuffled-referent-pairs
ablation: factor-specific heads preserve held-out combinations above every single-form
  and shuffled-referent control
metric: heldout_combo_acc
probe_dependency:
  factor: compositionality
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
taxonomy_category: 3
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f9_cross_form_compositional_binding.json
repro_level: R0
```
