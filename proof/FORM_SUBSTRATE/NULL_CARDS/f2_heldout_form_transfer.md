# Canonical null card: f2_heldout_form_transfer

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

heldout_form_transfer

## Machine-Readable Card

```yaml
exp_id: f2_heldout_form_transfer
title: heldout_form_transfer
hypothesis: substitute exactly one aligned training form per referent and optimizer
  step, matching labeled rows, initialization, head topology, and updates to single-form
  and Gaussian-augmentation controls
null_hypothesis: matched-exposure multi-form training fails to beat the strongest
  single-reference or matched-noise control by the preregistered margin, the held-out
  form remains near chance, or referent-shuffled alignment ties the treatment
baseline: single-reference-form, single-form-matched-noise, shuffled-heldout-alignment,
  chance, matched-rows-updates-head, disjoint-anchor-label-test
ablation: multi-form training beats both exposure-matched single-form controls and
  shuffled held-out alignment across five seeds
metric: heldout_form_acc
probe_dependency:
  factor: cross_form_transfer
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f2_heldout_form_transfer.json
repro_level: R0
```
