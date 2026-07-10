# Canonical null card: f1_form_alignment_gate

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

form_alignment_gate

## Machine-Readable Card

```yaml
exp_id: f1_form_alignment_gate
title: form_alignment_gate
hypothesis: fit a target-to-source affine map on unlabeled paired anchors disjoint
  from source-head training and final test rows, then freeze one source head across
  paired and unpaired coordinate controls
null_hypothesis: paired referent alignment fails to beat the strongest raw, moment-matched,
  or shuffled-anchor control by the preregistered margin, or remains near chance
baseline: raw-target-transfer, unpaired-moment-match, shuffled-referent-anchor-map,
  source-form-ceiling, target-supervised-oracle, disjoint-anchor-label-test
ablation: paired alignment beats every raw, moment-matched, and shuffled-anchor control
  across five seeds while staying below a separately trained target oracle
metric: aligned_transfer
probe_dependency:
  factor: cross_form_alignment
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f1_form_alignment_gate.json
repro_level: R0
```
