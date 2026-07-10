# Canonical null card: f5_cross_form_memory_binding

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

cross_form_memory_binding

## Machine-Readable Card

```yaml
exp_id: f5_cross_form_memory_binding
title: cross_form_memory_binding
hypothesis: store referents through one form, query through another, and compare content-addressed
  retrieval to per-form nearest neighbor and shuffled-referent controls
null_hypothesis: cross-form retrieval fails to beat raw or shuffled controls, or remains
  materially below same-form independent-view retrieval, so memory is form-local rather
  than referent-bound
baseline: same-form-independent-view, raw-cross-form, shuffled-referent-pairs, matched-memory-slots
ablation: cross-form retrieval beats raw and shuffled controls and approaches same-form
  independent-view retrieval at matched memory slots
metric: cross_form_recall_at_k
probe_dependency:
  factor: memory_binding
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f5_cross_form_memory_binding.json
repro_level: R0
```
