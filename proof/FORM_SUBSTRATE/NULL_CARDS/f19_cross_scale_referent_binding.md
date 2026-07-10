# Canonical null card: f19_cross_scale_referent_binding

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

cross_scale_referent_binding

## Machine-Readable Card

```yaml
exp_id: f19_cross_scale_referent_binding
title: cross_scale_referent_binding
hypothesis: nested referent ids (object within scene within episode within task) enforced
  at intake; store at one scale, query at another, compare hierarchical memory to
  flat and single-scale stores at matched bytes
null_hypothesis: hierarchical referent memory ties the strongest flat, single-scale,
  or random-hierarchy control at matched bytes, so scale structure buys no retrieval
  and memory stays clip-shaped
baseline: single-scale-memory, flat-clip-memory, random-hierarchy, matched-memory-bytes
ablation: bidirectional cross-scale retrieval beats every flat, single-scale, and
  random-hierarchy control across five seeds at exactly matched allocated bytes
metric: cross_scale_recall_at_k
probe_dependency:
  factor: scale_binding
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f19_cross_scale_referent_binding.json
repro_level: R0
```
