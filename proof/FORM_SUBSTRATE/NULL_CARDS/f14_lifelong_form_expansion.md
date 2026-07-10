# Canonical null card: f14_lifelong_form_expansion

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

lifelong_form_expansion

## Machine-Readable Card

```yaml
exp_id: f14_lifelong_form_expansion
title: lifelong_form_expansion
hypothesis: train on two forms, write immutable reference-form values, keys, and referent
  ids into ReplayBuffer/KVIndex, then align and replay a third form without rewriting
  any old memory tensor
null_hypothesis: new-form insertion changes any old-memory key, value, or referent
  id, changes old-memory retrieval, forgets old forms beyond the retention band, or
  fails to beat the strongest matched existing-head and raw or shuffled retrieval
  controls
baseline: frozen-zero-shot, new-only-no-replay-matched-compute, retrain-from-scratch-matched-cumulative-compute,
  no-alignment, shuffled-referents, raw-memory-query, shuffled-memory-query, immutable-old-memory-snapshot
ablation: the new form beats matched existing-head and retrieval floors while old-form
  forgetting stays bounded and every old memory tensor, id, count, hash, and retrieval
  result remains bit-exact
metric: old_form_bwt
probe_dependency:
  factor: continual_expansion
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
taxonomy_category: 5
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f14_lifelong_form_expansion.json
repro_level: R0
```
