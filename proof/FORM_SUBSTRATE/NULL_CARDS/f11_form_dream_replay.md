# Canonical null card: f11_form_dream_replay

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

form_dream_replay

## Machine-Readable Card

```yaml
exp_id: f11_form_dream_replay
title: form_dream_replay
hypothesis: fit a class-conditional diagonal generator over canonical Form tokens,
  replay at matched sample count, and compare with byte-capped stored tokens, raw
  exemplars, no replay, and a random generator
null_hypothesis: generated Form replay falls outside the preregistered retention band
  of stored replay at the same memory ceiling or fails the held-out Form-manifold
  validity floor
baseline: stored-form-replay, raw-exemplar-replay, random-generator, no-replay, matched-replay-samples,
  actual-byte-accounting
ablation: generated Form replay stays within the stored-replay retention band, clears
  the held-out manifold-validity floor, and improves retention per actual retained
  byte across five seeds
metric: retention_per_byte
probe_dependency:
  factor: memory
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
taxonomy_category: 4
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f11_form_dream_replay.json
repro_level: R0
```
