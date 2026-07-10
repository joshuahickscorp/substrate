# Canonical null card: f12_private_form_language_stability

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

private_form_language_stability

## Machine-Readable Card

```yaml
exp_id: f12_private_form_language_stability
title: private_form_language_stability
hypothesis: learn discrete form codes across multiple forms, then test cross-seed
  probe transfer, code agreement, and referent retrieval
null_hypothesis: cross-seed code agreement sits at or below the random-codebook floor
  and cross-seed probe transfer is at chance, so the form codes are private idiolects
  rather than a shared language
baseline: random-codebook, shuffled-referents, seed-sweep
ablation: code transfer and agreement beat random-code floors across seeds and forms
metric: cross_seed_code_transfer
probe_dependency:
  factor: code_stability
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f12_private_form_language_stability.json
repro_level: R0
```
