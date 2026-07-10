# Canonical null card: f20_substrate_crisis_test

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

substrate_crisis_test

## Machine-Readable Card

```yaml
exp_id: f20_substrate_crisis_test
title: substrate_crisis_test
hypothesis: build matrices containing known-insufficient arms (nuisance-carrier and
  predictor-wall exemplars) plus noisy-TV false-alarm streams; score crisis predictions
  against realized probe failure
null_hypothesis: the crisis detector fails to beat the strongest raw-error, fixed-confidence,
  or random baseline, or triggers on aleatoric noise, so prospective insufficiency
  is not established
baseline: raw-error-signal, fixed-confidence-threshold, random-trigger-matched-rate,
  noisy-tv-stream
ablation: a preregistered prospective crisis forecast beats every baseline, stays
  quiet on noise, and avoids a measured failed scale-up
metric: crisis_auroc
probe_dependency:
  factor: crisis_detection
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f20_substrate_crisis_test.json
repro_level: R0
```
