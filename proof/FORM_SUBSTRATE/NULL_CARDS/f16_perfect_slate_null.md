# Canonical null card: f16_perfect_slate_null

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

perfect_slate_null

## Machine-Readable Card

```yaml
exp_id: f16_perfect_slate_null
title: perfect_slate_null
hypothesis: initialize one blank encoder once, self-supervise one copy while freezing
  the identical-weight copy, then compare both with frozen inherited features and
  a larger inherited-feature shell under one explicit end-to-end FLOP estimator
null_hypothesis: the blank substrate fails to beat inherited frozen features, the
  larger shell, or its identical frozen-random initialization on licensed held-out
  transfer at a matched estimated end-to-end FLOP budget
baseline: frozen-inherited-substrate, larger-shell-on-frozen, shared-initialization-frozen-random-same-arch,
  matched-estimated-end-to-end-flops
ablation: the self-supervised blank encoder beats inherited-frozen, larger-shell,
  and shared-initialization frozen-random controls beyond the preregistered per-seed
  margin under the matched estimated FLOP convention
metric: blank_vs_inherited_delta
probe_dependency:
  factor: blank_slate
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
taxonomy_category: 10
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/PREFLIGHT/f16_perfect_slate_null.json
repro_level: R0
```
