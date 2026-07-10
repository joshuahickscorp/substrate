# Canonical null card: f8_plastic_substrate_rewrite

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

plastic_substrate_rewrite

## Machine-Readable Card

```yaml
exp_id: f8_plastic_substrate_rewrite
title: plastic_substrate_rewrite
hypothesis: update inherited encoder weights under a paired-view self-supervised objective,
  measure cosine representation rewrite, and compare held-out transfer with frozen
  inherited, larger-shell, and SSL-trained random-init controls under one explicit
  end-to-end FLOP estimator
null_hypothesis: a plastic substrate fails to beat the frozen inherited, larger frozen-shell,
  or SSL-trained random-init controls on licensed held-out factors at a matched estimated
  end-to-end FLOP budget
baseline: frozen-inherited-linear, larger-frozen-shell, ssl-trained-random-init-same-arch,
  matched-estimated-end-to-end-flops
ablation: plastic substrate beats every declared control on held-out factors beyond
  the preregistered per-seed margin under the matched estimated FLOP convention, while
  the registered rewrite metric reports actual representation shift
metric: heldout_factor_acc
probe_dependency:
  factor: representation_rewrite
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
taxonomy_category: 9
verdict: DOWNGRADE-TIE
badges:
- contract-locked
- structured-fixture
- reconstructed-after-audit
raw_run_id: proof/FORM_SUBSTRATE/PREFLIGHT/f8_plastic_substrate_rewrite.json
repro_level: R0
```
