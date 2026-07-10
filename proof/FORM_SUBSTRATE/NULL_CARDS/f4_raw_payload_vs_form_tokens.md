# Canonical null card: f4_raw_payload_vs_form_tokens

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

raw_payload_vs_canonical_form_tokens

## Machine-Readable Card

```yaml
exp_id: f4_raw_payload_vs_form_tokens
title: raw_payload_vs_canonical_form_tokens
hypothesis: render ordered heterogeneous token payloads, align each token position
  on disjoint paired anchors, preserve [N,T,D] geometry through a shared-token-transform
  probe, and compare matched token-shaped controls
null_hypothesis: ordered canonical form tokens fail to beat the strongest raw, handcrafted,
  shuffled-referent, or token-order control by the preregistered margin
baseline: raw-resized-token-payload, per-token-handcrafted-statistics, shuffled-referent-token-alignment,
  token-order-permutation, matched-token-shape-head-data-updates
ablation: ordered canonical tokens beat all four token-shaped controls across five
  seeds while the Form audit verifies preserved token geometry
metric: cross_form_transfer_per_dim
probe_dependency:
  factor: form_tokenization
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f4_raw_payload_vs_form_tokens.json
repro_level: R0
```
