# Canonical null card: f13_form_energy_budget

Locked from `registry/experiments.yaml` for the durable F-series campaign. This card
defines the null and controls; it is not a result receipt. Historical runs that predate
the audit do not acquire retrospective preregistration status. A positive verdict still
requires an independent verifier and verdict-gate receipt.

## Claim Under Test

form_energy_budget

## Machine-Readable Card

```yaml
exp_id: f13_form_energy_budget
title: form_energy_budget
hypothesis: sweep form width, token count, replay bytes, and shell size while charging
  form production, alignment, fusion, projection, training, inference, parameters,
  and retained storage; report the frontier rather than peak accuracy
null_hypothesis: every form interface lies on the same full-system accuracy-versus-cost
  frontier as raw or matched random features, so form structure buys no capability
  per retained byte, parameter, FLOP, or analytically estimated joule
baseline: raw-features, matched-width-random-features, matched-shell, matched-replay-bytes,
  full-system-cost-accounting
ablation: the form family beats the strongest raw or random frontier across seeds
  and at least one operating point Pareto-dominates a control on accuracy, retained
  bytes, parameters, FLOPs, and analytically estimated energy
metric: accuracy_per_byte
probe_dependency:
  factor: density
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
raw_run_id: proof/FORM_SUBSTRATE/RECEIPTS/f13_form_energy_budget.json
repro_level: R0
```
