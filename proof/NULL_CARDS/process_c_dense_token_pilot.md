# Null card: Process C dense-token pilot

Process C is not a default next step. It is a small, licensed fallback: a 1 to 10M object-centric
dense-token shell over frozen dense tokens, trained only after PR9 or DR1 leaves a receipt-backed
frozen-substrate wall. A tie is a null, and a missing license is a stop condition.

## Claim under test

On a real DR1 dense-token cache, a small object-centric slot shell can recover held-out bound-attribute
generalization that the frozen dense-token mean baseline cannot recover at matched capacity and compute.
The claim is about the licensed shell over frozen dense tokens, not about training a foundation video
model or replacing the whole substrate.

## Control

The control is the matched dense-without-slots baseline selected by `dense_hidden_for_target_params` on
the same dense cache, with the same labels, stream order, seeds, train/eval split, parameter budget, and
compute cap. A matched random-init dense cache and shuffle/permutation floors remain required for any
downstream substrate claim. Launch is blocked unless `runs/mot/process_c_license_gate.json` says
`launch_allowed: true`.

```yaml
exp_id: process_c_dense_token_pilot
title: Process C dense-token pilot only scores after PR9 or DR1 licenses the frozen-substrate wall
hypothesis: a 1 to 10M object-centric shell over frozen dense tokens improves held-out bound-attribute generalization beyond a matched dense mean baseline
null_hypothesis: the slot shell ties or loses to the matched dense-without-slots baseline, exceeds the 1 to 10M cap, lacks a binding-specificity gain, or is launched without a PR9 or DR1 license receipt
baseline: dense-without-slots mean-pooled token baseline with matched parameters, compute, seeds, split, and stream order
ablation: remove slots, exceed the budget cap, use an unlicensed PR9/DR1 state, or drop the matched random-init dense control; any such ablation downgrades to null
metric: recombination_generalization
probe_dependency:
  factor: relation
  encoder: all-three
  atlas_row: runs/mot/atlas_multi_encoder_grid.json
  decodable: "marginal"
  acc_above_chance: null
encoder_scale: all-three
seeds:
  n: 10
  sem: null
  sign_stability: preregistered seed-CI lower bound and no sign flip required for a Process C win
provenance_tag: natural-video
result: pending licensed Studio Process C run; no launch until runs/mot/process_c_license_gate.json has launch_allowed true and no positive until the pilot, dense gate, null card, artifact index, and verdict gate pass
taxonomy_category: 5
verdict: SUBSTRATE-BOUND
badges: [preregistered, substrate-blindspot]
raw_run_id: pending licensed Process C receipts under runs/mot plus proof/ARTIFACT_INDEX/pr9.json or a later Process C artifact index
repro_level: R0
```

## What this protects

This card prevents the presence of `src/mop/process_c/dense_tokens.py` from being treated as permission
to train. If PR9 finds a cheaper CBP positive, Process C is not licensed by that path. If DR1 lacks an
integrity-clean A6 wall, Process C is not licensed by that path. If the gate does license the pilot, the
result still has to beat the matched dense baseline, stay within the 1 to 10M cap, pass binding-specificity
checks, and survive the normal verdict-gate and artifact-bundle path.
