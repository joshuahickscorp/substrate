# Null card: PR9 long-stream plasticity restoration

PR9 tests whether continual backprop restores plasticity on a real DR1 latent stream. The run is
not a positive unless the plain baseline first certifies loss of plasticity under a CBP-blind tuned
learning rate, every CBP arm actually reinitializes units, LR integrals match, and the late-stream
adaptation gain clears seed spread without a retention tax. A tie is a null.

## Claim under test

On the DR1 real bound-video cache, utility-based selective reinitialization maintains or restores
late-stream plasticity in a frozen-latent shell where a well-tuned plain SGD shell loses plasticity.
The mechanism claim is about the plastic head on the same real latent stream, not about training the
frozen encoder.

## Control

The control is the identical plastic shell trained with plain SGD at the well-tuned learning rate,
selected by a CBP-blind tuning objective. CBP and plain arms must have matched LR integrals and the
same stream order, seeds, architecture, and task windows. A zero-reinitialization CBP arm is a config
error, not a null.

```yaml
exp_id: pr9_continual_backprop
title: continual backprop restores plasticity only if the tuned plain baseline certifies loss first
hypothesis: utility-based selective reinitialization restores late-stream plasticity on the DR1 real latent stream without paying a retention tax
null_hypothesis: given a fired plasticity-loss certificate under a well-tuned baseline, CBP does not restore plasticity beyond seed spread, or restores it only at a retention cost
baseline: identical plastic shell with plain SGD at the CBP-blind well-tuned LR on the same DR1 stream
ablation: CBP replacement-rate sweep with fractional reinit budget vs replacement_rate 0, matched LR integral, same seeds and stream order
metric: adaptation_steps_to_threshold
probe_dependency:
  factor: identity
  encoder: vjepa2_vitl_fpc64_256
  atlas_row: proof/atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable: "yes"
  acc_above_chance: 0.8414
encoder_scale: L
seeds:
  n: 10
  sem: null
  sign_stability: preregistered seed-CI lower bound and no sign flip required for a CBP win
provenance_tag: natural-video
result: pending Studio PR9 run; no positive until runs/mot/pr9_continual_backprop.json, runs/mot/pr9_continual_backprop.json.state.json, and runs/mot/pr9_verdict_ledger.json are present and pass on data/cache/vjepa2_vitl_comp_video
taxonomy_category: 3
verdict: SUBSTRATE-BOUND
badges: [preregistered, tuned-baseline-tie]
raw_run_id: pending Studio PR9 receipts under runs/mot plus proof/ARTIFACT_INDEX/pr9.json
repro_level: R0
```

## What this protects

This card prevents a short smoke run, a mistuned baseline, a no-op CBP arm, or an unmatched-compute
comparison from becoming moldability evidence. If the certificate does not fire, the correct verdict is
no plasticity loss to restore. If CBP fires but ties or pays a retention tax, the null is supported. A
candidate positive still needs the PR9 verdict ledger and the normal pre-ledger gates before publication.
