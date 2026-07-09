# Null card: dense multi-encoder atlas

The dense atlas asks whether structure survives the full registered multi-encoder grid after every
column and pair beats its own control. A positive is not available from a partial grid, a single real
dense cache without its matched random-init control, or an atlas JSON that has not passed the verdict
ledger. A tie is a null.

## Claim under test

On the Studio full registered atlas, frozen dense and pooled substrates carry non-random structure
across nuisance, alignment, within-encoder abstraction, and cross-substrate invariance axes. The claim
is about substrate evidence after matched random-init and permutation controls, not about training a
new foundation model.

## Control

Each atlas column must beat its own matched random-init control where applicable, each alignment pair
must beat the permutation floor, each cross-substrate pair must beat its matched random-to-random pair,
and the dense V-JEPA 2.1 column must include both real and matched random-init dense caches with matching
referents and sidecars. A missing dense control or partial registered grid withholds universal scope.

```yaml
exp_id: atlas_dense_multiencoder
title: dense multi-encoder atlas only scores after the full real/control registered grid passes
hypothesis: the full registered Studio atlas exposes non-random substrate structure across dense video, pooled video, image, text-rendered, and audio-rendered perspectives
null_hypothesis: no registered substrate beats its own random-init or permutation control across the atlas axes, or the apparent scope depends on a missing dense control or partial registered grid
baseline: each column's matched random-init control, permutation floor for alignment, and matched random-to-random transfer pair for cross-substrate invariance
ablation: remove the dense random-init cache, remove any registered column or arm, or allow a partial grid; any such ablation withholds universal scope
metric: dense_delta
probe_dependency:
  factor: identity
  encoder: all-three
  atlas_row: runs/mot/atlas_multi_encoder_grid.json
  decodable: "marginal"
  acc_above_chance: null
encoder_scale: all-three
seeds:
  n: 10
  sem: null
  sign_stability: preregistered atlas seeds and abstraction seeds 0-9; sign flips downgrade to null
provenance_tag: natural-video
result: pending Studio dense/atlas run; no positive until runs/mot/dense_atlas_cache_gate.json, runs/mot/atlas_multi_encoder_grid.json, runs/mot/atlas_verdict_ledger.json, and proof/ARTIFACT_INDEX/atlas.json pass
taxonomy_category: 5
verdict: SUBSTRATE-BOUND
badges: [preregistered, substrate-blindspot]
raw_run_id: pending Studio atlas receipts under runs/mot plus proof/ARTIFACT_INDEX/atlas.json
repro_level: R0
```

## What this protects

This card prevents a partial atlas, a missing matched dense control, a random-control artifact, or a
plain raw JSON from becoming density evidence. If the full atlas supports the null, the wall is preserved.
If it rejects the null, the result is only a candidate positive until the atlas verdict ledger, artifact
bundle, and normal verdict-gate path all pass.
