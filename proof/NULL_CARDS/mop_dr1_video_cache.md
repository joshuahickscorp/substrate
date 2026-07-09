# Null card: DR1 real bound-attribute video cache

DR1 is the prerequisite cache, not a result by itself. The cache only licenses downstream abstraction,
density, dense-token, and PR9 runs if the sidecar receipts prove real bound-attribute curation,
caption recoverability, real-encoder latents, PerspectiveMatrix alignment, A6 residual verification,
and independent adversarial verification. A tie or missing sidecar is a null.

## Claim under test

The Studio run creates a real non-additively bound video cache where pooled and dense frozen V-JEPA
latents can support off-ceiling compositional probes. The cache must beat the failure mode where the
video is real but the attribute binding is not recoverable, the encoder silently falls back to a random
backend, or the cross-modal alignment is just a factor or nuisance partition.

## Control

The cache has no positive interpretation until later probes compare real V-JEPA to matched random-init
and dense-without-slots controls. For the cache artifact itself, the control is stricter: failed caption
gate, non-contiguous shards, frozen-random backend, missing dense or pooled sidecars, missing
PerspectiveMatrix receipt, or A6 collapse. Any one of those blocks a positive downstream claim.

```yaml
exp_id: mop_dr1_video_cache
title: real bound-attribute video cache is only usable if caption, perspective, A6, and verification receipts pass
hypothesis: frozen V-JEPA pooled and dense latents over real bound-attribute video create a non-ceiling test bed for downstream compositional probes
null_hypothesis: even on real video the pooled and dense latents ceiling at 1.0 on held-out-combination decode, or the cache lacks durable sidecars proving bound factors, real backend, A6 survival, and adversarial verification
baseline: failed caption gate, matched random-init encoder floor, dense-without-slots floor, and A6 residualized-alignment collapse
ablation: remove one receipt at a time: caption recoverability, real-backend leg sidecar, PerspectiveMatrix alignment, A6 residual guard, or independent verifier; any removal blocks the positive read
metric: recombination_generalization
probe_dependency:
  factor: identity
  encoder: vjepa2_vitl_fpc64_256
  atlas_row: proof/atlas/vjepa2_vitl_fpc64_256/identity.json
  decodable: yes
  acc_above_chance: 0.8414
encoder_scale: L
seeds:
  n: 3
  sem: null
  sign_stability: preregistered cache gate, downstream probes must report stable at S>=3
provenance_tag: natural-video
result: preregistered before Studio DR1, no positive until data/cache/vjepa2_vitl_comp_video/merge_manifest.json, perspective_matrix_receipt.json, a6_residual_guard.json, and dr1_verification.json are all present and pass
taxonomy_category: 5
verdict: SUBSTRATE-BOUND
badges: [preregistered, substrate-blindspot]
raw_run_id: pending Studio DR1 sidecars under data/cache/vjepa2_vitl_comp_video plus proof/ARTIFACT_INDEX/dr1.json
repro_level: R0
```

## What this protects

This card prevents the DR1 cache from being scored as evidence merely because it exists. The cache is a
launchpad only when the verifier passes. If the verifier fails, downstream DR1-dependent claims inherit
the wall instead of quietly citing the cache.
