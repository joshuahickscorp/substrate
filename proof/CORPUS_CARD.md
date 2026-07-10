# Corpus card (the cached-latent corpus, made citable)

Makes the cached-latent corpus citable and prevents provenance uncertainty from
weakening the release. The license ledger is the release blocker until clean. Form per
BLACKHOLE.md: no em dashes; engineering vocabulary only. The encoder is frozen, so these
latents never go stale.

This is a STUB carrying the citable field list. A corpus release fills the real values only after
source authority and an untouched evaluation split exist. Historical pilot caches remain evidence
of mechanics, not members of the live corpus. Current intake targets official dense ViT-B, a pooled
control, and independently verified owned artifacts over exact shared referents.

```yaml
sources:            # per source: slug, license state, subset size
  - slug:        epic_kitchens_subset
    license:     CC BY-NC 4.0 (available, no signed terms)
    subset_size: TBD (stage small EPIC shard on laptop; 20k is a Studio job)
  - slug:        synthetic_controls
    license:     generated locally (zero license risk)
    subset_size: 9 families x {32,64} fixtures staged (data/controls/)
  - slug:        ssv2
    license:     manual (Qualcomm/20BN terms; HUMAN task pending, see README)
    subset_size: TBD (Studio, after access)
  - slug:        ego4d_subset
    license:     manual (signed Ego4D license + AWS creds; HUMAN task pending)
    subset_size: TBD (Studio, after access)
  - slug:        kinetics700_subset
    license:     metadata-only by default (ID/label CSVs open; video via licensed mirror)
    subset_size: CSVs only on laptop
encoders:           # exact registered ids, roles, dimensions, and authority receipts
  - id: vjepa21_vitb
    role: official_dense
    embed_dim: 768
  - id: vjepa2_vitl_fpc64_256
    role: pooled_control
    embed_dim: 1024
  - id: owned_artifact_id
    role: project_owned
    embed_dim: TBD
encoder_hashes:     TBD (weight hash per encoder, so a reader knows the exact frozen substrate)
preprocessing:
  frame_count:      64 (canonical) ; smaller frame counts tagged as throughput-lane caches
  resolution:       per registered encoder and immutable input manifest
  pooling:          pooled = latents + duplicated keys
latent_schema:
  shape:            [n_clips, embed_dim]
  dtype:            float32
  per_clip_bytes:   computed from exact token count, embed_dim, dtype, keys, and labels
  backend_tag:      exact backend or owned-artifact identity
seeds:              TBD (the seed set used across the campaign)
storage_size:       measured and projected per registered substrate before allocation
known_defects:      corrupt-file isolation, empty-class handling, short-clip handling (validator-enforced)
license_ledger:     see runs/studio_pipeline/latest/license_ledger.md once a plan is run
repro_level:        R0  (target R5 with a corpus tag/DOI if licensing allows)
```
