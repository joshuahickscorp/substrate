# Corpus card (the cached-latent corpus, made citable)

Makes the cached-latent corpus citable and prevents provenance uncertainty from
weakening the release. The license ledger is the release blocker until clean. Form per
BLACKHOLE.md: no em dashes; engineering vocabulary only. The encoder is frozen, so these
latents never go stale.

This is a STUB carrying the Section 10.10 field list. The Studio fills the real values
once the permanent multi-encoder pooled corpus is built. Until then the only real cached
latents on record are the 96-clip ViT-L probe cache (`data/cache/vjepa2_vitl_fpc64_256_real/`,
shape (96, 1024) float32, 6 classes, linear-probe acc 1.0 on record).

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
encoders:           # verified HF ids, embed dims
  - id: vjepa2_vitl_fpc64_256   # facebook/vjepa2-vitl-fpc64-256
    embed_dim: 1024
  - id: vjepa2_vith             # facebook/vjepa2-vith-fpc64-256
    embed_dim: 1280
  - id: vjepa2_vitg             # facebook/vjepa2-vitg-fpc64-384
    embed_dim: 1408
encoder_hashes:     TBD (weight hash per encoder, so a reader knows the exact frozen substrate)
preprocessing:
  frame_count:      64 (canonical) ; smaller frame counts tagged as throughput-lane caches
  resolution:       256 (L/H) / 384 (g)
  pooling:          pooled = latents + duplicated keys
latent_schema:
  shape:            [n_clips, embed_dim]
  dtype:            float32
  per_clip_pooled:  ~8 KB (L) / ~10 KB (H) / ~11 KB (g)
  backend_tag:      vjepa_hf
seeds:              TBD (the seed set used across the campaign)
storage_size:       pooled store across encoders is a few GB even at 100k clips
known_defects:      corrupt-file isolation, empty-class handling, short-clip handling (validator-enforced)
license_ledger:     see runs/studio_pipeline/latest/license_ledger.md once a plan is run
repro_level:        R0  (target R5 with a corpus tag/DOI if licensing allows)
```
