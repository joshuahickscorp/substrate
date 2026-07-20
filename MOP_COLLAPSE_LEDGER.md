# MOP Collapse Ledger

Compact generated view. Machine authority: `MOP_COLLAPSE_STATE.json`.

## Current

- Maintained Python: 7,401 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 5,200 LOC.
- Validation: 1,819 LOC.
- Verified reduction ledger: 400,449 LOC.
- Checklist: {"active": 1, "complete": 216, "pending": 1, "verified": 43}.
- Recovery: state `legacy_authorities` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-count-statistics-core | 330 | source_consumed_count_statistics_projection |
| mop-collapse-direct-count-producer | 182 | direct_single_consumer_count_production_vertical |
| mop-collapse-retired-onset-schema | 187 | retired_historical_onset_object_and_roundtrip_schema |
| mop-collapse-direct-count-intake | 97 | direct_real_count_intake_without_facades |
| mop-collapse-direct-count-artifact | 179 | direct_single_consumer_count_artifact_lifecycle |
| mop-collapse-direct-count-prereg | 68 | direct_count_preregistration_projection |
| mop-collapse-direct-count-gate | 68 | direct_fixed_count_gate_lifecycle |
| mop-collapse-direct-count-corpus | 113 | direct_count_corpus_staging |
| mop-collapse-retired-dormant-latent-store | 134 | retired_dormant_latent_store_and_git_probe |
| mop-collapse-single-durable-authority | -91 | normalized_single_durable_authority |
| mop-collapse-direct-count-pilot-authority | 1,732 | direct_count_production_and_compact_cm7_authority |
| mop-collapse-normalized-sanpo-attribute-map | 1,040 | normalized_sanpo_dr1_cm1_attribute_authority |
