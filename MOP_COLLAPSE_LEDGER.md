# MOP Collapse Ledger

Compact generated view. Machine authority: `MOP_COLLAPSE_STATE.json`.

## Current

- Maintained Python: 7,525 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 5,158 LOC.
- Validation: 1,969 LOC.
- Verified reduction ledger: 421,506 LOC.
- Checklist: {"active": 1, "complete": 216, "pending": 1, "verified": 43}.
- Recovery: state `legacy_authorities` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-direct-count-gate | 68 | direct_fixed_count_gate_lifecycle |
| mop-collapse-direct-count-corpus | 113 | direct_count_corpus_staging |
| mop-collapse-retired-dormant-latent-store | 134 | retired_dormant_latent_store_and_git_probe |
| mop-collapse-single-durable-authority | -91 | normalized_single_durable_authority |
| mop-collapse-direct-count-pilot-authority | 1,732 | direct_count_production_and_compact_cm7_authority |
| mop-collapse-normalized-sanpo-attribute-map | 1,040 | normalized_sanpo_dr1_cm1_attribute_authority |
| mop-collapse-compact-unbound-proof-json | 5,785 | canonical_unbound_proof_json |
| mop-collapse-compact-bound-run-json | 1,259 | compact_bound_run_json_and_retire_stale_mot_runner |
| mop-collapse-compact-bound-proof-merkle | 10,733 | compact_complete_bound_proof_merkle_graph |
| mop-collapse-compact-machine-authorities | 3,238 | compact_generated_machine_authorities |
| mop-collapse-retired-unused-count-verification-projection | 23 | retired_unused_count_verification_projection |
| mop-collapse-retired-count-artifact-detail | 19 | retired_unconsumed_count_artifact_detail |
