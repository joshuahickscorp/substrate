# MOP Collapse Ledger

Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json`, `collapse/MOP_REDUCTION_LOG.json`, and `collapse/MOP_COMPLETION_AUDIT.json`.

## Current

- Maintained Python: 15,121 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 10,715 LOC.
- Validation: 3,172 LOC.
- Verified reduction ledger: 389,971 LOC.
- Checklist: {"active": 1, "complete": 215, "pending": 2, "verified": 43}.
- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-35k | 14,566 | retired_one_off_entrypoints_and_dormant_development_verticals |
| mop-collapse-registry-config | 6,373 | single_registry_and_18k_source_kernel |
| mop-collapse-event-horizon | 4,612 | single_evidence_controller_and_event_horizon |
| mop-collapse-current-main | 1,768 | retired_stage_ladder_falsification_and_reconciled_current_main |
| mop-collapse-proof-index | 2,320 | retired_generic_substrate_facades_and_indexed_proof |
| mop-collapse-support-minimal | 1,445 | retired_unconsumed_support_cache_maintenance_and_docs_gates |
| mop-collapse-validation | 133 | retired_synthetic_acceptance_scaffold_and_reconciled_completion |
| mop-collapse-final | 192 | compacted_final_ledger_reconciliation |
| mop-collapse-declarative-engine | -307 | selected_record_engine_becomes_global |
| mop-collapse-declarative-engine | 57 | remove_blanket_completion_override |
| mop-collapse-pr9-protections | -166 | map_pr9_protections_and_add_shared_collapse_invariants |
| mop-collapse-ledger-compact | 1,965 | replace_duplicated_checklist_generator_with_compact_state_updater |
