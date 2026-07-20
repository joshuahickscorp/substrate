# MOP Collapse Ledger

Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json`, `collapse/MOP_REDUCTION_LOG.json`, and `collapse/MOP_COMPLETION_AUDIT.json`.

## Current

- Maintained Python: 13,751 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 9,640 LOC.
- Validation: 2,908 LOC.
- Verified reduction ledger: 391,341 LOC.
- Checklist: {"active": 1, "complete": 216, "pending": 1, "verified": 43}.
- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-final | 192 | compacted_final_ledger_reconciliation |
| mop-collapse-declarative-engine | -307 | selected_record_engine_becomes_global |
| mop-collapse-declarative-engine | 57 | remove_blanket_completion_override |
| mop-collapse-pr9-protections | -166 | map_pr9_protections_and_add_shared_collapse_invariants |
| mop-collapse-ledger-compact | 1,965 | replace_duplicated_checklist_generator_with_compact_state_updater |
| mop-collapse-custom-model | 46 | canonical_custom_substrate_model_authority |
| mop-collapse-budget-facade | 192 | retired_unconsumed_dual_budget_facade |
| mop-collapse-raw-once-attestation | 160 | raw_once_final_attestation_authority |
| mop-collapse-unused-surface | 680 | retired_unconsumed_support_and_synthetic_starss_adapter |
| mop-collapse-custom-chain | 145 | single_receipt_chain_validator_and_one_pass_snapshot_provenance |
| mop-collapse-cache-receipt-policy | 85 | single_cache_receipt_policy_and_condensed_mutation_fixtures |
| mop-collapse-workbench-checkpoint-state | 62 | single_workbench_checkpoint_identity_and_saved_state |
