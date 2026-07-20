# MOP Collapse Ledger

Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json`, `collapse/MOP_REDUCTION_LOG.json`, and `collapse/MOP_COMPLETION_AUDIT.json`.

## Current

- Maintained Python: 11,457 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 8,583 LOC.
- Validation: 2,394 LOC.
- Verified reduction ledger: 393,635 LOC.
- Checklist: {"active": 1, "complete": 216, "pending": 1, "verified": 43}.
- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-raw-once-attestation | 160 | raw_once_final_attestation_authority |
| mop-collapse-unused-surface | 680 | retired_unconsumed_support_and_synthetic_starss_adapter |
| mop-collapse-custom-chain | 145 | single_receipt_chain_validator_and_one_pass_snapshot_provenance |
| mop-collapse-cache-receipt-policy | 85 | single_cache_receipt_policy_and_condensed_mutation_fixtures |
| mop-collapse-workbench-checkpoint-state | 62 | single_workbench_checkpoint_identity_and_saved_state |
| mop-collapse-workbench-orchestration | 42 | single_workbench_stop_failure_and_teacher_audit_policy |
| mop-collapse-artifact-provenance-policy | 79 | typed_artifact_json_policy_and_single_export_finalization |
| mop-collapse-finalizer-chain-cli | 127 | canonical_finalizer_writes_and_retired_artifact_cli_wrapper |
| mop-collapse-finalizer-lifecycle-cli | 53 | direct_finalizer_copy_snapshot_and_promotion_policy_without_cli_dispatch |
| mop-collapse-retired-custom-finalizer | 710 | retired_completed_zero_consumer_custom_finalizer_lifecycle |
| mop-collapse-portable-hydration-core | 1,202 | minimal_content_addressed_portable_hydration_core |
| mop-collapse-workbench-snapshot-lifecycle | 81 | retired_orphaned_workbench_source_snapshot_copy_lifecycle |
