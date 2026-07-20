# MOP Collapse Ledger

Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json`, `collapse/MOP_REDUCTION_LOG.json`, and `collapse/MOP_COMPLETION_AUDIT.json`.

## Current

- Maintained Python: 7,707 LOC; ceiling: 50,000.
- Runtime and campaign kernel: 5,622 LOC.
- Validation: 1,772 LOC.
- Verified reduction ledger: 397,385 LOC.
- Checklist: {"active": 1, "complete": 216, "pending": 1, "verified": 43}.
- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-13: One campaign controller (AFTER live run terminal + PR30 closure) -> keep PR 31 draft; do not alter the protected checkout

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-portable-hydration-core | 1,202 | minimal_content_addressed_portable_hydration_core |
| mop-collapse-workbench-snapshot-lifecycle | 81 | retired_orphaned_workbench_source_snapshot_copy_lifecycle |
| mop-collapse-single-installed-interface | 14 | single_installed_cli_and_retired_parallel_developer_commands |
| mop-collapse-custom-substrate-exhaustion | 1,971 | retired_completed_cm7_and_blocked_cm8_execution_vertical |
| mop-collapse-cache-reader-only | 297 | retired_unconsumed_cache_writer_and_optional_validation_mode |
| mop-collapse-matched-budget-core | 241 | single_source_consumed_matched_budget_projection |
| mop-collapse-studio-readiness-core | 252 | single_studio_readiness_profile_memory_vertical |
| mop-collapse-count-statistics-core | 330 | source_consumed_count_statistics_projection |
| mop-collapse-direct-count-producer | 182 | direct_single_consumer_count_production_vertical |
| mop-collapse-retired-onset-schema | 187 | retired_historical_onset_object_and_roundtrip_schema |
| mop-collapse-direct-count-intake | 97 | direct_real_count_intake_without_facades |
| mop-collapse-direct-count-artifact | 179 | direct_single_consumer_count_artifact_lifecycle |
