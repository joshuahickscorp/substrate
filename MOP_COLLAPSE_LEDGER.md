# MOP Collapse Ledger

Compact view only. Machine authorities: `MOP_COLLAPSE_STATE.json` and `collapse/MOP_REDUCTION_LOG.json`.

## Current

- Maintained Python: 33,513 LOC; ceiling: 50,000.
- Verified net Python reduction: 371,684 LOC.
- Checklist: {"active": 48, "complete": 43, "partial": 9, "pending": 118, "verified": 43}.
- Recovery: `collapse/MOP_HISTORICAL_CODE_INDEX.json` and `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.

## Active boundaries

- SEC-9: One evidence authority (compact evidence core; verifier structurally independent) -> deletion map ready (collapse/MOP_EVIDENCE_EQUIVALENCE.json): 64 byte-identical primitive defs collapsible onto one core; implement core, redirect, delete, run parity+mutation+replay (HEAVY: queue behind live run per section 2)
- SEC-11: STARSS23 first high-pressure region collapse (12-step process) -> measure and collapse the next residual STARSS producer family

## Recent green reductions

| tag | net LOC | batch |
| --- | ---: | --- |
| mop-collapse-controller-orchestrator | 7,046 | general_run_and_legacy_v3_v7_orchestration |
| mop-collapse-full-generations | 8,018 | unexecuted_full_generations_future_phase |
| mop-collapse-categorized-wave | 8,141 | completed_categorized_wave_execution_framework |
| mop-collapse-successor-horizons | 8,154 | completed_successor_horizon_v1_v2_campaigns |
| mop-collapse-generation1-program | 37,457 | completed_generation1_program |
| mop-collapse-pre-generation1-campaign | 95,886 | retired_pre_generation1_campaign_and_escs_substrate |
| mop-collapse-form-program | 12,180 | retired_form_campaign_and_future_registry |
| mop-collapse-future-scaffolds | 15,265 | retired_future_form_scaffold_implementations |
| mop-collapse-legacy-experiment-bank | 51,089 | retired_legacy_experiment_and_preflight_bank |
| mop-collapse-50k | 28,268 | retired_completed_mechanism_and_studio_campaign_surfaces |
| mop-collapse-docs-compact | -426 | four_document_machine_first_surface |
| mop-collapse-35k | 14,566 | retired_one_off_entrypoints_and_dormant_development_verticals |

Older checkpoints, proof text, and exact accounting remain in the machine log.
