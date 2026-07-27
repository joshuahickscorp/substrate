# Substrate final ledger

Derived from the tree at `df62f885d2dad8b0adb15e634b9310ab6f9e0fc0`. Status is computed, never asserted.

66 requirements. implemented 66.

| id | category | status | classification | next action | rollback |
|---|---|---|---|---|---|
| A1 | authority | implemented | - | run and record tests/substrate/test_program.py::test_naming_authority_preserves_historical_programs | revert the commit that sealed SUBSTRATE_MASTER_AUTHORITY.json |
| A2 | authority | implemented | - | run and record tests/substrate/test_program.py::test_every_deliverable_binds_to_a_real_path | revert the commit that sealed SUBSTRATE_MASTER_AUTHORITY.json, SUBSTRA |
| A3 | authority | implemented | - | run and record tests/substrate/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven | revert the commit that sealed SUBSTRATE_EXPERIMENTAL_REQUIREMENTS.json |
| A4 | boundary | implemented | - | run and record tests/substrate/test_safety.py::test_protected_surfaces_cannot_be_removed_by_adaptation | revert the commit that sealed SUBSTRATE_DEVELOPMENTAL_SAFETY.json |
| A5 | boundary | implemented | - | run and record tests/substrate/test_safety.py::test_forbidden_claim_vocabulary_is_refused | revert the commit that sealed SUBSTRATE_SENTIENCE_RESEARCH_BOUNDARY.js |
| A6 | authority | implemented | - | run and record tests/substrate/test_program.py::test_batch_selection_is_dependency_ready_and_independent | revert the commit that sealed SUBSTRATE_STATE.json, SUBSTRATE_NEXT_FRO |
| A7 | authority | implemented | - | run and record tests/substrate/test_program.py::test_evidence_never_rises_from_implementation_alone | revert the commit that sealed SUBSTRATE_PROGRESS_SCORECARD.json |
| C1 | temporal_continuity | implemented | - | run and record tests/temporal/test_temporal_core.py | delete  and rerun substrate.deliverables write; the sealed inputs are  |
| C2 | workspace | implemented | - | run and record tests/substrate/test_workspace.py | delete SUBSTRATE_WORKSPACE.json and rerun substrate.deliverables write |
| C3 | perspective_diversity | implemented | - | run and record tests/substrate/test_perspectives.py | delete SUBSTRATE_PERSPECTIVE_SYSTEM.json and rerun substrate.deliverab |
| C4 | perspective_diversity | implemented | - | run and record tests/substrate/test_perspectives.py::test_learned_selector_stays_closed_without_headroom | delete SUBSTRATE_PERSPECTIVE_SYSTEM.json and rerun substrate.deliverab |
| C5 | perspective_arbitration | implemented | - | run and record tests/substrate/test_perspectives.py::test_minority_hypothesis_survives_arbitration | delete SUBSTRATE_ARBITRATION_SYSTEM.json and rerun substrate.deliverab |
| O1 | ontology | implemented | - | run and record tests/substrate/test_ontology.py::test_identity_preservation_over_time, tests/substrate/test_ontology.py::test_mistaken_merge_detection, tests/substrate/test_ontology.py::test_counterfactual_objects_never_merge_with_actual_ones, tests/substrate/test_ontology.py::test_the_self_environment_boundary_holds | delete SUBSTRATE_ONTOLOGY.json and rerun substrate.deliverables write; |
| O2 | epistemology | implemented | - | run and record tests/substrate/test_epistemology.py::test_hidden_dependency_failure_is_caught_by_retraction_propagating, tests/substrate/test_epistemology.py::test_circular_justification_is_refused_not_followed, tests/substrate/test_epistemology.py::test_the_dependency_aware_policy_beats_a_confident_but_undermined_claim | delete SUBSTRATE_EPISTEMOLOGY.json, SUBSTRATE_BELIEF_REVISION.json and |
| O3 | epistemology | implemented | - | run and record tests/substrate/test_epistemology.py::test_epistemic_value_and_not_confidence_chooses_the_action | delete SUBSTRATE_EPISTEMOLOGY.json and rerun substrate.deliverables wr |
| N1 | unity | implemented | - | run and record tests/substrate/test_runtime.py::test_one_cycle_runs_every_declared_stage, tests/substrate/test_runtime.py::test_a_skipped_stage_says_why_rather_than_looking_like_one_that_ran | delete SUBSTRATE_RUNTIME.json and rerun substrate.deliverables write;  |
| N2 | boundary | implemented | - | run and record tests/substrate/test_runtime.py::test_the_loop_never_acts_on_the_world, tests/substrate/test_runtime.py::test_a_tampered_checkpoint_is_refused_rather_than_silently_restored | revert the commit that sealed SUBSTRATE_RUNTIME.json |
| M1 | working_memory | implemented | - | run and record tests/substrate/test_memory.py::test_working_memory_capacity_interference_and_decay | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| M2 | episodic_memory | implemented | - | run and record tests/substrate/test_memory.py::test_generated_episode_cannot_be_promoted_without_verification | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| M3 | semantic_memory | implemented | - | run and record tests/substrate/test_memory.py::test_semantic_supersession_preserves_provenance | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| M4 | procedural_memory | implemented | - | run and record tests/substrate/test_memory.py::test_procedure_requires_transfer_beyond_source_episodes | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| M5 | consolidation | implemented | - | run and record tests/substrate/test_memory.py::test_consolidation_policies_are_distinct_and_ordered | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| M6 | semantic_memory | implemented | - | run and record tests/substrate/test_memory.py::test_hygiene_never_destroys_audit_required_records | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun substrate.deliverables w |
| W1 | world_model | implemented | - | run and record tests/substrate/test_world.py | delete SUBSTRATE_WORLD_MODEL.json and rerun substrate.deliverables wri |
| S1 | self_model | implemented | mechanism_null | run and record tests/substrate/test_selfmodel.py | delete SUBSTRATE_SELF_MODEL.json and rerun substrate.deliverables writ |
| K1 | metacognition | implemented | - | run and record tests/substrate/test_metacog.py | delete SUBSTRATE_METACOGNITION.json and rerun substrate.deliverables w |
| P1 | plasticity | implemented | - | run and record tests/substrate/test_plasticity.py::test_every_level_declares_the_seven_required_fields | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun substrate.deliverabl |
| P2 | plasticity | implemented | - | run and record tests/substrate/test_plasticity.py::test_fast_adaptation_does_not_touch_shared_parameters | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun substrate.deliverabl |
| P3 | plasticity | implemented | - | run and record tests/substrate/test_plasticity.py::test_slow_adaptation_requires_repeated_evidence_and_rollback | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun substrate.deliverabl |
| P4 | plasticity | implemented | - | run and record tests/substrate/test_plasticity.py::test_learned_policy_stays_closed_without_headroom | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun substrate.deliverabl |
| P5 | developmental_divergence | implemented | - | run and record tests/substrate/test_plasticity.py::test_learning_to_learn_requires_cross_task_generalization | delete SUBSTRATE_DEVELOPMENTAL_HISTORY.json and rerun substrate.delive |
| R1 | reorganization | implemented | - | run and record tests/substrate/test_plasticity.py::test_forbidden_reorganizations_are_refused | delete SUBSTRATE_REORGANIZATION.json and rerun substrate.deliverables  |
| B1 | model_body_integration | implemented | - | run and record tests/substrate/test_body.py | delete SUBSTRATE_MODEL_BODY_INTERFACE.json and rerun substrate.deliver |
| T1 | thinking | implemented | - | run and record tests/substrate/test_batteries.py::test_thinking_requires_a_declared_alternative_to_beat | delete SUBSTRATE_THINKING_BATTERY.json and rerun substrate.deliverable |
| E1 | continuity | implemented | - | run and record tests/substrate/test_batteries.py::test_continuity_survives_context_removal_from_owned_state | delete SUBSTRATE_CONTINUITY_BATTERY.json and rerun substrate.deliverab |
| E2 | unity | implemented | - | run and record tests/substrate/test_batteries.py::test_unity_measures_global_availability_not_shared_mutability | delete SUBSTRATE_UNITY_BATTERY.json and rerun substrate.deliverables w |
| E3 | reflective_access | implemented | - | run and record tests/substrate/test_batteries.py::test_reflective_report_fails_closed_without_provenance | delete SUBSTRATE_REFLECTIVE_ACCESS_BATTERY.json and rerun substrate.de |
| E4 | metacognition | implemented | - | run and record tests/substrate/test_metacog.py::test_attention_ranks_by_declared_drivers_under_budget | delete SUBSTRATE_METACOGNITION.json and rerun substrate.deliverables w |
| E5 | goal_continuity | implemented | - | run and record tests/substrate/test_safety.py::test_unauthorized_goal_creation_is_refused | delete SUBSTRATE_MASTER_AUTHORITY.json, SUBSTRATE_AGENCY_BATTERY.json  |
| V1 | authority | implemented | - | run and record tests/substrate/test_verification.py::test_recomputation_agrees_with_every_sealed_artifact | revert the commit that sealed SUBSTRATE_INDEPENDENT_VERIFICATION.json |
| V2 | authority | implemented | - | run and record tests/substrate/test_verification.py::test_every_mutation_names_a_distinct_guard_and_died | revert the commit that sealed SUBSTRATE_MUTATION_REPORT.json |
| X1 | authority | implemented | - | run and record tests/substrate/test_verification.py::test_the_capability_map_never_outruns_the_item_table | revert the commit that sealed SUBSTRATE_ARCHITECTURE.json, SUBSTRATE_C |
| X2 | authority | implemented | - | run and record tests/substrate/test_verification.py::test_the_entity_report_makes_no_forbidden_claim | revert the commit that sealed SUBSTRATE_CURRENT_ENTITY_SPEC.json, SUBS |
| Q1 | authority | implemented | - | run and record tests/substrate/test_experiments.py::test_sx1_is_refused_because_its_effect_is_true_by_construction, tests/substrate/test_experiments.py::test_a_refusal_is_not_recorded_as_a_null | revert the commit that sealed SUBSTRATE_NULL_MAP.json |
| F1 | authority | implemented | - | run and record tests/substrate/test_final_program.py::test_no_future_wave_exists_only_as_prose, tests/substrate/test_final_program.py::test_a_buildable_blocker_is_work_and_only_an_external_one_is_terminal | revert the commit that sealed SUBSTRATE_FINAL_PROGRAM_GRAPH.json |
| F2 | authority | implemented | - | run and record tests/substrate/test_final_program.py::test_every_requirement_carries_a_rollback, tests/substrate/test_final_program.py::test_the_authority_binds_the_final_plan_and_the_inherited_programs | revert the commit that sealed SUBSTRATE_FINAL_MASTER_AUTHORITY.json, S |
| F3 | temporal_continuity | implemented | - | run and record tests/substrate/test_final_program.py::test_no_licensed_core_exists_and_the_control_says_so, tests/substrate/test_final_program.py::test_the_five_information_sources_are_never_collapsed | delete SUBSTRATE_TEMPORAL_CORE.json and rerun substrate.deliverables w |
| F4 | episodic_memory | implemented | - | run and record tests/substrate/test_final_program.py::test_the_session_authority_is_real_and_certified | delete SUBSTRATE_REAL_SESSION_AUTHORITY.json and rerun substrate.deliv |
| F5 | world_model | implemented | - | run and record tests/substrate/test_worldbed.py::test_the_bed_is_admissible_only_if_the_best_action_varies, tests/substrate/test_worldbed.py::test_the_model_must_change_an_action_and_improve_it | delete SUBSTRATE_WORLD_MODEL_BED.json, SUBSTRATE_WORLD_MODEL_BATTERY.j |
| F6 | model_body_integration | implemented | - | run and record tests/substrate/test_final_program.py::test_three_body_classes_conform_through_one_interface, tests/substrate/test_final_program.py::test_the_frontier_body_is_recorded_as_externally_blocked_not_substituted | delete SUBSTRATE_BODY_COMPACT.json, SUBSTRATE_BODY_GENERAL.json, SUBST |
| F7 | goal_continuity | implemented | - | run and record tests/substrate/test_final_program.py::test_a_goal_cannot_authorize_itself_or_widen_its_parent | delete SUBSTRATE_GOAL_SYSTEM.json and rerun substrate.deliverables wri |
| F8 | valuation | implemented | - | run and record tests/substrate/test_final_program.py::test_valuation_is_authorized_and_refuses_to_be_fitted | delete SUBSTRATE_VALUATION_SYSTEM.json and rerun substrate.deliverable |
| F9 | grounding | implemented | - | run and record tests/substrate/test_final_program.py::test_grounding_refuses_a_symbol_with_no_referent | delete SUBSTRATE_GROUNDING.json and rerun substrate.deliverables write |
| F10 | causal_reasoning | implemented | - | run and record tests/substrate/test_world.py::test_intervening_is_not_the_same_operation_as_observing, tests/substrate/test_world.py::test_a_null_counterfactual_reproduces_the_factual_prediction | delete SUBSTRATE_WORLD_MODEL.json and rerun substrate.deliverables wri |
| F11 | developmental_divergence | implemented | - | run and record tests/substrate/test_final_program.py::test_divergence_has_a_working_control | delete SUBSTRATE_DEVELOPMENTAL_HISTORY.json and rerun substrate.delive |
| F12 | authority | implemented | - | run and record tests/substrate/test_worldbed.py::test_the_clean_clone_checks_are_declared | revert the commit that sealed SUBSTRATE_CLEAN_CLONE.json |
| Y1 | authority | implemented | - | run and record tests/substrate/test_certification.py::test_the_structural_audit_passes_every_check, tests/substrate/test_certification.py::test_a_dynamically_named_artifact_is_refused_by_the_producer_scan | revert the commit that sealed SUBSTRATE_STRUCTURAL_AUDIT.json |
| Y2 | perspective_diversity | implemented | - | run and record tests/substrate/test_certification.py::test_sx2_closes_because_the_oracle_ceiling_is_low, tests/substrate/test_certification.py::test_every_sx2_comparison_is_compute_matched | delete SUBSTRATE_SX2_DIVERSITY.json and rerun substrate.deliverables w |
| Y3 | unity | implemented | - | run and record tests/substrate/test_certification.py::test_every_ablatable_runtime_stage_is_active, tests/substrate/test_certification.py::test_a_stage_with_no_possible_null_control_says_so | delete SUBSTRATE_LONG_RUN_CERTIFICATION.json and rerun substrate.deliv |
| Y4 | model_body_integration | implemented | - | run and record tests/substrate/test_certification.py::test_every_session_canary_passes, tests/substrate/test_certification.py::test_the_three_bodies_are_pairwise_distinct | delete SUBSTRATE_LONG_RUN_CERTIFICATION.json and rerun substrate.deliv |
| Y5 | unity | implemented | mechanism_null | run and record tests/substrate/test_nous.py::test_the_closed_loop_has_no_missing_link, tests/substrate/test_nous.py::test_a_null_is_not_reported_as_an_instrument_failure, tests/substrate/test_nous.py::test_the_classification_never_reaches_a_claim_about_experience | delete SUBSTRATE_NOUS_CLOSURE.json and rerun substrate.deliverables wr |
| Z1 | authority | implemented | - | run and record tests/substrate/test_execution.py::test_every_unit_is_certified_or_necessary, tests/substrate/test_execution.py::test_a_live_edit_after_the_freeze_is_detectable | revert the commit that sealed SUBSTRATE_LONG_RUN_AUTHORITY.json, SUBST |
| Z2 | authority | implemented | - | run and record tests/substrate/test_execution.py::test_the_rehearsal_breaks_things_and_survives | revert the commit that sealed SUBSTRATE_LONG_RUN_REHEARSAL.json |
| Z3 | boundary | implemented | - | run and record tests/substrate/test_execution.py::test_the_claim_ceiling_forbids_what_it_must | revert the commit that sealed SUBSTRATE_LONG_RUN_CLAIM_BOUNDARY.json |
| X3 | temporal_continuity | implemented | - | run and record tests/substrate/test_verification.py::test_the_temporal_core_record_tracks_the_live_program | delete SUBSTRATE_TEMPORAL_CORE.json and rerun substrate.deliverables w |
| E6 | cognitive_integrity | implemented | - | run and record tests/substrate/test_safety.py::test_integrity_violation_is_detected_and_fails_closed | delete SUBSTRATE_INDEPENDENT_VERIFICATION.json, SUBSTRATE_COGNITIVE_IN |

## Program graph

26 nodes, 26 terminal, 0 buildable prerequisites, 0 externally blocked.

No future wave exists as prose. Every one is a node with an entry and an exit gate.

Activation remains false.
