# Substrate ledger

Generated from the tree at commit `f45abfd6f74e7a6bad2101b2773f6283e60417c0`. Status is derived, never asserted: an item is
implemented because its files exist, tested because a recorded test ledger says so, measured
because its evidence is sealed, and terminal because a scientific classification exists for it.

Items: 66. Levels: implemented 66.

| id | section | title | level | dependencies | next action |
|---|---|---|---|---|---|
| A1 | 2 | Naming and historical continuity authority | implemented | none | run and record tests/substrate/test_program.py::test_naming_authority_preserves_historical_programs |
| A2 | 21 | Master deliverable set exists and binds to real things | implemented | A1 | run and record tests/substrate/test_program.py::test_every_deliverable_binds_to_a_real_path |
| A3 | 18 | Experimental requirements bind every new Substrate experiment | implemented | A2 | run and record tests/substrate/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven |
| A4 | 19 | Developmental safety envelope | implemented | A2 | run and record tests/substrate/test_safety.py::test_protected_surfaces_cannot_be_removed_by_adaptation |
| A5 | 16 | Sentience research boundary | implemented | A2 | run and record tests/substrate/test_safety.py::test_forbidden_claim_vocabulary_is_refused |
| A6 | 17 | Continuous six batch research program | implemented | A2 | run and record tests/substrate/test_program.py::test_batch_selection_is_dependency_ready_and_independent |
| A7 | 20 | Scorecard separates implementation from evidence | implemented | A2 | run and record tests/substrate/test_program.py::test_evidence_never_rises_from_implementation_alone |
| C1 | 6.1 | Temporal core identified and selected | implemented | none | run and record tests/temporal/test_temporal_core.py |
| C2 | 6.2 | Typed cognitive workspace | implemented | A2 | run and record tests/substrate/test_workspace.py |
| C3 | 6.3 | Mixture of Perspectives, perspectives as processes | implemented | C2 | run and record tests/substrate/test_perspectives.py |
| C4 | 6.4 | Perspective selection ladder | implemented | C3 | run and record tests/substrate/test_perspectives.py::test_learned_selector_stays_closed_without_headroom |
| C5 | 6.5 | Perspective arbitration | implemented | C3 | run and record tests/substrate/test_perspectives.py::test_minority_hypothesis_survives_arbitration |
| O1 | 6 | Typed ontology with the nine distinctions it refuses to collapse | implemented | C2 | run and record tests/substrate/test_ontology.py::test_identity_preservation_over_time, tests/substrate/test_ontology.py::test_mistaken_merge_detection, tests/substrate/test_ontology.py::test_counterfactual_objects_never_merge_with_actual_ones, tests/substrate/test_ontology.py::test_the_self_environment_boundary_holds |
| O2 | 7 | Epistemology where justification is a graph, not a confidence table | implemented | O1 | run and record tests/substrate/test_epistemology.py::test_hidden_dependency_failure_is_caught_by_retraction_propagating, tests/substrate/test_epistemology.py::test_circular_justification_is_refused_not_followed, tests/substrate/test_epistemology.py::test_the_dependency_aware_policy_beats_a_confident_but_undermined_claim |
| O3 | 7.2 | Metacognitive control runs on epistemic value, not confidence | implemented | O2, K1 | run and record tests/substrate/test_epistemology.py::test_epistemic_value_and_not_confidence_chooses_the_action |
| N1 | 5 | The runtime loop that makes the modules one entity | implemented | C2, C5 | run and record tests/substrate/test_runtime.py::test_one_cycle_runs_every_declared_stage, tests/substrate/test_runtime.py::test_a_skipped_stage_says_why_rather_than_looking_like_one_that_ran |
| N2 | 19 | The loop has no path to acting on the world | implemented | A4, N1 | run and record tests/substrate/test_runtime.py::test_the_loop_never_acts_on_the_world, tests/substrate/test_runtime.py::test_a_tampered_checkpoint_is_refused_rather_than_silently_restored |
| M1 | 7.1 | Working memory | implemented | C2 | run and record tests/substrate/test_memory.py::test_working_memory_capacity_interference_and_decay |
| M2 | 7.2 | Episodic memory | implemented | C2 | run and record tests/substrate/test_memory.py::test_generated_episode_cannot_be_promoted_without_verification |
| M3 | 7.3 | Semantic memory | implemented | M2 | run and record tests/substrate/test_memory.py::test_semantic_supersession_preserves_provenance |
| M4 | 7.4 | Procedural memory | implemented | M2 | run and record tests/substrate/test_memory.py::test_procedure_requires_transfer_beyond_source_episodes |
| M5 | 7.5 | Consolidation | implemented | M2, M3, M4 | run and record tests/substrate/test_memory.py::test_consolidation_policies_are_distinct_and_ordered |
| M6 | 7.6 | Forgetting and hygiene | implemented | M3 | run and record tests/substrate/test_memory.py::test_hygiene_never_destroys_audit_required_records |
| W1 | 8 | World model | implemented | C2, M2 | run and record tests/substrate/test_world.py |
| S1 | 9 | Self model | implemented | C2, M2 | run and record tests/substrate/test_selfmodel.py |
| K1 | 10 | Metacognition | implemented | C5, S1 | run and record tests/substrate/test_metacog.py |
| P1 | 11.1 | Plasticity hierarchy | implemented | A4, C2 | run and record tests/substrate/test_plasticity.py::test_every_level_declares_the_seven_required_fields |
| P2 | 11.2 | Fast adaptation | implemented | P1, M1 | run and record tests/substrate/test_plasticity.py::test_fast_adaptation_does_not_touch_shared_parameters |
| P3 | 11.3 | Slow adaptation | implemented | P1, M5 | run and record tests/substrate/test_plasticity.py::test_slow_adaptation_requires_repeated_evidence_and_rollback |
| P4 | 11.4 | Plasticity policy | implemented | P2, P3 | run and record tests/substrate/test_plasticity.py::test_learned_policy_stays_closed_without_headroom |
| P5 | 11.5 | Learning to learn | implemented | P4 | run and record tests/substrate/test_plasticity.py::test_learning_to_learn_requires_cross_task_generalization |
| R1 | 12 | Bounded functional reorganization | implemented | P1, C4, A4 | run and record tests/substrate/test_plasticity.py::test_forbidden_reorganizations_are_refused |
| B1 | 13 | Model body interface | implemented | C2 | run and record tests/substrate/test_body.py |
| T1 | 14 | Operationalized thinking | implemented | C5, K1 | run and record tests/substrate/test_batteries.py::test_thinking_requires_a_declared_alternative_to_beat |
| E1 | 15.1 | Continuity | implemented | M2, S1 | run and record tests/substrate/test_batteries.py::test_continuity_survives_context_removal_from_owned_state |
| E2 | 15.2 | Unity | implemented | C2, C5 | run and record tests/substrate/test_batteries.py::test_unity_measures_global_availability_not_shared_mutability |
| E3 | 15.3 | Reflective access | implemented | S1, M3 | run and record tests/substrate/test_batteries.py::test_reflective_report_fails_closed_without_provenance |
| E4 | 15.4 | Endogenous attention | implemented | K1 | run and record tests/substrate/test_metacog.py::test_attention_ranks_by_declared_drivers_under_budget |
| E5 | 15.5 | Autonomous goal maintenance | implemented | A4, E1 | run and record tests/substrate/test_safety.py::test_unauthorized_goal_creation_is_refused |
| V1 | 18 | Independent recomputation of every sealed Substrate number | implemented | A3 | run and record tests/substrate/test_verification.py::test_recomputation_agrees_with_every_sealed_artifact |
| V2 | 18 | Mutation attacks on every declared guard | implemented | V1 | run and record tests/substrate/test_verification.py::test_every_mutation_names_a_distinct_guard_and_died |
| X1 | 21 | Architecture and capability map | implemented | A2 | run and record tests/substrate/test_verification.py::test_the_capability_map_never_outruns_the_item_table |
| X2 | 21 | Current entity specification and report | implemented | X1, A5 | run and record tests/substrate/test_verification.py::test_the_entity_report_makes_no_forbidden_claim |
| Q1 | 18 | The first Substrate experiment reaches a decision with its reason recorded | implemented | A3, C2 | run and record tests/substrate/test_experiments.py::test_sx1_is_refused_because_its_effect_is_true_by_construction, tests/substrate/test_experiments.py::test_a_refusal_is_not_recorded_as_a_null |
| F1 | 5 | One materialized program graph, no prose waves | implemented | A6 | run and record tests/substrate/test_final_program.py::test_no_future_wave_exists_only_as_prose, tests/substrate/test_final_program.py::test_a_buildable_blocker_is_work_and_only_an_external_one_is_terminal |
| F2 | 4 | A final authority that resumes without conversation history | implemented | F1 | run and record tests/substrate/test_final_program.py::test_every_requirement_carries_a_rollback, tests/substrate/test_final_program.py::test_the_authority_binds_the_final_plan_and_the_inherited_programs |
| F3 | 9 | Temporal core integrated through a versioned interface, as a declared control | implemented | C1, N1 | run and record tests/substrate/test_final_program.py::test_no_licensed_core_exists_and_the_control_says_so, tests/substrate/test_final_program.py::test_the_five_information_sources_are_never_collapsed |
| F4 | 8 | A real session authority built from a session nobody wrote for this | implemented | M2 | run and record tests/substrate/test_final_program.py::test_the_session_authority_is_real_and_certified |
| F5 | 12 | World model inside the decision loop on a state dependent bed | implemented | F4, W1 | run and record tests/substrate/test_worldbed.py::test_the_bed_is_admissible_only_if_the_best_action_varies, tests/substrate/test_worldbed.py::test_the_model_must_change_an_action_and_improve_it |
| F6 | 14 | Three model body classes conforming through one interface | implemented | B1, F3 | run and record tests/substrate/test_final_program.py::test_three_body_classes_conform_through_one_interface, tests/substrate/test_final_program.py::test_the_frontier_body_is_recorded_as_externally_blocked_not_substituted |
| F7 | 23 | Goal system where a goal cannot authorize itself | implemented | E5 | run and record tests/substrate/test_final_program.py::test_a_goal_cannot_authorize_itself_or_widen_its_parent |
| F8 | 24 | Valuation that is externally authorized and refuses to be fitted | implemented | F7 | run and record tests/substrate/test_final_program.py::test_valuation_is_authorized_and_refuses_to_be_fitted |
| F9 | 20 | Grounding where a verbal definition is not evidence | implemented | F4, O1 | run and record tests/substrate/test_final_program.py::test_grounding_refuses_a_symbol_with_no_referent |
| F10 | 21 | Intervention is distinguished from observation | implemented | W1 | run and record tests/substrate/test_world.py::test_intervening_is_not_the_same_operation_as_observing, tests/substrate/test_world.py::test_a_null_counterfactual_reproduces_the_factual_prediction |
| F11 | 29 | Developmental divergence with a working control | implemented | F4, P5 | run and record tests/substrate/test_final_program.py::test_divergence_has_a_working_control |
| F12 | 47 | Clean clone reproduces the evidence away from this machine | implemented | V1, V2 | run and record tests/substrate/test_worldbed.py::test_the_clean_clone_checks_are_declared |
| Y1 | 2 | Structural audit: exclusive producers, no stale outputs, no activation path | implemented | F1 | run and record tests/substrate/test_certification.py::test_the_structural_audit_passes_every_check, tests/substrate/test_certification.py::test_a_dynamically_named_artifact_is_refused_by_the_producer_scan |
| Y2 | 16 | SX2 diversity closed on a compute matched comparison | implemented | C4, A3 | run and record tests/substrate/test_certification.py::test_sx2_closes_because_the_oracle_ceiling_is_low, tests/substrate/test_certification.py::test_every_sx2_comparison_is_compute_matched |
| Y3 | 2 | Every runtime stage is active, with a null control where one is possible | implemented | N1, Y1 | run and record tests/substrate/test_certification.py::test_every_ablatable_runtime_stage_is_active, tests/substrate/test_certification.py::test_a_stage_with_no_possible_null_control_says_so |
| Y4 | 14 | Session and body canaries on the sealed authority | implemented | F4, F6 | run and record tests/substrate/test_certification.py::test_every_session_canary_passes, tests/substrate/test_certification.py::test_the_three_bodies_are_pairwise_distinct |
| Y5 | 3 | The closure gate: do the parts compose into one organization | implemented | Y3, Y4 | run and record tests/substrate/test_nous.py::test_the_closed_loop_has_no_missing_link, tests/substrate/test_nous.py::test_a_null_is_not_reported_as_an_instrument_failure, tests/substrate/test_nous.py::test_the_classification_never_reaches_a_claim_about_experience |
| Z1 | 3 | One frozen long run DAG of certified or necessary units | implemented | Y1, Y3, Y4 | run and record tests/substrate/test_execution.py::test_every_unit_is_certified_or_necessary, tests/substrate/test_execution.py::test_a_live_edit_after_the_freeze_is_detectable |
| Z2 | 4 | A rehearsal that tries to break the machinery | implemented | Z1 | run and record tests/substrate/test_execution.py::test_the_rehearsal_breaks_things_and_survives |
| Z3 | 45 | The claim ceiling the run cannot exceed | implemented | A5, Z1 | run and record tests/substrate/test_execution.py::test_the_claim_ceiling_forbids_what_it_must |
| X3 | 6.1 | Substrate Temporal Core v1 selection record | implemented | C1 | run and record tests/substrate/test_verification.py::test_the_temporal_core_record_tracks_the_live_program |
| E6 | 15.6 | Cognitive integrity | implemented | A4, M2 | run and record tests/substrate/test_safety.py::test_integrity_violation_is_detected_and_fails_closed |

## Selected next batch

Primary: A1 Naming and historical continuity authority. run and record tests/substrate/test_program.py::test_naming_authority_preserves_historical_programs
Secondary: A3 Experimental requirements bind every new Substrate experiment. run and record tests/substrate/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven

Activation remains false.
