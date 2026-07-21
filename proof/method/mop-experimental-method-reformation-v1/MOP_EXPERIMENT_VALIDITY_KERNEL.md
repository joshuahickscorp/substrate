# Experiment validity kernel

`src/mop/method`, 3060 lines of kernel plus 4351 lines of program
stages. No new CLI command, no new configuration root, no new registry, no second experiment engine.

## Admission sequence

scientific_premise -> causal_model -> measurement_model -> instrument_calibration -> arm_distinctness -> control_semantics -> bed_validity -> baseline_convergence -> oracle_headroom -> power_and_units -> scout -> independent_reproduction -> principal -> independent_verification -> mutation_attacks -> terminal_classification -> hypothesis_update

A stage opens only when the previous one passed. Everything before `principal` costs no training compute, so
an invalid experiment dies before it can spend any.

## Calibration

| property | holds |
|---|---|
| positive_passes | True |
| null_fails_positive_gate | True |
| harm_triggers_harm_classification | True |
| invalid_bed_does_not_become_mechanism_null | True |
| no_headroom_bed_becomes_invalid | True |
| leakage_is_rejected | True |
| arm_aliasing_is_rejected | True |
| inactive_mechanism_is_rejected | True |
| weak_estimator_distinguished_from_mechanism_failure | True |
| unconverged_baseline_blocks_comparison | True |
| wrong_control_is_rejected | True |
| all_pass | True |

## Historical defect mutations

| mutation | defect | caught at | blocks | outcome |
|---|---|---|---|---|
| temporal_conv_in_order_free_control | D1 | control_semantics | compute | rejected |
| inactive_replay | D2 | control_semantics | compute | rejected |
| buffer_that_stops_replacing | D3 | control_semantics | compute | rejected |
| aliased_lstm_and_lstm_gdumb | D4 | arm_distinctness | compute | rejected |
| phantom_parameter_group | D5 | causal_model | compute | rejected |
| variable_without_causal_path | D5 | causal_model | compute | rejected |
| analytic_value_marked_measured | D6 | causal_model | compute | rejected |
| missing_report_key | D7 | measurement_model | compute | rejected |
| wrong_baseline_comparison | D8 | baseline_convergence | compute | rejected |
| softened_verdict_wording | D9 | terminal_classification | claim | rejected |
| reviewer_consensus_overrides_reproduction | D10 | adjudication | claim | rejected |
| narrowed_coverage_scope | D11 | acceptance | compute | rejected |
| ignored_treatment_flag | D12 | arm_distinctness | compute | rejected |
| future_information_leakage | D13 | causal_model | compute | rejected |
| two_seed_false_headroom | D14 | oracle_headroom | compute | rejected |
| unconverged_baseline | D15 | baseline_convergence | compute | rejected |
| context_split_that_crosses_no_boundary | D16 | bed_validity | compute | rejected |
| brittle_plateau_criterion | D17 | baseline_convergence | compute | rejected |
| underpowered_design_admitted | D14 | power_and_units | compute | rejected |

report integrity and adjudication defects are caught automatically but after execution, because prose and reviewer votes do not exist before the run. They block the claim, not the compute, and no claim can be sealed while one is open.

## Contract vocabulary

ArmContract, BaselineContract, CausalModel, ClaimContract, ControlContract, DatasetContract, ExecutionContract, ExperimentQuestion, IndependentUnitContract, InstrumentContract, MeasurementModel, MutationContract, OracleContract, PowerContract, ResultContract, VerificationContract

## Quantity provenance

measured, recomputed, derived, analytic, assumed, structurally_guaranteed. A structurally guaranteed zero may not be reported as a measured zero,
which is defect D6.
