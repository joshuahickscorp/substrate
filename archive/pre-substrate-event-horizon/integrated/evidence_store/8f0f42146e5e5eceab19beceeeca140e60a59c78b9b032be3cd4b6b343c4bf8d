# Temporal core hypothesis graph

| id | premise |
|---|---|
| H1_recurrence | a recurrent state transition provides value a stateless model cannot recover, even with matched parameter count and explicit causal history |
| H2_explicit_history | the core effect is access to historical observations, and a stateless model given a matched history reproduces it |
| H3_capacity | the core effect is mainly increased representation capacity |
| H4_state_horizon | persistent state helps only because it carries information over a minimum useful temporal horizon |
| H5_optimization | recurrent models appear superior because they optimize more easily under the current training authority |
| H6_core_horizon_interaction | larger cores help only when state persists sufficiently long |
| H7_architecture_family | the effect is specific to one recurrent implementation rather than to persistent temporal computation generally |
| H8_bed_specificity | the effect is real on the two valid controlled beds and does not transfer to a third natural stream |

## Preregistered result mapping

| result | supports | weakens | closes |
|---|---|---|---|
| recurrent_beats_matched_history | H1_recurrence | H2_explicit_history | none |
| matched_history_matches_recurrent | H2_explicit_history | H1_recurrence | H1_recurrence |
| capacity_monotonic_and_large | H3_capacity | H1_recurrence | none |
| capacity_flat_or_saturating | none | H3_capacity | H3_capacity |
| horizon_threshold_at_dependency_length | H4_state_horizon | none | none |
| horizon_flat | none | H4_state_horizon, H1_recurrence | H4_state_horizon |
| capacity_helps_only_at_long_horizon | H6_core_horizon_interaction | none | none |
| capacity_and_horizon_independent | none | H6_core_horizon_interaction | H6_core_horizon_interaction |
| all_recurrent_families_agree | H1_recurrence | H7_architecture_family | H7_architecture_family |
| one_recurrent_family_dissents | H7_architecture_family | H1_recurrence | none |
| unconverged_arms_explain_the_gap | H5_optimization | H1_recurrence, H3_capacity | none |
| converged_everywhere_and_gap_remains | none | H5_optimization | H5_optimization |
| third_bed_agrees | none | H8_bed_specificity | H8_bed_specificity |
| third_bed_dissents | H8_bed_specificity | none | none |
| third_bed_invalid | none | none | none |
| readout_capacity_reproduces_the_effect | H3_capacity | H1_recurrence | none |
| readout_capacity_flat | none | H3_capacity | none |
