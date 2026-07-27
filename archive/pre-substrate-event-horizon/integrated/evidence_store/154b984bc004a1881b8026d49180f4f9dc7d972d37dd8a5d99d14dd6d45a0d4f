# Causal experiment graphs

## E1 

19 nodes, 19 edges, admissible: True.

| edge | kind |
|---|---|
| temporal_capability to recurrent_core | assumed_scientific_relation |
| recurrent_core to set_core | implemented_causal_path |
| readout_capacity to set_readout | implemented_causal_path |
| set_core to representation | implemented_causal_path |
| set_readout to logits | implemented_causal_path |
| representation to logits | implemented_causal_path |
| logits to test_accuracy | measured_relation |
| set_core to test_accuracy | measured_relation |
| set_readout to test_accuracy | measured_relation |
| pooled_control to representation | implemented_causal_path |
| reset_control to representation | implemented_causal_path |
| stream_order to recurrent_core | implemented_causal_path |
| final_window_label to set_core | forbidden_information_path |
| unit_identity to test_accuracy | assumed_scientific_relation |
| capacity to test_accuracy | structural_guarantee |
| segment_alignment to test_accuracy | assumed_scientific_relation |
| subject_or_speaker to test_accuracy | measured_relation |
| set_core to compute | measured_relation |
| recurrent_core to temporal_capability | assumed_scientific_relation |

## E4 

18 nodes, 16 edges, admissible: True.

| edge | kind |
|---|---|
| fast_adaptation to owned_state | assumed_scientific_relation |
| owned_state to recentre_state | implemented_causal_path |
| slow_parameters to gradient_update | implemented_causal_path |
| recentre_state to representation | implemented_causal_path |
| gradient_update to representation | implemented_causal_path |
| state_noise_control to representation | implemented_causal_path |
| no_adapt_control to representation | implemented_causal_path |
| context_B_statistics to recentre_state | implemented_causal_path |
| representation to acquisition_B | measured_relation |
| representation to retention_A | measured_relation |
| speaker_or_subject to acquisition_B | measured_relation |
| speaker_or_subject to retention_A | measured_relation |
| context_A_labels_during_B to recentre_state | forbidden_information_path |
| unit_identity to acquisition_B | assumed_scientific_relation |
| adaptation_budget to acquisition_B | structural_guarantee |
| gradient_update to compute | measured_relation |
