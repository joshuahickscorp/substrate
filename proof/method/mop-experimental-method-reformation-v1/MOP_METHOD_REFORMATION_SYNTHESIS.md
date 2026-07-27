# Method reformation synthesis

## Terminal questions

| question | answer |
|---|---|
| 1 which historical defects are now automatically detectable | ["D1", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"] |
| 2 which remain manually detectable only | [] |
| 3 can an aliased arm reach principal execution | false |
| 4 can an inactive mechanism reach principal execution | false |
| 5 can a semantically invalid control reach principal execution | false |
| 6 can an unconverged baseline produce a verdict | false |
| 7 can a missing report key pass | false |
| 8 can a report reference the wrong baseline | false |
| 9 can human prose soften a sealed verdict | false |
| 10 can reviewer consensus override a reproduced defect | false |
| 11 how many invalid runs were prevented | 208 |
| 12 how much compute was avoided | {"principal_runs_that_would_have_been_invalid": 208, "updates_avoided": 249600, "wall_seconds_avoided_at_measured_optimum": 678.9, "basis": "measured aggregate throughput from the  |
| 13 did coverage meet the target | true |
| 14 was the Fast State Forge terminal evidence preserved | true |
| 15 did the reaudit find new load bearing issues | ["R1", "R2"] |
| 16 which hypotheses remain open | ["H_shared_core_capacity"] |
| 17 which were closed | ["H_fast_state", "H_readout_capacity", "H_domain_specific_representation", "H_interference", "H_bed_insufficiency"] |
| 18 which two experiments had the highest information value | ["E1", "E4"] |
| 19 were their instruments valid | {"E1": true, "E4": true} |
| 20 were their beds valid | "har_stream and speech_stream carry the inherited sealed verdict temporal_headroom_present" |
| 21 were their baselines converged | {"E1": {"har_stream": {"fast_mlp": "plateau", "pooled_mlp": "plateau"}, "speech_stream": {"fast_mlp": "patience", "pooled_mlp": "plateau"}}, "E4": {"speech_stream": "patience", "ha |
| 22 were their mechanisms active | {"E1": {"har_stream": "active", "speech_stream": "active"}, "E4": {"speech_stream": "active", "har_stream": "active"}} |
| 23 what did the scouts establish | {"E1:har_stream": {"cell_means": {"pooled_linear": 0.43864, "pooled_mlp": 0.43593, "fast_linear": 0.89167, "fast_mlp": 0.89268, "reset3_linear": 0.89891, "reset3_mlp": 0.88558, "re |
| 24 what did the principal experiments establish | {"E1": ["H_fast_state"], "E4": ["H_fast_state", "H_interference"]} |
| 25 which explanation of the substrate nulls is now strongest | "not the readout and not the bed. The recurrent core carries the capability on both sealed valid temporal beds, readout capacity separates nothing, and an order free reader loses 0 |
| 26 is the bottleneck fast state, readout, capacity, interference, data or something else | "within a domain the load bearing component is the recurrent core and its long range state. Across contexts the binding constraint is interference: every locus that acquires the ne |
| 27 what experiment should run next | "E2, shared core capacity scaling against matched separate models, which is the only remaining eligible candidate in the value queue and the one hypothesis E1 held fixed by design" |
| 28 what experiment should never be repeated | "cross modality transfer of a shared fast core on activity recognition style beds. Five programs, the same null, and the value queue refuses it" |
| 29 how did the experimental method improve | {"defect_classes": 18, "discovered_by_this_program": ["D16", "D17", "D18"], "caught_before_principal_compute": 17, "caught_before_the_claim": 3, "invalid_principal_runs_prevented": |
| 30 is any substrate mechanism scientifically positive | {"E1": ["har_stream:core_effect_at_linear", "har_stream:core_effect_at_mlp", "har_stream:long_range_state_at_linear", "har_stream:long_range_state_at_mlp", "speech_stream:best_cell |
| 31 is any architecture selected | false |
| 32 is activation licensed | false |
| 33 what claims remain forbidden | ["any owned architecture beats strong matched baselines", "shared fast dynamics transfer across modalities", "a learned plasticity gate is licensed", "functional self reorganizatio |

## The next frontier

E2: shared core capacity scaling against matched separate models.

does more shared core capacity move the acquisition retention frontier or scale both

Still open: H_shared_core_capacity. Refused: E3, E5.

## Activation

False, and never separately granted.
