# Cognitive Material Genesis — Grok review archive

External activation is `false`. Grok opinions are proposals; evidence decides.

- invocations: **62** across **8** of the eight declared rounds
- distinct roles credited: **59** (minimum 32, preferred 48)
- feasibility grades out of 20: min 3, median 5, max 8
- blocking defects raised: **144**

## Rounds

- `architecture_proposals` — 12
- `blind_review` — 41
- `challenge_design` — 2
- `code_review` — 2
- `cross_examination` — 1
- `final_review` — 2
- `post_canary_review` — 1
- `post_pilot_review` — 1

## What the later rounds changed

Three findings from the cross-examination, post-canary and post-pilot rounds
changed the published conclusion rather than decorating it.

The cross-examination refused the reading that a monolith beat structure,
pointing out that K1 is itself a monolith and scores below the structured K2,
and that what separates the strongest control is exact content-addressable
association against the candidates' lossy low-bit projections. That confound
is now stated as the first limitation.

The post-canary round observed that twelve green canaries coexisted with a
counterfeit tying the best candidate, because each canary exercises one
mechanism in isolation and none ranks a counterfeit against a candidate.

The post-pilot round measured that the attempt repair reduced the gap from
roughly two hundredfold to roughly sevenfold rather than closing it. The
residual ratio and its direction are now published as numbers.

## Every credited invocation

| role | round | grade | blocking | session |
|---|---|---:|---:|---|
| `adaptive_topology_reviewer` | architecture_proposals | 5/20 | 2 | `019fadc1` |
| `cellular_field_reviewer` | architecture_proposals | 5/20 | 2 | `019fadc1` |
| `cognitive_material_architect` | architecture_proposals | 5/20 | 3 | `019fada8` |
| `continuous_time_field_reviewer` | architecture_proposals | 6/20 | 2 | `019fadc1` |
| `event_sourced_field_reviewer` | architecture_proposals | 6/20 | 2 | `019fadc1` |
| `exact_shell_constitution_reviewer` | architecture_proposals | 6/20 | 3 | `019fadb9` |
| `graph_field_reviewer` | architecture_proposals | 6/20 | 1 | `019fadc1` |
| `grok_original_material_author` | architecture_proposals | 7/20 | 1 | `019fada8` |
| `integrated_field_reviewer` | architecture_proposals | 6/20 | 2 | `019fadc7` |
| `monolithic_field_reviewer` | architecture_proposals | 5/20 | 3 | `019fadc1` |
| `predictive_field_reviewer` | architecture_proposals | 6/20 | 2 | `019fadc1` |
| `state_space_field_reviewer` | architecture_proposals | 5/20 | 2 | `019fadc1` |
| `catastrophic_forgetting_reviewer` | blind_review | 7/20 | 2 | `019fadb7` |
| `cognitive_compiler_reviewer` | blind_review | 5/20 | 2 | `019fadb4` |
| `cognitive_metabolism_reviewer` | blind_review | 7/20 | 2 | `019fadd4` |
| `consolidation_reviewer` | blind_review | 6/20 | 2 | `019fadd1` |
| `continual_learning_reviewer` | blind_review | 5/20 | 2 | `019fadb5` |
| `continuous_time_reviewer` | blind_review | 6/20 | 2 | `019fadd4` |
| `counterfactual_validity_reviewer` | blind_review | 5/20 | 2 | `019fadd4` |
| `curriculum_design_reviewer` | blind_review | 5/20 | 3 | `019fadb3` |
| `decompilation_safety_reviewer` | blind_review | 6/20 | 3 | `019fadd4` |
| `developmental_measurement_reviewer` | blind_review | 4/20 | 3 | `019fadb3` |
| `equal_resource_auditor` | blind_review | 4/20 | 4 | `019fadab` |
| `evaluation_security_reviewer` | blind_review | 4/20 | 3 | `019fadab` |
| `fast_plasticity_reviewer` | blind_review | 6/20 | 2 | `019fadc8` |
| `intermediate_plasticity_reviewer` | blind_review | 6/20 | 2 | `019fadd1` |
| `kernel_revision_reviewer` | blind_review | 6/20 | 1 | `019fadd2` |
| `learned_codebook_reviewer` | blind_review | 6/20 | 1 | `019fadc7` |
| `low_bit_arithmetic_reviewer` | blind_review | 5/20 | 1 | `019fadb7` |
| `metaplasticity_reviewer` | blind_review | 7/20 | 2 | `019fadb5` |
| `mixed_radix_packing_reviewer` | blind_review | 6/20 | 0 | `019fadc7` |
| `multiplicity_and_power_reviewer` | blind_review | 6/20 | 3 | `019fadb9` |
| `native_training_reviewer` | blind_review | 4/20 | 2 | `019fadab` |
| `numerical_stability_reviewer` | blind_review | 6/20 | 2 | `019fadc7` |
| `oracle_headroom_reviewer` | blind_review | 8/20 | 3 | `019fadb9` |
| `performance_reviewer` | blind_review | 5/20 | 1 | `019fadab` |
| `precision_economics_reviewer` | blind_review | 5/20 | 3 | `019fadb5` |
| `pruning_and_archival_reviewer` | blind_review | 7/20 | 1 | `019fadd4` |
| `red_team_answer_leakage` | blind_review | 4/20 | 4 | `019fadb3` |
| `red_team_checkpoint_coverage` | blind_review | 6/20 | 3 | `019fadb9` |
| `red_team_counterfeit_development` | blind_review | 5/20 | 4 | `019fadb3` |
| `red_team_history_laundering` | blind_review | 6/20 | 2 | `019fadb9` |
| `red_team_precision_gaming` | blind_review | 6/20 | 3 | `019fadb9` |
| `red_team_resource_parity` | blind_review | 5/20 | 3 | `019fadb4` |
| `red_team_shortcut_compilation` | blind_review | 5/20 | 3 | `019fadb9` |
| `red_team_topology_padding` | blind_review | 5/20 | 2 | `019fadab` |
| `rigidity_reviewer` | blind_review | 7/20 | 1 | `019fadd1` |
| `s2_fairness_reviewer` | blind_review | 5/20 | 4 | `019fada8` |
| `sensorium_and_modality_reviewer` | blind_review | 6/20 | 2 | `019fadb7` |
| `shadow_field_reviewer` | blind_review | 5/20 | 2 | `019fadb5` |
| `statistical_reviewer` | blind_review | 4/20 | 3 | `019fada8` |
| `topology_growth_economics_reviewer` | blind_review | 6/20 | 4 | `019fadb7` |
| `vector_quantization_reviewer` | blind_review | 7/20 | 2 | `019fadc7` |
| `challenge_generator_author` | challenge_design | 5/20 | 3 | `019fada8` |
| `hidden_composition_author` | challenge_design | 5/20 | 3 | `019fadab` |
| `code_review_campaign` | code_review | 5/20 | 4 | `019fadd4` |
| `code_review_core` | code_review | 5/20 | 2 | `019fadb9` |
| `cognitive_material_architect` | cross_examination | 4/20 | 1 | `019fae0e` |
| `falsification_reviewer` | final_review | 3/20 | 3 | `019fada8` |
| `publication_reviewer` | final_review | 4/20 | 2 | `019fadd4` |
| `red_team_counterfeit_development` | post_canary_review | 4/20 | 2 | `019fae0e` |
| `falsification_reviewer` | post_pilot_review | 4/20 | 3 | `019fae0e` |

## How to read the grades

Each reviewer graded whether this program could reach a defensible Outcome A
**on the evidence available when they read it**, not whether the plan sounded
good. A median of 5 out of 20 is the swarm saying, independently and
repeatedly, that it could not. The published result agrees with them.

Reviews that did not parse, did not carry the assigned role and round, or set
activation true were refused by the ledger rather than recorded.
