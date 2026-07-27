# Substrate final ledger

Derived from the tree at `c9daed7f6c76ae9ff2b47339dab9749b046e4ca3`. Status is computed, never asserted.

67 requirements. measured 43, terminal 23, tested 1.

| id | category | status | classification | next action | rollback |
|---|---|---|---|---|---|
| A1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_MASTER_AUTHORITY.json |
| A2 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_MASTER_AUTHORITY.json, SUBSTRA |
| A3 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_EXPERIMENTAL_REQUIREMENTS.json |
| A4 | boundary | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_DEVELOPMENTAL_SAFETY.json |
| A5 | boundary | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_SENTIENCE_RESEARCH_BOUNDARY.js |
| A6 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_STATE.json, SUBSTRATE_NEXT_FRO |
| A7 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_PROGRESS_SCORECARD.json |
| C1 | temporal_continuity | tested | - | seal temporal:MOP_OWNED_TEMPORAL_CORE_V1.json, temporal:MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json | delete  and rerun mop.cognition.deliverables write; the sealed inputs  |
| C2 | workspace | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_WORKSPACE.json and rerun mop.cognition.deliverables w |
| C3 | perspective_diversity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PERSPECTIVE_SYSTEM.json and rerun mop.cognition.deliv |
| C4 | perspective_diversity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PERSPECTIVE_SYSTEM.json and rerun mop.cognition.deliv |
| C5 | perspective_arbitration | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_ARBITRATION_SYSTEM.json and rerun mop.cognition.deliv |
| O1 | ontology | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_ONTOLOGY.json and rerun mop.cognition.deliverables wr |
| O2 | epistemology | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_EPISTEMOLOGY.json, SUBSTRATE_BELIEF_REVISION.json and |
| O3 | epistemology | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_EPISTEMOLOGY.json and rerun mop.cognition.deliverable |
| N1 | unity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_RUNTIME.json and rerun mop.cognition.deliverables wri |
| N2 | boundary | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_RUNTIME.json |
| M1 | working_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| M2 | episodic_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| M3 | semantic_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| M4 | procedural_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| M5 | consolidation | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| M6 | semantic_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MEMORY_SYSTEM.json and rerun mop.cognition.deliverabl |
| W1 | world_model | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_WORLD_MODEL.json and rerun mop.cognition.deliverables |
| S1 | self_model | terminal | mechanism_null | none | delete SUBSTRATE_SELF_MODEL.json and rerun mop.cognition.deliverables  |
| K1 | metacognition | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_METACOGNITION.json and rerun mop.cognition.deliverabl |
| P1 | plasticity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun mop.cognition.delive |
| P2 | plasticity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun mop.cognition.delive |
| P3 | plasticity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun mop.cognition.delive |
| P4 | plasticity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_PLASTICITY_SYSTEM.json and rerun mop.cognition.delive |
| P5 | developmental_divergence | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_DEVELOPMENTAL_HISTORY.json and rerun mop.cognition.de |
| R1 | reorganization | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_REORGANIZATION.json and rerun mop.cognition.deliverab |
| B1 | model_body_integration | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MODEL_BODY_INTERFACE.json and rerun mop.cognition.del |
| T1 | thinking | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_THINKING_BATTERY.json and rerun mop.cognition.deliver |
| E1 | continuity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_CONTINUITY_BATTERY.json and rerun mop.cognition.deliv |
| E2 | unity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_UNITY_BATTERY.json and rerun mop.cognition.deliverabl |
| E3 | reflective_access | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_REFLECTIVE_ACCESS_BATTERY.json and rerun mop.cognitio |
| E4 | metacognition | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_METACOGNITION.json and rerun mop.cognition.deliverabl |
| E5 | goal_continuity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_MASTER_AUTHORITY.json, SUBSTRATE_AGENCY_BATTERY.json  |
| V1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_INDEPENDENT_VERIFICATION.json |
| V2 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_MUTATION_REPORT.json |
| X1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_ARCHITECTURE.json, SUBSTRATE_C |
| X2 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_CURRENT_ENTITY_SPEC.json, SUBS |
| Q1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_NULL_MAP.json |
| F1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_FINAL_PROGRAM_GRAPH.json |
| F2 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_FINAL_MASTER_AUTHORITY.json, S |
| F3 | temporal_continuity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_TEMPORAL_CORE.json and rerun mop.cognition.deliverabl |
| F4 | episodic_memory | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_REAL_SESSION_AUTHORITY.json and rerun mop.cognition.d |
| F5 | world_model | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_WORLD_MODEL_BED.json, SUBSTRATE_WORLD_MODEL_BATTERY.j |
| F6 | model_body_integration | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_BODY_COMPACT.json, SUBSTRATE_BODY_GENERAL.json, SUBST |
| F7 | goal_continuity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_GOAL_SYSTEM.json and rerun mop.cognition.deliverables |
| F8 | valuation | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_VALUATION_SYSTEM.json and rerun mop.cognition.deliver |
| F9 | grounding | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_GROUNDING.json and rerun mop.cognition.deliverables w |
| F10 | causal_reasoning | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_WORLD_MODEL.json and rerun mop.cognition.deliverables |
| F11 | developmental_divergence | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_DEVELOPMENTAL_HISTORY.json and rerun mop.cognition.de |
| F12 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_CLEAN_CLONE.json |
| Y1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_STRUCTURAL_AUDIT.json |
| Y2 | perspective_diversity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_SX2_DIVERSITY.json and rerun mop.cognition.deliverabl |
| Y3 | unity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_LONG_RUN_CERTIFICATION.json and rerun mop.cognition.d |
| Y4 | model_body_integration | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_LONG_RUN_CERTIFICATION.json and rerun mop.cognition.d |
| Z1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_LONG_RUN_AUTHORITY.json, SUBST |
| Z2 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_LONG_RUN_REHEARSAL.json |
| Z3 | boundary | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_LONG_RUN_CLAIM_BOUNDARY.json |
| L1 | authority | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_CAMPAIGN_PLAN.json |
| L2 | boundary | terminal | - | none, an authority is terminal once sealed and tested | revert the commit that sealed SUBSTRATE_CAMPAIGN_PLAN.json |
| X3 | temporal_continuity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_TEMPORAL_CORE.json and rerun mop.cognition.deliverabl |
| E6 | cognitive_integrity | measured | - | classify the result through mop.method.gate.classify_result and record it | delete SUBSTRATE_INDEPENDENT_VERIFICATION.json, SUBSTRATE_COGNITIVE_IN |

## Program graph

26 nodes, 26 terminal, 0 buildable prerequisites, 0 externally blocked.

No future wave exists as prose. Every one is a node with an entry and an exit gate.

Activation remains false.
