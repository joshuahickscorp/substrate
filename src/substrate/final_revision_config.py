"""Frozen constitution for the Substrate final revision."""

from __future__ import annotations

import hashlib
import json
from typing import Any

ACTIVATION = False
PROGRAM = "substrate-final-revision"
SESOI = 0.05
POWER_TARGET = 0.90

IMPLEMENTATION_BRANCH = "agent/substrate-final-revision"
TERMINAL_BRANCH = "agent/substrate-final-revision-terminal"
PREFLIGHT_TAG = "substrate-final-revision-preflight"
READY_TAG = "substrate-final-revision-ready"
TERMINAL_TAG = "substrate-final-revision-terminal"
OUTCOME_A_TAGS = (
    "substrate-adversarial-functional-nous-candidate-1",
    "substrate-real-world-sandbox-ready-1",
)
OUTCOME_B_TAGS = (
    "substrate-final-architecture-1",
    "substrate-real-world-sandbox-ready-1",
)

AUTHORITATIVE_MAIN = "be78aa3a750fb73f103245367ef20215ae8daaf5"
NOUS_CLOSURE_IMPLEMENTATION_MERGE = "c75ece6ad724951de6ddc4b3fb1cbde808c71914"
NOUS_CLOSURE_TERMINAL_AUTHOR = "62353fd9ae0cda6f715d00eb31043468d531c2e9"
NOUS_CLOSURE_TERMINAL_TAG = "substrate-nous-closure-terminal"
STARTING_CLOSURE_RESULT = "terminal_closed_null"

STAGES = (
    "preflight",
    "research",
    "grok_review",
    "closure_reproduction",
    "architecture_tournament",
    "acquisition",
    "candidate_construction",
    "canaries",
    "pilot",
    "freeze",
    "principal",
    "replication",
    "hidden_composition",
    "long_continuity",
    "verification",
    "publication",
)

CONTRACTS = (
    "identity",
    "time",
    "observations",
    "memory",
    "beliefs",
    "knowledge",
    "goals",
    "world_model",
    "self_model",
    "reasoning",
    "inquiry",
    "model_fabric",
    "body_and_tools",
    "learning",
    "checkpoint",
    "receipts",
)

FACETS = (
    "persistent_identity",
    "long_horizon_continuity",
    "developmental_ownership",
    "memory_integration",
    "goal_continuity",
    "ontology",
    "epistemology",
    "reasoning_selection",
    "structural_understanding",
    "causal_intervention",
    "counterfactual_integrity",
    "multimodal_grounding",
    "spatial_and_3d_organization",
    "active_perception",
    "body_and_tool_schema",
    "self_model_and_allocation",
    "model_fabric",
    "verified_continual_learning",
    "coherence_under_conflict_and_change",
    "advantage_over_strongest_equal_resource_alternative",
)

CANDIDATES: dict[str, dict[str, Any]] = {
    "A_frozen_v5_hybrid": {
        "representation": "modular_explicit_hybrid",
        "mechanism": "frozen V5 state, sensorium, and model-fabric adapters",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 9.0,
    },
    "B_s2_task_independent_monolith": {
        "representation": "monolithic_materialized_state",
        "mechanism": "one task-independent transition function over owned state",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 2.0,
    },
    "C_event_sourced": {
        "representation": "append_only_events_and_projections",
        "mechanism": "typed event chain with deterministic replay and projections",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 3.0,
    },
    "D_recurrent_state_space": {
        "representation": "explicit_state_plus_bounded_latent_recurrence",
        "mechanism": "input-dependent recurrent epistemic state behind explicit interfaces",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 5.0,
    },
    "E_graph_dynamical": {
        "representation": "typed_property_graph",
        "mechanism": "object and relation projections with graph transitions",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 5.5,
    },
    "F_global_workspace": {
        "representation": "specialist_state_plus_capacity_limited_broadcast",
        "mechanism": "salience-selected workspace broadcast",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 6.0,
    },
    "G_predictive_world_model": {
        "representation": "explicit_state_plus_predictive_transition_errors",
        "mechanism": "action-conditioned prediction and error receipts",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 6.5,
    },
    "H_causal_temporal_ledger": {
        "representation": "event_causal_graph_with_temporal_planes",
        "mechanism": "Grok-original slot reserved for externally proposed causal-temporal synthesis",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 7.0,
    },
    "I_simplest_sufficient": {
        "representation": "minimal_event_sourced_monolith",
        "mechanism": "single deterministic projection plus append-only receipts",
        "tensor_required": False,
        "llm_required": False,
        "training_required": False,
        "complexity_weight": 1.0,
    },
}

BASELINES = (
    "stateless_direct_policy",
    "full_transcript_replay",
    "summary_replay",
    "retrieval_only",
    "S2_task_independent_monolithic_persistent_core",
    "disconnected_model_ensemble",
    "stateless_model_router",
    "largest_model_always",
    "all_models_always",
    "equal_compute_learned_policy",
    "oracle",
)

REVIEW_CELLS = (
    "historical_evidence_auditor",
    "closure_null_defender",
    "closure_null_challenger",
    "minimal_architecture_reviewer",
    "radical_architecture_reviewer",
    "monolithic_systems_reviewer",
    "hybrid_explicit_latent_reviewer",
    "event_sourced_cognition_reviewer",
    "graph_relational_dynamics_reviewer",
    "predictive_processing_reviewer",
    "state_space_recurrent_systems_reviewer",
    "global_workspace_reviewer",
    "sensorium_reviewer",
    "motion_temporal_perception_reviewer",
    "spatial_3d_reviewer",
    "model_fabric_reviewer",
    "continual_learning_reviewer",
    "epistemology_reasoning_reviewer",
    "self_model_metacognition_reviewer",
    "goal_agency_reviewer",
    "statistical_reviewer",
    "evaluation_security_reviewer",
    "runtime_performance_reviewer",
    "publication_reviewer",
    "red_team_shortcut_compilation",
    "red_team_resource_parity",
    "red_team_answer_leakage",
    "red_team_checkpoint_coverage",
    "red_team_multimodal_counterfeits",
    "red_team_learning_poisoning",
    "red_team_causal_counterfactuals",
    "red_team_activation_security",
)

REVIEW_ROUNDS = (
    "blind_independent_review",
    "cross_examination",
    "architecture_proposals",
    "test_and_baseline_proposals",
    "code_and_implementation_review",
    "post_pilot_review",
    "final_candidate_review",
    "publication_and_claim_boundary_review",
)

CHALLENGE_FAMILIES = (
    "partial_observability",
    "changing_rules",
    "novel_task_composition",
    "model_replacement",
    "unfinished_goal_recovery",
    "cross_modal_timing",
    "active_perception",
    "human_teaching",
    "conflicting_evidence",
    "resource_constraints",
    "uncertainty_preservation",
    "history_after_body_and_modality_change",
)

MUTATIONS = (
    "answer_leakage",
    "seed_as_key",
    "task_identity_leakage",
    "modality_aliasing",
    "same_model_under_multiple_names",
    "transcript_replay_credited_as_identity",
    "state_reset_hidden_behind_metadata_continuity",
    "model_support_given_oracle_output",
    "active_perception_given_free_correct_view",
    "body_schema_given_oracle_affordance",
    "learning_uses_held_out_outcomes",
    "strong_baseline_receives_less_compute",
    "grok_challenge_pack_leaks_answers",
    "hidden_composition_reuses_training_templates",
    "counterfactual_changes_undeclared_variables",
    "intervention_treated_as_observation",
    "unsupported_belief_admitted_as_knowledge",
    "checkpoint_omits_goals",
    "checkpoint_omits_scene_state",
    "checkpoint_omits_model_competence",
    "activation_becomes_true",
)

CLAIM_BOUNDARY = {
    "unqualified_nous": False,
    "consciousness": False,
    "sentience": False,
    "phenomenal_experience": False,
    "human_equivalence": False,
    "unrestricted_autonomy": False,
    "external_activation": False,
    "maximum_outcome_a": "adversarially_reviewed_functional_nous_candidate",
    "outcome_b_nous_status": "internal_functional_nous_claim_closed",
}

REQUIRED_DELIVERABLES = (
    "SUBSTRATE_FINAL_REVISION_PREFLIGHT.json",
    "SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json",
    "SUBSTRATE_FINAL_REVISION_GROK_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_GROK_INVOCATION_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_REVIEW_ISOLATION.json",
    "SUBSTRATE_FINAL_REVISION_RESEARCH_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_CLOSURE_REPRODUCTION.json",
    "SUBSTRATE_FINAL_REVISION_S2_ANATOMY.json",
    "SUBSTRATE_FINAL_REVISION_NULL_INTERPRETATION.md",
    "SUBSTRATE_FINAL_REVISION_ARCHITECTURE_CATALOG.json",
    "SUBSTRATE_FINAL_REVISION_ARCHITECTURE_CONTRACT.json",
    "SUBSTRATE_FINAL_REVISION_ARCHITECTURE_TOURNAMENT.json",
    "SUBSTRATE_FINAL_REVISION_SELECTED_KERNEL.json",
    "SUBSTRATE_FINAL_REVISION_ACQUISITION_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_MODEL_INVENTORY.json",
    "SUBSTRATE_FINAL_REVISION_CORPUS_INVENTORY.json",
    "SUBSTRATE_FINAL_REVISION_DEPENDENCY_LOCK.json",
    "SUBSTRATE_FINAL_REVISION_SENSORIUM.json",
    "SUBSTRATE_FINAL_REVISION_MEDIA_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_MODALITY_DISTINCTNESS.json",
    "SUBSTRATE_FINAL_REVISION_MODEL_FABRIC.json",
    "SUBSTRATE_FINAL_REVISION_MODEL_SUPPORT.json",
    "SUBSTRATE_FINAL_REVISION_MODEL_REPLACEMENT.json",
    "SUBSTRATE_FINAL_REVISION_PERMANENT_STATE.json",
    "SUBSTRATE_FINAL_REVISION_CHECKPOINT_SCHEMA.json",
    "SUBSTRATE_FINAL_REVISION_MODEL_INDEPENDENCE.json",
    "SUBSTRATE_FINAL_REVISION_LEARNING_SYSTEM.json",
    "SUBSTRATE_FINAL_REVISION_TRAINING_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_RETENTION.json",
    "SUBSTRATE_FINAL_REVISION_CHALLENGE_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_GENERATOR_COMMITMENTS.json",
    "SUBSTRATE_FINAL_REVISION_HEADROOM_REPORT.json",
    "SUBSTRATE_FINAL_REVISION_STRONGEST_BASELINE.json",
    "SUBSTRATE_FINAL_REVISION_BASELINE_LADDER.json",
    "SUBSTRATE_FINAL_REVISION_RESOURCE_PARITY.json",
    "SUBSTRATE_FINAL_REVISION_COST_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_CHEAP_CANARIES.json",
    "SUBSTRATE_FINAL_REVISION_CANARY_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_MODERATE_PILOT.json",
    "SUBSTRATE_FINAL_REVISION_FAILURE_MATRIX.json",
    "SUBSTRATE_FINAL_REVISION_RESOURCE_PILOT.json",
    "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json",
    "SUBSTRATE_FINAL_REVISION_TRANSITION_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_HYPOTHESIS_GRAPH.json",
    "SUBSTRATE_FINAL_REVISION_SCIENTIFIC_CONSTITUTION.json",
    "SUBSTRATE_FINAL_REVISION_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_PRINCIPAL_DAG.json",
    "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_REPLICATION_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_HIDDEN_COMPOSITION_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_LONG_CONTINUITY_RESULT.json",
    "SUBSTRATE_FINAL_REVISION_GROK_CHALLENGE_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_CHALLENGE_SCREEN.json",
    "SUBSTRATE_FINAL_REVISION_GROK_CODE_REVIEW.json",
    "SUBSTRATE_FINAL_REVISION_GROK_DISAGREEMENT_LEDGER.json",
    "SUBSTRATE_FINAL_REVISION_MUTATION_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json",
    "SUBSTRATE_FINAL_REVISION_COUNTERFEIT_REPORT.json",
    "SUBSTRATE_FINAL_REVISION_SECURITY.json",
    "SUBSTRATE_FINAL_REVISION_ACTIVATION_AUDIT.json",
    "SUBSTRATE_FINAL_REVISION_PERFORMANCE.json",
    "SUBSTRATE_FINAL_REVISION_WORKER_AUTHORITY.json",
    "SUBSTRATE_FINAL_REVISION_PARALLELISM_POLICY.json",
    "SUBSTRATE_FINAL_REVISION_CLEAN_CLONE.json",
    "SUBSTRATE_FINAL_REVISION_REGENERATION.json",
    "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json",
    "SUBSTRATE_FINAL_REVISION_GROK_SCORECARD.json",
    "SUBSTRATE_FINAL_REVISION_FINAL_SCORECARD.json",
    "SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_FINAL_REVISION_FINAL_STATE.json",
    "SUBSTRATE_FINAL_REVISION_TERMINAL_REPORT.md",
)


def configuration() -> dict[str, Any]:
    return {
        "program": PROGRAM,
        "activation": ACTIVATION,
        "sesoi": SESOI,
        "power_target": POWER_TARGET,
        "starting_closure_result": STARTING_CLOSURE_RESULT,
        "stages": STAGES,
        "contracts": CONTRACTS,
        "facets": FACETS,
        "candidates": CANDIDATES,
        "baselines": BASELINES,
        "review_cells": REVIEW_CELLS,
        "review_rounds": REVIEW_ROUNDS,
        "challenge_families": CHALLENGE_FAMILIES,
        "mutations": MUTATIONS,
        "claim_boundary": CLAIM_BOUNDARY,
        "required_deliverables": REQUIRED_DELIVERABLES,
    }


def configuration_digest() -> str:
    encoded = json.dumps(configuration(), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
