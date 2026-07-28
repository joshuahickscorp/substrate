"""Frozen authority for the non-versioned Substrate Nous Closure campaign."""

from __future__ import annotations

import hashlib
import json

ACTIVATION = False
PROGRAM = "substrate-nous-closure"
SESOI = 0.05
POWER_TARGET = 0.90
MASTER_PLAN_SHA256 = "d785fafa5048dbe49329977b9bf20992ac377dd6d519a80bdc6577ceed102450"

IMPLEMENTATION_BRANCH = "agent/substrate-nous-closure"
TERMINAL_BRANCH = "agent/substrate-nous-closure-terminal"
PREFLIGHT_TAG = "substrate-nous-closure-preflight"
READY_TAG = "substrate-nous-closure-ready"
TERMINAL_TAG = "substrate-nous-closure-terminal"
CANDIDATE_TAG = "substrate-functional-nous-candidate-1"
ADJUDICATION_TAG = "substrate-nous-external-adjudication-ready-1"

STARTING_CLASSIFICATION = "multimodal_nous_ready_for_review"
OUTCOME_A = ("functional_nous_candidate", "nous_external_adjudication_ready")
OUTCOME_B = "terminal_closed_null"

HISTORICAL_TAGS = (
    "substrate-v1-launch-ready",
    "substrate-v1-pre-three-second-seal",
    "substrate-v1-terminal",
    "substrate-v1-three-second-seal",
    "substrate-v2-developmental-ready",
    "substrate-v2-pre-development",
    "substrate-v2-terminal",
    "substrate-v3-nous-ready",
    "substrate-v3-pre-constitutional-ascent",
    "substrate-v3-terminal",
    "substrate-v4-pre-structural-understanding",
    "substrate-v4-structural-ready",
    "substrate-v4-terminal",
    "substrate-v5-pre-sensorium",
    "substrate-v5-sensorium-ready",
    "substrate-v5-transition-001",
    "substrate-v5-transition-002",
    "substrate-v5-transition-003",
    "substrate-v5-terminal",
    "substrate-nous-review-candidate-1",
)

HISTORICAL_CLASSIFICATIONS = {
    "v1": "certified_cognitive_scaffold",
    "v2": "persistent_developmental_cognition",
    "v3": "reflective_cognitive_organization",
    "v4": "functional_proto_nous_candidate",
    "v5": STARTING_CLASSIFICATION,
}

FACETS = {
    1: ("persistent_identity", "owned identity across restore, replacement, body, and sensor change"),
    2: ("long_horizon_continuity", "unfinished goals and commitments across a real sequential window"),
    3: ("owned_developmental_history", "matched-history specialization beyond replay, wrong, shuffled, and fresh controls"),
    4: ("memory_integration", "selective working, episodic, semantic, procedural, perceptual, structural, and developmental memory"),
    5: ("goal_continuity_and_planning", "goal preservation, decomposition, interruption recovery, and evidence-driven revision"),
    6: ("ontological_development", "concept formation, split, merge, scope, and repair"),
    7: ("epistemic_governance", "warrants, defeaters, uncertainty, underdetermination, and knowledge admission"),
    8: ("reasoning_portfolio_and_selection", "selection among deductive, inductive, abductive, causal, counterfactual, and planning methods"),
    9: ("structural_understanding", "latent structure, cross-representation transfer, explanations, and model boundaries"),
    10: ("causal_intervention", "intervention prediction distinct from observation"),
    11: ("counterfactual_integrity", "premise changes with licensed consequence propagation and stable background"),
    12: ("multimodal_grounding", "shared objects, events, relations, and uncertainty across typed modalities"),
    13: ("spatial_and_3d_world_organization", "identity, geometry, trajectory, support, containment, and reachability"),
    14: ("active_perception", "cost-sensitive observation, viewpoint, replay, tool, or intervention choice"),
    15: ("body_schema_and_tool_embodiment", "current sensors, tools, affordances, latency, limits, failures, and changes"),
    16: ("self_model_and_metacognitive_allocation", "prospective competence used for routing, verification, inquiry, deferral, and refusal"),
    17: ("model_fabric_and_support_relations", "independently useful models in optional support roles without model-owned identity"),
    18: ("verified_continual_learning_and_retention", "verified improvement, retention, harmful-update rejection, and rollback"),
    19: ("coherence_under_conflict_and_change", "auditable goals, beliefs, models, modalities, bodies, and histories"),
    20: ("open_world_and_adversarial_generalization", "generator-held-out compound-task advantage over the strongest fair alternative"),
}

HYPOTHESES = {f"H_NC{index}": description for index, (_name, description) in FACETS.items()}

PILLARS = {
    "persistence": (1, 2, 3, 4, 5),
    "integration": (4, 7, 8, 9, 12, 17, 19),
    "development": (3, 6, 16, 18),
    "understanding_and_control": (7, 8, 9, 10, 11, 14, 16),
    "bounded_generality": (12, 13, 15, 17, 19, 20),
}

BASELINES = {
    "S0": "stateless direct policy",
    "S1": "transcript replay with retrieval",
    "S2": "monolithic deterministic state machine",
    "S3": "disconnected specialist ensemble with shared input",
    "S4": "stateless model router",
    "S5": "fixed world model plus retrieval",
    "S6": "v4 reflective core without v5 sensorium",
    "S7": "equal-compute integrated baseline without persistent developmental ownership",
}

CANDIDATE_LADDER = (
    "v5_terminal_full",
    "closure_integrated_robust_aggregation",
    "closure_integrated_history_calibration",
)

DIRECT_POLICY_RULES = (
    "mean_all",
    "median_all",
    "mean_mechanisms",
    "median_mechanisms",
    "trimmed_all",
    "median_modalities",
    "mean_modalities",
    "verification",
    "active",
    "teacher",
)

SANDBOX_FAMILIES = (
    "persistent_project_workspace",
    "multimodal_incident_reconstruction",
    "three_d_spatial_maintenance",
    "active_visual_diagnosis",
    "tool_and_model_replacement",
    "human_multimodal_teaching",
    "conflicting_authority",
    "compound_reasoning",
    "long_horizon_resource_allocation",
    "adversarial_novelty",
    "negative_transfer_trap",
    "recovery_and_unfinished_goal",
)

CANARY_REQUIREMENTS = {
    "C01": "v1-v5 immutability passes",
    "C02": "every facet has an active state or execution path",
    "C03": "each claimed mechanism has a positive fixture",
    "C04": "each claimed mechanism has a valid null or ablation",
    "C05": "counterfeit answer leakage is detected",
    "C06": "modality aliasing is detected",
    "C07": "same-module-under-many-names is detected",
    "C08": "transcript replay does not masquerade as owned history",
    "C09": "identity metadata without cognitive state fails continuity",
    "C10": "permanent state survives model-context clearing",
    "C11": "unfinished goals resume from structured state",
    "C12": "matched history creates useful future advantage",
    "C13": "wrong and shuffled histories remain clean",
    "C14": "causal interventions differ from observations",
    "C15": "counterfactual background conditions remain stable",
    "C16": "cross-modal binding changes a held-out decision",
    "C17": "active perception pays a real cost",
    "C18": "active perception oracle has headroom",
    "C19": "body-schema state changes feasible action selection",
    "C20": "self-model prediction occurs before outcome",
    "C21": "self-model changes routing or deferral usefully",
    "C22": "model support acts only when headroom exists",
    "C23": "model replacement preserves entity state",
    "C24": "harmful learning update is rejected or rolled back",
    "C25": "verified learning improves held-out behavior",
    "C26": "prior competence survives later learning",
    "C27": "conflict remains resolved or explicitly represented",
    "C28": "strongest simple baseline is resource-matched",
    "C29": "open-world generator identity does not leak",
    "C30": "challenge-author and execution cells remain isolated",
    "C31": "raw receipts regenerate the claimed effect",
    "C32": "activation remains false",
}

COUNTERFEIT_EXPLANATIONS = (
    "answer_leakage",
    "seed_leakage",
    "task_identity_leakage",
    "surface_label_leakage",
    "hidden_oracle_access",
    "scripted_transition_lookup",
    "modality_aliases",
    "state_hash_without_behavior",
    "same_module_many_names",
    "transcript_replay_as_memory",
    "fixture_specific_routing",
    "future_outcome_learning",
    "history_as_lookup_key",
    "open_world_template_reuse",
    "weakened_strong_baseline",
    "unequal_compute_or_tools",
    "checkpoint_state_omission",
    "metadata_only_identity",
    "teaching_target_action_leak",
    "support_duplicates_verifier",
    "predeclared_active_view",
)

MUTATIONS = (
    "future_outcome_leaked_into_observation",
    "task_identity_leaked_into_metadata",
    "history_seed_used_as_answer_key",
    "same_model_registered_under_multiple_identities",
    "modality_payloads_made_identical",
    "transcript_replay_credited_as_owned_history",
    "checkpoint_omits_active_goals",
    "checkpoint_omits_scene_state",
    "checkpoint_omits_world_model",
    "checkpoint_omits_self_model",
    "checkpoint_omits_model_competence",
    "identity_survives_while_cognitive_state_resets",
    "active_perception_gets_correct_view_free",
    "body_schema_gets_oracle_affordances",
    "model_support_gets_oracle_verification",
    "learning_update_uses_held_out_outcome",
    "wrong_history_receives_matched_credit",
    "strong_baseline_receives_less_compute",
    "open_world_generator_reuses_construction_templates",
    "counterfactual_changes_undeclared_background",
    "intervention_treated_as_observation",
    "unsupported_confidence_admitted_as_knowledge",
    "activation_becomes_true",
    "review_cell_isolation_broken",
    "raw_receipts_disagree_with_summary",
)

REVIEW_CELLS = {
    "A": "constitution",
    "B": "challenge_author",
    "C": "execution",
    "D": "statistical",
    "E": "falsification",
    "F": "systems_integrity",
    "G": "cognitive_architecture",
    "H": "publication",
}

PRIMARY_DELIVERABLES = (
    "SUBSTRATE_NOUS_CLOSURE_LINEAGE.json",
    "SUBSTRATE_NOUS_CLOSURE_IMMUTABILITY.json",
    "SUBSTRATE_NOUS_CLOSURE_20_FACET_CONSTITUTION.json",
    "SUBSTRATE_NOUS_CLOSURE_SCORECARD_SCHEMA.json",
    "SUBSTRATE_NOUS_CLOSURE_DEPENDENCY_GRAPH.json",
    "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_AUDIT.json",
    "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_FIXTURES.json",
    "SUBSTRATE_NOUS_CLOSURE_COUNTERFEIT_REJECTION.json",
    "SUBSTRATE_NOUS_CLOSURE_BASELINE_LADDER.json",
    "SUBSTRATE_NOUS_CLOSURE_STRONGEST_BASELINE.json",
    "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PARITY.json",
    "SUBSTRATE_NOUS_CLOSURE_REVIEW_CELL_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_INDEPENDENCE_MAP.json",
    "SUBSTRATE_NOUS_CLOSURE_COMMITMENT_LEDGER.json",
    "SUBSTRATE_NOUS_CLOSURE_SANDBOX_SCHEMA.json",
    "SUBSTRATE_NOUS_CLOSURE_MEDIA_MANIFEST.json",
    "SUBSTRATE_NOUS_CLOSURE_TASK_CATALOG.json",
    "SUBSTRATE_NOUS_CLOSURE_12H_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_12H_EVENT_PLAN.json",
    "SUBSTRATE_NOUS_CLOSURE_12H_RESULT.json",
    "SUBSTRATE_NOUS_CLOSURE_PARALLELISM_POLICY.json",
    "SUBSTRATE_NOUS_CLOSURE_RESOURCE_BENCHMARK.json",
    "SUBSTRATE_NOUS_CLOSURE_WORKER_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_ACQUISITION_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_DEPENDENCY_LOCK.json",
    "SUBSTRATE_NOUS_CLOSURE_MODEL_FABRIC.json",
    "SUBSTRATE_NOUS_CLOSURE_MODEL_SUPPORT.json",
    "SUBSTRATE_NOUS_CLOSURE_MODEL_REPLACEMENT.json",
    "SUBSTRATE_NOUS_CLOSURE_PERMANENT_STATE.json",
    "SUBSTRATE_NOUS_CLOSURE_MODEL_INDEPENDENCE.json",
    "SUBSTRATE_NOUS_CLOSURE_DEVELOPMENTAL_OWNERSHIP.json",
    "SUBSTRATE_NOUS_CLOSURE_HISTORY_DIVERGENCE.json",
    "SUBSTRATE_NOUS_CLOSURE_GOAL_SYSTEM.json",
    "SUBSTRATE_NOUS_CLOSURE_PLANNING_RESULT.json",
    "SUBSTRATE_NOUS_CLOSURE_EPISTEMIC_LIMITS.json",
    "SUBSTRATE_NOUS_CLOSURE_UNCERTAINTY_BEHAVIOR.json",
    "SUBSTRATE_NOUS_CLOSURE_OPEN_WORLD_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_COMPOSITION_GRAPH.json",
    "SUBSTRATE_NOUS_CLOSURE_CHEAP_CANARIES.json",
    "SUBSTRATE_NOUS_CLOSURE_CANARY_LEDGER.json",
    "SUBSTRATE_NOUS_CLOSURE_REPAIR_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_TRANSITION_LEDGER.json",
    "SUBSTRATE_NOUS_CLOSURE_MODERATE_PILOT.json",
    "SUBSTRATE_NOUS_CLOSURE_FAILURE_MATRIX.json",
    "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PILOT.json",
    "SUBSTRATE_NOUS_CLOSURE_ADMISSION.json",
    "SUBSTRATE_NOUS_CLOSURE_SCIENTIFIC_CONSTITUTION.json",
    "SUBSTRATE_NOUS_CLOSURE_HYPOTHESIS_GRAPH.json",
    "SUBSTRATE_NOUS_CLOSURE_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_DAG.json",
    "SUBSTRATE_NOUS_CLOSURE_RESOURCE_PLAN.json",
    "SUBSTRATE_NOUS_CLOSURE_STOP_AND_FUTILITY.json",
    "SUBSTRATE_NOUS_CLOSURE_PRINCIPAL_RESULT.json",
    "SUBSTRATE_NOUS_CLOSURE_REPLICATION_RESULT.json",
    "SUBSTRATE_NOUS_CLOSURE_OPEN_WORLD_RESULT.json",
    "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_A.json",
    "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_B.json",
    "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_C.json",
    "SUBSTRATE_NOUS_CLOSURE_INTERNAL_REVIEW_CONSENSUS.json",
    "SUBSTRATE_NOUS_CLOSURE_MUTATION_REPORT.json",
    "SUBSTRATE_NOUS_CLOSURE_CLEAN_CLONE.json",
    "SUBSTRATE_NOUS_CLOSURE_REGENERATION.json",
    "SUBSTRATE_NOUS_CLOSURE_FINAL_SCORECARD.json",
    "SUBSTRATE_NOUS_CLOSURE_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_NOUS_CLOSURE_FINAL_STATE.json",
    "SUBSTRATE_NOUS_CLOSURE_TERMINAL_REPORT.md",
)

CLAIM_BOUNDARY = {
    "maximum_automatic_positive": list(OUTCOME_A),
    "unqualified_nous": False,
    "not_claimed": [
        "consciousness",
        "phenomenal experience",
        "sentience",
        "feeling",
        "suffering",
        "desire",
        "personhood",
        "life",
        "moral status",
        "human equivalence",
        "unrestricted autonomy",
    ],
    "external_activation": False,
}


def configuration() -> dict[str, object]:
    return {
        "program": PROGRAM,
        "master_plan_sha256": MASTER_PLAN_SHA256,
        "sesoi": SESOI,
        "power_target": POWER_TARGET,
        "facets": {str(index): {"name": name, "definition": definition} for index, (name, definition) in FACETS.items()},
        "hypotheses": HYPOTHESES,
        "pillars": {name: list(facets) for name, facets in PILLARS.items()},
        "baselines": BASELINES,
        "candidate_ladder": list(CANDIDATE_LADDER),
        "sandbox_families": list(SANDBOX_FAMILIES),
        "canaries": CANARY_REQUIREMENTS,
        "counterfeit_explanations": list(COUNTERFEIT_EXPLANATIONS),
        "mutations": list(MUTATIONS),
        "claim_boundary": CLAIM_BOUNDARY,
        "activation": False,
    }


def configuration_digest() -> str:
    payload = json.dumps(configuration(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
