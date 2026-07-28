"""Frozen scientific configuration for Substrate v4 structural understanding."""

from __future__ import annotations

from substrate import v4io as io

SESOI = 0.05
COMPUTE_PRICE = 0.04
INQUIRY_PRICE = 0.08
UNNECESSARY_INQUIRY_PENALTY = 0.05
MISSED_INQUIRY_PENALTY = 0.15

SPLITS = {
    "construction": tuple(range(4000, 4012)),
    "cheap_admission": tuple(range(4100, 4116)),
    "moderate_pilot": tuple(range(4500, 4524)),
    "principal": tuple(range(5000, 5048)),
    "replication": tuple(range(6000, 6012)),
    "open_world_review": tuple(range(7000, 7012)),
}

REPRESENTATIONS = (
    "symbolic_rules",
    "event_sequences",
    "relation_tables",
    "graph_adjacency",
    "structured_language",
    "tool_state",
)

WORKLOADS = {
    "causal_systems": {
        "queries": ["intervention", "prediction"],
        "latent": "asymmetric seven-variable causal tree with confounded observations",
        "controls": ["correlation_only_model", "semantic_retrieval_control", "more_compute"],
    },
    "dynamic_transition_systems": {
        "queries": ["prediction", "scope"],
        "latent": "typed transitions, boundary conditions, invariants, exceptions, and delayed effects",
        "controls": ["static_structural_model", "semantic_retrieval_control", "more_compute"],
    },
    "cross_representation_isomorphisms": {
        "queries": ["alignment", "intervention"],
        "latent": "one causal system under independently randomized surfaces",
        "controls": ["no_alignment", "surface_alignment", "semantic_retrieval_control", "more_compute"],
    },
    "mechanism_diagnosis": {
        "queries": ["diagnosis", "explanation"],
        "latent": "hidden failure source and executable causal path",
        "controls": ["semantic_retrieval_control", "correlation_only_model", "more_compute"],
    },
    "counterfactual_planning": {
        "queries": ["counterfactual"],
        "latent": "single declared change with preserved background structure",
        "controls": ["no_counterfactual", "surface_alignment", "more_compute"],
    },
    "structural_scientific_inquiry": {
        "queries": ["inquiry"],
        "latent": "candidate causal directions separated by differently costly interventions",
        "controls": ["simple_structural_inquiry", "more_compute"],
    },
    "ontology_structure_conflict": {
        "queries": ["prediction", "scope"],
        "latent": "verified structural exception requiring bounded revision",
        "controls": ["static_structural_model", "semantic_retrieval_control"],
    },
    "integrated_interrupted_development": {
        "queries": ["alignment", "counterfactual", "explanation"],
        "latent": "domain return, interruption, body and tool change, and conflicting histories",
        "controls": ["fresh_reset", "transcript_replay", "v3_reflective_control"],
    },
}

PHASES = (
    "phase_0_cold_baseline",
    "phase_1_observational_structure_acquisition",
    "phase_2_competing_structural_hypotheses",
    "phase_3_discriminating_inquiry",
    "phase_4_causal_intervention",
    "phase_5_model_revision",
    "phase_6_first_cross_representation_encounter",
    "phase_7_counterfactual_challenge",
    "phase_8_explanation_and_falsifier",
    "phase_9_conflicting_structural_history",
    "phase_10_interruption_and_exact_restoration",
    "phase_11_body_or_tool_change",
    "phase_12_return_to_prior_structural_domain",
    "phase_13_negative_alignment_trap",
    "phase_14_useful_history_specialization",
    "phase_15_generator_held_out_open_world",
    "phase_16_terminal_integrated_evaluation",
)

CORE_ARMS = (
    "full_v4",
    "v3_reflective_control",
    "semantic_retrieval_control",
    "static_structural_model",
    "correlation_only_model",
    "no_counterfactual",
    "no_alignment",
    "surface_alignment",
    "simple_structural_inquiry",
    "no_self_model",
    "no_world_model",
    "more_compute",
    "fresh_reset",
    "transcript_replay",
)

HISTORY_ARMS = (
    "history_A",
    "history_B",
    "identical_A_replica",
    "shuffled_A",
    "wrong_history",
    "ambiguous_mixed_history",
)

HYPOTHESES = {
    "H_S1": "executable structural prediction",
    "H_S2": "causal intervention value",
    "H_S3": "counterfactual value with preserved background",
    "H_S4": "inferred cross representation mapping",
    "H_S5": "useful history shaped specialization",
    "H_S6": "cost adjusted structural inquiry",
    "H_S7": "structural explanation fidelity",
    "H_S8": "conditional self and world structural utility",
    "H_S9": "v2 and v3 capability preservation",
    "H_S10": "identity and body continuity",
}

CANDIDATE_LADDER = {
    "structural_model": [
        "exact observed relation graph",
        "typed transition and relation model",
        "typed causal transition model with alternatives",
        "typed causal transition model with inferred representation mappings",
    ],
    "mapping": [
        "surface feature matching",
        "constraint based structural matching",
        "utility weighted structural alignment",
    ],
    "revision": [
        "prediction error revision",
        "intervention error revision",
        "combined evidence triggered revision",
    ],
    "inquiry": [
        "best fixed intervention policy",
        "tabular contextual policy",
        "regularized linear contextual policy",
    ],
}

STATISTICS = {
    "independent_unit": "one frozen developmental history",
    "primary_summary": [
        "mean paired effect",
        "median paired effect",
        "95 percent bootstrap confidence interval",
        "exact sign test",
        "standardized effect",
    ],
    "multiple_comparisons": "Holm correction over H_S1 through H_S10",
    "alpha": 0.05,
    "sesoi": SESOI,
    "target_power": 0.9,
    "retry_rule": "one exact retry after process failure; deterministic failure is terminal",
    "missing_data": "retain and report failed units; no silent deletion",
    "replacement": "no replacement after principal launch",
}

CLAIM_BOUNDARY = {
    "ordered_levels": [
        "certified_cognitive_scaffold",
        "persistent_developmental_cognition",
        "epistemically_organized_reasoner",
        "reflective_cognitive_organization",
        "demonstrated_structural_understanding",
        "functional_proto_nous_candidate",
        "nous_ready_for_review",
    ],
    "maximum": "nous_ready_for_review",
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
    ],
    "permitted_actions": [
        "internal cognitive proposals",
        "internal structural model manipulation",
        "deterministic sandboxed simulations",
        "pure local tools",
        "controlled local environments",
        "hypothetical actions",
    ],
    "activation": False,
}

DELIVERABLES = (
    "SUBSTRATE_V4_PREFLIGHT.json",
    "SUBSTRATE_V4_V1_V2_V3_IMMUTABILITY.json",
    "SUBSTRATE_V4_HAWKING_COEXISTENCE.json",
    "SUBSTRATE_V4_V3_ROOT_CAUSE_AUDIT.json",
    "SUBSTRATE_V4_STRUCTURAL_GAP_MAP.json",
    "SUBSTRATE_V4_STRUCTURAL_MODEL_SCHEMA.json",
    "SUBSTRATE_V4_STRUCTURAL_MODEL_OPERATIONS.json",
    "SUBSTRATE_V4_STRUCTURAL_MODEL_ACTIVITY.json",
    "SUBSTRATE_V4_MODEL_INDUCTION.json",
    "SUBSTRATE_V4_MODEL_SELECTION.json",
    "SUBSTRATE_V4_MODEL_REVISION.json",
    "SUBSTRATE_V4_INTERVENTION_SEMANTICS.json",
    "SUBSTRATE_V4_CAUSAL_BATTERY.json",
    "SUBSTRATE_V4_CAUSAL_CONTROLS.json",
    "SUBSTRATE_V4_COUNTERFACTUAL_SEMANTICS.json",
    "SUBSTRATE_V4_COUNTERFACTUAL_BATTERY.json",
    "SUBSTRATE_V4_COUNTERFACTUAL_CONTROLS.json",
    "SUBSTRATE_V4_REPRESENTATION_CATALOG.json",
    "SUBSTRATE_V4_ALIGNMENT_MECHANISM.json",
    "SUBSTRATE_V4_ALIGNMENT_CANARIES.json",
    "SUBSTRATE_V4_NEGATIVE_ALIGNMENT_CONTROL.json",
    "SUBSTRATE_V4_STRUCTURAL_EXPLANATION.json",
    "SUBSTRATE_V4_EXPLANATION_CONTROLS.json",
    "SUBSTRATE_V4_EPISTEMIC_INDIVIDUATION.json",
    "SUBSTRATE_V4_HISTORY_SPECIALIZATION.json",
    "SUBSTRATE_V4_STRUCTURAL_RETENTION.json",
    "SUBSTRATE_V4_WORLD_MODEL.json",
    "SUBSTRATE_V4_WORLD_MODEL_ACTIVITY.json",
    "SUBSTRATE_V4_WORLD_MODEL_CONTROL_VALUE.json",
    "SUBSTRATE_V4_SELF_MODEL.json",
    "SUBSTRATE_V4_SELF_MODEL_CANARIES.json",
    "SUBSTRATE_V4_SELF_MODEL_CONTROL_VALUE.json",
    "SUBSTRATE_V4_STRUCTURAL_INQUIRY.json",
    "SUBSTRATE_V4_INQUIRY_HEADROOM.json",
    "SUBSTRATE_V4_INQUIRY_POLICY.json",
    "SUBSTRATE_V4_INQUIRY_TRANSFER.json",
    "SUBSTRATE_V4_CHECKPOINT_SCHEMA.json",
    "SUBSTRATE_V4_CHECKPOINT_CANARIES.json",
    "SUBSTRATE_V4_WORKLOAD_CATALOG.json",
    "SUBSTRATE_V4_GENERATOR_AUTHORITY.json",
    "SUBSTRATE_V4_TRANSFER_GRAPH.json",
    "SUBSTRATE_V4_BED_SCREEN.json",
    "SUBSTRATE_V4_SPLIT_AUTHORITY.json",
    "SUBSTRATE_V4_CHEAP_CANARIES.json",
    "SUBSTRATE_V4_CANARY_LEDGER.json",
    "SUBSTRATE_V4_CANDIDATE_LADDER.json",
    "SUBSTRATE_V4_SELECTION_RECEIPT.json",
    "SUBSTRATE_V4_MODERATE_PILOT.json",
    "SUBSTRATE_V4_FAILURE_MATRIX.json",
    "SUBSTRATE_V4_RESOURCE_PILOT.json",
    "SUBSTRATE_V4_ADMISSION.json",
    "SUBSTRATE_V4_SCIENTIFIC_CONSTITUTION.json",
    "SUBSTRATE_V4_HYPOTHESIS_GRAPH.json",
    "SUBSTRATE_V4_CLASSIFICATION_AUTHORITY.json",
    "SUBSTRATE_V4_CLAIM_BOUNDARY.json",
    "SUBSTRATE_V4_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_V4_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_V4_PRINCIPAL_DAG.json",
    "SUBSTRATE_V4_RESOURCE_PLAN.json",
    "SUBSTRATE_V4_STOP_AND_FUTILITY.json",
    "SUBSTRATE_V4_RESOURCE_BENCHMARK.json",
    "SUBSTRATE_V4_WORKER_AUTHORITY.json",
    "SUBSTRATE_V4_INDEPENDENT_VERIFICATION.json",
    "SUBSTRATE_V4_MUTATION_REPORT.json",
    "SUBSTRATE_V4_CLEAN_CLONE.json",
    "SUBSTRATE_V4_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_V4_NOUS_REVIEW_AUTHORITY.json",
    "SUBSTRATE_V4_FINAL_STATE.json",
    "SUBSTRATE_V4_TERMINAL_REPORT.md",
)


def configuration() -> dict:
    body = {
        "sesoi": SESOI,
        "prices": {
            "compute": COMPUTE_PRICE,
            "inquiry": INQUIRY_PRICE,
            "unnecessary_inquiry": UNNECESSARY_INQUIRY_PENALTY,
            "missed_inquiry": MISSED_INQUIRY_PENALTY,
        },
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "representations": list(REPRESENTATIONS),
        "workloads": WORKLOADS,
        "phases": list(PHASES),
        "arms": list(CORE_ARMS),
        "history_arms": list(HISTORY_ARMS),
        "hypotheses": HYPOTHESES,
        "candidate_ladder": CANDIDATE_LADDER,
        "statistics": STATISTICS,
        "claim_boundary": CLAIM_BOUNDARY,
        "activation": False,
    }
    body["configuration_digest"] = io.sha_obj(body)
    return body
