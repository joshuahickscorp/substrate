"""Frozen scientific configuration for the Substrate v3 Nous constitutional ascent."""

from __future__ import annotations

from substrate import v3io as io

SESOI = 0.05
PREFERRED_INQUIRY_HEADROOM = 0.08
COMPUTE_PRICE = 0.04
LATENCY_PRICE = 0.01
UNNECESSARY_INQUIRY_PENALTY = 0.04
MISSED_INQUIRY_PENALTY = 0.14
CATASTROPHIC_ERROR_PENALTY = 0.50
BROKEN_GOAL_PENALTY = 0.30
INTERFERENCE_PENALTY = 0.25

SPLITS = {
    "construction": tuple(range(0, 12)),
    "cheap_admission": tuple(range(100, 116)),
    "moderate_pilot": tuple(range(500, 524)),
    "principal": tuple(range(1000, 1048)),
    "replication": tuple(range(2000, 2012)),
    "open_world_review": tuple(range(3000, 3012)),
}

WORKLOADS = {
    "ontology_garden": {
        "features": ["concept formation", "category revision", "identity through change", "relations", "exceptions", "split", "merge"],
        "latent_state": "feature defined categories with hidden exception regimes",
        "actions": ["form category", "split", "merge", "map representation", "preserve unknown"],
        "positive_transfer": ["cross_representation_systems"],
        "negative_transfer": ["adversarial_ambiguity"],
    },
    "epistemic_laboratory": {
        "features": ["source reliability", "warrant", "defeaters", "underdetermination", "knowledge admission", "inquiry"],
        "latent_state": "source conditionally reliable by regime",
        "actions": ["accept", "defer", "test", "seek counterexample", "preserve alternatives"],
        "positive_transfer": ["scientific_inquiry"],
        "negative_transfer": [],
    },
    "causal_micro_worlds": {
        "features": ["observation", "intervention", "confounding", "direction", "counterfactual"],
        "latent_state": "directed causal transition system",
        "actions": ["observe", "intervene", "simulate", "counterfactual"],
        "positive_transfer": ["cross_representation_systems"],
        "negative_transfer": ["adversarial_ambiguity"],
    },
    "cross_representation_systems": {
        "features": ["symbolic", "sequence", "graph", "structured statements", "tool state"],
        "latent_state": "one directed dependency system under five surface encodings",
        "actions": ["map concept", "predict", "explain", "reconstruct", "detect boundary"],
        "positive_transfer": ["reasoning_method_selection"],
        "negative_transfer": [],
    },
    "reasoning_method_selection": {
        "features": ["deduction", "induction", "abduction", "analogy", "diagnosis", "planning"],
        "latent_state": "task family determines valid inference operation",
        "actions": ["select reasoning mode", "execute trace", "verify"],
        "positive_transfer": ["scientific_inquiry"],
        "negative_transfer": ["adversarial_ambiguity"],
    },
    "scientific_inquiry": {
        "features": ["variable evidence reliability", "variable cost", "contradiction", "risk", "domain shift"],
        "latent_state": "context determines which evidence action has positive net value",
        "actions": ["inspect source", "test hypothesis", "simulate", "defer", "stop"],
        "positive_transfer": ["open_world_review"],
        "negative_transfer": ["no_headroom_control"],
    },
    "adversarial_ambiguity": {
        "features": ["surface trap", "conflicting evidence", "false explanation", "wrong analogy", "underpowered evidence"],
        "latent_state": "surface cues are anticorrelated with relational truth",
        "actions": ["reject surface cue", "preserve uncertainty", "seek discriminating evidence"],
        "positive_transfer": [],
        "negative_transfer": [],
    },
}

REASONING_MODES = (
    "deduction", "induction", "abduction", "analogy", "causal", "counterfactual",
    "temporal", "planning", "diagnostic", "dialectical", "symbolic", "meta",
)

CORE_ARMS = (
    "full_v3",
    "v2_developmental_control",
    "fixed_ontology",
    "confidence_only_epistemology",
    "fixed_reasoning",
    "no_understanding_structure",
    "simple_inquiry",
    "no_self_model",
    "no_world_model",
    "more_compute",
    "fresh_reset",
    "transcript_replay",
)

PHASES = (
    "phase_0_cold_baseline",
    "phase_1_ontology_acquisition",
    "phase_2_semantic_procedural_development",
    "phase_3_conflicting_evidence_defeaters",
    "phase_4_ontology_repair",
    "phase_5_cross_representation_transfer",
    "phase_6_causal_intervention",
    "phase_7_counterfactual",
    "phase_8_reasoning_method_switching",
    "phase_9_inquiry_under_cost",
    "phase_10_interruption_checkpoint",
    "phase_11_body_tool_change",
    "phase_12_return_prior_domains",
    "phase_13_misleading_analogy",
    "phase_14_open_world_structures",
    "phase_15_terminal_integrated_evaluation",
)

HYPOTHESES = {
    "H_N1": "active ontology revision improves held out prediction, explanation, or transfer over fixed ontology",
    "H_N2": "defeaters and underdetermination improve calibrated belief revision and inquiry over confidence only",
    "H_N3": "reasoning method selection improves utility over fixed reasoning and maximum compute",
    "H_N4": "structural understanding improves explanation, intervention, counterfactuals, and cross representation transfer",
    "H_N5": "inquiry driven allocation improves held out cost adjusted utility over the strongest simple policy",
    "H_N6": "v2 developmental transfer and retention remain preserved",
    "H_N7": "different verified epistemic histories produce useful predictable specialization",
    "H_N8": "conditional self and world models improve held out control",
    "H_N9": "one identity persists through ontology change, interruption, body change, and conflict",
}

CLAIM_BOUNDARY = {
    "ordered_levels": [
        "certified_cognitive_scaffold",
        "persistent_developmental_cognition",
        "epistemically_organized_reasoner",
        "demonstrated_structural_understanding",
        "reflective_cognitive_organization",
        "functional_proto_nous_candidate",
        "nous_ready_for_review",
    ],
    "maximum": "nous_ready_for_review",
    "unqualified_nous_automatically_assignable": False,
    "not_claimed": [
        "consciousness", "phenomenal experience", "sentience", "feeling", "suffering",
        "desire", "personhood", "life", "moral status",
    ],
    "permitted_actions": [
        "internal cognitive proposals",
        "internal reasoning selection",
        "deterministic sandboxed tools",
        "local immutable task queries",
        "hypothetical actions",
        "controlled simulated outcomes",
    ],
    "activation": False,
}

CANDIDATE_LADDER = {
    "ontology": ["fixed typed ontology", "evidence triggered split and merge", "regularized utility based revision"],
    "epistemology": ["dependency graph with direct defeaters", "source reliability", "underdetermination and inquiry"],
    "reasoning": ["fixed reasoning", "rule based contextual selector", "regularized linear selector"],
    "understanding": ["semantic structure", "semantic plus causal", "semantic plus causal plus cross representation"],
    "inquiry": ["best fixed", "tabular contextual value", "regularized linear contextual"],
}

STATISTICS = {
    "independent_unit": "one frozen developmental history",
    "primary_summary": ["mean paired effect", "median paired effect", "95 percent bootstrap confidence interval"],
    "exact_test": "paired exact sign test",
    "multiple_comparisons": "Holm correction over H_N1 through H_N9",
    "alpha": 0.05,
    "sesoi": SESOI,
    "target_power": 0.9,
    "retry_rule": "one exact retry after process failure; deterministic failure is terminal",
    "missing_data": "retain and report failed units; no silent deletion",
}

DELIVERABLES = (
    "SUBSTRATE_V3_PREFLIGHT.json",
    "SUBSTRATE_V3_V1_V2_IMMUTABILITY.json",
    "SUBSTRATE_V3_HAWKING_COEXISTENCE.json",
    "SUBSTRATE_V3_CONSTITUTIONAL_RETROSPECTIVE.json",
    "SUBSTRATE_V3_CAPABILITY_MATRIX.json",
    "SUBSTRATE_V3_EVIDENCE_GAP_MAP.json",
    "SUBSTRATE_V3_CLASSIFICATION_AUTHORITY.json",
    "SUBSTRATE_V3_CLAIM_BOUNDARY.json",
    "SUBSTRATE_V3_ONTOLOGY_SCHEMA.json",
    "SUBSTRATE_V3_ONTOLOGY_REVISION.json",
    "SUBSTRATE_V3_ONTOLOGY_CANARIES.json",
    "SUBSTRATE_V3_ONTOLOGY_CONTROL_REPORT.json",
    "SUBSTRATE_V3_EPISTEMIC_SCHEMA.json",
    "SUBSTRATE_V3_DEFEATER_SYSTEM.json",
    "SUBSTRATE_V3_KNOWLEDGE_ADMISSION.json",
    "SUBSTRATE_V3_INQUIRY_SYSTEM.json",
    "SUBSTRATE_V3_EPISTEMIC_CANARIES.json",
    "SUBSTRATE_V3_REASONING_CATALOG.json",
    "SUBSTRATE_V3_REASONING_PROCEDURES.json",
    "SUBSTRATE_V3_REASONING_SELECTION.json",
    "SUBSTRATE_V3_REASONING_CANARIES.json",
    "SUBSTRATE_V3_UNDERSTANDING_SCHEMA.json",
    "SUBSTRATE_V3_CROSS_REPRESENTATION_BED.json",
    "SUBSTRATE_V3_EXPLANATION_BATTERY.json",
    "SUBSTRATE_V3_COUNTERFACTUAL_BATTERY.json",
    "SUBSTRATE_V3_UNDERSTANDING_CANARIES.json",
    "SUBSTRATE_V3_INQUIRY_WORKLOADS.json",
    "SUBSTRATE_V3_ALLOCATION_HEADROOM.json",
    "SUBSTRATE_V3_ALLOCATION_POLICY.json",
    "SUBSTRATE_V3_ALLOCATION_TRANSFER.json",
    "SUBSTRATE_V3_ALLOCATION_CANARIES.json",
    "SUBSTRATE_V3_WORLD_MODEL.json",
    "SUBSTRATE_V3_WORLD_MODEL_CANARIES.json",
    "SUBSTRATE_V3_SELF_MODEL.json",
    "SUBSTRATE_V3_SELF_MODEL_CANARIES.json",
    "SUBSTRATE_V3_MODEL_CONTROL_VALUE.json",
    "SUBSTRATE_V3_INTEGRATED_RUNTIME.json",
    "SUBSTRATE_V3_CHECKPOINT_SCHEMA.json",
    "SUBSTRATE_V3_RUNTIME_ACTIVITY.json",
    "SUBSTRATE_V3_INTEGRATION_CANARIES.json",
    "SUBSTRATE_V3_WORKLOAD_CATALOG.json",
    "SUBSTRATE_V3_TRANSFER_GRAPH.json",
    "SUBSTRATE_V3_GENERATOR_AUTHORITY.json",
    "SUBSTRATE_V3_SPLIT_AUTHORITY.json",
    "SUBSTRATE_V3_BED_SCREEN.json",
    "SUBSTRATE_V3_CHEAP_CANARIES.json",
    "SUBSTRATE_V3_CANARY_LEDGER.json",
    "SUBSTRATE_V3_CANDIDATE_LADDER.json",
    "SUBSTRATE_V3_SELECTION_RECEIPT.json",
    "SUBSTRATE_V3_MODERATE_PILOT.json",
    "SUBSTRATE_V3_FAILURE_MATRIX.json",
    "SUBSTRATE_V3_RESOURCE_PILOT.json",
    "SUBSTRATE_V3_ADMISSION.json",
    "SUBSTRATE_V3_HYPOTHESIS_GRAPH.json",
    "SUBSTRATE_V3_SCIENTIFIC_CONSTITUTION.json",
    "SUBSTRATE_V3_STATISTICAL_AUTHORITY.json",
    "SUBSTRATE_V3_PRINCIPAL_AUTHORITY.json",
    "SUBSTRATE_V3_PRINCIPAL_DAG.json",
    "SUBSTRATE_V3_RESOURCE_PLAN.json",
    "SUBSTRATE_V3_STOP_AND_FUTILITY.json",
    "SUBSTRATE_V3_RESOURCE_BENCHMARK.json",
    "SUBSTRATE_V3_WORKER_AUTHORITY.json",
    "SUBSTRATE_V3_INDEPENDENT_VERIFICATION.json",
    "SUBSTRATE_V3_MUTATION_REPORT.json",
    "SUBSTRATE_V3_CLEAN_CLONE.json",
    "SUBSTRATE_V3_FINAL_CLASSIFICATION.json",
    "SUBSTRATE_V3_NOUS_REVIEW_AUTHORITY.json",
    "SUBSTRATE_V3_FINAL_STATE.json",
)


def configuration() -> dict:
    body = {
        "sesoi": SESOI,
        "preferred_inquiry_headroom": PREFERRED_INQUIRY_HEADROOM,
        "prices": {
            "compute": COMPUTE_PRICE,
            "latency": LATENCY_PRICE,
            "unnecessary_inquiry": UNNECESSARY_INQUIRY_PENALTY,
            "missed_inquiry": MISSED_INQUIRY_PENALTY,
            "catastrophic_error": CATASTROPHIC_ERROR_PENALTY,
            "broken_goal": BROKEN_GOAL_PENALTY,
            "interference": INTERFERENCE_PENALTY,
        },
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "workloads": WORKLOADS,
        "reasoning_modes": list(REASONING_MODES),
        "arms": list(CORE_ARMS),
        "phases": list(PHASES),
        "hypotheses": HYPOTHESES,
        "claim_boundary": CLAIM_BOUNDARY,
        "candidate_ladder": CANDIDATE_LADDER,
        "statistics": STATISTICS,
        "activation": False,
    }
    body["configuration_digest"] = io.sha_obj(body)
    return body
