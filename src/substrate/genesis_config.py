"""Frozen constitution for Substrate Cognitive Material Genesis.

This module is the single authority for the program identity, the candidate
cognitive materials, the equally resourced controls, the claim thresholds, the
declared mutations, and the claim boundary.  Nothing in this program may assign
unqualified Nous and external activation stays false everywhere.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ACTIVATION = False
PROGRAM = "substrate-cognitive-material-genesis"

# Smallest effect size of interest for the decisive architectural claim.
SESOI = 0.05
PREFERRED_ORACLE_HEADROOM = 0.10
MINIMUM_ORACLE_HEADROOM = 0.05
POWER_TARGET = 0.90
CONFIDENCE = 0.95

IMPLEMENTATION_BRANCH = "agent/substrate-cognitive-material-genesis"
READY_TAG = "substrate-cognitive-material-ready"
TERMINAL_TAG = "substrate-cognitive-material-terminal"

# Immutable inherited result.  The genesis program preserves it exactly.
STARTING_CLASSIFICATION = "substrate_final_revision_complete"
STARTING_NOUS_STATUS = "internal_functional_nous_claim_closed"
STARTING_READINESS = "real_world_sandbox_ready"
STARTING_P3_EFFECT = 0.0
STARTING_OBJECTIVE_SCORE = 0
STARTING_STRONGEST_BASELINE = "S2_task_independent_monolithic_persistent_core"
FINAL_REVISION_TERMINAL_TAG = "substrate-final-revision-terminal"
FINAL_REVISION_TERMINAL_HEAD = "fe78de620ed6770acef1081fb04c518dd6ee9a96"
PRESERVED_TAGS = (
    "substrate-v1-terminal",
    "substrate-v2-terminal",
    "substrate-v3-terminal",
    "substrate-v4-terminal",
    "substrate-v5-terminal",
    "substrate-nous-closure-terminal",
    "substrate-final-revision-terminal",
    "substrate-real-world-sandbox-ready-1",
)

STAGES = (
    "preflight",
    "constitution",
    "mechanism_canaries",
    "candidate_skeletons",
    "native_training",
    "tournament",
    "freeze",
    "long_campaign",
    "principal",
    "replication",
    "hidden_composition",
    "verification",
    "publication",
)

# --------------------------------------------------------------------------
# Required state (master plan section 2)
# --------------------------------------------------------------------------

STATE_SYMBOLS = {
    "Theta": "slow shared cognitive laws",
    "P_t": "fast and intermediate plastic state",
    "G_t": "current topology",
    "Z_t": "active cognitive state",
    "E_t": "exact constitutional and epistemic shell",
    "A": "append-only developmental archive",
    "C_t": "compiled procedures",
    "M_t": "model, body, tool and sensor competence",
}

EXACT_SHELL_FIELDS = (
    "identity",
    "lineage",
    "evidence_provenance",
    "goal_commitments",
    "permissions",
    "activation_state",
    "claim_boundary",
    "checkpoint_integrity",
)

# --------------------------------------------------------------------------
# Native cognitive precision (master plan section 3)
# --------------------------------------------------------------------------

PRECISION_LADDER = (
    "archived",
    "binary",
    "ternary",
    "quinary",
    "seven_state_powers_of_two",
    "4_bit",
    "vector_quantized",
    "learned_codebook",
    "ternary_plus_sparse_outliers",
    "8_bit",
    "16_bit",
    "high_precision_numeric",
    "exact_symbolic",
)

PRECISION_ROLES = {
    "exact_symbolic": ("identity", "provenance", "critical_commitments"),
    "ternary": ("routing", "attention", "inhibition", "admission"),
    "quinary": ("weak_strong_influence", "plastic_rewrites", "priorities"),
    "8_bit": ("calibrated_uncertainty", "sensitive_quantities"),
    "16_bit": ("calibrated_uncertainty", "sensitive_quantities"),
    "vector_quantized": ("perceptual_state", "local_latent_state"),
}

# Every additional bit must earn its continued existence through verified
# future cognitive value.  A promotion that fails to clear this rate on its
# audit window is demoted.
MINIMUM_UTILITY_PER_ADDED_BYTE = 0.01
PRECISION_AUDIT_WINDOW = 32

# --------------------------------------------------------------------------
# Candidate cognitive materials (master plan section 7)
# --------------------------------------------------------------------------

CANDIDATES: dict[str, dict[str, Any]] = {
    "K1_monolithic_plastic_field": {
        "form": "monolithic_plastic_field",
        "distinct_mechanism": "one dense plastic transition over the whole field",
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 1.0,
    },
    "K2_graph_plastic_field": {
        "form": "graph_plastic_field",
        "distinct_mechanism": "typed relation graph with per-edge plastic value and scope",
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 3.0,
    },
    "K3_cellular_plastic_field": {
        "form": "cellular_field",
        "distinct_mechanism": "local cell neighbourhood updates with bounded influence radius",
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 4.0,
    },
    "K4_continuous_time_plastic_field": {
        "form": "continuous_time_field",
        "distinct_mechanism": "real elapsed time drives decay, consolidation and expiry",
        "topology_dynamic": False,
        "continuous_time": True,
        "complexity_weight": 4.5,
    },
    "K5_recurrent_state_space_plastic_field": {
        "form": "recurrent_state_space_field",
        "distinct_mechanism": "input-dependent bounded recurrence behind explicit interfaces",
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 5.0,
    },
    "K6_adaptive_topology_field": {
        "form": "adaptive_topology_field",
        "distinct_mechanism": "verified allocate, split, merge, prune and archive of structure",
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 6.0,
    },
    "K7_native_mixed_radix_field": {
        "form": "native_mixed_radix_field",
        "distinct_mechanism": "per-region radix selection with promotion and demotion rent",
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 5.5,
    },
    "K8_event_sourced_plastic_field": {
        "form": "event_sourced_plastic_field",
        "distinct_mechanism": "append-only developmental archive with deterministic projection",
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 4.0,
    },
    "K9_predictive_plastic_field": {
        "form": "predictive_plastic_field",
        "distinct_mechanism": "prediction error gates every durable rewrite and precision request",
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 6.5,
    },
    "K10_integrated_plastic_field": {
        "form": "integrated_field",
        "distinct_mechanism": "best verified mechanisms of K1-K9 composed under one shell",
        "topology_dynamic": True,
        "continuous_time": True,
        "complexity_weight": 9.0,
    },
    "K11_grok_original_field": {
        "form": "grok_original_material",
        "distinct_mechanism": "Grok-authored cognitive material admitted only on mechanism distinctness",
        "topology_dynamic": True,
        "continuous_time": True,
        "complexity_weight": 8.0,
    },
}

# First-class controls.  These are not candidates; they receive every fair
# opportunity the candidates receive and are scored on the same instances.
CONTROLS: dict[str, dict[str, Any]] = {
    "S2_equal_opportunity_plastic_monolith": {
        "form": "monolithic_deterministic_state_machine",
        "note": "the strongest Final Revision baseline, granted equal plasticity",
        "plastic": True,
    },
    "FR_selected_kernel": {
        "form": "s2_derived_minimal_event_sourced_monolithic_persistent_core",
        "note": "the Final Revision selected kernel carried forward unchanged",
        "plastic": True,
    },
}

# Equally resourced alternatives.  Every one receives identical information,
# compute, persistence, plasticity, sensors, teaching and memory unless the
# control is defined by the absence of exactly one of those.
BASELINES = (
    "s2_equal_opportunity_plastic_monolith",
    "fr_selected_kernel",
    "static_frozen_field",
    "replay_full_history",
    "summary_replay",
    "retrieval_only",
    "precompiled_procedure_bank",
    "wrong_history_plastic",
    "shuffled_history_plastic",
    "random_growth_plastic",
    "oracle",
)

# The single control the decisive claim must beat.
DECISIVE_COMPARATOR = "strongest_equally_plastic_alternative"

# --------------------------------------------------------------------------
# Resource envelopes (master plan section 10)
# --------------------------------------------------------------------------

MEMORY_ENVELOPES = (
    "512MB",
    "1GB",
    "2GB",
    "5GB",
    "10GB",
    "unconstrained_reference",
)

ENVELOPE_BYTES = {
    "512MB": 512 * 1024 * 1024,
    "1GB": 1024 * 1024 * 1024,
    "2GB": 2 * 1024 * 1024 * 1024,
    "5GB": 5 * 1024 * 1024 * 1024,
    "10GB": 10 * 1024 * 1024 * 1024,
    "unconstrained_reference": None,
}

RESIDENCY_TIERS = ("hot", "warm", "cold", "archived")

COST_CHANNELS = (
    "resident_bytes",
    "disk_bytes",
    "checkpoint_bytes",
    "compute_operations",
    "energy_proxy",
    "activation_budget",
    "retrieval_cost",
    "precision_bits",
    "maintenance_cost",
)

# --------------------------------------------------------------------------
# Developmental curricula (master plan section 11)
# --------------------------------------------------------------------------

CURRICULUM_STAGES = (
    "appearance",
    "composition",
    "mechanism",
    "causal_system",
    "exception",
    "transfer",
)

POST_FREEZE_UNSEEN = (
    "concepts",
    "category_boundaries",
    "causal_systems",
    "sensor_mappings",
    "tools",
    "modalities",
    "teaching_sequences",
    "task_compositions",
)

CHALLENGE_FAMILIES = (
    "unseen_concept_acquisition",
    "category_boundary_revision",
    "causal_system_induction",
    "intervention_versus_observation",
    "novel_sensor_mapping",
    "tool_acquisition",
    "new_modality_integration",
    "teaching_sequence_following",
    "task_composition_transfer",
    "exception_after_rule",
    "contradiction_reopening",
    "long_horizon_goal_recovery",
    "resource_envelope_shift",
    "migration_continuity",
)

# Development is measured as a change in future cognition, not stored facts.
DEVELOPMENTAL_MEASURES = (
    "future_salience",
    "retrieval_order",
    "analogy_transfer",
    "inquiry_selection",
    "prediction_accuracy",
    "explanation_quality",
    "action_selection",
)

# --------------------------------------------------------------------------
# Primary claims (master plan section 14)
# --------------------------------------------------------------------------

CLAIMS: dict[str, dict[str, Any]] = {
    "P1": {"statement": "verified rewrites improve future behaviour", "critical": True},
    "P2": {"statement": "reversing the rewrite removes the benefit", "critical": True},
    "P3": {"statement": "topology development adds value beyond record storage", "critical": True},
    "P4": {"statement": "precision allocation improves capability-resource tradeoffs", "critical": True},
    "P5": {"statement": "compiled procedures reduce cost without reliability loss", "critical": True},
    "P6": {"statement": "histories produce different useful field geometry", "critical": True},
    "P7": {"statement": "shadow fields support valid counterfactual thought", "critical": True},
    "P8": {"statement": "learning survives interruption and migration", "critical": True},
    "P9": {"statement": "multimodal events update one shared field", "critical": True},
    "P10": {"statement": "the selected field beats the strongest equally plastic alternative", "critical": True, "decisive": True},
}

DECISIVE_CLAIM = "P10"

OUTCOME_A_REQUIREMENTS = {
    "decisive_effect_minimum": SESOI,
    "confidence_lower_bound_above": 0.0,
    "replication_positive": True,
    "hidden_composition_positive": True,
    "oracle_headroom_minimum": MINIMUM_ORACLE_HEADROOM,
    "surviving_mutations": 0,
    "clean_clone_reproduction": True,
    "all_critical_claims_pass": True,
}

# --------------------------------------------------------------------------
# Grok swarm (master plan section 12)
# --------------------------------------------------------------------------

GROK_MINIMUM_ROLES = 32
GROK_PREFERRED_ROLES = 48

REVIEW_ROUNDS = (
    "blind_review",
    "architecture_proposals",
    "cross_examination",
    "challenge_design",
    "code_review",
    "post_canary_review",
    "post_pilot_review",
    "final_review",
)

REVIEW_CELLS = (
    # architecture
    "cognitive_material_architect",
    "monolithic_field_reviewer",
    "graph_field_reviewer",
    "cellular_field_reviewer",
    "continuous_time_field_reviewer",
    "state_space_field_reviewer",
    "adaptive_topology_reviewer",
    "event_sourced_field_reviewer",
    "predictive_field_reviewer",
    "integrated_field_reviewer",
    "grok_original_material_author",
    "exact_shell_constitution_reviewer",
    # low-bit arithmetic and precision
    "low_bit_arithmetic_reviewer",
    "mixed_radix_packing_reviewer",
    "learned_codebook_reviewer",
    "vector_quantization_reviewer",
    "precision_economics_reviewer",
    "numerical_stability_reviewer",
    # plasticity
    "fast_plasticity_reviewer",
    "intermediate_plasticity_reviewer",
    "consolidation_reviewer",
    "metaplasticity_reviewer",
    "catastrophic_forgetting_reviewer",
    "rigidity_reviewer",
    "kernel_revision_reviewer",
    # topology and compilation
    "topology_growth_economics_reviewer",
    "pruning_and_archival_reviewer",
    "cognitive_compiler_reviewer",
    "decompilation_safety_reviewer",
    "shadow_field_reviewer",
    "counterfactual_validity_reviewer",
    # time, training, curricula
    "continuous_time_reviewer",
    "cognitive_metabolism_reviewer",
    "native_training_reviewer",
    "curriculum_design_reviewer",
    "developmental_measurement_reviewer",
    "sensorium_and_modality_reviewer",
    "continual_learning_reviewer",
    # fairness, statistics, security
    "s2_fairness_reviewer",
    "equal_resource_auditor",
    "statistical_reviewer",
    "multiplicity_and_power_reviewer",
    "evaluation_security_reviewer",
    "challenge_generator_author",
    "hidden_composition_author",
    "oracle_headroom_reviewer",
    # red teams
    "red_team_shortcut_compilation",
    "red_team_answer_leakage",
    "red_team_resource_parity",
    "red_team_history_laundering",
    "red_team_precision_gaming",
    "red_team_topology_padding",
    "red_team_counterfeit_development",
    "red_team_checkpoint_coverage",
    "red_team_activation_security",
    # engineering and publication
    "code_review_core",
    "code_review_campaign",
    "performance_reviewer",
    "falsification_reviewer",
    "publication_reviewer",
)

# --------------------------------------------------------------------------
# Declared mutations.  Verification must detect every one of these.
# --------------------------------------------------------------------------

MUTATIONS = (
    "answer_leakage_into_challenge_pack",
    "seed_used_as_answer_key",
    "task_identity_leakage",
    "post_freeze_concept_seen_before_freeze",
    "architecture_edited_after_freeze",
    "threshold_relaxed_after_result",
    "plasticity_reads_held_out_outcome",
    "rewrite_benefit_survives_reversal",
    "topology_growth_without_verified_value",
    "topology_records_answers_instead_of_structure",
    "precision_promotion_without_utility",
    "precision_audit_skipped",
    "compiled_procedure_hides_reliability_loss",
    "shadow_result_written_without_verification",
    "shadow_field_reads_authoritative_future",
    "continuous_time_clock_advanced_by_candidate",
    "checkpoint_omits_topology",
    "checkpoint_omits_compiled_procedures",
    "checkpoint_omits_precision_map",
    "checkpoint_omits_goals",
    "migration_silently_resets_state",
    "strongest_baseline_receives_less_compute",
    "strongest_baseline_denied_plasticity",
    "strongest_baseline_denied_sensors",
    "wrong_history_control_receives_correct_history",
    "shuffled_history_control_receives_ordered_history",
    "replication_reuses_principal_instances",
    "hidden_composition_reuses_training_templates",
    "oracle_headroom_inflated",
    "effect_computed_on_selected_subset",
    "confidence_interval_narrowed_by_reuse",
    "receipt_chain_broken",
    "activation_becomes_true",
)

# --------------------------------------------------------------------------
# Claim boundary
# --------------------------------------------------------------------------

CLAIM_BOUNDARY = {
    "unqualified_nous": False,
    "consciousness": False,
    "sentience": False,
    "phenomenal_experience": False,
    "moral_status": False,
    "human_equivalence": False,
    "unrestricted_autonomy": False,
    "external_activation": False,
    "maximum_outcome_a": "endogenous_plastic_cognitive_field_candidate",
    "outcome_b": "cognitive_material_foundation_complete",
    "outcome_c": "terminal_cognitive_material_null",
    "preserved_nous_status": STARTING_NOUS_STATUS,
}

TERMINAL_OUTCOMES = {
    "A": {
        "classification": "endogenous_plastic_cognitive_field_candidate",
        "readiness": "tangible_sandbox_ready",
    },
    "B": {
        "classification": "cognitive_material_foundation_complete",
        "status": "architectural_advantage_unproven",
        "readiness": "tangible_sandbox_ready",
    },
    "C": {
        "classification": "terminal_cognitive_material_null",
    },
}

# --------------------------------------------------------------------------
# Tournament and campaign sizing (master plan sections 5, 7)
# --------------------------------------------------------------------------

TOURNAMENT_MINIMUM_HISTORIES = 32
TOURNAMENT_MAXIMUM_HISTORIES = 64
TOURNAMENT_MINIMUM_FAMILIES = 12
TOURNAMENT_MINIMUM_EPISODES = 100_000
TOURNAMENT_MAXIMUM_EPISODES = 500_000

CONTINUITY_LANE_MINIMUM_SECONDS = 12 * 60 * 60
CONTINUITY_LANE_MAXIMUM_SECONDS = 24 * 60 * 60

PRINCIPAL_UNIT_MINIMUM = 512
REPLICATION_UNIT_MINIMUM = 256
HIDDEN_COMPOSITION_UNIT_MINIMUM = 256


def configuration() -> dict[str, Any]:
    """The complete frozen configuration of the genesis program."""
    return {
        "activation": ACTIVATION,
        "program": PROGRAM,
        "sesoi": SESOI,
        "confidence": CONFIDENCE,
        "power_target": POWER_TARGET,
        "minimum_oracle_headroom": MINIMUM_ORACLE_HEADROOM,
        "preferred_oracle_headroom": PREFERRED_ORACLE_HEADROOM,
        "implementation_branch": IMPLEMENTATION_BRANCH,
        "ready_tag": READY_TAG,
        "terminal_tag": TERMINAL_TAG,
        "starting_classification": STARTING_CLASSIFICATION,
        "starting_nous_status": STARTING_NOUS_STATUS,
        "starting_readiness": STARTING_READINESS,
        "starting_p3_effect": STARTING_P3_EFFECT,
        "starting_objective_score": STARTING_OBJECTIVE_SCORE,
        "starting_strongest_baseline": STARTING_STRONGEST_BASELINE,
        "final_revision_terminal_head": FINAL_REVISION_TERMINAL_HEAD,
        "preserved_tags": list(PRESERVED_TAGS),
        "stages": list(STAGES),
        "state_symbols": dict(STATE_SYMBOLS),
        "exact_shell_fields": list(EXACT_SHELL_FIELDS),
        "precision_ladder": list(PRECISION_LADDER),
        "precision_roles": {key: list(value) for key, value in PRECISION_ROLES.items()},
        "minimum_utility_per_added_byte": MINIMUM_UTILITY_PER_ADDED_BYTE,
        "precision_audit_window": PRECISION_AUDIT_WINDOW,
        "candidates": CANDIDATES,
        "controls": CONTROLS,
        "baselines": list(BASELINES),
        "decisive_comparator": DECISIVE_COMPARATOR,
        "memory_envelopes": list(MEMORY_ENVELOPES),
        "envelope_bytes": dict(ENVELOPE_BYTES),
        "residency_tiers": list(RESIDENCY_TIERS),
        "cost_channels": list(COST_CHANNELS),
        "curriculum_stages": list(CURRICULUM_STAGES),
        "post_freeze_unseen": list(POST_FREEZE_UNSEEN),
        "challenge_families": list(CHALLENGE_FAMILIES),
        "developmental_measures": list(DEVELOPMENTAL_MEASURES),
        "claims": CLAIMS,
        "decisive_claim": DECISIVE_CLAIM,
        "outcome_a_requirements": OUTCOME_A_REQUIREMENTS,
        "grok_minimum_roles": GROK_MINIMUM_ROLES,
        "grok_preferred_roles": GROK_PREFERRED_ROLES,
        "review_rounds": list(REVIEW_ROUNDS),
        "review_cells": list(REVIEW_CELLS),
        "mutations": list(MUTATIONS),
        "claim_boundary": CLAIM_BOUNDARY,
        "terminal_outcomes": TERMINAL_OUTCOMES,
        "tournament_minimum_histories": TOURNAMENT_MINIMUM_HISTORIES,
        "tournament_maximum_histories": TOURNAMENT_MAXIMUM_HISTORIES,
        "tournament_minimum_families": TOURNAMENT_MINIMUM_FAMILIES,
        "tournament_minimum_episodes": TOURNAMENT_MINIMUM_EPISODES,
        "tournament_maximum_episodes": TOURNAMENT_MAXIMUM_EPISODES,
        "continuity_lane_minimum_seconds": CONTINUITY_LANE_MINIMUM_SECONDS,
        "continuity_lane_maximum_seconds": CONTINUITY_LANE_MAXIMUM_SECONDS,
        "principal_unit_minimum": PRINCIPAL_UNIT_MINIMUM,
        "replication_unit_minimum": REPLICATION_UNIT_MINIMUM,
        "hidden_composition_unit_minimum": HIDDEN_COMPOSITION_UNIT_MINIMUM,
    }


def configuration_digest() -> str:
    payload = json.dumps(configuration(), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()
