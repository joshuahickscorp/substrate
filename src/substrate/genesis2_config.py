"""Frozen constitution for Substrate Cognitive Material Genesis II.

Genesis I established that individual plastic mechanisms carry real value and
that the integrated material nonetheless lost to a cheap exact associative
monolith by roughly 0.248 on the decisive claim.  This program does not rerun
that tournament at larger scale.  It separates the four questions Genesis I
conflated -- representation, update granularity, architecture and composition --
and then finishes the material or closes it as a terminal null.

Nothing in this program may assign unqualified Nous.  External activation stays
false everywhere.  Every Genesis I artifact, null, limitation and receipt is
preserved unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ACTIVATION = False
PROGRAM = "substrate-cognitive-material-genesis-ii"

# --------------------------------------------------------------------------
# Inherited terminal state.  Read from the Genesis I artifacts, not asserted.
# --------------------------------------------------------------------------

PARENT_PROGRAM = "substrate-cognitive-material-genesis"
PARENT_TERMINAL_TAG = "substrate-cognitive-material-terminal"
PARENT_CLASSIFICATION = "cognitive_material_foundation_complete"
PARENT_STATUS = "architectural_advantage_unproven"
PARENT_DECISIVE_EFFECT = -0.24776785714285712
PARENT_DECISIVE_CI = (-0.2567368847908243, -0.23839285714285713)
PARENT_SELECTED_CANDIDATE = "K8_event_sourced_plastic_field"
PARENT_STRONGEST_COMPARATOR = "S2_task_independent_monolithic_persistent_core"
PARENT_FAILING_CLAIMS = ("P4", "P10")
PARENT_INERT_MECHANISMS_OF_NINE = 7

PRESERVED_TAGS = (
    "substrate-v1-terminal",
    "substrate-v2-terminal",
    "substrate-v3-terminal",
    "substrate-v4-terminal",
    "substrate-v5-terminal",
    "substrate-nous-closure-terminal",
    "substrate-final-revision-terminal",
    "substrate-real-world-sandbox-ready-1",
    "substrate-cognitive-material-ready",
    "substrate-cognitive-material-terminal",
)

# --------------------------------------------------------------------------
# Decision thresholds.  Frozen before any principal instance exists.
# --------------------------------------------------------------------------

SESOI = 0.05
MINIMUM_ORACLE_HEADROOM = 0.05
PREFERRED_ORACLE_HEADROOM = 0.10
POWER_TARGET = 0.90
CONFIDENCE = 0.95

IMPLEMENTATION_BRANCH = "agent/substrate-cognitive-material-genesis-ii"
READY_TAG = "substrate-cognitive-material-ii-ready"
TERMINAL_TAG = "substrate-cognitive-material-ii-terminal"

STAGES = (
    "preflight",
    "constitution",
    "reconstruction",
    "factorial",
    "canaries",
    "envelope_calibration",
    "mechanism_matrix",
    "pilot",
    "freeze",
    "principal",
    "replication",
    "hidden_composition",
    "continuity",
    "mutations",
    "clean_clone",
    "publication",
)

# --------------------------------------------------------------------------
# The four questions Genesis I conflated (master plan section 1)
# --------------------------------------------------------------------------

DIAGNOSTIC_QUESTIONS = {
    "representation": "did S2 win because exact associative storage beats low-bit representation",
    "update_granularity": "did S2 win because it attempts many cheap writes while the field rewrites coarsely",
    "architecture": "does a plastic field add value once exact association and fair write granularity are available to both",
    "composition": "can the mechanisms cooperate causally rather than coexist decoratively",
}

# --------------------------------------------------------------------------
# Three scales of learning (master plan section 3)
# --------------------------------------------------------------------------

UPDATE_GRANULARITIES = (
    "micro_association",
    "local_low_bit_adjustment",
    "association_promotion",
    "structural_consolidation",
    "topology_revision",
)

#: Relative cost of each granularity, in the common ledger's units.  A material
#: that spends topology on a problem a micro-write solves is paying 64x for the
#: same information.  These weights are frozen; the allocator is scored against
#: them, never allowed to redefine them.
GRANULARITY_COST_WEIGHT = {
    "micro_association": 1.0,
    "local_low_bit_adjustment": 1.0,
    "association_promotion": 4.0,
    "structural_consolidation": 16.0,
    "topology_revision": 64.0,
}

MICRO_ASSOCIATION_KINDS = (
    "entity_value",
    "cue_outcome",
    "source_reliability",
    "object_identity",
    "local_prediction_error",
    "routing_preference",
)

MESO_STRUCTURE_KINDS = (
    "concept_relation",
    "causal_pathway",
    "procedure",
    "object_event_structure",
    "model_competence",
    "goal_dependency",
    "cross_modal_binding",
)

MACRO_TOPOLOGY_OPERATIONS = (
    "allocate_region",
    "split_concept",
    "merge_redundancy",
    "compile_procedure_family",
    "create_specialist_pathway",
    "prune_obsolete_organization",
)

# --------------------------------------------------------------------------
# Required layers (master plan section 4)
# --------------------------------------------------------------------------

REQUIRED_LAYERS = (
    "exact_constitutional_shell",
    "associative_microstore",
    "plastic_field",
    "structural_consolidator",
    "shadow_field",
    "cognitive_compiler",
    "developmental_archive",
    "model_and_body_fabric",
)

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

MICROSTORE_REQUIREMENTS = (
    "exact_and_low_bit_modes",
    "bounded_collision_detection",
    "provenance_pointers",
    "scope",
    "expiry",
    "confidence",
    "write_cost",
    "rollback",
)

# --------------------------------------------------------------------------
# Representation-architecture factorial (master plan section 5)
# --------------------------------------------------------------------------

FACTORIAL_CELLS: dict[str, dict[str, Any]] = {
    "A": {
        "arm": "S2_task_independent_monolithic_persistent_core",
        "representation": "exact",
        "organization": "monolithic",
        "reads": "the inherited strongest baseline, unchanged",
    },
    "B": {
        "arm": "S2_low_bit_constrained",
        "representation": "low_bit",
        "organization": "monolithic",
        "reads": "S2 with its associative values held to the candidate's radix",
    },
    "C": {
        "arm": "L0_prior_selected_field",
        "representation": "low_bit",
        "organization": "field",
        "reads": "the Genesis I selected field with its original representation",
    },
    "D": {
        "arm": "L2_associative_monolithic_plastic_field",
        "representation": "exact",
        "organization": "field",
        "reads": "the field granted an exact associative microstore",
    },
    "E": {
        "arm": "L7_exact_microstore_mixed_radix_field",
        "representation": "mixed",
        "organization": "field",
        "reads": "exact microstore with a low-bit structural layer above it",
    },
    "F": {
        "arm": "L1_associative_monolith",
        "representation": "exact",
        "organization": "monolithic_hybrid",
        "reads": "an equally plastic monolithic hybrid; allowed to win",
    },
}

FACTORIAL_INTERPRETATION = {
    "low_bit_s2_collapses": "representation explains substantial advantage",
    "exact_field_closes_gap": "write representation and granularity explain substantial advantage",
    "hybrid_monolith_still_wins": "field organization remains unsupported",
    "field_wins_only_after_consolidation": "structural compression is the missing value",
    "all_tie": "prefer the simplest material",
}

FACTORIAL_EQUAL_CHANNELS = (
    "observations",
    "history",
    "write_opportunities",
    "compute",
    "memory_accounting",
    "tool_access",
    "teaching",
)

# --------------------------------------------------------------------------
# Update economy (master plan section 6)
# --------------------------------------------------------------------------

LEDGER_FIELDS = (
    "proposal",
    "bytes_read",
    "bytes_written",
    "information_introduced",
    "compute",
    "latency",
    "scope",
    "durability",
    "rollback_cost",
    "future_utility",
)

LEDGER_REPORTS = (
    "attempt_count",
    "committed_count",
    "useful_commits",
    "utility_per_commit",
    "utility_per_written_byte",
    "utility_per_compute_unit",
    "utility_per_wall_time",
)

ALLOCATOR_COMPARATORS = (
    "always_micro",
    "always_structural",
    "fixed_thresholds",
    "s2_policy",
    "oracle_granularity",
)

#: A mechanism must not use topology revision where a one-byte association is
#: sufficient.  The audit fails a material whose topology operations exceed this
#: share of its committed updates on workloads a microstore already solves.
TOPOLOGY_MISALLOCATION_CEILING = 0.10

# --------------------------------------------------------------------------
# Compositional activation (master plan section 7)
# --------------------------------------------------------------------------

REQUIRED_MECHANISMS = (
    "micro_association",
    "plastic_relation_update",
    "precision_change",
    "topology_change",
    "shadow_field",
    "procedure_compilation",
    "self_model_allocation",
    "world_model_update",
    "memory_consolidation",
)

MECHANISM_DECLARATION_FIELDS = (
    "input_conditions",
    "activation_condition",
    "state_changed",
    "downstream_consumer",
    "cost",
    "positive_fixture",
    "null_fixture",
    "ablation",
    "interaction_partners",
)

MECHANISM_ADMISSION = {
    "standalone_activity": "required",
    "known_positive_causal_effect": "required",
    "integrated_activation_frequency": "nonzero_where_the_workload_calls_for_it",
    "integrated_ablation_effect": "nonzero_or_precisely_licensed_as_unnecessary",
}

#: A mechanism whose integrated ablation moves the score by less than this and
#: which has no licence is decorative.  Decorative machinery is removed or
#: disabled, never carried for appearance.
DECORATIVE_ABLATION_FLOOR = 0.005

SCHEDULER_COMPARATORS = (
    "always_on",
    "fixed_round_robin",
    "cost_ordered_static",
    "random_subset",
)

# --------------------------------------------------------------------------
# Calibrated resource envelopes (master plan section 8)
# --------------------------------------------------------------------------

#: Genesis I used absolute envelopes from 512MB to 10GB against materials whose
#: peak residency was measured in kilobytes.  None of them bound, so P4 was
#: instrumented by a budget that never applied.  Genesis II measures the
#: reference footprint first and then sets budgets as a fraction of it.
ENVELOPE_FRACTIONS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00, 1.50)

ENVELOPE_ADMISSION = (
    "memory_pressure_changes",
    "at_least_one_candidate_changes_behaviour",
    "at_least_one_nontrivial_capability_degrades_or_reallocates",
    "budget_is_enforced_rather_than_descriptive",
)

#: The fraction of arms whose behaviour must change for an envelope to bind.
ENVELOPE_BINDING_MINIMUM_ARMS = 1

PRECISION_ARMS = (
    "full_precision",
    "post_hoc_compression",
    "native_ternary",
    "native_quinary",
    "exact_microstore_plus_low_bit_field",
    "learned_codebook",
    "adaptive_mixed_radix",
)

FRONTIER_MEASURES = (
    "absolute_utility",
    "utility_per_resident_byte",
    "retained_history_per_byte",
    "learning_per_written_byte",
    "rare_case_accuracy",
    "calibration",
    "recovery",
    "latency",
)

#: Do not select the smallest artifact when absolute capability is materially
#: worse.  A frontier point that trades away more than this much absolute
#: utility is not eligible to be selected on density alone.
FRONTIER_ABSOLUTE_UTILITY_FLOOR = 0.05

# --------------------------------------------------------------------------
# Candidate family (master plan section 9)
# --------------------------------------------------------------------------

CANDIDATES: dict[str, dict[str, Any]] = {
    "L0_prior_selected_field": {
        "form": "event_sourced_plastic_field",
        "distinct_mechanism": "the Genesis I selected field carried forward byte-identically",
        "microstore_mode": "low_bit",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 4.0,
        "inherited_from": PARENT_SELECTED_CANDIDATE,
    },
    "L1_associative_monolith": {
        "form": "s2_derived_associative_monolith",
        "distinct_mechanism": "a flat exact associative microstore with unconditional consolidation and no field topology",
        "microstore_mode": "exact",
        "structural_layer": False,
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 1.5,
    },
    "L2_associative_monolithic_plastic_field": {
        "form": "associative_monolithic_plastic_field",
        "distinct_mechanism": "one dense plastic transition over an exact associative microstore",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 3.0,
    },
    "L3_associative_graph_plastic_field": {
        "form": "associative_graph_plastic_field",
        "distinct_mechanism": "typed relation graph over microstore addresses with per-edge scope",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 4.5,
    },
    "L4_associative_cellular_field": {
        "form": "associative_cellular_field",
        "distinct_mechanism": "bounded-radius neighbourhood rules over microstore cells",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 5.0,
    },
    "L5_associative_state_space_field": {
        "form": "associative_state_space_field",
        "distinct_mechanism": "input-dependent bounded recurrence reading the microstore",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 5.5,
    },
    "L6_associative_event_sourced_field": {
        "form": "associative_event_sourced_field",
        "distinct_mechanism": "append-only archive projecting into an exact microstore and structure",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 4.5,
    },
    "L7_exact_microstore_mixed_radix_field": {
        "form": "exact_microstore_mixed_radix_structural_field",
        "distinct_mechanism": "exact microstore beneath a per-region radix-selected structural layer",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 6.0,
    },
    "L8_consolidation_first_field": {
        "form": "consolidation_first_field",
        "distinct_mechanism": "repeated associations are induced into typed rules before anything else runs",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 5.0,
    },
    "L9_minimal_sufficient_field": {
        "form": "minimal_sufficient_field",
        "distinct_mechanism": "exact microstore plus rule induction and nothing else",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": False,
        "continuous_time": False,
        "complexity_weight": 2.0,
    },
    "L10_grok_original_compositional_field": {
        "form": "grok_original_compositional_field",
        "distinct_mechanism": "reserved for the Grok material author's own compositional proposal",
        "microstore_mode": "exact",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 6.5,
        "origin": "grok_original_material_author",
    },
    "L11_integrated_winner": {
        "form": "integrated_field",
        "distinct_mechanism": "the verified-active mechanisms of L1-L10 composed under one exact shell",
        "microstore_mode": "mixed",
        "structural_layer": True,
        "topology_dynamic": True,
        "continuous_time": False,
        "complexity_weight": 7.0,
    },
}

#: Prefer the simplest candidate when performance is equivalent.  Two arms are
#: equivalent when their paired difference falls inside this band.
SIMPLICITY_TIE_BAND = 0.01

# --------------------------------------------------------------------------
# Controls.  S2 gets both representations; a monolithic hybrid may win.
# --------------------------------------------------------------------------

CANONICAL_S2_ID = "S2_task_independent_monolithic_persistent_core"
S2_LOW_BIT_ID = "S2_low_bit_constrained"

S2_ALIASES = {
    "S2": CANONICAL_S2_ID,
    "s2": CANONICAL_S2_ID,
    "S2_exact_associative": CANONICAL_S2_ID,
    "S2_equal_opportunity_plastic_monolith": CANONICAL_S2_ID,
    CANONICAL_S2_ID: CANONICAL_S2_ID,
}

CONTROLS: dict[str, dict[str, Any]] = {
    CANONICAL_S2_ID: {
        "form": "monolithic_deterministic_state_machine",
        "note": "the inherited strongest baseline with exact associative writes, unchanged",
        "representation": "exact",
        "plastic": True,
        "eligible_decisive_comparator": True,
    },
    S2_LOW_BIT_ID: {
        "form": "monolithic_deterministic_state_machine_low_bit",
        "note": "identical to S2 except its stored associative values are held to the field radix",
        "representation": "low_bit",
        "plastic": True,
        "eligible_decisive_comparator": False,
    },
    "FR_selected_kernel": {
        "form": "s2_derived_minimal_event_sourced_monolithic_persistent_core",
        "note": "the Final Revision selected kernel carried forward unchanged",
        "representation": "exact",
        "plastic": True,
        "eligible_decisive_comparator": True,
    },
}

BASELINES = (
    CANONICAL_S2_ID,
    S2_LOW_BIT_ID,
    "FR_selected_kernel",
    "static_frozen_field",
    "replay_full_history",
    "summary_replay",
    "retrieval_only",
    "precompiled_procedure_bank",
    "wrong_history_plastic",
    "shuffled_history_plastic",
    "random_growth_plastic",
    "record_store_null",
    "oracle",
)

BASELINE_DEPRIVATION: dict[str, tuple[str, ...]] = {
    CANONICAL_S2_ID: (),
    # This is a representation ablation tied to the parent field's radix, not
    # an unrestricted capability comparator. It remains mandatory in the
    # diagnostic factorial but cannot become the decisive monolith merely
    # because sampling noise puts it above exact S2 on a pilot split.
    S2_LOW_BIT_ID: (),
    "FR_selected_kernel": (),
    "static_frozen_field": ("plasticity",),
    "replay_full_history": ("plasticity",),
    "summary_replay": ("plasticity",),
    "retrieval_only": ("plasticity",),
    "precompiled_procedure_bank": ("plasticity",),
    "wrong_history_plastic": ("correct_history",),
    "shuffled_history_plastic": ("history_order",),
    "random_growth_plastic": ("verified_growth",),
    "record_store_null": ("development",),
    "oracle": (),
}

#: The decisive comparator is the strongest equally plastic monolithic
#: alternative, resolved on the same instances before unblinding.  A monolith
#: is allowed to win: that is a real outcome, not a failure of the program.
DECISIVE_COMPARATOR_RULE = {
    "selection": "highest_scoring_eligible_control_on_the_same_instances",
    "eligibility": ("plastic", "no_deprivation", "parity_audit_passed", "separate_implementation"),
    "resolved_before_unblinding": True,
    "monolith_may_win": True,
    "fallback": CANONICAL_S2_ID,
}

PARITY_CHANNELS = (
    "information",
    "compute",
    "persistence",
    "plasticity",
    "sensors",
    "teaching",
    "memory",
)

PARITY_EXACT_CHANNELS = ("information", "sensors", "teaching")
PARITY_RELATIVE_TOLERANCE = 0.02

# --------------------------------------------------------------------------
# Developmental challenges (master plan section 10)
# --------------------------------------------------------------------------

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

#: Families whose held-out probe key is not present in the observation stream
#: as an exact association.  These are where a structural layer can pay: a
#: microstore alone must fall back to nearest-neighbour, while an induced rule
#: applies.  Named before any result is measured.
GENERALISATION_FAMILIES = (
    "tool_acquisition",
    "novel_sensor_mapping",
    "task_composition_transfer",
    "migration_continuity",
    "causal_system_induction",
    "category_boundary_revision",
)

DEVELOPMENTAL_ARC = (
    "micro_association",
    "repeated_evidence",
    "consolidation",
    "structural_transfer",
    "exception",
    "revision",
    "compiled_procedure",
    "later_reuse",
)

# --------------------------------------------------------------------------
# Grok program (master plan section 11)
# --------------------------------------------------------------------------

GROK_MINIMUM_ROLES = 40
GROK_PREFERRED_ROLES = 56
GROK_PREFERRED_ROLES_MAXIMUM = 72

REVIEW_ROUNDS = (
    "blind_diagnosis",
    "candidate_proposals",
    "cross_examination",
    "challenge_commitment",
    "code_review",
    "post_canary_review",
    "post_pilot_review",
    "final_hostile_review",
)

REVIEW_CELLS = (
    # representation versus architecture
    "representation_versus_architecture_diagnostician",
    "prior_result_reconstructor",
    "exact_versus_low_bit_adjudicator",
    "granularity_confound_reviewer",
    # associative memory design
    "associative_microstore_architect",
    "content_addressing_reviewer",
    "collision_detection_reviewer",
    "provenance_and_scope_reviewer",
    "expiry_and_eviction_reviewer",
    "confidence_calibration_reviewer",
    "microstore_rollback_reviewer",
    # low-bit representation
    "low_bit_arithmetic_reviewer",
    "mixed_radix_packing_reviewer",
    "learned_codebook_reviewer",
    "vector_quantization_reviewer",
    "numerical_stability_reviewer",
    "precision_economics_reviewer",
    # write economy
    "write_economy_ledger_author",
    "update_unit_commensurability_reviewer",
    "rewrite_allocator_author",
    "oracle_granularity_author",
    "attempt_parity_reviewer",
    "utility_per_byte_reviewer",
    # composition
    "mechanism_composition_architect",
    "inert_mechanism_hunter",
    "decorative_machinery_prosecutor",
    "interaction_partner_reviewer",
    "integrated_ablation_designer",
    # topology
    "topology_growth_economics_reviewer",
    "concept_split_reviewer",
    "redundancy_merge_reviewer",
    "pruning_and_archival_reviewer",
    "specialist_pathway_reviewer",
    # scheduling
    "conditional_scheduler_author",
    "fixed_policy_comparator_author",
    "scheduler_overhead_reviewer",
    # S2 fairness
    "s2_fairness_reviewer",
    "s2_low_bit_variant_author",
    "equal_resource_auditor",
    "monolith_advocate",
    "starvation_prosecutor",
    # challenge generation
    "challenge_generator_author",
    "generalisation_family_author",
    "hidden_composition_author",
    "curriculum_arc_reviewer",
    "counterfeit_detection_reviewer",
    # statistics
    "statistical_reviewer",
    "multiplicity_and_power_reviewer",
    "bootstrap_validity_reviewer",
    "oracle_headroom_reviewer",
    # performance
    "performance_reviewer",
    "multiprocessing_safety_reviewer",
    "worker_adaptation_reviewer",
    "checkpoint_restore_performance_reviewer",
    # continuity
    "continuity_authority_reviewer",
    "interruption_and_migration_reviewer",
    "developmental_archive_reviewer",
    "organ_replacement_reviewer",
    # security
    "evaluation_security_reviewer",
    "answer_leakage_prosecutor",
    "checkpoint_coverage_prosecutor",
    "activation_security_reviewer",
    # falsification
    "falsification_reviewer",
    "envelope_binding_prosecutor",
    "null_defence_reviewer",
    "mutation_designer",
    # publication
    "publication_reviewer",
    "limitations_author",
    "sandbox_handoff_reviewer",
    "code_review_core",
    "code_review_campaign",
)

#: Grok output is a claim, never an endpoint.  Every invocation is recorded with
#: its disposition; no analysis counts a Grok judgement as evidence.
GROK_EVIDENCE_POLICY = {
    "opinions_are_evidence": False,
    "record_every_invocation": True,
    "record_disposition": True,
    "dispositions": ("adopted", "adopted_with_changes", "rejected", "superseded", "deferred"),
}

# --------------------------------------------------------------------------
# Execution policy (master plan section 12)
# --------------------------------------------------------------------------

EXECUTION_POLICY = {
    "continuity_lane_nice": 5,
    "campaign_worker_nice": 19,
    "backfill_worker_nice": 19,
    "host_idle_target": (0.0, 0.05),
    "monitor_cumulative_cpu_not_pid": True,
    "multiprocessing_main_guard_required": True,
    "nested_uncontrolled_pools_forbidden": True,
    "spawn_safety_canary_before_campaign": True,
    "heavy_independent_backfill_first": True,
    "content_addressed_receipts": True,
    "anchored_checkpoint_restoration": True,
    "parallel_resumable_acquisition": True,
}

COLLECTOR_REFUSAL_CONDITIONS = (
    "campaign_log_incomplete",
    "required_artifact_missing",
    "checkpoint_does_not_restore",
    "raw_recomputation_differs",
    "mutation_survives",
    "source_digest_differs",
)

#: Long continuity is mechanism-driven.  A real 12-hour lane runs only when the
#: selected architecture has an active continuous-time mechanism; otherwise the
#: frozen event/cycle authority applies and wall time is not used as a proxy.
CONTINUITY_POLICY = {
    "real_wall_clock_lane_requires_active_continuous_time_mechanism": True,
    "real_lane_minimum_seconds": 12 * 60 * 60,
    "frozen_authority_when_no_continuous_time_mechanism": True,
    "duration_change_must_be_preregistered": True,
}

FROZEN_CONTINUITY_REQUIREMENTS = {
    "cycles": 512,
    "events": 250_000,
    "interruptions": 16,
    "checkpoints": 32,
    "migrations": 4,
    "model_organ_replacements": 4,
    "body_organ_replacements": 4,
    "developmental_history_stages": len(DEVELOPMENTAL_ARC),
}

# --------------------------------------------------------------------------
# Canaries (master plan section 13)
# --------------------------------------------------------------------------

CANARIES = {
    "C01": "exact micro-writes improve a held-out result",
    "C02": "low-bit micro-writes improve a held-out result",
    "C03": "micro-write reversal removes benefit",
    "C04": "repeated associations consolidate",
    "C05": "consolidation reduces future cost",
    "C06": "consolidated structure transfers",
    "C07": "exception reopens and revises structure",
    "C08": "granularity allocator chooses micro over topology",
    "C09": "topology is chosen when micro-writes are insufficient",
    "C10": "each integrated mechanism activates",
    "C11": "each necessary mechanism has an integrated ablation effect",
    "C12": "exact field closes or fails to close the S2 gap",
    "C13": "low-bit S2 isolates representation cost",
    "C14": "binding envelopes actually bind",
    "C15": "adaptive precision changes behaviour under pressure",
    "C16": "wrong-history controls remain clean",
    "C17": "shadow fields preserve authoritative state",
    "C18": "procedure failure returns to flexible reasoning",
    "C19": "checkpoint restore reproduces developed state",
    "C20": "process replacement preserves goals",
    "C21": "mutation detectors reject decorative mechanisms",
    "C22": "activation remains false",
}

#: Cheap results admit mechanisms.  They do not earn the terminal claim.
CANARY_POLICY = {"admits_mechanisms": True, "earns_terminal_claim": False}

# --------------------------------------------------------------------------
# Pilot and campaign sizing (master plan sections 14, 15)
# --------------------------------------------------------------------------

PILOT_HISTORIES_MINIMUM = 48
PILOT_HISTORIES_MAXIMUM = 96
PILOT_FAMILIES_MINIMUM = 12
PILOT_FAMILIES_MAXIMUM = 16
PILOT_EPISODES_MINIMUM = 250_000
PILOT_EPISODES_MAXIMUM = 1_000_000

PILOT_ELIMINATION_CONDITIONS = (
    "loses_decisively",
    "depends_on_decorative_mechanisms",
    "fails_binding_budgets",
    "has_invalid_controls",
    "cannot_restore",
)

PRINCIPAL_HISTORIES_MINIMUM = 128
PRINCIPAL_HISTORIES_MAXIMUM = 256
PRINCIPAL_UNITS_MINIMUM = 5_000
PRINCIPAL_UNITS_MAXIMUM = 30_000
REPLICATION_FRACTION_MINIMUM = 1.0 / 3.0
HIDDEN_COMPOSITION_FRACTION_MINIMUM = 1.0 / 3.0
CAMPAIGN_EPISODES_MINIMUM = 1_000_000
CAMPAIGN_EPISODES_MAXIMUM = 5_000_000

FREEZE_SUBJECTS = (
    "source",
    "architecture",
    "update_laws",
    "precision_rules",
    "budgets",
    "baselines",
    "challenge_generators",
    "splits",
    "seeds",
    "statistics",
    "mutations",
    "claim_boundary",
)

# --------------------------------------------------------------------------
# Primary claims (master plan section 16)
# --------------------------------------------------------------------------

CLAIMS: dict[str, dict[str, Any]] = {
    "P1": {
        "statement": "fine-grained associative plasticity matches or exceeds S2 write efficiency",
        "critical": True,
    },
    "P2": {
        "statement": "structural consolidation improves transfer and lowers repeated cost",
        "critical": True,
    },
    "P3": {
        "statement": "conditional granularity selection beats fixed update strategies",
        "critical": True,
    },
    "P4": {
        "statement": "integrated mechanisms have causal nondecorative value",
        "critical": True,
    },
    "P5": {
        "statement": "native precision improves a binding capability-resource frontier",
        "critical": True,
    },
    "P6": {
        "statement": "developmental histories create useful future organization",
        "critical": True,
    },
    "P7": {
        "statement": "the selected material survives interruption, migration and model replacement",
        "critical": True,
    },
    "P8": {
        "statement": "multimodal information updates one coherent field",
        "critical": True,
    },
    "P9": {
        "statement": "the selected field beats the strongest equally plastic monolithic alternative",
        "critical": True,
        "decisive": True,
    },
    "P10": {
        "statement": "the advantage replicates and survives hidden task composition",
        "critical": True,
        "decisive": True,
    },
}

DECISIVE_CLAIMS = ("P9", "P10")

OUTCOME_A_REQUIREMENTS = {
    "decisive_effect_minimum": SESOI,
    "confidence_lower_bound_above": 0.0,
    "oracle_headroom_minimum": MINIMUM_ORACLE_HEADROOM,
    "replication_positive": True,
    "hidden_composition_positive": True,
    "surviving_mutations": 0,
    "clean_clone_reproduction": True,
    "all_critical_claims_pass": True,
    "every_integrated_mechanism_active": True,
    "no_nonbinding_envelope_in_the_density_claim": True,
    "parity_audit_passed": True,
}

ROBUST_OUTCOME_A_REQUIREMENTS = {
    "confidence_lower_bound_at_least": SESOI,
    "oracle_headroom_minimum": PREFERRED_ORACLE_HEADROOM,
}

# --------------------------------------------------------------------------
# Declared mutations (master plan section 17)
# --------------------------------------------------------------------------

MUTATIONS = (
    "associative_store_reads_target_answers",
    "write_cost_is_undercounted",
    "candidate_receives_more_write_bandwidth",
    "s2_is_artificially_precision_limited",
    "nonbinding_envelope_reported_as_binding",
    "mechanism_registered_but_never_consumed",
    "ablation_bypassed_through_alias",
    "topology_stores_answers_instead_of_structure",
    "consolidation_copies_outputs",
    "precision_promotion_reads_outcomes",
    "procedure_loses_accuracy",
    "shadow_field_reads_future_authoritative_state",
    "checkpoint_omits_microstore",
    "checkpoint_omits_scheduler",
    "collector_accepts_unanchored_receipts",
    "multiprocessing_child_recursively_launches_main",
    "activation_becomes_true",
)

MUTATION_POLICY = {
    "every_detector_needs_an_injected_defect": True,
    "every_detector_needs_a_paired_clean_case": True,
    "required_survivors": 0,
}

# --------------------------------------------------------------------------
# Claim boundary and terminal outcomes (master plan section 2)
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
    "maximum_outcome_a": "associative_plastic_cognitive_field_candidate",
    "outcome_b": "cognitive_material_genesis_ii_complete",
    "outcome_c": "terminal_compositional_material_null",
    "preserved_parent_classification": PARENT_CLASSIFICATION,
    "earlier_negative_result_is_never_erased": True,
}

TERMINAL_OUTCOMES = {
    "A": {
        "classification": "associative_plastic_cognitive_field_candidate",
        "readiness": "tangible_sandbox_ready",
        "requires": "a positive replicated hidden-composition advantage over the strongest equally resourced equally plastic alternative",
    },
    "B": {
        "classification": "cognitive_material_genesis_ii_complete",
        "status": "compositional_advantage_unproven",
        "readiness": "tangible_sandbox_ready",
        "requires": "an improved, simplified, operationally ready architecture whose decisive advantage remains null",
    },
    "C": {
        "classification": "terminal_compositional_material_null",
        "requires": "a critical mechanism, instrument or architecture that cannot be made valid under the bounded candidate program",
    },
}

#: Outcome C is reserved for a broken prerequisite that makes the measurement
#: uninterpretable.  A sound program with a null decisive claim is Outcome B.
#: This is the rule the Genesis I adjudication panel established unanimously.
OUTCOME_C_RESERVED_FOR = "prerequisite_failure_that_makes_measurement_uninterpretable"

PREREQUISITES = (
    "record_store_null_scores_at_chance",
    "reference_learner_solves_every_admitted_family",
    "oracle_reaches_ceiling",
    "probe_splits_disjoint",
    "no_answer_leakage",
    "parity_audit_passed",
)

# --------------------------------------------------------------------------
# Frozen statistics
# --------------------------------------------------------------------------

STATISTICS = {
    "independent_unit": "family_history_cell",
    "pairing": "same_family_same_history_same_instances_candidate_versus_comparator",
    "primary_estimator": "mean_paired_difference_over_family_history_cells",
    "confidence_method": "bias_corrected_accelerated_bootstrap_over_family_history_cells",
    "bootstrap_resamples": 10_000,
    "resampling_unit": "family_history_cell",
    "episodes_are_not_independent": True,
    "multiplicity_correction": "intersection_union_gate",
    "multiplicity_family": "outcome_a_requires_all_ten_claims_and_no_marginal_familywise_claim_is_made",
    "decisive_claim_alpha": 0.05,
    "one_sided": False,
    "analysis_frozen_before_principal": True,
    "selection_uses_independent_pilot_only": True,
    "principal_sample_size_method": "normal_approximation_from_pilot_paired_cell_variance_clamped_to_frozen_bounds",
    "subgroup_analyses_are_exploratory_only": True,
}

SEALING = {
    "learner_reads_expected_label": False,
    "learner_sees_only_opaque_batch_digest": True,
    "verification_returns_scalar_outcome_only": True,
    "answer_key_committed_before_freeze": True,
    "commitment_scheme": "sha256_over_generator_source_seed_namespace_and_configuration",
    "principal_instances_generated_after_freeze_commit": True,
    "generator_source_digest_published": True,
}

# --------------------------------------------------------------------------
# Publication (master plan section 18)
# --------------------------------------------------------------------------

PUBLICATION_SUBJECTS = (
    "prior_result_reconstruction",
    "representation_versus_architecture_diagnosis",
    "write_economy_ledger",
    "candidate_tournament",
    "mechanism_activity_matrix",
    "binding_resource_frontiers",
    "developmental_curricula",
    "continuity",
    "principal",
    "replication",
    "hidden_composition",
    "grok_archive",
    "mutations",
    "clean_clone",
    "limitations",
    "tangible_sandbox_handoff",
)

THESIS = (
    "A viable cognitive material must learn cheaply at the associative scale, "
    "reorganize usefully at the structural scale, and invoke expensive topology "
    "only when simpler change is insufficient."
)


def configuration() -> dict[str, Any]:
    """The complete frozen configuration of the Genesis II program."""
    return {
        "activation": ACTIVATION,
        "program": PROGRAM,
        "parent_program": PARENT_PROGRAM,
        "parent_terminal_tag": PARENT_TERMINAL_TAG,
        "parent_classification": PARENT_CLASSIFICATION,
        "parent_status": PARENT_STATUS,
        "parent_decisive_effect": PARENT_DECISIVE_EFFECT,
        "parent_decisive_ci": list(PARENT_DECISIVE_CI),
        "parent_selected_candidate": PARENT_SELECTED_CANDIDATE,
        "parent_strongest_comparator": PARENT_STRONGEST_COMPARATOR,
        "parent_failing_claims": list(PARENT_FAILING_CLAIMS),
        "parent_inert_mechanisms_of_nine": PARENT_INERT_MECHANISMS_OF_NINE,
        "preserved_tags": list(PRESERVED_TAGS),
        "sesoi": SESOI,
        "confidence": CONFIDENCE,
        "power_target": POWER_TARGET,
        "minimum_oracle_headroom": MINIMUM_ORACLE_HEADROOM,
        "preferred_oracle_headroom": PREFERRED_ORACLE_HEADROOM,
        "implementation_branch": IMPLEMENTATION_BRANCH,
        "ready_tag": READY_TAG,
        "terminal_tag": TERMINAL_TAG,
        "stages": list(STAGES),
        "diagnostic_questions": dict(DIAGNOSTIC_QUESTIONS),
        "update_granularities": list(UPDATE_GRANULARITIES),
        "granularity_cost_weight": dict(GRANULARITY_COST_WEIGHT),
        "micro_association_kinds": list(MICRO_ASSOCIATION_KINDS),
        "meso_structure_kinds": list(MESO_STRUCTURE_KINDS),
        "macro_topology_operations": list(MACRO_TOPOLOGY_OPERATIONS),
        "required_layers": list(REQUIRED_LAYERS),
        "exact_shell_fields": list(EXACT_SHELL_FIELDS),
        "microstore_requirements": list(MICROSTORE_REQUIREMENTS),
        "factorial_cells": FACTORIAL_CELLS,
        "factorial_interpretation": dict(FACTORIAL_INTERPRETATION),
        "factorial_equal_channels": list(FACTORIAL_EQUAL_CHANNELS),
        "ledger_fields": list(LEDGER_FIELDS),
        "ledger_reports": list(LEDGER_REPORTS),
        "allocator_comparators": list(ALLOCATOR_COMPARATORS),
        "topology_misallocation_ceiling": TOPOLOGY_MISALLOCATION_CEILING,
        "required_mechanisms": list(REQUIRED_MECHANISMS),
        "mechanism_declaration_fields": list(MECHANISM_DECLARATION_FIELDS),
        "mechanism_admission": dict(MECHANISM_ADMISSION),
        "decorative_ablation_floor": DECORATIVE_ABLATION_FLOOR,
        "scheduler_comparators": list(SCHEDULER_COMPARATORS),
        "envelope_fractions": list(ENVELOPE_FRACTIONS),
        "envelope_admission": list(ENVELOPE_ADMISSION),
        "envelope_binding_minimum_arms": ENVELOPE_BINDING_MINIMUM_ARMS,
        "precision_arms": list(PRECISION_ARMS),
        "frontier_measures": list(FRONTIER_MEASURES),
        "frontier_absolute_utility_floor": FRONTIER_ABSOLUTE_UTILITY_FLOOR,
        "candidates": CANDIDATES,
        "simplicity_tie_band": SIMPLICITY_TIE_BAND,
        "canonical_s2_id": CANONICAL_S2_ID,
        "s2_low_bit_id": S2_LOW_BIT_ID,
        "s2_aliases": dict(S2_ALIASES),
        "controls": CONTROLS,
        "baselines": list(BASELINES),
        "baseline_deprivation": {key: list(value) for key, value in BASELINE_DEPRIVATION.items()},
        "decisive_comparator_rule": DECISIVE_COMPARATOR_RULE,
        "parity_channels": list(PARITY_CHANNELS),
        "parity_exact_channels": list(PARITY_EXACT_CHANNELS),
        "parity_relative_tolerance": PARITY_RELATIVE_TOLERANCE,
        "challenge_families": list(CHALLENGE_FAMILIES),
        "generalisation_families": list(GENERALISATION_FAMILIES),
        "developmental_arc": list(DEVELOPMENTAL_ARC),
        "grok_minimum_roles": GROK_MINIMUM_ROLES,
        "grok_preferred_roles": GROK_PREFERRED_ROLES,
        "grok_preferred_roles_maximum": GROK_PREFERRED_ROLES_MAXIMUM,
        "review_rounds": list(REVIEW_ROUNDS),
        "review_cells": list(REVIEW_CELLS),
        "grok_evidence_policy": dict(GROK_EVIDENCE_POLICY),
        "execution_policy": dict(EXECUTION_POLICY),
        "collector_refusal_conditions": list(COLLECTOR_REFUSAL_CONDITIONS),
        "continuity_policy": dict(CONTINUITY_POLICY),
        "frozen_continuity_requirements": dict(FROZEN_CONTINUITY_REQUIREMENTS),
        "canaries": dict(CANARIES),
        "canary_policy": dict(CANARY_POLICY),
        "pilot_histories_minimum": PILOT_HISTORIES_MINIMUM,
        "pilot_histories_maximum": PILOT_HISTORIES_MAXIMUM,
        "pilot_families_minimum": PILOT_FAMILIES_MINIMUM,
        "pilot_families_maximum": PILOT_FAMILIES_MAXIMUM,
        "pilot_episodes_minimum": PILOT_EPISODES_MINIMUM,
        "pilot_episodes_maximum": PILOT_EPISODES_MAXIMUM,
        "pilot_elimination_conditions": list(PILOT_ELIMINATION_CONDITIONS),
        "principal_histories_minimum": PRINCIPAL_HISTORIES_MINIMUM,
        "principal_histories_maximum": PRINCIPAL_HISTORIES_MAXIMUM,
        "principal_units_minimum": PRINCIPAL_UNITS_MINIMUM,
        "principal_units_maximum": PRINCIPAL_UNITS_MAXIMUM,
        "replication_fraction_minimum": REPLICATION_FRACTION_MINIMUM,
        "hidden_composition_fraction_minimum": HIDDEN_COMPOSITION_FRACTION_MINIMUM,
        "campaign_episodes_minimum": CAMPAIGN_EPISODES_MINIMUM,
        "campaign_episodes_maximum": CAMPAIGN_EPISODES_MAXIMUM,
        "freeze_subjects": list(FREEZE_SUBJECTS),
        "claims": CLAIMS,
        "decisive_claims": list(DECISIVE_CLAIMS),
        "outcome_a_requirements": OUTCOME_A_REQUIREMENTS,
        "robust_outcome_a_requirements": ROBUST_OUTCOME_A_REQUIREMENTS,
        "mutations": list(MUTATIONS),
        "mutation_policy": dict(MUTATION_POLICY),
        "claim_boundary": CLAIM_BOUNDARY,
        "terminal_outcomes": TERMINAL_OUTCOMES,
        "outcome_c_reserved_for": OUTCOME_C_RESERVED_FOR,
        "prerequisites": list(PREREQUISITES),
        "statistics": STATISTICS,
        "sealing": SEALING,
        "publication_subjects": list(PUBLICATION_SUBJECTS),
        "thesis": THESIS,
    }


def configuration_digest() -> str:
    payload = json.dumps(configuration(), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def demo() -> None:
    """Runnable self-check for the invariants the constitution is responsible for."""
    assert ACTIVATION is False
    assert CLAIM_BOUNDARY["unqualified_nous"] is False
    assert CLAIM_BOUNDARY["external_activation"] is False

    # Every factorial cell must name an arm the tournament can actually build.
    known = set(CANDIDATES) | set(CONTROLS)
    for cell, row in FACTORIAL_CELLS.items():
        assert row["arm"] in known, f"factorial cell {cell} names an unknown arm {row['arm']}"

    # Both S2 representations are diagnostic, but only unrestricted exact S2
    # is eligible to become the decisive capability comparator.
    assert CONTROLS[CANONICAL_S2_ID]["representation"] == "exact"
    assert CONTROLS[S2_LOW_BIT_ID]["representation"] == "low_bit"
    assert CONTROLS[CANONICAL_S2_ID]["eligible_decisive_comparator"] is True
    assert CONTROLS[S2_LOW_BIT_ID]["eligible_decisive_comparator"] is False
    assert DECISIVE_COMPARATOR_RULE["monolith_may_win"] is True

    # Grok roles must reach the preferred band and be distinct.
    assert len(REVIEW_CELLS) == len(set(REVIEW_CELLS)), "duplicate Grok role"
    assert GROK_MINIMUM_ROLES <= len(REVIEW_CELLS) <= GROK_PREFERRED_ROLES_MAXIMUM, len(REVIEW_CELLS)
    assert len(REVIEW_CELLS) >= GROK_PREFERRED_ROLES, len(REVIEW_CELLS)

    # Envelopes are relative fractions, never absolute byte counts.
    assert all(0.0 < fraction <= 1.5 for fraction in ENVELOPE_FRACTIONS)
    assert min(ENVELOPE_FRACTIONS) < 0.10, "no envelope small enough to bind"

    # Topology must be the most expensive granularity by a wide margin.
    assert GRANULARITY_COST_WEIGHT["topology_revision"] >= 16 * GRANULARITY_COST_WEIGHT["micro_association"]

    # The nine required mechanisms and the twenty-two canaries are complete.
    assert len(REQUIRED_MECHANISMS) == 9
    assert len(CANARIES) == 22
    assert sorted(CANARIES) == [f"C{index:02d}" for index in range(1, 23)]
    assert len(MUTATIONS) == 17 and len(set(MUTATIONS)) == 17

    # The digest must be stable across calls.
    assert configuration_digest() == configuration_digest()
    print(f"genesis2 constitution self-check passed: {len(REVIEW_CELLS)} Grok roles, digest {configuration_digest()[:12]}")


if __name__ == "__main__":
    demo()
