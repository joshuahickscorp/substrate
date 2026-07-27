"""Frozen scientific configuration for Substrate v2.

The constants in this module are the preregistration.  Principal execution consumes their digest and
refuses a mismatch.  Targets are generated privately by v2fabric and are never placed in observations.

House style: no dashes.
"""

from __future__ import annotations

from substrate import v2io as io

SESOI = 0.05
COMPUTE_PRICE = 0.08
UNNECESSARY_VERIFICATION_PENALTY = 0.04
MISSED_VERIFICATION_PENALTY = 0.12
INTERFERENCE_PENALTY = 0.20
BROKEN_GOAL_PENALTY = 0.20

SPLITS = {
    "development": tuple(range(0, 8)),
    "admission": tuple(range(100, 112)),
    "principal": tuple(range(1000, 1024)),
    "replication": tuple(range(2000, 2012)),
}

EPISODES_PER_PHASE = {
    "phase_0_cold_baseline": 8,
    "phase_1_domain_A_development": 12,
    "phase_2_domain_B_development": 12,
    "phase_3_return_held_out_A": 8,
    "phase_4_held_out_B_transfer": 8,
    "phase_5_positive_transfer_C": 12,
    "phase_6_positive_transfer_D": 8,
    "phase_7_negative_transfer_challenge": 8,
    "phase_8_interruption_exact_restore": 4,
    "phase_9_body_or_tool_change": 8,
    "phase_10_terminal_held_out": 16,
}

PHASES = tuple(EPISODES_PER_PHASE)

CORE_ARMS = (
    "full_v2",
    "fresh_control",
    "transcript_replay_control",
    "episodic_only",
    "semantic_only",
    "no_procedure",
    "more_compute",
    "simple_allocator",
    "no_self_model",
)

DIVERGENCE_ARMS = (
    "history_A",
    "history_B",
    "identical_history_A_replica",
    "shuffled_history",
    "wrong_history",
)

BODIES = ("general", "compact", "tool_dominant")

CLAIM_BOUNDARY = {
    "permitted_classifications": [
        "certified_cognitive_scaffold",
        "persistent_developmental_cognition",
        "reflective_cognitive_organization",
        "functional_or_proto_nous_candidate",
    ],
    "maximum": "functional_or_proto_nous_candidate",
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
        "internal cognitive choices",
        "sandboxed deterministic simulations",
        "read only retrieval",
        "pure local tool calls",
        "proposals never externally executed",
    ],
    "activation": False,
}

DOMAIN_CATALOG = {
    "A": {
        "name": "delayed sequence and boundary memory",
        "observation_schema": {
            "tokens": "three opaque tokens",
            "boundary": "boolean",
            "context": "opaque context label",
            "target": "absent",
        },
        "latent_state": "boundary conditioned successor index",
        "goal": "propose the correct delayed successor token",
        "available_cognitive_actions": ["preserve uncertainty", "retrieve episode", "retrieve procedure", "propose token"],
        "sandboxed_tool_actions": [],
        "outcome_timing": "after the proposal is committed",
        "cost_model": {"base": 1.0, "retrieval": 0.08, "verification": 0.20},
        "domain_boundary": "context label changes after a declared run",
        "independent_unit": "one sealed developmental history seed",
        "generator": "substrate.v2fabric.generate_task",
        "oracle_policy": "execute boundary_route",
        "simple_baselines": ["first token", "second token", "last observed token"],
        "random_baseline": "uniform over three tokens",
        "maximum_compute_baseline": "evaluate every registered operation then verify",
        "answer_leakage_test": "target and target digest absent from serialized observation",
        "floor_test": "uniform choice exceeds zero and remains below ceiling",
        "ceiling_test": "oracle executes the private operation exactly",
        "transfer_relationships": ["A_to_B_boundary_route"],
        "negative_transfer_relationships": ["A_to_D_surface_trap"],
        "task_signature": "conditional ordered selection",
        "required_operation": "boundary_route",
        "surface_family": "opaque sequence tokens",
    },
    "B": {
        "name": "symbolic rule induction",
        "observation_schema": {
            "glyphs": "three permuted opaque glyphs",
            "exception": "boolean",
            "rule_context": "opaque rule label",
            "target": "absent",
        },
        "latent_state": "exception conditioned symbolic transformation",
        "goal": "propose the transformed glyph after a delay",
        "available_cognitive_actions": ["test hypothesis", "retrieve current rule", "retrieve procedure", "propose glyph"],
        "sandboxed_tool_actions": [],
        "outcome_timing": "after the proposal is committed",
        "cost_model": {"base": 1.0, "retrieval": 0.08, "verification": 0.20},
        "domain_boundary": "rule context changes at a declared boundary",
        "independent_unit": "one sealed developmental history seed",
        "generator": "substrate.v2fabric.generate_task",
        "oracle_policy": "execute boundary_route over glyph positions",
        "simple_baselines": ["first glyph", "second glyph", "frequency rule"],
        "random_baseline": "uniform over three glyphs",
        "maximum_compute_baseline": "evaluate every registered operation then verify",
        "answer_leakage_test": "target and target digest absent from serialized observation",
        "floor_test": "uniform choice exceeds zero and remains below ceiling",
        "ceiling_test": "oracle executes the private operation exactly",
        "transfer_relationships": ["A_to_B_boundary_route"],
        "negative_transfer_relationships": [],
        "task_signature": "conditional ordered selection",
        "required_operation": "boundary_route",
        "surface_family": "permuted symbolic glyphs",
    },
    "C": {
        "name": "evidence routing under cost",
        "observation_schema": {
            "sources": "cheap and robust opaque sources",
            "risk": "bounded preoutcome estimate",
            "contradiction": "boolean",
            "budget": "available internal cost",
            "target": "absent",
        },
        "latent_state": "context dependent source sufficiency",
        "goal": "choose the cheapest sufficient evidence source",
        "available_cognitive_actions": ["preserve minority", "retrieve reliability", "verify", "choose source"],
        "sandboxed_tool_actions": ["pure compare"],
        "outcome_timing": "source sufficiency is revealed after commitment",
        "cost_model": {"cheap_source": 0.05, "robust_source": 0.16, "verification": 0.20},
        "domain_boundary": "reliability regime label changes",
        "independent_unit": "one sealed developmental history seed",
        "generator": "substrate.v2fabric.generate_task",
        "oracle_policy": "execute risk_route",
        "simple_baselines": ["always cheap", "always robust", "contradiction only"],
        "random_baseline": "uniform over two sources",
        "maximum_compute_baseline": "query both sources and compare",
        "answer_leakage_test": "private sufficient source absent from observation",
        "floor_test": "random source selection remains below oracle",
        "ceiling_test": "oracle executes private reliability regime",
        "transfer_relationships": ["C_to_D_risk_route"],
        "negative_transfer_relationships": [],
        "task_signature": "cost sensitive conditional routing",
        "required_operation": "risk_route",
        "surface_family": "contradictory evidence sources",
    },
    "D": {
        "name": "sandboxed tool selection",
        "observation_schema": {
            "tools": "cheap and robust deterministic local tools",
            "failure_risk": "bounded preoutcome estimate",
            "known_limitation": "boolean",
            "budget": "available internal cost",
            "target": "absent",
        },
        "latent_state": "tool capability and deterministic failure region",
        "goal": "choose the cheapest sufficient pure local tool",
        "available_cognitive_actions": ["predict competence", "retrieve procedure", "invoke sandboxed tool", "check result"],
        "sandboxed_tool_actions": ["lookup", "transform", "compare", "simulate", "check"],
        "outcome_timing": "tool success is returned after invocation is proposed",
        "cost_model": {"cheap_tool": 0.05, "robust_tool": 0.16, "check": 0.20},
        "domain_boundary": "capability availability changes",
        "independent_unit": "one sealed developmental history seed",
        "generator": "substrate.v2fabric.generate_task",
        "oracle_policy": "execute risk_route",
        "simple_baselines": ["always cheap", "always robust", "limitation only"],
        "random_baseline": "uniform over two tools",
        "maximum_compute_baseline": "invoke both pure tools and check",
        "answer_leakage_test": "private sufficient tool absent from observation",
        "floor_test": "random tool selection remains below oracle",
        "ceiling_test": "oracle executes private failure region",
        "transfer_relationships": ["C_to_D_risk_route"],
        "negative_transfer_relationships": ["A_to_D_surface_trap"],
        "task_signature": "cost sensitive conditional routing",
        "required_operation": "risk_route",
        "surface_family": "opaque selectable tokens resembling domain A",
    },
}

TRANSFER_GRAPH = {
    "positive": [
        {
            "id": "A_to_B_boundary_route",
            "source": "A",
            "target": "B",
            "latent_procedure": "boundary_route",
            "surface_difference": "sequence tokens versus permuted glyphs",
        },
        {
            "id": "C_to_D_risk_route",
            "source": "C",
            "target": "D",
            "latent_procedure": "risk_route",
            "surface_difference": "evidence sources versus pure local tools",
        },
    ],
    "negative": [
        {
            "id": "A_to_D_surface_trap",
            "source": "A",
            "target": "D",
            "source_procedure": "boundary_route",
            "target_procedure": "risk_route",
            "surface_similarity": "both expose opaque ordered candidates",
            "required_behavior": "reject the source procedure by task signature",
        }
    ],
}

CANDIDATE_LADDER = {
    "consolidation": [
        "verification_triggered",
        "boundary_triggered",
        "repetition_plus_verification",
        "hybrid_boundary_plus_verification",
    ],
    "procedure": [
        "exact_successful_motif",
        "typed_task_signature_motif",
        "utility_weighted_generalized_motif",
    ],
    "allocation": [
        "best_fixed_policy",
        "tabular_contextual_policy",
        "regularized_linear_contextual_policy",
    ],
    "self_model": [
        "global_estimate",
        "domain_conditional_estimate",
        "domain_plus_procedure_conditional_estimate",
    ],
    "selection_data": "development split only",
    "freeze_before": "admission split",
    "valid_repair_reasons": [
        "demonstrated software defect",
        "instrument violates declared semantics",
        "invalid control",
        "preregistered candidate not implemented",
    ],
    "valid_measured_null_is_terminal": True,
}

HYPOTHESES = {
    "H_D1": {
        "claim": "cross domain continuity",
        "primary": True,
        "endpoint": "held out B utility minus strongest fresh or replay control",
        "pass": "effect above SESOI, lower confidence bound above zero, retention within SESOI, exact identity",
    },
    "H_D2": {
        "claim": "procedural transfer",
        "primary": True,
        "endpoint": "full transferred utility minus strongest nonprocedural control",
        "pass": "two positive pairs above SESOI and clean negative pair",
    },
    "H_D3": {
        "claim": "endogenous allocation",
        "primary": True,
        "endpoint": "cost adjusted utility minus strongest simple policy",
        "pass": "oracle residual and effect above SESOI with held out transfer",
    },
    "H_D4": {
        "claim": "useful developmental divergence",
        "primary": False,
        "endpoint": "matched future specialization advantage",
        "pass": "advantage above SESOI, identical controls equivalent, wrong history clean",
    },
    "H_D5": {
        "claim": "developmental self model utility",
        "primary": False,
        "endpoint": "held out utility minus no self model",
        "pass": "at least one preregistered use above SESOI",
    },
}

STATISTICS = {
    "independent_unit": "one sealed developmental history generated from one seed",
    "analysis": [
        "paired effect per seed",
        "mean paired effect",
        "median paired effect",
        "95 percent bootstrap confidence interval over seeds",
        "exact paired sign permutation test",
        "standardized paired effect",
        "raw unit ledger",
    ],
    "bootstrap_repetitions": 10_000,
    "bootstrap_seed": 72_021,
    "familywise_policy": "Holm correction over H_D1 H_D2 H_D3",
    "missing_unit_policy": "never discard silently; retry exact checkpoint once then retain terminal failure",
    "replacement_rule": "none after launch",
    "valid_retry": "one exact resume for process failure",
    "terminal_failure": "second process failure or deterministic integrity refusal",
    "excluded_unit": "only preregistered invalid generator output before task exposure",
    "sesoi": SESOI,
}

STOP_AND_FUTILITY = {
    "stop": [
        "operator stop switch",
        "source or configuration drift",
        "activation audit failure",
        "resource floor breach",
        "invalid checkpoint",
        "no dependency ready",
    ],
    "futility": [
        "do not launch if procedural transfer admission is an active valid bed null",
        "after launch do not stop for observed effect size",
    ],
    "retry_limit": 1,
    "thresholds_immutable_after_launch": True,
}


def configuration() -> dict:
    """The content addressed configuration consumed by admission and principal execution."""
    body = {
        "schema": "substrate-v2-frozen-configuration/v1",
        "sesoi": SESOI,
        "compute_price": COMPUTE_PRICE,
        "penalties": {
            "unnecessary_verification": UNNECESSARY_VERIFICATION_PENALTY,
            "missed_verification": MISSED_VERIFICATION_PENALTY,
            "interference": INTERFERENCE_PENALTY,
            "broken_goal": BROKEN_GOAL_PENALTY,
        },
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "episodes_per_phase": EPISODES_PER_PHASE,
        "core_arms": list(CORE_ARMS),
        "divergence_arms": list(DIVERGENCE_ARMS),
        "bodies": list(BODIES),
        "domains": DOMAIN_CATALOG,
        "transfer_graph": TRANSFER_GRAPH,
        "hypotheses": HYPOTHESES,
        "statistics": STATISTICS,
        "stop_and_futility": STOP_AND_FUTILITY,
        "claim_boundary": CLAIM_BOUNDARY,
        "activation": False,
    }
    body["configuration_digest"] = io.sha_obj(body)
    return body
