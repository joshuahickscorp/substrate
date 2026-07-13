"""Unit tests for the memory-organization scaffold (epoch G1-R1).

These tests exercise the capability contracts, future-decision value contract, matched budget,
episodic-harm guard, and activation gate. They assert digest stability, fail-closed refusals,
determinism under seed, control-set completeness, and that the activation gate refuses by default.
No capability is claimed. The tests are pure Python: no network, no files, no heavy dependencies.
"""

from __future__ import annotations

import pytest

from mop.mechanisms.memory_organization_scaffold import (
    CLAIM_SCOPE,
    EPISODIC_HARM_NULL,
    FUTURE_DECISION_CONTROLS,
    FUTURE_DECISION_TARGET,
    PRIOR_NULLS,
    REPLAY_NULL,
    REQUIRED_CAPABILITIES,
    SCIENTIFIC_CAPABILITY_CLAIM,
    ActivationReceipt,
    CapabilityExercise,
    EpisodicHarmGuard,
    FutureDecisionValueContract,
    MatchedRetrievalBudget,
    MemoryActivationGate,
    MemoryOrganizationContract,
    MemoryOrganizationRefusal,
    build_memory_organization,
    coverage,
    default_activation_gate,
    default_episodic_harm_guard,
    default_future_decision_contract,
    simulate_capability_ops,
    toy_decision_values,
)

MEMORY_ORGANIZATION_SCHEMA = "mop-memory-organization/v1"


# ---------------------------------------------------------------------------
# Module-level invariants.
# ---------------------------------------------------------------------------


def test_claim_scope_is_exact() -> None:
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_scientific_capability_claim_is_false() -> None:
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_named_prior_nulls_are_fixed() -> None:
    assert PRIOR_NULLS == (REPLAY_NULL, EPISODIC_HARM_NULL)
    assert REQUIRED_CAPABILITIES == ("revision", "provenance", "deletion", "action")


# ---------------------------------------------------------------------------
# Section A. Capability exercises.
# ---------------------------------------------------------------------------


def test_capability_exercise_build_and_digest_stable() -> None:
    first = CapabilityExercise.build("revision", ("revision.op:a", "revision.op:b"))
    second = CapabilityExercise.build("revision", ("revision.op:a", "revision.op:b"))
    assert first.witness_sha256 == second.witness_sha256
    assert len(first.witness_sha256) == 64


def test_capability_exercise_detects_tampering() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="witness digest"):
        CapabilityExercise(
            capability="revision",
            exercised=True,
            op_log=("revision.op:a",),
            witness_sha256="0" * 64,
        )


def test_capability_exercise_rejects_declaration_without_exercise() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="must be exercised"):
        CapabilityExercise(
            capability="deletion",
            exercised=False,
            op_log=("deletion.op:a",),
            witness_sha256="0" * 64,
        )


def test_capability_exercise_rejects_unknown_capability() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="unsupported capability"):
        CapabilityExercise.build("telepathy", ("telepathy.op:a",))


# ---------------------------------------------------------------------------
# Section A. Memory organization contract.
# ---------------------------------------------------------------------------


def test_build_memory_organization_is_deterministic() -> None:
    assert build_memory_organization(4).sha256 == build_memory_organization(4).sha256


def test_memory_organization_covers_all_four_capabilities() -> None:
    org = build_memory_organization(1)
    assert tuple(e.capability for e in org.exercises) == REQUIRED_CAPABILITIES


def test_memory_organization_rejects_capability_drift() -> None:
    only_two = tuple(
        CapabilityExercise.build(cap, simulate_capability_ops(1, cap)) for cap in ("revision", "provenance")
    )
    with pytest.raises(MemoryOrganizationRefusal, match="coverage or order drift"):
        MemoryOrganizationContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            organization_id="memorg.partial",
            exercises=only_two,
            prediction_only_sufficient=False,
        )


def test_memory_organization_rejects_reordered_capabilities() -> None:
    reordered = tuple(
        CapabilityExercise.build(cap, simulate_capability_ops(1, cap))
        for cap in ("provenance", "revision", "deletion", "action")
    )
    with pytest.raises(MemoryOrganizationRefusal, match="coverage or order drift"):
        MemoryOrganizationContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            organization_id="memorg.reordered",
            exercises=reordered,
            prediction_only_sufficient=False,
        )


def test_memory_organization_rejects_prediction_only_sufficient() -> None:
    exercises = tuple(
        CapabilityExercise.build(cap, simulate_capability_ops(2, cap)) for cap in REQUIRED_CAPABILITIES
    )
    with pytest.raises(MemoryOrganizationRefusal, match="prediction-only"):
        MemoryOrganizationContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            organization_id="memorg.predonly",
            exercises=exercises,
            prediction_only_sufficient=True,
        )


# ---------------------------------------------------------------------------
# Section B. Future-decision value contract and matched budget.
# ---------------------------------------------------------------------------


def test_default_future_decision_contract_valid_and_stable() -> None:
    contract = default_future_decision_contract()
    assert contract.value_target == FUTURE_DECISION_TARGET
    assert contract.controls == FUTURE_DECISION_CONTROLS
    assert contract.sha256 == default_future_decision_contract().sha256


def test_future_decision_rejects_prediction_target() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="not prediction"):
        FutureDecisionValueContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            value_target="prediction",
            controls=FUTURE_DECISION_CONTROLS,
            matched=MatchedRetrievalBudget(params=1, retrieval_ops=1, memory_bytes=1, write_steps=1),
            matched_cost_required=True,
            replication_min=2,
            prediction_only_sufficient=False,
        )


def test_future_decision_rejects_control_drift() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="controls or order drift"):
        FutureDecisionValueContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            value_target=FUTURE_DECISION_TARGET,
            controls=("no-memory", "flat-memory"),
            matched=MatchedRetrievalBudget(params=1, retrieval_ops=1, memory_bytes=1, write_steps=1),
            matched_cost_required=True,
            replication_min=2,
            prediction_only_sufficient=False,
        )


def test_future_decision_requires_matched_cost() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="matched retrieval cost"):
        FutureDecisionValueContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            value_target=FUTURE_DECISION_TARGET,
            controls=FUTURE_DECISION_CONTROLS,
            matched=MatchedRetrievalBudget(params=1, retrieval_ops=1, memory_bytes=1, write_steps=1),
            matched_cost_required=False,
            replication_min=2,
            prediction_only_sufficient=False,
        )


def test_future_decision_rejects_single_replication() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="two independent replications"):
        FutureDecisionValueContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            value_target=FUTURE_DECISION_TARGET,
            controls=FUTURE_DECISION_CONTROLS,
            matched=MatchedRetrievalBudget(params=1, retrieval_ops=1, memory_bytes=1, write_steps=1),
            matched_cost_required=True,
            replication_min=1,
            prediction_only_sufficient=False,
        )


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="non-vacuous"):
        MatchedRetrievalBudget(params=0, retrieval_ops=1, memory_bytes=1, write_steps=1)


def test_future_decision_rejects_widened_claim_scope() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="claim scope cannot be widened"):
        FutureDecisionValueContract(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            value_target=FUTURE_DECISION_TARGET,
            controls=FUTURE_DECISION_CONTROLS,
            matched=MatchedRetrievalBudget(params=1, retrieval_ops=1, memory_bytes=1, write_steps=1),
            matched_cost_required=True,
            replication_min=2,
            prediction_only_sufficient=False,
            claim_scope="a capability was demonstrated",
        )


# ---------------------------------------------------------------------------
# Section C. Episodic-harm guard (the episodic-harm null).
# ---------------------------------------------------------------------------


def test_episodic_harm_guard_passes_within_tolerance() -> None:
    guard = default_episodic_harm_guard()
    assert guard.assert_no_episodic_harm(no_memory_value=0.5, episodic_value=0.6) == pytest.approx(-0.1)


def test_episodic_harm_guard_fails_closed_on_measured_harm() -> None:
    guard = EpisodicHarmGuard(schema=MEMORY_ORGANIZATION_SCHEMA, harm_tolerance=0.05)
    with pytest.raises(MemoryOrganizationRefusal, match="measurably harms"):
        guard.assert_no_episodic_harm(no_memory_value=0.9, episodic_value=0.2)


def test_episodic_harm_guard_rejects_negative_tolerance() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="nonnegative"):
        EpisodicHarmGuard(schema=MEMORY_ORGANIZATION_SCHEMA, harm_tolerance=-0.1)


# ---------------------------------------------------------------------------
# Section D. Activation gate and receipt.
# ---------------------------------------------------------------------------


def test_activation_gate_refuses_by_default() -> None:
    gate = default_activation_gate()
    with pytest.raises(MemoryOrganizationRefusal, match="not earned"):
        gate.authorize()


def test_activation_gate_rejects_permissive_construction() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="cannot be permitted"):
        MemoryActivationGate(activation_permitted=True)


def test_activation_gate_authorizes_with_valid_receipt() -> None:
    gate = default_activation_gate()
    receipt = ActivationReceipt.build(issuer="external-audit-lab", replications=3)
    gate.authorize(receipt)


def test_activation_receipt_detects_forgery() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="digest does not match"):
        ActivationReceipt(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            issuer="external-audit-lab",
            cleared_nulls=PRIOR_NULLS,
            future_decision_win_replications=3,
            matched_cost_confirmed=True,
            episodic_harm_absent=True,
            receipt_sha256="0" * 64,
        )


def test_activation_receipt_requires_both_named_nulls() -> None:
    with pytest.raises(MemoryOrganizationRefusal, match="both named prior nulls"):
        ActivationReceipt(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            issuer="external-audit-lab",
            cleared_nulls=(REPLAY_NULL,),
            future_decision_win_replications=3,
            matched_cost_confirmed=True,
            episodic_harm_absent=True,
            receipt_sha256="0" * 64,
        )


# ---------------------------------------------------------------------------
# Section E. Deterministic mechanics probes and coverage.
# ---------------------------------------------------------------------------


def test_simulate_capability_ops_is_deterministic() -> None:
    assert simulate_capability_ops(9, "action") == simulate_capability_ops(9, "action")
    assert simulate_capability_ops(9, "action") != simulate_capability_ops(10, "action")


def test_toy_decision_values_is_deterministic_and_covers_all_arms() -> None:
    values = toy_decision_values(5)
    assert values == toy_decision_values(5)
    assert set(values) == {"organized-memory", *FUTURE_DECISION_CONTROLS}


def test_coverage_lists_every_subquestion_with_two_bullets() -> None:
    cov = coverage()
    assert set(cov) == {
        "future-decision-value-not-prediction",
        "four-capabilities-exercised",
        "declared-decision-controls",
        "replay-parity-null",
        "episodic-harm-null",
    }
    for bullets in cov.values():
        assert len(bullets) >= 2
