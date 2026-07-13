"""Unit tests for the bounded causal messaging and contradiction repair scaffold.

These tests exercise the contracts, the seeded routing and repair mechanics, the control-set
completeness checks, and the activation gate. They assert fail-closed behavior and determinism.
No capability is claimed.
"""

from __future__ import annotations

import pytest

from mop.mechanisms.messaging_repair_scaffold import (
    BOUNDED_MESSAGE_CONTROLS,
    CLAIM_SCOPE,
    MESSAGING_REPAIR_SCHEMA,
    REPAIR_CONTROLS,
    REPAIR_METRICS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    VERIFICATION_CONTROLS,
    ActivationReceipt,
    BoundedMessageContract,
    ContradictionRepairContract,
    MatchedBudget,
    MessagePlan,
    MessagingActivationGate,
    MessagingRepairRefusal,
    RepairPlan,
    VerificationValueContract,
    assert_disagreement_present,
    causal_message_plan,
    coverage,
    default_bounded_message_contract,
    default_contradiction_repair_contract,
    default_verification_value_contract,
    detect_and_repair,
    verify_control_registry,
)

_BUDGET = MatchedBudget(messages=8, verify_calls=2, flops=64, memory_bytes=128)


# ---------------------------------------------------------------------------
# Section A. Claim scope, registry, matched budget.
# ---------------------------------------------------------------------------


def test_claim_scope_and_capability_flag_are_pinned() -> None:
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_control_registry_is_intact() -> None:
    assert verify_control_registry() is True


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(MessagingRepairRefusal):
        MatchedBudget(messages=0, verify_calls=1, flops=1, memory_bytes=1)


def test_matched_budget_digest_is_stable() -> None:
    assert _BUDGET.digest() == MatchedBudget(messages=8, verify_calls=2, flops=64, memory_bytes=128).digest()
    assert len(_BUDGET.digest()) == 64


# ---------------------------------------------------------------------------
# Section B. Bounded causal messaging (M1).
# ---------------------------------------------------------------------------


def test_bounded_message_contract_digest_is_stable() -> None:
    a = default_bounded_message_contract()
    b = default_bounded_message_contract()
    assert a.sha256 == b.sha256
    assert len(a.sha256) == 64


def test_bounded_message_refuses_unbounded_broadcast() -> None:
    with pytest.raises(MessagingRepairRefusal, match="unbounded broadcast is refused"):
        BoundedMessageContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            bandwidth_limit=3,
            max_fanout=2,
            routing_rule="causal-only",
            allow_unbounded_broadcast=True,
            controls=BOUNDED_MESSAGE_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="limited-broadcast",
        )


def test_bounded_message_requires_causal_routing_rule() -> None:
    with pytest.raises(MessagingRepairRefusal, match="causal-only routing rule"):
        BoundedMessageContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            bandwidth_limit=3,
            max_fanout=2,
            routing_rule="broadcast",
            allow_unbounded_broadcast=False,
            controls=BOUNDED_MESSAGE_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="limited-broadcast",
        )


def test_bounded_message_rejects_control_drift() -> None:
    with pytest.raises(MessagingRepairRefusal, match="controls or order drift"):
        BoundedMessageContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            bandwidth_limit=3,
            max_fanout=2,
            routing_rule="causal-only",
            allow_unbounded_broadcast=False,
            controls=("broadcast-all", "no-message", "random-route"),
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="limited-broadcast",
        )


def test_bounded_message_requires_matched_cost() -> None:
    with pytest.raises(MessagingRepairRefusal, match="matched full-system cost"):
        BoundedMessageContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            bandwidth_limit=3,
            max_fanout=2,
            routing_rule="causal-only",
            allow_unbounded_broadcast=False,
            controls=BOUNDED_MESSAGE_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=False,
            prior_null="limited-broadcast",
        )


def test_bounded_message_rejects_widened_claim_scope() -> None:
    with pytest.raises(MessagingRepairRefusal, match="claim scope cannot be widened"):
        BoundedMessageContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            bandwidth_limit=3,
            max_fanout=2,
            routing_rule="causal-only",
            allow_unbounded_broadcast=False,
            controls=BOUNDED_MESSAGE_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="limited-broadcast",
            claim_scope="a capability was demonstrated",
        )


def test_causal_message_plan_is_deterministic_under_seed() -> None:
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("b", "c")]

    def run() -> MessagePlan:
        return causal_message_plan(edges=edges, bandwidth_limit=2, seed=11)

    first = run()
    second = run()
    assert first.routes == second.routes
    assert first.sha256 == second.sha256


def test_causal_message_plan_respects_bandwidth_limit() -> None:
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")]
    plan = causal_message_plan(edges=edges, bandwidth_limit=2, seed=3)
    out_of_a = [dst for src, dst in plan.routes if src == "a"]
    assert len(out_of_a) == 2


def test_causal_message_plan_refuses_cycle() -> None:
    with pytest.raises(MessagingRepairRefusal, match="cyclic edge set"):
        causal_message_plan(edges=[("a", "b"), ("b", "a")], bandwidth_limit=2, seed=0)


# ---------------------------------------------------------------------------
# Section C. Value of verification (V1).
# ---------------------------------------------------------------------------


def test_verification_contract_valid_and_stable_digest() -> None:
    contract = default_verification_value_contract()
    assert contract.sha256 == default_verification_value_contract().sha256
    assert contract.controls == VERIFICATION_CONTROLS


def test_verification_refuses_always_on() -> None:
    with pytest.raises(MessagingRepairRefusal, match="strictly between"):
        VerificationValueContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            selective=True,
            verify_fraction=1.0,
            value_metric="held_out_error_reduction",
            controls=VERIFICATION_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="always-on-verification-suffices",
        )


def test_verification_refuses_no_verify_floor() -> None:
    with pytest.raises(MessagingRepairRefusal, match="strictly between"):
        VerificationValueContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            selective=True,
            verify_fraction=0.0,
            value_metric="held_out_error_reduction",
            controls=VERIFICATION_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="always-on-verification-suffices",
        )


def test_verification_requires_selective_flag() -> None:
    with pytest.raises(MessagingRepairRefusal, match="declared selective"):
        VerificationValueContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            selective=False,
            verify_fraction=0.25,
            value_metric="held_out_error_reduction",
            controls=VERIFICATION_CONTROLS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="always-on-verification-suffices",
        )


# ---------------------------------------------------------------------------
# Section D. Contradiction repair (K1).
# ---------------------------------------------------------------------------


def test_repair_contract_valid_and_covers_controls() -> None:
    contract = default_contradiction_repair_contract()
    assert contract.controls == REPAIR_CONTROLS
    assert contract.metrics == REPAIR_METRICS


def test_repair_contract_rejects_non_disagreement_trigger() -> None:
    with pytest.raises(MessagingRepairRefusal, match="only on detected disagreement"):
        ContradictionRepairContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            trigger_condition="always",
            controls=REPAIR_CONTROLS,
            metrics=REPAIR_METRICS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="disagreement-only",
        )


def test_repair_contract_rejects_control_drift() -> None:
    with pytest.raises(MessagingRepairRefusal, match="controls or order drift"):
        ContradictionRepairContract(
            schema=MESSAGING_REPAIR_SCHEMA,
            trigger_condition="detected-disagreement",
            controls=("broadcast-all", "no-message", "stale-message", "majority-vote"),
            metrics=REPAIR_METRICS,
            matched=_BUDGET,
            matched_cost_required=True,
            prior_null="disagreement-only",
        )


def test_detect_and_repair_no_disagreement_is_untriggered() -> None:
    plan = detect_and_repair(claims=(("a", 1), ("b", 1), ("c", 1)), seed=0)
    assert plan.triggered is False
    assert plan.routes == ()
    assert plan.dissenters == ()


def test_detect_and_repair_targets_only_dissenters() -> None:
    plan = detect_and_repair(claims=(("a", 1), ("b", 1), ("c", 9)), seed=0)
    assert plan.triggered is True
    assert plan.majority_value == 1
    assert plan.dissenters == ("c",)
    assert plan.routes == (("a", "c"),)


def test_detect_and_repair_is_deterministic_under_seed() -> None:
    claims = (("z", 5), ("y", 5), ("x", 2), ("w", 7))

    def run() -> RepairPlan:
        return detect_and_repair(claims=claims, seed=4)

    assert run().sha256 == run().sha256


def test_detect_and_repair_rejects_duplicate_ids() -> None:
    with pytest.raises(MessagingRepairRefusal, match="unique"):
        detect_and_repair(claims=(("a", 1), ("a", 2)), seed=0)


def test_assert_disagreement_present_fails_closed_on_agreement() -> None:
    with pytest.raises(MessagingRepairRefusal, match="no detected disagreement"):
        assert_disagreement_present((("a", 3), ("b", 3)))


def test_repair_plan_refuses_messages_without_trigger() -> None:
    with pytest.raises(MessagingRepairRefusal, match="untriggered repair plan cannot carry"):
        RepairPlan(
            schema=MESSAGING_REPAIR_SCHEMA,
            seed=0,
            triggered=False,
            majority_value=None,
            dissenters=(),
            routes=(("a", "b"),),
        )


# ---------------------------------------------------------------------------
# Section E. Activation gate.
# ---------------------------------------------------------------------------


def test_activation_gate_refuses_by_default() -> None:
    gate = MessagingActivationGate()
    with pytest.raises(MessagingRepairRefusal, match="activation not earned"):
        gate.authorize()


def test_activation_gate_cannot_be_constructed_preactivated() -> None:
    with pytest.raises(MessagingRepairRefusal, match="pre-activated"):
        MessagingActivationGate(activated=True)


def test_activation_gate_refuses_invalid_receipt() -> None:
    gate = MessagingActivationGate()
    receipt = ActivationReceipt(
        license_id="0" * 64,
        authority="external review board",
        confirmed=True,
        matched_cost_cleared=False,
        controls_cleared=True,
    )
    with pytest.raises(MessagingRepairRefusal, match="does not clear matched cost"):
        gate.authorize(receipt)


def test_activation_gate_accepts_valid_receipt() -> None:
    gate = MessagingActivationGate()
    receipt = ActivationReceipt(
        license_id="a" * 64,
        authority="external review board",
        confirmed=True,
        matched_cost_cleared=True,
        controls_cleared=True,
    )
    gate.authorize(receipt)  # must not raise


# ---------------------------------------------------------------------------
# Coverage record.
# ---------------------------------------------------------------------------


def test_coverage_lists_every_sub_question_with_two_bullets() -> None:
    cov = coverage()
    keys = set(cov)
    assert any(k.startswith("M1") for k in keys)
    assert any(k.startswith("V1") for k in keys)
    assert any(k.startswith("K1") for k in keys)
    for bullets in cov.values():
        assert len(bullets) >= 2
