from __future__ import annotations

import pytest

from mop.mechanisms.stage4_integration_advantage_scaffold import (
    CLAIM_SCOPE,
    MIN_CONFIRMED_MECHANISMS,
    PRIOR_NULL,
    REQUIRED_ABLATION_ARMS,
    STAGE4_SCHEMA,
    STRONG_BASELINES,
    AblationArm,
    AblationLadder,
    IntegrationBatteryContract,
    JointAdvantageContract,
    MatchedBudget,
    Stage3ConfirmationReceipt,
    Stage4EntryGate,
    Stage4Refusal,
    authorize_battery,
    build_ablation_ladder,
    build_stage3_receipt,
    distinct_confirmed_mechanisms,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _receipt(mechanism_id: str, gate_digest: str) -> Stage3ConfirmationReceipt:
    return build_stage3_receipt(mechanism_id=mechanism_id, promotion_gate_digest=gate_digest)


def _joint_advantage() -> JointAdvantageContract:
    budget = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    return JointAdvantageContract(
        schema=STAGE4_SCHEMA,
        integrated_budget=budget,
        baseline_budget=budget,
        baselines=STRONG_BASELINES,
        frontier_metric="held_out_transfer_accuracy",
        min_effect=0.03,
        matched_cost_required=True,
        replication_min=3,
        prior_null=PRIOR_NULL,
    )


def _battery() -> IntegrationBatteryContract:
    receipts = (_receipt("mech.alpha", _DIGEST_A), _receipt("mech.beta", _DIGEST_B))
    return IntegrationBatteryContract(
        schema=STAGE4_SCHEMA,
        receipts=receipts,
        joint_advantage=_joint_advantage(),
        ablation=build_ablation_ladder(("mech.alpha", "mech.beta")),
        prior_null=PRIOR_NULL,
    )


def test_receipt_digest_is_stable_and_deterministic() -> None:
    first = _receipt("mech.alpha", _DIGEST_A)
    second = _receipt("mech.alpha", _DIGEST_A)
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64


def test_receipt_rejects_unconfirmed() -> None:
    with pytest.raises(Stage4Refusal, match="not confirmed"):
        Stage3ConfirmationReceipt(
            mechanism_id="mech.alpha",
            promotion_gate_digest=_DIGEST_A,
            independent_replications=2,
            confirmed=False,
            confirmation_digest=_DIGEST_A,
        )


def test_receipt_rejects_replication_below_floor() -> None:
    with pytest.raises(Stage4Refusal, match="independent replications"):
        build_stage3_receipt(
            mechanism_id="mech.alpha", promotion_gate_digest=_DIGEST_A, independent_replications=1
        )


def test_receipt_detects_tampered_content_digest() -> None:
    with pytest.raises(Stage4Refusal, match="content digest does not match"):
        Stage3ConfirmationReceipt(
            mechanism_id="mech.alpha",
            promotion_gate_digest=_DIGEST_A,
            independent_replications=2,
            confirmed=True,
            confirmation_digest="0" * 64,
        )


def test_receipt_rejects_widened_claim_scope() -> None:
    good = _receipt("mech.alpha", _DIGEST_A)
    with pytest.raises(Stage4Refusal, match="claim scope cannot be widened"):
        Stage3ConfirmationReceipt(
            mechanism_id=good.mechanism_id,
            promotion_gate_digest=good.promotion_gate_digest,
            independent_replications=good.independent_replications,
            confirmed=True,
            confirmation_digest=good.confirmation_digest,
            claim_scope="a capability was demonstrated",
        )


def test_distinct_confirmed_mechanisms_rejects_duplicate() -> None:
    receipts = (_receipt("mech.alpha", _DIGEST_A), _receipt("mech.alpha", _DIGEST_B))
    with pytest.raises(Stage4Refusal, match="distinct mechanisms"):
        distinct_confirmed_mechanisms(receipts)


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(Stage4Refusal, match="non-vacuous"):
        MatchedBudget(params=0, flops=1, memory_bytes=1, wall_clock_ms=1)


def test_joint_advantage_digest_is_stable() -> None:
    assert _joint_advantage().digest() == _joint_advantage().digest()


def test_joint_advantage_requires_matched_compute() -> None:
    integrated = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    cheaper = MatchedBudget(params=1_000, flops=4_096, memory_bytes=65_536, wall_clock_ms=250)
    with pytest.raises(Stage4Refusal, match="matched compute"):
        JointAdvantageContract(
            schema=STAGE4_SCHEMA,
            integrated_budget=integrated,
            baseline_budget=cheaper,
            baselines=STRONG_BASELINES,
            frontier_metric="held_out_transfer_accuracy",
            min_effect=0.03,
            matched_cost_required=True,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_joint_advantage_requires_matched_cost_flag() -> None:
    budget = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    with pytest.raises(Stage4Refusal, match="matched full-system cost"):
        JointAdvantageContract(
            schema=STAGE4_SCHEMA,
            integrated_budget=budget,
            baseline_budget=budget,
            baselines=STRONG_BASELINES,
            frontier_metric="held_out_transfer_accuracy",
            min_effect=0.03,
            matched_cost_required=False,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_joint_advantage_rejects_baseline_drift() -> None:
    budget = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    with pytest.raises(Stage4Refusal, match="incomplete or out of canonical order"):
        JointAdvantageContract(
            schema=STAGE4_SCHEMA,
            integrated_budget=budget,
            baseline_budget=budget,
            baselines=("best-single-mechanism", "static-composition"),
            frontier_metric="held_out_transfer_accuracy",
            min_effect=0.03,
            matched_cost_required=True,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_joint_advantage_requires_the_named_prior_null() -> None:
    budget = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    with pytest.raises(Stage4Refusal, match="prior null"):
        JointAdvantageContract(
            schema=STAGE4_SCHEMA,
            integrated_budget=budget,
            baseline_budget=budget,
            baselines=STRONG_BASELINES,
            frontier_metric="held_out_transfer_accuracy",
            min_effect=0.03,
            matched_cost_required=True,
            replication_min=2,
            prior_null="some-other-null",
        )


def test_joint_advantage_requires_positive_effect() -> None:
    budget = MatchedBudget(params=1_000, flops=8_192, memory_bytes=65_536, wall_clock_ms=250)
    with pytest.raises(Stage4Refusal, match="strictly positive"):
        JointAdvantageContract(
            schema=STAGE4_SCHEMA,
            integrated_budget=budget,
            baseline_budget=budget,
            baselines=STRONG_BASELINES,
            frontier_metric="held_out_transfer_accuracy",
            min_effect=0.0,
            matched_cost_required=True,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_build_ablation_ladder_covers_all_arms() -> None:
    ladder = build_ablation_ladder(("mech.alpha", "mech.beta", "mech.gamma"))
    assert {arm.arm for arm in ladder.arms} == set(REQUIRED_ABLATION_ARMS)
    alone = [arm for arm in ladder.arms if arm.arm == "each-mechanism-alone"]
    assert {arm.mechanism_ids[0] for arm in alone} == {"mech.alpha", "mech.beta", "mech.gamma"}


def test_ablation_ladder_digest_is_stable() -> None:
    a = build_ablation_ladder(("mech.alpha", "mech.beta"))
    b = build_ablation_ladder(("mech.alpha", "mech.beta"))
    assert a.digest() == b.digest()


def test_ablation_arm_rejects_integration() -> None:
    with pytest.raises(Stage4Refusal, match="no ablation arm may be integrated"):
        AblationArm(arm="static-composition", mechanism_ids=("mech.alpha", "mech.beta"), integrated=True)


def test_ablation_ladder_fails_closed_on_missing_rung() -> None:
    arms = (
        AblationArm(arm="each-mechanism-alone", mechanism_ids=("mech.alpha",), integrated=False),
        AblationArm(arm="each-mechanism-alone", mechanism_ids=("mech.beta",), integrated=False),
        AblationArm(arm="best-single", mechanism_ids=("mech.alpha",), integrated=False),
    )
    with pytest.raises(Stage4Refusal, match="missing rungs"):
        AblationLadder(schema=STAGE4_SCHEMA, mechanism_ids=("mech.alpha", "mech.beta"), arms=arms)


def test_ablation_ladder_requires_full_coverage_by_each_mechanism_alone() -> None:
    arms = (
        AblationArm(arm="each-mechanism-alone", mechanism_ids=("mech.alpha",), integrated=False),
        AblationArm(arm="best-single", mechanism_ids=("mech.alpha",), integrated=False),
        AblationArm(arm="static-composition", mechanism_ids=("mech.alpha", "mech.beta"), integrated=False),
    )
    with pytest.raises(Stage4Refusal, match="cover every declared mechanism"):
        AblationLadder(schema=STAGE4_SCHEMA, mechanism_ids=("mech.alpha", "mech.beta"), arms=arms)


def test_battery_digest_is_stable() -> None:
    assert _battery().digest() == _battery().digest()


def test_battery_requires_at_least_two_receipts() -> None:
    with pytest.raises(Stage4Refusal, match="at least"):
        IntegrationBatteryContract(
            schema=STAGE4_SCHEMA,
            receipts=(_receipt("mech.alpha", _DIGEST_A),),
            joint_advantage=_joint_advantage(),
            ablation=build_ablation_ladder(("mech.alpha", "mech.beta")),
            prior_null=PRIOR_NULL,
        )


def test_battery_requires_ladder_to_match_confirmed_set() -> None:
    with pytest.raises(Stage4Refusal, match="cover exactly the confirmed mechanism set"):
        IntegrationBatteryContract(
            schema=STAGE4_SCHEMA,
            receipts=(_receipt("mech.alpha", _DIGEST_A), _receipt("mech.beta", _DIGEST_B)),
            joint_advantage=_joint_advantage(),
            ablation=build_ablation_ladder(("mech.alpha", "mech.gamma")),
            prior_null=PRIOR_NULL,
        )


def test_battery_exposes_confirmed_mechanism_ids() -> None:
    assert _battery().mechanism_ids == ("mech.alpha", "mech.beta")


def test_entry_gate_refuses_by_default_with_no_receipts() -> None:
    gate = Stage4EntryGate()
    with pytest.raises(Stage4Refusal, match="closed"):
        gate.authorize(())


def test_entry_gate_refuses_a_single_confirmed_mechanism() -> None:
    gate = Stage4EntryGate()
    with pytest.raises(Stage4Refusal, match="required"):
        gate.authorize((_receipt("mech.alpha", _DIGEST_A),))


def test_entry_gate_opens_on_two_confirmed_mechanisms() -> None:
    gate = Stage4EntryGate()
    receipts = (_receipt("mech.alpha", _DIGEST_A), _receipt("mech.beta", _DIGEST_B))
    activation = gate.authorize(receipts)
    assert len(activation) == 64
    assert gate.authorize(receipts) == activation


def test_entry_gate_rejects_too_low_a_threshold() -> None:
    with pytest.raises(Stage4Refusal, match="at least"):
        Stage4EntryGate(min_confirmed_mechanisms=1)


def test_authorize_battery_returns_activation_digest() -> None:
    battery = _battery()
    gate = Stage4EntryGate()
    activation = authorize_battery(gate, battery)
    assert len(activation) == 64


def test_claim_scope_constant_is_the_canonical_string() -> None:
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_min_confirmed_mechanisms_matches_the_bar() -> None:
    assert MIN_CONFIRMED_MECHANISMS == 2
