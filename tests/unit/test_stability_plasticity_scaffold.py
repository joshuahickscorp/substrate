
from __future__ import annotations

import pytest

from mop.mechanisms.stability_plasticity_scaffold import (
    CLAIM_SCOPE,
    DUAL_AXES,
    PRIOR_NULL,
    REQUIRED_CONTROLS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    AxisComparison,
    ConfirmationReceipt,
    ControlArm,
    ControlFamily,
    DualMetricReading,
    JointClaimGate,
    JointImprovementVerdict,
    MatchedCostBudget,
    StabilityPlasticityContract,
    StabilityPlasticityRefusal,
    assert_control_completeness,
    build_split_control_family,
    build_split_verdict,
    coverage,
    default_contract,
    evaluate_joint_improvement,
    simulate_reading,
)

SCHEMA = "mop-stability-plasticity/v1"
_BUDGET = MatchedCostBudget(params=4096, flops=1_048_576, replay_samples=256, update_steps=200)


def _reading(retention: float, future: float) -> DualMetricReading:
    return DualMetricReading(retention=retention, future_learnability=future)


def _family(readings: dict[str, DualMetricReading]) -> ControlFamily:
    arms = tuple(
        ControlArm(control=control, reading=readings[control], matched=_BUDGET)
        for control in REQUIRED_CONTROLS
    )
    return ControlFamily(schema=SCHEMA, arms=arms)


def test_claim_scope_constant_matches_house_value() -> None:
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_reading_digest_is_stable() -> None:
    first = _reading(0.5, 0.5)
    second = _reading(0.5, 0.5)
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64


def test_reading_rejects_out_of_unit_interval() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="unit interval"):
        _reading(1.5, 0.5)
    with pytest.raises(StabilityPlasticityRefusal, match="unit interval"):
        _reading(0.5, -0.1)


def test_reading_rejects_widened_claim_scope() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="claim scope"):
        DualMetricReading(retention=0.5, future_learnability=0.5, claim_scope="a capability was shown")


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="non-vacuous"):
        MatchedCostBudget(params=0, flops=1, replay_samples=1, update_steps=1)


def test_default_family_is_complete_and_ordered() -> None:
    family = build_split_control_family(seed=1)
    assert tuple(arm.control for arm in family.arms) == REQUIRED_CONTROLS


def test_control_family_fails_closed_on_membership_drift() -> None:
    readings = {control: _reading(0.5, 0.5) for control in REQUIRED_CONTROLS}
    arms = (
        ControlArm(control="fresh-init", reading=readings["fresh-init"], matched=_BUDGET),
        ControlArm(control="no-replay", reading=readings["no-replay"], matched=_BUDGET),
    )
    with pytest.raises(StabilityPlasticityRefusal, match="membership or order drift"):
        ControlFamily(schema=SCHEMA, arms=arms)


def test_control_family_fails_closed_on_order_drift() -> None:
    readings = {control: _reading(0.5, 0.5) for control in REQUIRED_CONTROLS}
    reordered = ("no-replay", "fresh-init", "full-retrain", "frozen-core")
    arms = tuple(ControlArm(control=c, reading=readings[c], matched=_BUDGET) for c in reordered)
    with pytest.raises(StabilityPlasticityRefusal, match="membership or order drift"):
        ControlFamily(schema=SCHEMA, arms=arms)


def test_control_family_requires_one_matched_budget() -> None:
    other = MatchedCostBudget(params=1, flops=1, replay_samples=1, update_steps=1)
    readings = {control: _reading(0.5, 0.5) for control in REQUIRED_CONTROLS}
    arms = tuple(
        ControlArm(
            control=c,
            reading=readings[c],
            matched=other if c == "frozen-core" else _BUDGET,
        )
        for c in REQUIRED_CONTROLS
    )
    with pytest.raises(StabilityPlasticityRefusal, match="one matched budget"):
        ControlFamily(schema=SCHEMA, arms=arms)


def test_assert_control_completeness_rejects_drift() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="drifted"):
        assert_control_completeness(("fresh-init", "no-replay"))
    assert_control_completeness(REQUIRED_CONTROLS)


def test_control_rejects_unknown_name() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="unsupported control"):
        ControlArm(control="warm-start", reading=_reading(0.5, 0.5), matched=_BUDGET)


def test_default_contract_is_valid_and_pins_null() -> None:
    contract = default_contract()
    assert contract.axes == DUAL_AXES
    assert contract.prior_null == PRIOR_NULL
    assert contract.digest() == default_contract().digest()


def test_contract_requires_both_axes() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="both axes must be required"):
        StabilityPlasticityContract(
            schema=SCHEMA,
            axes=DUAL_AXES,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            both_axes_required=False,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_contract_requires_matched_cost() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="matched full-system cost"):
        StabilityPlasticityContract(
            schema=SCHEMA,
            axes=DUAL_AXES,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=False,
            both_axes_required=True,
            replication_min=2,
            prior_null=PRIOR_NULL,
        )


def test_contract_rejects_single_replication() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="two independent replications"):
        StabilityPlasticityContract(
            schema=SCHEMA,
            axes=DUAL_AXES,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            both_axes_required=True,
            replication_min=1,
            prior_null=PRIOR_NULL,
        )


def test_contract_rejects_wrong_null() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="p6-stability-plasticity-split"):
        StabilityPlasticityContract(
            schema=SCHEMA,
            axes=DUAL_AXES,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            both_axes_required=True,
            replication_min=2,
            prior_null="some-other-null",
        )


def test_verdict_certifies_a_genuine_joint_win() -> None:
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.60),
            "no-replay": _reading(0.25, 0.55),
            "full-retrain": _reading(0.50, 0.50),
            "frozen-core": _reading(0.70, 0.20),
        }
    )
    candidate = _reading(0.80, 0.70)  # beats best retention 0.70 and best future 0.60
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    assert verdict.both_axes_improved
    assert verdict.certify() is verdict


def test_verdict_refuses_single_axis_win() -> None:
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.85),
            "no-replay": _reading(0.25, 0.80),
            "full-retrain": _reading(0.50, 0.50),
            "frozen-core": _reading(0.92, 0.20),
        }
    )
    candidate = _reading(0.95, 0.45)  # beats retention only; future 0.45 < 0.85
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    assert verdict.only_one_axis_improved
    with pytest.raises(StabilityPlasticityRefusal, match="single-axis win refused"):
        verdict.certify()


def test_verdict_refuses_no_axis_win() -> None:
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.85),
            "no-replay": _reading(0.25, 0.80),
            "full-retrain": _reading(0.55, 0.55),
            "frozen-core": _reading(0.92, 0.20),
        }
    )
    candidate = _reading(0.50, 0.50)
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    with pytest.raises(StabilityPlasticityRefusal, match="no-axis win refused"):
        verdict.certify()


def test_evaluate_refuses_unmatched_cost() -> None:
    family = build_split_control_family(seed=0)
    candidate = simulate_reading(seed=0, mechanism="candidate")
    wrong_budget = MatchedCostBudget(params=1, flops=1, replay_samples=1, update_steps=1)
    with pytest.raises(StabilityPlasticityRefusal, match="matched budget"):
        evaluate_joint_improvement(candidate=candidate, candidate_budget=wrong_budget, controls=family)


def test_verdict_digest_is_stable() -> None:
    first = build_split_verdict(seed=3)
    second = build_split_verdict(seed=3)
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64


def test_verdict_rejects_widened_claim_scope() -> None:
    cmp_r = AxisComparison(axis="retention", candidate_value=0.5, best_control_value=0.4)
    cmp_f = AxisComparison(axis="future_learnability", candidate_value=0.5, best_control_value=0.4)
    with pytest.raises(StabilityPlasticityRefusal, match="claim scope"):
        JointImprovementVerdict(
            schema=SCHEMA,
            retention=cmp_r,
            future_learnability=cmp_f,
            matched_cost_required=True,
            prior_null=PRIOR_NULL,
            claim_scope="capability shown",
        )


def test_toy_reading_is_deterministic_under_seed() -> None:
    first = simulate_reading(seed=11, mechanism="candidate")
    second = simulate_reading(seed=11, mechanism="candidate")
    assert first.payload() == second.payload()


def test_toy_exhibits_the_split_and_null_holds() -> None:
    verdict = build_split_verdict(seed=7)
    assert not verdict.both_axes_improved
    with pytest.raises(StabilityPlasticityRefusal, match=PRIOR_NULL):
        verdict.certify()


def test_toy_rejects_unknown_mechanism() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="unknown mechanism"):
        simulate_reading(seed=0, mechanism="teleport")


def test_activation_gate_is_off_by_default() -> None:
    gate = JointClaimGate()
    assert gate.activation_permitted is False
    verdict = build_split_verdict(seed=1)
    with pytest.raises(StabilityPlasticityRefusal, match="not earned"):
        gate.authorize(verdict)


def test_activation_gate_refuses_without_receipt_even_when_permitted() -> None:
    gate = JointClaimGate(activation_permitted=True)
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.60),
            "no-replay": _reading(0.25, 0.55),
            "full-retrain": _reading(0.50, 0.50),
            "frozen-core": _reading(0.70, 0.20),
        }
    )
    candidate = _reading(0.80, 0.70)
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    with pytest.raises(StabilityPlasticityRefusal, match="external confirmation receipt"):
        gate.authorize(verdict)


def test_activation_gate_opens_with_matching_receipt() -> None:
    gate = JointClaimGate(activation_permitted=True)
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.60),
            "no-replay": _reading(0.25, 0.55),
            "full-retrain": _reading(0.50, 0.50),
            "frozen-core": _reading(0.70, 0.20),
        }
    )
    candidate = _reading(0.80, 0.70)
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    receipt = ConfirmationReceipt(
        preregistration_sha256="a" * 64,
        verdict_digest=verdict.digest(),
        replication_count=3,
        matched_cost_attested=True,
        independent_reviewer="external.replication.lab",
    )
    assert gate.authorize(verdict, receipt) is verdict


def test_receipt_requires_two_replications() -> None:
    with pytest.raises(StabilityPlasticityRefusal, match="two independent replications"):
        ConfirmationReceipt(
            preregistration_sha256="a" * 64,
            verdict_digest="b" * 64,
            replication_count=1,
            matched_cost_attested=True,
            independent_reviewer="external.lab",
        )


def test_gate_refuses_receipt_for_a_different_verdict() -> None:
    gate = JointClaimGate(activation_permitted=True)
    family = _family(
        {
            "fresh-init": _reading(0.20, 0.60),
            "no-replay": _reading(0.25, 0.55),
            "full-retrain": _reading(0.50, 0.50),
            "frozen-core": _reading(0.70, 0.20),
        }
    )
    candidate = _reading(0.80, 0.70)
    verdict = evaluate_joint_improvement(candidate=candidate, candidate_budget=_BUDGET, controls=family)
    receipt = ConfirmationReceipt(
        preregistration_sha256="a" * 64,
        verdict_digest="c" * 64,  # does not match this verdict
        replication_count=2,
        matched_cost_attested=True,
        independent_reviewer="external.lab",
    )
    with pytest.raises(StabilityPlasticityRefusal, match="this exact verdict"):
        gate.authorize(verdict, receipt)


def test_coverage_lists_every_sub_question_with_bullets() -> None:
    cov = coverage()
    assert set(cov) == {
        "stable-core-coexists-with-rapid-adaptation",
        "retention-and-future-learning-improve-jointly",
        "improvement-is-at-matched-cost",
        "replay-is-not-a-complete-theory-of-plasticity",
    }
    for bullets in cov.values():
        assert len(bullets) >= 2
