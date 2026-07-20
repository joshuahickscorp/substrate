from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..substrate.events import canonical_sha256
from .joint_axis_runner import (
    AxisComparisonBase,
    ConfirmationReceiptBase,
    ControlArmBase,
    ControlFamilyBase,
    DualMetricReadingBase,
    JointAxisContractBase,
    JointClaimGateBase,
    JointImprovementVerdictBase,
    MatchedCostBudgetBase,
    evaluate_joint_axes,
)

CALIBRATED_UNCERTAINTY_SCHEMA = "mop-calibrated-uncertainty/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

PRIOR_NULL = "u1-decoupled-confidence-null"

DUAL_AXES: tuple[str, ...] = ("selective_risk_reduction", "decision_utility")

REQUIRED_CONTROLS: tuple[str, ...] = (
    "always_answer",
    "random_abstain",
    "overconfident_score",
    "frozen_uniform",
)


class CalibratedUncertaintyRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DualMetricReading(DualMetricReadingBase):
    selective_risk_reduction: float
    decision_utility: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = CALIBRATED_UNCERTAINTY_SCHEMA
    axes = DUAL_AXES
    expected_schema = CALIBRATED_UNCERTAINTY_SCHEMA
    refusal = CalibratedUncertaintyRefusal


@dataclass(frozen=True, slots=True)
class MatchedCostBudget(MatchedCostBudgetBase):
    params: int
    flops: int
    scored_items: int
    decision_steps: int
    budget_fields = ("params", "flops", "scored_items", "decision_steps")
    refusal = CalibratedUncertaintyRefusal


class ControlArm(ControlArmBase):
    __slots__ = ()
    controls = REQUIRED_CONTROLS
    refusal = CalibratedUncertaintyRefusal


class ControlFamily(ControlFamilyBase):
    __slots__ = ()
    expected_schema = CALIBRATED_UNCERTAINTY_SCHEMA
    controls = REQUIRED_CONTROLS
    axes = DUAL_AXES
    order_error = (
        "control family membership or order drift; expected always_answer, random_abstain, "
        "overconfident_score, frozen_uniform in that order"
    )
    refusal = CalibratedUncertaintyRefusal


def assert_control_completeness(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise CalibratedUncertaintyRefusal("declared control set drifted in membership or order")


class CalibratedUncertaintyContract(JointAxisContractBase):
    __slots__ = ()
    expected_schema = CALIBRATED_UNCERTAINTY_SCHEMA
    expected_axes = DUAL_AXES
    expected_controls = REQUIRED_CONTROLS
    expected_prior_null = PRIOR_NULL
    axes_error = "axis set or order drift; both selective risk reduction and decision utility required"
    matched_cost_error = "calibrated uncertainty claim must require matched full-system cost"
    both_axes_error = "a single-axis win is exactly the decoupled confidence null; both axes must be required"
    prior_null_error = "contract must name the u1-decoupled-confidence-null as its null"
    refusal = CalibratedUncertaintyRefusal


def default_contract() -> CalibratedUncertaintyContract:

    return CalibratedUncertaintyContract(
        schema=CALIBRATED_UNCERTAINTY_SCHEMA,
        axes=DUAL_AXES,
        controls=REQUIRED_CONTROLS,
        matched_cost_required=True,
        both_axes_required=True,
        replication_min=2,
        prior_null=PRIOR_NULL,
    )


class AxisComparison(AxisComparisonBase):
    __slots__ = ()
    axes = DUAL_AXES
    refusal = CalibratedUncertaintyRefusal


@dataclass(frozen=True, slots=True)
class JointImprovementVerdict(JointImprovementVerdictBase):
    schema: str
    selective_risk_reduction: AxisComparison
    decision_utility: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    expected_schema = CALIBRATED_UNCERTAINTY_SCHEMA
    axis_fields = DUAL_AXES
    axis_labels = ("selective risk reduction", "decision utility")
    expected_prior_null = PRIOR_NULL
    first_axis_error = "first comparison must be the selective risk axis"
    second_axis_error = "second comparison must be the decision utility axis"
    prior_null_error = "verdict must name the u1-decoupled-confidence-null as its null"
    refusal = CalibratedUncertaintyRefusal


def evaluate_joint_improvement(
    *,
    candidate: DualMetricReading,
    candidate_budget: MatchedCostBudget,
    controls: ControlFamily,
) -> JointImprovementVerdict:

    return evaluate_joint_axes(
        candidate=candidate,
        candidate_budget=candidate_budget,
        controls=controls,
        axes=DUAL_AXES,
        schema=CALIBRATED_UNCERTAINTY_SCHEMA,
        comparison_type=AxisComparison,
        verdict_type=JointImprovementVerdict,
        prior_null=PRIOR_NULL,
        refusal=CalibratedUncertaintyRefusal,
    )


_BASE_SCORES: dict[str, tuple[float, float]] = {
    "always_answer": (0.75, 0.75),
    "random_abstain": (0.74, 0.62),
    "overconfident_score": (0.70, 0.73),
    "frozen_uniform": (0.05, 0.50),
}
_CANDIDATE_BASE: tuple[float, float] = (0.85, 0.68)
_DEFAULT_BUDGET = MatchedCostBudget(params=1024, flops=262_144, scored_items=32, decision_steps=32)


def _seeded_jitter(seed: int, label: str) -> float:

    if seed < 0:
        raise CalibratedUncertaintyRefusal("toy seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 0.01


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def simulate_reading(*, seed: int, mechanism: str) -> DualMetricReading:

    if mechanism == "candidate":
        base = _CANDIDATE_BASE
    elif mechanism in _BASE_SCORES:
        base = _BASE_SCORES[mechanism]
    else:
        raise CalibratedUncertaintyRefusal(f"unknown mechanism {mechanism!r}")
    risk = _clamp_unit(base[0] + _seeded_jitter(seed, f"{mechanism}.selective_risk_reduction"))
    utility = _clamp_unit(base[1] + _seeded_jitter(seed, f"{mechanism}.decision_utility"))
    return DualMetricReading(selective_risk_reduction=risk, decision_utility=utility)


def build_null_control_family(*, seed: int) -> ControlFamily:

    arms = tuple(
        ControlArm(
            control=control,
            reading=simulate_reading(seed=seed, mechanism=control),
            matched=_DEFAULT_BUDGET,
        )
        for control in REQUIRED_CONTROLS
    )
    return ControlFamily(schema=CALIBRATED_UNCERTAINTY_SCHEMA, arms=arms)


def build_null_verdict(*, seed: int) -> JointImprovementVerdict:

    controls = build_null_control_family(seed=seed)
    candidate = simulate_reading(seed=seed, mechanism="candidate")
    return evaluate_joint_improvement(
        candidate=candidate, candidate_budget=_DEFAULT_BUDGET, controls=controls
    )


class ConfirmationReceipt(ConfirmationReceiptBase):
    __slots__ = ()
    refusal = CalibratedUncertaintyRefusal


class JointClaimGate(JointClaimGateBase):
    __slots__ = ()
    refusal = CalibratedUncertaintyRefusal


SCIENTIFIC_CAPABILITY_CLAIM = False
