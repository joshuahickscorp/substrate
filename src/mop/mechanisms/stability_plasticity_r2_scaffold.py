from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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

STABILITY_PLASTICITY_R2_SCHEMA = "mop-stability-plasticity-r2/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

PRIOR_NULL = "p6-stability-plasticity-split"

DUAL_AXES: tuple[str, ...] = ("retention", "future_learnability")

REQUIRED_CONTROLS: tuple[str, ...] = ("fresh_init", "frozen_core", "full_retrain", "no_replay")


class StabilityPlasticityR2Refusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DualMetricReading(DualMetricReadingBase):
    retention: float
    future_learnability: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = STABILITY_PLASTICITY_R2_SCHEMA
    axes = DUAL_AXES
    expected_schema = STABILITY_PLASTICITY_R2_SCHEMA
    refusal = StabilityPlasticityR2Refusal


@dataclass(frozen=True, slots=True)
class MatchedCostBudget(MatchedCostBudgetBase):
    params: int
    flops: int
    replay_samples: int
    update_steps: int
    budget_fields = ("params", "flops", "replay_samples", "update_steps")
    refusal = StabilityPlasticityR2Refusal


class ControlArm(ControlArmBase):
    __slots__ = ()
    controls = REQUIRED_CONTROLS
    refusal = StabilityPlasticityR2Refusal


class ControlFamily(ControlFamilyBase):
    __slots__ = ()
    expected_schema = STABILITY_PLASTICITY_R2_SCHEMA
    controls = REQUIRED_CONTROLS
    axes = DUAL_AXES
    order_error = (
        "control family membership or order drift; expected fresh_init, frozen_core, full_retrain, "
        "no_replay in that order"
    )
    refusal = StabilityPlasticityR2Refusal


def assert_control_completeness(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise StabilityPlasticityR2Refusal("declared control set drifted in membership or order")


class StabilityPlasticityR2Contract(JointAxisContractBase):
    __slots__ = ()
    expected_schema = STABILITY_PLASTICITY_R2_SCHEMA
    expected_axes = DUAL_AXES
    expected_controls = REQUIRED_CONTROLS
    expected_prior_null = PRIOR_NULL
    axes_error = "axis set or order drift; both retention and future learnability required"
    matched_cost_error = "stability vs plasticity claim must require matched full-system cost"
    both_axes_error = "a single-axis win is exactly the P6 split; both axes must be required"
    prior_null_error = "contract must name the p6-stability-plasticity-split as its null"
    refusal = StabilityPlasticityR2Refusal


def default_contract() -> StabilityPlasticityR2Contract:

    return StabilityPlasticityR2Contract(
        schema=STABILITY_PLASTICITY_R2_SCHEMA,
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
    refusal = StabilityPlasticityR2Refusal


@dataclass(frozen=True, slots=True)
class JointImprovementVerdict(JointImprovementVerdictBase):
    schema: str
    retention: AxisComparison
    future_learnability: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    expected_schema = STABILITY_PLASTICITY_R2_SCHEMA
    axis_fields = DUAL_AXES
    axis_labels = ("retention", "future learnability")
    expected_prior_null = PRIOR_NULL
    first_axis_error = "first comparison must be the retention axis"
    second_axis_error = "second comparison must be the future learnability axis"
    prior_null_error = "verdict must name the p6-stability-plasticity-split as its null"
    refusal = StabilityPlasticityR2Refusal


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
        schema=STABILITY_PLASTICITY_R2_SCHEMA,
        comparison_type=AxisComparison,
        verdict_type=JointImprovementVerdict,
        prior_null=PRIOR_NULL,
        refusal=StabilityPlasticityR2Refusal,
    )


class ConfirmationReceipt(ConfirmationReceiptBase):
    __slots__ = ()
    refusal = StabilityPlasticityR2Refusal


class JointClaimGate(JointClaimGateBase):
    __slots__ = ()
    refusal = StabilityPlasticityR2Refusal


SCIENTIFIC_CAPABILITY_CLAIM = False
