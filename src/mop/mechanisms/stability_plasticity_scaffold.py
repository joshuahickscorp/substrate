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

STABILITY_PLASTICITY_SCHEMA = "mop-stability-plasticity/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

PRIOR_NULL = "p6-stability-plasticity-split"

DUAL_AXES: tuple[str, ...] = ("retention", "future_learnability")

REQUIRED_CONTROLS: tuple[str, ...] = ("fresh-init", "no-replay", "full-retrain", "frozen-core")


class StabilityPlasticityRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DualMetricReading(DualMetricReadingBase):
    retention: float
    future_learnability: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = STABILITY_PLASTICITY_SCHEMA
    axes = DUAL_AXES
    expected_schema = STABILITY_PLASTICITY_SCHEMA
    refusal = StabilityPlasticityRefusal


@dataclass(frozen=True, slots=True)
class MatchedCostBudget(MatchedCostBudgetBase):
    params: int
    flops: int
    replay_samples: int
    update_steps: int
    budget_fields = ("params", "flops", "replay_samples", "update_steps")
    refusal = StabilityPlasticityRefusal


class ControlArm(ControlArmBase):
    __slots__ = ()
    controls = REQUIRED_CONTROLS
    refusal = StabilityPlasticityRefusal


class ControlFamily(ControlFamilyBase):
    __slots__ = ()
    expected_schema = STABILITY_PLASTICITY_SCHEMA
    controls = REQUIRED_CONTROLS
    axes = DUAL_AXES
    order_error = (
        "control family membership or order drift; expected fresh-init, no-replay, full-retrain, "
        "frozen-core in that order"
    )
    refusal = StabilityPlasticityRefusal


def assert_control_completeness(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise StabilityPlasticityRefusal("declared control set drifted in membership or order")


class StabilityPlasticityContract(JointAxisContractBase):
    __slots__ = ()
    expected_schema = STABILITY_PLASTICITY_SCHEMA
    expected_axes = DUAL_AXES
    expected_controls = REQUIRED_CONTROLS
    expected_prior_null = PRIOR_NULL
    axes_error = "axis set or order drift; both retention and future learnability required"
    matched_cost_error = "stability vs plasticity claim must require matched full-system cost"
    both_axes_error = "a single-axis win is exactly the P6 split; both axes must be required"
    prior_null_error = "contract must name the p6-stability-plasticity-split as its null"
    refusal = StabilityPlasticityRefusal


def default_contract() -> StabilityPlasticityContract:

    return StabilityPlasticityContract(
        schema=STABILITY_PLASTICITY_SCHEMA,
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
    refusal = StabilityPlasticityRefusal


@dataclass(frozen=True, slots=True)
class JointImprovementVerdict(JointImprovementVerdictBase):
    schema: str
    retention: AxisComparison
    future_learnability: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    expected_schema = STABILITY_PLASTICITY_SCHEMA
    axis_fields = DUAL_AXES
    axis_labels = ("retention", "future learnability")
    expected_prior_null = PRIOR_NULL
    first_axis_error = "first comparison must be the retention axis"
    second_axis_error = "second comparison must be the future learnability axis"
    prior_null_error = "verdict must name the p6-stability-plasticity-split as its null"
    refusal = StabilityPlasticityRefusal


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
        schema=STABILITY_PLASTICITY_SCHEMA,
        comparison_type=AxisComparison,
        verdict_type=JointImprovementVerdict,
        prior_null=PRIOR_NULL,
        refusal=StabilityPlasticityRefusal,
    )


_BASE_SCORES: dict[str, tuple[float, float]] = {
    "fresh-init": (0.20, 0.85),
    "no-replay": (0.26, 0.80),
    "full-retrain": (0.55, 0.55),
    "frozen-core": (0.92, 0.20),
}
_CANDIDATE_BASE: tuple[float, float] = (0.86, 0.45)
_DEFAULT_BUDGET = MatchedCostBudget(params=4096, flops=1_048_576, replay_samples=256, update_steps=200)


def _seeded_jitter(seed: int, label: str) -> float:

    if seed < 0:
        raise StabilityPlasticityRefusal("toy seed must be nonnegative")
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
        raise StabilityPlasticityRefusal(f"unknown mechanism {mechanism!r}")
    retention = _clamp_unit(base[0] + _seeded_jitter(seed, f"{mechanism}.retention"))
    future = _clamp_unit(base[1] + _seeded_jitter(seed, f"{mechanism}.future_learnability"))
    return DualMetricReading(retention=retention, future_learnability=future)


def build_split_control_family(*, seed: int) -> ControlFamily:

    arms = tuple(
        ControlArm(
            control=control,
            reading=simulate_reading(seed=seed, mechanism=control),
            matched=_DEFAULT_BUDGET,
        )
        for control in REQUIRED_CONTROLS
    )
    return ControlFamily(schema=STABILITY_PLASTICITY_SCHEMA, arms=arms)


def build_split_verdict(*, seed: int) -> JointImprovementVerdict:

    controls = build_split_control_family(seed=seed)
    candidate = simulate_reading(seed=seed, mechanism="candidate")
    return evaluate_joint_improvement(
        candidate=candidate, candidate_budget=_DEFAULT_BUDGET, controls=controls
    )


class ConfirmationReceipt(ConfirmationReceiptBase):
    __slots__ = ()
    refusal = StabilityPlasticityRefusal


class JointClaimGate(JointClaimGateBase):
    __slots__ = ()
    refusal = StabilityPlasticityRefusal


SCIENTIFIC_CAPABILITY_CLAIM = False
