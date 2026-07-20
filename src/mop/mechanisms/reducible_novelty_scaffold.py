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

REDUCIBLE_NOVELTY_SCHEMA = "mop-reducible-novelty/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

PRIOR_NULL = "irreducible-noise-trap"

DUAL_AXES: tuple[str, ...] = ("learning_progress", "allocation_efficiency")

REQUIRED_CONTROLS: tuple[str, ...] = (
    "uniform_allocation",
    "random_allocation",
    "novelty_chaser",
    "static_prior",
)


class ReducibleNoveltyRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DualMetricReading(DualMetricReadingBase):
    learning_progress: float
    allocation_efficiency: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = REDUCIBLE_NOVELTY_SCHEMA
    axes = DUAL_AXES
    expected_schema = REDUCIBLE_NOVELTY_SCHEMA
    refusal = ReducibleNoveltyRefusal


@dataclass(frozen=True, slots=True)
class MatchedCostBudget(MatchedCostBudgetBase):
    probes: int
    sources: int
    pilot_probes: int
    flops: int
    budget_fields = ("probes", "sources", "pilot_probes", "flops")
    refusal = ReducibleNoveltyRefusal


class ControlArm(ControlArmBase):
    __slots__ = ()
    controls = REQUIRED_CONTROLS
    refusal = ReducibleNoveltyRefusal


class ControlFamily(ControlFamilyBase):
    __slots__ = ()
    expected_schema = REDUCIBLE_NOVELTY_SCHEMA
    controls = REQUIRED_CONTROLS
    axes = DUAL_AXES
    order_error = (
        "control family membership or order drift; expected uniform_allocation, random_allocation, "
        "novelty_chaser, static_prior in that order"
    )
    refusal = ReducibleNoveltyRefusal


def assert_control_completeness(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise ReducibleNoveltyRefusal("declared control set drifted in membership or order")


class ReducibleNoveltyContract(JointAxisContractBase):
    __slots__ = ()
    expected_schema = REDUCIBLE_NOVELTY_SCHEMA
    expected_axes = DUAL_AXES
    expected_controls = REQUIRED_CONTROLS
    expected_prior_null = PRIOR_NULL
    axes_error = "axis set or order drift; both learning progress and allocation efficiency required"
    matched_cost_error = "reducible novelty claim must require matched full-system cost"
    both_axes_error = "a single-axis win is exactly the noise trap; both axes must be required"
    prior_null_error = "contract must name the irreducible-noise-trap as its null"
    refusal = ReducibleNoveltyRefusal


def default_contract() -> ReducibleNoveltyContract:

    return ReducibleNoveltyContract(
        schema=REDUCIBLE_NOVELTY_SCHEMA,
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
    refusal = ReducibleNoveltyRefusal


@dataclass(frozen=True, slots=True)
class JointImprovementVerdict(JointImprovementVerdictBase):
    schema: str
    learning_progress: AxisComparison
    allocation_efficiency: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    expected_schema = REDUCIBLE_NOVELTY_SCHEMA
    axis_fields = DUAL_AXES
    axis_labels = ("learning progress", "allocation efficiency")
    expected_prior_null = PRIOR_NULL
    first_axis_error = "first comparison must be the learning progress axis"
    second_axis_error = "second comparison must be the allocation efficiency axis"
    prior_null_error = "verdict must name the irreducible-noise-trap as its null"
    refusal = ReducibleNoveltyRefusal


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
        schema=REDUCIBLE_NOVELTY_SCHEMA,
        comparison_type=AxisComparison,
        verdict_type=JointImprovementVerdict,
        prior_null=PRIOR_NULL,
        refusal=ReducibleNoveltyRefusal,
    )


_BASE_SCORES: dict[str, tuple[float, float]] = {
    "uniform_allocation": (0.62, 0.50),
    "random_allocation": (0.55, 0.47),
    "novelty_chaser": (0.35, 0.18),
    "static_prior": (0.28, 0.88),
}
_CANDIDATE_BASE: tuple[float, float] = (0.70, 0.60)
_DEFAULT_BUDGET = MatchedCostBudget(probes=40, sources=8, pilot_probes=8, flops=1_048_576)


def _seeded_jitter(seed: int, label: str) -> float:

    if seed < 0:
        raise ReducibleNoveltyRefusal("toy seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    return (unit - 0.5) * 0.01


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def simulate_reading(*, seed: int, policy: str) -> DualMetricReading:

    if policy == "candidate":
        base = _CANDIDATE_BASE
    elif policy in _BASE_SCORES:
        base = _BASE_SCORES[policy]
    else:
        raise ReducibleNoveltyRefusal(f"unknown policy {policy!r}")
    progress = _clamp_unit(base[0] + _seeded_jitter(seed, f"{policy}.learning_progress"))
    efficiency = _clamp_unit(base[1] + _seeded_jitter(seed, f"{policy}.allocation_efficiency"))
    return DualMetricReading(learning_progress=progress, allocation_efficiency=efficiency)


def build_trap_control_family(*, seed: int) -> ControlFamily:

    arms = tuple(
        ControlArm(
            control=control,
            reading=simulate_reading(seed=seed, policy=control),
            matched=_DEFAULT_BUDGET,
        )
        for control in REQUIRED_CONTROLS
    )
    return ControlFamily(schema=REDUCIBLE_NOVELTY_SCHEMA, arms=arms)


def build_trap_verdict(*, seed: int) -> JointImprovementVerdict:

    controls = build_trap_control_family(seed=seed)
    candidate = simulate_reading(seed=seed, policy="candidate")
    return evaluate_joint_improvement(
        candidate=candidate, candidate_budget=_DEFAULT_BUDGET, controls=controls
    )


class ConfirmationReceipt(ConfirmationReceiptBase):
    __slots__ = ()
    refusal = ReducibleNoveltyRefusal


class JointClaimGate(JointClaimGateBase):
    __slots__ = ()
    refusal = ReducibleNoveltyRefusal


SCIENTIFIC_CAPABILITY_CLAIM = False
