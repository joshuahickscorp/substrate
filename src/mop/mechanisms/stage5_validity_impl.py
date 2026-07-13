"""Deterministic validity evaluator for the Stage 5 session-disjoint validity harness.

This module raises the SCAFFOLDING axis only. It reads a seeded regime from the bed and recomputes,
per disjointness axis, whether the axis cleared its pass threshold while staying disjoint from
calibration; per leak control, whether the control reproduced the result; and per measured resource,
whether the declared cost matches the measured cost within a relative tolerance. It returns a
verdict record. It certifies nothing and mints no receipt; the runner turns the verdict into a
mechanics-only demonstration.

A regime passes only when every axis passes, no leak control reproduces, and every measured resource
is backed within tolerance. Any single failing axis, any reproducing leak control, or any declared
cost that disagrees with its measured cost beyond tolerance makes the regime fail closed.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim.

House style: no em or en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256
from .stage5_validity_bed import AxisSample, ControlSample, RegimeEvidence, ResourceSample

STAGE5_VALIDITY_EVAL_SCHEMA = "mop-stage5-validity-eval/v1"

# An axis passes when its separation statistic clears this threshold and it is disjoint from
# calibration. A leak control reproduces when its statistic clears this threshold.
AXIS_PASS_THRESHOLD = 0.5
LEAK_REPRODUCE_THRESHOLD = 0.5
# The relative tolerance a declared resource cost may differ from its measured cost.
DEFAULT_RTOL = 0.05


class Stage5ValidityEvalRefusal(ValueError):
    """Raised when the evaluator is handed a tolerance outside the legal range."""


@dataclass(frozen=True, slots=True)
class AxisVerdict:
    """The recomputed verdict for one disjointness axis."""

    axis: str
    statistic: float
    threshold: float
    disjoint_from_calibration: bool
    passed: bool

    def payload(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "disjoint_from_calibration": self.disjoint_from_calibration,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ControlVerdict:
    """The recomputed verdict for one leak control."""

    control: str
    statistic: float
    threshold: float
    reproduced: bool

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "reproduced": self.reproduced,
        }


@dataclass(frozen=True, slots=True)
class EfficiencyVerdict:
    """The recomputed verdict for one measured resource, comparing declared vs measured cost."""

    kind: str
    declared: float
    measured: float
    rtol: float
    within_tolerance: bool

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "declared": self.declared,
            "measured": self.measured,
            "rtol": self.rtol,
            "within_tolerance": self.within_tolerance,
        }


def check_axis(sample: AxisSample, *, threshold: float = AXIS_PASS_THRESHOLD) -> AxisVerdict:
    """Recompute whether one axis passes: statistic clears the threshold and it is disjoint."""

    passed = sample.statistic >= threshold and sample.disjoint_from_calibration
    return AxisVerdict(
        axis=sample.axis,
        statistic=sample.statistic,
        threshold=threshold,
        disjoint_from_calibration=sample.disjoint_from_calibration,
        passed=passed,
    )


def check_control(
    sample: ControlSample, *, threshold: float = LEAK_REPRODUCE_THRESHOLD
) -> ControlVerdict:
    """Recompute whether one leak control reproduced: statistic clears the reproduce threshold."""

    reproduced = sample.statistic >= threshold
    return ControlVerdict(
        control=sample.control,
        statistic=sample.statistic,
        threshold=threshold,
        reproduced=reproduced,
    )


def check_efficiency(sample: ResourceSample, *, rtol: float) -> EfficiencyVerdict:
    """Recompute whether one declared cost matches its measured cost within a relative tolerance."""

    within = abs(sample.declared - sample.measured) <= rtol * sample.measured
    return EfficiencyVerdict(
        kind=sample.kind,
        declared=sample.declared,
        measured=sample.measured,
        rtol=rtol,
        within_tolerance=within,
    )


@dataclass(frozen=True, slots=True)
class ValidityEvaluation:
    """The composite verdict over one regime: every axis, every leak control, every resource."""

    regime: str
    rtol: float
    axis_verdicts: tuple[AxisVerdict, ...]
    control_verdicts: tuple[ControlVerdict, ...]
    efficiency_verdicts: tuple[EfficiencyVerdict, ...]
    schema: str = STAGE5_VALIDITY_EVAL_SCHEMA

    def failing_axes(self) -> tuple[str, ...]:
        return tuple(v.axis for v in self.axis_verdicts if not v.passed)

    def all_axes_pass(self) -> bool:
        return bool(self.axis_verdicts) and not self.failing_axes()

    def reproducing_controls(self) -> tuple[str, ...]:
        return tuple(v.control for v in self.control_verdicts if v.reproduced)

    def any_control_reproduced(self) -> bool:
        return bool(self.reproducing_controls())

    def clean_controls(self) -> tuple[str, ...]:
        return tuple(v.control for v in self.control_verdicts if not v.reproduced)

    def mismatching_resources(self) -> tuple[str, ...]:
        return tuple(v.kind for v in self.efficiency_verdicts if not v.within_tolerance)

    def efficiency_matches(self) -> bool:
        return bool(self.efficiency_verdicts) and not self.mismatching_resources()

    def passed(self) -> bool:
        return (
            self.all_axes_pass()
            and not self.any_control_reproduced()
            and self.efficiency_matches()
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "rtol": self.rtol,
            "axis_verdicts": [v.payload() for v in self.axis_verdicts],
            "control_verdicts": [v.payload() for v in self.control_verdicts],
            "efficiency_verdicts": [v.payload() for v in self.efficiency_verdicts],
            "passed": self.passed(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def evaluate_regime(regime: RegimeEvidence, *, rtol: float = DEFAULT_RTOL) -> ValidityEvaluation:
    """Evaluate every axis, every leak control, and the measured efficiency of one regime."""

    if not 0.0 < rtol <= 0.5:
        raise Stage5ValidityEvalRefusal("efficiency tolerance rtol must be in (0, 0.5]")
    axis_verdicts = tuple(check_axis(sample) for sample in regime.axes)
    control_verdicts = tuple(check_control(sample) for sample in regime.controls)
    efficiency_verdicts = tuple(check_efficiency(sample, rtol=rtol) for sample in regime.resources)
    return ValidityEvaluation(
        regime=regime.regime,
        rtol=rtol,
        axis_verdicts=axis_verdicts,
        control_verdicts=control_verdicts,
        efficiency_verdicts=efficiency_verdicts,
    )
