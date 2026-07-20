
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256
from .stage5_validity_bed import AxisSample, ControlSample, RegimeEvidence, ResourceSample

STAGE5_VALIDITY_EVAL_SCHEMA = "mop-stage5-validity-eval/v1"

AXIS_PASS_THRESHOLD = 0.5
LEAK_REPRODUCE_THRESHOLD = 0.5
DEFAULT_RTOL = 0.05


class Stage5ValidityEvalRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AxisVerdict:

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

    reproduced = sample.statistic >= threshold
    return ControlVerdict(
        control=sample.control,
        statistic=sample.statistic,
        threshold=threshold,
        reproduced=reproduced,
    )


def check_efficiency(sample: ResourceSample, *, rtol: float) -> EfficiencyVerdict:

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
