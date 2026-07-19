
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256

STAGE5_VALIDITY_BED_SCHEMA = "mop-stage5-validity-bed/v1"

MECHANISM_ID = "stage5_session_disjoint_validity"

VALIDITY_AXES: tuple[str, ...] = (
    "fresh-session",
    "new-seeds",
    "new-task-families",
    "lesions",
    "independent-reconstruction",
)

LEAK_CONTROLS: tuple[str, ...] = (
    "same-session-leak",
    "seed-reuse",
    "single-task-family",
)

RESOURCE_KINDS: tuple[str, ...] = (
    "wall_time_s",
    "flops",
)

NULL_FAILING_AXIS = "new-task-families"
NULL_REPRODUCING_CONTROL = "same-session-leak"

_PASS_FLOOR = 0.55
_FAIL_CEILING = 0.44
_MISMATCH_FACTOR = 2.0
_TWO_POW_32 = float(0x1_0000_0000)


class Stage5ValidityBedRefusal(ValueError):
    pass


def _unit_fraction(seed: int, label: str) -> float:

    digest = canonical_sha256({"seed": seed, "label": label, "schema": STAGE5_VALIDITY_BED_SCHEMA})
    return int(digest[:8], 16) / _TWO_POW_32


def _seeded_int(seed: int, label: str, modulo: int) -> int:

    if modulo <= 0:
        raise Stage5ValidityBedRefusal("modulo must be positive")
    digest = canonical_sha256({"seed": seed, "label": label, "schema": STAGE5_VALIDITY_BED_SCHEMA})
    return int(digest[8:16], 16) % modulo


@dataclass(frozen=True, slots=True)
class AxisSample:

    axis: str
    disjoint_from_calibration: bool
    replications: int
    statistic: float
    evidence_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "disjoint_from_calibration": self.disjoint_from_calibration,
            "replications": self.replications,
            "statistic": self.statistic,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class ControlSample:

    control: str
    statistic: float
    evidence_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "statistic": self.statistic,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResourceSample:

    kind: str
    declared: float
    measured: float
    unit: str
    measurement_source: str

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "declared": self.declared,
            "measured": self.measured,
            "unit": self.unit,
            "measurement_source": self.measurement_source,
        }


@dataclass(frozen=True, slots=True)
class RegimeEvidence:

    regime: str
    axes: tuple[AxisSample, ...]
    controls: tuple[ControlSample, ...]
    resources: tuple[ResourceSample, ...]
    matched: MatchedBudget
    schema: str = STAGE5_VALIDITY_BED_SCHEMA

    def axis(self, name: str) -> AxisSample:
        for sample in self.axes:
            if sample.axis == name:
                return sample
        raise Stage5ValidityBedRefusal(f"unknown validity axis {name!r}")

    def control(self, name: str) -> ControlSample:
        for sample in self.controls:
            if sample.control == name:
                return sample
        raise Stage5ValidityBedRefusal(f"unknown leak control {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "axes": [a.payload() for a in self.axes],
            "controls": [c.payload() for c in self.controls],
            "resources": [r.payload() for r in self.resources],
            "matched": self.matched.payload(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class Stage5ValidityBed:

    base_seed: int = 0
    failing_axes: tuple[str, ...] = ()
    reproducing_controls: tuple[str, ...] = ()
    mismatch_kinds: tuple[str, ...] = ()
    mechanism_id: str = MECHANISM_ID
    schema: str = STAGE5_VALIDITY_BED_SCHEMA

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise Stage5ValidityBedRefusal("bed base seed must be nonnegative")
        for axis in self.failing_axes:
            if axis not in VALIDITY_AXES:
                raise Stage5ValidityBedRefusal(f"unknown validity axis {axis!r}")
        for control in self.reproducing_controls:
            if control not in LEAK_CONTROLS:
                raise Stage5ValidityBedRefusal(f"unknown leak control {control!r}")
        for kind in self.mismatch_kinds:
            if kind not in RESOURCE_KINDS:
                raise Stage5ValidityBedRefusal(f"unknown resource kind {kind!r}")

    def controls(self) -> tuple[str, ...]:
        return LEAK_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(
            params=4096 + _seeded_int(self.base_seed, "params", 4096),
            flops=2_000_000 + _seeded_int(self.base_seed, "flops", 1_000_000),
            wall_ns=500_000 + _seeded_int(self.base_seed, "wall_ns", 500_000),
            seeds=4 + _seeded_int(self.base_seed, "seeds", 12),
        )

    def _axis_evidence(self, seed: int, axis: str) -> str:
        return canonical_sha256({"seed": seed, "axis": axis, "kind": "axis", "schema": self.schema})

    def _control_evidence(self, seed: int, control: str) -> str:
        return canonical_sha256(
            {"seed": seed, "control": control, "kind": "control", "schema": self.schema}
        )

    def _axis_sample(self, seed: int, axis: str, *, failing: bool) -> AxisSample:
        fraction = _unit_fraction(seed, f"axis:{axis}")
        if failing:
            statistic = _FAIL_CEILING * fraction
        else:
            statistic = _PASS_FLOOR + (1.0 - _PASS_FLOOR) * fraction
        return AxisSample(
            axis=axis,
            disjoint_from_calibration=not failing,
            replications=2 + _seeded_int(seed, f"reps:{axis}", 4),
            statistic=statistic,
            evidence_sha256=self._axis_evidence(seed, axis),
        )

    def _control_sample(self, seed: int, control: str, *, reproducing: bool) -> ControlSample:
        fraction = _unit_fraction(seed, f"control:{control}")
        if reproducing:
            statistic = _PASS_FLOOR + (1.0 - _PASS_FLOOR) * fraction
        else:
            statistic = _FAIL_CEILING * fraction
        return ControlSample(
            control=control,
            statistic=statistic,
            evidence_sha256=self._control_evidence(seed, control),
        )

    def _resource_sample(self, seed: int, kind: str, *, mismatched: bool) -> ResourceSample:
        if kind == "wall_time_s":
            declared = 1.0 + _seeded_int(seed, "wall_time_s", 500) / 100.0
            unit = "seconds"
            source = "perf_counter over the held-out session run"
        else:
            declared = float(1_000_000 + _seeded_int(seed, "flops", 1_000_000))
            unit = "flop"
            source = "hardware performance counter tally"
        measured = declared * _MISMATCH_FACTOR if mismatched else declared
        return ResourceSample(
            kind=kind,
            declared=declared,
            measured=measured,
            unit=unit,
            measurement_source=source,
        )

    def _regime(
        self,
        seed: int,
        label: str,
        failing_axes: tuple[str, ...],
        reproducing_controls: tuple[str, ...],
        mismatch_kinds: tuple[str, ...],
    ) -> RegimeEvidence:
        axes = tuple(
            self._axis_sample(seed, axis, failing=axis in failing_axes) for axis in VALIDITY_AXES
        )
        controls = tuple(
            self._control_sample(seed, control, reproducing=control in reproducing_controls)
            for control in LEAK_CONTROLS
        )
        resources = tuple(
            self._resource_sample(seed, kind, mismatched=kind in mismatch_kinds)
            for kind in RESOURCE_KINDS
        )
        return RegimeEvidence(
            regime=label,
            axes=axes,
            controls=controls,
            resources=resources,
            matched=self.matched_cost(),
        )

    def favorable_regime(self, seed: int) -> RegimeEvidence:

        return self._regime(seed, "favorable", (), (), ())

    def null_regime(self, seed: int) -> RegimeEvidence:

        return self._regime(
            seed, "null", (NULL_FAILING_AXIS,), (NULL_REPRODUCING_CONTROL,), ()
        )

    def candidate_regime(self, seed: int) -> RegimeEvidence:

        return self._regime(
            seed,
            "candidate",
            self.failing_axes,
            self.reproducing_controls,
            self.mismatch_kinds,
        )

    def with_failing_axis(self, axis: str) -> Stage5ValidityBed:
        if axis not in VALIDITY_AXES:
            raise Stage5ValidityBedRefusal(f"unknown validity axis {axis!r}")
        return replace(self, failing_axes=self.failing_axes + (axis,))

    def with_reproducing_control(self, control: str) -> Stage5ValidityBed:
        if control not in LEAK_CONTROLS:
            raise Stage5ValidityBedRefusal(f"unknown leak control {control!r}")
        return replace(self, reproducing_controls=self.reproducing_controls + (control,))

    def with_efficiency_mismatch(self, kind: str) -> Stage5ValidityBed:
        if kind not in RESOURCE_KINDS:
            raise Stage5ValidityBedRefusal(f"unknown resource kind {kind!r}")
        return replace(self, mismatch_kinds=self.mismatch_kinds + (kind,))


def build_bed(seed: int = 0) -> Stage5ValidityBed:

    return Stage5ValidityBed(base_seed=seed)


def build_null_bed(seed: int = 0) -> Stage5ValidityBed:

    return Stage5ValidityBed(
        base_seed=seed,
        failing_axes=(NULL_FAILING_AXIS,),
        reproducing_controls=(NULL_REPRODUCING_CONTROL,),
    )
