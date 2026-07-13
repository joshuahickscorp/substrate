"""Deterministic seeded bed for the Stage 5 session-disjoint general-validity harness.

This module raises the SCAFFOLDING axis only. It supplies a deterministic, seeded task environment
that produces per-axis disjointness evidence and per-control leak evidence, a measured-resource
block, and a non-vacuous matched full-system budget. It builds the evidence the validity evaluator
reads; it never certifies generality and never mints a receipt.

Two canonical regimes are exposed. The favorable regime carries evidence in which every disjointness
axis clears its pass band and every leak control stays under its reproduce band. The null regime
carries evidence in which at least one axis falls under its pass band and at least one leak control
reproduces the result, so the session-bound-leakage prior null explains it. A third candidate regime
mirrors the favorable regime but folds in any defect the bed was constructed with (a failing axis, a
reproducing leak control, a declared-versus-measured efficiency mismatch), so the fail-closed paths
of the runner can be exercised without editing any evidence in place.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. The statistics are seeded test vectors, not measurements of value.

House style: no em or en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256

STAGE5_VALIDITY_BED_SCHEMA = "mop-stage5-validity-bed/v1"

# Must stay byte-identical to the receipt mechanism id the runner mints under. Underscores are legal
# for the ladder id grammar; this id names the bed and every receipt it feeds.
MECHANISM_ID = "stage5_session_disjoint_validity"

# The disjointness axes a general-validity claim must clear. Order is load-bearing for the digests.
VALIDITY_AXES: tuple[str, ...] = (
    "fresh-session",
    "new-seeds",
    "new-task-families",
    "lesions",
    "independent-reconstruction",
)

# The leak controls that must stay clean; if any reproduces, the prior null explains the result.
LEAK_CONTROLS: tuple[str, ...] = (
    "same-session-leak",
    "seed-reuse",
    "single-task-family",
)

# The measured compute resources carried in every regime. Order is load-bearing for the digests.
RESOURCE_KINDS: tuple[str, ...] = (
    "wall_time_s",
    "flops",
)

# The canonical null regime fails one axis and reproduces one leak control.
NULL_FAILING_AXIS = "new-task-families"
NULL_REPRODUCING_CONTROL = "same-session-leak"

# Statistic bands. A passing axis clears the pass floor; a failing axis stays under the fail ceiling.
# A clean control stays under the fail ceiling; a reproducing control clears the pass floor. The
# thresholds the evaluator applies sit between these bands, so bed and evaluator stay consistent.
_PASS_FLOOR = 0.55
_FAIL_CEILING = 0.44
_MISMATCH_FACTOR = 2.0
_TWO_POW_32 = float(0x1_0000_0000)


class Stage5ValidityBedRefusal(ValueError):
    """Raised whenever a regime is asked for an unknown axis, leak control, or resource kind."""


def _unit_fraction(seed: int, label: str) -> float:
    """Deterministic fraction in [0, 1) from a digest. No wall clock, no OS entropy."""

    digest = canonical_sha256({"seed": seed, "label": label, "schema": STAGE5_VALIDITY_BED_SCHEMA})
    return int(digest[:8], 16) / _TWO_POW_32


def _seeded_int(seed: int, label: str, modulo: int) -> int:
    """Deterministic small integer from a digest. No wall clock, no OS entropy."""

    if modulo <= 0:
        raise Stage5ValidityBedRefusal("modulo must be positive")
    digest = canonical_sha256({"seed": seed, "label": label, "schema": STAGE5_VALIDITY_BED_SCHEMA})
    return int(digest[8:16], 16) % modulo


@dataclass(frozen=True, slots=True)
class AxisSample:
    """One disjointness axis of seeded evidence. The evaluator recomputes the pass verdict."""

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
    """One leak-control sample of seeded evidence. The evaluator recomputes the reproduce verdict."""

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
    """One measured resource whose declared cost is compared against its measured cost."""

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
    """A full seeded regime: every axis, every leak control, the resource block, and the budget."""

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
    """A deterministic, seeded bed for the Stage 5 session-disjoint validity harness.

    A bed carries an optional defect set (failing axes, reproducing leak controls, resource kinds
    whose measured cost drifts from the declared cost). A bed with no defects yields a favorable
    candidate regime in which every check passes; a bed with any defect yields a candidate regime
    that fails closed. The favorable and null regimes are always the pure canonical regimes.
    """

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
        """Return a non-vacuous matched full-system budget seeded from the bed base seed."""

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
        """The pure favorable regime: every axis passes, every leak control stays clean."""

        return self._regime(seed, "favorable", (), (), ())

    def null_regime(self, seed: int) -> RegimeEvidence:
        """The pure null regime: one axis falls under the pass band and one leak control reproduces."""

        return self._regime(
            seed, "null", (NULL_FAILING_AXIS,), (NULL_REPRODUCING_CONTROL,), ()
        )

    def candidate_regime(self, seed: int) -> RegimeEvidence:
        """The regime the runner evaluates: favorable, plus whatever defect the bed was built with."""

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
    """Return a clean bed whose candidate regime is favorable for every seed."""

    return Stage5ValidityBed(base_seed=seed)


def build_null_bed(seed: int = 0) -> Stage5ValidityBed:
    """Return a bed whose candidate regime carries the canonical null defect set."""

    return Stage5ValidityBed(
        base_seed=seed,
        failing_axes=(NULL_FAILING_AXIS,),
        reproducing_controls=(NULL_REPRODUCING_CONTROL,),
    )
