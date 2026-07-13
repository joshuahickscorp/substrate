"""Scaffold spine for Stage 5 session-disjoint general validity (the external-validity harness).

This module raises the SCAFFOLDING axis only. It supplies machine-checkable contracts that encode
the bar a general-validity claim must clear: apparent skill has to survive a fresh session, new
seeds, new task families, lesions, and an independent reconstruction, while three negative controls
(same-session leak, seed reuse, single task family) stay clean, and any efficiency claim is backed
by a real measured cost rather than a declared number. It builds the harness, not the result.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. A green contract here means the declaration set is complete and self-consistent.
Nothing here establishes general validity, transfer, or efficiency. The activation gate that would
certify a generality claim is off by default and refuses until a countersigned license is supplied.

Named prior null: session-bound leakage. Until every disjointness axis passes and every negative
control stays clean, apparent general validity is attributed to same-session leakage, seed
memorization, or single task-family overfit. The gate encodes that null as a fail-closed condition.

House style: no em or en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

STAGE5_VALIDITY_SCHEMA = "mop-stage5-session-disjoint-validity/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# The prior null this stage must defeat before any general-validity claim is licensed.
PRIOR_NULL = "session-bound-leakage"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SessionDisjointValidityRefusal(ValueError):
    """Raised whenever a declaration is missing, malformed, or outside its declared scope."""


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise SessionDisjointValidityRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise SessionDisjointValidityRefusal(f"{label} must use stable lowercase characters")


def _require_nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise SessionDisjointValidityRefusal(f"{label} must not be empty")


# Section A. Declared axes and controls. Order is load-bearing for the digests.

# Every disjointness axis a general-validity claim must clear.
VALIDITY_AXES: tuple[str, ...] = (
    "fresh-session",  # a session distinct from every calibration session
    "new-seeds",  # seeds never used during calibration
    "new-task-families",  # task families held out of calibration
    "lesions",  # ablations that should break a leak-driven result
    "independent-reconstruction",  # rebuilt by a disjoint team from the spec alone
)

# Negative controls that must stay clean; if any leaks, the prior null explains the result.
NEGATIVE_CONTROLS: tuple[str, ...] = (
    "same-session-leak",  # reuse the calibration session; must not confer validity
    "seed-reuse",  # reuse calibration seeds; must not confer validity
    "single-task-family",  # a single family only; must not confer general validity
)

# Real resource kinds; at least one compute kind must carry a measured cost.
RESOURCE_KINDS: tuple[str, ...] = (
    "wall_time_s",
    "flops",
    "peak_memory_bytes",
    "energy_joules",
)
COMPUTE_KINDS: frozenset[str] = frozenset({"wall_time_s", "flops"})

# Words that mark a cost as declared rather than measured; a real measurement source is required.
_DECLARED_MARKERS: frozenset[str] = frozenset(
    {"declared", "declared-only", "estimate", "estimated", "guess", "assumed", "planned"}
)


def assert_axis_completeness(axes: tuple[AxisOutcome, ...]) -> None:
    """Fail closed unless the axis set matches VALIDITY_AXES in membership and order."""

    if tuple(a.axis for a in axes) != VALIDITY_AXES:
        raise SessionDisjointValidityRefusal("validity axis coverage is incomplete or out of canonical order")


def assert_control_completeness(controls: tuple[ControlOutcome, ...]) -> None:
    """Fail closed unless the control set matches NEGATIVE_CONTROLS in membership and order."""

    if tuple(c.control for c in controls) != NEGATIVE_CONTROLS:
        raise SessionDisjointValidityRefusal(
            "negative control coverage is incomplete or out of canonical order"
        )


# Section B. Per-axis and per-control outcome declarations.


@dataclass(frozen=True, slots=True)
class AxisOutcome:
    """One disjointness axis outcome; a passing axis must be disjoint from calibration."""

    axis: str
    passed: bool
    replications: int
    disjoint_from_calibration: bool
    evidence_sha256: str
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE5_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE5_VALIDITY_SCHEMA:
            raise SessionDisjointValidityRefusal(f"unsupported axis schema {self.schema!r}")
        if self.axis not in VALIDITY_AXES:
            raise SessionDisjointValidityRefusal(f"unsupported validity axis {self.axis!r}")
        if self.replications < 2:
            raise SessionDisjointValidityRefusal("each axis needs at least two independent replications")
        _require_sha256(self.evidence_sha256, "AxisOutcome.evidence_sha256")
        if self.passed and not self.disjoint_from_calibration:
            raise SessionDisjointValidityRefusal(
                "a passing axis must be disjoint from the calibration session"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise SessionDisjointValidityRefusal("axis claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "passed": self.passed,
            "replications": self.replications,
            "disjoint_from_calibration": self.disjoint_from_calibration,
            "evidence_sha256": self.evidence_sha256,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """One negative-control outcome; a clean control did not confer spurious validity."""

    control: str
    leaked: bool
    evidence_sha256: str
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE5_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE5_VALIDITY_SCHEMA:
            raise SessionDisjointValidityRefusal(f"unsupported control schema {self.schema!r}")
        if self.control not in NEGATIVE_CONTROLS:
            raise SessionDisjointValidityRefusal(f"unsupported negative control {self.control!r}")
        _require_sha256(self.evidence_sha256, "ControlOutcome.evidence_sha256")
        if self.claim_scope != CLAIM_SCOPE:
            raise SessionDisjointValidityRefusal("control claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "control": self.control,
            "leaked": self.leaked,
            "evidence_sha256": self.evidence_sha256,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# Section C. Matched-cost discipline and measured efficiency.


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:
    """The matched full-system budget a comparison must be tuned to. Must be non-vacuous."""

    params: int
    flops: int
    wall_time_ms: int
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("wall_time_ms", self.wall_time_ms),
            ("peak_memory_bytes", self.peak_memory_bytes),
        ):
            if value <= 0:
                raise SessionDisjointValidityRefusal(
                    f"matched cost budget {name} must be positive (non-vacuous)"
                )

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "wall_time_ms": self.wall_time_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class MeasuredResource:
    """One resource whose declared cost must be backed by a real measurement."""

    kind: str
    declared: float
    measured: float
    unit: str
    measurement_source: str

    def __post_init__(self) -> None:
        if self.kind not in RESOURCE_KINDS:
            raise SessionDisjointValidityRefusal(f"unsupported resource kind {self.kind!r}")
        if self.declared <= 0.0 or self.measured <= 0.0:
            raise SessionDisjointValidityRefusal("resource declared and measured cost must be positive")
        _require_nonempty(self.unit, "MeasuredResource.unit")
        _require_nonempty(self.measurement_source, "MeasuredResource.measurement_source")
        if self.measurement_source.strip().lower() in _DECLARED_MARKERS:
            raise SessionDisjointValidityRefusal(
                "efficiency resource must cite a real measurement source, not a declaration"
            )

    def backed(self, rtol: float) -> bool:
        """True when the declared cost matches the measured cost within a relative tolerance."""

        return abs(self.declared - self.measured) <= rtol * self.measured

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "declared": self.declared,
            "measured": self.measured,
            "unit": self.unit,
            "measurement_source": self.measurement_source,
        }


@dataclass(frozen=True, slots=True)
class MeasuredEfficiencyContract:
    """An efficiency claim that is refused unless every declared cost is backed by measurement.

    Fails closed on a declared-only cost, on a missing measured baseline, or on a comparison that
    does not require matched cost. Claim scope: deterministic programmatic mechanics only; no
    capability claim.
    """

    efficiency_metric: str
    resources: tuple[MeasuredResource, ...]
    matched: MatchedCostBudget
    rtol: float
    matched_cost_required: bool
    baseline_measured: bool
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE5_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE5_VALIDITY_SCHEMA:
            raise SessionDisjointValidityRefusal(f"unsupported efficiency schema {self.schema!r}")
        _require_id(self.efficiency_metric, "MeasuredEfficiencyContract.efficiency_metric")
        if not self.resources:
            raise SessionDisjointValidityRefusal("efficiency claim needs at least one measured resource")
        kinds = [r.kind for r in self.resources]
        if len(set(kinds)) != len(kinds):
            raise SessionDisjointValidityRefusal("efficiency resource kinds must be unique")
        if not (COMPUTE_KINDS & set(kinds)):
            raise SessionDisjointValidityRefusal(
                "efficiency claim needs a measured compute cost (wall_time_s or flops)"
            )
        if not 0.0 < self.rtol <= 0.5:
            raise SessionDisjointValidityRefusal("efficiency tolerance rtol must be in (0, 0.5]")
        if not self.matched_cost_required:
            raise SessionDisjointValidityRefusal(
                "efficiency comparison must require matched full-system cost"
            )
        if not self.baseline_measured:
            raise SessionDisjointValidityRefusal("efficiency claim needs a measured baseline cost")
        for resource in self.resources:
            if not resource.backed(self.rtol):
                raise SessionDisjointValidityRefusal(
                    f"declared-only efficiency claim refused: declared {resource.kind} cost "
                    "does not match the measured cost"
                )
        if self.claim_scope != CLAIM_SCOPE:
            raise SessionDisjointValidityRefusal("efficiency claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "efficiency_metric": self.efficiency_metric,
            "resources": [r.payload() for r in self.resources],
            "matched": self.matched.payload(),
            "rtol": self.rtol,
            "matched_cost_required": self.matched_cost_required,
            "baseline_measured": self.baseline_measured,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# Section D. The composite session-disjoint validity contract.


@dataclass(frozen=True, slots=True)
class SessionDisjointValidityContract:
    """The external-validity harness: every axis, every control, and the efficiency backing.

    A valid instance means the declaration set is complete and self-consistent. It does not assert
    that generality was demonstrated; the activation gate holds that claim. Claim scope:
    deterministic programmatic mechanics only; no capability claim.
    """

    axes: tuple[AxisOutcome, ...]
    controls: tuple[ControlOutcome, ...]
    efficiency: MeasuredEfficiencyContract
    prior_null: str = PRIOR_NULL
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE5_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE5_VALIDITY_SCHEMA:
            raise SessionDisjointValidityRefusal(f"unsupported validity contract schema {self.schema!r}")
        assert_axis_completeness(self.axes)
        assert_control_completeness(self.controls)
        if self.prior_null != PRIOR_NULL:
            raise SessionDisjointValidityRefusal("the session-bound-leakage prior null cannot be reframed")
        if self.claim_scope != CLAIM_SCOPE:
            raise SessionDisjointValidityRefusal("validity contract claim scope cannot be widened")

    def unmet_axes(self) -> tuple[str, ...]:
        return tuple(a.axis for a in self.axes if not a.passed)

    def all_axes_pass(self) -> bool:
        return not self.unmet_axes()

    def leaked_controls(self) -> tuple[str, ...]:
        return tuple(c.control for c in self.controls if c.leaked)

    def any_control_leaked(self) -> bool:
        return bool(self.leaked_controls())

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prior_null": self.prior_null,
            "axes": [a.payload() for a in self.axes],
            "controls": [c.payload() for c in self.controls],
            "efficiency": self.efficiency.payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section E. Activation gate. Off by default; refuses any generality claim until
# every axis passes, every control is clean, and a countersigned license is supplied.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExternalValidityGate:
    """A fail-closed gate. Without a license it refuses every general-validity claim by default.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so
    that a passing harness can never quietly promote itself into a generality claim; activation is
    not earned until a disjoint reviewer countersigns the exact contract digest.
    """

    license_token: str | None = None
    default_state: str = "off"
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE5_VALIDITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE5_VALIDITY_SCHEMA:
            raise SessionDisjointValidityRefusal(f"unsupported gate schema {self.schema!r}")
        if self.default_state != "off":
            raise SessionDisjointValidityRefusal(
                "external validity gate is off by default and cannot be forced on at construction"
            )
        if self.license_token is not None and not self.license_token.strip():
            raise SessionDisjointValidityRefusal("license token must be non-empty when supplied")
        if self.claim_scope != CLAIM_SCOPE:
            raise SessionDisjointValidityRefusal("gate claim scope cannot be widened")

    def _require_licensed(self) -> str:
        if self.license_token is None:
            raise SessionDisjointValidityRefusal(
                "activation not earned: no confirmation license supplied; external general-validity "
                "claims are refused by default and attributed to the session-bound-leakage null"
            )
        return self.license_token

    def expected_receipt(self, contract: SessionDisjointValidityContract) -> str:
        """The receipt a disjoint reviewer must countersign for this exact contract and license."""

        token = self._require_licensed()
        return canonical_sha256({"contract": contract.digest(), "license": token})

    def certify_generality(self, contract: SessionDisjointValidityContract, receipt: str) -> dict[str, Any]:
        """Fail closed unless licensed, the receipt matches, every axis passes, no control leaked."""

        self._require_licensed()
        _require_sha256(receipt, "ExternalValidityGate.receipt")
        if receipt != self.expected_receipt(contract):
            raise SessionDisjointValidityRefusal(
                "confirmation receipt does not match the contract digest and license"
            )
        unmet = contract.unmet_axes()
        if unmet:
            raise SessionDisjointValidityRefusal(
                f"general-validity refused: disjointness axes did not pass: {list(unmet)}"
            )
        leaked = contract.leaked_controls()
        if leaked:
            raise SessionDisjointValidityRefusal(
                f"general-validity refused: negative controls leaked: {list(leaked)}"
            )
        return {
            "schema": self.schema,
            "certified": True,
            "contract_digest": contract.digest(),
            "receipt": receipt,
            "axes": list(VALIDITY_AXES),
            "controls": list(NEGATIVE_CONTROLS),
            "prior_null": contract.prior_null,
            "claim_scope": self.claim_scope,
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "default_state": self.default_state,
            "licensed": self.license_token is not None,
            "claim_scope": self.claim_scope,
        }


# Section F. Deterministic, seeded builders (used by tests and later wiring).


def _seeded_int(seed: int, label: str, modulo: int) -> int:
    """Deterministic small integer from a digest; no wall clock, no OS entropy."""

    if modulo <= 0:
        raise SessionDisjointValidityRefusal("modulo must be positive")
    digest = canonical_sha256({"seed": seed, "label": label, "schema": STAGE5_VALIDITY_SCHEMA})
    return int(digest[:8], 16) % modulo


def synthesize_axis_evidence(seed: int, axis: str) -> str:
    """Deterministic evidence digest for one axis. Not evidence of value; a byte-exact test vector."""

    if axis not in VALIDITY_AXES:
        raise SessionDisjointValidityRefusal(f"unsupported validity axis {axis!r}")
    if seed < 0:
        raise SessionDisjointValidityRefusal("evidence seed must be nonnegative")
    return canonical_sha256({"seed": seed, "axis": axis, "schema": STAGE5_VALIDITY_SCHEMA})


def synthesize_control_evidence(seed: int, control: str) -> str:
    """Deterministic evidence digest for one negative control."""

    if control not in NEGATIVE_CONTROLS:
        raise SessionDisjointValidityRefusal(f"unsupported negative control {control!r}")
    if seed < 0:
        raise SessionDisjointValidityRefusal("evidence seed must be nonnegative")
    return canonical_sha256({"seed": seed, "control": control, "schema": STAGE5_VALIDITY_SCHEMA})


def build_measured_efficiency(seed: int) -> MeasuredEfficiencyContract:
    """Return a non-vacuous efficiency contract whose declared costs equal their measured costs."""

    if seed < 0:
        raise SessionDisjointValidityRefusal("efficiency seed must be nonnegative")
    wall = 1.0 + _seeded_int(seed, "wall_time_s", 50) / 10.0
    flops = 1_000_000 + _seeded_int(seed, "flops", 1000) * 1000
    resources = (
        MeasuredResource(
            kind="wall_time_s",
            declared=wall,
            measured=wall,
            unit="seconds",
            measurement_source="perf_counter over the held-out session run",
        ),
        MeasuredResource(
            kind="flops",
            declared=float(flops),
            measured=float(flops),
            unit="flop",
            measurement_source="hardware performance counter tally",
        ),
    )
    matched = MatchedCostBudget(
        params=1024 + _seeded_int(seed, "params", 256),
        flops=flops,
        wall_time_ms=int(wall * 1000),
        peak_memory_bytes=8192 + _seeded_int(seed, "peak", 4096),
    )
    return MeasuredEfficiencyContract(
        efficiency_metric="held_out_accuracy_per_measured_flop",
        resources=resources,
        matched=matched,
        rtol=0.05,
        matched_cost_required=True,
        baseline_measured=True,
    )


def build_session_disjoint_contract(
    seed: int, *, all_axes_pass: bool = True, controls_clean: bool = True
) -> SessionDisjointValidityContract:
    """Build a complete, deterministic session-disjoint validity contract for one seed."""

    if seed < 0:
        raise SessionDisjointValidityRefusal("contract seed must be nonnegative")
    axes = tuple(
        AxisOutcome(
            axis=axis,
            passed=all_axes_pass,
            replications=2 + _seeded_int(seed, f"reps:{axis}", 3),
            disjoint_from_calibration=True,
            evidence_sha256=synthesize_axis_evidence(seed, axis),
        )
        for axis in VALIDITY_AXES
    )
    controls = tuple(
        ControlOutcome(
            control=control,
            leaked=not controls_clean,
            evidence_sha256=synthesize_control_evidence(seed, control),
        )
        for control in NEGATIVE_CONTROLS
    )
    return SessionDisjointValidityContract(
        axes=axes,
        controls=controls,
        efficiency=build_measured_efficiency(seed),
    )


def build_activation_gate(license_token: str | None = None) -> ExternalValidityGate:
    """Return the activation gate; off by default unless a non-empty license token is supplied."""

    return ExternalValidityGate(license_token=license_token)


# Section G. Epoch coverage record and capability flag.

SCIENTIFIC_CAPABILITY_CLAIM = False

SUB_QUESTIONS: tuple[str, ...] = (
    "does apparent skill survive a fresh session disjoint from calibration?",
    "does it survive new seeds and new task families?",
    "does it survive lesions and an independent reconstruction?",
    "is any efficiency claim backed by a real measured cost?",
    "are same-session-leak, seed-reuse, and single-task-family ruled out?",
)


def coverage() -> dict[str, list[str]]:
    """Map this epoch's sub-questions to the mechanics that encode each bar (readiness only)."""

    return {
        SUB_QUESTIONS[0]: [
            "the fresh-session axis in VALIDITY_AXES with a disjoint-from-calibration requirement",
            "AxisOutcome refuses a passing axis that is not disjoint from the calibration session",
        ],
        SUB_QUESTIONS[1]: [
            "the new-seeds and new-task-families axes carried in the ordered axis coverage check",
            "each axis requires at least two independent replications before it can pass",
        ],
        SUB_QUESTIONS[2]: [
            "the lesions and independent-reconstruction axes in VALIDITY_AXES",
            "assert_axis_completeness fails closed on any missing or reordered axis",
        ],
        SUB_QUESTIONS[3]: [
            "MeasuredEfficiencyContract refuses any declared-only cost via MeasuredResource.backed",
            "a measured baseline and matched full-system cost are both required, non-vacuously",
        ],
        SUB_QUESTIONS[4]: [
            "NEGATIVE_CONTROLS declares same-session-leak, seed-reuse, and single-task-family",
            "ExternalValidityGate refuses certification whenever any negative control leaked",
        ],
    }
