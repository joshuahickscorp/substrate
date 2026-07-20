from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..ladder.stage_ladder import FIRST_ACTIVATION_STAGE
from ..substrate.events import canonical_sha256

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_unit(value: float, label: str, refusal: type[ValueError]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise refusal(f"{label} must be a real number")
    if not math.isfinite(float(value)):
        raise refusal(f"{label} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise refusal(f"{label} must lie in the unit interval [0, 1]")


class DualMetricReadingBase:
    schema: str
    claim_scope: str
    axes: ClassVar[tuple[str, str]]
    expected_schema: ClassVar[str]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.expected_schema:
            raise self.refusal(f"unsupported reading schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("reading claim scope cannot be widened")
        for axis in self.axes:
            _require_unit(getattr(self, axis), f"reading.{axis}", self.refusal)

    def axis(self, name: str) -> float:
        if name not in self.axes:
            raise self.refusal(f"unknown metric axis {name!r}")
        return getattr(self, name)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            **{axis: getattr(self, axis) for axis in self.axes},
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


class MatchedCostBudgetBase:
    budget_fields: ClassVar[tuple[str, ...]]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        for name in self.budget_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise self.refusal(f"matched budget {name} must be an integer")
            if value <= 0:
                raise self.refusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.budget_fields}

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ControlArmBase:
    control: str
    reading: Any
    matched: Any
    claim_scope: str = CLAIM_SCOPE

    controls: ClassVar[tuple[str, ...]]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.control not in self.controls:
            raise self.refusal(f"unsupported control {self.control!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("control arm claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "reading": self.reading.payload(),
            "matched": self.matched.payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ControlFamilyBase:
    schema: str
    arms: tuple[Any, ...]
    claim_scope: str = CLAIM_SCOPE

    expected_schema: ClassVar[str]
    controls: ClassVar[tuple[str, ...]]
    axes: ClassVar[tuple[str, str]]
    order_error: ClassVar[str]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.expected_schema:
            raise self.refusal(f"unsupported control family schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("control family claim scope cannot be widened")
        if tuple(arm.control for arm in self.arms) != self.controls:
            raise self.refusal(self.order_error)
        if len({arm.matched.digest() for arm in self.arms}) != 1:
            raise self.refusal("control arms are not held to one matched budget")

    @property
    def matched(self) -> Any:
        return self.arms[0].matched

    def reading(self, control: str) -> Any:
        for arm in self.arms:
            if arm.control == control:
                return arm.reading
        raise self.refusal(f"control {control!r} absent from the family")

    def best_on(self, axis: str) -> float:
        if axis not in self.axes:
            raise self.refusal(f"unknown metric axis {axis!r}")
        return max(arm.reading.axis(axis) for arm in self.arms)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [arm.payload() for arm in self.arms],
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class AxisComparisonBase:
    axis: str
    candidate_value: float
    best_control_value: float

    axes: ClassVar[tuple[str, str]]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.axis not in self.axes:
            raise self.refusal(f"unknown metric axis {self.axis!r}")
        _require_unit(self.candidate_value, f"{self.axis}.candidate_value", self.refusal)
        _require_unit(self.best_control_value, f"{self.axis}.best_control_value", self.refusal)

    @property
    def margin(self) -> float:
        return self.candidate_value - self.best_control_value

    @property
    def improved(self) -> bool:
        return self.candidate_value > self.best_control_value

    def payload(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "candidate_value": self.candidate_value,
            "best_control_value": self.best_control_value,
            "margin": self.margin,
            "improved": self.improved,
        }


@dataclass(frozen=True, slots=True)
class JointAxisContractBase:
    schema: str
    axes: tuple[str, ...]
    controls: tuple[str, ...]
    matched_cost_required: bool
    both_axes_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    expected_schema: ClassVar[str]
    expected_axes: ClassVar[tuple[str, str]]
    expected_controls: ClassVar[tuple[str, ...]]
    expected_prior_null: ClassVar[str]
    axes_error: ClassVar[str]
    matched_cost_error: ClassVar[str]
    both_axes_error: ClassVar[str]
    prior_null_error: ClassVar[str]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.expected_schema:
            raise self.refusal(f"unsupported contract schema {self.schema!r}")
        if tuple(self.axes) != self.expected_axes:
            raise self.refusal(self.axes_error)
        if tuple(self.controls) != self.expected_controls:
            raise self.refusal("declared control set drifted in membership or order")
        if not self.matched_cost_required:
            raise self.refusal(self.matched_cost_error)
        if not self.both_axes_required:
            raise self.refusal(self.both_axes_error)
        if self.replication_min < 2:
            raise self.refusal("joint claim requires at least two independent replications")
        if self.prior_null != self.expected_prior_null:
            raise self.refusal(self.prior_null_error)
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("contract claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axes": list(self.axes),
            "controls": list(self.controls),
            "matched_cost_required": self.matched_cost_required,
            "both_axes_required": self.both_axes_required,
            "replication_min": self.replication_min,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


class JointImprovementVerdictBase:
    __slots__ = ()

    schema: str
    matched_cost_required: bool
    prior_null: str
    claim_scope: str
    expected_schema: ClassVar[str]
    axis_fields: ClassVar[tuple[str, str]]
    axis_labels: ClassVar[tuple[str, str]]
    expected_prior_null: ClassVar[str]
    first_axis_error: ClassVar[str]
    second_axis_error: ClassVar[str]
    prior_null_error: ClassVar[str]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.expected_schema:
            raise self.refusal(f"unsupported verdict schema {self.schema!r}")
        first, second = self._comparisons
        if first.axis != self.axis_fields[0]:
            raise self.refusal(self.first_axis_error)
        if second.axis != self.axis_fields[1]:
            raise self.refusal(self.second_axis_error)
        if not self.matched_cost_required:
            raise self.refusal("verdict must be computed at matched full-system cost")
        if self.prior_null != self.expected_prior_null:
            raise self.refusal(self.prior_null_error)
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("verdict claim scope cannot be widened")

    @property
    def _comparisons(self) -> tuple[Any, Any]:
        return getattr(self, self.axis_fields[0]), getattr(self, self.axis_fields[1])

    @property
    def both_axes_improved(self) -> bool:
        first, second = self._comparisons
        return first.improved and second.improved

    @property
    def only_one_axis_improved(self) -> bool:
        first, second = self._comparisons
        return first.improved != second.improved

    def certify(self) -> Any:
        first, second = self._comparisons
        if self.only_one_axis_improved:
            winner, loser = self.axis_fields if first.improved else reversed(self.axis_fields)
            raise self.refusal(
                f"single-axis win refused: {winner} improved but {loser} did not; "
                f"this is the {self.expected_prior_null}, not a joint improvement"
            )
        if not self.both_axes_improved:
            raise self.refusal(
                f"no-axis win refused: neither {self.axis_labels[0]} nor {self.axis_labels[1]} "
                f"improved; the {self.expected_prior_null} holds"
            )
        return self

    def payload(self) -> dict[str, Any]:
        first, second = self._comparisons
        return {
            "schema": self.schema,
            self.axis_fields[0]: first.payload(),
            self.axis_fields[1]: second.payload(),
            "matched_cost_required": self.matched_cost_required,
            "both_axes_improved": self.both_axes_improved,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def evaluate_joint_axes(
    *,
    candidate: Any,
    candidate_budget: Any,
    controls: ControlFamilyBase,
    axes: tuple[str, str],
    schema: str,
    comparison_type: type[AxisComparisonBase],
    verdict_type: type[Any],
    prior_null: str,
    refusal: type[ValueError],
) -> Any:
    if candidate_budget.digest() != controls.matched.digest():
        raise refusal("candidate is not held to the control family matched budget; comparison refused")
    comparisons = {
        axis: comparison_type(
            axis=axis,
            candidate_value=getattr(candidate, axis),
            best_control_value=controls.best_on(axis),
        )
        for axis in axes
    }
    return verdict_type(
        schema=schema,
        **comparisons,
        matched_cost_required=True,
        prior_null=prior_null,
    )


@dataclass(frozen=True, slots=True)
class ConfirmationReceiptBase:
    preregistration_sha256: str
    verdict_digest: str
    replication_count: int
    matched_cost_attested: bool
    independent_reviewer: str
    claim_scope: str = CLAIM_SCOPE

    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.preregistration_sha256) is None:
            raise self.refusal("receipt.preregistration_sha256 must be a lowercase SHA-256 digest")
        if _SHA256_RE.fullmatch(self.verdict_digest) is None:
            raise self.refusal("receipt.verdict_digest must be a lowercase SHA-256 digest")
        if self.replication_count < 2:
            raise self.refusal("receipt must attest at least two independent replications")
        if not self.matched_cost_attested:
            raise self.refusal("receipt must attest that the win held at matched cost")
        if _ID_RE.fullmatch(self.independent_reviewer) is None:
            raise self.refusal("receipt.independent_reviewer must use stable lowercase characters")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("receipt claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "preregistration_sha256": self.preregistration_sha256,
            "verdict_digest": self.verdict_digest,
            "replication_count": self.replication_count,
            "matched_cost_attested": self.matched_cost_attested,
            "independent_reviewer": self.independent_reviewer,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class JointClaimGateBase:
    activation_permitted: bool = False
    claim_scope: str = CLAIM_SCOPE

    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("gate claim scope cannot be widened")

    def authorize(self, verdict: Any, receipt: ConfirmationReceiptBase | None = None) -> Any:
        if not self.activation_permitted:
            raise self.refusal(
                "joint-claim activation is not earned; the gate is closed by default and local code "
                "cannot open it. Route to an external replication with a signed confirmation receipt."
            )
        if receipt is None:
            raise self.refusal("gate authorization requires an external confirmation receipt")
        verdict.certify()
        if receipt.verdict_digest != verdict.digest():
            raise self.refusal("receipt does not confirm this exact verdict")
        return verdict

    def payload(self) -> dict[str, Any]:
        return {"activation_permitted": self.activation_permitted, "claim_scope": self.claim_scope}


@dataclass(frozen=True, slots=True)
class JointAxisSpec:
    mechanism_id: str
    requirement_id: str
    schema: str
    axes: tuple[str, str]
    controls: tuple[str, ...]
    prior_null: str
    mechanism_arm: str
    result_margin_keys: tuple[str, str]
    receipt_margin_keys: tuple[str, str]
    regimes: tuple[str, str] = ("null", "favorable")


@dataclass(frozen=True, slots=True)
class JointAxisResult:
    regime: str
    seed: int
    mechanism_reading: Any
    control_readings: tuple[tuple[str, Any], ...]
    schema: str
    claim_scope: str

    spec: ClassVar[JointAxisSpec]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.spec.schema:
            raise self.refusal(f"unsupported run result schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("run result claim scope cannot be widened")
        if self.regime not in self.spec.regimes:
            raise self.refusal(f"unknown regime {self.regime!r}")
        if self.seed < 0:
            raise self.refusal("run result seed must be nonnegative")
        names = tuple(name for name, _ in self.control_readings)
        if not names:
            raise self.refusal("a run result needs at least one control to compare against")
        if len(set(names)) != len(names):
            raise self.refusal("control readings must be unique")
        for name in names:
            if name not in self.spec.controls:
                raise self.refusal(f"unsupported control {name!r}")

    def margin(self, axis: int) -> float:
        attribute = self.spec.axes[axis]
        candidate = getattr(self.mechanism_reading, attribute)
        return min(candidate - getattr(reading, attribute) for _, reading in self.control_readings)

    @property
    def both_axes_win(self) -> bool:
        return self.margin(0) > 0.0 and self.margin(1) > 0.0

    @property
    def controls_beaten_both(self) -> tuple[str, ...]:
        first, second = self.spec.axes
        beaten = {
            name
            for name, reading in self.control_readings
            if getattr(self.mechanism_reading, first) > getattr(reading, first)
            and getattr(self.mechanism_reading, second) > getattr(reading, second)
        }
        return tuple(control for control in self.spec.controls if control in beaten)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.spec.mechanism_id,
            "regime": self.regime,
            "seed": self.seed,
            "axes": list(self.spec.axes),
            "mechanism": self.mechanism_reading.payload(),
            "controls": [
                {"control": name, "reading": reading.payload()} for name, reading in self.control_readings
            ],
            self.spec.result_margin_keys[0]: self.margin(0),
            self.spec.result_margin_keys[1]: self.margin(1),
            "both_axes_win": self.both_axes_win,
            "prior_null": self.spec.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class JointAxisRunner:
    mechanism_id: str
    schema: str
    claim_scope: str

    spec: ClassVar[JointAxisSpec]
    refusal: ClassVar[type[ValueError]]
    result_type: ClassVar[type[JointAxisResult]]
    run_all: ClassVar[Callable[[Any], Mapping[str, Any]]]

    def __post_init__(self) -> None:
        if self.mechanism_id != self.spec.mechanism_id:
            raise self.refusal("runner mechanism_id drift")
        if self.schema != self.spec.schema:
            raise self.refusal(f"unsupported runner schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("runner claim scope cannot be widened")

    def run(self, bed: Bed, seed: int, regime: str = "favorable") -> JointAxisResult:
        if bed.mechanism_id != self.spec.mechanism_id:
            raise self.refusal("runner and bed mechanism_id mismatch")
        if regime == self.spec.regimes[0]:
            inputs = bed.null_regime(seed)
        elif regime == self.spec.regimes[1]:
            inputs = bed.favorable_regime(seed)
        else:
            raise self.refusal(f"unknown regime {regime!r}")
        readings = type(self).run_all(inputs)
        control_readings = tuple(
            (control, readings[control]) for control in bed.controls() if control in self.spec.controls
        )
        return self.result_type(
            regime=regime,
            seed=seed,
            mechanism_reading=readings[self.spec.mechanism_arm],
            control_readings=control_readings,
        )

    def mint(self, results: JointAxisResult) -> RunReceipt:
        verdict = (
            VERDICT_MECHANICS_OK
            if results.regime == self.spec.regimes[1] and results.both_axes_win
            else VERDICT_NULL
        )
        detail = {
            "regime": results.regime,
            "seed": results.seed,
            "axes": list(self.spec.axes),
            "controls": list(self.spec.controls),
            self.spec.receipt_margin_keys[0]: results.margin(0),
            self.spec.receipt_margin_keys[1]: results.margin(1),
            "both_axes_win": results.both_axes_win,
            "prior_null": self.spec.prior_null,
        }
        return mint_demonstration(
            mechanism_id=self.spec.mechanism_id,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=self.spec.requirement_id,
            controls_cleared=results.controls_beaten_both,
            evidence_digest=results.digest(),
            verdict=verdict,
            detail=detail,
        )
