
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

CALIBRATED_UNCERTAINTY_SCHEMA = "mop-calibrated-uncertainty/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise CalibratedUncertaintyRefusal(f"{label} must use stable lowercase characters")


def _require_unit(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalibratedUncertaintyRefusal(f"{label} must be a real number")
    if not math.isfinite(float(value)):
        raise CalibratedUncertaintyRefusal(f"{label} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise CalibratedUncertaintyRefusal(f"{label} must lie in the unit interval [0, 1]")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise CalibratedUncertaintyRefusal(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DualMetricReading:

    selective_risk_reduction: float
    decision_utility: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = CALIBRATED_UNCERTAINTY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CALIBRATED_UNCERTAINTY_SCHEMA:
            raise CalibratedUncertaintyRefusal(f"unsupported reading schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("reading claim scope cannot be widened")
        _require_unit(self.selective_risk_reduction, "reading.selective_risk_reduction")
        _require_unit(self.decision_utility, "reading.decision_utility")

    def axis(self, name: str) -> float:
        if name == "selective_risk_reduction":
            return self.selective_risk_reduction
        if name == "decision_utility":
            return self.decision_utility
        raise CalibratedUncertaintyRefusal(f"unknown metric axis {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selective_risk_reduction": self.selective_risk_reduction,
            "decision_utility": self.decision_utility,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:

    params: int
    flops: int
    scored_items: int
    decision_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("scored_items", self.scored_items),
            ("decision_steps", self.decision_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise CalibratedUncertaintyRefusal(f"matched budget {name} must be an integer")
            if value <= 0:
                raise CalibratedUncertaintyRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "scored_items": self.scored_items,
            "decision_steps": self.decision_steps,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ControlArm:

    control: str
    reading: DualMetricReading
    matched: MatchedCostBudget
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.control not in REQUIRED_CONTROLS:
            raise CalibratedUncertaintyRefusal(f"unsupported control {self.control!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("control arm claim scope cannot be widened")

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
class ControlFamily:

    schema: str
    arms: tuple[ControlArm, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CALIBRATED_UNCERTAINTY_SCHEMA:
            raise CalibratedUncertaintyRefusal(f"unsupported control family schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("control family claim scope cannot be widened")
        if tuple(arm.control for arm in self.arms) != REQUIRED_CONTROLS:
            raise CalibratedUncertaintyRefusal(
                "control family membership or order drift; expected always_answer, random_abstain, "
                "overconfident_score, frozen_uniform in that order"
            )
        budgets = {arm.matched.digest() for arm in self.arms}
        if len(budgets) != 1:
            raise CalibratedUncertaintyRefusal("control arms are not held to one matched budget")

    @property
    def matched(self) -> MatchedCostBudget:
        return self.arms[0].matched

    def reading(self, control: str) -> DualMetricReading:
        for arm in self.arms:
            if arm.control == control:
                return arm.reading
        raise CalibratedUncertaintyRefusal(f"control {control!r} absent from the family")

    def best_on(self, axis: str) -> float:
        if axis not in DUAL_AXES:
            raise CalibratedUncertaintyRefusal(f"unknown metric axis {axis!r}")
        return max(arm.reading.axis(axis) for arm in self.arms)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [arm.payload() for arm in self.arms],
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def assert_control_completeness(controls: Sequence[str]) -> None:

    if tuple(controls) != REQUIRED_CONTROLS:
        raise CalibratedUncertaintyRefusal("declared control set drifted in membership or order")


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyContract:

    schema: str
    axes: tuple[str, ...]
    controls: tuple[str, ...]
    matched_cost_required: bool
    both_axes_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CALIBRATED_UNCERTAINTY_SCHEMA:
            raise CalibratedUncertaintyRefusal(f"unsupported contract schema {self.schema!r}")
        if tuple(self.axes) != DUAL_AXES:
            raise CalibratedUncertaintyRefusal(
                "axis set or order drift; both selective risk reduction and decision utility required"
            )
        assert_control_completeness(self.controls)
        if not self.matched_cost_required:
            raise CalibratedUncertaintyRefusal(
                "calibrated uncertainty claim must require matched full-system cost"
            )
        if not self.both_axes_required:
            raise CalibratedUncertaintyRefusal(
                "a single-axis win is exactly the decoupled confidence null; both axes must be required"
            )
        if self.replication_min < 2:
            raise CalibratedUncertaintyRefusal(
                "joint claim requires at least two independent replications"
            )
        if self.prior_null != PRIOR_NULL:
            raise CalibratedUncertaintyRefusal(
                "contract must name the u1-decoupled-confidence-null as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("contract claim scope cannot be widened")

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


@dataclass(frozen=True, slots=True)
class AxisComparison:

    axis: str
    candidate_value: float
    best_control_value: float

    def __post_init__(self) -> None:
        if self.axis not in DUAL_AXES:
            raise CalibratedUncertaintyRefusal(f"unknown metric axis {self.axis!r}")
        _require_unit(self.candidate_value, f"{self.axis}.candidate_value")
        _require_unit(self.best_control_value, f"{self.axis}.best_control_value")

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
class JointImprovementVerdict:

    schema: str
    selective_risk_reduction: AxisComparison
    decision_utility: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CALIBRATED_UNCERTAINTY_SCHEMA:
            raise CalibratedUncertaintyRefusal(f"unsupported verdict schema {self.schema!r}")
        if self.selective_risk_reduction.axis != "selective_risk_reduction":
            raise CalibratedUncertaintyRefusal("first comparison must be the selective risk axis")
        if self.decision_utility.axis != "decision_utility":
            raise CalibratedUncertaintyRefusal("second comparison must be the decision utility axis")
        if not self.matched_cost_required:
            raise CalibratedUncertaintyRefusal("verdict must be computed at matched full-system cost")
        if self.prior_null != PRIOR_NULL:
            raise CalibratedUncertaintyRefusal(
                "verdict must name the u1-decoupled-confidence-null as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("verdict claim scope cannot be widened")

    @property
    def both_axes_improved(self) -> bool:
        return self.selective_risk_reduction.improved and self.decision_utility.improved

    @property
    def only_one_axis_improved(self) -> bool:
        return self.selective_risk_reduction.improved != self.decision_utility.improved

    def certify(self) -> JointImprovementVerdict:

        if self.only_one_axis_improved:
            winner = (
                "selective_risk_reduction"
                if self.selective_risk_reduction.improved
                else "decision_utility"
            )
            loser = (
                "decision_utility"
                if self.selective_risk_reduction.improved
                else "selective_risk_reduction"
            )
            raise CalibratedUncertaintyRefusal(
                f"single-axis win refused: {winner} improved but {loser} did not; "
                f"this is the {PRIOR_NULL}, not a joint improvement"
            )
        if not self.both_axes_improved:
            raise CalibratedUncertaintyRefusal(
                f"no-axis win refused: neither selective risk reduction nor decision utility "
                f"improved; the {PRIOR_NULL} holds"
            )
        return self

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selective_risk_reduction": self.selective_risk_reduction.payload(),
            "decision_utility": self.decision_utility.payload(),
            "matched_cost_required": self.matched_cost_required,
            "both_axes_improved": self.both_axes_improved,
            "prior_null": self.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def evaluate_joint_improvement(
    *,
    candidate: DualMetricReading,
    candidate_budget: MatchedCostBudget,
    controls: ControlFamily,
) -> JointImprovementVerdict:

    if candidate_budget.digest() != controls.matched.digest():
        raise CalibratedUncertaintyRefusal(
            "candidate is not held to the control family matched budget; comparison refused"
        )
    risk_cmp = AxisComparison(
        axis="selective_risk_reduction",
        candidate_value=candidate.selective_risk_reduction,
        best_control_value=controls.best_on("selective_risk_reduction"),
    )
    utility_cmp = AxisComparison(
        axis="decision_utility",
        candidate_value=candidate.decision_utility,
        best_control_value=controls.best_on("decision_utility"),
    )
    return JointImprovementVerdict(
        schema=CALIBRATED_UNCERTAINTY_SCHEMA,
        selective_risk_reduction=risk_cmp,
        decision_utility=utility_cmp,
        matched_cost_required=True,
        prior_null=PRIOR_NULL,
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


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:

    preregistration_sha256: str
    verdict_digest: str
    replication_count: int
    matched_cost_attested: bool
    independent_reviewer: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_sha256(self.preregistration_sha256, "receipt.preregistration_sha256")
        _require_sha256(self.verdict_digest, "receipt.verdict_digest")
        if self.replication_count < 2:
            raise CalibratedUncertaintyRefusal(
                "receipt must attest at least two independent replications"
            )
        if not self.matched_cost_attested:
            raise CalibratedUncertaintyRefusal("receipt must attest that the win held at matched cost")
        _require_id(self.independent_reviewer, "receipt.independent_reviewer")
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("receipt claim scope cannot be widened")

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
class JointClaimGate:

    activation_permitted: bool = False
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.claim_scope != CLAIM_SCOPE:
            raise CalibratedUncertaintyRefusal("gate claim scope cannot be widened")

    def authorize(
        self,
        verdict: JointImprovementVerdict,
        receipt: ConfirmationReceipt | None = None,
    ) -> JointImprovementVerdict:

        if not self.activation_permitted:
            raise CalibratedUncertaintyRefusal(
                "joint-claim activation is not earned; the gate is closed by default and local code "
                "cannot open it. Route to an external replication with a signed confirmation receipt."
            )
        if receipt is None:
            raise CalibratedUncertaintyRefusal(
                "gate authorization requires an external confirmation receipt"
            )
        verdict.certify()
        if receipt.verdict_digest != verdict.digest():
            raise CalibratedUncertaintyRefusal("receipt does not confirm this exact verdict")
        return verdict

    def payload(self) -> dict[str, Any]:
        return {"activation_permitted": self.activation_permitted, "claim_scope": self.claim_scope}


def coverage() -> dict[str, Sequence[str]]:

    return {
        "abstention-must-buy-selective-risk-reduction": (
            "always_answer is declared as the no-abstention pole of the control family",
            "DUAL_AXES fixes selective_risk_reduction as a jointly required axis",
        ),
        "abstention-must-not-destroy-decision-utility": (
            "JointImprovementVerdict.certify refuses any single-axis win",
            "frozen_uniform is declared as the abstain-everything degenerate pole",
        ),
        "improvement-is-at-matched-cost": (
            "MatchedCostBudget must be non-vacuous and equal across candidate and controls",
            "evaluate_joint_improvement refuses a candidate not held to the family budget",
        ),
        "an-uninformative-confidence-signal-buys-nothing": (
            "PRIOR_NULL pins the u1-decoupled-confidence-null as the default hypothesis",
            "the seeded toy lands the candidate below at least one axis maximum, so the null holds",
        ),
    }


SCIENTIFIC_CAPABILITY_CLAIM = False
