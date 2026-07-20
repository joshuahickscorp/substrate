
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

STABILITY_PLASTICITY_SCHEMA = "mop-stability-plasticity/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PRIOR_NULL = "p6-stability-plasticity-split"

DUAL_AXES: tuple[str, ...] = ("retention", "future_learnability")

REQUIRED_CONTROLS: tuple[str, ...] = ("fresh-init", "no-replay", "full-retrain", "frozen-core")


class StabilityPlasticityRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise StabilityPlasticityRefusal(f"{label} must use stable lowercase characters")


def _require_unit(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StabilityPlasticityRefusal(f"{label} must be a real number")
    if not math.isfinite(float(value)):
        raise StabilityPlasticityRefusal(f"{label} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise StabilityPlasticityRefusal(f"{label} must lie in the unit interval [0, 1]")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise StabilityPlasticityRefusal(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DualMetricReading:

    retention: float
    future_learnability: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = STABILITY_PLASTICITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_SCHEMA:
            raise StabilityPlasticityRefusal(f"unsupported reading schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("reading claim scope cannot be widened")
        _require_unit(self.retention, "reading.retention")
        _require_unit(self.future_learnability, "reading.future_learnability")

    def axis(self, name: str) -> float:
        if name == "retention":
            return self.retention
        if name == "future_learnability":
            return self.future_learnability
        raise StabilityPlasticityRefusal(f"unknown metric axis {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "retention": self.retention,
            "future_learnability": self.future_learnability,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:

    params: int
    flops: int
    replay_samples: int
    update_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("replay_samples", self.replay_samples),
            ("update_steps", self.update_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise StabilityPlasticityRefusal(f"matched budget {name} must be an integer")
            if value <= 0:
                raise StabilityPlasticityRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "replay_samples": self.replay_samples,
            "update_steps": self.update_steps,
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
            raise StabilityPlasticityRefusal(f"unsupported control {self.control!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("control arm claim scope cannot be widened")

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
        if self.schema != STABILITY_PLASTICITY_SCHEMA:
            raise StabilityPlasticityRefusal(f"unsupported control family schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("control family claim scope cannot be widened")
        if tuple(arm.control for arm in self.arms) != REQUIRED_CONTROLS:
            raise StabilityPlasticityRefusal(
                "control family membership or order drift; expected fresh-init, no-replay, "
                "full-retrain, frozen-core in that order"
            )
        budgets = {arm.matched.digest() for arm in self.arms}
        if len(budgets) != 1:
            raise StabilityPlasticityRefusal("control arms are not held to one matched budget")

    @property
    def matched(self) -> MatchedCostBudget:
        return self.arms[0].matched

    def reading(self, control: str) -> DualMetricReading:
        for arm in self.arms:
            if arm.control == control:
                return arm.reading
        raise StabilityPlasticityRefusal(f"control {control!r} absent from the family")

    def best_on(self, axis: str) -> float:
        if axis not in DUAL_AXES:
            raise StabilityPlasticityRefusal(f"unknown metric axis {axis!r}")
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
        raise StabilityPlasticityRefusal("declared control set drifted in membership or order")


@dataclass(frozen=True, slots=True)
class StabilityPlasticityContract:

    schema: str
    axes: tuple[str, ...]
    controls: tuple[str, ...]
    matched_cost_required: bool
    both_axes_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_SCHEMA:
            raise StabilityPlasticityRefusal(f"unsupported contract schema {self.schema!r}")
        if tuple(self.axes) != DUAL_AXES:
            raise StabilityPlasticityRefusal(
                "axis set or order drift; both retention and future learnability required"
            )
        assert_control_completeness(self.controls)
        if not self.matched_cost_required:
            raise StabilityPlasticityRefusal(
                "stability vs plasticity claim must require matched full-system cost"
            )
        if not self.both_axes_required:
            raise StabilityPlasticityRefusal(
                "a single-axis win is exactly the P6 split; both axes must be required"
            )
        if self.replication_min < 2:
            raise StabilityPlasticityRefusal("joint claim requires at least two independent replications")
        if self.prior_null != PRIOR_NULL:
            raise StabilityPlasticityRefusal(
                "contract must name the p6-stability-plasticity-split as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("contract claim scope cannot be widened")

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


@dataclass(frozen=True, slots=True)
class AxisComparison:

    axis: str
    candidate_value: float
    best_control_value: float

    def __post_init__(self) -> None:
        if self.axis not in DUAL_AXES:
            raise StabilityPlasticityRefusal(f"unknown metric axis {self.axis!r}")
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
    retention: AxisComparison
    future_learnability: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_SCHEMA:
            raise StabilityPlasticityRefusal(f"unsupported verdict schema {self.schema!r}")
        if self.retention.axis != "retention":
            raise StabilityPlasticityRefusal("first comparison must be the retention axis")
        if self.future_learnability.axis != "future_learnability":
            raise StabilityPlasticityRefusal("second comparison must be the future learnability axis")
        if not self.matched_cost_required:
            raise StabilityPlasticityRefusal("verdict must be computed at matched full-system cost")
        if self.prior_null != PRIOR_NULL:
            raise StabilityPlasticityRefusal(
                "verdict must name the p6-stability-plasticity-split as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("verdict claim scope cannot be widened")

    @property
    def both_axes_improved(self) -> bool:
        return self.retention.improved and self.future_learnability.improved

    @property
    def only_one_axis_improved(self) -> bool:
        return self.retention.improved != self.future_learnability.improved

    def certify(self) -> JointImprovementVerdict:

        if self.only_one_axis_improved:
            winner = "retention" if self.retention.improved else "future_learnability"
            loser = "future_learnability" if self.retention.improved else "retention"
            raise StabilityPlasticityRefusal(
                f"single-axis win refused: {winner} improved but {loser} did not; "
                f"this is the {PRIOR_NULL}, not a joint improvement"
            )
        if not self.both_axes_improved:
            raise StabilityPlasticityRefusal(
                f"no-axis win refused: neither retention nor future learnability improved; "
                f"the {PRIOR_NULL} holds"
            )
        return self

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "retention": self.retention.payload(),
            "future_learnability": self.future_learnability.payload(),
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
        raise StabilityPlasticityRefusal(
            "candidate is not held to the control family matched budget; comparison refused"
        )
    retention_cmp = AxisComparison(
        axis="retention",
        candidate_value=candidate.retention,
        best_control_value=controls.best_on("retention"),
    )
    future_cmp = AxisComparison(
        axis="future_learnability",
        candidate_value=candidate.future_learnability,
        best_control_value=controls.best_on("future_learnability"),
    )
    return JointImprovementVerdict(
        schema=STABILITY_PLASTICITY_SCHEMA,
        retention=retention_cmp,
        future_learnability=future_cmp,
        matched_cost_required=True,
        prior_null=PRIOR_NULL,
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
            raise StabilityPlasticityRefusal("receipt must attest at least two independent replications")
        if not self.matched_cost_attested:
            raise StabilityPlasticityRefusal("receipt must attest that the win held at matched cost")
        _require_id(self.independent_reviewer, "receipt.independent_reviewer")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityRefusal("receipt claim scope cannot be widened")

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
            raise StabilityPlasticityRefusal("gate claim scope cannot be widened")

    def authorize(
        self,
        verdict: JointImprovementVerdict,
        receipt: ConfirmationReceipt | None = None,
    ) -> JointImprovementVerdict:

        if not self.activation_permitted:
            raise StabilityPlasticityRefusal(
                "joint-claim activation is not earned; the gate is closed by default and local code "
                "cannot open it. Route to an external replication with a signed confirmation receipt."
            )
        if receipt is None:
            raise StabilityPlasticityRefusal("gate authorization requires an external confirmation receipt")
        verdict.certify()
        if receipt.verdict_digest != verdict.digest():
            raise StabilityPlasticityRefusal("receipt does not confirm this exact verdict")
        return verdict

    def payload(self) -> dict[str, Any]:
        return {"activation_permitted": self.activation_permitted, "claim_scope": self.claim_scope}


def coverage() -> dict[str, Sequence[str]]:

    return {
        "stable-core-coexists-with-rapid-adaptation": (
            "frozen-core and fresh-init arms are declared as the two poles of the P6 split",
            "ControlFamily forces both poles into one matched-budget comparison",
        ),
        "retention-and-future-learning-improve-jointly": (
            "JointImprovementVerdict.certify refuses any single-axis win",
            "DUAL_AXES fixes retention and future_learnability as jointly required",
        ),
        "improvement-is-at-matched-cost": (
            "MatchedCostBudget must be non-vacuous and equal across candidate and controls",
            "evaluate_joint_improvement refuses a candidate not held to the family budget",
        ),
        "replay-is-not-a-complete-theory-of-plasticity": (
            "PRIOR_NULL pins the p6-stability-plasticity-split as the default hypothesis",
            "the seeded toy lands the candidate below at least one axis maximum, so the null holds",
        ),
    }


SCIENTIFIC_CAPABILITY_CLAIM = False
