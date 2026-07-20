
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

REDUCIBLE_NOVELTY_SCHEMA = "mop-reducible-novelty/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise ReducibleNoveltyRefusal(f"{label} must use stable lowercase characters")


def _require_unit(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReducibleNoveltyRefusal(f"{label} must be a real number")
    if not math.isfinite(float(value)):
        raise ReducibleNoveltyRefusal(f"{label} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise ReducibleNoveltyRefusal(f"{label} must lie in the unit interval [0, 1]")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ReducibleNoveltyRefusal(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DualMetricReading:

    learning_progress: float
    allocation_efficiency: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = REDUCIBLE_NOVELTY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REDUCIBLE_NOVELTY_SCHEMA:
            raise ReducibleNoveltyRefusal(f"unsupported reading schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("reading claim scope cannot be widened")
        _require_unit(self.learning_progress, "reading.learning_progress")
        _require_unit(self.allocation_efficiency, "reading.allocation_efficiency")

    def axis(self, name: str) -> float:
        if name == "learning_progress":
            return self.learning_progress
        if name == "allocation_efficiency":
            return self.allocation_efficiency
        raise ReducibleNoveltyRefusal(f"unknown metric axis {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "learning_progress": self.learning_progress,
            "allocation_efficiency": self.allocation_efficiency,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:

    probes: int
    sources: int
    pilot_probes: int
    flops: int

    def __post_init__(self) -> None:
        for name, value in (
            ("probes", self.probes),
            ("sources", self.sources),
            ("pilot_probes", self.pilot_probes),
            ("flops", self.flops),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ReducibleNoveltyRefusal(f"matched budget {name} must be an integer")
            if value <= 0:
                raise ReducibleNoveltyRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "probes": self.probes,
            "sources": self.sources,
            "pilot_probes": self.pilot_probes,
            "flops": self.flops,
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
            raise ReducibleNoveltyRefusal(f"unsupported control {self.control!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("control arm claim scope cannot be widened")

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
        if self.schema != REDUCIBLE_NOVELTY_SCHEMA:
            raise ReducibleNoveltyRefusal(f"unsupported control family schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("control family claim scope cannot be widened")
        if tuple(arm.control for arm in self.arms) != REQUIRED_CONTROLS:
            raise ReducibleNoveltyRefusal(
                "control family membership or order drift; expected uniform_allocation, "
                "random_allocation, novelty_chaser, static_prior in that order"
            )
        budgets = {arm.matched.digest() for arm in self.arms}
        if len(budgets) != 1:
            raise ReducibleNoveltyRefusal("control arms are not held to one matched budget")

    @property
    def matched(self) -> MatchedCostBudget:
        return self.arms[0].matched

    def reading(self, control: str) -> DualMetricReading:
        for arm in self.arms:
            if arm.control == control:
                return arm.reading
        raise ReducibleNoveltyRefusal(f"control {control!r} absent from the family")

    def best_on(self, axis: str) -> float:
        if axis not in DUAL_AXES:
            raise ReducibleNoveltyRefusal(f"unknown metric axis {axis!r}")
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
        raise ReducibleNoveltyRefusal("declared control set drifted in membership or order")


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyContract:

    schema: str
    axes: tuple[str, ...]
    controls: tuple[str, ...]
    matched_cost_required: bool
    both_axes_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != REDUCIBLE_NOVELTY_SCHEMA:
            raise ReducibleNoveltyRefusal(f"unsupported contract schema {self.schema!r}")
        if tuple(self.axes) != DUAL_AXES:
            raise ReducibleNoveltyRefusal(
                "axis set or order drift; both learning progress and allocation efficiency required"
            )
        assert_control_completeness(self.controls)
        if not self.matched_cost_required:
            raise ReducibleNoveltyRefusal(
                "reducible novelty claim must require matched full-system cost"
            )
        if not self.both_axes_required:
            raise ReducibleNoveltyRefusal(
                "a single-axis win is exactly the noise trap; both axes must be required"
            )
        if self.replication_min < 2:
            raise ReducibleNoveltyRefusal("joint claim requires at least two independent replications")
        if self.prior_null != PRIOR_NULL:
            raise ReducibleNoveltyRefusal(
                "contract must name the irreducible-noise-trap as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("contract claim scope cannot be widened")

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


@dataclass(frozen=True, slots=True)
class AxisComparison:

    axis: str
    candidate_value: float
    best_control_value: float

    def __post_init__(self) -> None:
        if self.axis not in DUAL_AXES:
            raise ReducibleNoveltyRefusal(f"unknown metric axis {self.axis!r}")
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
    learning_progress: AxisComparison
    allocation_efficiency: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != REDUCIBLE_NOVELTY_SCHEMA:
            raise ReducibleNoveltyRefusal(f"unsupported verdict schema {self.schema!r}")
        if self.learning_progress.axis != "learning_progress":
            raise ReducibleNoveltyRefusal("first comparison must be the learning progress axis")
        if self.allocation_efficiency.axis != "allocation_efficiency":
            raise ReducibleNoveltyRefusal("second comparison must be the allocation efficiency axis")
        if not self.matched_cost_required:
            raise ReducibleNoveltyRefusal("verdict must be computed at matched full-system cost")
        if self.prior_null != PRIOR_NULL:
            raise ReducibleNoveltyRefusal(
                "verdict must name the irreducible-noise-trap as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("verdict claim scope cannot be widened")

    @property
    def both_axes_improved(self) -> bool:
        return self.learning_progress.improved and self.allocation_efficiency.improved

    @property
    def only_one_axis_improved(self) -> bool:
        return self.learning_progress.improved != self.allocation_efficiency.improved

    def certify(self) -> JointImprovementVerdict:

        if self.only_one_axis_improved:
            winner = "learning_progress" if self.learning_progress.improved else "allocation_efficiency"
            loser = "allocation_efficiency" if self.learning_progress.improved else "learning_progress"
            raise ReducibleNoveltyRefusal(
                f"single-axis win refused: {winner} improved but {loser} did not; "
                f"this is the {PRIOR_NULL}, not a joint improvement"
            )
        if not self.both_axes_improved:
            raise ReducibleNoveltyRefusal(
                f"no-axis win refused: neither learning progress nor allocation efficiency improved; "
                f"the {PRIOR_NULL} holds"
            )
        return self

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "learning_progress": self.learning_progress.payload(),
            "allocation_efficiency": self.allocation_efficiency.payload(),
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
        raise ReducibleNoveltyRefusal(
            "candidate is not held to the control family matched budget; comparison refused"
        )
    progress_cmp = AxisComparison(
        axis="learning_progress",
        candidate_value=candidate.learning_progress,
        best_control_value=controls.best_on("learning_progress"),
    )
    efficiency_cmp = AxisComparison(
        axis="allocation_efficiency",
        candidate_value=candidate.allocation_efficiency,
        best_control_value=controls.best_on("allocation_efficiency"),
    )
    return JointImprovementVerdict(
        schema=REDUCIBLE_NOVELTY_SCHEMA,
        learning_progress=progress_cmp,
        allocation_efficiency=efficiency_cmp,
        matched_cost_required=True,
        prior_null=PRIOR_NULL,
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
            raise ReducibleNoveltyRefusal("receipt must attest at least two independent replications")
        if not self.matched_cost_attested:
            raise ReducibleNoveltyRefusal("receipt must attest that the win held at matched cost")
        _require_id(self.independent_reviewer, "receipt.independent_reviewer")
        if self.claim_scope != CLAIM_SCOPE:
            raise ReducibleNoveltyRefusal("receipt claim scope cannot be widened")

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
            raise ReducibleNoveltyRefusal("gate claim scope cannot be widened")

    def authorize(
        self,
        verdict: JointImprovementVerdict,
        receipt: ConfirmationReceipt | None = None,
    ) -> JointImprovementVerdict:

        if not self.activation_permitted:
            raise ReducibleNoveltyRefusal(
                "joint-claim activation is not earned; the gate is closed by default and local code "
                "cannot open it. Route to an external replication with a signed confirmation receipt."
            )
        if receipt is None:
            raise ReducibleNoveltyRefusal("gate authorization requires an external confirmation receipt")
        verdict.certify()
        if receipt.verdict_digest != verdict.digest():
            raise ReducibleNoveltyRefusal("receipt does not confirm this exact verdict")
        return verdict

    def payload(self) -> dict[str, Any]:
        return {"activation_permitted": self.activation_permitted, "claim_scope": self.claim_scope}


def coverage() -> dict[str, Sequence[str]]:

    return {
        "curiosity-targets-reducible-not-raw-novelty": (
            "novelty_chaser is declared as the raw-novelty pole of the trap",
            "ControlFamily forces the chaser and the uniform pole into one matched-budget comparison",
        ),
        "learning-progress-and-efficiency-improve-jointly": (
            "JointImprovementVerdict.certify refuses any single-axis win",
            "DUAL_AXES fixes learning_progress and allocation_efficiency as jointly required",
        ),
        "improvement-is-at-matched-cost": (
            "MatchedCostBudget must be non-vacuous and equal across candidate and controls",
            "evaluate_joint_improvement refuses a candidate not held to the family budget",
        ),
        "raw-novelty-is-not-a-complete-theory-of-curiosity": (
            "PRIOR_NULL pins the irreducible-noise-trap as the default hypothesis",
            "the seeded toy lands the candidate below at least one axis maximum, so the null holds",
        ),
    }


SCIENTIFIC_CAPABILITY_CLAIM = False
