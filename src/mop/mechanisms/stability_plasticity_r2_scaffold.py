"""Scaffold spine for the redesigned stability vs plasticity mechanism cluster (lane G1-P1R).

This module raises the SCAFFOLDING axis only, for the REDESIGNED stable-core bed. It encodes, as
machine-checkable contracts, the exact bar the redesigned mechanism must clear before any claim: a
stable core, orthogonal per-task adapters, and selective consolidation driven by an honest
recurrence signal must improve BOTH retention AND future learnability, jointly, at matched cost,
against the declared control family (fresh_init, frozen_core, full_retrain, no_replay). It builds
the harness, not the result. Nothing here demonstrates that any mechanism clears the bar.

Why a redesign exists at all: the old G1-P1 bed gave the mechanism no exploitable recurrence
signal, so the old lane was canary-pruned. The r2 bed (doc 20 route: repaired bed under new
authority) carries an honest recurrence signal in its favorable regime; this scaffold pins the same
P6 prior null and the same joint bar, so the redesign changes the bed, never the standard.

Named prior null (forces the bar): the P6 stability vs plasticity split. Fresh actors adapt but
forget; frozen actors retain but fail to adapt. The split is the default hypothesis. A single-axis
win is exactly what the null predicts, so a single-axis win is refused. The scaffold fails closed
unless both axes strictly improve together at matched cost, and even then a claim stays quarantined
behind an activation gate that local code cannot open without an external confirmation receipt.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. The controls are declarations. The verdict is arithmetic over declared
readings, never a measurement of a real system.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

STABILITY_PLASTICITY_R2_SCHEMA = "mop-stability-plasticity-r2/v1"

# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this scaffold module has no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The named prior null this lane must clear. The redesign keeps the SAME null as the retired G1-P1
# lane; a repaired bed never gets a softer hypothesis.
PRIOR_NULL = "p6-stability-plasticity-split"

# The two axes that must improve jointly. Ordering is load-bearing for digests and completeness.
DUAL_AXES: tuple[str, ...] = ("retention", "future_learnability")

# The declared control family, same family as the retired G1-P1 lane. Ordering is load-bearing; a
# completeness check refuses drift.
REQUIRED_CONTROLS: tuple[str, ...] = ("fresh_init", "frozen_core", "full_retrain", "no_replay")


class StabilityPlasticityR2Refusal(ValueError):
    """Raised whenever a declaration is missing, malformed, widened, or below the joint bar."""


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise StabilityPlasticityR2Refusal(f"{label} must use stable lowercase characters")


def _require_unit(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StabilityPlasticityR2Refusal(f"{label} must be a real number")
    if not math.isfinite(float(value)):
        raise StabilityPlasticityR2Refusal(f"{label} must be finite")
    if not 0.0 <= float(value) <= 1.0:
        raise StabilityPlasticityR2Refusal(f"{label} must lie in the unit interval [0, 1]")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise StabilityPlasticityR2Refusal(f"{label} must be a lowercase SHA-256 digest")


# ---------------------------------------------------------------------------
# Section A. Dual metric reading. Retention and future learnability, normalized to [0, 1].
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualMetricReading:
    """One condition's normalized score on both axes. Neither axis stands alone.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    retention: float
    future_learnability: float
    claim_scope: str = CLAIM_SCOPE
    schema: str = STABILITY_PLASTICITY_R2_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_R2_SCHEMA:
            raise StabilityPlasticityR2Refusal(f"unsupported reading schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("reading claim scope cannot be widened")
        _require_unit(self.retention, "reading.retention")
        _require_unit(self.future_learnability, "reading.future_learnability")

    def axis(self, name: str) -> float:
        if name == "retention":
            return self.retention
        if name == "future_learnability":
            return self.future_learnability
        raise StabilityPlasticityR2Refusal(f"unknown metric axis {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "retention": self.retention,
            "future_learnability": self.future_learnability,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section B. Matched cost budget. A comparison is honest only at equal cost.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MatchedCostBudget:
    """The full-system budget every arm, candidate and control, must be held to before comparison."""

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
                raise StabilityPlasticityR2Refusal(f"matched budget {name} must be an integer")
            if value <= 0:
                raise StabilityPlasticityR2Refusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "replay_samples": self.replay_samples,
            "update_steps": self.update_steps,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Section C. Control arm and control family declaration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlArm:
    """One declared control condition with its reading and its held budget.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    control: str
    reading: DualMetricReading
    matched: MatchedCostBudget
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.control not in REQUIRED_CONTROLS:
            raise StabilityPlasticityR2Refusal(f"unsupported control {self.control!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("control arm claim scope cannot be widened")

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
    """The complete declared control family; membership and order must match REQUIRED_CONTROLS."""

    schema: str
    arms: tuple[ControlArm, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_R2_SCHEMA:
            raise StabilityPlasticityR2Refusal(f"unsupported control family schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("control family claim scope cannot be widened")
        if tuple(arm.control for arm in self.arms) != REQUIRED_CONTROLS:
            raise StabilityPlasticityR2Refusal(
                "control family membership or order drift; expected fresh_init, frozen_core, "
                "full_retrain, no_replay in that order"
            )
        budgets = {arm.matched.digest() for arm in self.arms}
        if len(budgets) != 1:
            raise StabilityPlasticityR2Refusal("control arms are not held to one matched budget")

    @property
    def matched(self) -> MatchedCostBudget:
        return self.arms[0].matched

    def reading(self, control: str) -> DualMetricReading:
        for arm in self.arms:
            if arm.control == control:
                return arm.reading
        raise StabilityPlasticityR2Refusal(f"control {control!r} absent from the family")

    def best_on(self, axis: str) -> float:
        if axis not in DUAL_AXES:
            raise StabilityPlasticityR2Refusal(f"unknown metric axis {axis!r}")
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
    """Module-level fail-closed check that a declared control set matches membership and order."""

    if tuple(controls) != REQUIRED_CONTROLS:
        raise StabilityPlasticityR2Refusal("declared control set drifted in membership or order")


# ---------------------------------------------------------------------------
# Section D. The contract that declares the joint bar.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StabilityPlasticityR2Contract:
    """Declares the joint-improvement rule: keep a mechanism only for a replicated, matched-cost,
    both-axes win against the full declared control family.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    schema: str
    axes: tuple[str, ...]
    controls: tuple[str, ...]
    matched_cost_required: bool
    both_axes_required: bool
    replication_min: int
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_R2_SCHEMA:
            raise StabilityPlasticityR2Refusal(f"unsupported contract schema {self.schema!r}")
        if tuple(self.axes) != DUAL_AXES:
            raise StabilityPlasticityR2Refusal(
                "axis set or order drift; both retention and future learnability required"
            )
        assert_control_completeness(self.controls)
        if not self.matched_cost_required:
            raise StabilityPlasticityR2Refusal(
                "stability vs plasticity claim must require matched full-system cost"
            )
        if not self.both_axes_required:
            raise StabilityPlasticityR2Refusal(
                "a single-axis win is exactly the P6 split; both axes must be required"
            )
        if self.replication_min < 2:
            raise StabilityPlasticityR2Refusal(
                "joint claim requires at least two independent replications"
            )
        if self.prior_null != PRIOR_NULL:
            raise StabilityPlasticityR2Refusal(
                "contract must name the p6-stability-plasticity-split as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("contract claim scope cannot be widened")

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


def default_contract() -> StabilityPlasticityR2Contract:
    """The canonical lane G1-P1R contract: both axes, matched cost, two replications, P6 null."""

    return StabilityPlasticityR2Contract(
        schema=STABILITY_PLASTICITY_R2_SCHEMA,
        axes=DUAL_AXES,
        controls=REQUIRED_CONTROLS,
        matched_cost_required=True,
        both_axes_required=True,
        replication_min=2,
        prior_null=PRIOR_NULL,
    )


# ---------------------------------------------------------------------------
# Section E. Axis comparison and joint verdict. The verdict certifies nothing on a single axis.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AxisComparison:
    """A candidate value vs the best control value on one axis, with a strict-improvement flag."""

    axis: str
    candidate_value: float
    best_control_value: float

    def __post_init__(self) -> None:
        if self.axis not in DUAL_AXES:
            raise StabilityPlasticityR2Refusal(f"unknown metric axis {self.axis!r}")
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
    """Arithmetic verdict over two axis comparisons. A single-axis win never certifies a claim.

    The verdict object itself is a neutral record; it does not raise on a single-axis result because
    a single-axis result is a valid, and expected, null outcome. The claim path, ``certify``, fails
    closed unless BOTH axes strictly improve at matched cost. That is the P6 split encoded as code.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    schema: str
    retention: AxisComparison
    future_learnability: AxisComparison
    matched_cost_required: bool
    prior_null: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STABILITY_PLASTICITY_R2_SCHEMA:
            raise StabilityPlasticityR2Refusal(f"unsupported verdict schema {self.schema!r}")
        if self.retention.axis != "retention":
            raise StabilityPlasticityR2Refusal("first comparison must be the retention axis")
        if self.future_learnability.axis != "future_learnability":
            raise StabilityPlasticityR2Refusal("second comparison must be the future learnability axis")
        if not self.matched_cost_required:
            raise StabilityPlasticityR2Refusal("verdict must be computed at matched full-system cost")
        if self.prior_null != PRIOR_NULL:
            raise StabilityPlasticityR2Refusal(
                "verdict must name the p6-stability-plasticity-split as its null"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("verdict claim scope cannot be widened")

    @property
    def both_axes_improved(self) -> bool:
        return self.retention.improved and self.future_learnability.improved

    @property
    def only_one_axis_improved(self) -> bool:
        return self.retention.improved != self.future_learnability.improved

    def certify(self) -> JointImprovementVerdict:
        """Fail closed unless both axes strictly improve. A single-axis win is the P6 split, refused."""

        if self.only_one_axis_improved:
            winner = "retention" if self.retention.improved else "future_learnability"
            loser = "future_learnability" if self.retention.improved else "retention"
            raise StabilityPlasticityR2Refusal(
                f"single-axis win refused: {winner} improved but {loser} did not; "
                f"this is the {PRIOR_NULL}, not a joint improvement"
            )
        if not self.both_axes_improved:
            raise StabilityPlasticityR2Refusal(
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
    """Compare a candidate reading against the best control per axis, at matched cost.

    Fails closed if the candidate is not held to the same budget as the control family. The verdict
    compares the candidate to the strongest control on EACH axis independently; the P6 split makes
    that pair of maxima come from different controls, which is exactly why a joint win is hard.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    if candidate_budget.digest() != controls.matched.digest():
        raise StabilityPlasticityR2Refusal(
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
        schema=STABILITY_PLASTICITY_R2_SCHEMA,
        retention=retention_cmp,
        future_learnability=future_cmp,
        matched_cost_required=True,
        prior_null=PRIOR_NULL,
    )


# ---------------------------------------------------------------------------
# Section F. Activation gate. A joint claim stays quarantined until an external receipt opens it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfirmationReceipt:
    """An external confirmation receipt; the only thing that can open the joint-claim gate.

    Claim scope: deterministic programmatic mechanics only; no capability claim. This receipt is a
    declaration that an independent party replicated the joint, matched-cost win. Local code cannot
    mint a valid one for itself; a test asserts the gate stays closed without it.
    """

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
            raise StabilityPlasticityR2Refusal(
                "receipt must attest at least two independent replications"
            )
        if not self.matched_cost_attested:
            raise StabilityPlasticityR2Refusal("receipt must attest that the win held at matched cost")
        _require_id(self.independent_reviewer, "receipt.independent_reviewer")
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("receipt claim scope cannot be widened")

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
    """A fail-closed activation gate. OFF by default; opening it needs a matching external receipt.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The gate exists so
    that a merely arithmetic both-axes win in this process can never be promoted to a standing claim
    without independent, matched-cost, replicated confirmation supplied from outside.
    """

    activation_permitted: bool = False
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.claim_scope != CLAIM_SCOPE:
            raise StabilityPlasticityR2Refusal("gate claim scope cannot be widened")

    def authorize(
        self,
        verdict: JointImprovementVerdict,
        receipt: ConfirmationReceipt | None = None,
    ) -> JointImprovementVerdict:
        """Fail closed. Raise unless activation is permitted AND a valid receipt matches this verdict."""

        if not self.activation_permitted:
            raise StabilityPlasticityR2Refusal(
                "joint-claim activation is not earned; the gate is closed by default and local code "
                "cannot open it. Route to an external replication with a signed confirmation receipt."
            )
        if receipt is None:
            raise StabilityPlasticityR2Refusal(
                "gate authorization requires an external confirmation receipt"
            )
        verdict.certify()
        if receipt.verdict_digest != verdict.digest():
            raise StabilityPlasticityR2Refusal("receipt does not confirm this exact verdict")
        return verdict

    def payload(self) -> dict[str, Any]:
        return {"activation_permitted": self.activation_permitted, "claim_scope": self.claim_scope}


# ---------------------------------------------------------------------------
# Section G. Coverage record for this lane's sub-questions (readiness only).
# ---------------------------------------------------------------------------


def coverage() -> dict[str, Sequence[str]]:
    """Static record of which G1-P1R sub-questions this scaffold arms. Readiness, not results."""

    return {
        "stable-core-coexists-with-rapid-adaptation": (
            "frozen_core and fresh_init arms are declared as the two poles of the P6 split",
            "ControlFamily forces both poles into one matched-budget comparison",
        ),
        "recurrence-signal-is-the-repaired-affordance": (
            "the r2 bed's favorable regime carries an honest recurrence signal the mechanism can",
            "exploit through selective consolidation; the old G1-P1 bed carried none, which is why",
            "the old lane canary-pruned; the bar itself is unchanged",
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
            "the null regime's same-subspace overwrites make the joint win impossible by construction",
        ),
    }


SCIENTIFIC_CAPABILITY_CLAIM = False
