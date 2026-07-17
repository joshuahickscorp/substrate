"""Deterministic task-stream bed for the REDESIGNED stability vs plasticity mechanism (lane G1-P1R).

This module is runnable machinery, not a scaffold and not evidence. It is the doc 20 route: a
repaired bed under new authority; the old G1-P1 lane was canary-pruned because its bed gave the
mechanism no exploitable recurrence signal. The r2 bed builds the two task-stream regimes the
redesigned mechanism must be measured on:

- NULL regime: conflicting same-subspace overwrites. Every task carries a fresh, sign-flipped core
  in the SAME core subspace, so retaining an old task and acquiring a new one pull against each
  other; the future task's core is sign-opposed to task zero's core by construction, so a frozen
  core is guaranteed wrong on the future task. The recurrence signal is silent (all flags False),
  so there is nothing for selective consolidation to exploit. On this stream the P6 split holds by
  construction: the mechanism cannot beat every control on both axes at once.
- FAVORABLE regime: a recurring shared core with small orthogonal per-task adapters AND an honest
  recurrence signal. All tasks share one seeded core; each task adds a bounded-magnitude adapter
  whose sign pattern is a fixed Hadamard row, so distinct tasks are guaranteed far apart in the
  adapter subspace. Exactly one interior history task is flagged as recurring, and the future task
  reuses that task's adapter byte-for-byte. A mechanism that consolidates the flagged adapter and
  reactivates it on recurrence can, in principle, win both axes; the honest flag is the repaired
  affordance the old bed lacked.

The regimes are mechanics only: seeded, byte-reproducible float tuples with no capability claim and
no natural data. The bed owns the task geometry; the learners live in the impl module and the
measurement and verdict live in the runner module.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .stability_plasticity_r2_scaffold import REQUIRED_CONTROLS

MECHANISM_ID = "stability_plasticity_r2"
BED_SCHEMA = "mop-stability-plasticity-r2-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# Geometry of the shared representation. Dims 0 to CORE_DIM-1 are the stable core subspace; dims
# CORE_DIM to DIM-1 are the plastic adapter subspace. History tasks are learned in order; one held
# out future task measures future-learnability.
DIM = 8
CORE_DIM = 4
HISTORY_TASKS = 4

# Bounded magnitude windows. Keeping every component's magnitude inside a window bounded away from
# zero is what makes the favorable joint win and the null impossibility PROVABLE for every seed:
# margins depend on sign geometry (fixed) and magnitudes (confined to these windows), never on a
# lucky draw.
CORE_MAG_LO = 0.55
CORE_MAG_HI = 0.90
ADAPT_MAG_LO = 0.60
ADAPT_MAG_HI = 0.90

# Fixed Hadamard sign rows for the adapter subspace, one row per history task index. Any two
# distinct rows differ in exactly two of the four adapter coordinates, so distinct tasks' adapters
# are at least 2 * ADAPT_MAG_LO apart per differing coordinate. Seed-independent by design.
ADAPTER_SIGN_ROWS: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1),
    (1, -1, 1, -1),
    (1, 1, -1, -1),
    (1, -1, -1, 1),
)

# Alternating core sign per history task in the null regime; the future core sign is forced to -1,
# the opposite of task zero, so a core frozen on task zero is wrong on the future task for every
# seed. This is the same-subspace overwrite made explicit.
NULL_CORE_SIGNS: tuple[int, ...] = (1, -1, 1, -1)
NULL_FUTURE_CORE_SIGN = -1

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)

# The honest recurrence index is always interior: never task zero (else a frozen_core control gets
# the recurrence for free) and never the last task (else fresh_init and no_replay end states get it
# for free). Interiority is part of the schema, validated by TaskStream.
NO_RECURRENCE = -1

Vector = tuple[float, ...]


class BedRefusal(ValueError):
    """Raised when a bed regime request is malformed or a declaration is widened."""


def _unit(seed: int, label: str) -> float:
    """A deterministic value in [0, 1) from a seeded digest; no wall clock, no rng."""

    if seed < 0:
        raise BedRefusal("bed seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


def _signed(seed: int, label: str) -> float:
    """A deterministic value in [-1, 1) from a seeded digest."""

    return 2.0 * _unit(seed, label) - 1.0


def _core_magnitude(seed: int, label: str) -> float:
    """A deterministic core magnitude inside [CORE_MAG_LO, CORE_MAG_HI)."""

    return CORE_MAG_LO + (CORE_MAG_HI - CORE_MAG_LO) * _unit(seed, label)


def _adapter_magnitude(seed: int, label: str) -> float:
    """A deterministic adapter magnitude inside [ADAPT_MAG_LO, ADAPT_MAG_HI)."""

    return ADAPT_MAG_LO + (ADAPT_MAG_HI - ADAPT_MAG_LO) * _unit(seed, label)


@dataclass(frozen=True, slots=True)
class TaskStream:
    """One regime rendered as concrete seeded task vectors plus the recurrence signal.

    ``history`` are the tasks learned in order; ``future`` is the held-out task on which future
    learnability is measured. ``recurrence_flags`` marks which history tasks recur; in the favorable
    regime the flag is honest (the future task reuses the flagged task's adapter exactly) and
    ``future_recurrence_index`` names the flagged task. In the null regime the signal is silent:
    all flags False and future_recurrence_index is NO_RECURRENCE. Claim scope: deterministic
    programmatic mechanics only.
    """

    regime: str
    seed: int
    history: tuple[Vector, ...]
    future: Vector
    recurrence_flags: tuple[bool, ...]
    future_recurrence_index: int
    dim: int = DIM
    core_dim: int = CORE_DIM
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported task stream schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("task stream claim scope cannot be widened")
        if self.regime not in REGIMES:
            raise BedRefusal(f"unknown regime {self.regime!r}")
        if not 0 < self.core_dim < self.dim:
            raise BedRefusal("core subspace must be a nonempty strict subset of the representation")
        if len(self.history) < 3:
            raise BedRefusal("a recurrence-bearing task stream needs at least three history tasks")
        for vector in (*self.history, self.future):
            if len(vector) != self.dim:
                raise BedRefusal("every task vector must match the declared dimension")
        if len(self.recurrence_flags) != len(self.history):
            raise BedRefusal("recurrence flags must cover every history task exactly once")
        flagged = tuple(index for index, flag in enumerate(self.recurrence_flags) if flag)
        if self.regime == REGIME_FAVORABLE:
            if len(flagged) != 1:
                raise BedRefusal("favorable regime must flag exactly one recurring history task")
            if self.future_recurrence_index != flagged[0]:
                raise BedRefusal("future recurrence index must name the flagged history task")
            if not 0 < self.future_recurrence_index < len(self.history) - 1:
                raise BedRefusal("the recurring task must be interior: never first, never last")
        else:
            if flagged:
                raise BedRefusal("null regime must keep the recurrence signal silent")
            if self.future_recurrence_index != NO_RECURRENCE:
                raise BedRefusal("null regime must carry future_recurrence_index NO_RECURRENCE")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "seed": self.seed,
            "dim": self.dim,
            "core_dim": self.core_dim,
            "history": [list(vector) for vector in self.history],
            "future": list(self.future),
            "recurrence_flags": list(self.recurrence_flags),
            "future_recurrence_index": self.future_recurrence_index,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _favorable_adapter(seed: int, task_index: int) -> Vector:
    """The adapter-subspace component for one favorable task: Hadamard signs, bounded magnitudes."""

    row = ADAPTER_SIGN_ROWS[task_index]
    return tuple(
        row[dim - CORE_DIM] * _adapter_magnitude(seed, f"favr2.adapter.{task_index}.{dim}")
        for dim in range(CORE_DIM, DIM)
    )


def _favorable_task(core: Vector, adapter: Vector) -> Vector:
    """Shared core plus a task-specific adapter delta orthogonal to the core."""

    return tuple(core[dim] if dim < CORE_DIM else adapter[dim - CORE_DIM] for dim in range(DIM))


def _null_task(seed: int, task_index: int, core_sign: int) -> Vector:
    """A conflicting task: a sign-flipped fresh core AND a fresh adapter, so nothing is shared."""

    values: list[float] = []
    for dim in range(DIM):
        if dim < CORE_DIM:
            values.append(core_sign * _core_magnitude(seed, f"nullr2.core.{task_index}.{dim}"))
        else:
            sign = 1 if _unit(seed, f"nullr2.adapter.sign.{task_index}.{dim}") < 0.5 else -1
            values.append(sign * _adapter_magnitude(seed, f"nullr2.adapter.{task_index}.{dim}"))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class StabilityPlasticityR2Bed:
    """The concrete redesigned bed. mechanism_id matches the scaffold and the runner.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

    mechanism_id: str = MECHANISM_ID
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise BedRefusal("bed mechanism_id drift")
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported bed schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("bed claim scope cannot be widened")

    def controls(self) -> tuple[str, ...]:
        """The declared control family: fresh_init, frozen_core, full_retrain, no_replay."""

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every arm is held to. Positive on every axis."""

        return MatchedBudget(params=DIM * DIM, flops=1_048_576, wall_ns=1_000_000, seeds=8)

    def null_regime(self, seed: int) -> TaskStream:
        """Conflicting same-subspace overwrites and a silent recurrence signal: the split holds.

        The future task's core sign is the opposite of task zero's, so any arm that freezes a core
        learned on task zero carries a guaranteed core error of at least CORE_MAG_LO + the frozen
        residual on every core coordinate of the future task; a full-dimension refit shrinks its
        error by the fit residual instead. That ordering is seed-independent, which is what makes
        the joint win impossible by construction rather than merely unlikely.
        """

        history = tuple(
            _null_task(seed, index, NULL_CORE_SIGNS[index]) for index in range(HISTORY_TASKS)
        )
        future = _null_task(seed, HISTORY_TASKS, NULL_FUTURE_CORE_SIGN)
        return TaskStream(
            regime=REGIME_NULL,
            seed=seed,
            history=history,
            future=future,
            recurrence_flags=tuple(False for _ in range(HISTORY_TASKS)),
            future_recurrence_index=NO_RECURRENCE,
        )

    def favorable_regime(self, seed: int) -> TaskStream:
        """A recurring shared core, orthogonal adapters, and an honest recurrence flag.

        The recurring task index is seeded but always interior (1 or 2), so no control's end state
        happens to sit on the recurring adapter: frozen_core holds task zero, fresh_init and
        no_replay end near the LAST task, and the last task's Hadamard row differs from every
        interior row in two adapter coordinates. Only an arm that reads the flag and consolidates
        can reactivate the recurring adapter on the future task.
        """

        core = tuple(
            (1 if _unit(seed, f"favr2.core.sign.{dim}") < 0.5 else -1)
            * _core_magnitude(seed, f"favr2.core.mag.{dim}")
            for dim in range(CORE_DIM)
        )
        adapters = tuple(_favorable_adapter(seed, index) for index in range(HISTORY_TASKS))
        recurring = 1 + (1 if _unit(seed, "favr2.recurring") >= 0.5 else 0)
        history = tuple(_favorable_task(core, adapters[index]) for index in range(HISTORY_TASKS))
        future = _favorable_task(core, adapters[recurring])
        return TaskStream(
            regime=REGIME_FAVORABLE,
            seed=seed,
            history=history,
            future=future,
            recurrence_flags=tuple(index == recurring for index in range(HISTORY_TASKS)),
            future_recurrence_index=recurring,
        )

    def regime(self, name: str, seed: int) -> TaskStream:
        """Dispatch to a regime by name. Fails closed on an unknown regime."""

        if name == REGIME_NULL:
            return self.null_regime(seed)
        if name == REGIME_FAVORABLE:
            return self.favorable_regime(seed)
        raise BedRefusal(f"unknown regime {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "dim": DIM,
            "core_dim": CORE_DIM,
            "history_tasks": HISTORY_TASKS,
            "controls": list(self.controls()),
            "matched_cost": self.matched_cost().payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
