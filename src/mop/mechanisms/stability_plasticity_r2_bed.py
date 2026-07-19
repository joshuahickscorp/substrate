
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .stability_plasticity_r2_scaffold import REQUIRED_CONTROLS

MECHANISM_ID = "stability_plasticity_r2"
BED_SCHEMA = "mop-stability-plasticity-r2-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

DIM = 8
CORE_DIM = 4
HISTORY_TASKS = 4

CORE_MAG_LO = 0.55
CORE_MAG_HI = 0.90
ADAPT_MAG_LO = 0.60
ADAPT_MAG_HI = 0.90

ADAPTER_SIGN_ROWS: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1),
    (1, -1, 1, -1),
    (1, 1, -1, -1),
    (1, -1, -1, 1),
)

NULL_CORE_SIGNS: tuple[int, ...] = (1, -1, 1, -1)
NULL_FUTURE_CORE_SIGN = -1

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)

NO_RECURRENCE = -1

Vector = tuple[float, ...]


class BedRefusal(ValueError):
    pass


def _unit(seed: int, label: str) -> float:

    if seed < 0:
        raise BedRefusal("bed seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


def _signed(seed: int, label: str) -> float:

    return 2.0 * _unit(seed, label) - 1.0


def _core_magnitude(seed: int, label: str) -> float:

    return CORE_MAG_LO + (CORE_MAG_HI - CORE_MAG_LO) * _unit(seed, label)


def _adapter_magnitude(seed: int, label: str) -> float:

    return ADAPT_MAG_LO + (ADAPT_MAG_HI - ADAPT_MAG_LO) * _unit(seed, label)


@dataclass(frozen=True, slots=True)
class TaskStream:

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

    row = ADAPTER_SIGN_ROWS[task_index]
    return tuple(
        row[dim - CORE_DIM] * _adapter_magnitude(seed, f"favr2.adapter.{task_index}.{dim}")
        for dim in range(CORE_DIM, DIM)
    )


def _favorable_task(core: Vector, adapter: Vector) -> Vector:

    return tuple(core[dim] if dim < CORE_DIM else adapter[dim - CORE_DIM] for dim in range(DIM))


def _null_task(seed: int, task_index: int, core_sign: int) -> Vector:

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

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(params=DIM * DIM, flops=1_048_576, wall_ns=1_000_000, seeds=8)

    def null_regime(self, seed: int) -> TaskStream:

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
