
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .stability_plasticity_scaffold import REQUIRED_CONTROLS

MECHANISM_ID = "stability_plasticity"
BED_SCHEMA = "mop-stability-plasticity-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

DIM = 8
CORE_DIM = 4
HISTORY_TASKS = 4

CORE_SCALE = 1.0
ADAPT_SCALE = 0.8

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)

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


@dataclass(frozen=True, slots=True)
class TaskStream:

    regime: str
    seed: int
    history: tuple[Vector, ...]
    future: Vector
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
        if len(self.history) < 2:
            raise BedRefusal("a task stream needs at least two history tasks")
        for vector in (*self.history, self.future):
            if len(vector) != self.dim:
                raise BedRefusal("every task vector must match the declared dimension")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "seed": self.seed,
            "dim": self.dim,
            "core_dim": self.core_dim,
            "history": [list(vector) for vector in self.history],
            "future": list(self.future),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _favorable_task(seed: int, core: Vector, task_index: int) -> Vector:

    values: list[float] = []
    for dim in range(DIM):
        if dim < CORE_DIM:
            values.append(core[dim])
        else:
            values.append(ADAPT_SCALE * _signed(seed, f"fav.adapter.{task_index}.{dim}"))
    return tuple(values)


def _null_task(seed: int, task_index: int) -> Vector:

    values: list[float] = []
    for dim in range(DIM):
        if dim < CORE_DIM:
            values.append(CORE_SCALE * _signed(seed, f"null.core.{task_index}.{dim}"))
        else:
            values.append(ADAPT_SCALE * _signed(seed, f"null.adapter.{task_index}.{dim}"))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class StabilityPlasticityBed:

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

        history = tuple(_null_task(seed, index) for index in range(HISTORY_TASKS))
        future = _null_task(seed, HISTORY_TASKS)
        return TaskStream(regime=REGIME_NULL, seed=seed, history=history, future=future)

    def favorable_regime(self, seed: int) -> TaskStream:

        core = tuple(
            CORE_SCALE * _signed(seed, f"fav.core.{dim}") if dim < CORE_DIM else 0.0
            for dim in range(DIM)
        )
        history = tuple(_favorable_task(seed, core, index) for index in range(HISTORY_TASKS))
        future = _favorable_task(seed, core, HISTORY_TASKS)
        return TaskStream(regime=REGIME_FAVORABLE, seed=seed, history=history, future=future)

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
