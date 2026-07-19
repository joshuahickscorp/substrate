
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import MatchedBudget
from ..substrate.events import canonical_sha256
from .memory_organization_scaffold import FUTURE_DECISION_CONTROLS

MECHANISM_ID = "memory_organization"

ORGANIZED_ARM = "organized-memory"
ARMS: tuple[str, ...] = (ORGANIZED_ARM,) + FUTURE_DECISION_CONTROLS

N_ACTIONS = 4

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"

STRUCTURE_REUSABLE = "reusable-structure"
STRUCTURE_ABSENT = "no-reusable-structure"
STRUCTURE_PREDICTION_ONLY = "prediction-only"
STRUCTURE_HARMFUL = "memory-harms"
FAVORABLE_STRUCTURES: tuple[str, ...] = (
    STRUCTURE_REUSABLE,
    STRUCTURE_ABSENT,
    STRUCTURE_PREDICTION_ONLY,
    STRUCTURE_HARMFUL,
)


class MemoryBedRefusal(ValueError):
    pass


def action_of(value: int) -> int:

    return value % N_ACTIONS


@dataclass(frozen=True, slots=True)
class MemoryOp:

    step: int
    key: int
    value: int
    reliable: bool
    retract: bool

    def payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "key": self.key,
            "value": self.value,
            "reliable": self.reliable,
            "retract": self.retract,
        }


@dataclass(frozen=True, slots=True)
class DecisionQuery:

    key: int
    default_feature: int
    optimal_action: int
    true_value: int

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "default_feature": self.default_feature,
            "optimal_action": self.optimal_action,
            "true_value": self.true_value,
        }


@dataclass(frozen=True, slots=True)
class DecisionStream:

    regime: str
    structure: str
    seed: int
    ops: tuple[MemoryOp, ...]
    queries: tuple[DecisionQuery, ...]

    def __post_init__(self) -> None:
        if not self.queries:
            raise MemoryBedRefusal("a decision stream must carry at least one future decision")

    def payload(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "structure": self.structure,
            "seed": self.seed,
            "ops": [op.payload() for op in self.ops],
            "queries": [query.payload() for query in self.queries],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())



_A = 1
_B = 2
_C = 3
_W = 2
_DEFAULT = 0
_PA = 1
_PB = 1 + N_ACTIONS
_PDEFAULT = 1


def _revision_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_B, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_B), true_value=_B
    )
    return ops, query


def _provenance_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_C, reliable=False, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_A), true_value=_A
    )
    return ops, query


def _deletion_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=True),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_DEFAULT), true_value=_DEFAULT
    )
    return ops, query


def _organized_only_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_B, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_C, reliable=False, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_B), true_value=_B
    )
    return ops, query


def _misleading_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_W, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_W, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_DEFAULT), true_value=_DEFAULT
    )
    return ops, query


def _prediction_only_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    ops = [
        MemoryOp(step=0, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_PB, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_PDEFAULT, optimal_action=action_of(_PB), true_value=_PB
    )
    return ops, query


def _structureless_scenario(seed: int, key: int) -> tuple[list[MemoryOp], DecisionQuery]:

    optimal = int(canonical_sha256({"seed": seed, "key": key})[:8], 16) % N_ACTIONS
    ops = [MemoryOp(step=0, key=key, value=_PDEFAULT, reliable=True, retract=False)]
    query = DecisionQuery(
        key=key, default_feature=_PDEFAULT, optimal_action=optimal, true_value=_PDEFAULT
    )
    return ops, query


def _assemble(
    regime: str,
    structure: str,
    seed: int,
    scenarios: list[tuple[list[MemoryOp], DecisionQuery]],
) -> DecisionStream:
    ops: list[MemoryOp] = []
    queries: list[DecisionQuery] = []
    for scenario_ops, query in scenarios:
        ops.extend(scenario_ops)
        queries.append(query)
    return DecisionStream(
        regime=regime,
        structure=structure,
        seed=seed,
        ops=tuple(ops),
        queries=tuple(queries),
    )


def _require_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise MemoryBedRefusal("regime seed must be a nonnegative integer")


def _key_base(seed: int) -> int:
    return seed * 1000


@dataclass(frozen=True, slots=True)
class MemoryOrganizationBed:

    structure: str = STRUCTURE_REUSABLE
    mechanism_id: str = MECHANISM_ID

    def __post_init__(self) -> None:
        if self.structure not in FAVORABLE_STRUCTURES:
            raise MemoryBedRefusal(f"unsupported favorable structure {self.structure!r}")
        if self.mechanism_id != MECHANISM_ID:
            raise MemoryBedRefusal("memory-organization bed mechanism id cannot be renamed")

    def controls(self) -> tuple[str, ...]:
        return FUTURE_DECISION_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(params=256, flops=4096, wall_ns=1024, seeds=8)

    def null_regime(self, seed: int) -> DecisionStream:

        _require_seed(seed)
        base = _key_base(seed)
        scenarios = [_structureless_scenario(seed, base + index) for index in range(6)]
        return _assemble(REGIME_NULL, STRUCTURE_ABSENT, seed, scenarios)

    def favorable_regime(self, seed: int) -> DecisionStream:

        _require_seed(seed)
        base = _key_base(seed)
        if self.structure == STRUCTURE_REUSABLE:
            scenarios = [_revision_scenario(base + index) for index in range(3)]
            scenarios += [_provenance_scenario(base + 3 + index) for index in range(3)]
            scenarios += [_deletion_scenario(base + 6 + index) for index in range(3)]
        elif self.structure == STRUCTURE_ABSENT:
            scenarios = [_structureless_scenario(seed, base + index) for index in range(6)]
        elif self.structure == STRUCTURE_PREDICTION_ONLY:
            scenarios = [_prediction_only_scenario(base + index) for index in range(6)]
        else:
            scenarios = [_misleading_scenario(base + index) for index in range(4)]
            scenarios += [_organized_only_scenario(base + 4 + index) for index in range(2)]
        return _assemble(REGIME_FAVORABLE, self.structure, seed, scenarios)


def favorable_bed() -> MemoryOrganizationBed:

    return MemoryOrganizationBed(structure=STRUCTURE_REUSABLE)


def structureless_bed() -> MemoryOrganizationBed:

    return MemoryOrganizationBed(structure=STRUCTURE_ABSENT)


def prediction_only_bed() -> MemoryOrganizationBed:

    return MemoryOrganizationBed(structure=STRUCTURE_PREDICTION_ONLY)


def harmful_bed() -> MemoryOrganizationBed:

    return MemoryOrganizationBed(structure=STRUCTURE_HARMFUL)
