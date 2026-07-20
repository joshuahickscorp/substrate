
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256

CONSTRUCTION_SEARCH_BED_SCHEMA = "mop-construction-search-bed/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

MECHANISM_ID = "construction_search"
REQUIREMENT_ID = "s3.construction_search"
STAGE = 3

CHEAP_CONTROLS: tuple[str, ...] = ("no-search", "random-construction", "greedy-only")
SEARCH_ARM = "construction-search"
ORACLE_REFERENCE = "oracle-headroom"
ALL_ARMS: tuple[str, ...] = (*CHEAP_CONTROLS, SEARCH_ARM, ORACLE_REFERENCE)

FORMATION_DEFAULT: tuple[int, ...] = (0, 1)
SPECIALISTS: tuple[int, ...] = (0, 1, 2)
SYNERGY_PAIR: tuple[int, ...] = (6, 7)

DEFAULT_NUM_MEMBERS = 12
DEFAULT_NUM_TASKS = 3
HIGH_AFFINITY = 0.9
LOW_AFFINITY = 0.1
JUNK_AFFINITY = 0.2
DEFAULT_SIZE_PENALTY = 0.05
DEFAULT_SYNERGY_BONUS = 1.0
DEFAULT_PER_EVAL_COST = 0.0002
DEFAULT_SEARCH_RESTARTS = 40
DEFAULT_RANDOM_SAMPLES = 2500
NOISE_SCALE = 0.02

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")

_MASK64 = (1 << 64) - 1
_LCG_MULT = 6364136223846793005
_LCG_ADD = 1442695040888963407
_MEMBER_SALT = 0x9E3779B97F4A7C15
_TASK_SALT = 0xD1B54A32D192ED03


class ConstructionSearchBedRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise ConstructionSearchBedRefusal(f"{label} must use stable lowercase characters")


def _mix(seed: int, member: int, task: int) -> float:

    state = (seed & _MASK64) * _LCG_MULT + _LCG_ADD
    state = (state + member * _MEMBER_SALT + task * _TASK_SALT) & _MASK64
    state = (state * _LCG_MULT + _LCG_ADD) & _MASK64
    return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def _base_affinity(member: int, task: int) -> float:

    if member in SPECIALISTS:
        return HIGH_AFFINITY if task == member else LOW_AFFINITY
    if member in SYNERGY_PAIR:
        return LOW_AFFINITY
    return JUNK_AFFINITY


def _build_affinity(seed: int, num_members: int, num_tasks: int) -> tuple[tuple[float, ...], ...]:

    return tuple(
        tuple(
            _base_affinity(member, task) + _mix(seed, member, task) * NOISE_SCALE
            for task in range(num_tasks)
        )
        for member in range(num_members)
    )


@dataclass(frozen=True, slots=True)
class RegimeSpec:

    name: str
    seed: int
    num_members: int
    num_tasks: int
    affinity: tuple[tuple[float, ...], ...]
    size_penalty: float
    synergy_pair: tuple[int, ...]
    synergy_bonus: float
    per_eval_cost: float
    search_restarts: int
    random_samples: int
    formation_default: tuple[int, ...]
    schema: str = CONSTRUCTION_SEARCH_BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != CONSTRUCTION_SEARCH_BED_SCHEMA:
            raise ConstructionSearchBedRefusal(f"unsupported regime schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchBedRefusal("regime claim scope cannot be widened")
        if self.name not in ("null", "favorable"):
            raise ConstructionSearchBedRefusal(f"unsupported regime name {self.name!r}")
        if self.seed < 0:
            raise ConstructionSearchBedRefusal("regime seed must be nonnegative")
        if self.num_members < 4:
            raise ConstructionSearchBedRefusal("regime needs at least four candidate members")
        if self.num_tasks < 2:
            raise ConstructionSearchBedRefusal("a multi-task regime needs at least two tasks")
        if len(self.affinity) != self.num_members:
            raise ConstructionSearchBedRefusal("affinity row count must equal the member count")
        for row in self.affinity:
            if len(row) != self.num_tasks:
                raise ConstructionSearchBedRefusal("each affinity row must cover every task")
        if not self.size_penalty > 0.0:
            raise ConstructionSearchBedRefusal("regime size penalty must be positive")
        if self.per_eval_cost < 0.0:
            raise ConstructionSearchBedRefusal("per-evaluation cost cannot be negative")
        if self.synergy_bonus < 0.0:
            raise ConstructionSearchBedRefusal("synergy bonus cannot be negative")
        if self.search_restarts <= 0 or self.random_samples <= 0:
            raise ConstructionSearchBedRefusal("search restarts and random samples must be positive")
        for member in (*self.synergy_pair, *self.formation_default):
            if not 0 <= member < self.num_members:
                raise ConstructionSearchBedRefusal("member index out of range")
        if self.synergy_bonus > 0.0 and len(self.synergy_pair) < 2:
            raise ConstructionSearchBedRefusal("a paying synergy needs at least two members")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "seed": self.seed,
            "num_members": self.num_members,
            "num_tasks": self.num_tasks,
            "affinity": [list(row) for row in self.affinity],
            "size_penalty": self.size_penalty,
            "synergy_pair": list(self.synergy_pair),
            "synergy_bonus": self.synergy_bonus,
            "per_eval_cost": self.per_eval_cost,
            "search_restarts": self.search_restarts,
            "random_samples": self.random_samples,
            "formation_default": list(self.formation_default),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ConstructionSearchBed:

    mechanism_id: str = MECHANISM_ID
    synergy_bonus: float = DEFAULT_SYNERGY_BONUS
    per_eval_cost: float = DEFAULT_PER_EVAL_COST
    size_penalty: float = DEFAULT_SIZE_PENALTY
    num_members: int = DEFAULT_NUM_MEMBERS
    num_tasks: int = DEFAULT_NUM_TASKS
    search_restarts: int = DEFAULT_SEARCH_RESTARTS
    random_samples: int = DEFAULT_RANDOM_SAMPLES
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_id(self.mechanism_id, "ConstructionSearchBed.mechanism_id")
        if self.claim_scope != CLAIM_SCOPE:
            raise ConstructionSearchBedRefusal("bed claim scope cannot be widened")
        if self.synergy_bonus < 0.0:
            raise ConstructionSearchBedRefusal("bed synergy bonus cannot be negative")
        if self.per_eval_cost < 0.0:
            raise ConstructionSearchBedRefusal("bed per-evaluation cost cannot be negative")
        if self.num_members < 4 or self.num_tasks < 2:
            raise ConstructionSearchBedRefusal("bed needs at least four members and two tasks")

    def controls(self) -> tuple[str, ...]:
        return CHEAP_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        return MatchedBudget(
            params=self.num_members * self.num_tasks,
            flops=self.random_samples * self.num_members,
            wall_ns=1000,
            seeds=8,
        )

    def _regime(self, name: str, seed: int, synergy_bonus: float) -> RegimeSpec:
        if seed < 0:
            raise ConstructionSearchBedRefusal("regime seed must be nonnegative")
        pair = SYNERGY_PAIR if synergy_bonus > 0.0 else ()
        return RegimeSpec(
            name=name,
            seed=seed,
            num_members=self.num_members,
            num_tasks=self.num_tasks,
            affinity=_build_affinity(seed, self.num_members, self.num_tasks),
            size_penalty=self.size_penalty,
            synergy_pair=pair,
            synergy_bonus=synergy_bonus,
            per_eval_cost=self.per_eval_cost,
            search_restarts=self.search_restarts,
            random_samples=self.random_samples,
            formation_default=FORMATION_DEFAULT,
        )

    def null_regime(self, seed: int) -> RegimeSpec:
        return self._regime("null", seed, synergy_bonus=0.0)

    def favorable_regime(self, seed: int) -> RegimeSpec:
        return self._regime("favorable", seed, synergy_bonus=self.synergy_bonus)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": CONSTRUCTION_SEARCH_BED_SCHEMA,
            "mechanism_id": self.mechanism_id,
            "synergy_bonus": self.synergy_bonus,
            "per_eval_cost": self.per_eval_cost,
            "size_penalty": self.size_penalty,
            "num_members": self.num_members,
            "num_tasks": self.num_tasks,
            "search_restarts": self.search_restarts,
            "random_samples": self.random_samples,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
