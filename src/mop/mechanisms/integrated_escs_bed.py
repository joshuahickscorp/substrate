
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import MatchedBudget
from ..substrate.events import canonical_sha256
from .integrated_escs_scaffold import (
    CLAIM_SCOPE,
    INTEGRATED_ESCS_SCHEMA,
    MECHANISM_LADDER,
    REQUIRED_BASELINES,
    IntegratedEscsRefusal,
)

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")

_BASE_DEMANDS: tuple[float, ...] = (0.40, 0.50, 0.60, 0.50)

NULL_ZERO_INDEX = len(MECHANISM_LADDER) - 1


def _seed_unit(seed: int, index: int) -> float:

    digest = canonical_sha256({"seed": seed, "index": index})
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _favorable_demand(seed: int, index: int) -> float:

    value = _BASE_DEMANDS[index] + _seed_unit(seed, index) * 0.10
    return round(min(0.90, value), 9)


@dataclass(frozen=True, slots=True)
class EscsRegime:

    name: str
    demands: tuple[float, ...]
    seed: int
    schema: str = INTEGRATED_ESCS_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported regime schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("regime claim scope cannot be widened")
        if _ID_RE.fullmatch(self.name) is None:
            raise IntegratedEscsRefusal("EscsRegime.name must use stable lowercase characters")
        if len(self.demands) != len(MECHANISM_LADDER):
            raise IntegratedEscsRefusal("regime demands must cover every mechanism on the ladder")
        if self.seed < 0:
            raise IntegratedEscsRefusal("regime seed must be nonnegative")
        for demand in self.demands:
            if demand != demand or demand in (float("inf"), float("-inf")):
                raise IntegratedEscsRefusal("regime demand must be a finite number")
            if not 0.0 <= demand <= 1.0:
                raise IntegratedEscsRefusal("regime demand must lie in [0.0, 1.0]")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "name": self.name,
            "demands": list(self.demands),
            "seed": self.seed,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class IntegratedEscsBed:

    mechanism_id: str = "integrated_escs"

    def controls(self) -> tuple[str, ...]:

        return REQUIRED_BASELINES

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(params=1000, flops=2000, wall_ns=100, seeds=8)

    def _favorable_demands(self, seed: int) -> tuple[float, ...]:
        return tuple(_favorable_demand(seed, index) for index in range(len(MECHANISM_LADDER)))

    def null_regime(self, seed: int) -> EscsRegime:

        demands = list(self._favorable_demands(seed))
        demands[NULL_ZERO_INDEX] = 0.0
        return EscsRegime(name="null", demands=tuple(demands), seed=seed)

    def favorable_regime(self, seed: int) -> EscsRegime:

        return EscsRegime(name="favorable", demands=self._favorable_demands(seed), seed=seed)


def build_default_bed() -> IntegratedEscsBed:

    return IntegratedEscsBed()
