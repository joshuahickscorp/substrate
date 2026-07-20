
from __future__ import annotations

from dataclasses import dataclass

from ..ladder.ladder_contracts import MatchedBudget
from .event_formation_impl import FAVORABLE_REGIME, NULL_REGIME, Tick, build_episode
from .event_formation_scaffold import REQUIRED_CONTROLS


@dataclass(frozen=True, slots=True)
class EventFormationBed:

    mechanism_id: str = "event_formation"

    def controls(self) -> tuple[str, ...]:

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(params=24, flops=4096, wall_ns=100_000, seeds=8)

    def null_regime(self, seed: int) -> tuple[Tick, ...]:

        return build_episode(NULL_REGIME, seed)

    def favorable_regime(self, seed: int) -> tuple[Tick, ...]:

        return build_episode(FAVORABLE_REGIME, seed)
