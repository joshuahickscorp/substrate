
from __future__ import annotations

from dataclasses import dataclass

from ..ladder.ladder_contracts import MatchedBudget
from .trace_stability_impl import (
    K_ITEMS,
    MECHANISM_ID,
    SeedMeasurement,
    favorable_measurement,
    null_measurement,
)
from .trace_stability_scaffold import MIN_SEEDS, REQUIRED_CONTROLS


@dataclass(frozen=True, slots=True)
class TraceStabilityBed:

    mechanism_id: str = MECHANISM_ID

    def controls(self) -> tuple[str, ...]:

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(
            params=K_ITEMS * MIN_SEEDS,
            flops=K_ITEMS * K_ITEMS * MIN_SEEDS,
            wall_ns=1_000_000,
            seeds=MIN_SEEDS,
        )

    def null_regime(self, seed: int) -> SeedMeasurement:

        return null_measurement(seed)

    def favorable_regime(self, seed: int) -> SeedMeasurement:

        return favorable_measurement(seed)
