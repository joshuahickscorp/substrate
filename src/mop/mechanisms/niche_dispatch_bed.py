
from __future__ import annotations

from dataclasses import dataclass

from ..ladder.stage_ladder import MatchedBudget
from .niche_dispatch_impl import (
    RegimeData,
    build_disjoint_regime,
    build_overlapping_regime,
)
from .niche_dispatch_scaffold import DISPATCH_CONTROLS

MECHANISM_ID = "niche_dispatch"


class NicheDispatchBedError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NicheDispatchBed:

    mechanism_id: str = MECHANISM_ID
    n_perspectives: int = 3
    contexts_per_cell: int = 4
    favorable_disjoint: bool = True

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise NicheDispatchBedError(f"unexpected mechanism id {self.mechanism_id!r}")
        if self.n_perspectives < 3:
            raise NicheDispatchBedError("a niche-dispatch bed needs at least three perspectives")
        if self.contexts_per_cell < 1:
            raise NicheDispatchBedError("each niche cell needs at least one context")

    def controls(self) -> tuple[str, ...]:

        return DISPATCH_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(params=256, flops=8192, wall_ns=64, seeds=8)

    def null_regime(self, seed: int) -> RegimeData:

        return build_overlapping_regime(
            "null",
            n_perspectives=self.n_perspectives,
            contexts_per_cell=self.contexts_per_cell,
        )

    def favorable_regime(self, seed: int) -> RegimeData:

        if self.favorable_disjoint:
            return build_disjoint_regime(
                "favorable",
                n_perspectives=self.n_perspectives,
                contexts_per_cell=self.contexts_per_cell,
            )
        return build_overlapping_regime(
            "favorable",
            n_perspectives=self.n_perspectives,
            contexts_per_cell=self.contexts_per_cell,
        )


def build_default_bed() -> NicheDispatchBed:

    return NicheDispatchBed()
