"""A deterministic niche_dispatch bed with a null regime and a favorable regime.

The bed presents the two seeded task environments a niche-dispatch mechanism is judged on. Its null
regime encodes the named prior null (an EDCM invalid bed: overlapping, partly harmful niches with no
uniquely positive perspective, where forced universal complementarity is the wrong assumption), so a
dispatcher there can never beat every matched control. Its favorable regime has clean disjoint niches
built so a value-of-compute dispatcher wins outright, which lets the runner mint a mechanics-only
demonstration. The bed makes no capability or natural-data claim; both regimes are toy partitions.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

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
    """Raised when the bed is configured with a shape that cannot exercise the matched controls."""


@dataclass(frozen=True, slots=True)
class NicheDispatchBed:
    """A seeded bed exposing controls, a matched cost, and a null vs favorable regime.

    Claim scope: deterministic programmatic mechanics only; no capability claim. Set
    ``favorable_disjoint`` to False to degrade the favorable regime into an overlapping one, which is
    used to prove the runner fails closed when dispatch cannot beat every control.
    """

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
        """The declared matched-control family, reused from the fail-closed scaffold."""

        return DISPATCH_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every dispatch arm is held to before comparison."""

        return MatchedBudget(params=256, flops=8192, wall_ns=64, seeds=8)

    def null_regime(self, seed: int) -> RegimeData:
        """The EDCM invalid-bed null regime: overlapping, partly harmful niches; dispatch cannot win."""

        return build_overlapping_regime(
            "null",
            n_perspectives=self.n_perspectives,
            contexts_per_cell=self.contexts_per_cell,
        )

    def favorable_regime(self, seed: int) -> RegimeData:
        """The favorable regime: clean disjoint niches a value-of-compute dispatcher wins outright."""

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
    """Return the canonical niche_dispatch bed with a winnable favorable regime."""

    return NicheDispatchBed()
