"""Deterministic event formation bed: a null regime, a favorable regime, and the declared controls.

The bed is the toy task environment the event formation runner is scored against. Its null regime
carries events with no relational-temporal signal, so any trigger loses utility and the strong null
holds. Its favorable regime carries relational-temporal structure, so a proper event former can
preserve utility while cutting charged compute. The bed only declares the control family and the
matched cost; it runs no arm and mints no receipt.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ladder.ladder_contracts import MatchedBudget
from .event_formation_impl import FAVORABLE_REGIME, NULL_REGIME, Tick, build_episode
from .event_formation_scaffold import REQUIRED_CONTROLS


@dataclass(frozen=True, slots=True)
class EventFormationBed:
    """A Bed for the event formation mechanism, positioned at the Stage 3 event-formation question."""

    mechanism_id: str = "event_formation"

    def controls(self) -> tuple[str, ...]:
        """The full declared control family, in canonical order."""

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every arm is held to before any comparison."""

        return MatchedBudget(params=24, flops=4096, wall_ns=100_000, seeds=8)

    def null_regime(self, seed: int) -> tuple[Tick, ...]:
        """Events carry no relational-temporal signal, so any trigger loses utility (null holds)."""

        return build_episode(NULL_REGIME, seed)

    def favorable_regime(self, seed: int) -> tuple[Tick, ...]:
        """Relational-temporal structure lets a proper former preserve utility while cutting compute."""

        return build_episode(FAVORABLE_REGIME, seed)
