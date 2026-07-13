"""The deterministic toy bed for the trace_stability epoch (a Bed per the ladder contract).

The bed supplies two seeded regimes and the epoch's declared control set. The null regime carries
no cross-seed structure so the named prior null holds (agreement stays at chance). The favorable
regime carries a real shared cross-seed structure so the metric can clear the toy threshold. Both
are mechanics only: no model weights, no network, no natural data. The control set and the matched
budget are the scaffold's declared bar, reused here without edit.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

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
    """A deterministic bed with a null regime, a favorable regime, and the declared controls."""

    mechanism_id: str = MECHANISM_ID

    def controls(self) -> tuple[str, ...]:
        """The scaffold's mandatory control set, in its load-bearing order."""

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched measurement budget covering at least the epoch minimum seeds."""

        return MatchedBudget(
            params=K_ITEMS * MIN_SEEDS,
            flops=K_ITEMS * K_ITEMS * MIN_SEEDS,
            wall_ns=1_000_000,
            seeds=MIN_SEEDS,
        )

    def null_regime(self, seed: int) -> SeedMeasurement:
        """A seeded trace with no cross-seed structure; the metric stays at chance and the null holds."""

        return null_measurement(seed)

    def favorable_regime(self, seed: int) -> SeedMeasurement:
        """A seeded trace with real cross-seed structure; the metric can clear the toy threshold."""

        return favorable_measurement(seed)
