
from __future__ import annotations

from dataclasses import dataclass

from ..ladder.ladder_contracts import MatchedBudget
from .messaging_repair_impl import (
    DECLARED_CONTROLS,
    DEFAULT_ACTION_BUDGET,
    FAVORABLE_REGIME,
    MESSAGING_REPAIR_IMPL_SCHEMA,
    NULL_REGIME,
    Agent,
    MessagingRepairImplError,
    Offer,
    Regime,
)

MECHANISM_ID = "messaging_repair"
REQUIREMENT_ID = "s3.messaging_repair"


def _shift(seed: int) -> int:

    if seed < 0:
        raise MessagingRepairImplError("bed seed must be nonnegative")
    return seed % 5


@dataclass(frozen=True, slots=True)
class MessagingRepairBed:

    mechanism_id: str = MECHANISM_ID
    action_budget: int = DEFAULT_ACTION_BUDGET

    def controls(self) -> tuple[str, ...]:
        return DECLARED_CONTROLS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(
            params=self.action_budget,
            flops=self.action_budget * self.action_budget,
            wall_ns=1,
            seeds=8,
        )

    def null_regime(self, seed: int) -> Regime:

        delta = _shift(seed)
        truth = 100 + delta
        agents = (
            Agent("a0", 70 + delta, False, "sink"),
            Agent("a1", 130 + delta, False, "sink"),
            Agent("a2", 70 + delta, False, "sink"),
            Agent("a3", 130 + delta, False, "sink"),
            Agent("a4", 100 + delta, False, "sink"),
            Agent("a5", 100 + delta, False, "sink"),
        )
        offers = (
            Offer("a0", "a1", 70 + delta, False),
            Offer("a2", "a3", 70 + delta, False),
            Offer("a4", "a5", 100 + delta, False),
        )
        return Regime(
            schema=MESSAGING_REPAIR_IMPL_SCHEMA,
            kind=NULL_REGIME,
            seed=seed,
            truth=truth,
            agents=agents,
            offers=offers,
            action_budget=self.action_budget,
        )

    def favorable_regime(self, seed: int) -> Regime:

        delta = _shift(seed)
        truth = 100 + delta
        agents = (
            Agent("s0", 100 + delta, True, "source"),
            Agent("s1", 100 + delta, True, "source"),
            Agent("f0", 40 + delta, False, "flaky"),
            Agent("k0", 160 + delta, False, "sink"),
            Agent("k1", 160 + delta, False, "sink"),
            Agent("k2", 160 + delta, False, "sink"),
            Agent("k3", 160 + delta, False, "sink"),
            Agent("k4", 160 + delta, False, "sink"),
        )
        offers = (
            Offer("s0", "k0", 100 + delta, True),
            Offer("s1", "k1", 100 + delta, True),
            Offer("s0", "k2", 100 + delta, True),
            Offer("s1", "k3", 100 + delta, True),
            Offer("f0", "k0", 40 + delta, False),
        )
        return Regime(
            schema=MESSAGING_REPAIR_IMPL_SCHEMA,
            kind=FAVORABLE_REGIME,
            seed=seed,
            truth=truth,
            agents=agents,
            offers=offers,
            action_budget=self.action_budget,
        )


def build_default_bed() -> MessagingRepairBed:

    return MessagingRepairBed()
