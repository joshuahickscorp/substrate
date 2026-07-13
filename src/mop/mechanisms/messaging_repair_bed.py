"""The seeded task bed for the messaging_repair Stage 3 workstream (epoch messaging_repair).

The bed supplies two deterministic regimes for one mechanism (mechanism_id "messaging_repair"):

- The NULL regime carries no genuine causal signal. Every parent estimate is as noisy as the child
  it points at and every offer is marked unreliable, so bounded messaging, selective verification,
  and repair all buy nothing: the mechanism ties the no-message floor. This encodes the named prior
  null that limited-broadcast and disagreement-only findings hold and that unbounded messaging and
  always-on verification buy nothing.
- The FAVORABLE regime carries real causal dependencies: reliable sources hold the target value, a
  flaky source would corrupt a sink, and one sink is reachable only by repair. Bounded causal
  messaging plus selective verification (to drop the flaky offer without paying to verify everything)
  plus disagreement-triggered repair fixes every agent within the matched budget, which no control
  can match. The advantage is mechanics-only: toy integers on a seeded bed, not a capability claim.

The controls the bed declares are exactly the metric's matched controls: no-message, broadcast-all,
stale-message, no-verify, always-verify, and majority-vote. The matched cost is non-vacuous.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

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
    """A deterministic seed offset applied uniformly to the target and every estimate.

    Shifting the target and all estimates by the same amount leaves every error magnitude and thus
    every policy ordering invariant, so the bed is genuinely seeded without changing the mechanics.
    """

    if seed < 0:
        raise MessagingRepairImplError("bed seed must be nonnegative")
    return seed % 5


@dataclass(frozen=True, slots=True)
class MessagingRepairBed:
    """The messaging_repair bed: null and favorable regimes, declared controls, matched cost.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    mechanism_id: str = MECHANISM_ID
    action_budget: int = DEFAULT_ACTION_BUDGET

    def controls(self) -> tuple[str, ...]:
        return DECLARED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched budget for the receipt boundary. All fields are strictly positive."""

        return MatchedBudget(
            params=self.action_budget,
            flops=self.action_budget * self.action_budget,
            wall_ns=1,
            seeds=8,
        )

    def null_regime(self, seed: int) -> Regime:
        """No genuine causal signal: symmetric noise, no reliable parent, every offer unreliable."""

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
        """Real causal dependencies: reliable sources, one flaky source, one repair-only sink."""

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
    """Return the canonical messaging_repair bed at the default matched action budget."""

    return MessagingRepairBed()
