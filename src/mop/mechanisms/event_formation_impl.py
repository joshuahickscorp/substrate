"""Deterministic seeded relational-temporal event former and the untrained control triggers.

This module supplies the working mechanics for the event formation workstream. It builds seeded
toy episodes for two regimes, a NULL regime whose events carry no relational-temporal signal and a
FAVORABLE regime whose relational-temporal structure lines an event former up with the moments that
matter. It then measures, per arm, the utility retained and the charged compute spent.

The event former acts only when a relation holds inside its bound temporal window, so on the
favorable regime it captures the opportunities while charging few full evaluations. The four
declared controls are wrong-time, wrong-event, appearance-only, and stateless-delayed-trigger; the
last two are the untrained controls a utility claim must beat on both utility and charged compute.

Nothing here claims that any event actually is useful. The episodes are toy, seeded records and the
measurements are deterministic mechanics. Claim scope: deterministic programmatic mechanics only.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256
from .event_formation_scaffold import (
    REQUIRED_CONTROLS,
    EventFormationRefusal,
    synthesize_relational_episode,
)

EVENT_FORMER_ID = "relational-temporal-former"

# The order is load-bearing for the measurement digest: the former first, then the declared controls
# in their canonical scaffold order.
ARM_IDS: tuple[str, ...] = (EVENT_FORMER_ID, *REQUIRED_CONTROLS)

# A utility below this floor is not preserved, so no useful event may be claimed.
UTILITY_FLOOR = 0.8

# Each false action costs half of one correctly handled opportunity when scoring utility.
FALSE_ACTION_PENALTY = 0.5

# The toy episode length. Both regimes share it so charged compute is comparable across arms.
EPISODE_TICKS = 24

NULL_REGIME = "null"
FAVORABLE_REGIME = "favorable"
REGIMES: tuple[str, ...] = (NULL_REGIME, FAVORABLE_REGIME)

# The ground-truth opportunity ticks: acting here yields utility, acting elsewhere is a false action.
_OPPORTUNITY_TICKS: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15)
# Surface-appearance distractor ticks: an appearance-only arm fires here and earns nothing.
_DISTRACTOR_TICKS: tuple[int, ...] = (16, 17, 18, 19, 20, 21, 22, 23)
# The stateless delayed trigger fires on this fixed periodic schedule, ignorant of the structure.
_DELAY_TICKS: tuple[int, ...] = tuple(range(0, EPISODE_TICKS, 2))
# A wrong-time arm fires one window off, so it never lands on an opportunity.
_WRONG_TIME_TICKS: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)
# On the null regime the former fires here: a relation-and-window signal decorrelated from the
# opportunities, so any trigger loses utility and the strong null holds.
_NULL_FORMER_TICKS: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)


@dataclass(frozen=True, slots=True)
class Tick:
    """One toy time step, with the ground truth and each arm's firing decision pinned in advance."""

    index: int
    opportunity: bool
    former_fire: bool
    wrong_time_fire: bool
    wrong_event_fire: bool
    appearance_fire: bool
    delay_fire: bool


def build_episode(regime: str, seed: int) -> tuple[Tick, ...]:
    """Build a deterministic, seeded episode for one regime. Identical inputs yield identical ticks."""

    if seed < 0:
        raise EventFormationRefusal("episode seed must be nonnegative")
    if regime not in REGIMES:
        raise EventFormationRefusal(f"unknown regime {regime!r}")
    opportunity = set(_OPPORTUNITY_TICKS)
    appearance = set(_OPPORTUNITY_TICKS) | set(_DISTRACTOR_TICKS)
    delay = set(_DELAY_TICKS)
    wrong_time = set(_WRONG_TIME_TICKS)
    wrong_event = set(_DISTRACTOR_TICKS)
    if regime == FAVORABLE_REGIME:
        former = set(_OPPORTUNITY_TICKS)
    else:
        former = set(_NULL_FORMER_TICKS)
    return tuple(
        Tick(
            index=index,
            opportunity=index in opportunity,
            former_fire=index in former,
            wrong_time_fire=index in wrong_time,
            wrong_event_fire=index in wrong_event,
            appearance_fire=index in appearance,
            delay_fire=index in delay,
        )
        for index in range(EPISODE_TICKS)
    )


def _arm_fires(tick: Tick, arm: str) -> bool:
    fires = {
        EVENT_FORMER_ID: tick.former_fire,
        "wrong-time": tick.wrong_time_fire,
        "wrong-event": tick.wrong_event_fire,
        "appearance-only": tick.appearance_fire,
        "stateless-delayed-trigger": tick.delay_fire,
    }
    if arm not in fires:
        raise EventFormationRefusal(f"unknown arm {arm!r}")
    return fires[arm]


def measure_arm(ticks: tuple[Tick, ...], arm: str) -> tuple[float, float]:
    """Return (utility_retained, charged_compute) for one arm over one episode.

    Utility credits a correctly handled opportunity and debits half a unit per false action, then
    clamps to the unit interval. Charged compute is the fraction of ticks the arm evaluates.
    """

    tick_count = len(ticks)
    if tick_count == 0:
        raise EventFormationRefusal("episode must contain at least one tick")
    total_opportunities = sum(1 for tick in ticks if tick.opportunity)
    if total_opportunities == 0:
        raise EventFormationRefusal("episode must contain at least one opportunity")
    acts = [tick for tick in ticks if _arm_fires(tick, arm)]
    hits = sum(1 for tick in acts if tick.opportunity)
    false_actions = len(acts) - hits
    raw = hits / total_opportunities - FALSE_ACTION_PENALTY * false_actions / total_opportunities
    utility = min(1.0, max(0.0, raw))
    charged_compute = len(acts) / tick_count
    return utility, charged_compute


def episode_sha256(seed: int) -> str:
    """A seeded relational-episode digest, so distinct seeds carry distinct episode identities."""

    return synthesize_relational_episode(seed).digest()


@dataclass(frozen=True, slots=True)
class RegimeMeasurement:
    """The utility and charged compute of every arm on one regime, plus the episode identity."""

    regime: str
    seed: int
    utility: Mapping[str, float]
    charged_compute: Mapping[str, float]
    episode_digest: str

    def __post_init__(self) -> None:
        if set(self.utility) != set(ARM_IDS):
            raise EventFormationRefusal("utility must cover exactly the arm ids")
        if set(self.charged_compute) != set(ARM_IDS):
            raise EventFormationRefusal("charged compute must cover exactly the arm ids")

    def payload(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "seed": self.seed,
            "utility": {arm: self.utility[arm] for arm in ARM_IDS},
            "charged_compute": {arm: self.charged_compute[arm] for arm in ARM_IDS},
            "episode_digest": self.episode_digest,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def measure_regime(ticks: tuple[Tick, ...], regime: str, seed: int) -> RegimeMeasurement:
    """Measure every arm on one regime and pin the result behind the seeded episode identity."""

    utility: dict[str, float] = {}
    charged_compute: dict[str, float] = {}
    for arm in ARM_IDS:
        arm_utility, arm_compute = measure_arm(ticks, arm)
        utility[arm] = arm_utility
        charged_compute[arm] = arm_compute
    return RegimeMeasurement(
        regime=regime,
        seed=seed,
        utility=utility,
        charged_compute=charged_compute,
        episode_digest=episode_sha256(seed),
    )
