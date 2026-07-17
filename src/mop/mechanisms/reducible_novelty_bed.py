"""Deterministic source-panel bed for the reducible novelty mechanism (lane reducible_novelty).

This module is runnable machinery, not a scaffold and not evidence. It builds the two source-panel
regimes a reducible-novelty curiosity mechanism must be measured on:

- NULL regime: every source is irreducible seeded noise by construction. Probing a source never
  reduces its error, so total learning progress is zero for EVERY allocation policy, and no policy
  can strictly beat uniform_allocation on that axis. The irreducible noise trap holds by
  construction: a joint win is impossible on this panel.
- FAVORABLE regime: a mixed panel. Even-indexed sources carry learnable seeded structure (a positive
  reducible signal that decays under probing); odd-indexed sources are pure noise whose raw novelty
  is deliberately LOUDER than any reducible source. Targeting predicted learning progress can, in
  principle, win both axes; chasing raw novelty parks the budget on the loud noise.

The regimes are mechanics only: seeded, byte-reproducible float panels with no capability claim and
no natural data. The bed owns the source geometry; the allocators live in the impl module and the
measurement and verdict live in the runner module.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .reducible_novelty_scaffold import REQUIRED_CONTROLS

MECHANISM_ID = "reducible_novelty"
BED_SCHEMA = "mop-reducible-novelty-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# Geometry of the source panel. Even indices are the reducible slots in the favorable regime; odd
# indices are always irreducible noise. In the null regime every slot is irreducible.
SOURCE_COUNT = 8
REDUCIBLE_INDICES: tuple[int, ...] = (0, 2, 4, 6)

# The fixed probe budget every allocation policy must spend exactly, and the per-source pilot spend
# the mechanism uses to estimate learning progress before committing the remainder.
PROBE_BUDGET = 40.0
PILOT_PROBES_PER_SOURCE = 1.0

# Seeded parameter bands. They are chosen so the strict joint win in the favorable regime holds BY
# CONSTRUCTION for every nonnegative seed (the impl module carries the margin arithmetic):
# - reducible signal in [1.0, 1.2) and decay in [0.55, 0.65), so a pilot probe always observes a
#   strictly positive progress gain of at least 0.35;
# - reducible noise floor in [0.10, 0.15), so reducible raw novelty stays at or below 1.35;
# - irreducible noise floor in [2.0, 2.5), so irreducible raw novelty stays at or above 2.0 and the
#   novelty chaser is always pulled toward the unlearnable sources.
REDUCIBLE_SIGNAL_BASE = 1.0
REDUCIBLE_SIGNAL_SPAN = 0.2
REDUCIBLE_DECAY_BASE = 0.55
REDUCIBLE_DECAY_SPAN = 0.1
REDUCIBLE_FLOOR_BASE = 0.1
REDUCIBLE_FLOOR_SPAN = 0.05
IRREDUCIBLE_FLOOR_BASE = 2.0
IRREDUCIBLE_FLOOR_SPAN = 0.5
IRREDUCIBLE_DECAY = 0.6

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)

Values = tuple[float, ...]


class BedRefusal(ValueError):
    """Raised when a bed regime request is malformed or a declaration is widened."""


def _unit(seed: int, label: str) -> float:
    """A deterministic value in [0, 1) from a seeded digest; no wall clock, no rng."""

    if seed < 0:
        raise BedRefusal("bed seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


@dataclass(frozen=True, slots=True)
class SourcePanel:
    """One regime rendered as concrete seeded source parameters.

    ``signals`` are the reducible components (zero means the source is pure irreducible noise);
    ``decays`` are the per-probe residual factors of the reducible component; ``noise_floors`` are
    the irreducible error levels. Raw novelty of a source is noise_floor + signal. Claim scope:
    deterministic programmatic mechanics only.
    """

    regime: str
    seed: int
    signals: Values
    decays: Values
    noise_floors: Values
    source_count: int = SOURCE_COUNT
    probe_budget: float = PROBE_BUDGET
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported source panel schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("source panel claim scope cannot be widened")
        if self.regime not in REGIMES:
            raise BedRefusal(f"unknown regime {self.regime!r}")
        if self.source_count < 2:
            raise BedRefusal("a source panel needs at least two sources")
        if not self.probe_budget > 0.0:
            raise BedRefusal("a source panel needs a positive probe budget")
        for name, values in (
            ("signals", self.signals),
            ("decays", self.decays),
            ("noise_floors", self.noise_floors),
        ):
            if len(values) != self.source_count:
                raise BedRefusal(f"panel {name} must match the declared source count")
        for value in self.signals:
            if value < 0.0:
                raise BedRefusal("panel signals must be nonnegative")
        for value in self.decays:
            if not 0.0 < value < 1.0:
                raise BedRefusal("panel decays must lie strictly inside (0, 1)")
        for value in self.noise_floors:
            if value <= 0.0:
                raise BedRefusal("panel noise floors must be positive")

    @property
    def novelties(self) -> Values:
        """Raw novelty per source: the loud irreducible floor plus any reducible signal."""

        return tuple(
            self.noise_floors[index] + self.signals[index] for index in range(self.source_count)
        )

    @property
    def reducible_sources(self) -> tuple[int, ...]:
        """The indices whose error a probe can actually reduce (strictly positive signal)."""

        return tuple(index for index in range(self.source_count) if self.signals[index] > 0.0)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "seed": self.seed,
            "source_count": self.source_count,
            "probe_budget": self.probe_budget,
            "signals": list(self.signals),
            "decays": list(self.decays),
            "noise_floors": list(self.noise_floors),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyBed:
    """The concrete reducible novelty bed. mechanism_id matches the scaffold and the runner.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

    mechanism_id: str = MECHANISM_ID
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise BedRefusal("bed mechanism_id drift")
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported bed schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("bed claim scope cannot be widened")

    def controls(self) -> tuple[str, ...]:
        """The declared control family: uniform, random, novelty chaser, static prior."""

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every arm is held to. Positive on every axis."""

        return MatchedBudget(
            params=SOURCE_COUNT * SOURCE_COUNT, flops=1_048_576, wall_ns=1_000_000, seeds=8
        )

    def null_regime(self, seed: int) -> SourcePanel:
        """A panel of pure irreducible noise: no probe reduces any error; the trap holds."""

        signals = tuple(0.0 for _ in range(SOURCE_COUNT))
        decays = tuple(IRREDUCIBLE_DECAY for _ in range(SOURCE_COUNT))
        noise_floors = tuple(
            IRREDUCIBLE_FLOOR_BASE + IRREDUCIBLE_FLOOR_SPAN * _unit(seed, f"null.floor.{index}")
            for index in range(SOURCE_COUNT)
        )
        return SourcePanel(
            regime=REGIME_NULL, seed=seed, signals=signals, decays=decays, noise_floors=noise_floors
        )

    def favorable_regime(self, seed: int) -> SourcePanel:
        """A mixed panel: quiet learnable sources next to louder pure-noise sources."""

        signals: list[float] = []
        decays: list[float] = []
        noise_floors: list[float] = []
        for index in range(SOURCE_COUNT):
            if index in REDUCIBLE_INDICES:
                signals.append(
                    REDUCIBLE_SIGNAL_BASE
                    + REDUCIBLE_SIGNAL_SPAN * _unit(seed, f"fav.signal.{index}")
                )
                decays.append(
                    REDUCIBLE_DECAY_BASE + REDUCIBLE_DECAY_SPAN * _unit(seed, f"fav.decay.{index}")
                )
                noise_floors.append(
                    REDUCIBLE_FLOOR_BASE + REDUCIBLE_FLOOR_SPAN * _unit(seed, f"fav.floor.{index}")
                )
            else:
                signals.append(0.0)
                decays.append(IRREDUCIBLE_DECAY)
                noise_floors.append(
                    IRREDUCIBLE_FLOOR_BASE
                    + IRREDUCIBLE_FLOOR_SPAN * _unit(seed, f"fav.noise.{index}")
                )
        return SourcePanel(
            regime=REGIME_FAVORABLE,
            seed=seed,
            signals=tuple(signals),
            decays=tuple(decays),
            noise_floors=tuple(noise_floors),
        )

    def regime(self, name: str, seed: int) -> SourcePanel:
        """Dispatch to a regime by name. Fails closed on an unknown regime."""

        if name == REGIME_NULL:
            return self.null_regime(seed)
        if name == REGIME_FAVORABLE:
            return self.favorable_regime(seed)
        raise BedRefusal(f"unknown regime {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "source_count": SOURCE_COUNT,
            "reducible_indices": list(REDUCIBLE_INDICES),
            "probe_budget": PROBE_BUDGET,
            "pilot_probes_per_source": PILOT_PROBES_PER_SOURCE,
            "controls": list(self.controls()),
            "matched_cost": self.matched_cost().payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
