"""Deterministic bed for the integrated ESCS advantage frontier (epoch integrated_escs).

This module builds the runnable task environment the integrated ESCS workstream runs against. It is
mechanics infrastructure only. It defines a task ("regime") as a vector of mechanism demands over the
four integration mechanisms multi-perspective, dynamic-composition, working-memory, and dense-mixing,
plus the two regimes the epoch needs:

- the NULL regime, where one mechanism is not demanded at all, so the matched baseline that ablates
  exactly that mechanism ties the integrated organization at matched cost and no integrated advantage
  exists (the named prior null holds);
- the FAVORABLE regime, where every mechanism is genuinely demanded, so the integrated organization
  strictly dominates every matched baseline on quality at matched cost, mechanics only.

The bed exposes the matched baselines as its controls and a non-vacuous matched full-system budget.
Nothing here asserts that any real system is capable on natural data.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import MatchedBudget
from ..substrate.events import canonical_sha256
from .integrated_escs_scaffold import (
    CLAIM_SCOPE,
    INTEGRATED_ESCS_SCHEMA,
    MECHANISM_LADDER,
    REQUIRED_BASELINES,
    IntegratedEscsRefusal,
)

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")

# Base demand for each mechanism in the favorable regime. Every entry is strictly positive so the
# integrated organization is genuinely demanded on every mechanism. Order tracks MECHANISM_LADDER.
_BASE_DEMANDS: tuple[float, ...] = (0.40, 0.50, 0.60, 0.50)

# The null regime zeroes exactly one mechanism demand. Index 3 (dense-mixing) is ablated by the
# sparse-only baseline, so under the null that baseline ties the integrated organization at matched cost.
NULL_ZERO_INDEX = len(MECHANISM_LADDER) - 1


def _seed_unit(seed: int, index: int) -> float:
    """Return a deterministic float in the closed unit interval keyed by seed and mechanism index."""

    digest = canonical_sha256({"seed": seed, "index": index})
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _favorable_demand(seed: int, index: int) -> float:
    """Return the strictly positive favorable-regime demand for one mechanism under one seed."""

    value = _BASE_DEMANDS[index] + _seed_unit(seed, index) * 0.10
    return round(min(0.90, value), 9)


@dataclass(frozen=True, slots=True)
class EscsRegime:
    """A task for the integrated ESCS frontier: a demand vector over the four mechanisms plus a seed.

    Claim scope: deterministic programmatic mechanics only; no capability claim. A regime is a
    bookkeeping description of what a task demands, never a measurement of any real system.
    """

    name: str
    demands: tuple[float, ...]
    seed: int
    schema: str = INTEGRATED_ESCS_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported regime schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("regime claim scope cannot be widened")
        if _ID_RE.fullmatch(self.name) is None:
            raise IntegratedEscsRefusal("EscsRegime.name must use stable lowercase characters")
        if len(self.demands) != len(MECHANISM_LADDER):
            raise IntegratedEscsRefusal("regime demands must cover every mechanism on the ladder")
        if self.seed < 0:
            raise IntegratedEscsRefusal("regime seed must be nonnegative")
        for demand in self.demands:
            if demand != demand or demand in (float("inf"), float("-inf")):
                raise IntegratedEscsRefusal("regime demand must be a finite number")
            if not 0.0 <= demand <= 1.0:
                raise IntegratedEscsRefusal("regime demand must lie in [0.0, 1.0]")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "name": self.name,
            "demands": list(self.demands),
            "seed": self.seed,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class IntegratedEscsBed:
    """The deterministic bed for the integrated ESCS advantage frontier.

    Claim scope: deterministic programmatic mechanics only; no capability claim. The bed reports the
    matched baseline controls, a non-vacuous matched budget, and the null and favorable regimes.
    """

    mechanism_id: str = "integrated_escs"

    def controls(self) -> tuple[str, ...]:
        """Return the matched baseline family the integrated organization must dominate."""

        return REQUIRED_BASELINES

    def matched_cost(self) -> MatchedBudget:
        """Return the non-vacuous matched full-system budget every organization is held to."""

        return MatchedBudget(params=1000, flops=2000, wall_ns=100, seeds=8)

    def _favorable_demands(self, seed: int) -> tuple[float, ...]:
        return tuple(_favorable_demand(seed, index) for index in range(len(MECHANISM_LADDER)))

    def null_regime(self, seed: int) -> EscsRegime:
        """Return the null regime: one mechanism is not demanded, so a matched baseline ties."""

        demands = list(self._favorable_demands(seed))
        demands[NULL_ZERO_INDEX] = 0.0
        return EscsRegime(name="null", demands=tuple(demands), seed=seed)

    def favorable_regime(self, seed: int) -> EscsRegime:
        """Return the favorable regime: every mechanism is genuinely demanded (strictly positive)."""

        return EscsRegime(name="favorable", demands=self._favorable_demands(seed), seed=seed)


def build_default_bed() -> IntegratedEscsBed:
    """Return the canonical integrated ESCS bed."""

    return IntegratedEscsBed()
