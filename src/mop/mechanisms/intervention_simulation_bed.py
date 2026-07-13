"""Deterministic toy bed for the intervention-simulation mechanism workstream.

The bed is the task environment for the EPOCH intervention_simulation. It exposes two regimes:

- the NULL regime has no causal structure and only irreducible noise, so intervention, simulation,
  and curiosity buy nothing and the named prior nulls hold: an observational-only control matches
  the do-operator, a random-action control matches a rollout, an overconfident control matches a
  calibrated estimator on fully determined outcomes, and curiosity toward irreducible noise gains
  nothing over random-curiosity or count-based exploration;
- the FAVORABLE regime carries genuine causal structure and genuine reducible novelty (mechanics
  only), so each arm has a real edge to exploit over its matched control.

The bed declares its five matched controls and a non-vacuous matched budget. It performs no
experiment and makes no capability claim: the regimes are seeded arithmetic constructions, not
measurements on natural data.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import MatchedBudget
from ..substrate.events import canonical_sha256

MECHANISM_ID = "intervention_simulation"
REQUIREMENT_ID = "s3.intervention_simulation"
STAGE = 3
SCHEMA = "mop-intervention-simulation-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# The five declared control policies, in a stable order. Membership is load-bearing for controls().
OBSERVATIONAL_ONLY = "observational-only"
RANDOM_ACTION = "random-action"
OVERCONFIDENT = "overconfident"
COUNT_BASED = "count-based"
RANDOM_CURIOSITY = "random-curiosity"
DECLARED_CONTROLS: tuple[str, ...] = (
    OBSERVATIONAL_ONLY,
    RANDOM_ACTION,
    OVERCONFIDENT,
    COUNT_BASED,
    RANDOM_CURIOSITY,
)

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIME_NAMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)

# Structural constants. The favorable values give each arm a comfortable, seed-robust margin, and
# the null values force an exact tie between every arm and its matched control.
NUM_ACTIONS = 4
HORIZON = 6
NULL_REWARD = 0.5
CAUSAL_EFFECT = 1.0
FAVORABLE_CONFOUND_BIAS = 0.6
NULL_CONFOUND_BIAS = 0.0
CURIOSITY_BUDGET = 3
FAVORABLE_BAYES_PROBS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
FAVORABLE_REDUCIBLE_NOVELTY: tuple[float, ...] = (0.9, 0.7, 0.5, 0.3, 0.1, 0.0)
COUNT_NOVELTY: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 1.0)


class InterventionSimulationBedError(ValueError):
    """Raised when the bed is built or queried with a malformed argument. The bed fails closed."""


def _require_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise InterventionSimulationBedError("seed must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    """A fully specified, seeded regime: the inputs each sub-mechanism and its control consume."""

    name: str
    seed: int
    causal_effect: float
    confound_bias: float
    test_inputs: tuple[float, ...]
    reward_table: tuple[tuple[float, ...], ...]
    bayes_probs: tuple[float, ...]
    reducible_novelty: tuple[float, ...]
    count_novelty: tuple[float, ...]
    curiosity_budget: int
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise InterventionSimulationBedError(f"unsupported regime schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise InterventionSimulationBedError("regime claim scope cannot be widened")
        if self.name not in REGIME_NAMES:
            raise InterventionSimulationBedError(f"unsupported regime name {self.name!r}")
        _require_seed(self.seed)
        if not self.test_inputs:
            raise InterventionSimulationBedError("a regime needs at least one test intervention")
        if not self.reward_table:
            raise InterventionSimulationBedError("a regime needs a non-empty reward table")
        for row in self.reward_table:
            if not row:
                raise InterventionSimulationBedError("each reward table row needs at least one action")
        if not self.bayes_probs:
            raise InterventionSimulationBedError("a regime needs at least one calibration item")
        for probability in self.bayes_probs:
            if not 0.0 <= probability <= 1.0:
                raise InterventionSimulationBedError("bayes probabilities must lie in [0, 1]")
        if not self.reducible_novelty:
            raise InterventionSimulationBedError("a regime needs at least one novelty state")
        if len(self.reducible_novelty) != len(self.count_novelty):
            raise InterventionSimulationBedError("novelty sequences must match in length")
        for value in self.reducible_novelty:
            if value < 0.0:
                raise InterventionSimulationBedError("reducible novelty cannot be negative")
        if self.curiosity_budget < 1:
            raise InterventionSimulationBedError("the curiosity budget must be at least one state")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "name": self.name,
            "seed": self.seed,
            "causal_effect": self.causal_effect,
            "confound_bias": self.confound_bias,
            "test_inputs": list(self.test_inputs),
            "reward_table": [list(row) for row in self.reward_table],
            "bayes_probs": list(self.bayes_probs),
            "reducible_novelty": list(self.reducible_novelty),
            "count_novelty": list(self.count_novelty),
            "curiosity_budget": self.curiosity_budget,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class InterventionSimulationBed:
    """The toy bed. Structured by default; set structured to False for a world with no structure."""

    mechanism_id: str = MECHANISM_ID
    structured: bool = True
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise InterventionSimulationBedError("bed mechanism id cannot drift")
        if self.schema != SCHEMA:
            raise InterventionSimulationBedError(f"unsupported bed schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise InterventionSimulationBedError("bed claim scope cannot be widened")

    def controls(self) -> tuple[str, ...]:
        """The five declared, matched control policies in stable order."""

        return DECLARED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every arm and control share before comparison."""

        return MatchedBudget(params=4096, flops=1_048_576, wall_ns=250_000, seeds=16)

    def null_regime(self, seed: int) -> RegimeSpec:
        """The null world: no causal structure, only irreducible noise. Every arm ties its control."""

        _require_seed(seed)
        return self._build(REGIME_NULL, seed, structured=False)

    def favorable_regime(self, seed: int) -> RegimeSpec:
        """The favorable world: genuine causal structure and reducible novelty, mechanics only."""

        _require_seed(seed)
        return self._build(REGIME_FAVORABLE, seed, structured=self.structured)

    def _build(self, name: str, seed: int, *, structured: bool) -> RegimeSpec:
        test_inputs = tuple(float(((seed + k) % NUM_ACTIONS) + 1) for k in range(5))
        if structured:
            confound_bias = FAVORABLE_CONFOUND_BIAS
            best_actions = tuple((seed + step) % NUM_ACTIONS for step in range(HORIZON))
            reward_table = tuple(
                tuple(1.0 if action == best_actions[step] else 0.0 for action in range(NUM_ACTIONS))
                for step in range(HORIZON)
            )
            bayes_probs = FAVORABLE_BAYES_PROBS
            reducible_novelty = FAVORABLE_REDUCIBLE_NOVELTY
        else:
            confound_bias = NULL_CONFOUND_BIAS
            reward_table = tuple(
                tuple(NULL_REWARD for _ in range(NUM_ACTIONS)) for _ in range(HORIZON)
            )
            bayes_probs = tuple(
                1.0 if (seed + index) % 2 == 0 else 0.0 for index in range(len(FAVORABLE_BAYES_PROBS))
            )
            reducible_novelty = tuple(0.0 for _ in FAVORABLE_REDUCIBLE_NOVELTY)
        return RegimeSpec(
            name=name,
            seed=seed,
            causal_effect=CAUSAL_EFFECT,
            confound_bias=confound_bias,
            test_inputs=test_inputs,
            reward_table=reward_table,
            bayes_probs=bayes_probs,
            reducible_novelty=reducible_novelty,
            count_novelty=COUNT_NOVELTY,
            curiosity_budget=CURIOSITY_BUDGET,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "mechanism_id": self.mechanism_id,
            "structured": self.structured,
            "controls": list(self.controls()),
            "matched_cost": self.matched_cost().payload(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
