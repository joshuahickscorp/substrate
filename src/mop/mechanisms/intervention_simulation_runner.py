
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from . import intervention_simulation_impl as impl
from .intervention_simulation_bed import (
    CLAIM_SCOPE,
    MECHANISM_ID,
    REQUIREMENT_ID,
    STAGE,
    InterventionSimulationBed,
    RegimeSpec,
)

MARGIN_REQUIRED = 0.02
RUN_SCHEMA = "mop-intervention-simulation-run/v1"


class InterventionSimulationRunnerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SubComparison:

    sub_mechanism: str
    arm_id: str
    control_id: str
    arm_score: float
    control_score: float
    margin_required: float

    def __post_init__(self) -> None:
        if self.margin_required <= 0.0:
            raise InterventionSimulationRunnerError("a non-vacuous comparison needs a positive margin")

    @property
    def margin(self) -> float:
        return self.arm_score - self.control_score

    @property
    def beats(self) -> bool:
        return self.margin >= self.margin_required

    def payload(self) -> dict[str, Any]:
        return {
            "sub_mechanism": self.sub_mechanism,
            "arm_id": self.arm_id,
            "control_id": self.control_id,
            "arm_score": self.arm_score,
            "control_score": self.control_score,
            "margin_required": self.margin_required,
            "margin": self.margin,
            "beats": self.beats,
        }


@dataclass(frozen=True, slots=True)
class RegimeEvaluation:

    regime_name: str
    comparisons: tuple[SubComparison, ...]

    def __post_init__(self) -> None:
        if not self.comparisons:
            raise InterventionSimulationRunnerError("a regime evaluation needs at least one comparison")

    @property
    def all_beat(self) -> bool:
        return all(comparison.beats for comparison in self.comparisons)

    @property
    def none_beat(self) -> bool:
        return not any(comparison.beats for comparison in self.comparisons)

    def payload(self) -> dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "comparisons": [comparison.payload() for comparison in self.comparisons],
        }


@dataclass(frozen=True, slots=True)
class RunResults:

    mechanism_id: str
    seed: int
    favorable: RegimeEvaluation
    null: RegimeEvaluation
    schema: str = RUN_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "mechanism_id": self.mechanism_id,
            "seed": self.seed,
            "favorable": self.favorable.payload(),
            "null": self.null.payload(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _evaluate_regime(regime: RegimeSpec) -> RegimeEvaluation:

    comparisons = (
        SubComparison(
            "causal-intervention",
            "do-operator",
            "observational-only",
            impl.do_operator_score(regime.causal_effect, regime.test_inputs),
            impl.observational_score(regime.causal_effect, regime.confound_bias, regime.test_inputs),
            MARGIN_REQUIRED,
        ),
        SubComparison(
            "simulation-for-action",
            "rollout-planner",
            "random-action",
            impl.rollout_score(regime.reward_table),
            impl.random_action_score(regime.reward_table),
            MARGIN_REQUIRED,
        ),
        SubComparison(
            "calibrated-uncertainty",
            "calibrated-uncertainty",
            "overconfident",
            impl.calibrated_score(regime.bayes_probs),
            impl.overconfident_score(regime.bayes_probs),
            MARGIN_REQUIRED,
        ),
        SubComparison(
            "reducible-novelty-vs-random",
            "reducible-novelty-curiosity",
            "random-curiosity",
            impl.reducible_novelty_score(regime.reducible_novelty, regime.curiosity_budget),
            impl.random_curiosity_score(regime.reducible_novelty, regime.curiosity_budget),
            MARGIN_REQUIRED,
        ),
        SubComparison(
            "reducible-novelty-vs-count",
            "reducible-novelty-curiosity",
            "count-based",
            impl.reducible_novelty_score(regime.reducible_novelty, regime.curiosity_budget),
            impl.count_based_score(
                regime.reducible_novelty, regime.count_novelty, regime.curiosity_budget
            ),
            MARGIN_REQUIRED,
        ),
    )
    return RegimeEvaluation(regime.name, comparisons)


@dataclass(frozen=True, slots=True)
class InterventionSimulationRunner:

    mechanism_id: str = MECHANISM_ID

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise InterventionSimulationRunnerError("runner mechanism id cannot drift")

    def run(self, bed: InterventionSimulationBed, seed: int) -> RunResults:

        favorable = _evaluate_regime(bed.favorable_regime(seed))
        null = _evaluate_regime(bed.null_regime(seed))
        return RunResults(mechanism_id=self.mechanism_id, seed=seed, favorable=favorable, null=null)

    def mint(self, results: RunResults) -> RunReceipt:

        favorable_all_beat = results.favorable.all_beat
        null_holds = results.null.none_beat
        verdict = VERDICT_MECHANICS_OK if (favorable_all_beat and null_holds) else VERDICT_NULL
        controls_cleared = tuple(
            sorted(
                {
                    comparison.control_id
                    for comparison in results.favorable.comparisons
                    if comparison.beats
                }
            )
        )
        detail: dict[str, Any] = {
            "requirement_id": REQUIREMENT_ID,
            "favorable_all_beat": favorable_all_beat,
            "null_holds": null_holds,
            "controls_cleared": list(controls_cleared),
            "favorable_margins": [
                {
                    "sub_mechanism": comparison.sub_mechanism,
                    "control": comparison.control_id,
                    "margin": comparison.margin,
                    "beats": comparison.beats,
                }
                for comparison in results.favorable.comparisons
            ],
            "null_margins": [
                {
                    "sub_mechanism": comparison.sub_mechanism,
                    "control": comparison.control_id,
                    "margin": comparison.margin,
                    "beats": comparison.beats,
                }
                for comparison in results.null.comparisons
            ],
            "note": "mechanics demonstration only; no capability or natural-data claim",
        }
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=results.digest(),
            verdict=verdict,
            detail=detail,
        )
