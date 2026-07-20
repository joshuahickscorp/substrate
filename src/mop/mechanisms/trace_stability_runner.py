
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    FIRST_ACTIVATION_STAGE,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .trace_stability_impl import (
    MECHANISM_ID,
    STABILITY_METRIC,
    THRESHOLD,
    control_rankings,
    derived_seeds,
    rank_agreement,
    recovered_ranking,
)
from .trace_stability_scaffold import MIN_SEEDS, TRACE_STABILITY_SCHEMA

REQUIREMENT_ID = "s3.trace_stability"
NAMED_PRIOR_NULL = (
    "few-seed trace studies over-claim; a trace is not real until it survives seed variation "
    "across at least the declared minimum seeds with every declared control dead"
)


@dataclass(frozen=True, slots=True)
class RunResults:

    mechanism_id: str
    base_seed: int
    threshold: float
    controls: tuple[str, ...]
    favorable_agreement: float
    null_agreement: float
    control_agreements: dict[str, float]


@dataclass(frozen=True, slots=True)
class TraceStabilityRunner:

    mechanism_id: str = MECHANISM_ID
    threshold: float = THRESHOLD
    leaked_controls: frozenset[str] = frozenset()

    def run(self, bed: Bed, seed: int) -> RunResults:

        seeds = derived_seeds(seed, MIN_SEEDS)
        favorable: list[Any] = [bed.favorable_regime(s) for s in seeds]
        null: list[Any] = [bed.null_regime(s) for s in seeds]
        favorable_agreement = rank_agreement([recovered_ranking(m) for m in favorable])
        null_agreement = rank_agreement([recovered_ranking(m) for m in null])
        control_agreements: dict[str, float] = {}
        for control in bed.controls():
            rankings = control_rankings(favorable, control, leak=control in self.leaked_controls)
            control_agreements[control] = rank_agreement(rankings)
        return RunResults(
            mechanism_id=self.mechanism_id,
            base_seed=seed,
            threshold=self.threshold,
            controls=tuple(bed.controls()),
            favorable_agreement=favorable_agreement,
            null_agreement=null_agreement,
            control_agreements=control_agreements,
        )

    def mint(self, results: RunResults) -> RunReceipt:

        controls_beaten = tuple(
            control
            for control in results.controls
            if results.control_agreements[control] < results.threshold
        )
        controls_all_dead = len(controls_beaten) == len(results.controls)
        favorable_clears = results.favorable_agreement >= results.threshold
        null_holds = results.null_agreement < results.threshold
        mechanics_ok = favorable_clears and controls_all_dead and null_holds
        verdict = VERDICT_MECHANICS_OK if mechanics_ok else VERDICT_NULL

        evidence = {
            "schema": TRACE_STABILITY_SCHEMA,
            "mechanism_id": results.mechanism_id,
            "base_seed": results.base_seed,
            "metric": STABILITY_METRIC,
            "threshold": results.threshold,
            "favorable_agreement": results.favorable_agreement,
            "null_agreement": results.null_agreement,
            "controls": list(results.controls),
            "control_agreements": {
                control: results.control_agreements[control] for control in results.controls
            },
            "verdict": verdict,
        }
        evidence_digest = canonical_sha256(evidence)

        detail = {
            "metric": STABILITY_METRIC,
            "threshold": results.threshold,
            "named_prior_null": NAMED_PRIOR_NULL,
            "favorable_agreement": results.favorable_agreement,
            "favorable_margin": round(results.favorable_agreement - results.threshold, 9),
            "null_agreement": results.null_agreement,
            "null_margin": round(results.threshold - results.null_agreement, 9),
            "control_margins": {
                control: round(results.threshold - results.control_agreements[control], 9)
                for control in results.controls
            },
            "controls_beaten": list(controls_beaten),
            "favorable_clears": favorable_clears,
            "controls_all_dead": controls_all_dead,
            "null_holds": null_holds,
        }

        return mint_demonstration(
            mechanism_id=results.mechanism_id,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_beaten,
            evidence_digest=evidence_digest,
            verdict=verdict,
            detail=detail,
        )
