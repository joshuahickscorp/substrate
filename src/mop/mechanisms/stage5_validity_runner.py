
from __future__ import annotations

from typing import Any

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .stage5_validity_bed import RegimeEvidence, Stage5ValidityBed
from .stage5_validity_impl import DEFAULT_RTOL, ValidityEvaluation, evaluate_regime

STAGE5_VALIDITY_RUNNER_SCHEMA = "mop-stage5-validity-runner/v1"

REQUIREMENT_ID = "s5.validity"
STAGE_INDEX = 5


class Stage5ValidityRunnerRefusal(ValueError):
    pass


def _evidence_digest(
    bed: Stage5ValidityBed, regime: RegimeEvidence, evaluation: ValidityEvaluation, seed: int
) -> str:

    return canonical_sha256(
        {
            "schema": STAGE5_VALIDITY_RUNNER_SCHEMA,
            "mechanism_id": bed.mechanism_id,
            "requirement_id": REQUIREMENT_ID,
            "stage": STAGE_INDEX,
            "seed": seed,
            "regime": regime.payload(),
            "evaluation": evaluation.payload(),
        }
    )


def run(bed: Stage5ValidityBed, seed: int) -> RunReceipt:

    if seed < 0:
        raise Stage5ValidityRunnerRefusal("stage 5 validity runner seed must be nonnegative")
    regime = bed.candidate_regime(seed)
    evaluation = evaluate_regime(regime, rtol=DEFAULT_RTOL)
    digest = _evidence_digest(bed, regime, evaluation, seed)
    passed = evaluation.passed()
    verdict = VERDICT_MECHANICS_OK if passed else VERDICT_NULL
    detail: dict[str, Any] = {
        "schema": STAGE5_VALIDITY_RUNNER_SCHEMA,
        "regime": regime.regime,
        "seed": seed,
        "all_axes_pass": evaluation.all_axes_pass(),
        "failing_axes": list(evaluation.failing_axes()),
        "reproducing_controls": list(evaluation.reproducing_controls()),
        "mismatching_resources": list(evaluation.mismatching_resources()),
        "efficiency_matches": evaluation.efficiency_matches(),
        "evaluation_digest": evaluation.digest(),
    }
    return mint_demonstration(
        mechanism_id=bed.mechanism_id,
        stage=STAGE_INDEX,
        requirement_id=REQUIREMENT_ID,
        controls_cleared=evaluation.clean_controls(),
        evidence_digest=digest,
        verdict=verdict,
        detail=detail,
    )
