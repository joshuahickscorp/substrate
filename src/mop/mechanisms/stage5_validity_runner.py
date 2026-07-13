"""Fail-closed runner for the Stage 5 session-disjoint general-validity harness.

This module raises the SCAFFOLDING axis only. It runs the deterministic validity evaluator against a
seeded bed and mints a mechanics-only demonstration receipt. It fails closed: the verdict is "null"
whenever any disjointness axis fails, any leak control reproduces the result, or a declared resource
cost disagrees with its measured cost beyond tolerance. It mints the "mechanics-ok" verdict only when
every axis passes, every leak control stays clean, and every measured resource is backed within
tolerance, on the favorable candidate regime.

The receipt is always a mechanics-demonstration, so it can never be a scientific confirmation and can
never open a stage gate. The honest current state is that Stage 5 is not entered; this runner
exercises the machinery and mints a null-by-default demonstration under that reality.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim.

House style: no em or en dashes. Use commas, semicolons, or "vs".
"""

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

# The Stage 5 entry requirement this runner reports against, and the ladder stage it sits at.
REQUIREMENT_ID = "s5.validity"
STAGE_INDEX = 5


class Stage5ValidityRunnerRefusal(ValueError):
    """Raised when the runner is handed a malformed seed."""


def _evidence_digest(
    bed: Stage5ValidityBed, regime: RegimeEvidence, evaluation: ValidityEvaluation, seed: int
) -> str:
    """Canonical digest over the seeded regime and its recomputed evaluation."""

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
    """Evaluate the candidate regime and mint a fail-closed mechanics-only demonstration receipt."""

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
