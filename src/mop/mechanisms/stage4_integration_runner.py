"""Fail-closed runner for the Stage 4 integrated architecture advantage harness.

This module raises the SCAFFOLDING axis only. It ties the deterministic bed and integration function
together behind the load-bearing composition precondition of Stage 4: an integrated advantage may not
even be measured until at least two component mechanisms have been INDEPENDENTLY CONFIRMED at Stage 3.
The runner counts scientific-confirmation receipts via their ``is_confirmation`` property; a toy
mechanics demonstration never counts. It runs no model, loads no weights, and touches no network.

Honest current state: zero Stage 3 confirmations exist, so the default call reports "not entered" and
mints a mechanics-demonstration receipt with verdict null. Supplying two or more genuine confirmations
exercises the composition machinery: on the favorable regime the joint point strictly dominates every
baseline and the receipt records verdict mechanics-ok; on the null regime the joint dominates nothing
and the receipt records verdict null. The output is ALWAYS a mechanics demonstration, never a
scientific confirmation, because a Stage 4 confirmation itself needs an independent verification pass.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    RunReceipt,
    mint_demonstration,
)
from ..ladder.stage_ladder import STAGE4_FORCING_NULL, STAGE_MIN_MECHANISM_RECEIPTS
from ..substrate.events import canonical_sha256
from .stage4_integration_bed import REGIME_FAVORABLE, REGIME_NULL, Stage4IntegrationBed
from .stage4_integration_impl import Composition, integrate

STAGE4_RUNNER_SCHEMA = "mop-stage4-integration-runner/v1"

# The mechanism id, stage, and requirement id the Stage 4 output receipt is minted under.
MECHANISM_ID = "stage4_integration"
STAGE = 4
REQUIREMENT_ID = "s4.integration"

# Stage 4 entry demands at least this many independently confirmed mechanisms (read from the ladder).
MIN_CONFIRMED_MECHANISMS = STAGE_MIN_MECHANISM_RECEIPTS[STAGE]

# Outcome tokens recorded in the receipt detail.
OUTCOME_NOT_ENTERED = "not-entered"
OUTCOME_ENTERED_MECHANICS_OK = "entered-mechanics-ok"
OUTCOME_ENTERED_NULL = "entered-null"

_ALLOWED_REGIMES = (REGIME_FAVORABLE, REGIME_NULL)


class Stage4RunnerRefusal(ValueError):
    """Raised when the runner is called with a malformed seed or an unsupported regime."""


def _count_confirmations(confirmations: Sequence[RunReceipt]) -> int:
    """Count scientific-confirmation receipts. A mechanics demonstration never counts."""

    return sum(1 for receipt in confirmations if receipt.is_confirmation)


def _mint(
    *,
    verdict: str,
    outcome: str,
    note: str,
    regime: str,
    confirmations: int,
    composition: Composition | None,
) -> RunReceipt:
    """Mint the Stage 4 mechanics-demonstration receipt with a canonical evidence digest."""

    detail: dict[str, Any] = {
        "schema": STAGE4_RUNNER_SCHEMA,
        "outcome": outcome,
        "note": note,
        "regime": regime,
        "confirmations": confirmations,
        "min_confirmed_mechanisms": MIN_CONFIRMED_MECHANISMS,
        "forcing_null": STAGE4_FORCING_NULL,
        "dominates_all": None if composition is None else composition.dominates_all(),
        "dominated_baselines": [] if composition is None else list(composition.dominated_baselines()),
        "composition_digest": None if composition is None else composition.digest(),
    }
    evidence_digest = canonical_sha256(
        {
            "schema": STAGE4_RUNNER_SCHEMA,
            "verdict": verdict,
            "outcome": outcome,
            "regime": regime,
            "confirmations": confirmations,
            "min_confirmed_mechanisms": MIN_CONFIRMED_MECHANISMS,
            "composition": None if composition is None else composition.payload(),
        }
    )
    return mint_demonstration(
        mechanism_id=MECHANISM_ID,
        stage=STAGE,
        requirement_id=REQUIREMENT_ID,
        controls_cleared=Stage4IntegrationBed().controls(),
        evidence_digest=evidence_digest,
        verdict=verdict,
        detail=detail,
    )


def run(
    confirmations: Sequence[RunReceipt],
    bed: Stage4IntegrationBed,
    seed: int,
    *,
    regime: str = REGIME_FAVORABLE,
) -> RunReceipt:
    """Run the Stage 4 integration harness and mint a mechanics-demonstration receipt.

    Fails closed: with fewer than the required confirmations it reports "not entered" and mints a null
    receipt without exercising the composition. With enough confirmations it composes the chosen regime
    and records verdict mechanics-ok on joint Pareto dominance, else verdict null. The output is never a
    scientific confirmation; a Stage 4 confirmation itself needs an independent verification pass.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Stage4RunnerRefusal("seed must be a nonnegative integer")
    if regime not in _ALLOWED_REGIMES:
        raise Stage4RunnerRefusal(f"unsupported regime {regime!r}")

    confirmed = _count_confirmations(confirmations)
    if confirmed < MIN_CONFIRMED_MECHANISMS:
        return _mint(
            verdict=VERDICT_NULL,
            outcome=OUTCOME_NOT_ENTERED,
            note=f"not entered: {confirmed} confirmations < {MIN_CONFIRMED_MECHANISMS}",
            regime=regime,
            confirmations=confirmed,
            composition=None,
        )

    sample = bed.favorable_regime(seed) if regime == REGIME_FAVORABLE else bed.null_regime(seed)
    composition = integrate(sample, seed)
    if composition.dominates_all():
        return _mint(
            verdict=VERDICT_MECHANICS_OK,
            outcome=OUTCOME_ENTERED_MECHANICS_OK,
            note="entered: joint point strictly dominates every matched baseline at matched compute",
            regime=regime,
            confirmations=confirmed,
            composition=composition,
        )
    return _mint(
        verdict=VERDICT_NULL,
        outcome=OUTCOME_ENTERED_NULL,
        note="entered: composition adds cost without quality; joint dominates no matched baseline",
        regime=regime,
        confirmations=confirmed,
        composition=composition,
    )
