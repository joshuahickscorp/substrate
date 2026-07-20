
from __future__ import annotations

import re

import pytest

from mop.ladder.ladder_contracts import (
    KIND_CONFIRMATION,
    KIND_DEMONSTRATION,
    RunReceipt,
    mint_confirmation,
    mint_demonstration,
)
from mop.ladder.stage_ladder import MatchedBudget
from mop.mechanisms.stage4_integration_bed import (
    BASELINE_ARMS,
    REGIME_FAVORABLE,
    REGIME_NULL,
    Stage4IntegrationBed,
    build_default_bed,
)
from mop.mechanisms.stage4_integration_runner import (
    MECHANISM_ID,
    MIN_CONFIRMED_MECHANISMS,
    OUTCOME_ENTERED_MECHANICS_OK,
    OUTCOME_ENTERED_NULL,
    OUTCOME_NOT_ENTERED,
    REQUIREMENT_ID,
    STAGE,
    Stage4RunnerRefusal,
    run,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST = "b" * 64


def a_confirmation(mechanism_id: str) -> RunReceipt:

    return mint_confirmation(
        mechanism_id=mechanism_id,
        stage=3,
        requirement_id="stage3.confirmed_useful_mechanism",
        controls_cleared=("untrained-control", "matched-sparse", "fixed-init"),
        overturns_null="gen0 first-mechanism nulls",
        matched=MatchedBudget(params=1, flops=1, wall_ns=1, seeds=8),
        evidence_digest=DIGEST,
    )


def two_confirmations() -> list[RunReceipt]:
    return [a_confirmation("event_formation"), a_confirmation("niche_dispatch")]


def test_zero_confirmations_is_null_and_not_entered() -> None:
    receipt = run([], build_default_bed(), 7)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == "null"
    assert receipt.stage == STAGE
    assert receipt.requirement_id == REQUIREMENT_ID
    assert receipt.mechanism_id == MECHANISM_ID
    assert receipt.detail["outcome"] == OUTCOME_NOT_ENTERED
    assert "not entered" in receipt.detail["note"]
    assert receipt.detail["confirmations"] == 0
    assert receipt.is_confirmation is False


def test_one_confirmation_still_not_entered() -> None:
    receipt = run([a_confirmation("event_formation")], build_default_bed(), 3)
    assert receipt.verdict == "null"
    assert receipt.detail["outcome"] == OUTCOME_NOT_ENTERED
    assert receipt.detail["confirmations"] == 1
    assert MIN_CONFIRMED_MECHANISMS == 2


def test_demonstrations_do_not_count_as_confirmations() -> None:
    demos = [
        mint_demonstration(
            mechanism_id="event_formation",
            stage=3,
            requirement_id="stage3.confirmed_useful_mechanism",
            controls_cleared=("untrained-control",),
            evidence_digest=DIGEST,
        )
        for _ in range(4)
    ]
    receipt = run(demos, build_default_bed(), 1)
    assert receipt.detail["outcome"] == OUTCOME_NOT_ENTERED
    assert receipt.detail["confirmations"] == 0


def test_two_confirmations_favorable_regime_is_mechanics_ok() -> None:
    receipt = run(two_confirmations(), build_default_bed(), 11, regime=REGIME_FAVORABLE)
    assert receipt.verdict == "mechanics-ok"
    assert receipt.detail["outcome"] == OUTCOME_ENTERED_MECHANICS_OK
    assert receipt.detail["dominates_all"] is True
    assert list(receipt.detail["dominated_baselines"]) == list(BASELINE_ARMS)
    assert receipt.detail["confirmations"] == 2
    assert receipt.is_confirmation is False


def test_two_confirmations_null_regime_is_null() -> None:
    receipt = run(two_confirmations(), build_default_bed(), 11, regime=REGIME_NULL)
    assert receipt.verdict == "null"
    assert receipt.detail["outcome"] == OUTCOME_ENTERED_NULL
    assert receipt.detail["dominates_all"] is False
    assert receipt.detail["dominated_baselines"] == []
    assert receipt.is_confirmation is False


def test_favorable_default_regime_reaches_mechanics_ok() -> None:
    receipt = run(two_confirmations(), build_default_bed(), 5)
    assert receipt.verdict == "mechanics-ok"
    assert receipt.detail["regime"] == REGIME_FAVORABLE


def test_stage4_output_is_never_a_confirmation() -> None:
    bed = build_default_bed()
    receipts = [
        run([], bed, 0),
        run(two_confirmations(), bed, 0, regime=REGIME_FAVORABLE),
        run(two_confirmations(), bed, 0, regime=REGIME_NULL),
    ]
    for receipt in receipts:
        assert receipt.kind != KIND_CONFIRMATION
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.is_confirmation is False


def test_runner_is_deterministic_per_seed() -> None:
    bed = build_default_bed()
    first = run(two_confirmations(), bed, 42, regime=REGIME_FAVORABLE)
    second = run(two_confirmations(), bed, 42, regime=REGIME_FAVORABLE)
    assert first.digest() == second.digest()
    assert first.evidence_digest == second.evidence_digest


def test_evidence_digest_is_a_stable_sha256() -> None:
    receipt = run(two_confirmations(), build_default_bed(), 9)
    assert _SHA256_RE.match(receipt.evidence_digest)
    assert len(receipt.digest()) == 64


def test_distinct_seeds_yield_distinct_evidence() -> None:
    bed = build_default_bed()
    one = run(two_confirmations(), bed, 1, regime=REGIME_FAVORABLE)
    two = run(two_confirmations(), bed, 2, regime=REGIME_FAVORABLE)
    assert one.evidence_digest != two.evidence_digest


def test_negative_seed_fails_closed() -> None:
    with pytest.raises(Stage4RunnerRefusal):
        run(two_confirmations(), build_default_bed(), -1)


def test_unknown_regime_fails_closed() -> None:
    with pytest.raises(Stage4RunnerRefusal):
        run(two_confirmations(), build_default_bed(), 0, regime="sideways")


def test_bed_conforms_to_expected_controls() -> None:
    bed = Stage4IntegrationBed()
    assert bed.controls() == BASELINE_ARMS
    assert bed.matched_cost().flops > 0
