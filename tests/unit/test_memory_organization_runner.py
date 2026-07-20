
from __future__ import annotations

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
)
from mop.mechanisms.memory_organization_bed import (
    FUTURE_DECISION_CONTROLS,
    favorable_bed,
    harmful_bed,
    prediction_only_bed,
    structureless_bed,
)
from mop.mechanisms.memory_organization_runner import MemoryOrganizationRunner

SEED = 7


def _runner() -> MemoryOrganizationRunner:
    return MemoryOrganizationRunner()


def test_run_and_mint_are_deterministic() -> None:
    runner = _runner()
    bed = favorable_bed()
    first = runner.mint(runner.run(bed, SEED))
    second = runner.mint(runner.run(bed, SEED))
    assert first.evidence_digest == second.evidence_digest
    assert first.digest() == second.digest()
    assert first.payload() == second.payload()


def test_null_regime_yields_null_verdict() -> None:
    runner = _runner()
    receipt = runner.mint(runner.run(structureless_bed(), SEED))
    assert receipt.verdict == VERDICT_NULL
    assert receipt.detail["null_holds"] is True
    assert receipt.detail["favorable_win"] is False
    assert receipt.is_confirmation is False


def test_favorable_regime_yields_mechanics_ok() -> None:
    runner = _runner()
    receipt = runner.mint(runner.run(favorable_bed(), SEED))
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.detail["favorable_win"] is True
    assert receipt.detail["null_holds"] is True
    assert receipt.detail["harmed"] is False
    assert tuple(receipt.controls_cleared) == FUTURE_DECISION_CONTROLS
    for control in FUTURE_DECISION_CONTROLS:
        assert receipt.detail["favorable_margins"][control] > 0.0
    assert receipt.is_confirmation is False


def test_prediction_only_improvement_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = runner.run(prediction_only_bed(), SEED)
    receipt = runner.mint(result)
    assert receipt.detail["favorable_prediction_margins"]["stale-memory"] > 0.0
    assert receipt.detail["favorable_margins"]["stale-memory"] == 0.0
    assert receipt.detail["favorable_win"] is False
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_harming_case_blocks_mechanics_ok() -> None:
    runner = _runner()
    result = runner.run(harmful_bed(), SEED)
    receipt = runner.mint(result)
    assert result.harmed is True
    assert receipt.detail["harmed"] is True
    assert receipt.detail["harm_value"] > 0.0
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_evidence_digest_is_stable_across_beds_but_kind_is_demonstration() -> None:
    runner = _runner()
    favorable = runner.mint(runner.run(favorable_bed(), SEED))
    null = runner.mint(runner.run(structureless_bed(), SEED))
    assert len(favorable.evidence_digest) == 64
    assert favorable.evidence_digest != null.evidence_digest
    assert favorable.kind == KIND_DEMONSTRATION
    assert null.kind == KIND_DEMONSTRATION


def test_no_receipt_is_ever_a_confirmation() -> None:
    runner = _runner()
    for bed in (favorable_bed(), structureless_bed(), prediction_only_bed(), harmful_bed()):
        receipt = runner.mint(runner.run(bed, SEED))
        assert receipt.is_confirmation is False
        assert receipt.kind == KIND_DEMONSTRATION
