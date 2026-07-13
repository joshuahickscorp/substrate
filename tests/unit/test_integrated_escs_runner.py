"""Tests for the integrated ESCS runner: determinism, the null and favorable verdicts, and fail-closed.

The runner mints mechanics demonstrations only. The verdict is mechanics-ok exactly when the
integrated organization Pareto-dominates every matched baseline on the favorable regime and every
added mechanism justifies its marginal compute, while the null regime still holds null. Any baseline
that ties or beats the integrated point at matched cost blocks mechanics-ok. No capability is claimed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.mechanisms.integrated_escs_bed import EscsRegime, build_default_bed
from mop.mechanisms.integrated_escs_runner import (
    RunResult,
    build_default_runner,
    evaluate_regime,
)
from mop.mechanisms.integrated_escs_scaffold import REQUIRED_BASELINES


def test_bed_and_runner_satisfy_protocols() -> None:
    assert isinstance(build_default_bed(), Bed)
    assert isinstance(build_default_runner(), MechanismRunner)


def test_run_is_deterministic() -> None:
    runner = build_default_runner()
    bed = build_default_bed()
    first = runner.run(bed, 7)
    second = runner.run(bed, 7)
    assert first.digest() == second.digest()
    assert runner.mint(first).digest() == runner.mint(second).digest()
    assert runner.mint(first).evidence_digest == runner.mint(second).evidence_digest


def test_null_regime_yields_null() -> None:
    runner = build_default_runner()
    results = runner.run(build_default_bed(), 3)
    assert results.null_report.dominates_all() is False
    receipt = runner.demonstration_for(results.null_report)
    assert receipt.verdict == VERDICT_NULL
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.is_confirmation is False


def test_favorable_regime_yields_mechanics_ok() -> None:
    runner = build_default_runner()
    results = runner.run(build_default_bed(), 3)
    assert results.favorable_report.dominates_all() is True
    assert results.favorable_report.ablation_all_justified() is True
    receipt = runner.demonstration_for(results.favorable_report)
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.is_confirmation is False
    assert set(receipt.controls_cleared) == set(REQUIRED_BASELINES)


def test_combined_mint_is_a_mechanics_demonstration() -> None:
    runner = build_default_runner()
    results = runner.run(build_default_bed(), 11)
    receipt = runner.mint(results)
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.is_confirmation is False
    assert receipt.mechanism_id == "integrated_escs"
    assert receipt.requirement_id == "s3.integrated_escs"
    assert set(receipt.controls_cleared) == set(REQUIRED_BASELINES)
    assert "favorable" in receipt.detail
    assert "null" in receipt.detail


def test_fail_closed_when_a_baseline_is_not_dominated() -> None:
    runner = build_default_runner()
    # A regime where the first mechanism is not demanded: the single-perspective baseline, which
    # ablates that mechanism, ties the integrated organization at matched cost, so it is not dominated.
    tie_regime = EscsRegime(name="tie", demands=(0.0, 0.5, 0.6, 0.5), seed=1)
    report = evaluate_regime(tie_regime, 1)
    assert "single-perspective" not in report.dominated
    assert report.dominates_all() is False
    assert runner.demonstration_for(report).verdict == VERDICT_NULL


def test_combined_mint_fails_closed_when_favorable_does_not_dominate() -> None:
    runner = build_default_runner()
    bed = build_default_bed()
    tie_regime = EscsRegime(name="tie", demands=(0.0, 0.5, 0.6, 0.5), seed=1)
    tie_report = evaluate_regime(tie_regime, 1)
    forced = RunResult(
        seed=1,
        null_report=evaluate_regime(bed.null_regime(1), 1),
        favorable_report=tie_report,
        matched=bed.matched_cost(),
    )
    receipt = runner.mint(forced)
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_evidence_digest_is_stable_and_sha256() -> None:
    runner = build_default_runner()
    results = runner.run(build_default_bed(), 5)
    receipt = runner.mint(results)
    again = runner.mint(runner.run(build_default_bed(), 5))
    assert receipt.evidence_digest == again.evidence_digest
    assert len(receipt.evidence_digest) == 64
    assert receipt.digest() == again.digest()


def test_no_receipt_is_ever_a_confirmation() -> None:
    runner = build_default_runner()
    results = runner.run(build_default_bed(), 2)
    for receipt in (
        runner.mint(results),
        runner.demonstration_for(results.null_report),
        runner.demonstration_for(results.favorable_report),
    ):
        assert receipt.is_confirmation is False
        assert receipt.kind == KIND_DEMONSTRATION
