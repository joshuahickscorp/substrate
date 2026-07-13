"""Tests for the niche_dispatch runner: determinism, the null vs favorable verdicts, fail-closed.

These pin the honesty boundary for the niche_dispatch workstream. The named prior null (an EDCM
invalid bed: overlapping, partly harmful niches) must hold on the null regime, so dispatch there can
never beat every matched control. Only a genuine favorable win, where the dispatcher strictly beats
all four declared controls, yields a mechanics-ok demonstration, and even then the receipt is never a
scientific confirmation. A tie or loss on any control fails closed to the null verdict.
"""

from __future__ import annotations

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.mechanisms.niche_dispatch_bed import NicheDispatchBed, build_default_bed
from mop.mechanisms.niche_dispatch_runner import NicheDispatchRunner
from mop.mechanisms.niche_dispatch_scaffold import DISPATCH_CONTROLS

SEED = 7


def test_bed_and_runner_satisfy_the_protocols() -> None:
    assert isinstance(build_default_bed(), Bed)
    assert isinstance(NicheDispatchRunner(), MechanismRunner)


def test_bed_controls_are_the_declared_family() -> None:
    assert build_default_bed().controls() == DISPATCH_CONTROLS


def test_matched_cost_is_non_vacuous() -> None:
    matched = build_default_bed().matched_cost()
    assert matched.params > 0
    assert matched.flops > 0
    assert matched.wall_ns > 0
    assert matched.seeds > 0


def test_run_is_deterministic() -> None:
    runner = NicheDispatchRunner()
    bed = build_default_bed()
    first = runner.run(bed, SEED)
    second = runner.run(bed, SEED)
    assert first == second
    assert first.evidence_digest() == second.evidence_digest()
    assert runner.mint(first).digest() == runner.mint(second).digest()


def test_null_regime_yields_null_and_named_null_holds() -> None:
    # On the null regime the named prior null holds: dispatch cannot beat every matched control.
    runner = NicheDispatchRunner()
    results = runner.run(build_default_bed(), SEED)
    assert results.null_holds is True
    assert results.null_report.beats_every_control is False
    # A single-best control ties dispatch on the overlapping bed, so no strict win exists.
    assert results.null_report.controls_beaten != DISPATCH_CONTROLS


def test_favorable_regime_beats_every_control_and_is_mechanics_ok() -> None:
    runner = NicheDispatchRunner()
    results = runner.run(build_default_bed(), SEED)
    assert results.favorable_win is True
    assert results.favorable_report.beats_every_control is True
    receipt = runner.mint(results)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.controls_cleared == DISPATCH_CONTROLS
    assert receipt.is_confirmation is False


def test_favorable_dispatch_strictly_beats_each_control() -> None:
    runner = NicheDispatchRunner()
    report = runner.run(build_default_bed(), SEED).favorable_report
    control_scores = dict(report.control_accuracies)
    for name in DISPATCH_CONTROLS:
        assert report.dispatch_accuracy > control_scores[name]


def test_tie_or_loss_fails_closed_to_null_verdict() -> None:
    # Degrade the favorable regime into an overlapping one; a single-best control now ties dispatch,
    # so the runner must refuse mechanics-ok and fall back to the null verdict.
    runner = NicheDispatchRunner()
    degraded = NicheDispatchBed(favorable_disjoint=False)
    results = runner.run(degraded, SEED)
    assert results.favorable_win is False
    assert results.favorable_report.beats_every_control is False
    receipt = runner.mint(results)
    assert receipt.verdict == VERDICT_NULL
    assert receipt.controls_cleared != DISPATCH_CONTROLS
    assert receipt.is_confirmation is False


def test_receipt_never_a_confirmation_on_either_verdict() -> None:
    runner = NicheDispatchRunner()
    ok_receipt = runner.mint(runner.run(build_default_bed(), SEED))
    null_receipt = runner.mint(runner.run(NicheDispatchBed(favorable_disjoint=False), SEED))
    assert ok_receipt.is_confirmation is False
    assert null_receipt.is_confirmation is False
    assert ok_receipt.kind == KIND_DEMONSTRATION
    assert null_receipt.kind == KIND_DEMONSTRATION


def test_receipt_metadata_and_digest_are_stable() -> None:
    runner = NicheDispatchRunner()
    receipt = runner.mint(runner.run(build_default_bed(), SEED))
    assert receipt.mechanism_id == "niche_dispatch"
    assert receipt.requirement_id == "s3.niche_dispatch"
    assert receipt.stage == 3
    assert len(receipt.evidence_digest) == 64
    again = runner.mint(runner.run(build_default_bed(), SEED))
    assert receipt.evidence_digest == again.evidence_digest
