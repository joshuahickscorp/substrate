
from __future__ import annotations

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.mechanisms.construction_search_bed import (
    CHEAP_CONTROLS,
    MECHANISM_ID,
    REQUIREMENT_ID,
    STAGE,
    ConstructionSearchBed,
)
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner

SEEDS = (0, 1, 2, 7, 42)


def _runner() -> ConstructionSearchRunner:
    return ConstructionSearchRunner()


def test_bed_and_runner_satisfy_ladder_protocols() -> None:
    assert isinstance(ConstructionSearchBed(), Bed)
    assert isinstance(_runner(), MechanismRunner)
    assert ConstructionSearchBed().mechanism_id == MECHANISM_ID
    assert _runner().mechanism_id == MECHANISM_ID
    assert ConstructionSearchBed().controls() == CHEAP_CONTROLS
    budget = ConstructionSearchBed().matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


def test_run_is_deterministic_per_seed() -> None:
    runner = _runner()
    for seed in SEEDS:
        first = runner.run(ConstructionSearchBed(), seed)
        second = runner.run(ConstructionSearchBed(), seed)
        assert first == second
        assert first.evidence_digest() == second.evidence_digest()
        assert first.favorable_nets == second.favorable_nets
        assert first.null_nets == second.null_nets


def test_mint_digest_is_stable() -> None:
    runner = _runner()
    first = runner.mint(runner.run(ConstructionSearchBed(), 42))
    second = runner.mint(runner.run(ConstructionSearchBed(), 42))
    assert first.digest() == second.digest()
    assert first.evidence_digest == second.evidence_digest
    assert len(first.evidence_digest) == 64
    assert first.digest() == first.digest()
    assert len(first.digest()) == 64


def test_favorable_regime_is_mechanics_ok() -> None:
    runner = _runner()
    for seed in SEEDS:
        results = runner.run(ConstructionSearchBed(), seed)
        assert results.favorable_beats_all is True
        assert results.null_holds is True
        for control in CHEAP_CONTROLS:
            assert results.favorable_margins[control] > 0.0
        receipt = runner.mint(results)
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.verdict == VERDICT_MECHANICS_OK
        assert receipt.mechanism_id == MECHANISM_ID
        assert receipt.requirement_id == REQUIREMENT_ID
        assert receipt.stage == STAGE
        assert receipt.controls_cleared == CHEAP_CONTROLS
        assert receipt.is_confirmation is False


def test_search_reaches_the_synergy_greedy_cannot() -> None:
    results = _runner().run(ConstructionSearchBed(), 42)
    assert results.favorable_margins["greedy-only"] > 0.0
    assert results.favorable_headroom_gap == 0.0


def test_null_regime_prior_holds_on_default_bed() -> None:
    for seed in SEEDS:
        results = _runner().run(ConstructionSearchBed(), seed)
        assert results.null_margins["greedy-only"] <= 0.0
        assert results.null_holds is True


def test_flat_bed_with_no_synergy_mints_null() -> None:
    runner = _runner()
    flat_bed = ConstructionSearchBed(synergy_bonus=0.0)
    for seed in SEEDS:
        results = runner.run(flat_bed, seed)
        assert results.favorable_beats_all is False
        receipt = runner.mint(results)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.controls_cleared == ()
        assert receipt.is_confirmation is False


def test_high_charged_cost_erases_the_gain_and_is_not_mechanics_ok() -> None:
    runner = _runner()
    costly_bed = ConstructionSearchBed(per_eval_cost=0.01)
    for seed in SEEDS:
        results = runner.run(costly_bed, seed)
        assert results.favorable_beats_all is False
        assert results.favorable_margins["greedy-only"] <= 0.0
        receipt = runner.mint(results)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.verdict != VERDICT_MECHANICS_OK
        assert receipt.is_confirmation is False


def test_demonstration_is_never_a_confirmation() -> None:
    runner = _runner()
    for bed in (
        ConstructionSearchBed(),
        ConstructionSearchBed(synergy_bonus=0.0),
        ConstructionSearchBed(per_eval_cost=0.01),
    ):
        receipt = runner.mint(runner.run(bed, 7))
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.is_confirmation is False
