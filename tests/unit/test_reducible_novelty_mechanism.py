
from __future__ import annotations

import pytest

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.ladder.stage3_registry import _discover
from mop.ladder.stage_ladder import FIRST_ACTIVATION_STAGE
from mop.mechanisms import reducible_novelty_bed as bed_module
from mop.mechanisms import reducible_novelty_runner as runner_module
from mop.mechanisms.reducible_novelty_bed import (
    REGIME_FAVORABLE,
    REGIME_NULL,
    BedRefusal,
    ReducibleNoveltyBed,
)
from mop.mechanisms.reducible_novelty_impl import ImplRefusal, run_all, run_control
from mop.mechanisms.reducible_novelty_runner import (
    REQUIREMENT_ID,
    ReducibleNoveltyRunner,
    RunnerRefusal,
    RunResult,
)
from mop.mechanisms.reducible_novelty_scaffold import (
    REQUIRED_CONTROLS,
    DualMetricReading,
    JointClaimGate,
    ReducibleNoveltyRefusal,
    build_trap_verdict,
)

SEEDS = (0, 1, 2, 7, 42)
BAND_SEEDS = (171000001, 182000001, 193000001, 1600000001, 10600000001)


def _bed() -> ReducibleNoveltyBed:
    return ReducibleNoveltyBed()


def _runner() -> ReducibleNoveltyRunner:
    return ReducibleNoveltyRunner()


def _crafted(
    *, regime: str, mechanism: DualMetricReading, control: DualMetricReading
) -> RunResult:
    return RunResult(
        regime=regime,
        seed=0,
        mechanism_reading=mechanism,
        control_readings=(("uniform_allocation", control),),
    )


def test_bed_and_runner_conform_to_protocols() -> None:
    assert isinstance(_bed(), Bed)
    assert isinstance(_runner(), MechanismRunner)
    assert _bed().mechanism_id == _runner().mechanism_id == "reducible_novelty"


def test_bed_and_runner_are_discoverable_by_the_registry_pattern() -> None:
    bed = _discover(bed_module, "Bed")
    runner = _discover(runner_module, "Runner")
    assert isinstance(bed, ReducibleNoveltyBed)
    assert isinstance(runner, ReducibleNoveltyRunner)


def test_bed_declares_the_control_family_and_a_non_vacuous_budget() -> None:
    bed = _bed()
    assert bed.controls() == REQUIRED_CONTROLS
    budget = bed.matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


def test_run_is_deterministic() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
        first = runner.run(bed, seed, REGIME_FAVORABLE)
        second = runner.run(bed, seed, REGIME_FAVORABLE)
        assert first == second
        assert first.digest() == second.digest()
        assert runner.mint(first).digest() == runner.mint(second).digest()


def test_readings_are_reproducible_at_the_source() -> None:
    bed = _bed()
    panel = bed.favorable_regime(7)
    assert run_all(panel) == run_all(panel)


def test_null_regime_holds_the_trap_and_mints_null() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_NULL)
        assert result.both_axes_win is False
        assert result.mechanism_reading.learning_progress == 0.0
        assert result.mechanism_reading.allocation_efficiency == 0.0
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.is_confirmation is False


def test_favorable_regime_mints_mechanics_ok_over_every_control() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_FAVORABLE)
        assert result.progress_margin > 0.0
        assert result.efficiency_margin > 0.0
        assert result.both_axes_win is True
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_MECHANICS_OK
        assert set(receipt.controls_cleared) == set(REQUIRED_CONTROLS)
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.stage == FIRST_ACTIVATION_STAGE
        assert receipt.requirement_id == REQUIREMENT_ID
        assert receipt.is_confirmation is False


def test_favorable_novelty_chaser_is_pulled_toward_the_noise() -> None:
    bed = _bed()
    panel = bed.favorable_regime(0)
    chaser = run_control("novelty_chaser", panel)
    uniform = run_control("uniform_allocation", panel)
    assert chaser.allocation_efficiency < uniform.allocation_efficiency


def test_only_progress_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(learning_progress=0.90, allocation_efficiency=0.40),
        control=DualMetricReading(learning_progress=0.50, allocation_efficiency=0.60),
    )
    assert result.progress_margin > 0.0
    assert result.efficiency_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_only_efficiency_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(learning_progress=0.40, allocation_efficiency=0.90),
        control=DualMetricReading(learning_progress=0.60, allocation_efficiency=0.50),
    )
    assert result.efficiency_margin > 0.0
    assert result.progress_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_a_tie_on_an_axis_is_not_a_strict_win() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(learning_progress=0.80, allocation_efficiency=0.80),
        control=DualMetricReading(learning_progress=0.80, allocation_efficiency=0.50),
    )
    assert result.progress_margin == 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_evidence_digest_is_stable_and_well_formed() -> None:
    runner, bed = _runner(), _bed()
    result = runner.run(bed, 0, REGIME_FAVORABLE)
    receipt = runner.mint(result)
    assert len(receipt.evidence_digest) == 64
    assert receipt.evidence_digest == runner.mint(runner.run(bed, 0, REGIME_FAVORABLE)).evidence_digest


def test_mint_is_never_a_confirmation_on_either_regime() -> None:
    runner, bed = _runner(), _bed()
    for regime in (REGIME_NULL, REGIME_FAVORABLE):
        receipt = runner.mint(runner.run(bed, 0, regime))
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.is_confirmation is False


def test_run_result_refuses_an_empty_control_family() -> None:
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=0,
            mechanism_reading=DualMetricReading(learning_progress=0.9, allocation_efficiency=0.9),
            control_readings=(),
        )


def test_runner_refuses_an_unknown_regime_and_a_negative_seed() -> None:
    runner, bed = _runner(), _bed()
    with pytest.raises(RunnerRefusal):
        runner.run(bed, 0, "widened")
    with pytest.raises(BedRefusal):
        runner.run(bed, -1, REGIME_FAVORABLE)


def test_bed_refuses_an_unknown_regime() -> None:
    with pytest.raises(BedRefusal):
        _bed().regime("widened", 0)


def test_impl_refuses_an_unknown_control() -> None:
    panel = _bed().favorable_regime(0)
    with pytest.raises(ImplRefusal):
        run_control("clever_new_control", panel)


def test_scaffold_toy_holds_the_null_and_the_gate_stays_closed() -> None:
    verdict = build_trap_verdict(seed=0)
    assert verdict.both_axes_improved is False
    with pytest.raises(ReducibleNoveltyRefusal):
        verdict.certify()
    with pytest.raises(ReducibleNoveltyRefusal):
        JointClaimGate().authorize(verdict)
