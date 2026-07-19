
from __future__ import annotations

import importlib

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
from mop.mechanisms.stability_plasticity_r2_bed import (
    HISTORY_TASKS,
    NO_RECURRENCE,
    REGIME_FAVORABLE,
    REGIME_NULL,
    BedRefusal,
    StabilityPlasticityR2Bed,
    TaskStream,
)
from mop.mechanisms.stability_plasticity_r2_impl import ImplRefusal, run_all, run_control
from mop.mechanisms.stability_plasticity_r2_runner import (
    REQUIREMENT_ID,
    RunnerRefusal,
    RunResult,
    StabilityPlasticityR2Runner,
)
from mop.mechanisms.stability_plasticity_r2_scaffold import (
    REQUIRED_CONTROLS,
    DualMetricReading,
    StabilityPlasticityR2Refusal,
    default_contract,
)

SEEDS = (0, 1, 2, 7, 42, 255)
BAND_SEEDS = (171000001, 182000001, 193000001, 1600000001, 10600000001)


def _bed() -> StabilityPlasticityR2Bed:
    return StabilityPlasticityR2Bed()


def _runner() -> StabilityPlasticityR2Runner:
    return StabilityPlasticityR2Runner()


def _crafted(
    *, regime: str, mechanism: DualMetricReading, control: DualMetricReading
) -> RunResult:
    return RunResult(
        regime=regime,
        seed=0,
        mechanism_reading=mechanism,
        control_readings=(("fresh_init", control),),
    )




def test_bed_and_runner_conform_to_protocols() -> None:
    assert isinstance(_bed(), Bed)
    assert isinstance(_runner(), MechanismRunner)
    assert _bed().mechanism_id == _runner().mechanism_id == "stability_plasticity_r2"


def test_bed_declares_the_control_family_and_a_non_vacuous_budget() -> None:
    bed = _bed()
    assert bed.controls() == REQUIRED_CONTROLS
    assert REQUIRED_CONTROLS == ("fresh_init", "frozen_core", "full_retrain", "no_replay")
    budget = bed.matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


def test_bed_and_runner_are_discoverable_via_the_registry_pattern() -> None:
    bed_module = importlib.import_module("mop.mechanisms.stability_plasticity_r2_bed")
    runner_module = importlib.import_module("mop.mechanisms.stability_plasticity_r2_runner")
    bed = _discover(bed_module, "Bed")
    runner = _discover(runner_module, "Runner")
    assert isinstance(bed, StabilityPlasticityR2Bed)
    assert isinstance(runner, StabilityPlasticityR2Runner)
    receipt = runner.mint(runner.run(bed, 0))
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.is_confirmation is False


def test_default_contract_pins_the_joint_bar() -> None:
    contract = default_contract()
    assert contract.both_axes_required is True
    assert contract.matched_cost_required is True
    assert contract.prior_null == "p6-stability-plasticity-split"




def test_run_is_deterministic() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
        first = runner.run(bed, seed, REGIME_FAVORABLE)
        second = runner.run(bed, seed, REGIME_FAVORABLE)
        assert first == second
        assert first.digest() == second.digest()
        assert runner.mint(first).digest() == runner.mint(second).digest()


def test_null_run_is_deterministic_too() -> None:
    runner, bed = _runner(), _bed()
    for seed in (0, 7, 255):
        assert (
            runner.mint(runner.run(bed, seed, REGIME_NULL)).digest()
            == runner.mint(runner.run(bed, seed, REGIME_NULL)).digest()
        )


def test_readings_are_reproducible_at_the_source() -> None:
    bed = _bed()
    stream = bed.favorable_regime(7)
    assert run_all(stream) == run_all(stream)




def test_favorable_stream_carries_an_honest_interior_recurrence() -> None:
    bed = _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        stream = bed.favorable_regime(seed)
        recurring = stream.future_recurrence_index
        assert 0 < recurring < HISTORY_TASKS - 1
        assert stream.recurrence_flags[recurring] is True
        assert sum(stream.recurrence_flags) == 1
        assert stream.future[stream.core_dim :] == stream.history[recurring][stream.core_dim :]
        for task in (*stream.history, stream.future):
            assert task[: stream.core_dim] == stream.history[0][: stream.core_dim]


def test_null_stream_keeps_the_signal_silent_and_conflicts_the_core() -> None:
    bed = _bed()
    for seed in (0, 7, 255):
        stream = bed.null_regime(seed)
        assert not any(stream.recurrence_flags)
        assert stream.future_recurrence_index == NO_RECURRENCE
        for dim in range(stream.core_dim):
            assert stream.history[0][dim] > 0.0
            assert stream.future[dim] < 0.0




def test_null_regime_holds_the_split_and_mints_null() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_NULL)
        assert result.both_axes_win is False
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.is_confirmation is False




def test_favorable_regime_mints_mechanics_ok_over_every_control() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_FAVORABLE)
        assert result.retention_margin > 0.0
        assert result.plasticity_margin > 0.0
        assert result.both_axes_win is True
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_MECHANICS_OK
        assert set(receipt.controls_cleared) == set(REQUIRED_CONTROLS)
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.stage == FIRST_ACTIVATION_STAGE
        assert receipt.requirement_id == REQUIREMENT_ID
        assert receipt.is_confirmation is False




def test_only_retention_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(retention=0.90, future_learnability=0.40),
        control=DualMetricReading(retention=0.50, future_learnability=0.60),
    )
    assert result.retention_margin > 0.0
    assert result.plasticity_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_only_plasticity_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(retention=0.40, future_learnability=0.90),
        control=DualMetricReading(retention=0.60, future_learnability=0.50),
    )
    assert result.plasticity_margin > 0.0
    assert result.retention_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_a_tie_on_an_axis_is_not_a_strict_win() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(retention=0.80, future_learnability=0.80),
        control=DualMetricReading(retention=0.80, future_learnability=0.50),
    )
    assert result.retention_margin == 0.0
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




def test_runner_refuses_an_unknown_regime_and_a_foreign_bed() -> None:
    runner, bed = _runner(), _bed()
    with pytest.raises(RunnerRefusal):
        runner.run(bed, 0, "sideways")
    foreign = StabilityPlasticityR2Bed.__new__(StabilityPlasticityR2Bed)
    object.__setattr__(foreign, "mechanism_id", "stability_plasticity")
    with pytest.raises(RunnerRefusal):
        runner.run(foreign, 0, REGIME_FAVORABLE)


def test_bed_refuses_negative_seeds_and_unknown_regimes() -> None:
    bed = _bed()
    with pytest.raises(BedRefusal):
        bed.favorable_regime(-1)
    with pytest.raises(BedRefusal):
        bed.null_regime(-1)
    with pytest.raises(BedRefusal):
        bed.regime("sideways", 0)


def test_stream_refuses_a_dishonest_recurrence_signal() -> None:
    bed = _bed()
    honest = bed.favorable_regime(0)
    with pytest.raises(BedRefusal):
        TaskStream(
            regime=REGIME_FAVORABLE,
            seed=0,
            history=honest.history,
            future=honest.future,
            recurrence_flags=tuple(False for _ in honest.history),
            future_recurrence_index=honest.future_recurrence_index,
        )
    with pytest.raises(BedRefusal):
        TaskStream(
            regime=REGIME_NULL,
            seed=0,
            history=honest.history,
            future=honest.future,
            recurrence_flags=honest.recurrence_flags,
            future_recurrence_index=honest.future_recurrence_index,
        )
    boundary_flags = tuple(index == 0 for index in range(len(honest.history)))
    with pytest.raises(BedRefusal):
        TaskStream(
            regime=REGIME_FAVORABLE,
            seed=0,
            history=honest.history,
            future=honest.future,
            recurrence_flags=boundary_flags,
            future_recurrence_index=0,
        )


def test_impl_refuses_an_unknown_control() -> None:
    stream = _bed().favorable_regime(0)
    with pytest.raises(ImplRefusal):
        run_control("fresh-init", stream)


def test_run_result_refuses_an_empty_control_family() -> None:
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=0,
            mechanism_reading=DualMetricReading(retention=0.9, future_learnability=0.9),
            control_readings=(),
        )


def test_scaffold_refuses_widened_claim_scope_and_drifted_controls() -> None:
    with pytest.raises(StabilityPlasticityR2Refusal):
        DualMetricReading(retention=0.5, future_learnability=0.5, claim_scope="anything goes")
    from mop.mechanisms.stability_plasticity_r2_scaffold import assert_control_completeness

    with pytest.raises(StabilityPlasticityR2Refusal):
        assert_control_completeness(("fresh_init", "frozen_core", "full_retrain"))
