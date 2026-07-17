"""Tests for the calibrated uncertainty mechanism: determinism and fail-closed on the decoupled null.

The runner measures selective risk reduction and decision utility for the mechanism and each control
on both regimes and mints a mechanics-demonstration receipt. The load-bearing guarantees pinned here:

- Determinism: the same bed and seed give a byte-identical result and receipt digest.
- The NULL regime always mints ``null``: the decoupled signal sits below the answer bar, so the
  mechanism collapses onto frozen_uniform and a strict joint win is impossible by construction.
- The FAVORABLE regime mints ``mechanics-ok`` for a strict both-axes win over every control, at
  fixed engineered margins that hold for every nonnegative seed by construction.
- Improving only one axis is NOT mechanics-ok: that is the decoupled confidence null.
- The receipt is never a scientific confirmation.
- The bed and runner are discoverable via the stage3_registry _discover pattern.

No capability is claimed.
"""

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
from mop.mechanisms import calibrated_uncertainty_bed as bed_module
from mop.mechanisms import calibrated_uncertainty_runner as runner_module
from mop.mechanisms.calibrated_uncertainty_bed import (
    ANSWER_THRESHOLD,
    INCORRECT_COUNT,
    REGIME_FAVORABLE,
    REGIME_NULL,
    TASK_COUNT,
    BedRefusal,
    CalibratedUncertaintyBed,
)
from mop.mechanisms.calibrated_uncertainty_impl import ImplRefusal, run_all, run_control
from mop.mechanisms.calibrated_uncertainty_runner import (
    REQUIREMENT_ID,
    CalibratedUncertaintyRunner,
    RunnerRefusal,
    RunResult,
)
from mop.mechanisms.calibrated_uncertainty_scaffold import REQUIRED_CONTROLS, DualMetricReading

SEEDS = (0, 1, 2, 7, 42, 255)
BAND_SEEDS = (171000001, 182000001, 193000001, 1600000001, 10600000001)


def _bed() -> CalibratedUncertaintyBed:
    return CalibratedUncertaintyBed()


def _runner() -> CalibratedUncertaintyRunner:
    return CalibratedUncertaintyRunner()


def _crafted(
    *, regime: str, mechanism: DualMetricReading, control: DualMetricReading
) -> RunResult:
    return RunResult(
        regime=regime,
        seed=0,
        mechanism_reading=mechanism,
        control_readings=(("always_answer", control),),
    )


# ---------------------------------------------------------------------------
# Structural conformance and registry compatibility.
# ---------------------------------------------------------------------------


def test_bed_and_runner_conform_to_protocols() -> None:
    assert isinstance(_bed(), Bed)
    assert isinstance(_runner(), MechanismRunner)
    assert _bed().mechanism_id == _runner().mechanism_id == "calibrated_uncertainty"


def test_bed_declares_the_control_family_and_a_non_vacuous_budget() -> None:
    bed = _bed()
    assert bed.controls() == REQUIRED_CONTROLS
    budget = bed.matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


def test_bed_and_runner_are_discoverable_via_the_registry_pattern() -> None:
    assert isinstance(_discover(bed_module, "Bed"), CalibratedUncertaintyBed)
    assert isinstance(_discover(runner_module, "Runner"), CalibratedUncertaintyRunner)


# ---------------------------------------------------------------------------
# Bed construction guarantees the win and the null are structural, not sampled.
# ---------------------------------------------------------------------------


def test_favorable_batch_separates_confidence_by_correctness() -> None:
    bed = _bed()
    for seed in SEEDS:
        batch = bed.favorable_regime(seed)
        assert sum(1 for bit in batch.correctness if bit == 0) == INCORRECT_COUNT
        for bit, conf in zip(batch.correctness, batch.confidence, strict=True):
            if bit == 1:
                assert conf > ANSWER_THRESHOLD
            else:
                assert conf < ANSWER_THRESHOLD


def test_favorable_incorrect_tasks_cover_both_parities_for_every_seed() -> None:
    bed = _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        batch = bed.favorable_regime(seed)
        incorrect = [index for index, bit in enumerate(batch.correctness) if bit == 0]
        assert sum(1 for index in incorrect if index % 2 == 0) == INCORRECT_COUNT // 2
        assert sum(1 for index in incorrect if index % 2 == 1) == INCORRECT_COUNT // 2


def test_null_batch_confidence_is_wholly_below_the_answer_bar() -> None:
    bed = _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        batch = bed.null_regime(seed)
        assert all(conf < ANSWER_THRESHOLD for conf in batch.confidence)


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_run_is_deterministic() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
        for regime in (REGIME_FAVORABLE, REGIME_NULL):
            first = runner.run(bed, seed, regime)
            second = runner.run(bed, seed, regime)
            assert first == second
            assert first.digest() == second.digest()
            assert runner.mint(first).digest() == runner.mint(second).digest()


def test_readings_are_reproducible_at_the_source() -> None:
    bed = _bed()
    batch = bed.favorable_regime(7)
    assert run_all(batch) == run_all(batch)


# ---------------------------------------------------------------------------
# The null regime holds the decoupled null: always null, never a both-axes win.
# ---------------------------------------------------------------------------


def test_null_regime_holds_the_null_and_mints_null() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_NULL)
        assert result.both_axes_win is False
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.is_confirmation is False


def test_null_regime_mechanism_ties_frozen_uniform_by_construction() -> None:
    bed = _bed()
    for seed in SEEDS:
        readings = run_all(bed.null_regime(seed))
        assert readings["mechanism"] == readings["frozen_uniform"]


# ---------------------------------------------------------------------------
# The favorable regime mints mechanics-ok on a strict both-axes win.
# ---------------------------------------------------------------------------


def test_favorable_regime_mints_mechanics_ok_over_every_control() -> None:
    runner, bed = _runner(), _bed()
    for seed in (*SEEDS, *BAND_SEEDS):
        result = runner.run(bed, seed, REGIME_FAVORABLE)
        assert result.risk_margin > 0.0
        assert result.utility_margin > 0.0
        assert result.both_axes_win is True
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_MECHANICS_OK
        assert set(receipt.controls_cleared) == set(REQUIRED_CONTROLS)
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.stage == FIRST_ACTIVATION_STAGE
        assert receipt.requirement_id == REQUIREMENT_ID
        assert receipt.is_confirmation is False


def test_favorable_margins_are_the_engineered_constants() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
        result = runner.run(bed, seed, REGIME_FAVORABLE)
        assert result.risk_margin == pytest.approx(0.25, abs=1e-12)
        assert result.utility_margin == pytest.approx(0.125, abs=1e-12)
        assert result.mechanism_reading.selective_risk_reduction == pytest.approx(1.0)
        expected_error_rate = INCORRECT_COUNT / TASK_COUNT
        assert result.mechanism_reading.decision_utility == pytest.approx(
            (1.0 - expected_error_rate + 1.0) / 2.0
        )


# ---------------------------------------------------------------------------
# Fail-closed on the null: a single-axis win is never mechanics-ok.
# ---------------------------------------------------------------------------


def test_only_risk_reduction_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(selective_risk_reduction=0.90, decision_utility=0.40),
        control=DualMetricReading(selective_risk_reduction=0.50, decision_utility=0.60),
    )
    assert result.risk_margin > 0.0
    assert result.utility_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_only_utility_improved_is_not_mechanics_ok() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(selective_risk_reduction=0.40, decision_utility=0.90),
        control=DualMetricReading(selective_risk_reduction=0.60, decision_utility=0.50),
    )
    assert result.utility_margin > 0.0
    assert result.risk_margin < 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


def test_a_tie_on_an_axis_is_not_a_strict_win() -> None:
    runner = _runner()
    result = _crafted(
        regime=REGIME_FAVORABLE,
        mechanism=DualMetricReading(selective_risk_reduction=0.80, decision_utility=0.80),
        control=DualMetricReading(selective_risk_reduction=0.80, decision_utility=0.50),
    )
    assert result.risk_margin == 0.0
    assert result.both_axes_win is False
    assert runner.mint(result).verdict == VERDICT_NULL


# ---------------------------------------------------------------------------
# Digest stability and receipt honesty.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Refusals: unknown regimes, negative seeds, malformed results, unknown controls.
# ---------------------------------------------------------------------------


def test_bed_refuses_an_unknown_regime_and_a_negative_seed() -> None:
    bed = _bed()
    with pytest.raises(BedRefusal):
        bed.regime("widened", 0)
    with pytest.raises(BedRefusal):
        bed.favorable_regime(-1)
    with pytest.raises(BedRefusal):
        bed.null_regime(-1)


def test_runner_refuses_an_unknown_regime_and_a_mismatched_bed() -> None:
    runner, bed = _runner(), _bed()
    with pytest.raises(RunnerRefusal):
        runner.run(bed, 0, "widened")

    class WrongBed:
        mechanism_id = "some_other_mechanism"

    with pytest.raises(RunnerRefusal):
        runner.run(WrongBed(), 0, REGIME_FAVORABLE)


def test_impl_refuses_an_unknown_control() -> None:
    bed = _bed()
    with pytest.raises(ImplRefusal):
        run_control("widened_control", bed.favorable_regime(0))


def test_run_result_refuses_an_empty_control_family() -> None:
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=0,
            mechanism_reading=DualMetricReading(
                selective_risk_reduction=0.9, decision_utility=0.9
            ),
            control_readings=(),
        )


def test_run_result_refuses_an_unknown_control_and_a_negative_seed() -> None:
    reading = DualMetricReading(selective_risk_reduction=0.9, decision_utility=0.9)
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=0,
            mechanism_reading=reading,
            control_readings=(("widened_control", reading),),
        )
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=-1,
            mechanism_reading=reading,
            control_readings=(("always_answer", reading),),
        )
