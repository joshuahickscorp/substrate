"""Tests for the stability vs plasticity runner: determinism and fail-closed on the P6 split.

The runner measures retention and future-learnability for the mechanism and each control on both
regimes and mints a mechanics-demonstration receipt. The load-bearing guarantees pinned here:

- Determinism: the same bed and seed give a byte-identical result and receipt digest.
- The NULL regime always mints ``null``; the split genuinely holds there (not a both-axes win).
- The FAVORABLE regime mints ``mechanics-ok`` only for a strict both-axes win over every control.
- Improving only retention, or only future-learnability, is NOT mechanics-ok: that is the split.
- The receipt is never a scientific confirmation.

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
from mop.ladder.stage_ladder import FIRST_ACTIVATION_STAGE
from mop.mechanisms.stability_plasticity_bed import (
    REGIME_FAVORABLE,
    REGIME_NULL,
    StabilityPlasticityBed,
)
from mop.mechanisms.stability_plasticity_impl import run_all
from mop.mechanisms.stability_plasticity_runner import (
    REQUIREMENT_ID,
    RunnerRefusal,
    RunResult,
    StabilityPlasticityRunner,
)
from mop.mechanisms.stability_plasticity_scaffold import REQUIRED_CONTROLS, DualMetricReading

SEEDS = (0, 1, 2, 7, 42)


def _bed() -> StabilityPlasticityBed:
    return StabilityPlasticityBed()


def _runner() -> StabilityPlasticityRunner:
    return StabilityPlasticityRunner()


def _crafted(
    *, regime: str, mechanism: DualMetricReading, control: DualMetricReading
) -> RunResult:
    return RunResult(
        regime=regime,
        seed=0,
        mechanism_reading=mechanism,
        control_readings=(("fresh-init", control),),
    )


# ---------------------------------------------------------------------------
# Structural conformance.
# ---------------------------------------------------------------------------


def test_bed_and_runner_conform_to_protocols() -> None:
    assert isinstance(_bed(), Bed)
    assert isinstance(_runner(), MechanismRunner)
    assert _bed().mechanism_id == _runner().mechanism_id == "stability_plasticity"


def test_bed_declares_the_control_family_and_a_non_vacuous_budget() -> None:
    bed = _bed()
    assert bed.controls() == REQUIRED_CONTROLS
    budget = bed.matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


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
    stream = bed.favorable_regime(7)
    assert run_all(stream) == run_all(stream)


# ---------------------------------------------------------------------------
# The null regime holds the split: always null, never a both-axes win.
# ---------------------------------------------------------------------------


def test_null_regime_holds_the_split_and_mints_null() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
        result = runner.run(bed, seed, REGIME_NULL)
        assert result.both_axes_win is False
        receipt = runner.mint(result)
        assert receipt.verdict == VERDICT_NULL
        assert receipt.is_confirmation is False


# ---------------------------------------------------------------------------
# The favorable regime mints mechanics-ok on a strict both-axes win.
# ---------------------------------------------------------------------------


def test_favorable_regime_mints_mechanics_ok_over_every_control() -> None:
    runner, bed = _runner(), _bed()
    for seed in SEEDS:
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


# ---------------------------------------------------------------------------
# Fail-closed on the split: a single-axis win is never mechanics-ok.
# ---------------------------------------------------------------------------


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


def test_run_result_refuses_an_empty_control_family() -> None:
    with pytest.raises(RunnerRefusal):
        RunResult(
            regime=REGIME_FAVORABLE,
            seed=0,
            mechanism_reading=DualMetricReading(retention=0.9, future_learnability=0.9),
            control_readings=(),
        )
