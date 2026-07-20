
from __future__ import annotations

from typing import Any

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.mechanisms.trace_stability_bed import TraceStabilityBed
from mop.mechanisms.trace_stability_impl import SeedMeasurement
from mop.mechanisms.trace_stability_runner import RunResults, TraceStabilityRunner
from mop.mechanisms.trace_stability_scaffold import REQUIRED_CONTROLS


class NullOnlyBed:

    mechanism_id = "trace_stability"

    def __init__(self, base: TraceStabilityBed) -> None:
        self._base = base

    def controls(self) -> tuple[str, ...]:
        return self._base.controls()

    def matched_cost(self) -> Any:
        return self._base.matched_cost()

    def null_regime(self, seed: int) -> SeedMeasurement:
        return self._base.null_regime(seed)

    def favorable_regime(self, seed: int) -> SeedMeasurement:
        return self._base.null_regime(seed)


def test_bed_and_runner_conform_to_protocols() -> None:
    assert isinstance(TraceStabilityBed(), Bed)
    assert isinstance(TraceStabilityRunner(), MechanismRunner)


def test_run_is_deterministic_per_seed() -> None:
    bed = TraceStabilityBed()
    runner = TraceStabilityRunner()
    first = runner.run(bed, 7)
    second = runner.run(bed, 7)
    assert first == second
    assert isinstance(first, RunResults)
    assert runner.mint(first).digest() == runner.mint(second).digest()


def test_distinct_seeds_give_distinct_receipt_digests() -> None:
    bed = TraceStabilityBed()
    runner = TraceStabilityRunner()
    assert runner.mint(runner.run(bed, 3)).digest() != runner.mint(runner.run(bed, 4)).digest()


def test_measured_agreements_separate_favorable_from_controls() -> None:
    results = TraceStabilityRunner().run(TraceStabilityBed(), 3)
    assert results.favorable_agreement >= results.threshold
    assert results.null_agreement < results.threshold
    for control in REQUIRED_CONTROLS:
        assert results.control_agreements[control] < results.threshold


def test_favorable_regime_is_mechanics_ok_with_controls_beaten() -> None:
    bed = TraceStabilityBed()
    runner = TraceStabilityRunner()
    receipt = runner.mint(runner.run(bed, 3))
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.controls_cleared == REQUIRED_CONTROLS
    assert receipt.requirement_id == "s3.trace_stability"
    assert receipt.is_confirmation is False


def test_null_regime_is_null() -> None:
    bed = NullOnlyBed(TraceStabilityBed())
    runner = TraceStabilityRunner()
    receipt = runner.mint(runner.run(bed, 3))
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_leaking_control_blocks_mechanics_ok() -> None:
    bed = TraceStabilityBed()
    runner = TraceStabilityRunner(leaked_controls=frozenset({"shuffled-seed"}))
    results = runner.run(bed, 3)
    assert results.favorable_agreement >= results.threshold  # favorable still clears
    assert results.control_agreements["shuffled-seed"] >= results.threshold  # but the control leaked
    receipt = runner.mint(results)
    assert receipt.verdict == VERDICT_NULL
    assert "shuffled-seed" not in receipt.controls_cleared
    assert receipt.is_confirmation is False


def test_every_leaking_control_blocks_mechanics_ok() -> None:
    bed = TraceStabilityBed()
    for control in REQUIRED_CONTROLS:
        runner = TraceStabilityRunner(leaked_controls=frozenset({control}))
        receipt = runner.mint(runner.run(bed, 5))
        assert receipt.verdict == VERDICT_NULL, control


def test_receipt_digest_is_stable() -> None:
    receipt = TraceStabilityRunner().mint(TraceStabilityRunner().run(TraceStabilityBed(), 11))
    assert receipt.digest() == receipt.digest()
    assert len(receipt.digest()) == 64


def test_receipt_is_never_a_confirmation() -> None:
    bed = TraceStabilityBed()
    runner = TraceStabilityRunner()
    leaking = TraceStabilityRunner(leaked_controls=frozenset({"permuted-trace"}))
    null_bed = NullOnlyBed(bed)
    assert runner.mint(runner.run(bed, 1)).is_confirmation is False
    assert runner.mint(runner.run(null_bed, 1)).is_confirmation is False
    assert leaking.mint(leaking.run(bed, 1)).is_confirmation is False
