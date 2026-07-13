"""Tests for the intervention-simulation MechanismRunner.

These tests pin the honest two-regime demonstration: the run is deterministic per seed with a
stable digest, the null regime forces an exact tie for every sub-mechanism so the verdict is null,
the favorable regime lets every sub-mechanism beat its matched control so the verdict is
mechanics-ok, a single tie anywhere on the favorable regime blocks mechanics-ok, and the minted
receipt is a mechanics demonstration that can never be a scientific confirmation. No capability is
claimed anywhere.
"""

from __future__ import annotations

import re

import pytest

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
    RunReceipt,
)
from mop.mechanisms.intervention_simulation_bed import (
    DECLARED_CONTROLS,
    MECHANISM_ID,
    REQUIREMENT_ID,
    InterventionSimulationBed,
)
from mop.mechanisms.intervention_simulation_runner import (
    MARGIN_REQUIRED,
    InterventionSimulationRunner,
    RegimeEvaluation,
    RunResults,
    SubComparison,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_bed_and_runner_satisfy_the_ladder_protocols() -> None:
    assert isinstance(InterventionSimulationBed(), Bed)
    assert isinstance(InterventionSimulationRunner(), MechanismRunner)


def test_run_is_deterministic_per_seed() -> None:
    bed = InterventionSimulationBed()
    runner = InterventionSimulationRunner()
    first = runner.run(bed, 7)
    second = runner.run(bed, 7)
    assert first.digest() == second.digest()
    assert _SHA256_RE.match(first.digest())


def test_receipt_digest_and_evidence_digest_are_stable() -> None:
    bed = InterventionSimulationBed()
    runner = InterventionSimulationRunner()
    first = runner.mint(runner.run(bed, 5))
    second = runner.mint(runner.run(bed, 5))
    assert first.digest() == second.digest()
    assert first.evidence_digest == second.evidence_digest
    assert _SHA256_RE.match(first.evidence_digest)


def test_null_regime_forces_an_exact_tie_for_every_sub_mechanism() -> None:
    bed = InterventionSimulationBed()
    runner = InterventionSimulationRunner()
    results = runner.run(bed, 3)
    assert results.null.none_beat is True
    for comparison in results.null.comparisons:
        assert comparison.margin == pytest.approx(0.0)
        assert comparison.beats is False


def test_favorable_regime_mints_mechanics_ok() -> None:
    bed = InterventionSimulationBed()
    runner = InterventionSimulationRunner()
    results = runner.run(bed, 0)
    assert results.favorable.all_beat is True
    receipt = runner.mint(results)
    assert isinstance(receipt, RunReceipt)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.mechanism_id == MECHANISM_ID
    assert receipt.requirement_id == REQUIREMENT_ID
    assert set(receipt.controls_cleared) == set(DECLARED_CONTROLS)
    assert receipt.is_confirmation is False


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 11, 41])
def test_mechanics_ok_holds_across_seeds(seed: int) -> None:
    bed = InterventionSimulationBed()
    runner = InterventionSimulationRunner()
    receipt = runner.mint(runner.run(bed, seed))
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.is_confirmation is False


def test_null_world_bed_mints_null() -> None:
    # A degenerate world with no causal structure and no reducible novelty anywhere: the favorable
    # regime carries nothing to exploit, so no sub-mechanism beats its control and the verdict is null.
    bed = InterventionSimulationBed(structured=False)
    runner = InterventionSimulationRunner()
    results = runner.run(bed, 0)
    assert results.favorable.all_beat is False
    receipt = runner.mint(results)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_a_single_tie_on_the_favorable_regime_blocks_mechanics_ok() -> None:
    runner = InterventionSimulationRunner()
    favorable = RegimeEvaluation(
        "favorable",
        (
            SubComparison("first", "arm", "control-a", 1.0, 0.0, MARGIN_REQUIRED),
            SubComparison("second", "arm", "control-b", 1.0, 0.0, MARGIN_REQUIRED),
            SubComparison("tied", "arm", "control-c", 0.5, 0.5, MARGIN_REQUIRED),
        ),
    )
    null = RegimeEvaluation(
        "null",
        (SubComparison("first", "arm", "control-a", 0.5, 0.5, MARGIN_REQUIRED),),
    )
    results = RunResults(mechanism_id=MECHANISM_ID, seed=0, favorable=favorable, null=null)
    receipt = runner.mint(results)
    assert receipt.verdict != VERDICT_MECHANICS_OK
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_a_leaky_null_regime_also_blocks_mechanics_ok() -> None:
    # Even with the favorable regime clean, if any sub-mechanism beats its control on the null
    # regime the null does not hold, so mechanics-ok is withheld.
    runner = InterventionSimulationRunner()
    favorable = RegimeEvaluation(
        "favorable",
        (SubComparison("first", "arm", "control-a", 1.0, 0.0, MARGIN_REQUIRED),),
    )
    null = RegimeEvaluation(
        "null",
        (SubComparison("first", "arm", "control-a", 1.0, 0.0, MARGIN_REQUIRED),),
    )
    results = RunResults(mechanism_id=MECHANISM_ID, seed=0, favorable=favorable, null=null)
    receipt = runner.mint(results)
    assert receipt.verdict == VERDICT_NULL
