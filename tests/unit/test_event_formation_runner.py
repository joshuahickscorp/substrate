
from __future__ import annotations

from mop.ladder.ladder_contracts import KIND_DEMONSTRATION, Bed, MechanismRunner
from mop.mechanisms.event_formation_bed import EventFormationBed
from mop.mechanisms.event_formation_impl import (
    EVENT_FORMER_ID,
    FAVORABLE_REGIME,
    NULL_REGIME,
    RegimeMeasurement,
)
from mop.mechanisms.event_formation_runner import EventFormationRunner, EventFormationRunResults

SEED = 7
EXPECTED_CONTROLS = ("wrong-time", "wrong-event", "appearance-only", "stateless-delayed-trigger")


def _runner_and_results(seed: int = SEED) -> tuple[EventFormationRunner, EventFormationRunResults]:
    bed = EventFormationBed()
    runner = EventFormationRunner()
    return runner, runner.run(bed, seed)


def test_bed_and_runner_satisfy_protocols() -> None:
    assert isinstance(EventFormationBed(), Bed)
    assert isinstance(EventFormationRunner(), MechanismRunner)


def test_run_is_deterministic() -> None:
    runner, first = _runner_and_results()
    _, second = _runner_and_results()
    assert first.payload() == second.payload()
    assert runner.mint(first).digest() == runner.mint(second).digest()


def test_null_regime_yields_null() -> None:
    runner, results = _runner_and_results()
    receipt = runner.mint_regime(results, NULL_REGIME)
    assert receipt.verdict == "null"
    assert receipt.controls_cleared == ()
    assert receipt.is_confirmation is False


def test_favorable_regime_yields_mechanics_ok() -> None:
    runner, results = _runner_and_results()
    receipt = runner.mint_regime(results, FAVORABLE_REGIME)
    assert receipt.verdict == "mechanics-ok"
    assert receipt.requirement_id == "s3.event_formation"
    assert receipt.kind == KIND_DEMONSTRATION
    assert set(receipt.controls_cleared) == set(EXPECTED_CONTROLS)


def test_canonical_mint_requires_favorable_and_null() -> None:
    runner, results = _runner_and_results()
    receipt = runner.mint(results)
    assert receipt.verdict == "mechanics-ok"
    assert receipt.detail["favorable_claims_useful"] is True
    assert receipt.detail["null_holds"] is True
    assert receipt.is_confirmation is False


def test_utility_leak_by_control_blocks_mechanics_ok() -> None:
    runner, results = _runner_and_results()
    favorable = results.favorable
    leaked_utility = dict(favorable.utility)
    leaked_utility["appearance-only"] = favorable.utility[EVENT_FORMER_ID]
    leaked_favorable = RegimeMeasurement(
        regime=FAVORABLE_REGIME,
        seed=favorable.seed,
        utility=leaked_utility,
        charged_compute=dict(favorable.charged_compute),
        episode_digest=favorable.episode_digest,
    )
    leaked = EventFormationRunResults(seed=results.seed, null=results.null, favorable=leaked_favorable)
    assert runner.mint_regime(leaked, FAVORABLE_REGIME).verdict != "mechanics-ok"
    assert runner.mint(leaked).verdict == "null"


def test_evidence_digest_is_stable_and_canonical() -> None:
    runner, results = _runner_and_results()
    receipt = runner.mint(results)
    assert receipt.evidence_digest == runner.mint(results).evidence_digest
    assert len(receipt.evidence_digest) == 64
    assert receipt.digest() == runner.mint(results).digest()


def test_receipts_are_never_confirmations() -> None:
    runner, results = _runner_and_results()
    receipts = (
        runner.mint(results),
        runner.mint_regime(results, NULL_REGIME),
        runner.mint_regime(results, FAVORABLE_REGIME),
    )
    for receipt in receipts:
        assert receipt.kind == KIND_DEMONSTRATION
        assert receipt.is_confirmation is False
