
from __future__ import annotations

from mop.ladder.ladder_contracts import (
    KIND_DEMONSTRATION,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MechanismRunner,
)
from mop.mechanisms.messaging_repair_bed import (
    MessagingRepairBed,
    build_default_bed,
)
from mop.mechanisms.messaging_repair_impl import (
    CONTROL_POLICIES,
    DECLARED_CONTROLS,
    MECHANISM_POLICY,
    MatchedCost,
    Regime,
)
from mop.mechanisms.messaging_repair_runner import (
    MessagingRepairRunner,
    RunResult,
    build_default_runner,
)


def _run(seed: int = 7) -> tuple[MessagingRepairRunner, MessagingRepairBed, RunResult]:
    runner = build_default_runner()
    bed = build_default_bed()
    return runner, bed, runner.run(bed, seed)


def test_bed_and_runner_satisfy_the_ladder_protocols() -> None:
    assert isinstance(build_default_bed(), Bed)
    assert isinstance(build_default_runner(), MechanismRunner)


def test_matched_cost_is_non_vacuous() -> None:
    budget = build_default_bed().matched_cost()
    assert budget.params > 0 and budget.flops > 0 and budget.wall_ns > 0 and budget.seeds > 0


def test_declared_controls_are_exactly_the_metric_controls() -> None:
    assert build_default_bed().controls() == DECLARED_CONTROLS
    assert DECLARED_CONTROLS == (
        "no-message",
        "broadcast-all",
        "stale-message",
        "no-verify",
        "always-verify",
        "majority-vote",
    )


def test_run_is_deterministic_for_a_fixed_seed() -> None:
    runner, bed, first = _run(11)
    second = runner.run(bed, 11)
    assert first == second
    assert runner.mint(first) == runner.mint(second)


def test_evidence_digest_is_stable() -> None:
    runner, bed, _ = _run(3)
    digest_a = runner.mint(runner.run(bed, 3)).evidence_digest
    digest_b = runner.mint(runner.run(bed, 3)).evidence_digest
    assert digest_a == digest_b
    assert len(digest_a) == 64


def test_favorable_regime_earns_mechanics_ok_beating_all_controls() -> None:
    runner, _, results = _run(7)
    receipt = runner.mint(results)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert results.favorable.mechanism_beats_all is True
    assert set(receipt.controls_cleared) == set(DECLARED_CONTROLS)
    for name, margin in results.favorable.margins:
        assert margin > 0, f"mechanism failed to beat control {name} on the favorable regime"


def test_mechanism_strictly_beats_each_control_at_matched_cost() -> None:
    bed = build_default_bed()
    regime = bed.favorable_regime(0)
    budget = MatchedCost(action_budget=bed.action_budget)
    mechanism = MECHANISM_POLICY(regime, budget).improvement
    for name, policy in CONTROL_POLICIES.items():
        control = policy(regime, budget).improvement
        assert mechanism > control, f"mechanism did not strictly beat {name}"


def test_mechanics_ok_receipt_is_not_a_confirmation() -> None:
    runner, _, results = _run(7)
    receipt = runner.mint(results)
    assert receipt.is_confirmation is False
    assert receipt.requirement_id == "s3.messaging_repair"
    assert receipt.stage == 3


def test_null_regime_holds_the_prior_null() -> None:
    runner, _, results = _run(7)
    assert results.null.mechanism_beats_all is False
    assert results.null.mechanism_improvement == 0


class _NullFavorableBed:

    mechanism_id = "messaging_repair"

    def __init__(self, base: MessagingRepairBed) -> None:
        self._base = base

    def controls(self) -> tuple[str, ...]:
        return self._base.controls()

    def matched_cost(self) -> object:
        return self._base.matched_cost()

    def null_regime(self, seed: int) -> Regime:
        return self._base.null_regime(seed)

    def favorable_regime(self, seed: int) -> Regime:
        return self._base.null_regime(seed)


def test_a_no_signal_favorable_regime_yields_a_null_verdict() -> None:
    runner = build_default_runner()
    bed = _NullFavorableBed(build_default_bed())
    receipt = runner.mint(runner.run(bed, 7))
    assert receipt.verdict == VERDICT_NULL
    assert receipt.controls_cleared == ()
    assert receipt.is_confirmation is False


def test_a_leaking_control_blocks_mechanics_ok() -> None:
    bed = build_default_bed()
    leaking = dict(CONTROL_POLICIES)
    leaking["always-verify"] = MECHANISM_POLICY  # a control that secretly runs the mechanism
    runner = MessagingRepairRunner(control_policies=leaking)
    results = runner.run(bed, 7)
    receipt = runner.mint(results)
    assert results.favorable.mechanism_beats_all is False
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_mechanics_ok_holds_across_several_seeds() -> None:
    runner = build_default_runner()
    bed = build_default_bed()
    for seed in (0, 1, 2, 5, 13, 21):
        receipt = runner.mint(runner.run(bed, seed))
        assert receipt.verdict == VERDICT_MECHANICS_OK
