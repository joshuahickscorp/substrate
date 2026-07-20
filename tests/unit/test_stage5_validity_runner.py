
from __future__ import annotations

import pytest

from mop.ladder.ladder_contracts import KIND_DEMONSTRATION, VERDICT_MECHANICS_OK, VERDICT_NULL
from mop.mechanisms.stage5_validity_bed import (
    LEAK_CONTROLS,
    VALIDITY_AXES,
    build_bed,
    build_null_bed,
)
from mop.mechanisms.stage5_validity_runner import REQUIREMENT_ID, STAGE_INDEX, run

SEED = 7


def test_favorable_regime_mints_mechanics_ok() -> None:
    receipt = run(build_bed(SEED), SEED)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.requirement_id == REQUIREMENT_ID
    assert receipt.stage == STAGE_INDEX
    assert set(receipt.controls_cleared) == set(LEAK_CONTROLS)
    assert receipt.is_confirmation is False


def test_null_regime_is_null() -> None:
    receipt = run(build_null_bed(SEED), SEED)
    assert receipt.verdict == VERDICT_NULL
    assert receipt.is_confirmation is False


def test_dropping_one_axis_fails_closed() -> None:
    receipt = run(build_bed(SEED).with_failing_axis("lesions"), SEED)
    assert receipt.verdict == VERDICT_NULL
    assert "lesions" in receipt.detail["failing_axes"]


def test_reproducing_leak_control_is_null() -> None:
    receipt = run(build_bed(SEED).with_reproducing_control("seed-reuse"), SEED)
    assert receipt.verdict == VERDICT_NULL
    assert "seed-reuse" in receipt.detail["reproducing_controls"]
    assert "seed-reuse" not in receipt.controls_cleared


def test_declared_not_equal_measured_efficiency_is_null() -> None:
    receipt = run(build_bed(SEED).with_efficiency_mismatch("wall_time_s"), SEED)
    assert receipt.verdict == VERDICT_NULL
    assert "wall_time_s" in receipt.detail["mismatching_resources"]


def test_run_is_deterministic() -> None:
    first = run(build_bed(SEED), SEED)
    second = run(build_bed(SEED), SEED)
    assert first.payload() == second.payload()
    assert first.digest() == second.digest()


def test_digest_is_stable_and_sha256() -> None:
    receipt = run(build_bed(SEED), SEED)
    assert receipt.digest() == run(build_bed(SEED), SEED).digest()
    assert len(receipt.evidence_digest) == 64
    assert len(receipt.digest()) == 64


def test_is_confirmation_always_false_across_regimes() -> None:
    beds = [
        build_bed(SEED),
        build_null_bed(SEED),
        build_bed(SEED).with_failing_axis("fresh-session"),
        build_bed(SEED).with_reproducing_control("same-session-leak"),
        build_bed(SEED).with_efficiency_mismatch("flops"),
    ]
    for bed in beds:
        assert run(bed, SEED).is_confirmation is False


def test_every_axis_can_break_the_run() -> None:
    for axis in VALIDITY_AXES:
        receipt = run(build_bed(SEED).with_failing_axis(axis), SEED)
        assert receipt.verdict == VERDICT_NULL


def test_every_leak_control_can_break_the_run() -> None:
    for control in LEAK_CONTROLS:
        receipt = run(build_bed(SEED).with_reproducing_control(control), SEED)
        assert receipt.verdict == VERDICT_NULL


def test_negative_seed_is_refused() -> None:
    with pytest.raises(ValueError):
        run(build_bed(), -1)
