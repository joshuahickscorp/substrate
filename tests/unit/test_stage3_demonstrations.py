
from __future__ import annotations

import re

import pytest

from mop.ladder.stage3_demonstrations import (
    CLAIM_SCOPE,
    SCHEMA,
    STAGE3_EPOCHS,
    DemonstrationResult,
    Stage3DemonstrationError,
    coverage,
    run_demonstration,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERDICT_BEARING = ("event_formation", "stability_plasticity", "integrated_escs")


def test_stage3_epochs_are_the_nine_in_ladder_order() -> None:
    assert STAGE3_EPOCHS == (
        "trace_stability",
        "niche_dispatch",
        "event_formation",
        "messaging_repair",
        "intervention_simulation",
        "memory_organization",
        "stability_plasticity",
        "construction_search",
        "integrated_escs",
    )


@pytest.mark.parametrize("epoch", STAGE3_EPOCHS)
def test_run_demonstration_is_deterministic_per_seed(epoch: str) -> None:
    first = run_demonstration(epoch, 7)
    second = run_demonstration(epoch, 7)
    assert isinstance(first, DemonstrationResult)
    assert first.digest() == second.digest()
    assert _SHA256_RE.match(first.digest())
    assert first.schema == SCHEMA
    assert first.claim_scope == CLAIM_SCOPE
    assert first.epoch == epoch
    assert first.seed == 7


@pytest.mark.parametrize("epoch", STAGE3_EPOCHS)
def test_honest_default_null_holds_and_controls_complete(epoch: str) -> None:
    result = run_demonstration(epoch, 3)
    assert result.null_holds is True
    assert result.controls_ok is True
    assert result.verdict_status  # a non empty null/split/no-advantage token


@pytest.mark.parametrize("epoch", _VERDICT_BEARING)
def test_verdict_bearing_epochs_report_null_holds(epoch: str) -> None:
    result = run_demonstration(epoch, 0)
    assert result.null_holds is True


def test_distinct_seeds_yield_distinct_digests() -> None:
    assert run_demonstration("trace_stability", 1).digest() != run_demonstration(
        "trace_stability", 2
    ).digest()


def test_integrated_escs_reports_the_no_advantage_token() -> None:
    result = run_demonstration("integrated_escs", 0)
    assert result.verdict_status == "no-integrated-advantage-at-matched-cost"


def test_stability_plasticity_reports_the_split_token() -> None:
    result = run_demonstration("stability_plasticity", 0)
    assert result.verdict_status == "p6-stability-plasticity-split"


def test_unknown_epoch_fails_closed() -> None:
    with pytest.raises(Stage3DemonstrationError):
        run_demonstration("not_an_epoch", 0)


def test_negative_seed_fails_closed() -> None:
    with pytest.raises(Stage3DemonstrationError):
        run_demonstration("trace_stability", -1)


def test_coverage_lists_all_nine_epochs() -> None:
    cov = coverage()
    assert set(cov) == set(STAGE3_EPOCHS)
    assert len(cov) == 9
    for entries in cov.values():
        assert entries
        assert all(isinstance(item, str) and item for item in entries)
