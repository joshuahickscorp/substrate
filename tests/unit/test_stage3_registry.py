from __future__ import annotations

import pytest

from mop.ladder.ladder_contracts import KIND_DEMONSTRATION
from mop.ladder.stage3_registry import (
    STAGE3_EPOCHS,
    Stage3RegistryError,
    build_pair,
    run_demonstration,
)


def test_registry_lists_active_epochs() -> None:
    assert len(STAGE3_EPOCHS) == 7
    assert len(set(STAGE3_EPOCHS)) == 7


@pytest.mark.parametrize("epoch", STAGE3_EPOCHS)
def test_every_epoch_runs_and_is_an_honest_demonstration(epoch: str) -> None:
    receipt = run_demonstration(epoch, 0)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.is_confirmation is False
    assert receipt.mechanism_id == epoch
    assert receipt.verdict in ("null", "pending", "mechanics-ok")


@pytest.mark.parametrize("epoch", STAGE3_EPOCHS)
def test_runs_are_deterministic_per_seed(epoch: str) -> None:
    assert run_demonstration(epoch, 3).digest() == run_demonstration(epoch, 3).digest()


@pytest.mark.parametrize("epoch", STAGE3_EPOCHS)
def test_build_pair_returns_matched_mechanism_ids(epoch: str) -> None:
    bed, runner = build_pair(epoch)
    assert bed.mechanism_id == epoch
    assert runner.mechanism_id == epoch


def test_unknown_epoch_fails_closed() -> None:
    with pytest.raises(Stage3RegistryError):
        run_demonstration("not_an_epoch", 0)


def test_negative_seed_fails_closed() -> None:
    with pytest.raises(Stage3RegistryError):
        run_demonstration("trace_stability", -1)
