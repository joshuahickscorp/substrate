
from __future__ import annotations

import pytest

from mop.ladder import stage3_successor_registry as registry
from mop.ladder.ladder_contracts import KIND_DEMONSTRATION
from mop.ladder.stage3_registry import STAGE3_EPOCHS
from mop.ladder.stage3_successor_registry import (
    SUCCESSOR_EPOCHS,
    SuccessorRegistryError,
    build_pair,
    run_demonstration,
)


def test_registry_lists_three_new_epochs_disjoint_from_stage3() -> None:
    assert SUCCESSOR_EPOCHS == ("calibrated_uncertainty", "reducible_novelty", "stability_plasticity_r2")
    assert not set(SUCCESSOR_EPOCHS) & set(STAGE3_EPOCHS)


@pytest.mark.parametrize("epoch", SUCCESSOR_EPOCHS)
def test_every_epoch_runs_and_is_an_honest_demonstration(epoch: str) -> None:
    receipt = run_demonstration(epoch, 0)
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.is_confirmation is False
    assert receipt.mechanism_id == epoch
    assert receipt.verdict in ("null", "pending", "mechanics-ok")


@pytest.mark.parametrize("epoch", SUCCESSOR_EPOCHS)
def test_runs_are_deterministic_per_seed(epoch: str) -> None:
    assert run_demonstration(epoch, 3).digest() == run_demonstration(epoch, 3).digest()


@pytest.mark.parametrize("epoch", SUCCESSOR_EPOCHS)
def test_build_pair_discovers_matched_mechanism_ids(epoch: str) -> None:
    bed, runner = build_pair(epoch)
    assert bed.mechanism_id == epoch
    assert runner.mechanism_id == epoch


def test_unknown_epoch_fails_closed() -> None:
    with pytest.raises(SuccessorRegistryError):
        build_pair("not_an_epoch")
    with pytest.raises(SuccessorRegistryError):
        run_demonstration("stability_plasticity", 0)


def test_negative_seed_fails_closed() -> None:
    with pytest.raises(SuccessorRegistryError):
        run_demonstration("calibrated_uncertainty", -1)


def test_confirmation_from_a_toy_bed_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReceipt:
        is_confirmation = True

    class FakeRunner:
        def run(self, bed: object, seed: int) -> tuple[str, ...]:
            return ("fake",)

        def mint(self, results: object) -> FakeReceipt:
            return FakeReceipt()

    monkeypatch.setattr(registry, "build_pair", lambda epoch: (object(), FakeRunner()))
    with pytest.raises(SuccessorRegistryError, match="confirmation"):
        registry.run_demonstration("calibrated_uncertainty", 0)
