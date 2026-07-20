
from __future__ import annotations

import importlib
from typing import cast

from .ladder_contracts import Bed, MechanismRunner, RunReceipt
from .stage3_registry import Stage3RegistryError, _discover

SUCCESSOR_EPOCHS: tuple[str, ...] = (
    "calibrated_uncertainty",
    "reducible_novelty",
    "stability_plasticity_r2",
)


class SuccessorRegistryError(RuntimeError):
    pass


def build_pair(epoch: str) -> tuple[Bed, MechanismRunner]:

    if epoch not in SUCCESSOR_EPOCHS:
        raise SuccessorRegistryError(f"unknown successor Stage 3 epoch {epoch!r}")
    bed_module = importlib.import_module(f"mop.mechanisms.{epoch}_bed")
    runner_module = importlib.import_module(f"mop.mechanisms.{epoch}_runner")
    try:
        bed = cast(Bed, _discover(bed_module, "Bed"))
        runner = cast(MechanismRunner, _discover(runner_module, "Runner"))
    except Stage3RegistryError as exc:
        raise SuccessorRegistryError(str(exc)) from exc
    return bed, runner


def run_demonstration(epoch: str, seed: int) -> RunReceipt:

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise SuccessorRegistryError("seed must be a nonnegative integer")
    bed, runner = build_pair(epoch)
    results = runner.run(bed, seed)
    receipt = runner.mint(results)
    if receipt.is_confirmation:
        raise SuccessorRegistryError(
            f"runner for {epoch!r} minted a confirmation from a toy bed, which is forbidden"
        )
    return receipt
