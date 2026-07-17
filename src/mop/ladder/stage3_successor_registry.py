"""Registry that drives the three successor Stage 3 mechanism runners uniformly.

The successor epochs (calibrated_uncertainty, reducible_novelty, stability_plasticity_r2) ship beds and
runners that satisfy the ladder_contracts Bed and MechanismRunner protocols, exactly like the original
Stage 3 epochs. This module discovers and constructs them with the sealed stage3_registry discovery
helper (read-only import; the sealed registry itself is never modified) so successor campaigns can run
any new epoch by name and collect a RunReceipt. Runners mint mechanics-demonstration receipts only;
this module never turns one into a confirmation.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

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
    """Raised when an epoch is unknown or its bed or runner cannot be discovered."""


def build_pair(epoch: str) -> tuple[Bed, MechanismRunner]:
    """Construct the (bed, runner) pair for one successor epoch. Fails closed on an unknown epoch."""

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
    """Run one successor epoch on one seed and return its (demonstration) RunReceipt."""

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
