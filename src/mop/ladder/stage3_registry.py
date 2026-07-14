"""Registry that drives every Stage 3 mechanism runner uniformly.

Each epoch ships a bed and a runner that satisfy the ladder_contracts Bed and MechanismRunner protocols.
This module discovers and constructs them (via a build_default helper where present, else the single
public class named <Camel>Bed or <Camel>Runner) so the campaign can run any epoch by name and collect a
RunReceipt. Runners mint mechanics-demonstration receipts only; this module never turns one into a
confirmation.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType
from typing import Any, cast

from .ladder_contracts import Bed, MechanismRunner, RunReceipt

STAGE3_EPOCHS: tuple[str, ...] = (
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


class Stage3RegistryError(RuntimeError):
    """Raised when an epoch is unknown or its bed or runner cannot be discovered."""


def _discover(module: ModuleType, suffix: str) -> Any:
    builder = getattr(module, f"build_default_{suffix.lower()}", None)
    if builder is not None:
        return builder()
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if (
            name.endswith(suffix)
            and not name.endswith(("Error", "Refusal"))
            and cls.__module__ == module.__name__
        ):
            return cls()
    raise Stage3RegistryError(f"no public {suffix} found in {module.__name__}")


def build_pair(epoch: str) -> tuple[Bed, MechanismRunner]:
    """Construct the (bed, runner) pair for one epoch. Fails closed on an unknown epoch."""

    if epoch not in STAGE3_EPOCHS:
        raise Stage3RegistryError(f"unknown Stage 3 epoch {epoch!r}")
    bed_module = importlib.import_module(f"mop.mechanisms.{epoch}_bed")
    runner_module = importlib.import_module(f"mop.mechanisms.{epoch}_runner")
    bed = cast(Bed, _discover(bed_module, "Bed"))
    runner = cast(MechanismRunner, _discover(runner_module, "Runner"))
    return bed, runner


def run_demonstration(epoch: str, seed: int) -> RunReceipt:
    """Run one epoch on one seed and return its (demonstration) RunReceipt."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Stage3RegistryError("seed must be a nonnegative integer")
    bed, runner = build_pair(epoch)
    results = runner.run(bed, seed)
    receipt = runner.mint(results)
    if receipt.is_confirmation:
        raise Stage3RegistryError(
            f"runner for {epoch!r} minted a confirmation from a toy bed, which is forbidden"
        )
    return receipt
