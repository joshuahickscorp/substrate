"""Determinism sanity loop. Apple-Silicon Metal at temperature zero is only ~50% byte-
identical, so we MEASURE run-to-run spread instead of assuming bit-exactness, and the suite
sizes its tolerances from the result. Run this before trusting any cross-condition delta.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..seeding import VarianceReport, variance_of


def determinism_loop(fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0) -> VarianceReport:
    """Run a seeded computation several times; return the spread report (max/mean abs diff,
    bit_identical flag, and a suggested tolerance)."""
    return variance_of(fn, runs=runs, seed=seed)


def assert_reproducible(
    fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0, floor: float = 1e-6
) -> VarianceReport:
    """Soft reproducibility check: passes if spread is within a tolerance derived from the
    measured spread itself (never asserts bit-exactness on Metal)."""
    rep = determinism_loop(fn, runs, seed)
    assert rep.max_abs <= rep.tol(floor), (
        f"run-to-run spread {rep.max_abs} exceeds tolerance {rep.tol(floor)}"
    )
    return rep
