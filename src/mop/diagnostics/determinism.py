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
    fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0, tol: float = 1e-4
) -> VarianceReport:
    """Reproducibility check against an ABSOLUTE tolerance (default fp32-friendly 1e-4), so
    the assertion can actually fail. Never asserts bit-exactness on Metal: pass a larger tol
    for ops with known Metal nondeterminism, sized from determinism_loop on this machine."""
    rep = determinism_loop(fn, runs, seed)
    assert rep.max_abs <= tol, f"run-to-run spread {rep.max_abs} exceeds tolerance {tol}"
    return rep
