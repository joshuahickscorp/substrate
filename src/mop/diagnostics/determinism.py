from __future__ import annotations

from collections.abc import Callable

import torch

from ..seeding import VarianceReport, variance_of


def determinism_loop(fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0) -> VarianceReport:
    return variance_of(fn, runs=runs, seed=seed)


def assert_reproducible(
    fn: Callable[[], torch.Tensor], runs: int = 5, seed: int = 0, tol: float = 1e-4
) -> VarianceReport:
    rep = determinism_loop(fn, runs, seed)
    assert rep.max_abs <= tol, f"run-to-run spread {rep.max_abs} exceeds tolerance {tol}"
    return rep
