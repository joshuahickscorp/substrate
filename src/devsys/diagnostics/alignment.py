"""AlignmentSuite (WP-02): a THIN aggregator over the existing geometry battery (linear/kernel CKA,
RSA) and seed_consistency (cross-seed CKA), plus a permutation p-value so an alignment score comes with
its own null. This module does NOT reimplement CKA; it calls `diagnostics.geometry` and reports.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .geometry import kernel_cka, linear_cka, rsa
from .seed_consistency import cross_seed_cka


def permutation_pvalue(observed: float, null_samples: Sequence[float], greater: bool = True) -> float:
    """Permutation p-value with the add-one correction: (1 + #null >= observed) / (1 + n). `greater`
    False flips the tail. Never returns exactly 0 (honest at small n)."""
    n = len(null_samples)
    if greater:
        hits = sum(1 for s in null_samples if s >= observed)
    else:
        hits = sum(1 for s in null_samples if s <= observed)
    return (1 + hits) / (1 + n)


def alignment_null(x: torch.Tensor, y: torch.Tensor, n_permutations: int = 200, seed: int = 0) -> list[float]:
    """Null distribution for linear CKA under row-shuffled y (breaks the x<->y correspondence while
    preserving both geometries). Returns n_permutations CKA samples."""
    g = torch.Generator().manual_seed(seed)
    out: list[float] = []
    for _ in range(n_permutations):
        perm = torch.randperm(y.shape[0], generator=g)
        out.append(linear_cka(x, y[perm]))
    return out


def alignment_suite(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """The standard alignment report between two representations [N, D1], [N, D2]: linear CKA, kernel
    CKA, RSA, and a permutation p-value for linear CKA against the row-shuffle null. Keys:
    linear_cka, kernel_cka, rsa, null_mean, p_value, n_permutations."""
    lc = linear_cka(x, y)
    null = alignment_null(x, y, n_permutations=n_permutations, seed=seed)
    return {
        "linear_cka": round(lc, 4),
        "kernel_cka": round(kernel_cka(x, y), 4),
        "rsa": round(rsa(x, y), 4),
        "null_mean": round(sum(null) / max(len(null), 1), 4),
        "p_value": round(permutation_pvalue(lc, null), 4),
        "n_permutations": int(n_permutations),
    }


def cross_seed_alignment(reps: list[torch.Tensor]) -> dict:
    """Delegates to seed_consistency.cross_seed_cka: pairwise CKA across seeds of the SAME pipeline
    (a stability report, not a substrate comparison)."""
    return cross_seed_cka(reps)
