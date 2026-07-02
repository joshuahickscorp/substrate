"""Matched-capacity constructor (WP-02, H-CAPMATCH): solve the hidden width (or any monotone integer
knob) so a module lands within 2% of a target parameter count, plus the fixed-total-params bandwidth
sweep (vary slot count, resolve per-slot width so TOTAL params stay constant). This is the helper the
honesty doctrine names: capacity matching is done by construction here, never by eyeballing.

All searches are exact integer binary search over `param_count(make(width))`, which must be
non-decreasing in width (true for every MLP/head/slot module in the shell).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

from torch import nn

from ..diagnostics.compute import param_count


def width_for_param_count(
    make: Callable[[int], nn.Module],
    target_params: int,
    *,
    tol: float = 0.02,
    lo: int = 1,
    hi: int = 1 << 16,
) -> tuple[int, int]:
    """Binary-search the integer width w in [lo, hi] minimizing |param_count(make(w)) - target|.
    Returns (width, achieved_params). Raises ValueError if the best achievable count misses the
    target by more than `tol` (relative), so a silent capacity mismatch cannot enter an experiment."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    a, b = int(lo), int(hi)
    while a < b:
        mid = (a + b) // 2
        if param_count(make(mid)) < target_params:
            a = mid + 1
        else:
            b = mid
    best_w, best_p = a, param_count(make(a))
    for w in (a - 1, a + 1):
        if lo <= w <= hi:
            p = param_count(make(w))
            if abs(p - target_params) < abs(best_p - target_params):
                best_w, best_p = w, p
    if abs(best_p - target_params) > tol * target_params:
        raise ValueError(
            f"capmatch failed: best width {best_w} gives {best_p} params, "
            f"target {target_params} (tol {tol:.0%})"
        )
    return best_w, best_p


def matched_capacity(
    reference: nn.Module,
    make: Callable[[int], nn.Module],
    *,
    tol: float = 0.02,
) -> nn.Module:
    """Construct make(w) with the width solved so its param count matches `reference` within tol.
    The one-call form of the doctrine: 'matched capacity via the capmatch helper'."""
    w, _ = width_for_param_count(make, param_count(reference), tol=tol)
    return make(w)


def fixed_total_params_sweep(
    make: Callable[[int, int], nn.Module],
    total_params: int,
    slots: Sequence[int],
    *,
    tol: float = 0.02,
) -> dict[int, dict]:
    """The bandwidth-sweep variant (WS4): for each slot count n in `slots`, solve the per-slot width w
    so param_count(make(n, w)) hits `total_params` within tol. Returns n -> {"width": w, "params": p}.
    Sweeping slots at FIXED total params isolates bandwidth allocation from raw capacity."""
    out: dict[int, dict] = {}
    for n in slots:
        w, p = width_for_param_count(functools.partial(make, int(n)), total_params, tol=tol)
        out[int(n)] = {"width": w, "params": p}
    return out
