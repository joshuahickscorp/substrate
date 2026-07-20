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
    w, _ = width_for_param_count(make, param_count(reference), tol=tol)
    return make(w)


def fixed_total_params_sweep(
    make: Callable[[int, int], nn.Module],
    total_params: int,
    slots: Sequence[int],
    *,
    tol: float = 0.02,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for n in slots:
        w, p = width_for_param_count(functools.partial(make, int(n)), total_params, tol=tol)
        out[int(n)] = {"width": w, "params": p}
    return out
