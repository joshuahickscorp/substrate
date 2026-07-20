
from __future__ import annotations

import math

import torch


def _contraction_factor(norms: list[float]) -> float:
    pts = [(i, math.log(n)) for i, n in enumerate(norms) if n > 1e-12]
    if len(pts) < 2:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mx) ** 2 for x in xs) or 1e-12
    return num / den


def convergence_report(refiner, z: torch.Tensor, steps: int = 64) -> dict:
    _, norms = refiner.unroll(z, steps)
    cf = _contraction_factor(norms)
    resid = norms[-1] if norms else 0.0
    first = norms[0] if norms else 1.0
    converged = bool(resid < 0.05 * max(first, 1e-9) and cf < 0)
    if converged:
        cls = "converges"
    elif cf >= -1e-3 and resid > 0.2 * max(first, 1e-9):
        cls = "limit_cycle" if resid <= first * 1.5 else "drifts"
    else:
        cls = "drifts"
    return {
        "update_norms": [round(n, 6) for n in norms],
        "contraction_factor": round(cf, 6),
        "fixed_point_residual": round(resid, 6),
        "initial_update_norm": round(first, 6),
        "converged": converged,
        "classification": cls,
        "n_steps": int(steps),
    }


def basin_stability(refiner, z: torch.Tensor, eps: float = 0.1, steps: int = 64, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(z.shape, generator=g) * eps
    z0, _ = refiner.unroll(z, steps)
    z1, _ = refiner.unroll(z + noise, steps)
    pert = float(noise.norm(dim=-1).mean())
    fpd = float((z0 - z1).norm(dim=-1).mean())
    ratio = fpd / pert if pert > 0 else float("inf")
    return {
        "perturbation": round(pert, 6),
        "fixed_point_distance": round(fpd, 6),
        "contraction_ratio": round(ratio, 6),
        "stable": bool(ratio < 1.0),
    }
