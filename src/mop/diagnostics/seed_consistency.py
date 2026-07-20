from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from .geometry import linear_cka


def cross_seed_cka(reps: list[torch.Tensor]) -> dict:
    if len(reps) < 2:
        return {"mean_cka": 1.0, "min_cka": 1.0, "n": len(reps)}
    vals = [linear_cka(a, b) for a, b in combinations(reps, 2)]
    return {"mean_cka": round(float(np.mean(vals)), 4), "min_cka": round(float(min(vals)), 4), "n": len(reps)}


def _hungarian(cost: np.ndarray) -> list[int]:
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"cost must be square, got shape {matrix.shape}")
    n = int(matrix.shape[0])
    if n == 0:
        return []

    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[int(p[j]) - 1] = j - 1
    return assignment


def hungarian_code_agreement(codes_a: torch.Tensor, codes_b: torch.Tensor, k: int) -> float:
    a = codes_a.cpu().numpy()
    b = codes_b.cpu().numpy()
    conf = np.zeros((k, k))
    for ia, ib in zip(a, b, strict=True):
        conf[int(ia), int(ib)] += 1
    col = _hungarian(-conf)
    matched = sum(conf[r, col[r]] for r in range(k))
    return float(matched / max(1, len(a)))


def code_stability(code_assignments: list[torch.Tensor], k: int) -> dict:
    if len(code_assignments) < 2:
        return {"mean_agreement": 1.0, "chance": 1.0 / k, "stable": True}
    vals = [hungarian_code_agreement(a, b, k) for a, b in combinations(code_assignments, 2)]
    mean = float(np.mean(vals))
    return {
        "mean_agreement": round(mean, 4),
        "chance": round(1.0 / k, 4),
        "stable": bool(mean > 1.0 / k + 0.2),
    }
