from __future__ import annotations

from collections.abc import Sequence


def backward_transfer(acc_matrix: Sequence[Sequence[float]]) -> float:
    t = len(acc_matrix)
    if t < 2:
        return 0.0
    last = acc_matrix[t - 1]
    return sum(float(last[j]) - float(acc_matrix[j][j]) for j in range(t - 1)) / (t - 1)


def forward_transfer(acc_matrix: Sequence[Sequence[float]], scratch_acc: Sequence[float]) -> float:
    t = len(acc_matrix)
    if t < 2:
        return 0.0
    return sum(float(acc_matrix[j - 1][j]) - float(scratch_acc[j]) for j in range(1, t)) / (t - 1)


def forgetting_area(acc_curve: Sequence[float]) -> float:
    curve = [float(x) for x in acc_curve]
    if not curve:
        return 0.0
    area, peak = 0.0, curve[0]
    for a in curve:
        peak = max(peak, a)
        area += peak - a
    return area / len(curve)


def adaptation_speed(acc_curve: Sequence[float], target_frac: float = 0.9) -> dict:
    curve = [float(x) for x in acc_curve]
    if not curve:
        return {"steps": 0, "reached": False, "target": 0.0}
    target = target_frac * curve[-1]
    for i, a in enumerate(curve):
        if a >= target:
            return {"steps": i, "reached": True, "target": round(target, 4)}
    return {"steps": len(curve), "reached": False, "target": round(target, 4)}


class LRIntegralAccumulator:
    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    def add(self, lr: float, steps: int = 1, weight: float = 1.0, partition: str = "all") -> None:
        self._totals[partition] = self._totals.get(partition, 0.0) + float(lr) * int(steps) * float(weight)

    def total(self, partition: str | None = None) -> float:
        if partition is not None:
            return self._totals.get(partition, 0.0)
        return sum(self._totals.values())

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in sorted(self._totals.items())}

    def matched(self, other: LRIntegralAccumulator, tol: float = 0.02) -> bool:
        a, b = self.total(), other.total()
        return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)
