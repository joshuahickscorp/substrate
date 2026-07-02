"""Continual-learning metrics (WP-02): the factored BWT core (accuracy-matrix form of what
buffer_compression._bwt_at_bits computes inline), plus the four metrics that exist nowhere else yet:
forward transfer (FWT), forgetting-area, adaptation-speed, and the LR-integral accumulator that makes
the PR4 gate-vs-ungated allocation comparison honest (matched LR-integral per partition, the e4
conflation guard). All pure python over accuracy curves and matrices; nothing here trains.

Conventions: `acc_matrix[i][j]` is accuracy on task j evaluated right after finishing training task i
(rows = time, cols = tasks, lower-triangle-plus-diagonal is what a single pass produces). Curves are
per-step scalars over a stream.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from collections.abc import Sequence


def backward_transfer(acc_matrix: Sequence[Sequence[float]]) -> float:
    """BWT (GEM convention): mean over tasks j < T-1 of acc_matrix[T-1][j] - acc_matrix[j][j].
    Negative = forgetting. The 2-task special case reduces to the _bwt_at_bits scalar."""
    t = len(acc_matrix)
    if t < 2:
        return 0.0
    last = acc_matrix[t - 1]
    return sum(float(last[j]) - float(acc_matrix[j][j]) for j in range(t - 1)) / (t - 1)


def forward_transfer(acc_matrix: Sequence[Sequence[float]], scratch_acc: Sequence[float]) -> float:
    """FWT: mean over tasks j > 0 of acc_matrix[j-1][j] - scratch_acc[j], where scratch_acc[j] is the
    from-scratch (or random-init) accuracy on task j. Positive = the stream prepared the learner."""
    t = len(acc_matrix)
    if t < 2:
        return 0.0
    return sum(float(acc_matrix[j - 1][j]) - float(scratch_acc[j]) for j in range(1, t)) / (t - 1)


def forgetting_area(acc_curve: Sequence[float]) -> float:
    """Area between the running maximum and the current accuracy over a stream, normalized by curve
    length: mean_t (max_{s<=t} acc_s - acc_t). Zero iff accuracy never drops below its own peak; large
    = deep or long forgetting troughs that endpoint BWT cannot see."""
    curve = [float(x) for x in acc_curve]
    if not curve:
        return 0.0
    area, peak = 0.0, curve[0]
    for a in curve:
        peak = max(peak, a)
        area += peak - a
    return area / len(curve)


def adaptation_speed(acc_curve: Sequence[float], target_frac: float = 0.9) -> dict:
    """Steps to reach target_frac of the curve's final accuracy. Keys: steps (int, len(curve) if never
    reached), reached (bool), target (float). Lower steps = faster adaptation on a new task."""
    curve = [float(x) for x in acc_curve]
    if not curve:
        return {"steps": 0, "reached": False, "target": 0.0}
    target = target_frac * curve[-1]
    for i, a in enumerate(curve):
        if a >= target:
            return {"steps": i, "reached": True, "target": round(target, 4)}
    return {"steps": len(curve), "reached": False, "target": round(target, 4)}


class LRIntegralAccumulator:
    """Accumulates the LR-integral (sum of learning rate x steps, optionally weighted by update norm)
    per named partition. The PR4/PR5 comparisons are only honest at MATCHED LR-integral, so both arms
    log through one of these and the experiment asserts the totals match before comparing outcomes."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    def add(self, lr: float, steps: int = 1, weight: float = 1.0, partition: str = "all") -> None:
        """Record `steps` optimizer steps at learning rate `lr` for `partition`, contributing
        lr * steps * weight (weight defaults to 1; pass an update norm for norm-weighted variants)."""
        self._totals[partition] = self._totals.get(partition, 0.0) + float(lr) * int(steps) * float(weight)

    def total(self, partition: str | None = None) -> float:
        """LR-integral for one partition, or the grand total across partitions when None."""
        if partition is not None:
            return self._totals.get(partition, 0.0)
        return sum(self._totals.values())

    def as_dict(self) -> dict[str, float]:
        """partition -> LR-integral, for the run-dir json."""
        return {k: round(v, 6) for k, v in sorted(self._totals.items())}

    def matched(self, other: LRIntegralAccumulator, tol: float = 0.02) -> bool:
        """True iff grand totals agree within relative tol (the matched-LR-integral precondition)."""
        a, b = self.total(), other.total()
        return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)
