"""Metrics: continual-learning (BWT/FWT/adaptation) and the adaptation-retention frontier.

Merged from the former metrics/{continual,frontier}.py package on the sibling-concern collapse; the
import path mop.metrics is unchanged.

Continual-learning metrics are computed from the accuracy matrix R, where R[i][j] = accuracy on task j
after finishing training on task i (i,j in 0..T-1):
  avg_accuracy      = mean_j R[T-1][j]                      (final mean accuracy)
  backward_transfer = mean_{j<T-1} (R[T-1][j] - R[j][j])    (negative => forgetting)
  forward_transfer  = mean_{j>0}  (R[j-1][j] - chance)      (zero-shot help from prior tasks)
  adaptation_speed  = steps to reach a threshold on a new task (tracked during training)

The adaptation-retention frontier is the program's central metric: each method is a point
(adaptation, retention); a method wins by Pareto-dominating or being clearly better at matched budget.
We compute the Pareto front and a frontier AUC (area under the retention-vs-adaptation curve) as a
single comparable scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ContinualResult:
    R: list[list[float]]  # accuracy matrix
    chance: float = 0.5
    adapt_steps: list[int] = field(default_factory=list)  # steps-to-threshold per task

    @property
    def T(self) -> int:
        return len(self.R)

    def avg_accuracy(self) -> float:
        return float(sum(self.R[-1]) / self.T)

    def backward_transfer(self) -> float:
        if self.T < 2:
            return 0.0
        return float(sum(self.R[-1][j] - self.R[j][j] for j in range(self.T - 1)) / (self.T - 1))

    def forward_transfer(self) -> float:
        if self.T < 2:
            return 0.0
        return float(sum(self.R[j - 1][j] - self.chance for j in range(1, self.T)) / (self.T - 1))

    def adaptation_speed(self) -> float:
        """Mean steps-to-threshold; lower is faster. inf-safe."""
        valid = [s for s in self.adapt_steps if s >= 0]
        return float(sum(valid) / len(valid)) if valid else float("inf")

    def summary(self) -> dict[str, float]:
        return {
            "avg_accuracy": self.avg_accuracy(),
            "backward_transfer": self.backward_transfer(),
            "forward_transfer": self.forward_transfer(),
            "adaptation_speed": self.adaptation_speed(),
            "final_first_task_acc": float(self.R[-1][0]),
            "first_task_peak_acc": float(self.R[0][0]),
        }


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


@dataclass
class FrontierPoint:
    name: str
    adaptation: float
    retention: float


def dominates(a: FrontierPoint, b: FrontierPoint) -> bool:
    """a Pareto-dominates b if >= on both axes and > on at least one."""
    return (
        a.adaptation >= b.adaptation
        and a.retention >= b.retention
        and (a.adaptation > b.adaptation or a.retention > b.retention)
    )


def pareto_front(points: list[FrontierPoint]) -> list[FrontierPoint]:
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def frontier_auc(points: list[FrontierPoint]) -> float:
    """Area under the retention(adaptation) staircase of the Pareto front, adaptation in
    [0,1]. A scalar where higher = a better frontier. Robust to a single point."""
    front = sorted(pareto_front(points), key=lambda p: p.adaptation)
    if not front:
        return 0.0
    if len(front) == 1:
        return front[0].adaptation * front[0].retention
    auc = 0.0
    for i in range(1, len(front)):
        dx = front[i].adaptation - front[i - 1].adaptation
        auc += dx * (front[i].retention + front[i - 1].retention) / 2
    auc += front[0].adaptation * front[0].retention  # rectangle from 0 to first point
    return float(auc)


def retention_from_bwt(bwt: float) -> float:
    """Map backward transfer (<=0 forgetting) to a retention score in roughly [0,1]."""
    return float(max(0.0, 1.0 + bwt))


__all__ = [
    "ContinualResult",
    "accuracy",
    "FrontierPoint",
    "dominates",
    "frontier_auc",
    "pareto_front",
    "retention_from_bwt",
]
