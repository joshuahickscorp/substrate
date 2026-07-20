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
    return (
        a.adaptation >= b.adaptation
        and a.retention >= b.retention
        and (a.adaptation > b.adaptation or a.retention > b.retention)
    )


def pareto_front(points: list[FrontierPoint]) -> list[FrontierPoint]:
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def frontier_auc(points: list[FrontierPoint]) -> float:
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
