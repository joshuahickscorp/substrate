"""The adaptation-retention frontier: the program's central metric. Each method is a point
(adaptation, retention). A method "wins" by Pareto-dominating or being clearly better at
matched budget. We compute the Pareto front of a set of method-points and a frontier AUC
(area under the retention-vs-adaptation curve) as a single comparable scalar.

  adaptation = how fast/well new tasks are learned (e.g. mean diagonal accuracy, or
               1/normalized-adaptation-steps)
  retention  = how well old tasks survive (e.g. final mean accuracy, or 1+BWT)
"""

from __future__ import annotations

from dataclasses import dataclass


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
