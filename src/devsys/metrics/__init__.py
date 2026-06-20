"""Metrics: continual-learning (BWT/FWT/adaptation) and the adaptation-retention frontier."""

from __future__ import annotations

from .continual import ContinualResult, accuracy
from .frontier import FrontierPoint, frontier_auc, pareto_front, retention_from_bwt

__all__ = [
    "ContinualResult",
    "accuracy",
    "FrontierPoint",
    "frontier_auc",
    "pareto_front",
    "retention_from_bwt",
]
