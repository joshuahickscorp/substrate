
from __future__ import annotations

import torch

from .linear_probe import linear_probe


def reference_separation(x: torch.Tensor, y: torch.Tensor, seed: int = 0, margin: float = 0.1) -> dict:
    ref = linear_probe(x, y, seed=seed, epochs=400)
    gap = ref["score"] - ref["chance"]
    return {
        "reference_score": round(ref["score"], 4),
        "chance": round(ref["chance"], 4),
        "gap": round(gap, 4),
        "regime_calibrated": bool(gap > margin),
    }


def calibrated_tie(
    x: torch.Tensor, y: torch.Tensor, score_a: float, score_b: float, seed: int = 0, tie_tol: float = 0.03
) -> dict:
    cal = reference_separation(x, y, seed=seed)
    is_tie = abs(score_a - score_b) <= tie_tol
    return {
        **cal,
        "score_a": round(float(score_a), 4),
        "score_b": round(float(score_b), 4),
        "is_tie": bool(is_tie),
        "tie_is_meaningful": bool(is_tie and cal["regime_calibrated"]),
    }
