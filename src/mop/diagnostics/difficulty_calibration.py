"""D3: difficulty calibration. A tie between two methods is only interpretable if the regime is hard
enough that a KNOWN-SEPARABLE reference method actually separates in it; otherwise a tie could mean
either "the methods are equivalent" or "nothing can be measured here" and the two are indistinguishable.
This certifies the regime before a tie elsewhere (E7/E8-style structural comparisons) is trusted.
"""

from __future__ import annotations

import torch

from .linear_probe import linear_probe


def reference_separation(x: torch.Tensor, y: torch.Tensor, seed: int = 0, margin: float = 0.1) -> dict:
    """Run a strong, known-separable reference (the linear probe itself, at generous epochs) on (x, y).
    If it clears chance by `margin`, the regime is calibrated: a tie between two OTHER methods on this
    same data is a real finding, not measurement noise. If the reference also fails, no tie here is
    interpretable (the regime itself is too hard or the factor is not present)."""
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
    """Certify whether an observed near-tie between two method scores (score_a, score_b) on this (x, y)
    regime is interpretable: only if the reference separates AND the two scores are actually close."""
    cal = reference_separation(x, y, seed=seed)
    is_tie = abs(score_a - score_b) <= tie_tol
    return {
        **cal,
        "score_a": round(float(score_a), 4),
        "score_b": round(float(score_b), 4),
        "is_tie": bool(is_tie),
        "tie_is_meaningful": bool(is_tie and cal["regime_calibrated"]),
    }
