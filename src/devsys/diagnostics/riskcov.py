"""Risk-coverage and discrimination metrics (WP-02, H-RISKCOV): AUROC (rank-based, tie-corrected),
equal-mass ECE (fixed-width bins hide miscalibration where confidence mass concentrates), the
risk-coverage curve and its area (AURC, lower is better), a Pareto-frontier area for accuracy-vs-cost
tradeoffs, and the seed-CI plus sign-flip report every MoT verdict table requires. All pure torch and
python, CPU-fine, deterministic, nothing trains.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def _as1d(x) -> torch.Tensor:
    return torch.as_tensor(x).detach().to(torch.float64).reshape(-1)


def auroc(scores, labels) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) formulation, ties handled by average
    rank. `scores` higher = more positive; `labels` in {0, 1}. Returns 0.5 for a degenerate class."""
    s, y = _as1d(scores), _as1d(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = s.argsort()
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float64)
    # average ranks over ties
    for v in s.unique():
        m = s == v
        if int(m.sum()) > 1:
            ranks[m] = ranks[m].mean()
    u = float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def ece_equal_mass(confidences, correct, bins: int = 10) -> float:
    """Expected calibration error with EQUAL-MASS bins (each bin holds ~n/bins points, sorted by
    confidence): sum over bins of (mass fraction) * |mean confidence - mean accuracy|."""
    c, k = _as1d(confidences), _as1d(correct)
    n = len(c)
    order = c.argsort()
    c, k = c[order], k[order]
    edges = [round(i * n / bins) for i in range(bins + 1)]
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        err += (hi - lo) / n * abs(float(c[lo:hi].mean()) - float(k[lo:hi].mean()))
    return err


def risk_coverage(confidences, correct) -> dict:
    """Selective-prediction curve: sort by descending confidence, at each coverage c the risk is the
    error rate on the top-c fraction. Keys: coverage (list), risk (list), aurc (area under the curve,
    lower is better), risk_at_full (error at coverage 1.0)."""
    c, k = _as1d(confidences), _as1d(correct)
    order = c.argsort(descending=True)
    k = k[order]
    n = len(k)
    cum_err = torch.cumsum(1.0 - k, dim=0)
    counts = torch.arange(1, n + 1, dtype=torch.float64)
    risk = cum_err / counts
    coverage = counts / n
    aurc = float(risk.mean())
    return {
        "coverage": [round(float(v), 4) for v in coverage],
        "risk": [round(float(v), 4) for v in risk],
        "aurc": round(aurc, 4),
        "risk_at_full": round(float(risk[-1]), 4),
    }


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated subset of (cost, quality) points: keep a point iff no other point has cost <= and
    quality >= with one strict. Returned sorted by cost ascending."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    front: list[tuple[float, float]] = []
    best = -math.inf
    for x, y in pts:
        if y > best:
            front.append((x, y))
            best = y
    return front


def pareto_area(points: Sequence[tuple[float, float]], *, x_max: float | None = None) -> float:
    """Area under the (cost, quality) Pareto frontier by step interpolation up to x_max (default: the
    max cost seen). A scalar for 'how much quality per unit budget the family buys'; compare arms at
    identical x_max only."""
    front = pareto_frontier(points)
    if not front:
        return 0.0
    hi = float(x_max) if x_max is not None else front[-1][0]
    area, prev_x, prev_y = 0.0, front[0][0], front[0][1]
    for x, y in front[1:] + [(hi, front[-1][1])]:
        if x > prev_x:
            area += (min(x, hi) - prev_x) * prev_y
        prev_x, prev_y = x, y
        if prev_x >= hi:
            break
    return area


def seed_ci(values: Sequence[float], z: float = 1.96) -> dict:
    """Mean, sample SD, and a normal-approximation CI half-width over per-seed values. Keys: n, mean,
    sd, ci_half, lo, hi. Small-n honesty: with n < 3 the CI is reported but flagged unstable."""
    v = [float(x) for x in values]
    n = len(v)
    mean = sum(v) / max(n, 1)
    sd = (sum((x - mean) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    half = z * sd / n**0.5 if n > 0 else 0.0
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci_half": round(half, 4),
        "lo": round(mean - half, 4),
        "hi": round(mean + half, 4),
        "unstable": n < 3,
    }


def sign_flip_report(deltas: Sequence[float]) -> dict:
    """Per-seed effect deltas -> the sign-consistency verdict the aggregate table renders. Keys:
    n, n_pos, n_neg, n_zero, any_flip (both signs present), consistent_sign (+1, -1, or 0)."""
    d = [float(x) for x in deltas]
    n_pos = sum(1 for x in d if x > 0)
    n_neg = sum(1 for x in d if x < 0)
    n_zero = len(d) - n_pos - n_neg
    any_flip = n_pos > 0 and n_neg > 0
    consistent = 0 if any_flip or (n_pos == 0 and n_neg == 0) else (1 if n_pos > 0 else -1)
    return {
        "n": len(d),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "any_flip": any_flip,
        "consistent_sign": consistent,
    }
