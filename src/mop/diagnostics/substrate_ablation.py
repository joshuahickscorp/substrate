"""Substrate-ablation control (D2). Take any latent-level metric and recompute it under controlled
substrate variants, so a result that is genuinely present in the frozen latent is distinguished from one
that shuffled (chance-floor) inputs would produce. The single honest verdict this control can license at
the LATENT level is decodability above the shuffled floor: real beats the label-permuted floor by a
material margin. That is the meaning of needs_real here.

What this control CANNOT license (and no longer claims to): "the result needed the REAL pretrained V-JEPA
substrate rather than any encoder". That is an encoder-level claim and requires a random-init-ENCODER
comparison (a different frozen encoder producing different latents), which lives in the caching / adapter
path, NOT here. In particular a fixed random LINEAR map of the latents is the WRONG control for that
question when the metric is a linear probe: a full-rank d x d map is invertible, so a linear probe simply
re-learns the inverse and scores identically (delta_frozen_random == 0 by construction). frozen_random is
therefore retained only as a descriptive, KNOWN-VACUOUS-FOR-LINEAR-METRICS field; it is never gated on.

Variants (all latent-level, no encoder access needed):
  real          : latents as-is.
  shuffled      : the x-y correspondence is broken (label permutation), the chance floor. THE gate.
  compressed    : latents quantized to k bits and dequantized (precision sensitivity).
  frozen_random : a fixed FULL-RANK random linear projection. Vacuous for a linear metric (invertible),
                  reported for transparency but NOT used in any verdict. See beats_frozen_random.
  rank_reduced  : a fixed random projection to a LOWER rank (loses information). Unlike frozen_random this
                  can actually move a linear metric, so it is a genuine "does the metric survive a lossy
                  bottleneck" descriptive control. Reported, not gated (rank is a free parameter).

The default metric is linear-probe decodability, so substrate_ablation answers "is the target decodable
above the shuffled floor, or is it a chance artifact". Any metric_fn(x, y) -> float plugs in.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .linear_probe import linear_probe


def frozen_random_projection(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A fixed FULL-RANK random linear projection of the latents into the same dimension (renormalized).

    WARNING (known vacuity): a square d x d Gaussian map is almost surely invertible, so for a LINEAR
    metric (the default linear probe) it is a no-op: the probe re-learns the inverse and scores the same,
    hence delta_frozen_random ~ 0 by construction. It is NOT a test of whether a result needed the real
    substrate. It is kept for descriptive transparency and for the few consumers that use it as an
    honest full-rank re-mixing arm on a NON-linear or geometry metric (where invariance is not free). For
    a genuine lossy control on a linear metric, use rank_reduced_projection."""
    x = torch.as_tensor(x).float()
    d = x.shape[1]
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, d, generator=g) / (d**0.5)
    return x @ w


def rank_reduced_projection(x: torch.Tensor, rank: int | None = None, seed: int = 0) -> torch.Tensor:
    """A fixed random projection to a LOWER rank, embedded back into the original dimension (renormalized).

    Unlike frozen_random_projection this is genuinely lossy: it maps to rank < d dimensions and back, so a
    linear probe cannot recover components that fell in the null space. It is therefore a real "does the
    metric survive a lossy bottleneck" control (a rank_reduced that ties real means the target lives in a
    low-dimensional subspace, which is informative), reported descriptively. rank defaults to max(1, d//4)."""
    x = torch.as_tensor(x).float()
    d = x.shape[1]
    r = rank if rank is not None else max(1, d // 4)
    r = min(r, d)
    g = torch.Generator().manual_seed(seed)
    down = torch.randn(d, r, generator=g) / (d**0.5)
    up = torch.randn(r, d, generator=g) / (r**0.5)
    return x @ down @ up


def shuffled_pairing(x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Break the x-y correspondence by permuting labels: the chance floor. Any decodability that
    survives this is an artifact of the metric, not the representation."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(y.shape[0], generator=g)
    return x, y[perm]


def quantize_dequantize(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-feature uniform quantization to `bits` then dequantization (precision-sensitivity control)."""
    x = torch.as_tensor(x).float()
    levels = max(2, 2**bits)
    lo = x.min(0, keepdim=True).values
    hi = x.max(0, keepdim=True).values
    scale = (hi - lo).clamp(min=1e-8) / (levels - 1)
    q = torch.round((x - lo) / scale)
    return q * scale + lo


def _probe_score(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(linear_probe(x, y)["score"])


def substrate_ablation(
    x: torch.Tensor,
    y: torch.Tensor,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
    seed: int = 0,
    bits: int = 4,
    rank: int | None = None,
) -> dict:
    """Recompute `metric_fn` (default linear-probe score) under each substrate variant and report the
    delta vs real.

    needs_real is True IFF the real score materially exceeds the SHUFFLED (label-permuted) chance floor:
    the target is genuinely decodable, not a metric artifact. This is the only substrate-specificity claim
    a latent-level control can honestly make. It is NOT gated on frozen_random (which is vacuous for a
    linear metric, see frozen_random_projection); delta_frozen_random and beats_frozen_random are reported
    for transparency only. rank_reduced / beats_rank_reduced are a descriptive lossy-bottleneck control.
    """
    metric = metric_fn or _probe_score
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y)
    real = metric(x, y)
    fr = metric(frozen_random_projection(x, seed), y)
    rr = metric(rank_reduced_projection(x, rank, seed), y)
    sx, sy = shuffled_pairing(x, y, seed)
    sh = metric(sx, sy)
    comp = metric(quantize_dequantize(x, bits), y)
    margin = 0.05
    return {
        "real": round(real, 4),
        "frozen_random": round(fr, 4),
        "rank_reduced": round(rr, 4),
        "shuffled": round(sh, 4),
        "compressed": round(comp, 4),
        # descriptive deltas; delta_frozen_random is ~0 for a linear metric by construction (see module doc)
        "delta_frozen_random": round(real - fr, 4),
        "delta_rank_reduced": round(real - rr, 4),
        "delta_shuffled": round(real - sh, 4),
        "delta_compressed": round(real - comp, 4),
        # THE verdict: decodable above the shuffled chance floor (the honest latent-level claim)
        "needs_real": bool(real - sh > margin),
        # transparency-only flags, NEVER gate a verdict on beats_frozen_random for a linear metric
        "beats_frozen_random": bool(real - fr > margin),
        "beats_rank_reduced": bool(real - rr > margin),
        "bits": bits,
    }
