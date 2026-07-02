"""Substrate-ablation control (D2). The cheapest devastating control in the program: take any
latent-level metric and recompute it under controlled substrate variants, so a result that needs the
REAL frozen V-JEPA structure is distinguished from one that any projection (or even shuffled inputs)
would produce. A claim that survives on real latents but collapses under frozen-random/shuffled is a
real representational claim; a claim unchanged under frozen-random did not need V-JEPA (taxonomy 3).

Variants (all latent-level, no encoder access needed):
  real         : latents as-is.
  frozen_random: a fixed random linear projection of the latents (the "any projection" baseline).
  shuffled     : the x-y correspondence is broken (label permutation), the chance floor.
  compressed   : latents quantized to k bits and dequantized (precision sensitivity).

The default metric is linear-probe decodability, so substrate_ablation answers "is the target decodable
because of V-JEPA, or trivially". Any metric_fn(x, y) -> float plugs in.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .linear_probe import linear_probe


def frozen_random_projection(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A fixed random linear projection of the latents into the same dimension (renormalized). The
    'would any projection do' control: it preserves dimensionality and a rotated/mixed view of the
    information, so a result that is unchanged here did not need the specific V-JEPA geometry."""
    x = torch.as_tensor(x).float()
    d = x.shape[1]
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, d, generator=g) / (d**0.5)
    return x @ w


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
) -> dict:
    """Recompute `metric_fn` (default linear-probe score) under each substrate variant and report the
    delta vs real. Returns {real, frozen_random, shuffled, compressed, delta_frozen_random,
    delta_shuffled, delta_compressed, needs_real}. needs_real is True when the real score materially
    exceeds BOTH the frozen-random and shuffled variants (the result needed the real substrate)."""
    metric = metric_fn or _probe_score
    x = torch.as_tensor(x).float()
    y = torch.as_tensor(y)
    real = metric(x, y)
    fr = metric(frozen_random_projection(x, seed), y)
    sx, sy = shuffled_pairing(x, y, seed)
    sh = metric(sx, sy)
    comp = metric(quantize_dequantize(x, bits), y)
    margin = 0.05
    return {
        "real": round(real, 4),
        "frozen_random": round(fr, 4),
        "shuffled": round(sh, 4),
        "compressed": round(comp, 4),
        "delta_frozen_random": round(real - fr, 4),
        "delta_shuffled": round(real - sh, 4),
        "delta_compressed": round(real - comp, 4),
        "needs_real": bool(real - fr > margin and real - sh > margin),
        "bits": bits,
    }
