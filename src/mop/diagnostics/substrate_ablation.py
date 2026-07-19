
from __future__ import annotations

from collections.abc import Callable

import torch

from .linear_probe import linear_probe


def frozen_random_projection(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    d = x.shape[1]
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, d, generator=g) / (d**0.5)
    return x @ w


def rank_reduced_projection(x: torch.Tensor, rank: int | None = None, seed: int = 0) -> torch.Tensor:
    x = torch.as_tensor(x).float()
    d = x.shape[1]
    r = rank if rank is not None else max(1, d // 4)
    r = min(r, d)
    g = torch.Generator().manual_seed(seed)
    down = torch.randn(d, r, generator=g) / (d**0.5)
    up = torch.randn(r, d, generator=g) / (r**0.5)
    return x @ down @ up


def shuffled_pairing(x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(y.shape[0], generator=g)
    return x, y[perm]


def quantize_dequantize(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
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
        "delta_frozen_random": round(real - fr, 4),
        "delta_rank_reduced": round(real - rr, 4),
        "delta_shuffled": round(real - sh, 4),
        "delta_compressed": round(real - comp, 4),
        "needs_real": bool(real - sh > margin),
        "beats_frozen_random": bool(real - fr > margin),
        "beats_rank_reduced": bool(real - rr > margin),
        "bits": bits,
    }
