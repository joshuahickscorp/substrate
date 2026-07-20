from __future__ import annotations

import torch
import torch.nn.functional as F

from ..seeding import seed_everything
from .linear_probe import linear_probe
from .substrate_ablation import frozen_random_projection, quantize_dequantize


def _bottleneck_probe(
    x: torch.Tensor, y: torch.Tensor, width: int, epochs: int = 150, lr: float = 0.02, seed: int = 0
) -> float:
    seed_everything(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm].float(), y[perm]
    cut = int(n * 0.7)
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]
    nc = int(y.max()) + 1
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], width), torch.nn.Linear(width, nc))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((net(xte).argmax(-1) == yte).float().mean())


def capability_per_bit(x: torch.Tensor, y: torch.Tensor, widths=(1, 2, 4, 8, 16, 32), seed: int = 0) -> dict:
    xr = x.float()
    xf = frozen_random_projection(xr, seed)
    real = {w: round(_bottleneck_probe(xr, y, w, seed=seed), 4) for w in widths}
    fr = {w: round(_bottleneck_probe(xf, y, w, seed=seed), 4) for w in widths}
    full = real[max(widths)]
    knee = next((w for w in widths if real[w] >= 0.95 * full), max(widths))
    small = [w for w in widths if w <= 4]
    sep = max((real[w] - fr[w]) for w in small) if small else 0.0
    return {
        "widths": list(widths),
        "acc_real": real,
        "acc_frozen_random": fr,
        "knee_width_95pct": int(knee),
        "max_small_width_separation": round(sep, 4),
        "real_packs_tighter": bool(sep > 0.1),
    }


def quantization_robustness(x: torch.Tensor, y: torch.Tensor, bits=(8, 4, 3, 2), seed: int = 0) -> dict:
    xr = x.float()
    xf = frozen_random_projection(xr, seed)
    real = {b: round(linear_probe(quantize_dequantize(xr, b), y, seed=seed)["score"], 4) for b in bits}
    fr = {b: round(linear_probe(quantize_dequantize(xf, b), y, seed=seed)["score"], 4) for b in bits}
    gap = sum(real[b] - fr[b] for b in bits) / len(bits)
    return {
        "bits": list(bits),
        "acc_real": real,
        "acc_frozen_random": fr,
        "mean_gap": round(gap, 4),
        "real_more_robust": bool(gap > 0.05),
    }
