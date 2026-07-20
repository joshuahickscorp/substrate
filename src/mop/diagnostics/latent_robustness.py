from __future__ import annotations

import torch

from .linear_probe import linear_probe
from .substrate_ablation import quantize_dequantize


def _shuffle_features(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    for j in range(x.shape[1]):
        perm = torch.randperm(x.shape[0], generator=g)
        out[:, j] = x[perm, j]
    return out


def _add_noise(x: torch.Tensor, level: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    std = x.std().clamp(min=1e-8)
    return x + level * std * torch.randn(x.shape, generator=g)


def _dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(x.shape, generator=g) > p).float()
    return x * mask


def degradation_curve(
    x: torch.Tensor,
    y: torch.Tensor,
    noise_levels=(0.0, 0.25, 0.5, 1.0, 2.0),
    dropout_levels=(0.0, 0.1, 0.3, 0.5, 0.8),
    bits=(32, 8, 4, 2),
    seed: int = 0,
) -> dict:
    base = linear_probe(x, y, seed=seed)["score"]
    noise_curve = {
        lv: round(linear_probe(_add_noise(x, lv, seed), y, seed=seed)["score"], 4) for lv in noise_levels
    }
    dropout_curve = {
        lv: round(linear_probe(_dropout(x, lv, seed), y, seed=seed)["score"], 4) for lv in dropout_levels
    }
    bit_curve = {
        b: round(linear_probe(x if b >= 32 else quantize_dequantize(x, b), y, seed=seed)["score"], 4)
        for b in bits
    }
    shuffled = linear_probe(_shuffle_features(x, seed), y, seed=seed)["score"]
    return {
        "base_accuracy": round(base, 4),
        "noise_curve": noise_curve,
        "dropout_curve": dropout_curve,
        "bit_curve": bit_curve,
        "shuffled_feature_floor": round(shuffled, 4),
        "flat_under_perturbation": bool(min(noise_curve.values()) > base - 0.05),
        "collapses_immediately": bool(noise_curve[noise_levels[1]] < base - 0.3),
    }
