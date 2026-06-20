"""Task heads on frozen latents. A plain classifier, and a probabilistic (Gaussian) head
that emits mean+logvar so we can separate epistemic from aleatoric uncertainty and run
calibration. The probabilistic head is the C-cluster prerequisite for E4 / noisy-TV.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .predictor import mlp


class ClassHead(nn.Module):
    """latent -> class logits."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 512, depth: int = 1):
        super().__init__()
        self.net = mlp(dim, n_classes, hidden, depth) if depth else nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):
    """latent -> (mean, logvar) over a target. nll separates aleatoric (logvar) from the
    epistemic spread an ensemble of these would show. logvar is clamped for stability."""

    def __init__(self, dim: int, out: int, hidden: int = 512, depth: int = 1, logvar_bounds=(-8.0, 6.0)):
        super().__init__()
        self.body = mlp(dim, hidden, hidden, max(0, depth - 1)) if depth > 1 else nn.Identity()
        feat = hidden if depth > 1 else dim
        self.mean = nn.Linear(feat, out)
        self.logvar = nn.Linear(feat, out)
        self.lo, self.hi = logvar_bounds

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mean(h), self.logvar(h).clamp(self.lo, self.hi)

    def nll(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, logvar = self(x)
        return 0.5 * (logvar + (target - mean) ** 2 / logvar.exp() + math.log(2 * math.pi)).mean()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self(x)
        return mean, logvar.exp()  # mean, aleatoric variance


def build_head(cfg, dim: int, n_classes: int, out_dim: int | None = None):
    """Probabilistic head iff cfg.probabilistic, regressing a latent of size out_dim; else a
    classifier of n_classes."""
    if bool(cfg.probabilistic) and out_dim is not None:
        return GaussianHead(dim, out_dim, hidden=int(cfg.hidden))
    return ClassHead(dim, n_classes, hidden=int(cfg.hidden))
