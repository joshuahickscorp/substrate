"""Ensemble of small predictors/heads. Disagreement (variance across members) is the
epistemic-uncertainty signal: high where the model is ignorant-but-learnable, near-zero on
irreducible aleatoric noise once members agree it is noise. This is what lets E4 pass the
noisy-TV test where a point predictor cannot.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, make_member: Callable[[], nn.Module], size: int = 5, bootstrap: bool = False):
        super().__init__()
        self.members = nn.ModuleList([make_member() for _ in range(size)])
        self.bootstrap = bootstrap

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Stack member outputs: [size, B, ...]."""
        return torch.stack([m(x, *args) for m in self.members], dim=0)

    def mean_and_disagreement(self, x: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(x, *args)  # [S, B, D]
        mean = out.mean(0)
        disagreement = out.var(0).mean(dim=-1)  # [B] epistemic, per-sample
        return mean, disagreement

    def bootstrap_mask(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Per-member bernoulli mask over a batch for bootstrapped training (if enabled)."""
        if not self.bootstrap:
            return torch.ones(len(self.members), n)
        return torch.bernoulli(torch.full((len(self.members), n), 0.5), generator=generator)
