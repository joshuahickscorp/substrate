
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
        return torch.stack([m(x, *args) for m in self.members], dim=0)

    def mean_and_disagreement(self, x: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(x, *args)  # [S, B, D]
        mean = out.mean(0)
        disagreement = out.var(0).mean(dim=-1)  # [B] epistemic, per-sample
        return mean, disagreement

    def bootstrap_mask(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if not self.bootstrap:
            return torch.ones(len(self.members), n)
        return torch.bernoulli(torch.full((len(self.members), n), 0.5), generator=generator)
