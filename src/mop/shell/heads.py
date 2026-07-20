
from __future__ import annotations

import math

import torch
from torch import nn

from .predictor import mlp


class ClassHead(nn.Module):

    def __init__(self, dim: int, n_classes: int, hidden: int = 512, depth: int = 1):
        super().__init__()
        self.net = mlp(dim, n_classes, hidden, depth) if depth else nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):

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


class KWTAHead(nn.Module):

    def __init__(self, dim: int, hidden: int, n_classes: int, k: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.k = max(1, min(int(k), int(hidden)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        thresh = h.topk(self.k, dim=-1).values[..., -1:]
        return self.fc2(h * (h >= thresh))


class MoEHead(nn.Module):

    def __init__(self, dim: int, n_classes: int, n_experts: int, expert_hidden: int):
        super().__init__()
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, expert_hidden), nn.GELU(), nn.Linear(expert_hidden, n_classes))
            for _ in range(n_experts)
        )
        self.last_gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.router(x), dim=-1)  # [B, E]
        self.last_gates = gates.detach()
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C]
        return (gates.unsqueeze(-1) * stacked).sum(dim=1)


def routing_entropy(gates: torch.Tensor) -> float:
    p = gates.clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean())


def moe_expert_hidden_for_dense(dim: int, hidden: int, n_classes: int, n_experts: int) -> int:
    dense = dim * hidden + hidden + n_classes * hidden + n_classes
    router = dim * n_experts + n_experts
    per_expert_budget = max(1.0, (dense - router) / n_experts)
    w = (per_expert_budget - n_classes) / (dim + n_classes + 1.0)
    return max(1, int(round(w)))


def build_head(cfg, dim: int, n_classes: int, out_dim: int | None = None):
    if bool(cfg.probabilistic) and out_dim is not None:
        return GaussianHead(dim, out_dim, hidden=int(cfg.hidden))
    return ClassHead(dim, n_classes, hidden=int(cfg.hidden))
