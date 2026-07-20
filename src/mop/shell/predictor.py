
from __future__ import annotations

import torch
from torch import nn


def mlp(din: int, dout: int, hidden: int, depth: int, dropout: float = 0.0, ln: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = din
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        if ln:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = hidden
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


class Predictor(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden: int = 1024,
        depth: int = 2,
        dropout: float = 0.0,
        action_dim: int = 0,
        layernorm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.net = mlp(dim + action_dim, dim, hidden, depth, dropout, layernorm)

    def forward(self, x: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if self.action_dim:
            assert action is not None, "action-conditioned predictor needs an action"
            x = torch.cat([x, action], dim=-1)
        return self.net(x)

    @classmethod
    def from_cfg(cls, cfg, dim: int) -> Predictor:
        return cls(
            dim=dim,
            hidden=int(cfg.hidden),
            depth=int(cfg.depth),
            dropout=float(cfg.dropout),
            action_dim=int(cfg.action_dim),
            layernorm=bool(cfg.layernorm),
        )
