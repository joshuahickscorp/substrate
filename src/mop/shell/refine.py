from __future__ import annotations

import torch
from torch import nn

from .predictor import mlp


class IterativeRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        steps: int = 4,
        halt: bool = False,
        halt_threshold: float = 0.9,
        mode: str = "residual",
        pc_rate: float = 0.5,
    ):
        super().__init__()
        if mode not in ("residual", "predictive_coding"):
            raise ValueError(f"mode must be residual or predictive_coding, got {mode!r}")
        self.dim = dim
        self.steps = int(steps)
        self.halt = bool(halt)
        self.halt_threshold = float(halt_threshold)
        self.mode = mode
        self.pc_rate = float(pc_rate)
        self.norm = nn.LayerNorm(dim)
        self.block = mlp(dim, dim, hidden, depth=1, ln=True)  # one dim->hidden->dim residual block
        self.halt_head = nn.Linear(dim, 1) if halt else None

    def _update(self, z: torch.Tensor) -> torch.Tensor:
        pred = self.block(self.norm(z))
        if self.mode == "predictive_coding":
            return self.pc_rate * (pred - z)
        return pred

    def forward(self, z: torch.Tensor, max_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        n = max_steps or self.steps
        b = z.shape[0]
        used = torch.zeros(b, dtype=torch.float32, device=z.device)
        if not self.halt:
            for _ in range(n):
                z = z + self._update(z)
            used = used + n
            return z, used
        assert self.halt_head is not None  # constructed iff halt=True (the branch guard above)
        cum = torch.zeros(b, device=z.device)
        active = torch.ones(b, dtype=torch.bool, device=z.device)
        for _ in range(n):
            upd = self._update(z)
            z = torch.where(active.unsqueeze(-1), z + upd, z)
            used = used + active.float()
            p = torch.sigmoid(self.halt_head(z)).squeeze(-1)
            cum = cum + torch.where(active, p, torch.zeros_like(p))
            active = active & (cum < self.halt_threshold)
            if not active.any():
                break
        return z, used

    @torch.no_grad()
    def unroll(self, z: torch.Tensor, steps: int) -> tuple[torch.Tensor, list[float]]:
        norms: list[float] = []
        for _ in range(int(steps)):
            u = self._update(z)
            norms.append(float(u.norm(dim=-1).mean()))
            z = z + u
        return z, norms

    def block_count(self, max_steps: int | None = None) -> int:
        return int(max_steps or self.steps)


class Verifier(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(dim, 1, hidden, depth=1, ln=True)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z)
