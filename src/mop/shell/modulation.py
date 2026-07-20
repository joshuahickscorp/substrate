
from __future__ import annotations

import torch
from torch import nn


class ContextGating(nn.Module):

    def __init__(self, dim: int, n_contexts: int):
        super().__init__()
        self.gate = nn.Embedding(n_contexts, dim)
        nn.init.zeros_(self.gate.weight)  # start near gate=0.5 (mild), learn specialization

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return h * torch.sigmoid(self.gate(context))


class WorkingMemory(nn.Module):

    def __init__(self, dim: int, slots: int = 4):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.write = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, slots)

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        if mem is None:
            mem = x.new_zeros(b, self.slots, self.dim)
        w = torch.softmax(self.gate(x), dim=-1).unsqueeze(-1)  # [B, slots, 1]
        upd = self.write(x).unsqueeze(1)  # [B, 1, dim]
        mem = mem + w * (upd - mem)  # gated write
        return mem.mean(1), mem  # read, new mem


class Chunking(nn.Module):

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def boundaries(self, seq: torch.Tensor) -> torch.Tensor:
        d = (seq[:, 1:] - seq[:, :-1]).norm(dim=-1)  # [B, T-1]
        return (d > self.threshold).float()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.boundaries(seq)


def build_modulation(cfg, dim: int) -> dict[str, nn.Module]:
    mods: dict[str, nn.Module] = {}
    if bool(cfg.context_gating):
        mods["context_gating"] = ContextGating(dim, int(cfg.n_contexts))
    if bool(cfg.working_memory):
        mods["working_memory"] = WorkingMemory(dim, int(cfg.wm_slots))
    if bool(cfg.chunking):
        mods["chunking"] = Chunking()
    return mods
