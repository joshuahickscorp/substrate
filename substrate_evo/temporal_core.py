"""Temporal substrate kernel + architectures A-T and C + converged temporal baselines, on genuine sequences.

Input is a (batch, T, channels) sequence where temporal order matters (validated by the order-headroom gate).
A-T = Temporal Shared Latent Workspace: owned per-step projection -> GRU fast state -> medium EMA context ->
shared slow workspace -> head. C = Multi-Horizon Predictive State: GRU core trained with an auxiliary future-
latent prediction objective (the premise: predictive state retains temporal info useful for adaptation). Both
own their projection/core/slow params. Memory is GDumb over windows (established, no learned selector).
House style: no dashes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64


class OwnedProj(nn.Module):
    """Owned per-timestep projection: Conv1d over channels + linear to latent (trainable, MOP-owned)."""

    def __init__(self, ch, latent=LATENT):
        super().__init__()
        self.c = nn.Conv1d(ch, 32, 5, padding=2)
        self.l = nn.Linear(32, latent)

    def forward(self, x):  # x: (B,T,ch)
        z = F.relu(self.c(x.transpose(1, 2))).transpose(1, 2)  # (B,T,32)
        return F.relu(self.l(z))  # (B,T,latent)


class GRUBase(nn.Module):
    name = "gru"

    def __init__(self, ch, n_out):
        super().__init__()
        self.proj = OwnedProj(ch); self.core = nn.GRU(LATENT, LATENT, batch_first=True); self.head = nn.Linear(LATENT, n_out)
        self.param_groups = {"all": list(self.parameters())}

    def forward(self, x, ctx=None):
        o, _ = self.core(self.proj(x)); return self.head(o[:, -1]), None


class LSTMBase(nn.Module):
    name = "lstm"

    def __init__(self, ch, n_out):
        super().__init__()
        self.proj = OwnedProj(ch); self.core = nn.LSTM(LATENT, LATENT, batch_first=True); self.head = nn.Linear(LATENT, n_out)
        self.param_groups = {"all": list(self.parameters())}

    def forward(self, x, ctx=None):
        o, _ = self.core(self.proj(x)); return self.head(o[:, -1]), None


class BagBase(nn.Module):
    name = "bag"

    def __init__(self, ch, n_out):
        super().__init__()
        self.proj = OwnedProj(ch); self.net = nn.Sequential(nn.Linear(LATENT * 3, LATENT), nn.ReLU(), nn.Linear(LATENT, n_out))
        self.param_groups = {"all": list(self.parameters())}

    def forward(self, x, ctx=None):
        z = self.proj(x); f = torch.cat([z.mean(1), z.std(1), z.amax(1)], 1); return self.net(f), None


class ArchAT(nn.Module):
    """Temporal Shared Latent Workspace: fast GRU state + medium EMA context + slow shared workspace."""

    name = "AT"

    def __init__(self, ch, n_out):
        super().__init__()
        self.proj = OwnedProj(ch); self.fast = nn.GRU(LATENT, LATENT, batch_first=True)
        self.workspace = nn.Sequential(nn.Linear(LATENT * 2, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT))
        self.head = nn.Linear(LATENT, n_out)
        self.aux = nn.Linear(LATENT, LATENT)  # multi-horizon predictive aux head
        self.register_buffer("medium", torch.zeros(LATENT))  # medium-timescale EMA context (persistent state)
        self.param_groups = {"proj": list(self.proj.parameters()), "fast": list(self.fast.parameters()),
                             "workspace": list(self.workspace.parameters()), "head": list(self.head.parameters()) + list(self.aux.parameters())}

    def forward(self, x, ctx=None, update_medium=False):
        z = self.proj(x); o, h = self.fast(z); last = o[:, -1]
        if update_medium and self.training:
            with torch.no_grad():
                self.medium.mul_(0.95).add_(0.05 * last.mean(0))
        med = self.medium.unsqueeze(0).expand(last.shape[0], -1)
        rep = F.relu(self.workspace(torch.cat([last, med], 1))) + last
        return self.head(rep), (o, z)  # return internals for aux prediction


class ArchC(nn.Module):
    """Multi-Horizon Predictive State: GRU core with an auxiliary future-latent prediction objective."""

    name = "C"

    def __init__(self, ch, n_out):
        super().__init__()
        self.proj = OwnedProj(ch); self.core = nn.GRU(LATENT, LATENT, batch_first=True)
        self.pred = nn.Linear(LATENT, LATENT)  # predict future latent
        self.head = nn.Linear(LATENT, n_out)
        self.param_groups = {"all": list(self.parameters())}

    def forward(self, x, ctx=None):
        z = self.proj(x); o, _ = self.core(z); return self.head(o[:, -1]), (o, z)


ARCHS = {"gru": GRUBase, "lstm": LSTMBase, "bag": BagBase, "AT": ArchAT, "C": ArchC}


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


class SeqMemory:
    """GDumb over windows (established policy)."""

    def __init__(self, cap=600):
        self.cap = cap; self.x = []; self.y = []

    def add(self, x, y, rng):
        for i in range(len(x)):
            if len(self.x) < self.cap:
                self.x.append(x[i]); self.y.append(int(y[i]))
        # gdumb rebalance
        if len(self.x) >= self.cap:
            yy = np.array(self.y); cls = np.unique(yy); per = max(1, self.cap // len(cls)); keep = []
            for c in cls:
                idx = np.where(yy == c)[0]; keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
            keep = keep[:self.cap]; self.x = [self.x[i] for i in keep]; self.y = [self.y[i] for i in keep]

    def sample(self, n, rng):
        if not self.x:
            return None
        idx = rng.choice(len(self.x), min(n, len(self.x)), replace=False)
        return torch.stack([self.x[i] for i in idx]), torch.tensor([self.y[i] for i in idx])

    def size(self):
        return len(self.x)
