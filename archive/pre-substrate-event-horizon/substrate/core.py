"""Owned Substrate v0 core: shared contract + two architectures + memory + policies + continual engine.

Both architectures own their trainable observation projection, latent state, and slow representation. Frozen or
raw providers only supply observations. Four timescales: immediate (fast per-sequence state), episodic (bounded
replay memory + local stats), slow (owned representation parameters, updated under eligibility), and policy
(consolidation / eligibility / routing schedule, strong SIMPLE rules by default).

Binding null constraints (Gen2/Gen3): active replay is an established policy (GDumb/reservoir/uniform), NOT a
learned selector; no learned retrieval, simulation, or plasticity controller by default. Complexity is never
credited as capability; all extra compute and memory is charged. House style: no dashes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 128
FAST = 64


# ---------------- owned components ----------------
class OwnedProjection(nn.Module):
    """Trainable projection from a (possibly frozen) observation into owned substrate space. Timescale: slow."""

    def __init__(self, obs_dim, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, latent), nn.ReLU(), nn.Linear(latent, latent))

    def forward(self, x):
        return self.net(x)


class TaskHeads(nn.Module):
    def __init__(self, latent, n_out):
        super().__init__()
        self.head = nn.Linear(latent, n_out)

    def forward(self, z):
        return self.head(z)


# ---------------- Architecture A: Shared Latent Workspace ----------------
class ArchA(nn.Module):
    """One shared workspace: owned projection -> shared slow MLP -> fast GRU core -> shared head.

    Timescales: fast = GRU hidden (resets per sequence); slow = projection + workspace params; episodic and
    policy handled by the engine. Context routing is a simple context-conditioned gain on the workspace.
    """

    name = "A_shared_latent_workspace"

    def __init__(self, obs_dim, n_out, n_ctx=8):
        super().__init__()
        self.proj = OwnedProjection(obs_dim)
        self.workspace = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT))
        self.fast = nn.GRUCell(LATENT, FAST)
        self.ctx_gain = nn.Embedding(n_ctx, LATENT)  # simple context-conditioned routing
        self.to_head = nn.Linear(FAST + LATENT, LATENT)
        self.heads = TaskHeads(LATENT, n_out)
        self.n_ctx = n_ctx
        self.param_groups = {"proj": list(self.proj.parameters()), "workspace": list(self.workspace.parameters()),
                             "fast": list(self.fast.parameters()), "ctx": list(self.ctx_gain.parameters()),
                             "head": list(self.to_head.parameters()) + list(self.heads.parameters())}

    def init_fast(self, b):
        return torch.zeros(b, FAST)

    def forward(self, x, ctx, h=None):
        z = self.proj(x)
        z = z * torch.sigmoid(self.ctx_gain(ctx))          # context routing (simple, owned)
        z = self.workspace(z) + z
        if h is None:
            h = self.init_fast(x.shape[0])
        h = self.fast(z, h)
        rep = F.relu(self.to_head(torch.cat([h, z], 1)))
        return self.heads(rep), h, rep


# ---------------- Architecture B: Sparse Modular Plastic Substrate ----------------
class ArchB(nn.Module):
    """K bounded latent modules with local slow params and a simple sparse context router.

    Timescales: fast = per-module hidden mix (engine-managed); slow = module params; policy = routing/eligibility.
    Top-k sparse routing gives conditional compute and specialization; dormant modules are preallocated.
    """

    name = "B_sparse_modular_substrate"

    def __init__(self, obs_dim, n_out, n_modules=6, topk=2, n_ctx=8):
        super().__init__()
        self.proj = OwnedProjection(obs_dim)
        self.modules_ = nn.ModuleList([nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(),
                                                     nn.Linear(LATENT, LATENT)) for _ in range(n_modules)])
        self.router = nn.Sequential(nn.Linear(LATENT, 64), nn.ReLU(), nn.Linear(64, n_modules))
        self.fast = nn.GRUCell(LATENT, FAST)
        self.to_head = nn.Linear(FAST + LATENT, LATENT)
        self.heads = TaskHeads(LATENT, n_out)
        self.n_modules = n_modules; self.topk = topk
        self.last_route = None
        self.param_groups = {"proj": list(self.proj.parameters()), "router": list(self.router.parameters()),
                             "fast": list(self.fast.parameters()),
                             "head": list(self.to_head.parameters()) + list(self.heads.parameters())}
        for i, m in enumerate(self.modules_):
            self.param_groups[f"module_{i}"] = list(m.parameters())

    def init_fast(self, b):
        return torch.zeros(b, FAST)

    def forward(self, x, ctx, h=None):
        z = self.proj(x)
        logits = self.router(z)
        w = torch.softmax(logits, 1)
        topv, topi = w.topk(self.topk, 1)
        topv = topv / (topv.sum(1, keepdim=True) + 1e-9)
        self.last_route = topi.detach()
        out = torch.zeros_like(z)
        for k in range(self.topk):
            for m in range(self.n_modules):
                mask = (topi[:, k] == m)
                if mask.any():
                    out[mask] += topv[mask, k:k + 1] * self.modules_[m](z[mask])
        z2 = out + z
        if h is None:
            h = self.init_fast(x.shape[0])
        h = self.fast(z2, h)
        rep = F.relu(self.to_head(torch.cat([h, z2], 1)))
        return self.heads(rep), h, rep


def make_arch(name, obs_dim, n_out):
    return ArchA(obs_dim, n_out) if name.startswith("A") else ArchB(obs_dim, n_out)


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ---------------- episodic memory (established policies only) ----------------
class Memory:
    def __init__(self, cap=1000, policy="gdumb"):
        self.cap = cap; self.policy = policy; self.x = []; self.y = []; self.c = []; self.seen = 0

    def add(self, x, y, ctx, rng):
        for i in range(len(x)):
            self.seen += 1
            if self.policy == "none":
                return
            if len(self.x) < self.cap:
                self.x.append(x[i]); self.y.append(int(y[i])); self.c.append(int(ctx[i]))
            elif self.policy == "reservoir":
                j = int(rng.integers(0, self.seen))
                if j < self.cap:
                    self.x[j] = x[i]; self.y[j] = int(y[i]); self.c[j] = int(ctx[i])
            elif self.policy == "recent":
                self.x.pop(0); self.y.pop(0); self.c.pop(0)
                self.x.append(x[i]); self.y.append(int(y[i])); self.c.append(int(ctx[i]))
            # gdumb rebalancing handled in rebalance()

    def rebalance(self, rng):
        if self.policy != "gdumb" or len(self.x) < self.cap:
            return
        y = np.array(self.y); classes = np.unique(y); per = max(1, self.cap // len(classes)); keep = []
        for cc in classes:
            idx = np.where(y == cc)[0]; keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
        keep = keep[:self.cap]
        self.x = [self.x[i] for i in keep]; self.y = [self.y[i] for i in keep]; self.c = [self.c[i] for i in keep]

    def sample(self, n, rng):
        if not self.x:
            return None
        idx = rng.choice(len(self.x), min(n, len(self.x)), replace=False)
        return (torch.stack([self.x[i] for i in idx]), torch.tensor([self.y[i] for i in idx]),
                torch.tensor([self.c[i] for i in idx]))

    def size(self):
        return len(self.x)
