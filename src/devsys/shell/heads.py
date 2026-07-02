"""Task heads on frozen latents. A plain classifier, a probabilistic (Gaussian) head
that emits mean+logvar so we can separate epistemic from aleatoric uncertainty and run
calibration (the C-cluster prerequisite for E4 / noisy-TV), and the e7 sparse head family
(k-WTA, gated MoE) promoted into the shell (WP-02) so DR2/PR3, WS5, CM4 and the routing
metrics build against one implementation instead of per-script copies. The sparse heads
are only meaningful against a PARAM-MATCHED dense head (shell/capmatch or
moe_expert_hidden_for_dense); interference reduction must never be bought capacity.
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


class KWTAHead(nn.Module):
    """latent -> hidden (GELU, top-k winner-take-all mask) -> classes. Identical shape (and so
    identical param count) to a dense two-layer head; sparsity of ACTIVATION is the only change,
    which is exactly what makes the dense arm a genuine capacity-matched control."""

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
    """A softmax-gated mixture of small expert MLPs (dim -> expert_hidden -> classes) plus a linear
    router. The router's per-sample gate distribution is kept on `last_gates` so routing_entropy is
    readable after any forward (the C1-C3 routing metrics). Expert width should be solved with
    moe_expert_hidden_for_dense (or capmatch) so the total matches the dense reference."""

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
    """Mean entropy (nats) of a per-sample gate distribution [B, E]. High and flat: the router never
    specialized (a corpus null signature); dropping over a stream: modules carved up the tasks."""
    p = gates.clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean())


def moe_expert_hidden_for_dense(dim: int, hidden: int, n_classes: int, n_experts: int) -> int:
    """Per-expert width so an MoEHead (router + experts) matches a dense dim->hidden->classes head's
    param count. Dense = dim*hidden + hidden + hidden*n_classes + n_classes; router = dim*E + E; each
    expert = w*(dim + n_classes + 1) + n_classes. Solve for w, floor at 1 (the e7 matching rule)."""
    dense = dim * hidden + hidden + n_classes * hidden + n_classes
    router = dim * n_experts + n_experts
    per_expert_budget = max(1.0, (dense - router) / n_experts)
    w = (per_expert_budget - n_classes) / (dim + n_classes + 1.0)
    return max(1, int(round(w)))


def build_head(cfg, dim: int, n_classes: int, out_dim: int | None = None):
    """Probabilistic head iff cfg.probabilistic, regressing a latent of size out_dim; else a
    classifier of n_classes."""
    if bool(cfg.probabilistic) and out_dim is not None:
        return GaussianHead(dim, out_dim, hidden=int(cfg.hidden))
    return ClassHead(dim, n_classes, hidden=int(cfg.hidden))
