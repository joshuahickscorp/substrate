from __future__ import annotations

import torch
import torch.nn.functional as F

from ..seeding import seed_everything
from .linear_probe import linear_probe
from .substrate_ablation import frozen_random_projection


def nonlinear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    hidden: int = 64,
    epochs: int = 150,
    lr: float = 0.02,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict:
    seed_everything(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm].float(), y[perm]
    cut = int(n * (1 - test_frac))
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]
    nc = int(y.max()) + 1
    head = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, nc)
    )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(xte).argmax(-1) == yte).float().mean())
    return {"metric": "accuracy", "score": acc, "chance": 1.0 / nc, "decodable": acc > 1.0 / nc + 0.1}


def readout_contribution(x: torch.Tensor, y: torch.Tensor, hidden: int = 64, seed: int = 0) -> dict:
    xr = x.float()
    xf = frozen_random_projection(xr, seed)
    lr_real = linear_probe(xr, y, seed=seed)["score"]
    nl_real = nonlinear_probe(xr, y, hidden=hidden, seed=seed)["score"]
    lr_fr = linear_probe(xf, y, seed=seed)["score"]
    nl_fr = nonlinear_probe(xf, y, hidden=hidden, seed=seed)["score"]
    gain_real = nl_real - lr_real
    gain_fr = nl_fr - lr_fr
    idx = gain_real - gain_fr
    return {
        "linear_real": round(lr_real, 4),
        "nonlinear_real": round(nl_real, 4),
        "linear_frozen_random": round(lr_fr, 4),
        "nonlinear_frozen_random": round(nl_fr, 4),
        "nonlinear_gain_real": round(gain_real, 4),
        "nonlinear_gain_frozen_random": round(gain_fr, 4),
        "readout_contribution_index": round(idx, 4),
        "substrate_carries_nonlinear_structure": bool(idx > 0.05),
    }
