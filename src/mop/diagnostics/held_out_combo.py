from __future__ import annotations

import torch

from .linear_probe import linear_probe
from .substrate_ablation import frozen_random_projection


def factorized_latents(
    n_a: int = 3, n_b: int = 3, per_cell: int = 40, dim: int = 64, interaction: float = 0.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    cA = torch.randn(n_a, dim, generator=g) * 2.0
    cB = torch.randn(n_b, dim, generator=g) * 2.0
    cAB = torch.randn(n_a, n_b, dim, generator=g) * 2.0
    xs, ya, yb = [], [], []
    for a in range(n_a):
        for b in range(n_b):
            base = cA[a] + cB[b] + interaction * cAB[a, b]
            x = base.unsqueeze(0) + 0.5 * torch.randn(per_cell, dim, generator=g)
            xs.append(x)
            ya += [a] * per_cell
            yb += [b] * per_cell
    return torch.cat(xs), torch.tensor(ya), torch.tensor(yb)


def held_out_combination(
    x: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, held_b_for_a=None, seed: int = 0
) -> dict:
    n_a = int(y_a.max()) + 1
    n_b = int(y_b.max()) + 1
    held = held_b_for_a or {a: a % n_b for a in range(n_a)}
    is_held = torch.tensor([held.get(int(a)) == int(b) for a, b in zip(y_a, y_b, strict=True)])
    xtr, ytr = x[~is_held], y_a[~is_held]
    xte, yte = x[is_held], y_a[is_held]
    if xte.shape[0] < 4 or xtr.shape[0] < 8:
        return {"error": "not enough samples for the held-out split"}
    seen = linear_probe(xtr, ytr, seed=seed)
    head = torch.nn.Linear(x.shape[1], n_a)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    import torch.nn.functional as F

    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(head(xtr.float()), ytr).backward()
        opt.step()
    with torch.no_grad():
        ho = float((head(xte.float()).argmax(-1) == yte).float().mean())
    chance = 1.0 / n_a
    return {
        "seen_acc": round(seen["score"], 4),
        "heldout_acc": round(ho, 4),
        "gap": round(seen["score"] - ho, 4),
        "chance": round(chance, 4),
        "compositional": bool(ho > chance + 0.15),
        "n_held": int(xte.shape[0]),
    }


def compositionality_report(
    n_a: int = 3, n_b: int = 3, dim: int = 64, interaction: float = 0.0, seed: int = 0
) -> dict:
    x, ya, yb = factorized_latents(n_a, n_b, dim=dim, interaction=interaction, seed=seed)
    real = held_out_combination(x, ya, yb, seed=seed)
    fr = held_out_combination(frozen_random_projection(x.float(), seed), ya, yb, seed=seed)
    return {
        "interaction": interaction,
        "real": real,
        "frozen_random": fr,
        "compositional_above_floor": bool(
            real.get("heldout_acc", 0) > fr.get("heldout_acc", 0) + 0.1 and real.get("compositional")
        ),
    }
