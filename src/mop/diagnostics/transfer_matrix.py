
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..seeding import seed_everything


def _fit_linear_head(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, seed: int) -> nn.Linear:
    seed_everything(seed)
    n_classes = int(y.max()) + 1
    head = nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()
    return head


def transfer_matrix(
    tasks: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 150, lr: float = 0.05, seed: int = 0
) -> dict:
    t = len(tasks)
    heads = [_fit_linear_head(x, y, epochs, lr, seed + i) for i, (x, y) in enumerate(tasks)]
    grid = [[0.0] * t for _ in range(t)]
    with torch.no_grad():
        for i, head in enumerate(heads):
            for j, (xj, yj) in enumerate(tasks):
                acc = float((head(xj).argmax(-1) == yj).float().mean())
                grid[i][j] = round(acc, 4)
    off_diag = [grid[i][j] for i in range(t) for j in range(t) if i != j]
    diag = [grid[i][i] for i in range(t)]
    chance = 1.0 / int(max(int(y.max()) + 1 for _, y in tasks))
    asym = 0.0
    pairs = 0
    for i in range(t):
        for j in range(i + 1, t):
            asym += abs(grid[i][j] - grid[j][i])
            pairs += 1
    asymmetry_index = round(asym / pairs, 4) if pairs else 0.0
    mean_off_diag = round(sum(off_diag) / len(off_diag), 4) if off_diag else 0.0
    return {
        "transfer_matrix": grid,
        "diag_mean": round(sum(diag) / t, 4) if t else 0.0,
        "off_diag_mean": mean_off_diag,
        "chance": round(chance, 4),
        "asymmetry_index": asymmetry_index,
        "cross_task_structure": bool(mean_off_diag > chance + 0.1),
    }
