from __future__ import annotations

import torch
import torch.nn.functional as F

from ..seeding import seed_everything


def linear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    classification: bool = True,
    epochs: int = 200,
    lr: float = 0.05,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict:
    seed_everything(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm], y[perm]
    cut = int(n * (1 - test_frac))
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]

    if classification:
        n_classes = int(y.max()) + 1
        head = torch.nn.Linear(x.shape[1], n_classes)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()
        with torch.no_grad():
            acc = float((head(xte).argmax(-1) == yte).float().mean())
        return {
            "metric": "accuracy",
            "score": acc,
            "chance": 1.0 / n_classes,
            "decodable": acc > 1.0 / n_classes + 0.1,
        }

    out = y.shape[1] if y.dim() > 1 else 1
    head = torch.nn.Linear(x.shape[1], out)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    ytr2, yte2 = ytr.float().view(cut, -1), yte.float().view(n - cut, -1)
    for _ in range(epochs):
        opt.zero_grad()
        F.mse_loss(head(xtr), ytr2).backward()
        opt.step()
    with torch.no_grad():
        pred = head(xte)
        ss_res = ((pred - yte2) ** 2).sum()
        ss_tot = ((yte2 - yte2.mean(0)) ** 2).sum() + 1e-8
        r2 = float(1 - ss_res / ss_tot)
    return {"metric": "r2", "score": r2, "chance": 0.0, "decodable": r2 > 0.1}
