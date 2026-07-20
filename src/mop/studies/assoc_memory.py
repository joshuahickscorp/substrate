from __future__ import annotations

import torch

from ..seeding import seed_everything


def _store(k: int, dim: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(k, dim, generator=gen)
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def _corrupt(patterns: torch.Tensor, level: float, gen: torch.Generator) -> torch.Tensor:
    q = patterns + level * torch.randn(patterns.shape, generator=gen)
    return q / (q.norm(dim=1, keepdim=True) + 1e-8)


def hopfield_retrieve(x_store: torch.Tensor, query: torch.Tensor, beta: float, steps: int) -> torch.Tensor:
    x = query
    for _ in range(steps):
        sim = beta * (x @ x_store.t())  # [B, K]
        x = torch.softmax(sim, dim=1) @ x_store  # [B, D]
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    return x


def ff_autoassociator(x_store: torch.Tensor, queries: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    q = queries  # [K, D] one corrupted exemplar per stored pattern
    d = x_store.shape[1]
    a = q.t() @ q + ridge * torch.eye(d)  # [D, D]
    w = torch.linalg.solve(a, q.t() @ x_store)  # least-squares W s.t. q W ~= x_store
    out = q @ w
    return out / (out.norm(dim=1, keepdim=True) + 1e-8)


def _retrieval_acc(retrieved: torch.Tensor, targets: torch.Tensor, thresh: float) -> float:
    cos_self = (retrieved * targets).sum(dim=1)  # both unit-norm
    sims = retrieved @ targets.t()  # [B, K]
    nearest = sims.argmax(dim=1)
    correct = (cos_self > thresh) & (nearest == torch.arange(targets.shape[0]))
    return float(correct.float().mean())


def _capacity(curve: list[dict], floor: float) -> float:
    ok = [pt["load"] for pt in curve if pt["acc"] >= floor]
    return max(ok) if ok else 0.0


def assoc_memory(
    dim: int = 64,
    loads: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0),
    betas: tuple[float, ...] = (8.0, 16.0, 32.0),
    corruption: float = 0.3,
    steps: int = 3,
    thresh: float = 0.9,
    floor: float = 0.5,
    seed: int = 0,
    toy: bool = True,
) -> dict:
    seed_everything(seed)
    gen = torch.Generator().manual_seed(seed)
    kmax = 6 * dim if toy else 32 * dim

    hop_curve: list[dict] = []
    ff_curve: list[dict] = []
    for load in loads:
        k = max(2, min(int(round(load * dim)), kmax))
        x_store = _store(k, dim, gen)
        queries = _corrupt(x_store, corruption, gen)

        best = -1.0
        best_beta = betas[0]
        for beta in betas:
            r = hopfield_retrieve(x_store, queries, beta=beta, steps=steps)
            acc = _retrieval_acc(r, x_store, thresh)
            if acc > best:
                best, best_beta = acc, beta
        hop_curve.append({"load": float(load), "k": k, "acc": best, "beta": best_beta})

        ff = ff_autoassociator(x_store, queries)
        ff_curve.append({"load": float(load), "k": k, "acc": _retrieval_acc(ff, x_store, thresh)})

    hop_cap = _capacity(hop_curve, floor)
    ff_cap = _capacity(ff_curve, floor)
    return {
        "dim": dim,
        "corruption": corruption,
        "steps": steps,
        "thresh": thresh,
        "floor": floor,
        "hopfield": {"capacity": hop_cap, "curve": hop_curve},
        "feedforward": {"capacity": ff_cap, "curve": ff_curve},
        "hopfield_wins": hop_cap > ff_cap,
        "provenance": "provisional",
    }


def main() -> int:
    import json

    out = assoc_memory()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
