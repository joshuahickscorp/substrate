from __future__ import annotations

import torch
import torch.nn.functional as F

from ..seeding import seed_everything


def _task(dim: int, classes: int, samples: int, sep: float, seed: int):
    from ..substrate.datasets import make_task_stream

    t = make_task_stream(
        n_tasks=1,
        dim=dim,
        classes_per_task=classes,
        samples_per_task=samples,
        separation=sep,
        incremental="class",
        seed=seed,
    )[0]
    return t.x, t.y, int(t.y.max()) + 1


def _prep(x, y, test_frac=0.3, seed=0):
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    g = torch.Generator().manual_seed(seed)  # own generator: split is independent of global RNG
    perm = torch.randperm(x.shape[0], generator=g)
    x, y = x[perm], y[perm]
    cut = int(x.shape[0] * (1 - test_frac))
    return x[:cut], y[:cut], x[cut:], y[cut:]


def _acc(o, y):
    return float((o.argmax(-1) == y).float().mean())


def _init_weights(d: int, hidden: int, depth: int, classes: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    dims = [d] + [hidden] * depth + [classes]
    return [torch.randn(dims[i + 1], dims[i], generator=g) / dims[i] ** 0.5 for i in range(len(dims) - 1)]


def _train_backprop(xtr, ytr, xte, yte, W0, depth, epochs, lr):
    Ws = [w.clone().requires_grad_(True) for w in W0]
    opt = torch.optim.Adam(Ws, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        a = xtr
        for li in range(depth):
            a = torch.tanh(a @ Ws[li].T)
        o = a @ Ws[depth].T
        F.cross_entropy(o, ytr).backward()
        opt.step()

    @torch.no_grad()
    def fwd(xb):
        a = xb
        for li in range(depth):
            a = torch.tanh(a @ Ws[li].T)
        return a @ Ws[depth].T

    return _acc(fwd(xte), yte), _acc(fwd(xtr), ytr)


def _train_pc(xtr, ytr, xte, yte, W0, depth, classes, epochs, lr, infer, infer_lr=0.1):
    Ws = [w.clone() for w in W0]
    f = torch.tanh

    def fprime(z):
        return 1 - torch.tanh(z) ** 2

    T = F.one_hot(ytr, classes).float()
    n = xtr.shape[0]
    L = depth + 1  # number of weight layers

    for _ in range(epochs):
        v = [xtr]
        for li in range(depth):
            v.append(f(v[-1] @ Ws[li].T))
        v.append(v[-1] @ Ws[depth].T)  # output value node
        v = [vi.clone() for vi in v]

        for _ in range(infer):  # relax hidden value nodes 1..depth toward fixed point
            zpre = [v[li] @ Ws[li].T for li in range(L)]  # pre-activations into v_{l+1}
            mu = [f(zpre[li]) if li < depth else zpre[li] for li in range(L)]
            eps = [v[li + 1] - mu[li] for li in range(L)]
            eps_out = T - torch.softmax(v[L], -1)  # label-driven top error (clamped target)
            for li in range(1, L):  # update hidden value nodes only
                top_err = eps[li] if li < depth else eps_out  # readout error is label-driven
                top = (top_err @ Ws[li]) * fprime(zpre[li - 1])  # backward error onto v[li]
                dv = -eps[li - 1] + top
                v[li] = v[li] + infer_lr * dv

        zpre = [v[li] @ Ws[li].T for li in range(L)]
        mu = [f(zpre[li]) if li < depth else zpre[li] for li in range(L)]
        eps = [v[li + 1] - mu[li] for li in range(L)]
        eps[L - 1] = T - torch.softmax(v[L], -1)  # readout trained against the label error
        for li in range(L):
            pre = v[li]
            post_err = eps[li] * (fprime(zpre[li]) if li < depth else 1.0)
            Ws[li] = Ws[li] + lr * (post_err.T @ pre) / n

    @torch.no_grad()
    def fwd(xb):
        a = xb
        for li in range(depth):
            a = f(a @ Ws[li].T)
        return a @ Ws[depth].T

    return _acc(fwd(xte), yte), _acc(fwd(xtr), ytr)


def pc_depth(
    depths: tuple[int, ...] = (1, 2, 3),
    toy: bool = True,
    dim: int = 32,
    classes: int = 8,
    hidden: int = 16,
    seed: int = 0,
) -> dict:
    samples = 400 if toy else 1200
    bp_epochs = 200 if toy else 400
    pc_epochs = 200 if toy else 400
    infer = 8 if toy else 16

    seed_everything(seed)
    x, y, C = _task(dim, classes, samples, sep=0.7, seed=seed)
    xtr, ytr, xte, yte = _prep(x, y, seed=seed)
    chance = 1.0 / C

    per_depth = []
    for depth in depths:
        W0 = _init_weights(dim, hidden, depth, C, seed=seed + depth)
        bp_te, _ = _train_backprop(xtr, ytr, xte, yte, W0, depth, bp_epochs, lr=0.05)
        pc_te, _ = _train_pc(xtr, ytr, xte, yte, W0, depth, C, pc_epochs, lr=0.05, infer=infer)
        per_depth.append({"depth": int(depth), "bp_acc": bp_te, "pc_acc": pc_te, "gap": bp_te - pc_te})

    gaps = [d["gap"] for d in per_depth]
    monotone = all(gaps[i + 1] >= gaps[i] - 1e-9 for i in range(len(gaps) - 1))
    gap_widens = len(gaps) >= 2 and monotone and gaps[-1] > gaps[0]

    return {
        "per_depth": per_depth,
        "gap_widens_with_depth": bool(gap_widens),
        "chance": chance,
        "depths": [int(d) for d in depths],
        "provenance": "provisional",
    }


def main() -> dict:
    return pc_depth(depths=(1, 2, 3, 4), toy=True)
