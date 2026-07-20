
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..seeding import seed_everything


@dataclass
class RuleResult:
    name: str
    test_acc: float
    train_acc: float
    seconds: float
    weight_transport: bool  # uses W^T in the backward path (backprop does)
    separate_backward: bool  # needs a distinct backward pass
    local: bool  # updates use only local pre/post activity
    activation_memory: float  # relative: 1.0 = stores all activations (backprop)
    chance: float

    def gap(self, ceiling: float) -> float:
        return ceiling - self.test_acc

    def asdict(self) -> dict:
        return {
            "test_acc": self.test_acc,
            "train_acc": self.train_acc,
            "seconds": self.seconds,
            "weight_transport": self.weight_transport,
            "separate_backward": self.separate_backward,
            "local": self.local,
            "activation_memory": self.activation_memory,
            "chance": self.chance,
        }


def _prep(x, y, test_frac=0.3, seed=0):
    seed_everything(seed)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    perm = torch.randperm(x.shape[0])
    x, y = x[perm], y[perm]
    cut = int(x.shape[0] * (1 - test_frac))
    return x[:cut], y[:cut], x[cut:], y[cut:], int(y.max()) + 1


def _acc(o, y):
    return float((o.argmax(-1) == y).float().mean())


def _relu(z):
    return z.clamp_min(0.0)


def train_backprop(x, y, hidden=64, epochs=200, lr=0.05, seed=0) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, C)
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(net(xtr), ytr).backward()
        opt.step()
    return RuleResult(
        "backprop",
        _acc(net(xte), yte),
        _acc(net(xtr), ytr),
        time.perf_counter() - t0,
        True,
        True,
        False,
        1.0,
        1.0 / C,
    )


def train_feedback_alignment(x, y, hidden=64, epochs=400, lr=0.05, seed=0) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    g = torch.Generator().manual_seed(seed)
    d = xtr.shape[1]
    W1 = torch.randn(hidden, d, generator=g) * (1 / d**0.5)
    W2 = torch.randn(C, hidden, generator=g) * (1 / hidden**0.5)
    b1, b2 = torch.zeros(hidden), torch.zeros(C)
    B = torch.randn(hidden, C, generator=g) * (1 / C**0.5)  # fixed random feedback
    T = F.one_hot(ytr, C).float()
    n = xtr.shape[0]
    t0 = time.perf_counter()
    for _ in range(epochs):
        z1 = xtr @ W1.T + b1
        a1 = _relu(z1)
        o = a1 @ W2.T + b2
        e = torch.softmax(o, -1) - T
        gW2 = e.T @ a1 / n
        gb2 = e.mean(0)
        delta1 = (e @ B.T) * (z1 > 0).float()  # random feedback, no transport
        gW1 = delta1.T @ xtr / n
        gb1 = delta1.mean(0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
    te = _relu(xte @ W1.T + b1) @ W2.T + b2
    tr = _relu(xtr @ W1.T + b1) @ W2.T + b2
    return RuleResult(
        "feedback_alignment",
        _acc(te, yte),
        _acc(tr, ytr),
        time.perf_counter() - t0,
        False,
        True,
        False,
        1.0,
        1.0 / C,
    )


def train_dfa(x, y, hidden=64, epochs=400, lr=0.05, seed=0) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    g = torch.Generator().manual_seed(seed)
    d = xtr.shape[1]
    W1 = torch.randn(hidden, d, generator=g) / d**0.5
    W2 = torch.randn(hidden, hidden, generator=g) / hidden**0.5
    W3 = torch.randn(C, hidden, generator=g) / hidden**0.5
    b1, b2, b3 = torch.zeros(hidden), torch.zeros(hidden), torch.zeros(C)
    B1 = torch.randn(hidden, C, generator=g) / C**0.5
    B2 = torch.randn(hidden, C, generator=g) / C**0.5
    T = F.one_hot(ytr, C).float()
    n = xtr.shape[0]
    t0 = time.perf_counter()

    def fwd(x_):
        z1 = x_ @ W1.T + b1
        a1 = _relu(z1)
        z2 = a1 @ W2.T + b2
        a2 = _relu(z2)
        o = a2 @ W3.T + b3
        return z1, a1, z2, a2, o

    for _ in range(epochs):
        z1, a1, z2, a2, o = fwd(xtr)
        e = torch.softmax(o, -1) - T
        gW3 = e.T @ a2 / n
        d2 = (e @ B2.T) * (z2 > 0).float()  # direct from output
        d1 = (e @ B1.T) * (z1 > 0).float()  # direct from output
        gW2 = d2.T @ a1 / n
        gW1 = d1.T @ xtr / n
        W3 -= lr * gW3
        b3 -= lr * e.mean(0)
        W2 -= lr * gW2
        b2 -= lr * d2.mean(0)
        W1 -= lr * gW1
        b1 -= lr * d1.mean(0)
    _, _, _, _, te = fwd(xte)
    _, _, _, _, tr = fwd(xtr)
    return RuleResult(
        "dfa", _acc(te, yte), _acc(tr, ytr), time.perf_counter() - t0, False, True, False, 0.5, 1.0 / C
    )


def train_forward_forward(x, y, hidden=128, epochs=300, lr=0.03, seed=0) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    d = xtr.shape[1]
    torch.manual_seed(seed)
    L1 = torch.nn.Linear(d + C, hidden)
    L2 = torch.nn.Linear(hidden, hidden)
    opt = torch.optim.Adam(list(L1.parameters()) + list(L2.parameters()), lr=lr)
    theta = 2.0

    def overlay(xb, labels):
        oh = F.one_hot(labels, C).float() * 3.0
        return torch.cat([oh, xb], dim=-1)

    def norm(a):
        return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

    def goodness(layer, inp):
        a = _relu(layer(inp))
        return (a**2).mean(-1), norm(a)

    t0 = time.perf_counter()
    n = xtr.shape[0]
    for _ in range(epochs):
        wrong = (ytr + torch.randint(1, C, (n,))) % C
        pos1 = overlay(xtr, ytr)
        neg1 = overlay(xtr, wrong)
        opt.zero_grad()
        gp1, hp = goodness(L1, pos1)
        gn1, hn = goodness(L1, neg1)
        gp2, _ = goodness(L2, hp.detach())
        gn2, _ = goodness(L2, hn.detach())
        loss = (
            F.softplus(theta - gp1)
            + F.softplus(gn1 - theta)
            + F.softplus(theta - gp2)
            + F.softplus(gn2 - theta)
        ).mean()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def predict(xb):
        scores = []
        for c in range(C):
            lab = torch.full((xb.shape[0],), c)
            g1, h1 = goodness(L1, overlay(xb, lab))
            g2, _ = goodness(L2, h1)
            scores.append(g1 + g2)
        return torch.stack(scores, dim=-1)

    return RuleResult(
        "forward_forward",
        _acc(predict(xte), yte),
        _acc(predict(xtr), ytr),
        time.perf_counter() - t0,
        False,
        False,
        True,
        0.0,
        1.0 / C,
    )


def train_target_prop(x, y, hidden=64, epochs=300, lr=0.05, seed=0) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    d = xtr.shape[1]
    torch.manual_seed(seed)
    W1 = torch.nn.Linear(d, hidden)  # x -> h
    W2 = torch.nn.Linear(hidden, C)  # h -> o
    V = torch.nn.Linear(C, hidden)  # learned inverse o -> h
    oW1 = torch.optim.Adam(W1.parameters(), lr=lr)
    oW2 = torch.optim.Adam(W2.parameters(), lr=lr)
    oV = torch.optim.Adam(V.parameters(), lr=lr)
    T = F.one_hot(ytr, C).float()
    t0 = time.perf_counter()
    for _ in range(epochs):
        with torch.no_grad():
            h = torch.tanh(W1(xtr))
            o = W2(h)
            o_tgt = o - 0.5 * (torch.softmax(o, -1) - T)  # nudge output toward target
            h_tgt = h - torch.tanh(V(o)) + torch.tanh(V(o_tgt))  # difference correction
        oW2.zero_grad()
        F.mse_loss(W2(h.detach()), o_tgt).backward()
        oW2.step()
        oW1.zero_grad()
        F.mse_loss(torch.tanh(W1(xtr)), h_tgt.detach()).backward()
        oW1.step()
        oV.zero_grad()
        F.mse_loss(torch.tanh(V(o.detach())), h.detach()).backward()
        oV.step()  # inverse recon

    @torch.no_grad()
    def fwd(xb):
        return W2(torch.tanh(W1(xb)))

    return RuleResult(
        "target_prop",
        _acc(fwd(xte), yte),
        _acc(fwd(xtr), ytr),
        time.perf_counter() - t0,
        False,
        True,
        True,
        0.5,
        1.0 / C,
    )


def train_equilibrium_prop(x, y, hidden=64, epochs=120, lr=0.05, seed=0, beta=0.5, relax=12) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    g = torch.Generator().manual_seed(seed)
    d = xtr.shape[1]
    W1 = torch.randn(hidden, d, generator=g) / d**0.5  # x -> h
    W2 = torch.randn(C, hidden, generator=g) / hidden**0.5  # h <-> o (symmetric)

    def rho(s):
        return s.clamp(0, 1)

    T = F.one_hot(ytr, C).float()

    def settle(xb, target=None, b=0.0):
        h = torch.zeros(xb.shape[0], hidden)
        o = torch.zeros(xb.shape[0], C)
        for _ in range(relax):
            h = rho(0.5 * (xb @ W1.T + rho(o) @ W2))
            o_drive = rho(h) @ W2.T
            if target is not None:
                o_drive = o_drive + b * (target - o)
            o = rho(o_drive)
        return h, o

    t0 = time.perf_counter()
    n = xtr.shape[0]
    for _ in range(epochs):
        hf, of = settle(xtr)
        hn, on = settle(xtr, T, beta)
        gW2 = (rho(hn).T @ rho(on) - rho(hf).T @ rho(of)).T / (beta * n)
        gW1 = (rho(hn).T @ xtr - rho(hf).T @ xtr) / (beta * n)
        W2 += lr * gW2
        W1 += lr * gW1

    @torch.no_grad()
    def predict(xb):
        _, o = settle(xb)
        return o

    return RuleResult(
        "equilibrium_prop",
        _acc(predict(xte), yte),
        _acc(predict(xtr), ytr),
        time.perf_counter() - t0,
        False,
        False,
        True,
        0.0,
        1.0 / C,
    )


def train_predictive_coding(x, y, hidden=64, epochs=200, lr=0.05, seed=0, infer=20) -> RuleResult:
    xtr, ytr, xte, yte, C = _prep(x, y, seed=seed)
    g = torch.Generator().manual_seed(seed)
    d = xtr.shape[1]
    W1 = torch.randn(hidden, d, generator=g) / d**0.5
    W2 = torch.randn(C, hidden, generator=g) / hidden**0.5
    f = torch.tanh

    def fp(deriv, z):
        return 1 - torch.tanh(z) ** 2 if deriv else torch.tanh(z)

    T = F.one_hot(ytr, C).float()
    t0 = time.perf_counter()
    n = xtr.shape[0]
    for _ in range(epochs):
        z1 = xtr @ W1.T
        h = f(z1)  # init hidden value from feedforward pass
        for _ in range(infer):  # relax hidden value nodes
            o_pred = h @ W2.T
            eps_o = T - torch.softmax(o_pred, -1)
            eps_h = h - f(z1)
            dh = -eps_h + (eps_o @ W2) * fp(True, z1)
            h = h + 0.1 * dh
        o_pred = h @ W2.T
        eps_o = T - torch.softmax(o_pred, -1)
        eps_h = h - f(z1)
        W2 += lr * (eps_o.T @ h) / n
        W1 += lr * ((eps_h * fp(True, z1)).T @ xtr) / n

    @torch.no_grad()
    def fwd(xb):
        return f(xb @ W1.T) @ W2.T

    return RuleResult(
        "predictive_coding",
        _acc(fwd(xte), yte),
        _acc(fwd(xtr), ytr),
        time.perf_counter() - t0,
        False,
        False,
        True,
        0.5,
        1.0 / C,
    )


RULES = {
    "backprop": train_backprop,
    "feedback_alignment": train_feedback_alignment,
    "dfa": train_dfa,
    "forward_forward": train_forward_forward,
    "target_prop": train_target_prop,
    "equilibrium_prop": train_equilibrium_prop,
    "predictive_coding": train_predictive_coding,
}

__all__ = ["RULES", "RuleResult"] + [
    f"train_{k}"
    for k in (
        "backprop",
        "feedback_alignment",
        "dfa",
        "forward_forward",
        "target_prop",
        "equilibrium_prop",
        "predictive_coding",
    )
]
