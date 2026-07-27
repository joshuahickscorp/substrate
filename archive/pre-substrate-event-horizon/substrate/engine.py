"""Shared continual train/eval engine + strong baselines + moldability metrics. One matched CL loop for every
model (substrate A/B and baselines) so parameters, updates, memory, replay, and data order are matched.
House style: no dashes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import ArchA, ArchB, Memory, OwnedProjection, count_params

LATENT = 128


# ---------------- baselines ----------------
class MLPBaseline(nn.Module):
    name = "baseline_mlp"

    def __init__(self, obs_dim, n_out, n_ctx=8):
        super().__init__()
        self.proj = OwnedProjection(obs_dim)
        self.body = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT), nn.ReLU())
        self.head = nn.Linear(LATENT, n_out)
        self.param_groups = {"all": list(self.parameters())}

    def init_fast(self, b):
        return None

    def forward(self, x, ctx, h=None):
        z = self.body(self.proj(x))
        return self.head(z), None, z


class GRUBaseline(nn.Module):
    name = "baseline_gru"

    def __init__(self, obs_dim, n_out, n_ctx=8):
        super().__init__()
        self.proj = OwnedProjection(obs_dim)
        self.gru = nn.GRUCell(LATENT, LATENT)
        self.head = nn.Linear(LATENT, n_out)
        self.param_groups = {"all": list(self.parameters())}

    def init_fast(self, b):
        return torch.zeros(b, LATENT)

    def forward(self, x, ctx, h=None):
        z = self.proj(x)
        if h is None:
            h = self.init_fast(x.shape[0])
        h = self.gru(z, h)
        return self.head(h), h, h


MODELS = {"A": ArchA, "B": ArchB, "mlp": MLPBaseline, "gru": GRUBaseline}


def make(name, obs_dim, n_out):
    return MODELS[name](obs_dim, n_out)


# ---------------- continual engine ----------------
def _eligible_params(model, policy, frozen):
    """Return the parameter list eligible to update under the policy (default all)."""
    if policy in ("all", None):
        return [p for p in model.parameters() if p.requires_grad]
    if policy == "fixed_group":  # freeze projection after task 0 (simple consolidation of slow input map)
        keep = []
        for gname, ps in model.param_groups.items():
            if gname == "proj" and frozen:
                continue
            keep += ps
        return keep
    return [p for p in model.parameters() if p.requires_grad]


def run_stream(model_name, tasks, cfg, seed, obs_dim, n_out):
    """tasks: list of dicts {x,y,ctx (train)}; plus tasks[i]['test'] = (x,y,ctx). cfg controls policies.
    Returns acc_matrix, plus substrate metrics."""
    torch.manual_seed(1000 + seed)
    model = make(model_name, obs_dim, n_out)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100)
    mem = Memory(cap=cfg.get("mem_cap", 1000), policy=cfg.get("memory", "gdumb"))
    nT = len(tasks); acc = np.zeros((nT, nT))
    BATCH = 64; steps = cfg.get("steps", 120); replay_b = cfg.get("replay_batch", 64)
    ewc_lambda = cfg.get("ewc_lambda", 0.0); star = {}; fisher = {}
    changed_frac = []; consolidations = 0; routes = []
    n_params = count_params(model)
    updates = 0
    for t in range(nT):
        x, y, ctx = tasks[t]["x"], tasks[t]["y"], tasks[t]["ctx"]
        frozen = (t > 0 and cfg.get("consolidation") in ("boundary", "fixed"))
        elig = _eligible_params(model, cfg.get("eligibility", "all"), frozen)
        before = {id(p): p.detach().clone() for p in elig}
        model.train()
        for _ in range(steps):
            bi = rng.choice(len(x), min(BATCH, len(x)), replace=False)
            xb, yb, cb = x[bi], y[bi], ctx[bi]
            if mem.size() and cfg.get("memory") != "none":
                s = mem.sample(replay_b, rng)
                if s is not None:
                    xb = torch.cat([xb, s[0]]); yb = torch.cat([yb, s[1]]); cb = torch.cat([cb, s[2]])
            out, _, _ = model(xb, cb)
            loss = F.cross_entropy(out, yb)
            if ewc_lambda > 0 and star:
                for n_, p in model.named_parameters():
                    if n_ in star:
                        loss = loss + 0.5 * ewc_lambda * (fisher[n_] * (p - star[n_]) ** 2).sum()
            opt.zero_grad(); loss.backward()
            if cfg.get("eligibility") not in ("all", None):
                elig_ids = {id(p) for p in elig}
                for p in model.parameters():
                    if p.grad is not None and id(p) not in elig_ids:
                        p.grad = None
            opt.step(); updates += 1
            if model_name == "B" and getattr(model, "last_route", None) is not None:
                routes.append(model.last_route.numpy())
        # episodic memory update (established policy)
        mem.add(x, y, ctx, rng); mem.rebalance(rng)
        # consolidation
        if cfg.get("consolidation") in ("boundary", "fixed", "perf") and ewc_lambda > 0:
            star = {n_: p.detach().clone() for n_, p in model.named_parameters()}
            fisher = _estimate_fisher(model, x, y, ctx, rng)
            consolidations += 1
        # measure changed fraction
        after = {id(p): p.detach() for p in elig}
        num = sum(float((after[k] - before[k]).abs().sum()) for k in before)
        den = sum(float(before[k].abs().sum()) + 1e-9 for k in before)
        changed_frac.append(num / den)
        # evaluate retention
        for j in range(t + 1):
            xt, yt, ct = tasks[j]["test"]
            acc[t, j] = _acc(model, xt, ct, yt)
    metrics = {"n_params": n_params, "updates": updates, "mem_size": mem.size(),
               "changed_fraction_mean": float(np.mean(changed_frac)) if changed_frac else 0.0,
               "consolidations": consolidations,
               "route_entropy": float(_route_entropy(routes)) if routes else None}
    return acc, metrics, model, tasks


def _estimate_fisher(model, x, y, ctx, rng, n=128):
    fisher = {n_: torch.zeros_like(p) for n_, p in model.named_parameters()}
    idx = rng.choice(len(x), min(n, len(x)), replace=False)
    for i in idx:
        model.zero_grad()
        out, _, _ = model(x[i:i + 1], ctx[i:i + 1])
        F.cross_entropy(out, y[i:i + 1]).backward()
        for n_, p in model.named_parameters():
            if p.grad is not None:
                fisher[n_] += p.grad.detach() ** 2
    for n_ in fisher:
        fisher[n_] /= max(1, len(idx))
    return fisher


@torch.no_grad()
def _acc(model, x, ctx, y):
    model.eval()
    out = torch.cat([model(x[s:s + 256], ctx[s:s + 256])[0].argmax(1) for s in range(0, len(x), 256)])
    return float((out == y).float().mean())


def _route_entropy(routes):
    r = np.concatenate(routes); counts = np.bincount(r.ravel(), minlength=1) + 1e-9
    p = counts / counts.sum(); return -(p * np.log(p + 1e-9)).sum()


def moldability(acc, future_acc=None):
    """Standard continual-learning metrics from the accuracy matrix."""
    nT = acc.shape[0]
    final = acc[nT - 1, :]
    avg_final = float(final.mean())
    new = float(np.mean([acc[t, t] for t in range(nT)]))
    retention = float(final[:nT - 1].mean()) if nT > 1 else avg_final
    peak = np.array([max(acc[t, j] for t in range(j, nT)) for j in range(nT)])
    forgetting = float(np.mean([peak[j] - final[j] for j in range(nT - 1)])) if nT > 1 else 0.0
    worst = float(final.min())
    return {"avg_final": round(avg_final, 4), "new_task": round(new, 4), "retention": round(retention, 4),
            "forgetting": round(forgetting, 4), "worst_task": round(worst, 4),
            "future_adaptation": round(float(future_acc), 4) if future_acc is not None else None}
