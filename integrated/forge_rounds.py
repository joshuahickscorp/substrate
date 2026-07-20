"""Remaining Forge science on the valid HAR bed: Architecture F (Predictive Consolidation), A-T improvement
rounds (state attribution, interference-localized updates), timescale ablations, against LSTM+GDumb.

Reuses the integrated substrate kernel. 5 seeds, cost-adjusted, lower-95pct-CB decisions. A tie is a null.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-substrate-forge/substrate_evo")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from temporal_core import LATENT, OwnedProj, SeqMemory, count_params  # noqa: E402
import run_temporal_domain as RT  # noqa: E402

OUT = "/Users/scammermike/Downloads/mop-substrate-forge/integrated"
SEEDS = [0, 1, 2, 3, 4]
STEPS = 160
FASTH = 64


def sha_obj(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class LSTMBaseline(nn.Module):
    def __init__(s, ch, no):
        super().__init__(); s.proj = OwnedProj(ch); s.core = nn.LSTM(LATENT, LATENT, batch_first=True)
        s.head = nn.Linear(LATENT, no)

    def forward(s, x):
        o, _ = s.core(s.proj(x)); return s.head(o[:, -1]), None


class ATAttrib(nn.Module):
    """A-T1: fast / medium / slow readouts kept separate so each timescale is independently removable."""

    def __init__(s, ch, no, use_fast=True, use_medium=True, use_slow=True):
        super().__init__(); s.proj = OwnedProj(ch); s.fast = nn.GRU(LATENT, FASTH, batch_first=True)
        s.slow = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, LATENT))
        s.rf, s.rm, s.rs = nn.Linear(FASTH, no), nn.Linear(LATENT, no), nn.Linear(LATENT, no)
        s.register_buffer("medium", torch.zeros(LATENT))
        s.use = (use_fast, use_medium, use_slow)
        s.param_groups = {"proj": list(s.proj.parameters()), "fast": list(s.fast.parameters()),
                          "slow": list(s.slow.parameters()),
                          "heads": list(s.rf.parameters()) + list(s.rm.parameters()) + list(s.rs.parameters())}

    def forward(s, x, update_medium=True):
        z = s.proj(x); o, _ = s.fast(z); last = o[:, -1]
        if update_medium and s.training:
            with torch.no_grad():
                s.medium.mul_(0.95).add_(0.05 * z[:, -1].mean(0))
        logits = 0
        if s.use[0]:
            logits = logits + s.rf(last)
        if s.use[1]:
            logits = logits + s.rm(s.medium.unsqueeze(0).expand(x.shape[0], -1))
        if s.use[2]:
            logits = logits + s.rs(s.slow(z[:, -1]))
        return logits, (o, z)


class FPredCons(nn.Module):
    """Architecture F: predictive consolidation. Slow state consolidates future-predictive summaries."""

    def __init__(s, ch, no, objective="multi_horizon"):
        super().__init__(); s.proj = OwnedProj(ch); s.core = nn.GRU(LATENT, LATENT, batch_first=True)
        s.pred = nn.Linear(LATENT, LATENT); s.head = nn.Linear(LATENT * 2, no)
        s.register_buffer("consolidated", torch.zeros(LATENT))
        s.objective = objective
        s.param_groups = {"proj": list(s.proj.parameters()), "core": list(s.core.parameters()),
                          "pred": list(s.pred.parameters()), "head": list(s.head.parameters())}

    def forward(s, x, update_medium=True):
        z = s.proj(x); o, _ = s.core(z); last = o[:, -1]
        if update_medium and s.training:
            with torch.no_grad():  # consolidate predictive summary, not raw activity
                s.consolidated.mul_(0.95).add_(0.05 * torch.tanh(s.pred(last)).mean(0))
        cons = s.consolidated.unsqueeze(0).expand(x.shape[0], -1)
        return s.head(torch.cat([last, cons], 1)), (o, z)


def aux_loss(internals, objective, rng):
    if internals is None or objective == "none":
        return 0.0
    o, z = internals; T = o.shape[1]
    if objective == "shuffled_time":
        perm = torch.tensor(rng.permutation(T))
        z = z[:, perm]
    horizons = (4,) if objective == "next_step" else (4, 8, 16)
    loss, n = 0.0, 0
    for h in horizons:
        if T - h > 0:
            loss = loss + F.mse_loss(o[:, :T - h], z[:, h:].detach()); n += 1
    return loss / max(1, n)


def run_arm(kind, seed, steps=STEPS, objective="multi_horizon", eligibility="all", use=(True, True, True)):
    torch.manual_seed(1000 + seed); tasks, ch, nout = RT.tasks_for(seed)
    if kind == "lstm":
        model = LSTMBaseline(ch, nout)
    elif kind == "AT":
        model = ATAttrib(ch, nout, *use)
    else:
        model = FPredCons(ch, nout, objective)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100); mem = SeqMemory(cap=600)
    nT = len(tasks); acc = np.zeros((nT, nT)); prev = None
    for t in range(nT):
        x, y = tasks[t]["x"], tasks[t]["y"]; model.train()
        for _ in range(steps):
            bi = rng.choice(len(x), min(64, len(x)), replace=False); xb, yb = x[bi], y[bi]
            if mem.size():
                sp = mem.sample(64, rng)
                if sp:
                    xb = torch.cat([xb, sp[0]]); yb = torch.cat([yb, sp[1]])
            out = model(xb) if kind == "lstm" else model(xb, update_medium=True)
            logits, internals = out
            loss = F.cross_entropy(logits, yb)
            if kind == "F":
                loss = loss + 0.3 * aux_loss(internals, objective, rng)
            opt.zero_grad(); loss.backward()
            # A-T2: interference-localized slow updates. Freeze the slow group when its gradient direction
            # conflicts with the previous task's accumulated direction.
            if eligibility == "localized" and prev is not None and hasattr(model, "param_groups"):
                g = torch.cat([p.grad.flatten() for p in model.param_groups["slow"] if p.grad is not None])
                if g.numel() and torch.dot(g, prev) < 0:
                    for p in model.param_groups["slow"]:
                        p.grad = None
            opt.step()
        if eligibility == "localized" and hasattr(model, "param_groups"):
            gs = [p.grad.flatten() for p in model.param_groups["slow"] if p.grad is not None]
            prev = torch.cat(gs).detach() if gs else prev
        mem.add(x, y, rng); model.eval()
        for j in range(t + 1):
            xt, yt = tasks[j]["test"]
            with torch.no_grad():
                pred = torch.cat([(model(xt[k:k + 256])[0] if kind == "lstm"
                                   else model(xt[k:k + 256], update_medium=False)[0]).argmax(1)
                                  for k in range(0, len(xt), 256)])
            acc[t, j] = float((pred == yt).float().mean())
    fin = acc[nT - 1]
    return {"avg_final": float(fin.mean()), "retention": float(fin[:nT - 1].mean()),
            "new": float(np.mean([acc[t, t] for t in range(nT)])), "params": count_params(model)}


ARMS = {
    "lstm_gdumb": dict(kind="lstm"),
    "AT1_attrib": dict(kind="AT"),
    "AT1_no_fast": dict(kind="AT", use=(False, True, True)),
    "AT1_no_medium": dict(kind="AT", use=(True, False, True)),
    "AT1_no_slow": dict(kind="AT", use=(True, True, False)),
    "AT2_localized": dict(kind="AT", eligibility="localized"),
    "F_multi_horizon": dict(kind="F", objective="multi_horizon"),
    "F_next_step": dict(kind="F", objective="next_step"),
    "F_no_objective": dict(kind="F", objective="none"),
    "F_shuffled_time": dict(kind="F", objective="shuffled_time"),
}
COST = 0.05


def lcb(e):
    e = np.asarray(e, float); n = len(e); sd = e.std(ddof=1) if n > 1 else 0.0
    return float(e.mean() - 1.833 * sd / np.sqrt(n))


def main():
    t0 = time.time(); per = {a: [] for a in ARMS}
    for s in SEEDS:
        for a, kw in ARMS.items():
            per[a].append(run_arm(seed=s, **kw))
        print(f"  seed{s} done", flush=True)
    means = {a: {k: float(np.mean([r[k] for r in per[a]])) for k in per[a][0]} for a in ARMS}
    bp = means["lstm_gdumb"]["params"]

    def util(a, i):
        return per[a][i]["avg_final"] - COST * (per[a][i]["params"] / bp)

    um = {a: float(np.mean([util(a, i) for i in range(len(SEEDS))])) for a in ARMS}
    base = "lstm_gdumb"

    def eff(a):
        e = [util(a, i) - util(base, i) for i in range(len(SEEDS))]
        return {"mean": round(float(np.mean(e)), 4), "lower_95_cb": round(lcb(e), 4)}

    results = {a: {"means": {k: round(v, 4) for k, v in means[a].items()}, "util": round(um[a], 4),
                   "effect_vs_lstm_gdumb": eff(a)} for a in ARMS}
    # timescale value: removing a timescale should cost material accuracy if it contributes
    ts = {name: round(um["AT1_attrib"] - um[f"AT1_no_{name}"], 4) for name in ("fast", "medium", "slow")}
    verdicts = {
        "AT1_state_attribution": "null" if results["AT1_attrib"]["effect_vs_lstm_gdumb"]["lower_95_cb"] < 0.05 else "positive",
        "AT2_interference_localized": "null" if results["AT2_localized"]["effect_vs_lstm_gdumb"]["lower_95_cb"] < 0.05 else "positive",
        "F_predictive_consolidation": "null" if max(results[a]["effect_vs_lstm_gdumb"]["lower_95_cb"]
                                                    for a in ("F_multi_horizon", "F_next_step")) < 0.05 else "positive",
        "timescale_contributions": ts,
        "decorative_timescales": [k for k, v in ts.items() if v <= 0.01],
    }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-substrate-forge").stdout.strip()
    doc = {"schema": "mop-integrated-forge-rounds/v1", "bed": "HAR_raw (valid temporal, order matters)",
           "source_commit": commit, "seeds": SEEDS, "steps": STEPS, "baseline": base,
           "results": results, "verdicts": verdicts, "SESOI": 0.05,
           "wall_seconds": round(time.time() - t0, 1)}
    doc["sha256"] = sha_obj(doc)
    open(f"{OUT}/MOP_INTEGRATED_FORGE_ROUNDS.json", "w").write(json.dumps(doc, indent=2))
    print("verdicts:", json.dumps(verdicts))
    print("util:", {a: round(um[a], 3) for a in um})
    print("ROUNDS_DONE", flush=True)


if __name__ == "__main__":
    main()
