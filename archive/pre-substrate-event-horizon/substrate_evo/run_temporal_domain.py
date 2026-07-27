"""Temporal moldability comparison on HAR raw sequences: A-T and C substrates vs converged GRU/LSTM/bag
baselines, continual over activity subsets, subject-disjoint test, GDumb replay, 5 seeds, cost-adjusted utility.
House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-autonomous-substrate-evolution/substrate_evo")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from temporal_core import ARCHS, SeqMemory, count_params  # noqa: E402

REP = "/Users/scammermike/Downloads/mop-autonomous-substrate-evolution/substrate_evo/reports"
HARD = "/Users/scammermike/Downloads/mop-gen3/runs/generation3/mop-generation3-discovery-v1/data/UCI HAR Dataset"
CH = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
SEEDS = [0, 1, 2, 3, 4]
STEPS = 180
AUXW = 0.3
_D = {}


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load():
    if "d" in _D:
        return _D["d"]
    def ld(sp):
        X = np.stack([np.loadtxt(f"{HARD}/{sp}/Inertial Signals/{c}_{sp}.txt") for c in CH], -1).astype(np.float32)
        y = np.loadtxt(f"{HARD}/{sp}/y_{sp}.txt").astype(int) - 1
        return X, y
    Xtr, ytr = ld("train"); Xte, yte = ld("test")
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    _D["d"] = (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte)); return _D["d"]


def tasks_for(seed, cpt=2):
    Xtr, ytr, Xte, yte = load(); rng = np.random.default_rng(seed); order = rng.permutation(6)
    tasks = []
    for t in range(6 // cpt):
        cs = order[t * cpt:(t + 1) * cpt]
        itr = np.concatenate([np.where(ytr.numpy() == c)[0] for c in cs]); ite = np.concatenate([np.where(yte.numpy() == c)[0] for c in cs])
        tasks.append({"x": Xtr[itr], "y": ytr[itr], "test": (Xte[ite], yte[ite])})
    return tasks, 9, 6


def aux_loss(internals, arch_name):
    if internals is None:
        return 0.0
    o, z = internals  # o: (B,T,L) hidden; z: (B,T,L) projected
    # multi-horizon future-latent prediction: predict z[t+h] from o[t]
    T = o.shape[1]; loss = 0.0; n = 0
    for h in (4, 8, 16):
        if T - h <= 0:
            continue
        loss = loss + F.mse_loss(o[:, :T - h], z[:, h:].detach()); n += 1
    return loss / max(1, n)


def run_arm(arch, seed, steps=STEPS):
    torch.manual_seed(1000 + seed); tasks, ch, nout = tasks_for(seed)
    model = ARCHS[arch](ch, nout); opt = torch.optim.Adam(model.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100); mem = SeqMemory(cap=600)
    nT = len(tasks); acc = np.zeros((nT, nT))
    for t in range(nT):
        x, y = tasks[t]["x"], tasks[t]["y"]; model.train()
        for _ in range(steps):
            bi = rng.choice(len(x), min(64, len(x)), replace=False); xb, yb = x[bi], y[bi]
            if mem.size():
                s = mem.sample(64, rng)
                if s:
                    xb = torch.cat([xb, s[0]]); yb = torch.cat([yb, s[1]])
            out = model(xb, update_medium=True) if arch == "AT" else model(xb)
            logits, internals = out
            loss = F.cross_entropy(logits, yb)
            if arch in ("AT", "C"):
                loss = loss + AUXW * aux_loss(internals, arch)
            opt.zero_grad(); loss.backward(); opt.step()
        mem.add(x, y, rng)
        model.eval()
        for j in range(t + 1):
            xt, yt = tasks[j]["test"]
            with torch.no_grad():
                pred = torch.cat([model(xt[s:s + 256])[0].argmax(1) for s in range(0, len(xt), 256)])
            acc[t, j] = float((pred == yt).float().mean())
    final = acc[nT - 1]
    return {"avg_final": float(final.mean()), "retention": float(final[:nT - 1].mean()),
            "new": float(np.mean([acc[t, t] for t in range(nT)])), "params": count_params(model)}


ARMS = ["gru", "lstm", "bag", "AT", "C"]
BASELINES = ["gru", "lstm", "bag"]
COST = 0.05


def re(e):
    e = np.asarray(e, float); n = len(e); sd = e.std(ddof=1) if n > 1 else 0
    return {"mean": round(float(e.mean()), 4), "lower_95_cb": round(float(e.mean() - 1.833 * sd / np.sqrt(n)), 4), "n": n}


def main():
    t0 = time.time(); per = {a: [] for a in ARMS}
    for s in SEEDS:
        for a in ARMS:
            per[a].append(run_arm(a, s)); print(f"  seed{s} {a:4s} avg_final={per[a][-1]['avg_final']:.4f}", flush=True)
    means = {a: {k: float(np.mean([r[k] for r in per[a]])) for k in per[a][0]} for a in ARMS}
    gp = means["gru"]["params"]
    def util(a, si): return per[a][si]["avg_final"] - COST * (per[a][si]["params"] / gp)
    um = {a: float(np.mean([util(a, si) for si in range(len(SEEDS))])) for a in ARMS}
    bb = max(BASELINES, key=lambda a: um[a])
    def eff(a):
        e = [util(a, si) - util(bb, si) for si in range(len(SEEDS))]
        floors = (means[a]["retention"] >= means[bb]["retention"] - 0.02 and means[a]["new"] >= means[bb]["new"] - 0.02)
        return re(e), floors
    eAT, fAT = eff("AT"); eC, fC = eff("C")
    posAT = eAT["lower_95_cb"] >= 0.05 and fAT; posC = eC["lower_95_cb"] >= 0.05 and fC
    cls = ("both_positive" if posAT and posC else "AT_positive" if posAT else "C_positive" if posC else "substrate_candidate_null")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-autonomous-substrate-evolution").stdout.strip()
    out = {"schema": "mop-substrate-temporal-domain-result/v1", "domain": "HAR_raw_inertial", "source_commit": commit,
           "seeds": SEEDS, "steps": STEPS, "arm_means": {a: {k: round(v, 4) for k, v in means[a].items()} for a in ARMS},
           "util_mean": {a: round(v, 4) for a, v in um.items()}, "best_baseline": bb,
           "AT_effect_vs_best_baseline": eAT, "AT_floors_ok": fAT, "C_effect_vs_best_baseline": eC, "C_floors_ok": fC,
           "classification": cls, "SESOI": 0.05, "wall_seconds": round(time.time() - t0, 1)}
    out["sha256"] = sha(out)
    open(f"{REP}/MOP_SUBSTRATE_TEMPORAL_DOMAIN_RESULT.json", "w").write(json.dumps(out, indent=2))
    print(f"[HAR_raw] {cls} | best_baseline={bb}({um[bb]:.3f}) AT_util={um['AT']:.3f}(lcb {eAT['lower_95_cb']}) "
          f"C_util={um['C']:.3f}(lcb {eC['lower_95_cb']}) [{out['wall_seconds']}s]", flush=True)
    print("TEMPORAL_DOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
