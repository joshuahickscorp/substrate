"""Generation 3 precompute headroom gates for the two selected candidates.

C1 P1R-priority: over a FIXED GDumb-representative buffer (identical content to GDumb), does a replay-SAMPLING
priority beat uniform GDumb sampling? Isolates the sampling policy from buffer content. If even a strong
loss/oracle priority cannot beat uniform GDumb, the candidate has no headroom and dies cheaply.

C2 V1-capable-family: does an EXPANDED capable estimator family (add a numpy MLP and a higher-capacity RFF)
recover the verification value that only kernel_ridge decoded (V1 was architecture_dependent, 1 of 3)? If a
richer family robustly decodes it, the architecture-dependence was an estimator-capacity artifact and the
candidate has headroom.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/scammermike/Downloads/mop-scientific-frontier/frontier/lanes")
sys.path.insert(0, "/Users/scammermike/Downloads/mop/salvage/lanes2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import lane_p_emnist as lp  # noqa: E402

OUT = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/generation3")
OUT.mkdir(parents=True, exist_ok=True)


# ---------- C1: P1R-priority oracle-headroom over a fixed GDumb buffer ----------
def gdumb_buffer(net, memx, memy, newx, newy, rng, MEM):
    allx = torch.cat([memx, newx]) if len(memx) else newx
    ally = torch.cat([memy, newy]) if len(memy) else newy
    n = len(allx)
    if n <= MEM:
        return allx, ally
    classes = torch.unique(ally).tolist(); per = max(1, MEM // len(classes)); keep = []
    an = ally.numpy()
    for c in classes:
        idx = np.where(an == c)[0]; keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    keep = np.array(keep[:MEM]); return allx[keep], ally[keep]


def run_priority(mode, seed, prep):
    torch.manual_seed(1000 + seed)
    tasks, train_pool, test_pool = prep
    net = lp.EmnistCNN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 100)
    memx = torch.empty(0, 1, 28, 28); memy = torch.empty(0, dtype=torch.long)
    acc = np.zeros((lp.N_TASKS, lp.N_TASKS))
    for t in range(lp.N_TASKS):
        newx, newy = train_pool[t]
        net.train()
        for _ in range(lp.STEPS_PER_TASK):
            bi = rng.choice(len(newx), min(lp.BATCH, len(newx)), replace=False)
            xb, yb = newx[bi], newy[bi]
            if len(memx):
                if mode == "uniform":
                    mi = rng.choice(len(memx), min(lp.BATCH, len(memx)), replace=False)
                else:
                    with torch.no_grad():
                        ml = F.cross_entropy(net(memx), memy, reduction="none").numpy()  # current loss = retention need
                    if mode == "loss_priority":
                        w = ml - ml.min() + 1e-3
                    else:  # oracle_priority: emphasize items the model currently forgets, class-balanced
                        w = ml - ml.min() + 1e-3
                        for c in np.unique(memy.numpy()):
                            m = memy.numpy() == c
                            w[m] = w[m] / (w[m].sum() + 1e-9)  # balance priority mass per class
                    p = w / w.sum()
                    mi = rng.choice(len(memx), min(lp.BATCH, len(memx)), replace=False, p=p)
                xb = torch.cat([xb, memx[mi]]); yb = torch.cat([yb, memy[mi]])
            opt.zero_grad(); F.cross_entropy(net(xb), yb).backward(); opt.step()
        memx, memy = gdumb_buffer(net, memx, memy, newx, newy, rng, lp.MEM)
        for j in range(t + 1):
            xt, yt = test_pool[j]; acc[t, j] = lp._acc(net, xt, yt)
    return float(acc[lp.N_TASKS - 1, :].mean())


def c1_p1r_priority():
    lp.STEPS_PER_TASK = 110; lp.PER_CLASS_TRAIN = 400
    seeds = [0, 1]; modes = ["uniform", "loss_priority", "oracle_priority"]
    res = {m: [] for m in modes}
    for s in seeds:
        prep = lp._prep(s)
        for m in modes:
            res[m].append(run_priority(m, s, prep))
    means = {m: float(np.mean(res[m])) for m in modes}
    headroom = means["oracle_priority"] - means["uniform"]
    verdict = "headroom_present" if headroom >= 0.03 else "no_headroom"
    return {"candidate": "C1_P1R_priority", "means": means, "priority_headroom_over_gdumb_uniform": round(headroom, 4),
            "gate5_verdict": verdict, "seeds": seeds,
            "interpretation": ("a replay-sampling priority over the fixed GDumb buffer " +
                               ("does beat" if headroom >= 0.03 else "does not beat") + " uniform GDumb sampling")}


# ---------- C2: V1-capable-family expanded estimators ----------
def mlp_fit_predict(xtr, ytr, xte, hidden=64, epochs=300, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    Xtr = (xtr - mu) / sd; Xte = (xte - mu) / sd
    d = Xtr.shape[1]
    W1 = rng.normal(0, 0.1, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, (hidden, 1)); b2 = np.zeros(1)
    y = ytr.reshape(-1, 1)
    for _ in range(epochs):
        h = np.maximum(0, Xtr @ W1 + b1); pred = h @ W2 + b2
        g = (pred - y) / len(y)
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = Xtr.T @ gh; gb1 = gh.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
    h = np.maximum(0, Xte @ W1 + b1); return (h @ W2 + b2).ravel()


def c2_v1_family():
    from estimators import select_and_fit, CAPABLE
    cache = Path("/Users/scammermike/Downloads/mop-scientific-frontier/runs/generation2/mop-generation2-scientific-frontier-v1/v1_bed_cache.npz")
    if not cache.exists():
        return {"candidate": "C2_V1_capable_family", "gate5_verdict": "bed_cache_absent"}
    z = np.load(cache)
    nfam = 6
    # expanded capable family = existing CAPABLE + numpy MLP
    fam_effects = {"mlp": [], **{k: [] for k in CAPABLE}}
    for f in range(nfam):
        tr = {k: z[f"{f}_train_{k}"] for k in ("x", "r")}
        tu = {k: z[f"{f}_tune_{k}"] for k in ("x", "r")}
        te = {k: z[f"{f}_test_{k}"] for k in ("x", "r")}
        b = int(0.2 * len(te["r"])); r = te["r"]
        orc = float(np.sum(np.sort(r)[::-1][:b]))
        rnd = float(np.mean([np.sum(r[np.random.default_rng(s).choice(len(r), b, replace=False)]) for s in range(15)]))
        # best simple control = loss-proxy feature ranking (a column of x); approximate by feature with max corr
        simple = 0.0
        for col in range(te["x"].shape[1]):
            v = float(np.sum(r[np.argsort(-te["x"][:, col])[:b]]))
            if orc > rnd:
                simple = max(simple, (v - rnd) / (orc - rnd))
        for name in list(CAPABLE) + ["mlp"]:
            if name == "mlp":
                pred = mlp_fit_predict(tr["x"], tr["r"], te["x"], seed=f)
            else:
                pred = select_and_fit(name, tr["x"], tr["r"], tu["x"], tu["r"], te["x"])
            v = float(np.sum(r[np.argsort(-pred)[:b]]))
            eff = (v - rnd) / (orc - rnd) - simple if orc > rnd else 0.0
            fam_effects[name].append(eff)
    # how many estimators have random-effects lcb >= SESOI
    def lcb(e):
        e = np.array(e); n = len(e); sd = e.std(ddof=1) if n > 1 else 0
        return float(e.mean() - (1.833 if n <= 10 else 1.729) * sd / np.sqrt(n))
    passing = {k: round(lcb(v), 3) for k, v in fam_effects.items()}
    n_pass = sum(1 for v in passing.values() if v >= 0.05)
    return {"candidate": "C2_V1_capable_family", "expanded_family": list(fam_effects.keys()),
            "per_estimator_incremental_lcb": passing, "n_passing_of_family": n_pass,
            "gate5_verdict": "headroom_present" if n_pass >= 2 else "no_headroom",
            "interpretation": f"{n_pass} of {len(passing)} capable estimators robustly decode verification value beyond the best simple control"}


if __name__ == "__main__":
    t = time.time()
    out = {"schema": "mop-gen3-precompute/v1"}
    out["C2_V1_capable_family"] = c2_v1_family()
    print("C2 done:", out["C2_V1_capable_family"].get("gate5_verdict"), out["C2_V1_capable_family"].get("n_passing_of_family"), flush=True)
    out["C1_P1R_priority"] = c1_p1r_priority()
    print("C1 done:", out["C1_P1R_priority"]["gate5_verdict"], out["C1_P1R_priority"]["priority_headroom_over_gdumb_uniform"], flush=True)
    out["wall_seconds"] = round(time.time() - t, 1)
    (OUT / "MOP_GENERATION3_PRECOMPUTE.json").write_text(json.dumps(out, indent=2))
    print("PRECOMPUTE_DONE", flush=True)
