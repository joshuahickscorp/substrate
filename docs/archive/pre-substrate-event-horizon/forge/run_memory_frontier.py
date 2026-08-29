"""Memory-capacity frontier + Architecture E (dual-memory compression) on HAR raw. Tests whether any BOUNDED
representation (reservoir, GDumb, coreset, prototype, dual recent+prototype) recovers a material fraction of
the full-memory continual gap at matched stored-item budget. This is compression, NOT learned retrieval
(R1/P1R closed). House style: no dashes."""

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
import torch.nn.functional as F  # noqa: E402
from temporal_core import LSTMBase, count_params  # noqa: E402
import run_temporal_domain as RT  # noqa: E402 (HAR loader + tasks_for)

REP = "/Users/scammermike/Downloads/mop-substrate-forge/forge/reports"
SEEDS = [0, 1, 2, 3, 4]
STEPS = 160
BUDGETS = [40, 150, 375, 750]        # stored-item budgets (windows)
POLICIES = ["reservoir", "gdumb", "coreset", "prototype", "dual"]


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_memory(policy, allx, ally, cap, rng):
    """Return (mx, my) memory of <= cap items under the policy. Compression policies may synthesize items."""
    allx = torch.stack(allx); ally = torch.tensor(ally); n = len(allx); yy = ally.numpy()
    if policy in ("reservoir",):
        keep = rng.choice(n, min(cap, n), replace=False); return allx[keep], ally[keep]
    if policy == "gdumb":
        cls = np.unique(yy); per = max(1, cap // len(cls)); keep = []
        for c in cls:
            idx = np.where(yy == c)[0]; keep.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
        keep = np.array(keep[:cap]); return allx[keep], ally[keep]
    if policy == "coreset":
        # farthest-point (k-center greedy) in flattened-signal space, class-balanced
        cls = np.unique(yy); per = max(1, cap // len(cls)); keep = []
        flat = allx.reshape(n, -1).numpy()
        for c in cls:
            idx = np.where(yy == c)[0]
            if len(idx) <= per:
                keep.extend(idx.tolist()); continue
            sel = [int(rng.choice(idx))]; d = np.linalg.norm(flat[idx] - flat[sel[0]], axis=1)
            while len(sel) < per:
                j = int(idx[np.argmax(d)]); sel.append(j); d = np.minimum(d, np.linalg.norm(flat[idx] - flat[j], axis=1))
            keep.extend(sel)
        keep = np.array(keep[:cap]); return allx[keep], ally[keep]
    if policy == "prototype":
        # K synthetic class-mean windows per class (compressed): cluster each class into K means
        cls = np.unique(yy); K = max(1, cap // len(cls)); mx = []; my = []
        for c in cls:
            idx = np.where(yy == c)[0]; xc = allx[idx]
            if len(idx) <= K:
                mx.append(xc); my.extend([c] * len(idx)); continue
            # simple k-means on flattened signals
            flat = xc.reshape(len(idx), -1).numpy(); cen = flat[rng.choice(len(idx), K, replace=False)]
            for _ in range(5):
                a = np.argmin(((flat[:, None] - cen[None]) ** 2).sum(-1), 1)
                cen = np.stack([flat[a == k].mean(0) if (a == k).any() else cen[k] for k in range(K)])
            proto = torch.tensor(cen.reshape(K, *allx.shape[1:]).astype(np.float32)); mx.append(proto); my.extend([c] * K)
        return torch.cat(mx), torch.tensor(my)
    if policy == "dual":
        # Architecture E: half recent exact cache + half prototype compression
        half = cap // 2
        rmx, rmy = allx[-half:], ally[-half:]
        pmx, pmy = build_memory("prototype", [allx[i] for i in range(n)], yy.tolist(), cap - half, rng)
        return torch.cat([rmx, pmx]), torch.cat([rmy, pmy])
    return allx[:cap], ally[:cap]


def run(policy, cap, seed, full=False, nomem=False):
    torch.manual_seed(1000 + seed); tasks, ch, nout = RT.tasks_for(seed)
    m = LSTMBase(ch, nout); opt = torch.optim.Adam(m.parameters(), 1e-3); rng = np.random.default_rng(seed + 100)
    seen_x, seen_y = [], []; nT = len(tasks); acc = np.zeros((nT, nT)); mx = my = None
    for t in range(nT):
        x, y = tasks[t]["x"], tasks[t]["y"]; m.train()
        for _ in range(STEPS):
            bi = rng.choice(len(x), min(64, len(x)), replace=False); xb, yb = x[bi], y[bi]
            if mx is not None and not nomem:
                mi = rng.choice(len(mx), min(64, len(mx)), replace=False); xb = torch.cat([xb, mx[mi]]); yb = torch.cat([yb, my[mi]])
            opt.zero_grad(); F.cross_entropy(m(xb)[0], yb).backward(); opt.step()
        for i in range(len(x)):
            seen_x.append(x[i]); seen_y.append(int(y[i]))
        if not nomem:
            if full:
                mx, my = torch.stack(seen_x), torch.tensor(seen_y)
            else:
                mx, my = build_memory(policy, seen_x, seen_y, cap, rng)
        m.eval()
        for j in range(t + 1):
            xt, yt = tasks[j]["test"]
            with torch.no_grad():
                pred = torch.cat([m(xt[k:k + 256])[0].argmax(1) for k in range(0, len(xt), 256)])
            acc[t, j] = float((pred == yt).float().mean())
    return float(acc[nT - 1].mean())


def main():
    t0 = time.time()
    none_m = float(np.mean([run("none", 0, s, nomem=True) for s in SEEDS]))
    full_m = float(np.mean([run("gdumb", 0, s, full=True) for s in SEEDS]))
    gap = full_m - none_m
    frontier = {}
    for cap in BUDGETS:
        frontier[cap] = {}
        for p in POLICIES:
            vals = [run(p, cap, s) for s in SEEDS]
            mean = float(np.mean(vals)); frac = (mean - none_m) / gap if gap > 0 else 0.0
            frontier[cap][p] = {"avg_final": round(mean, 4), "frac_of_full_gap": round(frac, 4)}
            print(f"  cap{cap:4d} {p:10s} avg_final={mean:.4f} frac_gap={frac:.3f}", flush=True)
    # does any compression beat GDumb at matched budget?
    best_over_gdumb = {}
    for cap in BUDGETS:
        g = frontier[cap]["gdumb"]["avg_final"]
        best = max(POLICIES, key=lambda p: frontier[cap][p]["avg_final"])
        best_over_gdumb[cap] = {"best_policy": best, "best_avg_final": frontier[cap][best]["avg_final"],
                                "gdumb_avg_final": g, "compression_beats_gdumb": round(frontier[cap][best]["avg_final"] - g, 4)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-substrate-forge").stdout.strip()
    out = {"schema": "mop-substrate-memory-frontier/v1", "domain": "HAR_raw", "source_commit": commit, "seeds": SEEDS,
           "no_memory": round(none_m, 4), "full_memory": round(full_m, 4), "full_gap": round(gap, 4),
           "budgets_windows": BUDGETS, "frontier": frontier, "best_over_gdumb_per_budget": best_over_gdumb,
           "primary_question": "can bounded compression recover a material fraction of the full-memory gap and beat GDumb at matched budget",
           "wall_seconds": round(time.time() - t0, 1)}
    out["sha256"] = sha(out)
    open(f"{REP}/MOP_SUBSTRATE_MEMORY_FRONTIER.json", "w").write(json.dumps(out, indent=2))
    print(f"[frontier] none={none_m:.3f} full={full_m:.3f} gap={gap:.3f} | best_over_gdumb={best_over_gdumb} [{out['wall_seconds']}s]", flush=True)
    print("MEMORY_FRONTIER_DONE", flush=True)


if __name__ == "__main__":
    main()
