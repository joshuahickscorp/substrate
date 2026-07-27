"""PAMAP2 second temporal domain: window continuous IMU streams, subject-disjoint, run the temporal-order gate
(does order matter) and a continual forgetting/headroom preflight. House style: no dashes."""

from __future__ import annotations

import glob
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

REP = "/Users/scammermike/Downloads/mop-substrate-forge/forge/reports"
PROTO = "/Users/scammermike/Downloads/mop-autonomous-substrate-evolution/runs/substrate/mop-autonomous-substrate-evolution-v1/data/pamap2/PAMAP2_Dataset/Protocol"
# acc16 + gyro from hand/chest/ankle IMUs = 18 channels
COLS = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46]
WIN, STRIDE = 128, 64
SEEDS = [0, 1, 2, 3, 4]
_C = {}


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_windows():
    if "w" in _C:
        return _C["w"]
    X, Y, S = [], [], []
    for f in sorted(glob.glob(f"{PROTO}/subject*.dat")):
        sid = int(f.split("subject")[1][:3]); d = np.loadtxt(f)
        act = d[:, 1].astype(int); sig = d[:, COLS]
        # forward-fill NaNs
        for c in range(sig.shape[1]):
            col = sig[:, c]; m = np.isnan(col)
            if m.any():
                idx = np.where(~m)[0]
                if len(idx):
                    col[m] = np.interp(np.where(m)[0], idx, col[idx])
                sig[:, c] = col
        sig = np.nan_to_num(sig)
        for st in range(0, len(sig) - WIN, STRIDE):
            a = act[st:st + WIN]
            if a[0] == 0 or not (a == a[0]).all():
                continue  # drop transient/mixed windows
            X.append(sig[st:st + WIN]); Y.append(int(a[0])); S.append(sid)
    X = np.array(X, np.float32); Y = np.array(Y); S = np.array(S)
    # keep the most common activities, relabel 0..K-1
    acts, counts = np.unique(Y, return_counts=True); keep = acts[np.argsort(-counts)[:6]]
    m = np.isin(Y, keep); X, Y, S = X[m], Y[m], S[m]
    remap = {a: i for i, a in enumerate(sorted(keep))}; Y = np.array([remap[y] for y in Y])
    mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-6; X = (X - mu) / sd
    _C["w"] = (torch.tensor(X), torch.tensor(Y), S, len(keep)); return _C["w"]


class GRU(nn.Module):
    def __init__(s, ch, no):
        super().__init__(); s.g = nn.GRU(ch, 64, batch_first=True); s.h = nn.Linear(64, no)

    def forward(s, x):
        o, _ = s.g(x); return s.h(o[:, -1])


class Bag(nn.Module):
    def __init__(s, ch, no):
        super().__init__(); s.n = nn.Sequential(nn.Linear(ch * 3, 64), nn.ReLU(), nn.Linear(64, no))

    def forward(s, x):
        return s.n(torch.cat([x.mean(1), x.std(1), x.amax(1)], 1))


def train_eval(Model, tr_idx, te_idx, X, Y, ch, no, seed, transform=None):
    torch.manual_seed(seed); m = Model(ch, no); opt = torch.optim.Adam(m.parameters(), 1e-3); rng = np.random.default_rng(seed)
    xt = X[tr_idx] if transform is None else transform(X[tr_idx], rng)
    yt = Y[tr_idx]
    for _ in range(300):
        bi = rng.choice(len(xt), min(128, len(xt)), replace=False)
        opt.zero_grad(); F.cross_entropy(m(xt[bi]), yt[bi]).backward(); opt.step()
    m.eval(); xe = X[te_idx] if transform is None else transform(X[te_idx], rng)
    with torch.no_grad():
        acc = (torch.cat([m(xe[k:k + 256]).argmax(1) for k in range(0, len(xe), 256)]) == Y[te_idx]).float().mean().item()
    return acc


def shuffle_t(X, rng):
    return X[:, torch.tensor(rng.permutation(X.shape[1])), :]


def main():
    t0 = time.time(); X, Y, S, no = load_windows(); ch = X.shape[2]
    subs = np.unique(S); rng0 = np.random.default_rng(0)
    # subject-disjoint split: ~70% train subjects
    tr_subs = set(rng0.choice(subs, int(0.7 * len(subs)), replace=False).tolist())
    tr_idx = np.where(np.isin(S, list(tr_subs)))[0]; te_idx = np.where(~np.isin(S, list(tr_subs)))[0]
    gc = [train_eval(GRU, tr_idx, te_idx, X, Y, ch, no, s) for s in SEEDS]
    sh = [train_eval(GRU, tr_idx, te_idx, X, Y, ch, no, s, shuffle_t) for s in SEEDS]
    bg = [train_eval(Bag, tr_idx, te_idx, X, Y, ch, no, s) for s in SEEDS]
    gc, sh, bg = map(np.array, (gc, sh, bg))
    th = float((gc - bg).mean()); th_lcb = float((gc - bg).mean() - 1.833 * (gc - bg).std(ddof=1) / np.sqrt(len(SEEDS)))
    om = float((gc - sh).mean())
    verdict = "temporal_headroom_present" if (th_lcb >= 0.03 and om > 0.02) else "invalid_no_temporal_headroom"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-substrate-forge").stdout.strip()
    out = {"schema": "mop-substrate-pamap2-preflight/v1", "domain": "PAMAP2", "source_commit": commit,
           "n_windows": int(len(X)), "n_channels": ch, "n_classes": no, "n_subjects": int(len(subs)),
           "units": "subject-disjoint", "seeds": SEEDS,
           "gru_correct": round(float(gc.mean()), 4), "gru_shuffled": round(float(sh.mean()), 4), "bag_order_free": round(float(bg.mean()), 4),
           "temporal_headroom_gru_vs_bag": round(th, 4), "temporal_headroom_lcb": round(th_lcb, 4),
           "order_matters_gru_vs_shuffled": round(om, 4), "verdict": verdict, "wall_seconds": round(time.time() - t0, 1)}
    out["sha256"] = sha(out)
    open(f"{REP}/MOP_SUBSTRATE_PAMAP2_REPORT.json", "w").write(json.dumps(out, indent=2))
    print(f"[PAMAP2] {verdict} | gru {gc.mean():.3f} bag {bg.mean():.3f} shuffled {sh.mean():.3f} "
          f"th_lcb {th_lcb:.3f} order {om:.3f} | {len(X)} windows {ch}ch {no}cls {len(subs)}subj [{out['wall_seconds']}s]", flush=True)
    print("PAMAP2_DONE", flush=True)


if __name__ == "__main__":
    main()
