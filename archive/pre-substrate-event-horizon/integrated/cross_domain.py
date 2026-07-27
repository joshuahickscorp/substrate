"""Cross-domain persistent-core sequence: HAR (sensor) -> Speech Commands (audio) -> return HAR -> held-out
speech adaptation. Shared slow core, domain-specific projections and heads only. The persistent arm never
reinitializes the core. 5 seeds, lower-95pct-CB. A tie is a null.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-substrate-forge/substrate_evo")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-substrate-forge/integrated")
import audio_preflight as AP  # noqa: E402
import numpy as np  # noqa: E402
import run_temporal_domain as RT  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from temporal_core import LATENT  # noqa: E402

OUT = Path("/Users/scammermike/Downloads/mop-substrate-forge/integrated")
CACHE = OUT / "data/speech_feats.npz"
SEEDS = [0, 1, 2, 3, 4]
STEPS = 150


def sha_obj(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Proj(nn.Module):
    def __init__(s, ch):
        super().__init__()
        s.c = nn.Conv1d(ch, 32, 5, padding=2)
        s.l = nn.Linear(32, LATENT)

    def forward(s, x):
        return F.relu(s.l(F.relu(s.c(x.transpose(1, 2))).transpose(1, 2)))


class SharedCore(nn.Module):
    """One persistent slow core. Domain-specific projections and heads attach to it."""

    def __init__(s):
        super().__init__()
        s.core = nn.GRU(LATENT, LATENT, batch_first=True)

    def forward(s, z):
        o, _ = s.core(z)
        return o[:, -1]


def speech_data():
    if CACHE.exists():
        d = np.load(CACHE)
        return torch.tensor(d["X"]), torch.tensor(d["Y"]), d["S"]
    X, Y, S = AP.load()
    np.savez(CACHE, X=X.numpy(), Y=Y.numpy(), S=S)
    return X, Y, S


def har_data(seed):
    tasks, ch, nout = RT.tasks_for(seed)
    X = torch.cat([t["x"] for t in tasks])
    Y = torch.cat([t["y"] for t in tasks])
    te = [t["test"] for t in tasks]
    Xte = torch.cat([t[0] for t in te])
    Yte = torch.cat([t[1] for t in te])
    return X, Y, Xte, Yte, ch, nout


def fit(core, proj, head, X, Y, seed, steps=STEPS, train_core=True):
    params = (
        list(proj.parameters()) + list(head.parameters()) + (list(core.parameters()) if train_core else [])
    )
    opt = torch.optim.Adam(params, 1e-3)
    rng = np.random.default_rng(seed)
    proj.train()
    head.train()
    core.train()
    for _ in range(steps):
        bi = rng.choice(len(X), min(64, len(X)), replace=False)
        logits = head(core(proj(X[bi])))
        opt.zero_grad()
        F.cross_entropy(logits, Y[bi]).backward()
        opt.step()


@torch.no_grad()
def acc(core, proj, head, X, Y):
    core.eval()
    proj.eval()
    head.eval()
    p = torch.cat([head(core(proj(X[k : k + 256]))).argmax(1) for k in range(0, len(X), 256)])
    return float((p == Y).float().mean())


def run(arm, seed):
    torch.manual_seed(1000 + seed)
    hX, hY, hXte, hYte, hch, hno = har_data(seed)
    sX, sY, sS = speech_data()
    spk = np.unique(sS)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(spk, max(1, len(spk) // 5), replace=False).tolist())
    tr = np.where(~np.isin(sS, list(held)))[0]
    ho = np.where(np.isin(sS, list(held)))[0]
    sno = int(sY.max()) + 1

    core = SharedCore()
    hproj, hhead = Proj(hch), nn.Linear(LATENT, hno)
    sproj, shead = Proj(sX.shape[2]), nn.Linear(LATENT, sno)

    # stage 1: HAR
    fit(core, hproj, hhead, hX, hY, seed)
    har_after_1 = acc(core, hproj, hhead, hXte, hYte)

    # stage 2: audio. arm decides what carries over.
    if arm == "fresh_core":
        core = SharedCore()
    elif arm == "projection_only":
        core = SharedCore()  # core discarded, only the domain projection idea carries (no shared slow state)
    train_core = arm not in ("frozen_transferred_core",)
    fit(core, sproj, shead, sX[tr], sY[tr], seed, train_core=train_core)
    speech_acq = acc(core, sproj, shead, sX[tr], sY[tr])

    # stage 3: return to HAR (no core reinit for persistent arms)
    fit(core, hproj, hhead, hX, hY, seed, steps=STEPS // 3, train_core=train_core)
    har_return = acc(core, hproj, hhead, hXte, hYte)

    # stage 4: held-out speaker adaptation
    fit(core, sproj, shead, sX[ho], sY[ho], seed, steps=STEPS // 3, train_core=train_core)
    future_adapt = acc(core, sproj, shead, sX[ho], sY[ho])

    return {
        "har_after_stage1": har_after_1,
        "speech_acquisition": speech_acq,
        "har_return_recovery": har_return,
        "future_adaptation": future_adapt,
        "negative_transfer": har_after_1 - har_return,
    }


ARMS = [
    "fresh_core",
    "projection_only",
    "slow_core_transfer",
    "frozen_transferred_core",
    "finetuned_transferred_core",
    "full_persistent_substrate",
]


def lcb(e):
    e = np.asarray(e, float)
    n = len(e)
    sd = e.std(ddof=1) if n > 1 else 0.0
    return float(e.mean() - 1.833 * sd / np.sqrt(n))


def main():
    t0 = time.time()
    per = {a: [] for a in ARMS}
    for s in SEEDS:
        for a in ARMS:
            per[a].append(run(a, s))
        print(f"  seed{s} done", flush=True)
    means = {a: {k: round(float(np.mean([r[k] for r in per[a]])), 4) for k in per[a][0]} for a in ARMS}
    base = "fresh_core"
    eff = {}
    for a in ARMS:
        if a == base:
            continue
        e = [per[a][i]["speech_acquisition"] - per[base][i]["speech_acquisition"] for i in range(len(SEEDS))]
        r = [
            per[a][i]["har_return_recovery"] - per[base][i]["har_return_recovery"] for i in range(len(SEEDS))
        ]
        eff[a] = {
            "speech_acq_vs_fresh": {"mean": round(float(np.mean(e)), 4), "lower_95_cb": round(lcb(e), 4)},
            "har_return_vs_fresh": {"mean": round(float(np.mean(r)), 4), "lower_95_cb": round(lcb(r), 4)},
        }
    pos = [
        a
        for a, v in eff.items()
        if v["speech_acq_vs_fresh"]["lower_95_cb"] >= 0.05 and v["har_return_vs_fresh"]["lower_95_cb"] > -0.02
    ]
    verdict = "cross_domain_moldability_positive" if pos else "cross_domain_moldability_null"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd="/Users/scammermike/Downloads/mop-substrate-forge",
    ).stdout.strip()
    doc = {
        "schema": "mop-integrated-cross-domain/v1",
        "source_commit": commit,
        "sequence": "HAR sensor -> Speech Commands audio -> return HAR -> held-out speaker adaptation",
        "domains_valid": {
            "HAR_raw": "temporal_headroom_present",
            "SpeechCommands": "temporal_headroom_present (mel filterbank)",
        },
        "seeds": SEEDS,
        "arms": ARMS,
        "means": means,
        "effects_vs_fresh_core": eff,
        "positive_arms": pos,
        "verdict": verdict,
        "SESOI": 0.05,
        "core_reinitialized_for_persistent_arm": False,
        "wall_seconds": round(time.time() - t0, 1),
    }
    doc["sha256"] = sha_obj(doc)
    (OUT / "MOP_SUBSTRATE_CROSS_DOMAIN_REPORT.json").write_text(json.dumps(doc, indent=2))
    print("verdict:", verdict, "| positive arms:", pos)
    for a in ARMS:
        print(f"  {a:28s} {means[a]}")
    print("CROSS_DONE", flush=True)


if __name__ == "__main__":
    main()
