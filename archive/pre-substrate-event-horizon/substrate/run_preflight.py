"""Domain validity preflight over >=5 seeds: measure forgetting and residual headroom beyond the STRONG
baseline (GDumb), not naive fine-tuning. Honors the C1 lesson: no headroom decision on two seeds.
House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-substrate-genesis-v2/substrate")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import domains as D  # noqa: E402
from engine import make, moldability, run_stream  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-substrate-genesis-v2/substrate/reports")
SEEDS = [0, 1, 2, 3, 4]
CANDIDATES = ["emnist", "cifar100", "har_class", "har_shift"]


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def joint_upper(model_name, tasks, obs_dim, n_out, seed, steps=120):
    """Train on all tasks pooled (no forgetting ceiling), eval per task."""
    torch.manual_seed(1000 + seed)
    m = make(model_name, obs_dim, n_out); opt = torch.optim.Adam(m.parameters(), 1e-3)
    rng = np.random.default_rng(seed + 7)
    X = torch.cat([t["x"] for t in tasks]); Y = torch.cat([t["y"] for t in tasks]); C = torch.cat([t["ctx"] for t in tasks])
    m.train()
    for _ in range(steps * len(tasks)):
        bi = rng.choice(len(X), min(64, len(X)), replace=False)
        out, _, _ = m(X[bi], C[bi]); opt.zero_grad(); F.cross_entropy(out, Y[bi]).backward(); opt.step()
    m.eval(); accs = []
    for t in tasks:
        xt, ct, yt = t["test"][0], t["test"][2], t["test"][1]
        with torch.no_grad():
            pred = torch.cat([m(xt[s:s + 256], ct[s:s + 256])[0].argmax(1) for s in range(0, len(xt), 256)])
        accs.append(float((pred == yt).float().mean()))
    return float(np.mean(accs))


def run():
    t0 = time.time(); out = {}
    for dom in CANDIDATES:
        naive, gdumb, joint = [], [], []
        for s in SEEDS:
            tasks, obs_dim, n_out = D.PROVIDERS[dom](s)
            steps = 80
            a_naive, _, _, _ = run_stream("mlp", tasks, {"memory": "none", "steps": steps}, s, obs_dim, n_out)
            a_gdumb, _, _, _ = run_stream("mlp", tasks, {"memory": "gdumb", "steps": steps, "mem_cap": 1000}, s, obs_dim, n_out)
            naive.append(moldability(a_naive)["avg_final"]); gdumb.append(moldability(a_gdumb)["avg_final"])
            joint.append(joint_upper("mlp", tasks, obs_dim, n_out, s, steps=steps))
        naive, gdumb, joint = map(np.array, (naive, gdumb, joint))
        forgetting = float((joint - naive).mean()); residual = float((joint - gdumb).mean())
        f_lcb = float((joint - naive).mean() - 1.833 * (joint - naive).std(ddof=1) / np.sqrt(len(SEEDS)))
        r_lcb = float((joint - gdumb).mean() - 1.833 * (joint - gdumb).std(ddof=1) / np.sqrt(len(SEEDS)))
        if f_lcb < 0.05:
            cls = "invalid_no_forgetting"
        elif r_lcb < 0.05:
            cls = "invalid_no_headroom"
        else:
            cls = "eligible_principal"
        out[dom] = {"is_image": D.IS_IMAGE[dom], "naive_mean": round(float(naive.mean()), 4),
                    "gdumb_mean": round(float(gdumb.mean()), 4), "joint_upper_mean": round(float(joint.mean()), 4),
                    "forgetting_mean": round(forgetting, 4), "forgetting_lcb": round(f_lcb, 4),
                    "residual_headroom_beyond_gdumb_mean": round(residual, 4), "residual_headroom_lcb": round(r_lcb, 4),
                    "classification": cls, "seeds": len(SEEDS)}
        print(f"  {dom:10s} naive={naive.mean():.3f} gdumb={gdumb.mean():.3f} joint={joint.mean():.3f} "
              f"forget_lcb={f_lcb:.3f} resid_lcb={r_lcb:.3f} -> {cls}", flush=True)
    commit = __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          cwd="/Users/scammermike/Downloads/mop-substrate-genesis-v2").stdout.strip()
    rep = {"schema": "mop-substrate-domain-preflight/v1", "source_commit": commit, "seeds": SEEDS,
           "strong_baseline": "GDumb (class-balanced replay), not naive fine-tuning", "domains": out,
           "eligible_principal": [d for d in out if out[d]["classification"] == "eligible_principal"],
           "eligible_non_image": [d for d in out if out[d]["classification"] == "eligible_principal" and not out[d]["is_image"]],
           "wall_seconds": round(time.time() - t0, 1)}
    rep["sha256"] = sha(rep)
    (REPORTS / "MOP_SUBSTRATE_DOMAIN_PREFLIGHT.json").write_text(json.dumps(rep, indent=2))
    print(f"eligible principal: {rep['eligible_principal']} | non-image: {rep['eligible_non_image']}", flush=True)
    print("PREFLIGHT_DONE", flush=True)


if __name__ == "__main__":
    run()
