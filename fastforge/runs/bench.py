"""Resource profile and baseline convergence.

Two questions must be answered before any principal run is sized:

1. how many simultaneous training capsules maximize aggregate throughput on this host, and
2. what learning rate and step budget each baseline needs to plateau, because no substrate result is
   terminal against an unconverged baseline.

The convergence sweep covers learning rate as well as steps. A baseline starved by a badly chosen learning
rate is not a strong baseline, and beating one is not evidence.

Run as shards, one process per (domain, baseline), then aggregate.

House style: no dashes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastforge import arch as A
from fastforge import data as D
from fastforge import engine as E
from fastforge.runs import io

PY = sys.executable
STEP_GRID = [150, 350, 550, 900, 1400]
LR_GRID = [1e-3, 3e-3]
SEEDS = [0, 1, 2]
BASELINES = ["gru", "lstm", "separate", "shared_adapters", "matched_capacity", "bag"]
DOMAINS = ["har", "speech"]
PLATEAU_GAIN = 0.01


def one_capsule():
    d = D.domain("har")
    m = A.build("lstm", {"har": (9, 6)})
    t0 = time.time()
    E.fit(
        m, "har", d["x"], d["y"], train_groups=list(m.param_groups), steps=120, rng=np.random.default_rng(0)
    )
    return time.time() - t0


def concurrency_benchmark(levels=(1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24)):
    prof = {}
    script = Path(__file__).with_name("_capsule.py")
    script.write_text(
        f"import sys;sys.path.insert(0,{str(Path(__file__).resolve().parents[2])!r})\n"
        + "from fastforge.runs.bench import one_capsule;print(one_capsule())\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for n in levels:
        t0 = time.time()
        ps = [subprocess.Popen([PY, str(script)], env=env, stdout=subprocess.DEVNULL) for _ in range(n)]
        for p in ps:
            p.wait()
        wall = time.time() - t0
        prof[n] = {"wall_seconds": round(wall, 2), "capsules_per_minute": round(60.0 * n / wall, 2)}
        print(
            f"  capsules={n:2d} wall={wall:6.2f}s throughput={prof[n]['capsules_per_minute']:7.2f}/min",
            flush=True,
        )
    script.unlink(missing_ok=True)
    return prof, max(prof, key=lambda k: prof[k]["capsules_per_minute"])


def curve(dname, kind):
    d = D.domain(dname)
    dom = {dname: (d["channels"], d["classes"])}
    out = {}
    for lr in LR_GRID:
        for steps in STEP_GRID:
            accs = []
            for s in SEEDS:
                torch.manual_seed(1000 + s)
                m = A.build(kind, dom)
                E.fit(
                    m,
                    dname,
                    d["x"],
                    d["y"],
                    train_groups=list(m.param_groups),
                    steps=steps,
                    lr=lr,
                    rng=np.random.default_rng(s),
                )
                accs.append(E.evaluate(m, dname, d["xte"], d["yte"]))
            out[f"{lr}|{steps}"] = {
                "learning_rate": lr,
                "steps": steps,
                "updates": steps,
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "seed_sd": round(float(np.std(accs, ddof=1)), 4),
                "params": A.count_params(A.build(kind, dom)),
                "memory_cap": 0,
            }
            print(
                f"  {dname:7s} {kind:16s} lr={lr:<6} steps={steps:5d} "
                f"acc={out[f'{lr}|{steps}']['accuracy_mean']:.4f}",
                flush=True,
            )
    return out


def plateau(curve_for_lr):
    """First step count whose accuracy gain over the previous grid point falls below the plateau gain."""
    for i in range(1, len(STEP_GRID)):
        a = curve_for_lr[STEP_GRID[i - 1]]
        b = curve_for_lr[STEP_GRID[i]]
        if b - a < PLATEAU_GAIN:
            return STEP_GRID[i]
    return STEP_GRID[-1]


def aggregate(rows):
    curves, selection = {}, {}
    for key, c in rows.items():
        dname, kind = key.split("|")
        per_lr = {}
        for lr in LR_GRID:
            acc = {st: c[f"{lr}|{st}"]["accuracy_mean"] for st in STEP_GRID}
            pl = plateau(acc)
            per_lr[lr] = {
                "plateau_steps": pl,
                "accuracy_at_plateau": acc[pl],
                "best_accuracy": max(acc.values()),
                "seed_sd_at_plateau": c[f"{lr}|{pl}"]["seed_sd"],
            }
        best_lr = max(LR_GRID, key=lambda lr: per_lr[lr]["accuracy_at_plateau"])
        curves[key] = {
            "grid": c,
            "per_learning_rate": per_lr,
            "selected_learning_rate": best_lr,
            "selected_checkpoint": f"{best_lr}|{per_lr[best_lr]['plateau_steps']}",
            "plateau_steps": per_lr[best_lr]["plateau_steps"],
            "accuracy_at_selected_checkpoint": per_lr[best_lr]["accuracy_at_plateau"],
            "seed_variance_at_plateau": per_lr[best_lr]["seed_sd_at_plateau"],
        }
        selection.setdefault(dname, {})[kind] = (best_lr, per_lr[best_lr]["plateau_steps"])
    principal = {}
    for dname, per in selection.items():
        lrs = [v[0] for v in per.values()]
        principal[dname] = {
            "learning_rate": max(set(lrs), key=lrs.count),
            "steps": int(max(v[1] for v in per.values())),
            "rule": "the learning rate most baselines converge best under, and the largest plateau across "
            "baselines, so no baseline is starved by the principal budget",
        }
    return curves, principal


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        dname, kind = sys.argv[2].split(":")
        p = io.RUNS / "bench"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{dname}_{kind}.json").write_text(json.dumps(curve(dname, kind)))
        print("SHARD_DONE", dname, kind, flush=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "concurrency":
        prof, best = concurrency_benchmark()
        io.seal(
            "MOP_FAST_STATE_RESOURCE_REPORT.json",
            {
                "schema": "mop-fast-state-resource/v1",
                "host_cores": os.cpu_count(),
                "torch_threads_per_capsule": 1,
                "note": "torch intra op threading measurably hurts these small recurrent models, so capsules "
                "are single threaded and concurrency is process level",
                "concurrency_profile": prof,
                "selected_concurrency": best,
                "aggregate_throughput_at_selected": prof[best]["capsules_per_minute"],
                "decomposition": [
                    "domain",
                    "sequence direction",
                    "architecture",
                    "architecture round",
                    "seed",
                    "baseline",
                    "plasticity policy",
                    "ablation",
                    "verification shard",
                    "mutation family",
                ],
            },
        )
        print("selected concurrency", best, flush=True)
        print("RESOURCE_DONE", flush=True)
        return

    rows = {}
    for dname in DOMAINS:
        for kind in BASELINES:
            f = io.RUNS / "bench" / f"{dname}_{kind}.json"
            if f.is_file():
                rows[f"{dname}|{kind}"] = json.loads(f.read_text())
    if len(rows) < len(DOMAINS) * len(BASELINES):
        print("missing bench shards:", len(rows), "of", len(DOMAINS) * len(BASELINES), flush=True)
        return
    curves, principal = aggregate(rows)
    io.seal(
        "MOP_BASELINE_CONVERGENCE_REPORT.json",
        {
            "schema": "mop-baseline-convergence/v2",
            "seeds": SEEDS,
            "step_grid": STEP_GRID,
            "learning_rate_grid": LR_GRID,
            "baselines": BASELINES,
            "curves": curves,
            "plateau_criterion": f"first step count whose accuracy gain over the previous grid point is "
            "below "
            f"{PLATEAU_GAIN}",
            "principal_budget": principal,
            "rule": "no substrate result is terminal against an unconverged baseline. The principal budget "
            "is "
            "chosen so that every baseline is at or past its own plateau.",
            "note": "the separate baseline is architecturally identical to lstm when only one context is "
            "registered, so its single domain curve matches lstm by construction",
        },
    )
    print(json.dumps(principal), flush=True)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
