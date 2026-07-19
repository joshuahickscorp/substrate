"""Lane P runner: run every replay method on the EMNIST-balanced class-incremental stream under matched memory
and compute, then compare P1R to the best established method per past task (retention units). House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-scientific-frontier/frontier/lanes")
import numpy as np  # noqa: E402
import lane_p_emnist as lp  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")
ESTABLISHED = ["reservoir", "gdumb", "loss_based", "recency"]
ALL_METHODS = ["none"] + ESTABLISHED + ["p1r"]


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_effects(effects, sesoi=0.05):
    e = np.asarray(effects, float); n = len(e)
    mean = float(e.mean()); sd = float(e.std(ddof=1)) if n > 1 else 0.0
    t95 = 1.833 if n <= 10 else 1.729
    se = sd / np.sqrt(n) if n else 0.0
    lcb = mean - t95 * se
    return {"mean": round(mean, 4), "between_task_sd": round(sd, 4), "lower_95_cb": round(lcb, 4),
            "min_detectable_effect": round(t95 * se, 4), "favorable_fraction": round(float((e > 0).mean()), 3),
            "n": n, "pass": bool(lcb >= sesoi)}


def run():
    t0 = time.time()
    mats = {}
    for m in ALL_METHODS:
        mats[m] = lp.run_stream(m, seed=0)
        fa = float(mats[m][lp.N_TASKS - 1, :].mean())
        print(f"  {m:11s} final_avg_acc={fa:.4f}")
    finals = {m: lp.joint_utility(mats[m]) for m in ALL_METHODS}
    # plasticity: accuracy on each task right after learning it (diagonal)
    plast = {m: float(np.mean([mats[m][t, t] for t in range(lp.N_TASKS)])) for m in ALL_METHODS}
    # retention units = past tasks 0..N-2 (last task trivially retained)
    units = list(range(lp.N_TASKS - 1))
    best_est_per_task = np.array([max(finals[m][j] for m in ESTABLISHED) for j in units])
    p1r_per_task = np.array([finals["p1r"][j] for j in units])
    effects = (p1r_per_task - best_est_per_task).tolist()
    re = random_effects(effects)
    p1r_fa = float(finals["p1r"].mean()); best_est_fa = max(float(finals[m].mean()) for m in ESTABLISHED)
    best_est_name = max(ESTABLISHED, key=lambda m: float(finals[m].mean()))
    harm = re["mean"] < -0.05 and re["lower_95_cb"] < 0
    if re["pass"] and p1r_fa >= best_est_fa:
        cls = "same_team_external_method_positive"
    elif harm:
        cls = "replication_harm"
    else:
        cls = "replication_null"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-scientific-frontier").stdout.strip()
    res = {
        "schema": "mop-frontier-lane-p-result/v1", "lane": "P", "mechanism": "G1-P1R plasticity/replay selection",
        "question": "P1R improves joint retention+plasticity beyond established replay under matched memory and compute on a stronger external source",
        "source_commit": commit, "dataset": "EMNIST-balanced class-incremental (9 tasks, 45 classes)",
        "external_source": "NIST Special Database 19 / EMNIST-balanced, distinct from split-MNIST, CIFAR-100, KMNIST",
        "matched_memory": lp.MEM, "matched_compute_steps_per_task": lp.STEPS_PER_TASK,
        "independent_units": "past tasks (retention measurements)",
        "established_methods": {"reservoir": "Vitter/Chaudhry tiny episodic memory", "gdumb": "Prabhu 2020 greedy balancing",
                                "loss_based": "hard-example replay", "recency": "recency buffer"},
        "final_avg_accuracy": {m: round(float(finals[m].mean()), 4) for m in ALL_METHODS},
        "plasticity_diag_accuracy": {m: round(plast[m], 4) for m in ALL_METHODS},
        "best_established": best_est_name, "best_established_final_avg": round(best_est_fa, 4),
        "p1r_final_avg": round(p1r_fa, 4),
        "primary_comparison_per_task_effect": re,
        "classification": cls, "SESOI": 0.05, "tie_is_null": True,
        "licenses_next": cls == "same_team_external_method_positive",
        "wall_seconds": round(time.time() - t0, 1),
    }
    res["result_sha256"] = sha(res)
    (REPORTS / "MOP_FRONTIER_P_RESULT.json").write_text(json.dumps(res, indent=2))
    print(f"[P] {cls} p1r_fa={p1r_fa:.4f} best_est={best_est_name}({best_est_fa:.4f}) "
          f"per_task_incr_mean={re['mean']} lcb={re['lower_95_cb']} favorable={re['favorable_fraction']} [{res['wall_seconds']}s]")
    return cls


if __name__ == "__main__":
    run()
