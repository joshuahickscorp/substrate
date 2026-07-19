"""Lane P runner (post-audit, faithful P1R): run every replay method on EMNIST-balanced class-incremental over
several torch-seeded streams under matched memory and compute, then compare P1R to the best established method
per past task (retention units), aggregating across seeds. House style: no dashes."""

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
SEEDS = [0, 1, 2, 3, 4]


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
    per_seed_final = {m: [] for m in ALL_METHODS}   # list over seeds of final per-task accuracy arrays
    per_seed_fa = {m: [] for m in ALL_METHODS}
    plast = {m: [] for m in ALL_METHODS}
    for seed in SEEDS:
        prep = lp._prep(seed)
        for m in ALL_METHODS:
            mat = lp.run_stream(m, seed=seed, prepared=prep)
            per_seed_final[m].append(lp.joint_utility(mat))
            fa = float(mat[lp.N_TASKS - 1, :].mean()); per_seed_fa[m].append(fa)
            plast[m].append(float(np.mean([mat[t, t] for t in range(lp.N_TASKS)])))
            print(f"  seed{seed} {m:11s} final_avg={fa:.4f}", flush=True)
    # aggregate: per-task retention effect averaged across seeds; units = past tasks 0..N-2
    units = list(range(lp.N_TASKS - 1))
    p1r_task = np.mean([per_seed_final["p1r"][s] for s in range(len(SEEDS))], axis=0)
    best_est_task = np.mean([[max(per_seed_final[m][s][j] for m in ESTABLISHED) for j in range(lp.N_TASKS)]
                             for s in range(len(SEEDS))], axis=0)
    effects = [float(p1r_task[j] - best_est_task[j]) for j in units]
    re = random_effects(effects)
    p1r_fa = float(np.mean(per_seed_fa["p1r"])); p1r_fa_sd = float(np.std(per_seed_fa["p1r"]))
    est_fa = {m: float(np.mean(per_seed_fa[m])) for m in ESTABLISHED}
    best_est_name = max(est_fa, key=est_fa.get); best_est_fa = est_fa[best_est_name]
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
        "schema": "mop-frontier-lane-p-result/v2-faithful", "lane": "P", "mechanism": "G1-P1R plasticity/replay-value selection",
        "question": "P1R (learned replay-value predictor with toxic gate) improves joint retention+plasticity beyond established replay under matched memory and compute on a stronger external source",
        "source_commit": commit, "dataset": "EMNIST-balanced class-incremental (9 tasks, 45 classes)",
        "external_source": "NIST Special Database 19 / EMNIST-balanced, distinct from split-MNIST, CIFAR-100, KMNIST",
        "repair_applied": "faithful P1R = validated learned replay-value predictor (rff_ridge over [prob,entropy,feat-norm]) with validated toxic-value gate, replacing the unfaithful hard-coded raw-loss+prototype filter; instrumentation hardened (torch-seeded, 5 seeds, true Vitter reservoir)",
        "seeds": SEEDS, "matched_memory": lp.MEM, "matched_compute_steps_per_task": lp.STEPS_PER_TASK,
        "independent_units": "past tasks (retention), seed-averaged",
        "established_methods": {"reservoir": "Vitter reservoir sampling", "gdumb": "Prabhu 2020 greedy balancing",
                                "loss_based": "hard-example replay", "recency": "recency buffer"},
        "final_avg_accuracy_mean": {m: round(float(np.mean(per_seed_fa[m])), 4) for m in ALL_METHODS},
        "final_avg_accuracy_sd": {m: round(float(np.std(per_seed_fa[m])), 4) for m in ALL_METHODS},
        "plasticity_diag_mean": {m: round(float(np.mean(plast[m])), 4) for m in ALL_METHODS},
        "best_established": best_est_name, "best_established_final_avg": round(best_est_fa, 4),
        "p1r_final_avg": round(p1r_fa, 4), "p1r_final_avg_sd": round(p1r_fa_sd, 4),
        "primary_comparison_per_task_effect": re,
        "classification": cls, "SESOI": 0.05, "tie_is_null": True,
        "licenses_next": cls in ("same_team_external_method_positive", "external_replication_positive"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    res["result_sha256"] = sha(res)
    (REPORTS / "MOP_FRONTIER_P_RESULT.json").write_text(json.dumps(res, indent=2))
    print(f"[P] {cls} p1r_fa={p1r_fa:.4f}(sd {p1r_fa_sd:.3f}) best_est={best_est_name}({best_est_fa:.4f}) "
          f"per_task_incr_mean={re['mean']} lcb={re['lower_95_cb']} favorable={re['favorable_fraction']} [{res['wall_seconds']}s]", flush=True)
    print("LANE_P_DONE", flush=True)
    return cls


if __name__ == "__main__":
    run()
