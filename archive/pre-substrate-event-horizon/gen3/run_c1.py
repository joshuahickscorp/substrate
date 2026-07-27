"""C1 stages A (calibration) and B (canary) on the fixed GDumb buffer, plus C (confirmation) on HAR if licensed.
Primary comparison: the learned P1R sampling-priority main effect vs the strongest simple sampling control
(uniform / loss-priority) on a fixed established buffer, with >=2 capable estimators required to agree.
House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-gen3/gen3")
import numpy as np  # noqa: E402
import c1_priority as c1  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-gen3/gen3/reports")
REPORTS.mkdir(parents=True, exist_ok=True)
ESTIMATORS = ["rff_ridge", "kernel_ridge", "knn"]


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_effects(effects, sesoi=0.05):
    e = np.asarray(effects, float); n = len(e)
    mean = float(e.mean()); sd = float(e.std(ddof=1)) if n > 1 else 0.0
    t95 = 1.833 if n <= 10 else 1.729
    lcb = mean - t95 * (sd / np.sqrt(n) if n else 0)
    return {"mean": round(mean, 4), "sd": round(sd, 4), "lower_95_cb": round(lcb, 4),
            "favorable_fraction": round(float((e > 0).mean()), 3), "n": n, "pass": bool(lcb >= sesoi)}


def run_source(source, seeds, steps):
    """Return per-policy final-avg per seed and per-task final accuracies."""
    policies = [("uniform", None), ("loss_priority", None), ("oracle_priority", None),
                ("shuffled_p1r", "rff_ridge"), ("random_rate_matched", None)]
    for e in ESTIMATORS:
        policies.append((f"learned_p1r:{e}", e))
    fa = {p[0]: [] for p in policies}; per_task = {p[0]: [] for p in policies}
    for s in seeds:
        for name, est in policies:
            v, acc = c1.run_c1(source, "gdumb", name.split(":")[0], est or "rff_ridge", s, steps=steps)
            fa[name].append(v); per_task[name].append(acc[acc.shape[0] - 1, :])
            print(f"  [{source}] seed{s} {name:22s} final_avg={v:.4f}", flush=True)
    return fa, per_task


def classify(source, fa, per_task):
    nT = len(per_task["uniform"][0]); units = list(range(nT - 1))
    def task_mean(name):
        return np.mean([per_task[name][si] for si in range(len(per_task[name]))], axis=0)
    best_simple_task = np.maximum(task_mean("uniform"), task_mean("loss_priority"))
    m = {p: float(np.mean(fa[p])) for p in fa}
    # bed validity: oracle must beat uniform (headroom present)
    bed_valid = m["oracle_priority"] > m["uniform"] + 0.02
    # controls: shuffled and random must not beat uniform
    controls_ok = (m["shuffled_p1r"] <= m["uniform"] + 0.02) and (m["random_rate_matched"] <= m["uniform"] + 0.02)
    est_effects = {}
    for e in ESTIMATORS:
        eff = [float(task_mean(f"learned_p1r:{e}")[j] - best_simple_task[j]) for j in units]
        est_effects[e] = random_effects(eff)
    n_agree = sum(1 for e in ESTIMATORS if est_effects[e]["pass"])
    beats_simple = all(m[f"learned_p1r:{e}"] >= max(m["uniform"], m["loss_priority"]) for e in ESTIMATORS if est_effects[e]["pass"])
    if not bed_valid:
        cls = "invalid_bed"
    elif not controls_ok:
        cls = "instrumentation_failure"
    elif n_agree >= 2 and beats_simple:
        cls = "priority_canary_positive"
    else:
        cls = "priority_canary_null"
    return cls, {"final_avg_means": {k: round(v, 4) for k, v in m.items()}, "bed_valid": bed_valid,
                 "controls_ok": controls_ok, "estimator_effects": est_effects, "n_estimators_agree": n_agree,
                 "strongest_simple_control": "max(uniform, loss_priority)",
                 "oracle_headroom_over_uniform": round(m["oracle_priority"] - m["uniform"], 4),
                 "learned_vs_loss_priority": {e: round(m[f"learned_p1r:{e}"] - m["loss_priority"], 4) for e in ESTIMATORS}}


def seal(stage, source, cls, detail, wall):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-gen3").stdout.strip()
    res = {"schema": "mop-gen3-c1-result/v1", "stream": "C1", "stage": stage, "source": source,
           "source_commit": commit, "classification": cls, "SESOI": 0.05, "tie_is_null": True,
           "primary": "learned P1R sampling-priority main effect on fixed GDumb buffer vs strongest simple control",
           "detail": detail, "wall_seconds": wall}
    res["sha256"] = sha(res)
    (REPORTS / f"MOP_GEN3_C1_{stage}_{source}_RESULT.json").write_text(json.dumps(res, indent=2))
    print(f"[C1-{stage} {source}] {cls} | oracle_headroom={detail['oracle_headroom_over_uniform']} "
          f"n_agree={detail['n_estimators_agree']} learned_vs_loss={detail['learned_vs_loss_priority']} [{wall}s]", flush=True)
    return cls


if __name__ == "__main__":
    t0 = time.time()
    # Stage C1-A + C1-B: calibration and canary on EMNIST (image)
    fa, pt = run_source("emnist", [0, 1, 2], steps=120)
    cls_e = classify("emnist", fa, pt)
    seal("B_canary", "emnist", cls_e[0], cls_e[1], round(time.time() - t0, 1))
    # Stage C1-C: HAR confirmation (non-image principal source) runs regardless, so the evidence base is not all-image
    t1 = time.time()
    fah, pth = run_source("har", [0, 1, 2], steps=150)
    cls_h = classify("har", fah, pth)
    seal("C_confirm", "har", cls_h[0], cls_h[1], round(time.time() - t1, 1))
    print("C1_DONE", flush=True)
