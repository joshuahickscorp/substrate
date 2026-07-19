"""Temporal admission runner for E1 and C0. Applies the random-effects lower-95pct-CB rule to
mechanism-minus-best-simple-control downstream accuracy across independent sessions, with control rejection.
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
sys.path.insert(0, "/Users/scammermike/Downloads/mop-scientific-frontier/frontier/lanes")
import numpy as np  # noqa: E402
import temporal_beds as tb  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_effects(effects, sesoi=0.05):
    e = np.asarray(effects, dtype=float); n = len(e)
    mean = float(e.mean()); sd = float(e.std(ddof=1)) if n > 1 else 0.0
    t95 = 1.833 if n <= 10 else 1.729
    se = sd / np.sqrt(n) if n > 0 else 0.0
    lcb = mean - t95 * se
    return {"mean": round(mean, 4), "between_session_sd": round(sd, 4),
            "lower_95_cb": round(lcb, 4), "min_detectable_effect": round(t95 * se, 4),
            "favorable_fraction": round(float((e > 0).mean()), 3), "n": n, "pass": bool(lcb >= sesoi)}


SIMPLE_E1 = ["fixed_window", "uniform", "novelty_threshold", "prederr_threshold", "change_point"]
SIMPLE_C0 = ["direct_state", "last_observation", "ema_smoothing", "matched_memory"]


def run_e1():
    t = time.time()
    sessions = tb.build_sessions()
    n = len(sessions); train_idx = list(range(4)); test_idx = list(range(4, n))
    # tune learned_thresh on train sessions to maximize E1 downstream accuracy
    cand = np.linspace(0.05, 1.2, 24)
    best_thr, best_acc = cand[0], -1
    for thr in cand:
        accs = []
        for s in train_idx:
            emb, lab, tb_ = sessions[s]
            arms = tb.e1_session_arms(emb, lab, tb_, s, thr)
            accs.append(arms["E1"])
        if np.mean(accs) > best_acc:
            best_acc, best_thr = np.mean(accs), thr
    # evaluate on untouched test sessions
    per = {"E1": [], "best_simple": [], "oracle": [], "none": [], "shuffled": [], "random": []}
    effects = []
    for s in test_idx:
        emb, lab, tb_ = sessions[s]
        a = tb.e1_session_arms(emb, lab, tb_, s, best_thr)
        best_simple = max(a[k] for k in SIMPLE_E1)
        per["E1"].append(a["E1"]); per["best_simple"].append(best_simple)
        per["oracle"].append(a["oracle"]); per["none"].append(a["none"])
        per["shuffled"].append(a["E1_shuffled"]); per["random"].append(a["random_rate_matched"])
        effects.append(a["E1"] - best_simple)
    re = random_effects(effects)
    oracle_headroom = float(np.mean(per["oracle"]) - np.mean(per["best_simple"]))
    beats_shuffled = float(np.mean(per["E1"]) - np.mean(per["shuffled"]))
    beats_random = float(np.mean(per["E1"]) - np.mean(per["random"]))
    if oracle_headroom < 0.02:
        cls = "invalid_bed"
    elif n < 8:
        cls = "insufficient_independent_units"
    elif re["pass"] and beats_shuffled > 0 and beats_random > 0:
        cls = "admitted"
    else:
        cls = "pruned_mechanism"
    return _seal("E", "G1-E1 relational temporal event formation",
                 "relational temporal boundaries improve downstream prediction beyond fixed windows and change detectors",
                 "externally-ordered KMNIST class-run stream", "sessions", cls,
                 {"tuned_threshold": round(float(best_thr), 3), "n_sessions": n, "n_test": len(test_idx),
                  "incremental_over_best_simple": re, "oracle_headroom": round(oracle_headroom, 4),
                  "E1_mean": round(float(np.mean(per["E1"])), 4), "best_simple_mean": round(float(np.mean(per["best_simple"])), 4),
                  "oracle_mean": round(float(np.mean(per["oracle"])), 4), "none_mean": round(float(np.mean(per["none"])), 4),
                  "beats_shuffled": round(beats_shuffled, 4), "beats_random_rate_matched": round(beats_random, 4)},
                 "controls: none, fixed_window, uniform, random_rate_matched, novelty_threshold, prederr_threshold, change_point, shuffled_boundaries, oracle",
                 round(time.time() - t, 1))


def run_c0():
    t = time.time()
    sessions = tb.build_sessions()
    n = len(sessions)
    per = {"C0": [], "best_simple": [], "oracle": [], "random": [], "shuffled": []}
    effects = []
    for s in range(n):
        emb, lab, _ = sessions[s]
        a = tb.c0_session_arms(emb, lab, s)
        best_simple = max(a[k] for k in SIMPLE_C0)
        per["C0"].append(a["C0"]); per["best_simple"].append(best_simple)
        per["oracle"].append(a["oracle_stable_state"]); per["random"].append(a["random_trace"]); per["shuffled"].append(a["shuffled_trace"])
        effects.append(a["C0"] - best_simple)
    re = random_effects(effects)
    oracle_headroom = float(np.mean(per["oracle"]) - np.mean(per["best_simple"]))
    beats_random = float(np.mean(per["C0"]) - np.mean(per["random"]))
    beats_shuffled = float(np.mean(per["C0"]) - np.mean(per["shuffled"]))
    if oracle_headroom < 0.02:
        cls = "invalid_bed"
    elif n < 8:
        cls = "insufficient_independent_units"
    elif re["pass"] and beats_random > 0 and beats_shuffled > 0:
        cls = "admitted"
    else:
        cls = "pruned_mechanism"
    return _seal("C", "G1-C0 trace stability",
                 "stable persistent trace improves downstream decisions beyond direct state, smoothing, matched-memory",
                 "noisy KMNIST temporal stream", "sessions", cls,
                 {"n_sessions": n, "incremental_over_best_simple": re, "oracle_headroom": round(oracle_headroom, 4),
                  "C0_mean": round(float(np.mean(per["C0"])), 4), "best_simple_mean": round(float(np.mean(per["best_simple"])), 4),
                  "oracle_mean": round(float(np.mean(per["oracle"])), 4),
                  "beats_random_trace": round(beats_random, 4), "beats_shuffled_trace": round(beats_shuffled, 4)},
                 "controls: no_trace, direct_state, last_observation, ema_smoothing, matched_memory, random_trace, shuffled_trace, oracle",
                 round(time.time() - t, 1))


def _seal(lane, mech, q, dataset, units, cls, detail, controls, wall):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-scientific-frontier").stdout.strip()
    res = {"schema": "mop-frontier-admission-result/v1", "lane": lane, "mechanism": mech, "question": q,
           "source_commit": commit, "dataset": dataset, "independent_units": units, "harness": "temporal-sequence",
           "classification": cls, "detail": detail, "controls": controls, "SESOI": 0.05, "tie_is_null": True,
           "licenses_canary": cls == "admitted", "wall_seconds": wall}
    res["result_sha256"] = sha(res)
    (REPORTS / f"MOP_FRONTIER_{lane}_ADMISSION_RESULT.json").write_text(json.dumps(res, indent=2))
    re = detail["incremental_over_best_simple"]
    print(f"[{lane}] {cls} incr_mean={re['mean']} lcb={re['lower_95_cb']} oracle_headroom={detail['oracle_headroom']} "
          f"mech_mean={detail.get('E1_mean', detail.get('C0_mean'))} best_simple={detail['best_simple_mean']} [{wall}s]")
    return cls


if __name__ == "__main__":
    run_e1()
    run_c0()
    print("TEMPORAL_BATCH_DONE")
