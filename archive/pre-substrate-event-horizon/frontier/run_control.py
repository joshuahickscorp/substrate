"""Control admission runner for A1 and S1 on gymnasium classic-control units. Real-rollout evaluation,
normalized per-unit effects, random-effects lower-95pct-CB decision rule. House style: no dashes."""

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
import control_beds as cb  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def random_effects(effects, sesoi=0.05):
    e = np.asarray(effects, float); n = len(e)
    mean = float(e.mean()); sd = float(e.std(ddof=1)) if n > 1 else 0.0
    t95 = 1.833 if n <= 10 else 1.729
    se = sd / np.sqrt(n) if n else 0.0
    lcb = mean - t95 * se
    return {"mean": round(mean, 4), "between_unit_sd": round(sd, 4), "lower_95_cb": round(lcb, 4),
            "min_detectable_effect": round(t95 * se, 4), "favorable_fraction": round(float((e > 0).mean()), 3),
            "n": n, "pass": bool(lcb >= sesoi)}


def run_control():
    t0 = time.time()
    units = cb.ENV_SPECS
    # build all policies once per unit (fit on offline data)
    pol = [cb.build_policies(u, seed=i) for i, u in enumerate(units)]
    ARMS_A = ["A1", "reactive_bc", "greedy_reward", "value_estimator", "random", "wrong_context", "onestep_planner"]
    ARMS_S = ["S1", "reactive_bc", "greedy_reward", "onestep_planner", "value_estimator", "random", "actionblind_sim"]
    need = sorted(set(ARMS_A + ARMS_S))
    # wrong-context partner must share the base env (same observation dimension), but differ in dynamics
    def partner(i):
        base = units[i][0]
        for j in range(len(units)):
            if j != i and units[j][0] == base:
                return j
        return i
    ret = []  # per-unit dict of arm->return
    for i, u in enumerate(units):
        r = {}
        for arm in need:
            if arm == "wrong_context":
                fn = pol[partner(i)]["A1"]  # same env family, different (wrong) dynamics parameters
            else:
                fn = pol[i][arm]
            r[arm] = cb.rollout(u, fn, episodes=15, seed=1000 + i * 50)
        ret.append(r)

    eff_a, eff_s, informative = [], [], 0
    for r in ret:
        upper = max(r["A1"], r["S1"], r["value_estimator"], r["reactive_bc"], r["greedy_reward"], r["onestep_planner"])
        rnd = r["random"]; span = upper - rnd
        # a unit is informative only if some policy achieves headroom over random; otherwise it cannot
        # discriminate mechanism from control and is excluded (e.g. an env no linear policy can solve)
        if span < max(0.05 * abs(rnd), 1.0):
            continue
        informative += 1
        best_simple_a = max(r["reactive_bc"], r["greedy_reward"], r["value_estimator"])
        best_simple_s = max(r["reactive_bc"], r["greedy_reward"], r["onestep_planner"])
        eff_a.append(float(np.clip((r["A1"] - best_simple_a) / span, -2, 2)))
        eff_s.append(float(np.clip((r["S1"] - best_simple_s) / span, -2, 2)))

    def classify(effs, arms_mech_beats, ret_key, mech, extra_controls):
        re = random_effects(effs)
        beats_rand = float(np.mean([r[ret_key] - r["random"] for r in ret]))
        beats_extra = float(np.mean([r[ret_key] - r[extra_controls] for r in ret]))
        if informative < 5:
            cls = "insufficient_independent_units"
        elif beats_rand <= 0:
            cls = "pruned_mechanism"
        elif re["pass"] and beats_extra > 0:
            cls = "admitted"
        else:
            cls = "pruned_mechanism"
        return cls, re, beats_rand, beats_extra

    cls_a, re_a, br_a, bx_a = classify(eff_a, ARMS_A, "A1", "A1", "wrong_context")
    cls_s, re_s, br_s, bx_s = classify(eff_s, ARMS_S, "S1", "S1", "actionblind_sim")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-scientific-frontier").stdout.strip()
    wall = round(time.time() - t0, 1)
    for lane, mech, q, cls, re, br, bx, ctrl in [
        ("A", "G1-A1 read affordance from latent",
         "reading action-relevance from the latent improves action selection beyond reactive/greedy/value/predictor",
         cls_a, re_a, br_a, bx_a, "reactive_bc, greedy_reward, value_estimator, random, wrong_context, onestep_planner"),
        ("S", "G1-S1 simulate consequence vs react",
         "simulating consequences beats acting reactively and one-step planning under a fixed sim budget",
         cls_s, re_s, br_s, bx_s, "reactive_bc, greedy_reward, onestep_planner, value_estimator, random, actionblind_sim"),
    ]:
        res = {"schema": "mop-frontier-admission-result/v1", "lane": lane, "mechanism": mech, "question": q,
               "source_commit": commit, "dataset": "gymnasium classic-control (parameter-perturbed variants)",
               "independent_units": f"{len(units)} distinct dynamical systems ({informative} informative after excluding degenerate)",
               "harness": "real-rollout offline-fit",
               "classification": cls, "incremental_over_best_simple": re, "beats_random_return": round(br, 3),
               "beats_extra_control": round(bx, 3), "controls": ctrl, "SESOI": 0.05, "tie_is_null": True,
               "licenses_canary": cls == "admitted", "wall_seconds": wall}
        res["result_sha256"] = sha(res)
        (REPORTS / f"MOP_FRONTIER_{lane}_ADMISSION_RESULT.json").write_text(json.dumps(res, indent=2))
        print(f"[{lane}] {cls} incr_mean={re['mean']} lcb={re['lower_95_cb']} favorable={re['favorable_fraction']} "
              f"beats_random={round(br,2)} beats_extra={round(bx,3)} [{wall}s]")
    print("CONTROL_BATCH_DONE")


if __name__ == "__main__":
    run_control()
