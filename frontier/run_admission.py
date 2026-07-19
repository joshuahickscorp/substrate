"""Generic frontier admission runner: run a bed through the calibrated Phase 4B battery, independently
recompute the effect with separate estimator code, classify, and seal a result. House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop/salvage/lanes2")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-scientific-frontier/frontier/lanes")
import numpy as np  # noqa: E402
from estimators import CAPABLE, select_and_fit  # noqa: E402
from repaired_battery import classify, run_repaired_battery  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")
REPORTS.mkdir(parents=True, exist_ok=True)


def nat(o):
    if isinstance(o, dict):
        return {k: nat(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nat(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return nat(o.tolist())
    return o


def sha(v):
    return hashlib.sha256(json.dumps(nat(v), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def independent_effect(bed):
    """Separately authored recomputation of mechanism-minus-random normalized effect across units.

    Shares no primary-metric code with the battery: it recomputes the oracle, random baseline, and the best
    capable estimator's allocated value directly here.
    """
    inc = []
    for f in range(bed.n_families):
        fam = bed.family(f); te = fam["test"]; b = fam["budget"]
        r = te["r"]
        orc = float(np.sum(np.sort(r)[::-1][:b]))
        rnd = float(np.mean([np.sum(r[np.random.default_rng(s).choice(len(r), b, replace=False)]) for s in range(20)]))
        best = 0.0
        for name in CAPABLE:
            pred = select_and_fit(name, fam["train"]["x"], fam["train"]["r"],
                                  fam["tune"]["x"], fam["tune"]["r"], te["x"])
            v = float(np.sum(r[np.argsort(-pred)[:b]]))
            if orc > rnd:
                best = max(best, (v - rnd) / (orc - rnd))
        inc.append(best)
    return float(np.mean(inc)), inc


# battery classification -> lane terminal class
ADMISSION_MAP = {
    "admitted": "admitted",
    "pruned_mechanism": "pruned_mechanism",
    "invalid_bed": "invalid_bed",
    "insufficient_independent_units": "insufficient_independent_units",
    "architecture_dependent": "architecture_dependent",
    "instrumentation_failure": "instrumentation_failure",
}


def run_lane(lane_id, mechanism, question, dataset, units, BedClass, controls_note):
    t = time.time()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-scientific-frontier").stdout.strip()
    bed = BedClass()
    battery = run_repaired_battery(bed)
    verdict = classify(battery)
    indep_mean, indep_per = independent_effect(bed)
    prod = battery["clauses"]["J_rate_matched_random"]["mechanism_minus_random"]
    cls = ADMISSION_MAP.get(verdict["classification"], verdict["classification"])
    c = battery["clauses"]
    res = {
        "schema": "mop-frontier-admission-result/v1",
        "lane": lane_id, "mechanism": mechanism, "question": question,
        "source_commit": commit, "dataset": dataset, "independent_units": units,
        "classification": cls, "battery_classification": verdict["classification"],
        "clauses_passed": f"{verdict['clauses_passed']}/{verdict['clauses_total']}",
        "first_failing_clause": verdict["first_failing_clause"],
        "clauses": battery["clauses"], "residual_learnable_headroom": battery["residual_learnable_headroom"],
        "controls": controls_note,
        "independent_verification": {
            "independent_mean_effect": round(indep_mean, 3),
            "producer_mean_effect": prod,
            "per_unit_independent": [round(x, 3) for x in indep_per],
            "sign_agrees": (indep_mean > 0) == (prod > 0),
        },
        "SESOI": 0.05, "tie_is_null": True,
        "licenses_canary": cls == "admitted",
        "wall_seconds": round(time.time() - t, 1),
    }
    res["result_sha256"] = sha(res)
    out = REPORTS / f"MOP_FRONTIER_{lane_id}_ADMISSION_RESULT.json"
    out.write_text(json.dumps(nat(res), indent=2))
    d = c["D_incremental_value"]
    print(f"[{lane_id}] {cls} ({verdict['clauses_passed']}/{verdict['clauses_total']}) "
          f"first_fail={verdict['first_failing_clause']} D_incr={d['mean_incr_over_simple']} "
          f"lcb={d['lower_95_cb']} F={c['F_architecture_independence']['n_passing']} "
          f"B={c['B_oracle_headroom']['raw_headroom_fraction']} indep_agree={res['independent_verification']['sign_agrees']} "
          f"[{res['wall_seconds']}s]")
    return cls
