"""Phase 4B runner: calibrate, preregister (new authority), run repaired batteries, classify, decide Phase 5.

Emits sealed: battery calibration report, three new preregistrations (append-only, SESOI unchanged), three
repaired admission-battery reports (V2), independent verification, and the Phase 4B summary with the exact
Phase 5 canary set. The original Phase 4 seals are untouched.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import CAPABLE, select_and_fit  # noqa: E402
from mechanism_beds import MechanismBed  # noqa: E402
from repaired_battery import (  # noqa: E402
    SESOI,
    classify,
    run_repaired_battery,
)

REPORTS = Path("/Users/scammermike/Downloads/mop/salvage/reports")
LANES = ("G1-P1R", "G1-U1", "G1-N1")
QUESTION = {
    "G1-P1R": "retain prior capability while learning materially new capability without sacrificing either",
    "G1-U1": "does uncertainty change decisions (verify/abstain/gather) with positive downstream value",
    "G1-N1": "distinguish reducible novelty from irreducible noise, difficulty, and shift",
}


def _native(o):
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    return o


def sha(v):
    return hashlib.sha256(json.dumps(_native(v), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preregister(lane):
    core = {
        "schema": "mop-canary-preregistration/v2", "lane": lane, "question": QUESTION[lane],
        "supersedes": "does NOT reuse the Phase 4 authority; new append-only authority; SESOI unchanged at 0.05",
        "mechanism_definition": "best of capacity-matched capable estimators (knn, kernel_ridge, rff_ridge) "
                                "predicting lane reducible value from decision-time features; tuned on a tuning "
                                "partition",
        "independent_units": "12 independent generator families, group-disjoint train/tune/test",
        "architectures": list(CAPABLE) + ["linear (baseline only, not counted for architecture independence)"],
        "capacity_matching": "same features, training units, tuning information, allocation and inference budget",
        "oracle_definition": "top-budget by true reducible value", "primary_metric": "fraction of oracle "
        "headroom over random captured", "sesoi": SESOI,
        "group_generalization_rule": "random-effects lower 95% confidence bound on the mean family effect >= SESOI",
        "architecture_independence_rule": ">= 2 capable estimators pass the group rule",
        "multiplicity": "all clauses A-J plus residual-headroom must pass jointly",
        "futility": "oracle raw headroom < 10% -> invalid_bed", "harm_rule": "excess noisy-TV allocation > SESOI prunes",
        "compute_ceiling": "one battery over 12 families x 3 capable estimators (small)",
        "claim_ceiling": "controlled-bed mechanism plausibility only; admission licenses a SMALL real-data or "
                         "independently sourced canary, not scientific confirmation",
    }
    return {**core, "prereg_sha256": sha(core)}


def independent_verify(bed):
    """Independent recompute of the best-capable-estimator mean family effect over random (separate loop)."""
    per_family = []
    for f in range(bed.n_families):
        fam = bed.family(f, 7)
        b = fam["budget"]
        te = fam["test"]
        orc = np.argsort(-te["r"])[:b]
        orc_v = float(np.sum(te["r"][orc]))
        rnd_v = np.mean([np.sum(te["r"][np.random.default_rng(s).choice(len(te["r"]), b, replace=False)])
                         for s in range(15)])
        best = 0.0
        for name in CAPABLE:
            pred = select_and_fit(name, fam["train"]["x"], fam["train"]["r"], fam["tune"]["x"],
                                  fam["tune"]["r"], te["x"])
            ch = np.argsort(-pred)[:b]
            v = float(np.sum(te["r"][ch]))
            if orc_v > rnd_v:
                best = max(best, (v - rnd_v) / (orc_v - rnd_v))
        per_family.append(best)
    return float(np.mean(per_family))


def calibrate():
    """Re-run the sealed calibration and record the pass/fail."""
    # capture the calibration result by calling its main components
    from calibration import AllocationBed
    pos = run_repaired_battery(AllocationBed(decodability=0.75, noisy_tv_frac=0.12))
    nul = run_repaired_battery(AllocationBed(decodability=0.0, noisy_tv_frac=0.0))
    ofb = run_repaired_battery(AllocationBed(oracle_free=True))
    out = {"positive_control": classify(pos)["classification"], "null_bed": classify(nul)["classification"],
           "oracle_free_bed": classify(ofb)["classification"]}
    out["calibration_passed"] = (out["positive_control"] == "admitted"
                                 and out["null_bed"] in ("pruned_mechanism", "invalid_bed")
                                 and out["oracle_free_bed"] == "invalid_bed")
    core = {"schema": "mop-phase4b-battery-calibration/v1", **out,
            "meaning": "the battery passes a known-good mechanism and rejects null and oracle-free beds, so it "
                       "is trustworthy to prune scientific lanes"}
    rep = {**core, "calibration_sha256": sha(core)}
    (REPORTS / "MOP_PHASE4B_BATTERY_CALIBRATION.json").write_text(json.dumps(_native(rep), indent=2))
    return out


def main():
    calib = calibrate()
    if not calib["calibration_passed"]:
        print("CALIBRATION FAILED; not running principal lanes")
        return 1
    lanes_out = {}
    for lane in LANES:
        bed = MechanismBed(lane)
        prereg = preregister(lane)
        (REPORTS / f"MOP_{lane.replace('-', '_')}_PREREGISTRATION_V2.json").write_text(json.dumps(_native(prereg), indent=2))
        battery = run_repaired_battery(bed, sesoi=prereg["sesoi"])
        verdict = classify(battery, sesoi=prereg["sesoi"])
        indep = independent_verify(bed)
        prod = battery["clauses"]["D_incremental_value"]["mean_incr_over_simple"]
        indep_agree = (indep > 0) == (prod > 0)
        report_core = {"schema": "mop-admission-battery-report/v2", "lane": lane, "prereg_sha256": prereg["prereg_sha256"],
                       "classification": verdict["classification"], "first_failing_clause": verdict["first_failing_clause"],
                       "clauses_passed": f"{verdict['clauses_passed']}/{verdict['clauses_total']}",
                       "clauses": battery["clauses"], "best_arch": battery["best_arch"],
                       "residual_learnable_headroom": battery["residual_learnable_headroom"],
                       "independent_verification": {"independent_mean_effect": round(indep, 3),
                                                    "producer_mean_effect": prod, "agree": indep_agree},
                       "claim_ceiling": prereg["claim_ceiling"]}
        report = {**report_core, "report_sha256": sha(report_core)}
        (REPORTS / f"MOP_{lane.replace('-', '_')}_ADMISSION_BATTERY_V2.json").write_text(json.dumps(_native(report), indent=2))
        lanes_out[lane] = {"classification": verdict["classification"], "first_fail": verdict["first_failing_clause"],
                           "passed": f"{verdict['clauses_passed']}/{verdict['clauses_total']}",
                           "D_incr": battery["clauses"]["D_incremental_value"]["mean_incr_over_simple"],
                           "D_lcb": battery["clauses"]["D_incremental_value"]["lower_95_cb"],
                           "F_n_arch_pass": battery["clauses"]["F_architecture_independence"]["n_passing"],
                           "F_linear_baseline": battery["clauses"]["F_architecture_independence"]["linear_baseline_incr"],
                           "residual_headroom": battery["residual_learnable_headroom"], "indep_agree": indep_agree}
    admitted = [k for k, v in lanes_out.items() if v["classification"] == "admitted"]
    print(json.dumps({"calibration": calib, "lanes": lanes_out}, indent=2, default=float))
    print("\nADMITTED (license a SMALL real-data canary):", admitted or "NONE")
    return admitted, lanes_out


if __name__ == "__main__":
    main()
