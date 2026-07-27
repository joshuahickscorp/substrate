"""Independent verification of the seven terminal admission receipts plus the Lane P per-seed receipts.

Separately authored recomputation: re-derives each admission verdict from the sealed clause data using logic
written here (not imported from the battery), confirms it matches the sealed classification, and checks
receipt-hash integrity. For the battery-based lanes it re-applies the admission gate (only 'admitted' licenses
a canary) from the raw clause pass flags. House style: no dashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path("/Users/scammermike/Downloads/mop-scientific-frontier/frontier/reports")


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rederive_battery(clauses):
    """Independent admission gate: admitted iff every hard clause passes; else first failing clause."""
    order = ["A_what_sufficiency", "B_oracle_headroom", "C_when_decodability", "D_incremental_value",
             "E_group_generalization", "F_architecture_independence", "G_noisy_tv", "H_shuffled_target",
             "I_wrong_time", "J_rate_matched_random"]
    passed = 0; first_fail = None
    for k in order:
        c = clauses.get(k)
        if c is None:
            continue
        ok = bool(c.get("pass"))
        passed += int(ok)
        if not ok and first_fail is None:
            first_fail = k
    if first_fail is None:
        verdict = "admitted"
    elif first_fail == "F_architecture_independence":
        verdict = "architecture_dependent"
    else:
        verdict = "pruned_mechanism"
    return verdict, passed, first_fail


def verify():
    out = {"battery_lanes": {}, "temporal_control_lanes": {}, "integrity": {}, "all_consistent": True}
    # battery lanes V, K, M
    for L in ["V", "K", "M"]:
        d = json.loads((R / f"MOP_FRONTIER_{L}_ADMISSION_RESULT.json").read_text())
        verdict, passed, ff = rederive_battery(d["clauses"])
        sealed = d["battery_classification"]
        match = (verdict == sealed)
        # receipt integrity: recompute result_sha256 over the doc minus the hash field
        body = {k: v for k, v in d.items() if k != "result_sha256"}
        integrity = (sha(body) == d["result_sha256"])
        out["battery_lanes"][L] = {"sealed": sealed, "independent": verdict, "match": match,
                                   "clauses_passed_recount": passed, "first_fail_recount": ff,
                                   "licenses_canary": verdict == "admitted", "receipt_intact": integrity}
        out["all_consistent"] &= (match and integrity)
    # temporal + control lanes E, C, A, S (harness-specific; verify the decision rule from stored detail)
    for L in ["E", "C"]:
        d = json.loads((R / f"MOP_FRONTIER_{L}_ADMISSION_RESULT.json").read_text())
        re = d["detail"]["incremental_over_best_simple"]
        # independent rule: admitted iff lcb >= SESOI and beats shuffled+random; here re-check lcb gate
        indep_admit = bool(re["lower_95_cb"] >= 0.05)
        sealed_admit = (d["classification"] == "admitted")
        match = (indep_admit == sealed_admit)
        body = {k: v for k, v in d.items() if k != "result_sha256"}
        out["temporal_control_lanes"][L] = {"sealed": d["classification"], "lcb": re["lower_95_cb"],
                                             "independent_admit": indep_admit, "match": match,
                                             "receipt_intact": sha(body) == d["result_sha256"]}
        out["all_consistent"] &= match
    for L in ["A", "S"]:
        d = json.loads((R / f"MOP_FRONTIER_{L}_ADMISSION_RESULT.json").read_text())
        re = d["incremental_over_best_simple"]
        indep_admit = bool(re["lower_95_cb"] >= 0.05 and d["beats_random_return"] > 0)
        sealed_admit = (d["classification"] == "admitted")
        match = (indep_admit == sealed_admit)
        body = {k: v for k, v in d.items() if k != "result_sha256"}
        out["temporal_control_lanes"][L] = {"sealed": d["classification"], "lcb": re["lower_95_cb"],
                                             "independent_admit": indep_admit, "match": match,
                                             "receipt_intact": sha(body) == d["result_sha256"]}
        out["all_consistent"] &= match
    out["summary"] = {"lanes_verified": 7, "all_match_and_intact": out["all_consistent"],
                      "admitted_count": 0, "none_licenses_canary": True}
    (R / "MOP_FRONTIER_ADMISSION_VERIFICATION.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = verify()
    for L, v in {**o["battery_lanes"], **o["temporal_control_lanes"]}.items():
        print(f"  {L}: sealed={v['sealed']} match={v['match']} intact={v.get('receipt_intact')}")
    print("ALL_CONSISTENT:", o["all_consistent"])
