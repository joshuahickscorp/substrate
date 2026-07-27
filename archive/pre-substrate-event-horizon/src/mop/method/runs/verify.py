"""Role C: the independent scientific verifier.

This module recomputes every reported effect from the raw run receipts with its own arithmetic. It does not
import the producer's contrast code, its own t table is written out rather than imported, and it forms its
own verdicts before comparing them to the sealed ones. File reading and hashing are shared on purpose;
scientific logic is not.

House style: no dashes.
"""

from __future__ import annotations

import json
import math
import time

from mop.method import io

# one sided 95 percent t values, written here rather than imported, so an error in one table cannot hide
T = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}
SESOI = 0.05


def mean(v):
    return sum(v) / len(v) if v else 0.0


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def lower_bound(v):
    if len(v) < 2:
        return mean(v)
    return mean(v) - T.get(len(v), 1.729) * sd(v) / math.sqrt(len(v))


def verdict(v, sesoi=SESOI):
    if len(v) < 2:
        return "insufficient_power"
    m, lo = mean(v), lower_bound(v)
    if m <= -abs(sesoi):
        return "harm"
    if lo >= sesoi:
        return "positive"
    if m < 0:
        return "wrong_direction_failure"
    if m <= 0.01:
        return "null_futile"
    return "null"


# ---------------------------------------------------------------- E1


def verify_e1() -> dict:
    sealed = io.load("MOP_PRINCIPAL_EXPERIMENT_1.json")
    checks, mismatches = {}, []
    for bedname, a in sealed["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        files = sorted((io.RUNS / "principal").glob(f"{bedname}_*.json"))
        runs = [json.loads(p.read_text()) for p in files]
        cells: dict[str, list[float]] = {}
        units_seen: dict[str, set] = {}
        for r in runs:
            for c in r["runs"]:
                cells.setdefault(f"{c['core']}_{c['readout']}", []).append(c["accuracy"])
                units_seen.setdefault(f"{c['core']}_{c['readout']}", set()).update(c["per_unit_accuracy"])
            cells.setdefault("external", []).append(r["external_baseline"]["accuracy"])
        checks[f"{bedname}:seed_count"] = len(runs) == len(sealed["seeds"])
        checks[f"{bedname}:no_undeclared_parameter_changes"] = all(
            not c["undeclared_changes"] for r in runs for c in r["runs"]
        )
        checks[f"{bedname}:every_cell_present"] = set(sealed["cells"]) <= set(cells)
        checks[f"{bedname}:capacity_matched_within_1_percent"] = all(
            abs(c["core_params"] - r["runs"][0]["core_params"]) / max(1, r["runs"][0]["core_params"]) < 0.01
            for r in runs
            for c in r["runs"]
        )
        for name, d in a["contrasts"].items():
            left, right = d["contrast"].split(" minus ")
            if left not in cells or right not in cells:
                continue
            eff = [x - y for x, y in zip(cells[left], cells[right], strict=True)]
            mine = {"mean": round(mean(eff), 5), "lower_95_cb": round(lower_bound(eff), 5),
                    "verdict": verdict(eff)}
            same = (abs(mine["mean"] - d["mean"]) < 1e-4 and abs(mine["lower_95_cb"] - d["lower_95_cb"]) < 1e-4
                    and mine["verdict"].split("_")[0] == d["verdict"].split("_")[0])
            checks[f"{bedname}:{name}"] = same
            if not same:
                mismatches.append({"bed": bedname, "contrast": name, "sealed": d, "recomputed": mine})
    return {"role": "C scientific verifier", "experiment": "E1", "checks": checks,
            "mismatches": mismatches, "all_pass": all(checks.values()) and not mismatches}


# ---------------------------------------------------------------- E4


def verify_e4() -> dict:
    sealed = io.load("MOP_PRINCIPAL_EXPERIMENT_2.json")
    checks, mismatches = {}, []
    for bedname, a in sealed["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        runs = [json.loads(p.read_text()) for p in sorted((io.RUNS / "e4_principal").glob(f"{bedname}_*.json"))]
        checks[f"{bedname}:seed_count"] = len(runs) == len(sealed["seeds"])
        checks[f"{bedname}:state_arms_never_updated_a_parameter"] = all(
            r["arms"][arm]["parameter_updates"] == 0 and r["arms"][arm]["changed_param_count"] == 0
            for r in runs
            for arm in ("state_only", "state_noise")
        )
        checks[f"{bedname}:no_undeclared_parameter_changes"] = all(
            not r["arms"][arm]["undeclared_changes"] for r in runs for arm in sealed["arms"]
        )
        checks[f"{bedname}:budget_matched"] = len({r["arms"][arm]["trace"].get("updates")
                                                   for r in runs for arm in sealed["arms"]}) == 1
        for group, key in (("acquisition_contrasts", "acquisition_B"), ("retention_contrasts", "retention_A")):
            for name, d in a[group].items():
                left, right = name.rsplit("_vs_", 1)
                eff = [r["arms"][left][key] - r["arms"][right][key] for r in runs]
                mine = {"mean": round(mean(eff), 5), "lower_95_cb": round(lower_bound(eff), 5),
                        "verdict": verdict(eff)}
                same = (abs(mine["mean"] - d["mean"]) < 1e-4
                        and abs(mine["lower_95_cb"] - d["lower_95_cb"]) < 1e-4
                        and mine["verdict"].split("_")[0] == d["verdict"].split("_")[0])
                checks[f"{bedname}:{group}:{name}"] = same
                if not same:
                    mismatches.append({"bed": bedname, "contrast": f"{group}:{name}", "sealed": d,
                                       "recomputed": mine})
    return {"role": "C scientific verifier", "experiment": "E4", "checks": checks,
            "mismatches": mismatches, "all_pass": all(checks.values()) and not mismatches}


def main():
    t0 = time.time()
    out = {}
    if io.exists("MOP_PRINCIPAL_EXPERIMENT_1.json"):
        out["E1"] = verify_e1()
    if io.exists("MOP_PRINCIPAL_EXPERIMENT_2.json"):
        out["E4"] = verify_e4()
    doc = {
        "schema": "mop-independent-verification/v1",
        "role": "C",
        "independence": (
            "this module recomputes effects, lower bounds and verdicts with its own arithmetic and its own t "
            "table, imports no contrast code from the producer, and forms its verdicts before comparing"
        ),
        "results": out,
        "all_pass": all(v["all_pass"] for v in out.values()) if out else False,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_METHOD_INDEPENDENT_VERIFICATION.json", doc)
    print(f"role C: {[(k, v['all_pass']) for k, v in out.items()]}", flush=True)
    for k, v in out.items():
        for c, ok in v["checks"].items():
            if not ok:
                print(f"  FAIL {k} {c}", flush=True)
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
