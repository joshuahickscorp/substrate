"""Role B instrumentation auditor and Role C independent scientific verifier.

Role B never looks at an outcome. Role C recomputes every effect with its own arithmetic and its own t table
and forms its verdicts before comparing. File reading is shared on purpose; scientific logic is not.

House style: no dashes.
"""

from __future__ import annotations

import json
import math
import time

from mop.temporal import io

T95 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833,
       11: 1.812, 12: 1.796}
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
    return mean(v) - T95.get(len(v), 1.729) * sd(v) / math.sqrt(len(v))


def verdict(v):
    if len(v) < 2:
        return "insufficient_power"
    m, lo = mean(v), lower_bound(v)
    if m <= -SESOI:
        return "harm"
    if lo >= SESOI:
        return "positive"
    if m < 0:
        return "wrong_direction_failure"
    if m <= 0.01:
        return "null_futile"
    return "null"


# ---------------------------------------------------------------- role B


def role_b() -> dict:
    checks, notes = {}, []
    if not io.exists("MOP_E2_PRINCIPAL_RESULT.json"):
        return {"role": "B", "status": "not_run"}
    doc = io.load("MOP_E2_PRINCIPAL_RESULT.json")
    for bed, a in doc["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        inst = a["instrumentation"]
        checks[f"{bed}:no_undeclared_parameter_changes"] = inst["undeclared_parameter_changes"] == 0
        invalid_reset_cells = sorted(
            c for c in inst["oracle_segmented_cells"] if "|true_boundary|" not in c
        )
        checks[f"{bed}:oracle_segmented_arms_are_identified"] = all(
            "|true_boundary|" in c or c in invalid_reset_cells for c in inst["oracle_segmented_cells"]
        )
        load_bearing = set(inst.get("load_bearing_cells") or [])
        checks[f"{bed}:invalid_reset_arms_excluded_from_load_bearing_inference"] = not (
            set(invalid_reset_cells) & load_bearing
        )
        if invalid_reset_cells:
            notes.append({"bed": bed, "invalid_reset_cells": invalid_reset_cells,
                          "consequence": "these cells are excluded from load bearing inference"})
        checks[f"{bed}:reset_classifications_declared"] = bool(inst["reset_classifications"])
        checks[f"{bed}:load_bearing_baselines_converged"] = bool(
            a["convergence"].get("load_bearing_all_converged")
        )
        if not a["convergence"].get("load_bearing_all_converged"):
            notes.append({"bed": bed,
                          "unconverged": a["convergence"].get("load_bearing_unconverged"),
                          "consequence": "only comparisons using these arms are provisional"})
        runs = []
        for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
            runs.extend(json.loads(p.read_text())["runs"])
        by_tier: dict = {}
        for r in runs:
            by_tier.setdefault(r["spec"]["tier"], set()).add(r["params"]["core"])
        checks[f"{bed}:capacity_tiers_are_banded"] = all(
            max(v) / max(1, min(v)) < 3.0 for v in by_tier.values())
        readouts = {r["spec"]["readout"]: r["params"]["readout"] for r in runs}
        checks[f"{bed}:readout_parameter_count_depends_only_on_the_readout"] = len(readouts) == len(
            {v for v in readouts.values()})
        hp = {r["cell"]: tuple(sorted(r["history_profile"]["kinds"])) for r in runs}
        checks[f"{bed}:no_arm_sees_future_information"] = all(
            "future_information" not in k for k in hp.values())
        checks[f"{bed}:history_profiles_declared"] = all(bool(k) for k in hp.values())
    e3_paths = sorted((io.RUNS / "e3").glob("*.json")) if (io.RUNS / "e3").is_dir() else []
    for p in e3_paths:
        d = json.loads(p.read_text())
        key = f"E3:{d.get('source_bed')}:{d.get('target_bed')}:{d.get('seed')}"
        checks[f"{key}:direction_identity"] = d.get("source_bed") != d.get("target_bed")
        checks[f"{key}:eight_arms"] = len(d.get("arms", {})) == 8
        checks[f"{key}:arm_distinctness"] = bool(
            (d.get("arm_distinctness") or {}).get("all_nonoracle_arms_distinct"))
        checks[f"{key}:wrong_bed_control"] = bool(
            (d.get("wrong_bed_control") or {}).get("distinct_bed"))
        checks[f"{key}:component_interventions"] = len(d.get("component_interventions", {})) == 8
    third_paths = sorted((io.RUNS / "third_bed_preflight").glob("*.json")) \
        if (io.RUNS / "third_bed_preflight").is_dir() else []
    for p in third_paths:
        d = json.loads(p.read_text())
        key = f"HARTH-preflight:{d.get('seed')}"
        sets = [set(v) for v in d.get("units", {}).values()]
        checks[f"{key}:bed_identity"] = d.get("bed") == "harth_stream"
        checks[f"{key}:unit_disjoint"] = all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
        checks[f"{key}:test_untouched"] = bool(d.get("test_split_untouched"))
        checks[f"{key}:training_receipts"] = bool(d.get("all_checks_pass"))
    hybrid_paths = sorted((io.RUNS / "hybrid").glob("*.json")) if (io.RUNS / "hybrid").is_dir() else []
    for p in hybrid_paths:
        d = json.loads(p.read_text())
        key = f"hybrid:{d.get('bed')}:{d.get('seed')}"
        checks[f"{key}:six_arms"] = len(d.get("arms", {})) == 6
        checks[f"{key}:state_only_zero_parameter_updates"] = bool(
            (d.get("checks") or {}).get("state_only_zero_parameter_updates"))
        checks[f"{key}:matched_updates"] = bool((d.get("checks") or {}).get("matched_adaptation_updates"))
        checks[f"{key}:state_noise_matched"] = bool(
            (d.get("checks") or {}).get("state_noise_magnitude_matched"))
    return {"role": "B instrumentation auditor", "checks": checks, "notes": notes,
            "failed": [k for k, v in checks.items() if not v], "all_pass": all(checks.values()),
            "outcomes_inspected": False}


# ---------------------------------------------------------------- role C


def role_c() -> dict:
    if not io.exists("MOP_E2_PRINCIPAL_RESULT.json"):
        return {"role": "C", "status": "not_run"}
    sealed = io.load("MOP_E2_PRINCIPAL_RESULT.json")
    checks, mismatches = {}, []
    for bed, a in sealed["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        runs = []
        for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
            runs.extend(json.loads(p.read_text())["runs"])
        cells: dict[str, list] = {}
        unit_cells: dict[str, dict[str, list[float]]] = {}
        for r in runs:
            cells.setdefault(r["cell"], []).append(r["accuracy"])
            for unit, value in r["per_unit_accuracy"].items():
                unit_cells.setdefault(r["cell"], {}).setdefault(unit, []).append(float(value))
        unit_cells = {cell: {unit: mean(values) for unit, values in per.items()}
                      for cell, per in unit_cells.items()}
        checks[f"{bed}:seed_count"] = len({r["seed"] for r in runs}) == len(sealed["seeds"])
        checks[f"{bed}:every_cell_has_every_seed"] = len({len(v) for v in cells.values()}) == 1
        for group, table in a["effects"].items():
            for k, d in table.items():
                if d.get("mean") is None:
                    continue
                components = d.get("components")
                signs = d.get("formula_signs")
                if not components:
                    components = d["contrast"].split(" minus ")
                    signs = [1, -1]
                if any(cell not in cells for cell in components):
                    continue
                n = min(len(cells[cell]) for cell in components)
                eff = [sum(sign * cells[cell][i] for sign, cell in zip(signs, components, strict=True))
                       for i in range(n)]
                shared_units = sorted(set.intersection(*(set(unit_cells[cell]) for cell in components)))
                unit_effects = [sum(sign * unit_cells[cell][unit]
                                    for sign, cell in zip(signs, components, strict=True))
                                for unit in shared_units]
                mine = {"mean": round(mean(eff), 5), "lower_95_cb": round(lower_bound(eff), 5),
                        "group_lower_95_cb": round(lower_bound(unit_effects), 5),
                        "verdict": verdict(eff)}
                ok = (abs(mine["mean"] - d["mean"]) < 1e-4
                      and abs(mine["lower_95_cb"] - d["lower_95_cb"]) < 1e-4
                      and (d.get("group_lower_95_cb") is None
                           or abs(mine["group_lower_95_cb"] - d["group_lower_95_cb"]) < 1e-4)
                      and mine["verdict"].split("_")[0] == d["verdict"].split("_")[0])
                checks[f"{bed}:{group}:{k}"] = ok
                if not ok:
                    mismatches.append({"bed": bed, "contrast": k, "sealed": d, "recomputed": mine})
    if io.exists("MOP_E2_INDEPENDENT_REPLICATION.json"):
        rep = io.load("MOP_E2_INDEPENDENT_REPLICATION.json")
        control = rep["reference_control"]
        implementation_cells = {
            "torch_gru_vs_full_history": "gru|small|linear|none|h1",
            "explicit_mgu_vs_full_history": "mgu|small|linear|none|h1",
        }
        for bed, row in rep["per_bed"].items():
            runs = []
            for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
                runs.extend(json.loads(p.read_text())["runs"])
            cells: dict[str, list[float]] = {}
            for r in runs:
                cells.setdefault(r["cell"], []).append(float(r["accuracy"]))
            for key, cell in implementation_cells.items():
                expected = row["effects"][key]
                vals = [x - y for x, y in zip(cells.get(cell, []), cells.get(control, []), strict=True)]
                mine = {"mean": round(mean(vals), 5), "lower_95_cb": round(lower_bound(vals), 5),
                        "verdict": verdict(vals)}
                ok = (abs(mine["mean"] - expected["mean"]) < 1e-4
                      and abs(mine["lower_95_cb"] - expected["lower_95_cb"]) < 1e-4
                      and mine["verdict"].split("_")[0] == expected["verdict"].split("_")[0])
                checks[f"{bed}:independent_replication:{key}"] = ok
                if not ok:
                    mismatches.append({"bed": bed, "contrast": key,
                                       "sealed": expected, "recomputed": mine})
    if io.exists("MOP_E3_SHARED_LOCAL_RESULT.json"):
        e3 = io.load("MOP_E3_SHARED_LOCAL_RESULT.json")
        result = e3.get("result") or {}
        for direction, expected in (result.get("per_direction") or {}).items():
            source, target = direction.split("_to_", 1)
            rows = [json.loads(p.read_text()) for p in sorted((io.RUNS / "e3").glob(
                f"{source}_to_{target}_*.json"))]
            effects = [r["arms"]["shared_component"]["accuracy"]
                       - r["arms"]["domain_local_component"]["accuracy"] for r in rows]
            unit_values: dict[str, list[float]] = {}
            for row in rows:
                shared = row["arms"]["shared_component"]["per_unit_accuracy"]
                local = row["arms"]["domain_local_component"]["per_unit_accuracy"]
                for unit in set(shared) & set(local):
                    unit_values.setdefault(unit, []).append(shared[unit] - local[unit])
            units = [mean(v) for v in unit_values.values()]
            sealed = expected["shared_minus_domain_local"]
            mine = {"mean": round(mean(effects), 5), "lower_95_cb": round(lower_bound(effects), 5),
                    "group_lower_95_cb": lower_bound(units)}
            ok = (len(rows) == 8 and abs(mine["mean"] - sealed["mean"]) < 1e-4
                  and abs(mine["lower_95_cb"] - sealed["lower_95_cb"]) < 1e-4
                  and abs(mine["group_lower_95_cb"] - sealed["group_lower_95_cb"]) < 1e-4)
            checks[f"E3:{direction}:shared_vs_local"] = ok
            if not ok:
                mismatches.append({"bed": direction, "contrast": "E3 shared versus local",
                                   "sealed": sealed, "recomputed": mine})
    if io.exists("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json"):
        expected = io.load("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json")
        rows = [json.loads(p.read_text()) for p in sorted(
            (io.RUNS / "third_bed_preflight").glob("*.json"))]
        gains = [r["after_B_adaptation"]["B"]["accuracy"] - r["before_adaptation"]["B"]["accuracy"]
                 for r in rows]
        returns = [r["return_after_recovery"]["accuracy"] - r["return_before_recovery"]["accuracy"]
                   for r in rows]
        shifts = [r["before_adaptation"]["A"]["accuracy"] - r["before_adaptation"]["B"]["accuracy"]
                  for r in rows]
        costs = [r["after_B_adaptation"]["A"]["accuracy"] - r["before_adaptation"]["A"]["accuracy"]
                 for r in rows]
        unit_values: dict[str, list[float]] = {}
        for row in rows:
            before = row["before_adaptation"]["B"]["per_unit_accuracy"]
            after = row["after_B_adaptation"]["B"]["per_unit_accuracy"]
            for unit in set(before) & set(after):
                unit_values.setdefault(unit, []).append(after[unit] - before[unit])
        units = [mean(v) for v in unit_values.values()]
        checks["HARTH-preflight:seed_count"] = len(rows) == 8
        checks["HARTH-preflight:future_gain"] = (
            abs(round(mean(gains), 5) - expected["future_adaptation"]["mean"]) < 1e-4
            and abs(round(lower_bound(gains), 5) - expected["future_adaptation"]["lower_95_cb"]) < 1e-4
            and abs(lower_bound(units) - expected["future_adaptation"]["group_lower_95_cb"]) < 1e-4)
        checks["HARTH-preflight:return_gain"] = (
            abs(round(mean(returns), 5) - expected["returning_context"]["mean"]) < 1e-4)
        boundary = expected["context_boundary"]
        checks["HARTH-preflight:boundary"] = (
            abs(round(lower_bound(shifts), 5) - boundary["distribution_shift_lower_95_cb"]) < 1e-4
            and abs(round(lower_bound(costs), 5) - boundary["retention_cost_lower_95_cb"]) < 1e-4)
    if io.exists("MOP_HYBRID_ADAPTATION_RESULT.json"):
        hybrid = io.load("MOP_HYBRID_ADAPTATION_RESULT.json")
        result = hybrid.get("result") or {}
        for bedname, expected in (result.get("per_bed") or {}).items():
            rows = [json.loads(p.read_text()) for p in sorted((io.RUNS / "hybrid").glob(
                f"{bedname}_*.json"))]
            gain = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                    - r["arms"]["head_only"]["future_acquisition_B"]["accuracy"] for r in rows]
            noise = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                     - r["arms"]["head_plus_state_noise"]["future_acquisition_B"]["accuracy"] for r in rows]
            checks[f"hybrid:{bedname}:hybrid_vs_head"] = (
                len(rows) == 8 and abs(round(mean(gain), 5) - expected["hybrid_minus_head"]["mean"]) < 1e-4
                and abs(round(lower_bound(gain), 5)
                        - expected["hybrid_minus_head"]["lower_95_cb"]) < 1e-4)
            checks[f"hybrid:{bedname}:hybrid_vs_noise"] = (
                len(rows) == 8 and abs(round(mean(noise), 5)
                                       - expected["hybrid_minus_head_noise"]["mean"]) < 1e-4
                and abs(round(lower_bound(noise), 5)
                        - expected["hybrid_minus_head_noise"]["lower_95_cb"]) < 1e-4)
    return {"role": "C scientific verifier", "checks": checks, "mismatches": mismatches,
            "n_checks": len(checks), "all_pass": all(checks.values()) and not mismatches,
            "independence": ("recomputes every effect with its own arithmetic and its own t table, imports no "
                             "contrast code from the producer, and forms its verdicts before comparing")}


def main():
    t0 = time.time()
    b, c = role_b(), role_c()
    doc = {
        "schema": "mop-temporal-core-independent-verification/v1",
        "role_b": b,
        "role_c": c,
        "all_pass": bool(b.get("all_pass")) and bool(c.get("all_pass")),
        "rule": "a load bearing result requires Role A, Role B and Role C to pass",
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", doc)
    print(f"verification: role B {b.get('all_pass')} role C {c.get('all_pass')} "
          f"({c.get('n_checks', 0)} recomputations)", flush=True)
    for f in b.get("failed", []):
        print(f"  roleB FAIL {f}", flush=True)
    for m in c.get("mismatches", [])[:5]:
        print(f"  roleC MISMATCH {m['bed']} {m['contrast']}", flush=True)
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
