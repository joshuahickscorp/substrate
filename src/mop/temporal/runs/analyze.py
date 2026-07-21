"""E2 principal analysis: factorial effects, the per factor reports, and the hypothesis fold.

Every per factor report is a projection of the same sealed effect table, so a report that disagrees with the
factorial is a bug rather than a second opinion.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mop.method import gate, power
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import hypotheses as H
from mop.temporal import io
from mop.temporal.runs import e2

BEDS = ("har_stream", "speech_stream", "harth_stream")
PRINCIPAL_BEDS = ("har_stream", "speech_stream")


def interaction(series: dict, units: dict, cells: tuple[str, str, str, str], label: str) -> dict:
    """Difference in differences from four already sealed factorial cells."""
    return AN.interaction(series, cells, e2.PREREG, units, label)


def load_runs(bed: str) -> list[dict]:
    """Load immutable principal receipts with exact key corrections taking precedence."""
    out = {}
    for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            out[(int(row["seed"]), row["cell"])] = row
    for p in sorted((io.RUNS / "e2_principal_corrections").glob(f"capacity_{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            key = (int(row["seed"]), row["cell"])
            original = out.get(key) or {}
            lo, hi = A.TIER_RANGE["large"]
            if not lo <= int((original.get("params") or {}).get("core", 0)) <= hi:
                out[key] = row
    return list(out.values())


def convergence(bed: str) -> dict:
    p = io.RUNS / "e2_converge" / f"converge_{bed}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def independent_bed_difference(left: dict, right: dict, label: str) -> dict:
    """Welch bounded difference between effects measured on unrelated bed units."""
    seed_left, seed_right = left.get("per_seed_effects", []), right.get("per_seed_effects", [])
    unit_left = list((left.get("per_unit_effects") or {}).values())
    unit_right = list((right.get("per_unit_effects") or {}).values())

    def estimate(a, b):
        if len(a) < 2 or len(b) < 2:
            return None, None, None, None
        ma, mb = float(np.mean(a)), float(np.mean(b))
        va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
        term_a, term_b = va / len(a), vb / len(b)
        se = float(np.sqrt(term_a + term_b))
        denom = term_a ** 2 / (len(a) - 1) + term_b ** 2 / (len(b) - 1)
        df = (term_a + term_b) ** 2 / denom if denom else len(a) + len(b) - 2
        critical = power.t95(max(2, int(np.floor(df)) + 1))
        return ma - mb, ma - mb - critical * se, ma - mb + critical * se, df

    mean_seed, lower_seed, _, seed_df = estimate(seed_left, seed_right)
    mean_unit, lower_unit, upper_unit, unit_df = estimate(unit_left, unit_right)
    if mean_seed is None or mean_unit is None:
        return {"contrast": label, "estimand": "independent_bed_difference_in_differences",
                "verdict": "insufficient_power", "mean": None}
    if mean_seed <= -e2.PREREG["harm_boundary"]:
        verdict = "harm"
    elif lower_seed >= io.SESOI and lower_unit >= io.SESOI:
        verdict = "positive"
    elif mean_seed < 0:
        verdict = "wrong_direction_failure"
    elif mean_seed <= e2.PREREG["futility_boundary"]:
        verdict = "null_futile"
    else:
        verdict = "null"
    component_parameters = int(left.get("component_parameter_sum") or 0) + int(
        right.get("component_parameter_sum") or 0)
    converged = all((d.get("convergence") or {}).get("all_converged") for d in (left, right))
    sufficient = all(d.get("estimator_sufficient") is True for d in (left, right))
    if not sufficient:
        verdict = "insufficient_power"
    group_floor = lower_unit >= io.SESOI
    return {"contrast": label, "estimand": "independent_bed_difference_in_differences",
            "mean": round(mean_seed, 5), "lower_95_cb": round(lower_seed, 5),
            "group_mean": round(mean_unit, 5), "group_lower_95_cb": round(lower_unit, 5),
            "group_upper_95_cb": round(upper_unit, 5),
            "group_heterogeneity": round(float(np.sqrt((np.var(unit_left, ddof=1) +
                                                         np.var(unit_right, ddof=1)) / 2)), 5),
            "n_units": len(unit_left) + len(unit_right), "seed_welch_df": round(seed_df, 3),
            "group_welch_df": round(unit_df, 3), "verdict": verdict,
            "component_convergence": {
                "left": (left.get("convergence") or {}).get("classification"),
                "right": (right.get("convergence") or {}).get("classification")},
            "convergence": {"all_converged": converged,
                            "classification": "converged" if converged else "provisional_unconverged"},
            "estimator_sufficient": sufficient,
            "component_parameter_sum": component_parameters,
            "cost_adjusted_effect_per_100k_parameters": (
                round(mean_seed * 100_000 / component_parameters, 5) if component_parameters else None),
            "component_floor_status": "passes" if converged and group_floor else "provisional_or_below_floor",
            "left_per_seed_effects": seed_left, "right_per_seed_effects": seed_right,
            "left_per_unit_effects": left.get("per_unit_effects", {}),
            "right_per_unit_effects": right.get("per_unit_effects", {})}


def between_bed_interactions(per_bed: dict, group: str) -> dict:
    left_name, right_name = PRINCIPAL_BEDS
    left = per_bed.get(left_name, {}).get("effects", {}).get(group, {})
    right = per_bed.get(right_name, {}).get("effects", {}).get(group, {})
    return {key: independent_bed_difference(left[key], right[key],
            f"{key} on {left_name} minus {key} on {right_name}")
            for key in sorted(set(left) & set(right))}


def optimization_interaction(bed: str) -> dict:
    """Capacity by optimization DID from same-family strict curves and an exact compute match."""
    paths = {tier: io.RUNS / "e2_optimization_corrections" /
             f"optimization_{bed}_{tier}.json" for tier in ("large", "small")}
    if not all(p.is_file() for p in paths.values()):
        return {"estimand": "optimization_by_capacity", "verdict": "missing_receipt", "mean": None}
    large, small = (json.loads(paths[tier].read_text()) for tier in ("large", "small"))
    budgets = {"large_same_update": int(large["same_update_anchor"]),
               "large_convergence": int(large["selected_checkpoint"]),
               "small_same_compute": int(small["compute_match"]["small_steps"]),
               "small_convergence": int(small["selected_checkpoint"])}
    arms = {"large_convergence": (large, budgets["large_convergence"],
                                   "large_model_at_strict_selected_convergence"),
            "large_same_update": (large, budgets["large_same_update"],
                                  "large_model_at_same_update_count"),
            "small_convergence": (small, budgets["small_convergence"],
                                   "small_model_at_strict_selected_convergence"),
            "small_same_compute": (small, budgets["small_same_compute"],
                                    "small_model_at_same_compute")}

    def at(document, key, budget):
        table = document.get(key) or {}
        return table.get(str(budget), table.get(budget))

    records, checks = {}, {}
    expected_seeds = set(e2.CONVERGENCE_SEEDS)
    for name, (document, budget, role) in arms.items():
        rows = at(document, "arm_records", budget) or []
        seed_ids = [int(row.get("seed", -1)) for row in rows]
        role_row = (document.get("four_contrast_roles") or {}).get(role) or {}
        records[name] = {int(row["seed"]): row for row in rows if "seed" in row}
        checks[f"{name}_exact_seed_identity"] = set(seed_ids) == expected_seeds \
            and len(seed_ids) == len(set(seed_ids)) == len(expected_seeds)
        checks[f"{name}_role_identity"] = role_row.get("budget") == budget \
            and role_row.get("records") == rows
        checks[f"{name}_parameter_exposure_identity"] = bool(rows) and all(
            int(row.get("updates", -1)) == budget
            and int(row.get("parameter_update_exposure", -1))
            == int(row.get("trainable_param_count", 0)) * budget for row in rows)
    checks["exact_role_inventory"] = (
        set(large.get("four_contrast_roles") or {}) == {
            "large_model_at_same_update_count", "large_model_at_strict_selected_convergence"}
        and set(small.get("four_contrast_roles") or {}) == {
            "small_model_at_same_compute", "small_model_at_strict_selected_convergence"})
    checks["paired_specs_differ_only_by_tier"] = (
        large.get("bed") == small.get("bed") == bed
        and large.get("spec", {}).get("family") == small.get("spec", {}).get("family") == "gru"
        and large.get("spec", {}).get("tier") == "large" and small.get("spec", {}).get("tier") == "small"
        and {k: v for k, v in large.get("spec", {}).items() if k != "tier"}
        == {k: v for k, v in small.get("spec", {}).items() if k != "tier"})
    compute_errors = []
    for seed in expected_seeds:
        large_row = records.get("large_same_update", {}).get(seed, {})
        small_row = records.get("small_same_compute", {}).get(seed, {})
        large_exposure = float(large_row.get("parameter_update_exposure", 0))
        small_exposure = float(small_row.get("parameter_update_exposure", 0))
        compute_errors.append(abs(large_exposure - small_exposure) / large_exposure
                              if large_exposure else float("inf"))
    checks["actual_compute_match_within_tolerance"] = bool(compute_errors) and max(compute_errors) <= 0.0001
    checks["producer_design_checks_pass"] = bool(large.get("all_checks_pass")) and bool(
        small.get("all_checks_pass"))
    for seed in expected_seeds:
        inventories = [set(records.get(name, {}).get(seed, {}).get("per_unit_accuracy") or {})
                       for name in arms]
        checks[f"seed_{seed}_complete_identical_unit_inventory"] = (
            all(inventories) and all(units == inventories[0] for units in inventories[1:]))
    receipts_valid = all(checks.values())
    components = ["large_convergence", "large_same_update", "small_convergence", "small_same_compute"]
    if not receipts_valid:
        return {"contrast": "optimization by capacity", "estimand": "difference_in_differences",
                "formula_signs": [1, -1, -1, 1], "components": components, "budgets": budgets,
                "verdict": "invalid_receipt", "mean": None, "receipt_checks": checks,
                "classification": "invalid_receipt"}
    effects = [records[components[0]][seed]["score"] - records[components[1]][seed]["score"]
               - records[components[2]][seed]["score"] + records[components[3]][seed]["score"]
               for seed in e2.CONVERGENCE_SEEDS]
    per_unit = {}
    for seed in e2.CONVERGENCE_SEEDS:
        units = set(records[components[0]][seed]["per_unit_accuracy"])
        for unit in units:
            per_unit.setdefault(unit, []).append(
                records[components[0]][seed]["per_unit_accuracy"][unit]
                - records[components[1]][seed]["per_unit_accuracy"][unit]
                - records[components[2]][seed]["per_unit_accuracy"][unit]
                + records[components[3]][seed]["per_unit_accuracy"][unit])
    units = {unit: float(np.mean(values)) for unit, values in per_unit.items()}
    unit_values = list(units.values())
    out = power.decide(effects, e2.PREREG)
    out["upper_95_cb"] = round(-power.lcb([-x for x in effects]), 5)
    raw_verdict = out["verdict"]
    group_lcb = round(power.lcb(unit_values), 5) if len(unit_values) > 1 else None
    group_ucb = round(-power.lcb([-x for x in unit_values]), 5) if len(unit_values) > 1 else None
    both_converged = large.get("classification") == small.get("classification") == "converged"
    scientific_verdict = ("provisional_unconverged" if not both_converged else
                          "positive_seed_only_group_floor_not_met" if raw_verdict == "positive"
                          and (group_lcb is None or group_lcb < io.SESOI) else raw_verdict)
    exposure_by_arm = {name: sum(float(row["parameter_update_exposure"])
                                 for row in records[name].values()) for name in components}
    exposure_denominator = sum(exposure_by_arm.values())
    raw_adjusted = (round(float(out["mean"]) * 1_000_000_000 / exposure_denominator, 5)
                    if exposure_denominator else None)
    out.update({"contrast": "(large convergence minus same update) minus (small convergence minus same compute)",
                "estimand": "difference_in_differences", "formula_signs": [1, -1, -1, 1],
                "components": components, "budgets": budgets, "compute_match": small["compute_match"],
                "receipt_checks": checks, "receipts_valid": receipts_valid,
                "per_seed_effects": [round(x, 5) for x in effects],
                "per_unit_effects": {k: round(v, 5) for k, v in units.items()},
                "group_mean": round(float(np.mean(unit_values)), 5) if unit_values else None,
                "group_lower_95_cb": group_lcb, "group_upper_95_cb": group_ucb,
                "group_heterogeneity": round(float(np.std(unit_values, ddof=1)), 5)
                if len(unit_values) > 1 else None, "n_units": len(unit_values),
                "raw_statistical_verdict": raw_verdict, "verdict": scientific_verdict,
                "classification": "converged" if both_converged else "provisional_unconverged",
                "component_convergence": {"large": large.get("classification"),
                                          "small": small.get("classification")},
                "parameter_update_exposure_by_arm": exposure_by_arm,
                "parameter_update_exposure_denominator": exposure_denominator,
                "raw_cost_adjusted_effect_per_billion_parameter_updates": raw_adjusted,
                "cost_adjusted_effect_per_billion_parameter_updates": raw_adjusted if both_converged else None,
                "cost_adjusted_status": "scientific" if both_converged else "provisional_unconverged",
                "component_floor_status": ("passes" if both_converged and raw_verdict == "positive"
                                           and group_lcb is not None and group_lcb >= io.SESOI
                                           else "provisional_or_below_floor")})
    return out


def analyse_bed(bed: str) -> dict:
    runs = load_runs(bed)
    if not runs:
        return {"bed": bed, "status": "no_runs"}
    series, units = e2._series(runs)
    conv = convergence(bed)
    rec = AN.recover(series, e2.PREREG, units)
    ref = AN.name()
    rec["effects"]["core_by_readout"] = {
        f"gru_vs_{f}_{r}": interaction(series, units,
            (AN.name(readout=r), ref, AN.name(family=f, readout=r), AN.name(family=f)),
            f"(gru {r} minus linear) minus ({f} {r} minus linear)")
        for f in ("pooled", "histmlp") for r in ("mlp1", "mlp_strong")}
    rec["effects"]["core_by_capacity"] = {
        f"gru_vs_{f}_{t}": interaction(series, units,
            (AN.name(tier=t), ref, AN.name(family=f, tier=t), AN.name(family=f)),
            f"(gru {t} minus small) minus ({f} {t} minus small)")
        for f in ("pooled", "histmlp", "tcn") for t in ("micro", "medium", "large")}
    rec["effects"]["core_by_horizon"] = {
        f"mgu_vs_gru_h{h}": interaction(series, units,
            (AN.name(family="mgu", reset=f"horizon_{h}"),
             AN.name(family="mgu", reset="horizon_full"),
             AN.name(reset=f"horizon_{h}"), AN.name(reset="horizon_full")),
            f"(mgu h{h} minus full) minus (gru h{h} minus full)")
        for h in (5, 45, 90)}
    rec["effects"]["readout_by_capacity"] = {
        f"{f}_strong_large_vs_small": interaction(series, units,
            (AN.name(family=f, tier="large", readout="mlp_strong"),
             AN.name(family=f, tier="large"), AN.name(family=f, readout="mlp_strong"),
             AN.name(family=f)),
            f"({f} large strong minus linear) minus ({f} small strong minus linear)")
        for f in ("gru", "pooled", "histmlp")}
    rec["effects"]["history_by_architecture"] = {
        f"histmlp_vs_tcn_k{k}": interaction(series, units,
            (AN.name(family="histmlp", history_k=k), AN.name(family="histmlp"),
             AN.name(family="tcn", history_k=k), AN.name(family="tcn")),
            f"(histmlp k{k} minus k1) minus (tcn k{k} minus k1)")
        for k in (5, 20, "full_window")}
    params = {r["cell"]: r["params"] for r in runs}
    conv_configs = conv.get("configs") or {}

    def conv_status(cell: str) -> str:
        if cell in conv_configs:
            return conv_configs[cell].get("classification", "not_measured")
        # horizon full and no reset are the same call path and reset index list under the sealed factorial.
        alias = cell.replace("|horizon_full|", "|none|")
        return (conv_configs.get(alias) or {}).get("classification", "not_measured")

    for table in rec["effects"].values():
        for effect in table.values():
            contrast = effect.get("contrast") or ""
            cells = effect.get("components") or (contrast.split(" minus ") if " minus " in contrast else [])
            statuses = {c: conv_status(c) for c in cells}
            effect["convergence"] = {
                "cells": statuses,
                "all_converged": bool(statuses) and all(v == "converged" for v in statuses.values()),
                "classification": ("converged" if statuses and all(v == "converged" for v in statuses.values())
                                   else "provisional_unconverged_or_unmeasured"),
            }
            totals = [params.get(c, {}).get("total", 0) for c in cells]
            effect["component_parameter_sum"] = sum(totals)
            effect["cost_adjusted_effect_per_100k_parameters"] = (
                round(float(effect["mean"]) * 100_000 / max(1, sum(totals)), 5)
                if effect.get("mean") is not None else None)
            effect["component_floor_status"] = (
                "passes" if effect["convergence"]["all_converged"]
                and (effect.get("group_lower_95_cb") or float("-inf")) >= io.SESOI
                else "provisional_or_below_floor")
    means = {k: round(float(np.mean(v)), 5) for k, v in series.items()}
    wall = {}
    for r in runs:
        wall.setdefault(r["cell"], []).append(float(r.get("wall_seconds", 0)))
    best = max(means, key=means.get)
    return {
        "bed": bed,
        "n_seeds": len({r["seed"] for r in runs}),
        "n_cells": len(series),
        "cell_means": means,
        "cell_sds": {k: round(float(np.std(v, ddof=1)), 5) for k, v in series.items() if len(v) > 1},
        "cell_params": params,
        "metric_panel": {
            "primary_task_performance": means,
            "transition_performance": {"status": "not_measured_under_E2_classification_authority"},
            "future_adaptation": {"status": "not_measured_under_E2_classification_authority"},
            "retention": {"status": "not_measured_under_E2_classification_authority"},
            "return_recovery": {"status": "not_measured_under_E2_classification_authority"},
            "latency_wall_seconds_mean": {k: round(float(np.mean(v)), 5) for k, v in wall.items()},
            "memory_parameter_bytes": {k: v["total"] * 4 for k, v in params.items()},
            "training_compute_updates": {r["cell"]: r["updates"] for r in runs},
            "parameter_count": {k: v["total"] for k, v in params.items()},
            "rule": "unmeasured outcomes remain explicit and are never folded into one utility",
        },
        "best_cell": best,
        "best_mean": means[best],
        "majority_rate": round(float(np.mean([r.get("prediction_concentration", 0) for r in runs])), 5),
        "effects": rec["effects"],
        "findings": rec["findings"],
        "recovered": rec["recovered"],
        "convergence": {"all_converged": conv.get("all_converged"),
                        "unconverged": conv.get("unconverged", []),
                        "grid": conv.get("grid"),
                        "load_bearing_cells": conv.get("load_bearing_cells", []),
                        "load_bearing_all_converged": conv.get("load_bearing_all_converged", False),
                        "load_bearing_unconverged": conv.get("load_bearing_unconverged", []),
                        "configs": {k: {kk: v.get(kk) for kk in (
                            "classification", "selected_checkpoint", "second_half_movement",
                            "residual_slope", "seeds", "curve")}
                                    for k, v in conv_configs.items()}},
        "instrumentation": {
            "undeclared_parameter_changes": sum(len(r["undeclared_changes"]) for r in runs),
            "reset_classifications": sorted({r["reset_witness"]["classification"] for r in runs}),
            "oracle_segmented_cells": sorted({r["cell"] for r in runs
                                              if r["reset_witness"]["classification"] == "oracle_segmented"}),
            "load_bearing_cells": list(e2.LOAD_BEARING_CONVERGENCE_CELLS),
        },
    }


def per_factor_reports(per_bed: dict) -> dict:
    """One projection per factor, each carrying the numbers a reader would otherwise have to assemble."""
    out = {}

    def gather(group: str) -> dict:
        return {b: {k: {kk: v.get(kk) for kk in (
                    "mean", "lower_95_cb", "upper_95_cb", "group_mean", "group_lower_95_cb", "group_upper_95_cb",
                    "group_heterogeneity", "per_seed_effects", "per_unit_effects", "components",
                    "formula_signs", "estimand", "verdict", "estimator_sufficient",
                    "component_parameter_sum",
                    "cost_adjusted_effect_per_100k_parameters", "component_floor_status", "convergence")}
                    for k, v in a["effects"][group].items()}
                for b, a in per_bed.items() if a.get("status") != "no_runs"}

    def optimization_arms(bed: str) -> dict:
        docs = {}
        for tier in ("large", "small"):
            p = io.RUNS / "e2_optimization_corrections" / f"optimization_{bed}_{tier}.json"
            docs[tier] = json.loads(p.read_text()) if p.is_file() else {}
        if not all(docs.values()):
            return {"status": "missing_append_only_optimization_corrections"}
        large, small = docs["large"], docs["small"]
        return {
            "large_model_at_same_update_count": large["four_contrast_roles"][
                "large_model_at_same_update_count"],
            "large_model_at_convergence": large["four_contrast_roles"][
                "large_model_at_strict_selected_convergence"],
            "small_model_at_same_compute": small["four_contrast_roles"]["small_model_at_same_compute"],
            "small_model_at_convergence": small["four_contrast_roles"][
                "small_model_at_strict_selected_convergence"],
            "compute_match": small["compute_match"], "same_family": "gru",
            "capacity_levels": ["small", "large"]}

    out["MOP_CORE_ARCHITECTURE_REPORT.json"] = {
        "schema": "mop-core-architecture-report/v1",
        "families": list(A.FAMILIES),
        "recurrent": list(A.RECURRENT),
        "stateless": list(A.STATELESS),
        "versus_reference_gru": gather("architecture"),
        "recurrent_versus_stateless": gather("recurrence_versus_best_stateless"),
        "materially_independent_implementations": ["gru uses torch fused kernels",
                                                   "mgu is stepped in an explicit python loop here"],
    }
    out["MOP_CORE_CAPACITY_REPORT.json"] = {
        "schema": "mop-core-capacity-report/v1",
        "tiers": {t: list(A.TIER_RANGE[t]) for t in A.CAPACITY_TIERS},
        "effects": gather("capacity"),
        "measured_parameters": {b: a["cell_params"] for b, a in per_bed.items()
                                if a.get("status") != "no_runs"},
        "monotonic": {b: a["findings"]["capacity_monotonic"] for b, a in per_bed.items()
                      if a.get("status") != "no_runs"},
    }
    out["MOP_READOUT_CAPACITY_REPORT.json"] = {
        "schema": "mop-readout-capacity-report/v1",
        "readouts": list(A.READOUTS),
        "effects": gather("readout"),
        "capacity_by_readout": gather("capacity_by_readout"),
    }
    horizon_gate = state_horizon_gate(per_bed)
    out["MOP_STATE_HORIZON_REPORT.json"] = {
        "schema": "mop-state-horizon-report/v1",
        "horizons": ["1", "2", "5", "10", "20", "45", "90", "full"],
        "effects": gather("horizon"),
        "shortest_sufficient_horizon": {b: a["findings"]["horizon_threshold"]
                                        for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "principal_gate": horizon_gate,
        "classification": ("two_bed_persistent_state_mechanism" if horizon_gate["all_pass"]
                           else "bed_specific_or_unresolved_horizon_effect"),
        "rule": ("a principal mechanism positive requires converged group bounds showing that full state "
                 "beats horizon 45 and both a fixed misaligned reset and a rate matched random reset by "
                 "the SESOI on each principal bed; one bed cannot license a two bed claim"),
    }
    out["MOP_RESET_SEMANTICS_REPORT.json"] = {
        "schema": "mop-reset-semantics-report/v1",
        "schedules": list(e2.Fx.RESET_KINDS) if hasattr(e2, "Fx") else [],
        "effects": gather("reset"),
        "alignment_witness": {b: a["instrumentation"] for b, a in per_bed.items()
                              if a.get("status") != "no_runs"},
        "rule": "period three is retained only as a historical defect mutation and is never an arm",
    }
    out["MOP_EXPLICIT_HISTORY_REPORT.json"] = {
        "schema": "mop-explicit-history-report/v1",
        "history_lengths": [str(k) for k in e2.Fx.HISTORY_K] if hasattr(e2, "Fx") else [],
        "effects": gather("history"),
        "recurrent_versus_matched_history": gather("recurrent_versus_matched_history"),
        "explicit_history_sufficient": {b: a["findings"]["explicit_history_sufficient"]
                                        for b, a in per_bed.items() if a.get("status") != "no_runs"},
    }
    out["MOP_OPTIMIZATION_CONVERGENCE_REPORT.json"] = {
        "schema": "mop-optimization-convergence-report/v1",
        "grid": list(e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID),
        "per_bed": {b: a["convergence"] for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "required_optimization_contrasts": {
            b: optimization_arms(b) for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "rule": ("an unconverged arm cannot determine a scientific classification, and the remedy is a "
                 "longer budget rather than a looser plateau rule"),
    }
    out["MOP_FACTORIAL_INTERACTION_REPORT.json"] = {
        "schema": "mop-factorial-interaction-report/v1",
        "core_by_readout": gather("core_by_readout"),
        "core_by_capacity": gather("core_by_capacity"),
        "core_by_horizon": gather("core_by_horizon"),
        "readout_by_capacity": gather("readout_by_capacity"),
        "capacity_by_horizon": gather("capacity_by_horizon"),
        "capacity_by_readout": gather("capacity_by_readout"),
        "history_by_architecture": gather("history_by_architecture"),
        "architecture_by_bed": between_bed_interactions(per_bed, "architecture"),
        "horizon_by_bed": between_bed_interactions(per_bed, "horizon"),
        "optimization_by_capacity": {b: optimization_interaction(b) for b in PRINCIPAL_BEDS},
        "omission_status": {
            "reset_by_readout": "not_run_under_sealed_omission_map",
            "history_by_recurrent_architecture": "not_defined_because_recurrent_cores_have_no_history_k_parameter",
        },
    }
    return out


def state_horizon_gate(per_bed: dict) -> dict:
    """Require the preregistered persistence and destructive reset witnesses on both principal beds."""
    per = {}
    for bed in PRINCIPAL_BEDS:
        a = per_bed.get(bed, {})

        def destructive(d: dict) -> bool:
            return (d.get("convergence", {}).get("all_converged")
                    and d.get("group_upper_95_cb") is not None
                    and d["group_upper_95_cb"] <= -io.SESOI)

        horizon = a.get("effects", {}).get("horizon", {})
        reset = a.get("effects", {}).get("reset", {})
        fixed = {k: destructive(reset.get(k, {})) for k in ("misaligned_a", "misaligned_b")}
        checks = {
            "full_beats_horizon_45_by_sesoi": destructive(horizon.get("gru_h45_vs_full", {})),
            "full_beats_a_misaligned_fixed_reset_by_sesoi": any(fixed.values()),
            "full_beats_rate_matched_random_reset_by_sesoi": destructive(
                reset.get("random_rate_matched", {})),
        }
        per[bed] = {"checks": checks, "fixed_reset_checks": fixed, "all_pass": all(checks.values())}
    return {
        "per_bed": per,
        "two_principal_beds_agree": len(per) == len(PRINCIPAL_BEDS) and all(v["all_pass"] for v in per.values()),
        "all_pass": len(per) == len(PRINCIPAL_BEDS) and all(v["all_pass"] for v in per.values()),
    }


def result_keys(per_bed: dict) -> list[str]:
    """Fold the measured effects into the preregistered result vocabulary. No new keys are invented here."""
    keys = []
    principal = {b: a for b, a in per_bed.items() if b in PRINCIPAL_BEDS and a.get("status") != "no_runs"}
    if set(principal) != set(PRINCIPAL_BEDS):
        return keys

    def sufficient(d: dict) -> bool:
        return (d.get("mean") is not None and d.get("estimator_sufficient") is True
                and (d.get("convergence") or {}).get("all_converged") is True
                and all(d.get(field) is not None for field in (
                    "lower_95_cb", "upper_95_cb", "group_lower_95_cb", "group_upper_95_cb")))

    def positive(d: dict) -> bool:
        return sufficient(d) and d.get("verdict") == "positive" \
            and d["group_lower_95_cb"] >= io.SESOI

    def equivalent(d: dict) -> bool:
        return sufficient(d) and AN.equivalent(d)

    mh = [a["effects"]["recurrent_versus_matched_history"] for a in principal.values()]
    matched_full = [g.get("gru_vs_histmlp_kfull_window", {}) for g in mh]
    if all(positive(v) for v in matched_full):
        keys.append("recurrent_beats_matched_history")
    if all(equivalent(v) for v in matched_full):
        keys.append("matched_history_matches_recurrent")
    capacity_tables = [a["effects"]["capacity"] for a in principal.values()]
    if all(all(positive(a["effects"]["capacity"].get(f"gru_{tier}_vs_small", {}))
                   for tier in ("medium", "large"))
           and a["effects"]["capacity"]["gru_medium_vs_small"]["mean"]
           <= a["effects"]["capacity"]["gru_large_vs_small"]["mean"] + 1e-9
           for a in principal.values()):
        keys.append("capacity_monotonic_and_large")
    elif all(table and all(equivalent(d) for d in table.values()) for table in capacity_tables):
        keys.append("capacity_flat_or_saturating")
    horizon_gate = state_horizon_gate(per_bed)
    horizon_effects = [[a["effects"]["horizon"].get(k, {}) for k in (
        "gru_h45_vs_full", "gru_h90_vs_full")] for a in principal.values()]
    horizon_ready = all(sufficient(d) for rows in horizon_effects for d in rows)
    threshold_rows = []
    for a in principal.values():
        threshold = a["findings"]["horizon_threshold"]
        threshold_rows.append(a["effects"]["horizon"].get(f"gru_h{threshold}_vs_full", {})
                              if threshold is not None else {})
    if horizon_ready and horizon_gate["all_pass"] and all(
            sufficient(d) and equivalent(d) for d in threshold_rows):
        keys.append("horizon_threshold_at_dependency_length")
    elif horizon_ready and all(equivalent(d) for rows in horizon_effects for d in rows):
        keys.append("horizon_flat")
    interaction_tables = [a["effects"]["capacity_by_horizon"] for a in principal.values()]
    interaction_ready = all(table and all(sufficient(d) for d in table.values())
                            for table in interaction_tables)
    if interaction_ready and all(any(positive(d) for d in table.values()) for table in interaction_tables):
        keys.append("capacity_helps_only_at_long_horizon")
    elif interaction_ready and all(all(equivalent(d) for d in table.values())
                                   for table in interaction_tables):
        keys.append("capacity_and_horizon_independent")
    family_pass = {r: all(positive(principal[b]["effects"]["recurrence_versus_best_stateless"].get(
        f"{r}_vs_{s}", {})) for b in PRINCIPAL_BEDS for s in A.STATELESS) for r in A.RECURRENT}
    family_ready = {r: all(sufficient(principal[b]["effects"]["recurrence_versus_best_stateless"].get(
        f"{r}_vs_{s}", {})) for b in PRINCIPAL_BEDS for s in A.STATELESS) for r in A.RECURRENT}
    if set(family_pass) == set(A.RECURRENT) and all(family_pass.values()):
        keys.append("all_recurrent_families_agree")
    elif all(family_ready.values()) and any(family_pass.values()):
        keys.append("one_recurrent_family_dissents")
    optimization = {bed: optimization_interaction(bed) for bed in PRINCIPAL_BEDS}
    optimization_ready = all(d.get("classification") == "converged" for d in optimization.values())
    if optimization_ready and all(positive(v) for v in matched_full):
        keys.append("converged_everywhere_and_gap_remains")
    third = per_bed.get("harth_stream")
    third_preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") if io.exists(
        "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {}
    third_admitted = "harth_stream" in (third_preflight.get("selected") or [])
    if third and third.get("status") != "no_runs" and third_admitted:
        effect = third["effects"]["recurrent_versus_matched_history"].get(
            "gru_vs_histmlp_kfull_window", {})
        if positive(effect):
            keys.append("third_bed_agrees")
        elif sufficient(effect) and (equivalent(effect)
                                     or effect["group_upper_95_cb"] <= -io.SESOI):
            keys.append("third_bed_dissents")
    elif not third_admitted:
        keys.append("third_bed_invalid")
    readout_tables = [a["effects"]["readout"] for a in principal.values()]
    readout_ready = all(table and all(sufficient(d) for d in table.values()) for table in readout_tables)
    if readout_ready and all(any(positive(d) for d in table.values()) for table in readout_tables):
        keys.append("readout_capacity_reproduces_the_effect")
    elif readout_ready and all(all(equivalent(d) for d in table.values()) for table in readout_tables):
        keys.append("readout_capacity_flat")
    return keys


def main():
    t0 = time.time()
    per_bed = {b: analyse_bed(b) for b in BEDS}
    keys = result_keys(per_bed)
    fold = H.apply(keys)
    reports = per_factor_reports(per_bed)
    for name, doc in reports.items():
        io.seal(name, doc)
    principal = {b: a for b, a in per_bed.items() if b in PRINCIPAL_BEDS and a.get("status") != "no_runs"}
    correction = io.load("MOP_E2_CAPACITY_TIER_CORRECTION.json") if io.exists(
        "MOP_E2_CAPACITY_TIER_CORRECTION.json") else {"all_pass": False}
    factorial = io.load("MOP_E2_FACTORIAL_AUTHORITY.json") if io.exists("MOP_E2_FACTORIAL_AUTHORITY.json") else {}
    mutations = io.load("MOP_TEMPORAL_CORE_MUTATION_REPORT.json") if io.exists(
        "MOP_TEMPORAL_CORE_MUTATION_REPORT.json") else {}
    replication = io.load("MOP_E2_INDEPENDENT_REPLICATION.json") if io.exists(
        "MOP_E2_INDEPENDENT_REPLICATION.json") else {}
    classifications = {}
    for b, a in principal.items():
        for group in ("recurrence_versus_best_stateless", "capacity", "horizon", "readout",
                      "recurrent_versus_matched_history"):
            for k, d in a["effects"][group].items():
                if d.get("mean") is None:
                    continue
                classifications[f"{b}:{group}:{k}"] = gate.classify_result(
                    effect=d, instrument_valid=bool(correction.get("all_pass")), bed_valid=bool(
                        (factorial.get("principal_beds", {}).get(b, {}).get("checks") or {}).get("all_pass")),
                    mechanism_active=True,
                    baseline_valid=bool(d.get("convergence", {}).get("all_converged")),
                    verifier_agrees=False,
                    mutations_rejected=bool(mutations.get("all_rejected")),
                    implementations_agreeing=2 if replication.get("all_pass") else 0,
                    estimator_sufficient=bool(d.get("estimator_sufficient")),
                )["classification"]
    expected_cells = {AN.name(**c) for c in e2.Fx.sweep_cells()["_all"]}
    shard_index = []
    for b in BEDS:
        for seed in e2.PRINCIPAL_SEEDS:
            p = io.RUNS / "e2_principal" / f"{b}_{seed}.json"
            checks = {"exists": p.is_file()}
            if p.is_file():
                raw = json.loads(p.read_text())
                cells = [r.get("cell") for r in raw.get("runs", [])]
                checks.update({
                    "bed_identity": raw.get("bed") == b and all(r.get("bed") == b for r in raw["runs"]),
                    "seed_identity": raw.get("seed") == seed and all(r.get("seed") == seed for r in raw["runs"]),
                    "factorial_identity": len(cells) == len(expected_cells) and set(cells) == expected_cells,
                    "training_budget": all(r.get("steps") == (raw.get("convergence_authority", {})
                        .get("selected_checkpoints", {}).get(r.get("cell"))) for r in raw["runs"]),
                    "convergence_binding": raw.get("convergence_authority", {}).get("sha256") ==
                        io.sha_file(io.RUNS / "e2_converge" / f"converge_{b}.json"),
                    "checkpoint_receipts": all(r.get("checkpoint_sha_after") for r in raw["runs"]),
                    "parameter_inventory": all(r.get("params", {}).get("total") for r in raw["runs"]),
                    "no_undeclared_changes": all(not r.get("undeclared_changes") for r in raw["runs"]),
                })
            checks["all_pass"] = all(checks.values())
            shard_index.append({"path": p.relative_to(io.ROOT).as_posix(), "bed": b, "seed": seed,
                                "sha256": io.sha_file(p) if p.is_file() else None, "checks": checks})
    correction_index = []
    for b in BEDS:
        for seed in e2.PRINCIPAL_SEEDS:
            p = io.RUNS / "e2_principal_corrections" / f"capacity_{b}_{seed}.json"
            d = json.loads(p.read_text()) if p.is_file() else {}
            correction_index.append({"path": p.relative_to(io.ROOT).as_posix(), "bed": b, "seed": seed,
                                     "sha256": io.sha_file(p) if p.is_file() else None,
                                     "all_pass": bool(d.get("all_checks_pass"))})
    doc = {
        "schema": "mop-e2-principal-result/v1",
        "beds": list(BEDS),
        "principal_beds": list(PRINCIPAL_BEDS),
        "seeds": list(e2.PRINCIPAL_SEEDS),
        "sesoi": io.SESOI,
        "equivalence_margin": io.EQUIVALENCE_MARGIN,
        "preregistration": e2.PREREG,
        "per_bed": per_bed,
        "observed_result_keys": keys,
        "hypothesis_fold": fold,
        "classification_phase": "pre_independent_verification",
        "terminal_classification": classifications,
        "shard_index": shard_index,
        "capacity_tier_correction": correction,
        "correction_shard_index": correction_index,
        "n_expected_shards": len(BEDS) * len(e2.PRINCIPAL_SEEDS),
        "n_verified_shards": sum(s["checks"]["all_pass"] for s in shard_index),
        "all_shards_verified": (all(s["checks"]["all_pass"] for s in shard_index)
                                and correction.get("all_pass")
                                and all(s["all_pass"] for s in correction_index)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_E2_PRINCIPAL_RESULT.json", doc)
    io.seal("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", {
        "schema": "mop-substrate-hypothesis-graph/v2-temporal",
        "hypotheses": fold["hypotheses"],
        "observed_results": fold["observed_results"],
        "rule": "the mapping from result to hypothesis state was sealed before the first principal cell ran",
    })
    rows = "\n".join(
        f"| {name} | {row['state']} | {', '.join(row['evidence']['supports']) or 'none'} | "
        f"{', '.join(row['evidence']['weakens']) or 'none'} | {', '.join(row['evidence']['closes']) or 'none'} |"
        for name, row in fold["hypotheses"].items())
    io.seal_md("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md", f"""# Temporal substrate hypothesis graph

| hypothesis | state | supports | weakens | closes |
|---|---|---|---|---|
{rows}

The result to hypothesis mapping was sealed before principal execution. Unknown result keys are refused.
""")
    print(f"E2 analysis: results {keys}", flush=True)
    for h, v in fold["hypotheses"].items():
        print(f"  {h:32s} {v['state']}", flush=True)
    print("ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
