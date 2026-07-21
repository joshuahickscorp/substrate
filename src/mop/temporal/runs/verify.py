"""Role B instrumentation auditor and Role C independent scientific verifier.

Role B never looks at an outcome. Role C recomputes every effect with its own arithmetic and its own t table
and forms its verdicts before comparing. File reading is shared on purpose; scientific logic is not.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io

T95 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}
SESOI = 0.05
CORRECTED_CELLS = {Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large")),
                   Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large",
                                       readout="mlp_strong"))}


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


def upper_bound(v):
    return -lower_bound([-x for x in v]) if v else 0.0


def summarize(v: list[float]) -> dict:
    """Independent arithmetic used for both seed and natural-unit estimates."""
    return {"mean": round(mean(v), 5), "lower_95_cb": round(lower_bound(v), 5),
            "upper_95_cb": round(upper_bound(v), 5),
            "heterogeneity": round(sd(v), 5) if len(v) > 1 else None,
            "n": len(v), "verdict": verdict(v)}


def _close(a, b, tol: float = 1e-4) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < tol


def _unit_inventory(payload) -> tuple[dict[str, float], dict]:
    """Normalize unit scores while refusing duplicate identities before they can be averaged away."""
    if isinstance(payload, dict):
        records = [{"unit": str(k), "accuracy": v} for k, v in payload.items()]
    elif isinstance(payload, list):
        records = payload
    else:
        records = []
    ids = [str(r.get("unit")) for r in records if isinstance(r, dict) and "unit" in r]
    duplicate = sorted({u for u in ids if ids.count(u) > 1})
    values = {str(r["unit"]): float(r["accuracy"]) for r in records
              if isinstance(r, dict) and "unit" in r and "accuracy" in r}
    valid_values = len(values) == len(records) and all(math.isfinite(v) and 0 <= v <= 1
                                                       for v in values.values())
    return values, {"n_records": len(records), "n_unique_units": len(set(ids)),
                    "duplicate_units": duplicate, "all_values_valid": valid_values,
                    "all_pass": bool(records) and not duplicate and valid_values}


def _readout_inventory(runs: list[dict]) -> dict:
    inventory: dict[str, set[int]] = {}
    for row in runs:
        inventory.setdefault(row["spec"]["readout"], set()).add(int(row["params"]["readout"]))
    return {"counts": {k: sorted(v) for k, v in sorted(inventory.items())},
            "complete": set(inventory) == set(A.READOUTS),
            "depends_only_on_readout": bool(inventory) and all(len(v) == 1 for v in inventory.values())}


def _test_unit_weights(bed: str) -> dict[str, int]:
    units = [str(u) for u in np.asarray(B.load(bed)["ute"])]
    return {u: units.count(u) for u in sorted(set(units))}


def _weighted_metric(per_unit: dict[str, float], weights: dict[str, int]) -> float | None:
    if set(per_unit) != set(weights) or not weights:
        return None
    return sum(per_unit[u] * weights[u] for u in weights) / sum(weights.values())


def _partition_units(bed: str, seed: int, *, offset: int, eval_fraction: float,
                     fixed_total_eval: int | None = None) -> dict[str, list]:
    """Recreate successor unit custody directly from bed data, without importing either producer."""
    unique = np.random.default_rng(offset + seed).permutation(np.unique(np.asarray(B.load(bed)["u"])))
    half = len(unique) // 2
    if fixed_total_eval is None:
        a_n = max(1, int(round(eval_fraction * len(unique[:half]))))
        b_n = max(1, int(round(eval_fraction * len(unique[half:]))))
    else:
        a_n, b_n = (fixed_total_eval + 1) // 2, fixed_total_eval // 2
    return {"A_train": unique[:half][a_n:].tolist(), "A_eval": unique[:half][:a_n].tolist(),
            "B_train": unique[half:][b_n:].tolist(), "B_eval": unique[half:][:b_n].tolist()}


def _context_weights(bed: str, units: dict[str, list]) -> dict[str, dict[str, int]]:
    raw = [str(u) for u in np.asarray(B.load(bed)["u"])]
    return {name: {str(u): raw.count(str(u)) for u in group} for name, group in units.items()}


def _batch_plan_hash(n: int, seed: int, steps: int, batch: int) -> str:
    rng, digest = np.random.default_rng(seed), hashlib.sha256()
    for _ in range(steps):
        digest.update(np.asarray(rng.choice(n, min(batch, n), replace=False), dtype=np.int64).tobytes())
    return digest.hexdigest()


def _welch(left: list[float], right: list[float]) -> dict | None:
    if len(left) < 2 or len(right) < 2:
        return None
    ml, mr, vl, vr = mean(left), mean(right), sd(left) ** 2, sd(right) ** 2
    tl, tr = vl / len(left), vr / len(right)
    se = math.sqrt(tl + tr)
    denominator = tl ** 2 / (len(left) - 1) + tr ** 2 / (len(right) - 1)
    degrees = (tl + tr) ** 2 / denominator if denominator else len(left) + len(right) - 2
    critical = T95.get(max(2, math.floor(degrees) + 1), 1.729)
    return {"mean": ml - mr, "lower_95_cb": ml - mr - critical * se,
            "upper_95_cb": ml - mr + critical * se, "degrees": degrees}


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


def _principal_runs(bed: str) -> list[dict]:
    """Apply only exact bed, seed and cell corrections while retaining original files."""
    rows = {}
    for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            rows[(int(row["seed"]), row["cell"])] = row
    for p in sorted((io.RUNS / "e2_principal_corrections").glob(f"capacity_{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            rows[(int(row["seed"]), row["cell"])] = row
    return list(rows.values())


def _effect_inputs(runs: list[dict]) -> tuple[dict[str, dict[int, dict]], dict[str, dict[str, float]]]:
    by_cell: dict[str, dict[int, dict]] = {}
    unit_values: dict[str, dict[str, list[float]]] = {}
    for row in runs:
        by_cell.setdefault(row["cell"], {})[int(row["seed"])] = row
        values, _ = _unit_inventory(row.get("per_unit_accuracy"))
        for unit, value in values.items():
            unit_values.setdefault(row["cell"], {}).setdefault(unit, []).append(value)
    return by_cell, {cell: {unit: mean(values) for unit, values in cells.items()}
                     for cell, cells in unit_values.items()}


def _recompute_effect(by_cell: dict[str, dict[int, dict]], unit_cells: dict[str, dict[str, float]],
                      components: list[str], signs: list[int], seeds: list[int]) -> dict | None:
    if (not components or len(components) != len(signs)
            or any(set(by_cell.get(cell, {})) != set(seeds) for cell in components)):
        return None
    seed_effects = [sum(sign * float(by_cell[cell][seed]["accuracy"])
                        for sign, cell in zip(signs, components, strict=True)) for seed in seeds]
    shared = sorted(set.intersection(*(set(unit_cells.get(cell, {})) for cell in components)))
    unit_effects = [sum(sign * unit_cells[cell][unit]
                        for sign, cell in zip(signs, components, strict=True)) for unit in shared]
    seed_summary, group_summary = summarize(seed_effects), summarize(unit_effects)
    return {"mean": seed_summary["mean"], "lower_95_cb": seed_summary["lower_95_cb"],
            "upper_95_cb": seed_summary["upper_95_cb"], "verdict": seed_summary["verdict"],
            "per_seed_effects": [round(v, 5) for v in seed_effects],
            "per_unit_effects": {unit: round(value, 5) for unit, value in zip(
                shared, unit_effects, strict=True)},
            "group_mean": group_summary["mean"],
            "group_lower_95_cb": group_summary["lower_95_cb"],
            "group_upper_95_cb": group_summary["upper_95_cb"],
            "group_heterogeneity": group_summary["heterogeneity"], "n_units": len(unit_effects)}


def _effect_matches(mine: dict | None, expected: dict, *, require_group: bool = True) -> bool:
    if mine is None:
        return False
    pairs = (("mean", "mean"), ("lower_95_cb", "lower_95_cb"))
    ok = all(_close(mine[a], expected.get(b)) for a, b in pairs)
    ok = ok and mine["verdict"] == expected.get("verdict")
    if "per_seed_effects" in expected:
        ok = ok and len(mine["per_seed_effects"]) == len(expected["per_seed_effects"]) and all(
            _close(a, b) for a, b in zip(mine["per_seed_effects"], expected["per_seed_effects"], strict=True))
    group_fields = ("group_mean", "group_lower_95_cb", "group_upper_95_cb", "group_heterogeneity")
    if require_group:
        ok = ok and all(field in expected and _close(mine[field], expected.get(field))
                        for field in group_fields)
    else:
        ok = ok and all(field not in expected or _close(mine[field], expected.get(field))
                        for field in group_fields)
    return ok


def _equivalent(effect: dict, margin: float) -> bool:
    return (abs(effect["mean"]) <= margin and effect["lower_95_cb"] >= -margin
            and effect["upper_95_cb"] <= margin)


def _terminal_classification(effect: dict, *, instrument_valid: bool, bed_valid: bool,
                             verifier_agrees: bool, mutations_rejected: bool,
                             implementations_agree: bool) -> str:
    """Independent statement of the sealed terminal contract, without importing gate code."""
    if not instrument_valid:
        return "invalid_instrument"
    if not bed_valid:
        return "invalid_bed"
    if not (effect.get("convergence") or {}).get("all_converged"):
        return "unconverged_baseline"
    if not effect.get("estimator_sufficient"):
        return "insufficient_power"
    value = effect.get("verdict")
    if value == "positive":
        return ("positive" if verifier_agrees and mutations_rejected and implementations_agree
                else "provisional_positive")
    if value == "harm":
        return "harm"
    if value == "wrong_direction_failure":
        return "failure_wrong_direction"
    return "mechanism_null" if verifier_agrees else "scientifically_unresolved"


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
        original = []
        for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
            original.extend(json.loads(p.read_text())["runs"])
        invalid_originals = [r for r in original if r["cell"] in CORRECTED_CELLS and not (
            A.TIER_RANGE["large"][0] <= r["params"]["core"] <= A.TIER_RANGE["large"][1])]
        correction = io.load("MOP_E2_CAPACITY_TIER_CORRECTION.json") if io.exists(
            "MOP_E2_CAPACITY_TIER_CORRECTION.json") else {}
        checks[f"{bed}:original_capacity_defects_quarantined"] = len(invalid_originals) == 16
        checks[f"{bed}:capacity_correction_authority_passes"] = bool(correction.get("all_pass"))
        runs = _principal_runs(bed)
        by_tier: dict = {}
        for r in runs:
            by_tier.setdefault(r["spec"]["tier"], set()).add(r["params"]["core"])
        checks[f"{bed}:capacity_tiers_are_banded"] = all(
            A.TIER_RANGE[tier][0] <= value <= A.TIER_RANGE[tier][1]
            for tier, values in by_tier.items() for value in values)
        checks[f"{bed}:factorial_cell_identity"] = all(
            r["cell"] == Fx.cell_name(**r["spec"]) for r in runs)
        checks[f"{bed}:training_budget_matches_updates"] = all(
            r["steps"] == r["updates"] == Fx.STEPS for r in runs)
        checks[f"{bed}:checkpoint_receipts_present"] = all(r.get("checkpoint_sha_after") for r in runs)
        checks[f"{bed}:parameter_inventory_sums"] = all(
            r["params"]["total"] == r["params"]["core"] + r["params"]["readout"] for r in runs)
        readouts = _readout_inventory(runs)
        checks[f"{bed}:readout_inventory_complete"] = readouts["complete"]
        checks[f"{bed}:readout_parameter_count_depends_only_on_the_readout"] = readouts[
            "depends_only_on_readout"]
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
        nonoracle = {k: v for k, v in (d.get("arms") or {}).items() if k != "oracle_assignment"}
        totals = {v.get("params", {}).get("total") for v in nonoracle.values()}
        checks[f"{key}:matched_model_resources"] = len(totals) == 1 and None not in totals
        checks[f"{key}:capacity_band"] = bool(totals) and all(
            d.get("capacity_band", [1, 0])[0] <= v.get("params", {}).get("core", -1)
            <= d.get("capacity_band", [1, 0])[1] for v in nonoracle.values())
        full = [d.get("source_training") or {},
                ((d.get("wrong_bed_control") or {}).get("donor_training") or {}),
                ((d.get("arms") or {}).get("domain_local_component") or {}).get("training") or {},
                ((d.get("arms") or {}).get("frozen_transferred_component") or {}).get("training") or {},
                ((d.get("arms") or {}).get("fine_tuned_transferred_component") or {}).get("training") or {}]
        shared = ((d.get("arms") or {}).get("shared_component") or {}).get("training") or {}
        checks[f"{key}:declared_compute_is_charged"] = all(
            r.get("steps") == r.get("updates") == Fx.STEPS for r in [*full, shared])
        resource_receipts = [r for r in [*full, shared] if "lr" in r or "batch" in r]
        checks[f"{key}:recorded_lr_and_batches_match"] = all(
            ("lr" not in r or _close(r["lr"], Fx.LR))
            and ("batch" not in r or r["batch"] == Fx.BATCH) for r in resource_receipts)
        target_match = d.get("target_training_match") or {}
        target_checks = target_match.get("checks") or {}
        checks[f"{key}:target_training_match"] = (
            target_match.get("optimizer") == "Adam"
            and target_match.get("batch_seed") == 50_000 + int(d.get("seed"))
            and target_match.get("parameter_exposure_per_arm", 0) > 0
            and bool(target_checks) and all(target_checks.values()) and target_match.get("all_matched") is True
            and ((d.get("arms") or {}).get("domain_local_component") or {}).get("training", {}).get(
                "trainable_params") == ((d.get("arms") or {}).get("shared_component") or {}).get(
                    "training", {}).get("trainable_params"))
    expected_seeds = set(doc.get("seeds") or [])
    directions: dict[tuple, set] = {}
    for p in e3_paths:
        d = json.loads(p.read_text())
        directions.setdefault((d.get("source_bed"), d.get("target_bed")), set()).add(d.get("seed"))
    for direction, seeds in directions.items():
        checks[f"E3:{direction[0]}:{direction[1]}:exact_seed_set"] = seeds == expected_seeds
    third_paths = sorted((io.RUNS / "third_bed_preflight").glob("*.json")) \
        if (io.RUNS / "third_bed_preflight").is_dir() else []
    for p in third_paths:
        d = json.loads(p.read_text())
        key = f"HARTH-preflight:{d.get('seed')}"
        sets = [set(v) for v in d.get("units", {}).values()]
        expected_units = _partition_units("harth_stream", int(d.get("seed")), offset=70_000,
                                          eval_fraction=0.0, fixed_total_eval=7)
        checks[f"{key}:bed_identity"] = d.get("bed") == "harth_stream"
        checks[f"{key}:unit_disjoint"] = all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
        checks[f"{key}:test_untouched"] = bool(d.get("test_split_untouched"))
        checks[f"{key}:training_receipts"] = bool(d.get("all_checks_pass"))
        checks[f"{key}:split_identity"] = d.get("units") == expected_units
    if third_paths:
        checks["HARTH-preflight:exact_seed_set"] = {
            json.loads(p.read_text()).get("seed") for p in third_paths} == expected_seeds
    hybrid_paths = sorted((io.RUNS / "hybrid").glob("*.json")) if (io.RUNS / "hybrid").is_dir() else []
    for p in hybrid_paths:
        d = json.loads(p.read_text())
        key = f"hybrid:{d.get('bed')}:{d.get('seed')}"
        expected_units = _partition_units(str(d.get("bed")), int(d.get("seed")), offset=110_000,
                                          eval_fraction=0.25)
        checks[f"{key}:six_arms"] = len(d.get("arms", {})) == 6
        checks[f"{key}:state_only_zero_parameter_updates"] = bool(
            (d.get("checks") or {}).get("state_only_zero_parameter_updates"))
        checks[f"{key}:matched_updates"] = bool((d.get("checks") or {}).get("matched_adaptation_updates"))
        checks[f"{key}:state_noise_matched"] = bool(
            (d.get("checks") or {}).get("state_noise_magnitude_matched"))
        checks[f"{key}:split_identity"] = d.get("units") == expected_units
        arms = d.get("arms") or {}
        head_traces = [(arms.get(name) or {}).get("trace") or {} for name in (
            "head_only", "head_plus_state", "head_plus_state_noise")]
        lr_values = [r.get("head_lr") for r in head_traces]
        plans = [r.get("batch_plan_sha") for r in head_traces]
        expected_plan = _batch_plan_hash(sum(_context_weights(str(d.get("bed")), d["units"])[
            "B_train"].values()), 140_000 + int(d.get("seed")), Fx.STEPS // 4, Fx.BATCH)
        checks[f"{key}:head_learning_rates_match"] = all(_close(v, Fx.LR) for v in lr_values)
        checks[f"{key}:head_batches_match"] = all(plan == expected_plan for plan in plans)
        checks[f"{key}:noise_norm_reconstructed"] = _close(
            (arms.get("head_plus_state_noise") or {}).get("state_norm"),
            (arms.get("head_plus_state") or {}).get("state_norm"), tol=1.1e-5)
        noisy_trace = (arms.get("head_plus_state_noise") or {}).get("trace") or {}
        checks[f"{key}:noise_uses_learned_hybrid_state"] = bool(
            noisy_trace.get("magnitude_matched_to_learned_head_plus_state"))
        checks[f"{key}:noise_preserves_matched_head"] = bool(
            noisy_trace.get("head_parameters_match_learned_head_plus_state"))
    hybrid_by_bed: dict[str, set] = {}
    for p in hybrid_paths:
        d = json.loads(p.read_text())
        hybrid_by_bed.setdefault(str(d.get("bed")), set()).add(d.get("seed"))
    for bed, seeds in hybrid_by_bed.items():
        checks[f"hybrid:{bed}:exact_seed_set"] = seeds == expected_seeds
    return {"role": "B instrumentation auditor", "checks": checks, "notes": notes,
            "failed": [k for k, v in checks.items() if not v], "all_pass": all(checks.values()),
            "outcomes_inspected": False}


# ---------------------------------------------------------------- role C


def role_c() -> dict:
    if not io.exists("MOP_E2_PRINCIPAL_RESULT.json"):
        return {"role": "C", "status": "not_run"}
    sealed = io.load("MOP_E2_PRINCIPAL_RESULT.json")
    expected_seeds = [int(s) for s in sealed["seeds"]]
    checks, mismatches, recomputed = {}, [], {}
    for bed, a in sealed["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        runs = _principal_runs(bed)
        by_cell, unit_cells = _effect_inputs(runs)
        weights = _test_unit_weights(bed)
        checks[f"{bed}:exact_seed_set"] = {int(r["seed"]) for r in runs} == set(expected_seeds)
        checks[f"{bed}:every_cell_has_exact_seed_set"] = bool(by_cell) and all(
            set(rows) == set(expected_seeds) for rows in by_cell.values())
        receipt_identity, unit_identity, metrics = [], [], []
        for r in runs:
            values, inventory = _unit_inventory(r.get("per_unit_accuracy"))
            receipt_identity.append(r.get("bed") == bed and int(r.get("seed")) in expected_seeds
                                    and r.get("eval_on") == "test"
                                    and r.get("cell") == Fx.cell_name(**r.get("spec", {})))
            unit_identity.append(inventory["all_pass"] and set(values) == set(weights))
            metric = _weighted_metric(values, weights)
            metrics.append(metric is not None and _close(metric, r.get("accuracy")))
        checks[f"{bed}:receipt_bed_seed_cell_and_split_identity"] = bool(runs) and all(receipt_identity)
        checks[f"{bed}:exact_evaluation_unit_identity"] = bool(runs) and all(unit_identity)
        checks[f"{bed}:accuracy_reconstructed_from_per_unit_receipts"] = bool(runs) and all(metrics)
        recomputed[bed] = {}
        for group, table in a["effects"].items():
            recomputed[bed][group] = {}
            for k, d in table.items():
                if d.get("mean") is None:
                    continue
                components = list(d.get("components") or [])
                signs = list(d.get("formula_signs") or [])
                if not components:
                    components = d["contrast"].split(" minus ")
                    signs = [1, -1]
                mine = _recompute_effect(by_cell, unit_cells, components, signs, expected_seeds)
                recomputed[bed][group][k] = mine
                ok = _effect_matches(mine, d)
                if group in {"core_by_readout", "core_by_capacity", "core_by_horizon",
                             "readout_by_capacity", "history_by_architecture", "capacity_by_horizon"}:
                    did = (len(components) == 4 and signs == [1, -1, -1, 1]
                           and d.get("estimand") == "difference_in_differences")
                    checks[f"{bed}:{group}:{k}:did_identity"] = did
                    ok = ok and did
                convergence = d.get("convergence") or {}
                declared = convergence.get("cells") or {}
                checks[f"{bed}:{group}:{k}:convergence_classification"] = (
                    bool(declared) and convergence.get("all_converged") == all(
                        status == "converged" for status in declared.values())
                    and convergence.get("classification") == (
                        "converged" if all(status == "converged" for status in declared.values())
                        else "provisional_unconverged_or_unmeasured"))
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
            runs = _principal_runs(bed)
            by_cell, unit_cells = _effect_inputs(runs)
            for key, cell in implementation_cells.items():
                expected = row["effects"][key]
                mine = _recompute_effect(by_cell, unit_cells, [cell, control], [1, -1], expected_seeds)
                ok = _effect_matches(mine, expected)
                checks[f"{bed}:independent_replication:{key}"] = ok
                if not ok:
                    mismatches.append({"bed": bed, "contrast": key,
                                       "sealed": expected, "recomputed": mine})

    if io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json"):
        owned = io.load("MOP_OWNED_TEMPORAL_CORE_V1.json")
        selection = owned.get("selection") or {}
        evidence = selection.get("equivalence_evidence") or {}
        # A stale preselection artifact may exist before the supervisor reaches core selection. Once the
        # current selector emits raw equivalence evidence, every row is independently checked below.
        checks["owned_core:equivalence_evidence_present"] = (
            not selection.get("selected") or "equivalence_evidence" not in selection or bool(evidence))
        best = selection.get("best_cell")
        independently_equivalent = []
        for candidate, expected in evidence.items():
            per_bed, candidate_pass = {}, True
            for bed in sealed["principal_beds"]:
                runs = _principal_runs(bed)
                by_cell, unit_cells = _effect_inputs(runs)
                effect = _recompute_effect(by_cell, unit_cells, [candidate, best], [1, -1], expected_seeds)
                if effect is None:
                    candidate_pass = False
                    continue
                seed_equivalent = _equivalent(effect, io.EQUIVALENCE_MARGIN)
                group_equivalent = (effect["group_lower_95_cb"] >= -io.EQUIVALENCE_MARGIN
                                    and effect["group_upper_95_cb"] <= io.EQUIVALENCE_MARGIN)
                per_bed[bed] = {"seed_equivalent": seed_equivalent,
                                "group_equivalent": group_equivalent, **effect}
                sealed_bed = (expected.get("per_bed") or {}).get(bed, {})
                candidate_pass = candidate_pass and seed_equivalent == sealed_bed.get("seed_equivalent") \
                    and group_equivalent == sealed_bed.get("group_equivalent") \
                    and all(_close(effect[field], sealed_bed.get(field)) for field in (
                        "mean", "lower_95_cb", "group_lower_95_cb", "group_upper_95_cb"))
            candidate_pass = candidate_pass and all(
                row["seed_equivalent"] and row["group_equivalent"] for row in per_bed.values())
            checks[f"owned_core:equivalence:{candidate}"] = candidate_pass == bool(expected.get("passes"))
            if candidate_pass:
                independently_equivalent.append(candidate)
        if evidence:
            checks["owned_core:equivalent_region"] = sorted(independently_equivalent) == sorted(
                selection.get("equivalent_region") or [])
        checks["owned_core:claim_ceiling"] = owned.get("evidence_ceiling") == (
            "this is a substrate component with evidence on the beds named here. It does not establish a "
            "complete substrate architecture, continual plasticity, cross domain transfer, functional "
            "reorganization or activation") and owned.get("activation") is False

    if io.exists("MOP_FACTORIAL_INTERACTION_REPORT.json"):
        interactions = io.load("MOP_FACTORIAL_INTERACTION_REPORT.json")
        for report_key, group in (("architecture_by_bed", "architecture"),
                                  ("horizon_by_bed", "horizon")):
            for key, expected in (interactions.get(report_key) or {}).items():
                left = recomputed.get("har_stream", {}).get(group, {}).get(key)
                right = recomputed.get("speech_stream", {}).get(group, {}).get(key)
                seed_welch = _welch((left or {}).get("per_seed_effects", []),
                                    (right or {}).get("per_seed_effects", []))
                unit_left = list(((left or {}).get("per_unit_effects") or {}).values())
                unit_right = list(((right or {}).get("per_unit_effects") or {}).values())
                unit_welch = _welch(unit_left, unit_right)
                ok = seed_welch is not None and unit_welch is not None
                if ok:
                    heterogeneity = math.sqrt((sd(unit_left) ** 2 + sd(unit_right) ** 2) / 2)
                    independent_verdict = (
                        "harm" if seed_welch["mean"] <= -SESOI else
                        "positive" if seed_welch["lower_95_cb"] >= SESOI
                        and unit_welch["lower_95_cb"] >= SESOI else
                        "wrong_direction_failure" if seed_welch["mean"] < 0 else
                        "null_futile" if seed_welch["mean"] <= 0.01 else "null")
                    ok = (expected.get("estimand") == "independent_bed_difference_in_differences"
                          and _close(seed_welch["mean"], expected.get("mean"))
                          and _close(seed_welch["lower_95_cb"], expected.get("lower_95_cb"))
                          and _close(unit_welch["mean"], expected.get("group_mean"))
                          and _close(unit_welch["lower_95_cb"], expected.get("group_lower_95_cb"))
                          and _close(unit_welch["upper_95_cb"], expected.get("group_upper_95_cb"))
                          and _close(heterogeneity, expected.get("group_heterogeneity"))
                          and independent_verdict == expected.get("verdict"))
                checks[f"interaction:{report_key}:{key}"] = ok
                if not ok:
                    mismatches.append({"bed": "cross_bed", "contrast": f"{report_key}:{key}",
                                       "sealed": expected,
                                       "recomputed": {"seed": seed_welch, "group": unit_welch}})
        for bed, expected in (interactions.get("optimization_by_capacity") or {}).items():
            path = io.RUNS / "e2_converge_corrections" / f"convergence_{bed}.json"
            raw = json.loads(path.read_text()) if path.is_file() else {}
            control = raw.get("optimization_control") or {}
            budgets = expected.get("budgets") or []

            def at(document, key, budget):
                table = document.get(key) or {}
                return table.get(str(budget), table.get(budget))

            if len(budgets) == 2:
                low, high = budgets
                vectors = [at(raw, "seed_scores", high), at(raw, "seed_scores", low),
                           at(control, "seed_scores", high), at(control, "seed_scores", low)]
                effects = ([vectors[0][i] - vectors[1][i] - vectors[2][i] + vectors[3][i]
                            for i in range(min(map(len, vectors)))] if all(vectors) else [])
                tables = [at(raw, "per_unit_seed_scores", high), at(raw, "per_unit_seed_scores", low),
                          at(control, "per_unit_seed_scores", high),
                          at(control, "per_unit_seed_scores", low)]
                per_unit: dict[str, list[float]] = {}
                if all(tables):
                    for seed in map(str, raw.get("seeds") or []):
                        shared = set.intersection(*(set(table.get(seed, {})) for table in tables))
                        for unit in shared:
                            per_unit.setdefault(unit, []).append(
                                tables[0][seed][unit] - tables[1][seed][unit]
                                - tables[2][seed][unit] + tables[3][seed][unit])
                units = [mean(values) for values in per_unit.values()]
                seed_summary, group_summary = summarize(effects), summarize(units)
                ok = (raw.get("seeds") == [0, 1, 2]
                      and expected.get("estimand") == "difference_in_differences"
                      and expected.get("formula_signs") == [1, -1, -1, 1]
                      and seed_summary["verdict"] == expected.get("verdict")
                      and _close(seed_summary["mean"], expected.get("mean"))
                      and _close(seed_summary["lower_95_cb"], expected.get("lower_95_cb"))
                      and _close(group_summary["mean"], expected.get("group_mean"))
                      and _close(group_summary["lower_95_cb"], expected.get("group_lower_95_cb"))
                      and _close(group_summary["upper_95_cb"], expected.get("group_upper_95_cb"))
                      and _close(group_summary["heterogeneity"], expected.get("group_heterogeneity")))
            else:
                ok = expected.get("mean") is None and not raw
            checks[f"interaction:optimization_by_capacity:{bed}"] = ok
            if not ok:
                mismatches.append({"bed": bed, "contrast": "optimization_by_capacity",
                                   "sealed": expected, "recomputed": None})

    prior_verification = io.load("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json") if io.exists(
        "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json") else {}
    correction = io.load("MOP_E2_CAPACITY_TIER_CORRECTION.json") if io.exists(
        "MOP_E2_CAPACITY_TIER_CORRECTION.json") else {}
    factorial = io.load("MOP_E2_FACTORIAL_AUTHORITY.json") if io.exists(
        "MOP_E2_FACTORIAL_AUTHORITY.json") else {}
    mutations = io.load("MOP_TEMPORAL_CORE_MUTATION_REPORT.json") if io.exists(
        "MOP_TEMPORAL_CORE_MUTATION_REPORT.json") else {}
    replication = io.load("MOP_E2_INDEPENDENT_REPLICATION.json") if io.exists(
        "MOP_E2_INDEPENDENT_REPLICATION.json") else {}
    for identity, expected in (sealed.get("terminal_classification") or {}).items():
        bed, group, key = identity.split(":", 2)
        effect = sealed["per_bed"][bed]["effects"][group][key]
        bed_valid = bool(((factorial.get("principal_beds") or {}).get(bed, {}).get("checks") or {}).get(
            "all_pass"))
        mine = _terminal_classification(
            effect, instrument_valid=bool(correction.get("all_pass")), bed_valid=bed_valid,
            verifier_agrees=bool(prior_verification.get("all_pass")),
            mutations_rejected=bool(mutations.get("all_rejected")),
            implementations_agree=bool(replication.get("all_pass")))
        checks[f"terminal:{identity}"] = mine == expected
        if mine != expected:
            mismatches.append({"bed": bed, "contrast": identity, "sealed": expected, "recomputed": mine})

    if io.exists("MOP_E3_SHARED_LOCAL_RESULT.json"):
        e3 = io.load("MOP_E3_SHARED_LOCAL_RESULT.json")
        result = e3.get("result") or {}
        e3_classes = []
        for direction, expected in (result.get("per_direction") or {}).items():
            source, target = direction.split("_to_", 1)
            rows = [json.loads(p.read_text()) for p in sorted((io.RUNS / "e3").glob(
                f"{source}_to_{target}_*.json"))]
            checks[f"E3:{direction}:exact_seed_set"] = {r.get("seed") for r in rows} == set(expected_seeds)
            effects = [r["arms"]["shared_component"]["accuracy"]
                       - r["arms"]["domain_local_component"]["accuracy"] for r in rows]
            unit_values: dict[str, list[float]] = {}
            metric_checks, retention_checks, floors = [], [], []
            target_weights = _test_unit_weights(target)
            for row in rows:
                metric_checks.append(row.get("source_bed") == source and row.get("target_bed") == target
                                     and int(row.get("seed")) in expected_seeds)
                shared = row["arms"]["shared_component"]["per_unit_accuracy"]
                local = row["arms"]["domain_local_component"]["per_unit_accuracy"]
                for unit in set(shared) & set(local):
                    unit_values.setdefault(unit, []).append(shared[unit] - local[unit])
                for arm in row["arms"].values():
                    values, inventory = _unit_inventory(arm.get("per_unit_accuracy"))
                    score = _weighted_metric(values, target_weights)
                    metric_checks.append(inventory["all_pass"] and score is not None
                                         and _close(score, arm.get("accuracy")))
                source_values, source_inventory = _unit_inventory(
                    (row.get("source_baseline") or {}).get("per_unit_accuracy"))
                source_score = _weighted_metric(source_values, _test_unit_weights(source))
                metric_checks.append(source_inventory["all_pass"] and source_score is not None
                                     and _close(source_score, (row.get("source_baseline") or {}).get("accuracy")))
                retention = row["arms"]["shared_component"]["source_retention"]
                floor = float(retention["after"]) >= float(retention["before"]) - io.SESOI
                floors.append(floor)
                retention_checks.append(floor == bool(retention.get("floor_met")))
            units = [mean(v) for v in unit_values.values()]
            seed_summary, group_summary = summarize(effects), summarize(units)
            inverse_group = summarize([-v for v in units])
            mine = {"mean": seed_summary["mean"], "lower_95_cb": seed_summary["lower_95_cb"],
                    "upper_95_cb": seed_summary["upper_95_cb"], "verdict": seed_summary["verdict"],
                    "group_mean": group_summary["mean"],
                    "group_lower_95_cb": group_summary["lower_95_cb"],
                    "group_upper_95_cb": group_summary["upper_95_cb"],
                    "group_heterogeneity": group_summary["heterogeneity"]}
            sealed_effect = expected["shared_minus_domain_local"]
            inverse = verdict([-v for v in effects])
            classification = ("shared_component_supported" if seed_summary["verdict"] == "positive"
                              and group_summary["lower_95_cb"] >= io.SESOI and all(floors)
                              else "domain_local_component_supported" if inverse == "positive"
                              and inverse_group["lower_95_cb"] >= io.SESOI
                              else "shared_and_domain_local_inconclusive")
            e3_classes.append(classification)
            mine["per_seed_effects"] = [round(v, 5) for v in effects]
            ok = (len(rows) == len(expected_seeds) and _effect_matches(mine, sealed_effect,
                                                                       require_group=False)
                  and _close(inverse_group["lower_95_cb"],
                             sealed_effect.get("inverse_group_lower_95_cb"))
                  and classification == expected.get("classification"))
            checks[f"E3:{direction}:shared_vs_local"] = ok
            checks[f"E3:{direction}:metrics_reconstructed"] = bool(metric_checks) and all(metric_checks)
            checks[f"E3:{direction}:retention_floor_reconstructed"] = bool(retention_checks) and all(
                retention_checks)
            recomputed[f"E3:{direction}"] = mine
            if not ok:
                mismatches.append({"bed": direction, "contrast": "E3 shared versus local",
                                   "sealed": sealed_effect, "recomputed": mine})
        if result:
            overall = ("shared_temporal_representation_supported" if e3_classes and all(
                c == "shared_component_supported" for c in e3_classes)
                else "domain_local_temporal_representation_supported" if e3_classes and all(
                    c == "domain_local_component_supported" for c in e3_classes)
                else "direction_dependent_or_inconclusive")
            checks["E3:terminal_classification"] = bool(e3.get("experiment_terminal")) \
                and overall == result.get("classification")
            checks["E3:claim_ceiling"] = result.get("claim_ceiling") == (
                "causal component sharing on the two principal controlled beds")
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
        order_values: dict[str, list[float]] = {}
        pooled_values: dict[str, list[float]] = {}
        order_seed_effects, split_checks, order_resource_checks = [], [], []
        for row in rows:
            before = row["before_adaptation"]["B"]["per_unit_accuracy"]
            after = row["after_B_adaptation"]["B"]["per_unit_accuracy"]
            for unit in set(before) & set(after):
                unit_values.setdefault(unit, []).append(after[unit] - before[unit])
            split_checks.append(row.get("units") == _partition_units(
                "harth_stream", int(row["seed"]), offset=70_000, eval_fraction=0.0, fixed_total_eval=7))
            witness = row["temporal_order_permutation"]
            ordered, permuted = witness["ordered_per_unit_accuracy"], witness["permuted_per_unit_accuracy"]
            pooled_ordered = witness["pooled_ordered_per_unit_accuracy"]
            pooled_permuted = witness["pooled_permuted_per_unit_accuracy"]
            for unit in set(ordered) & set(permuted):
                order_values.setdefault(unit, []).append(ordered[unit] - permuted[unit])
            for unit in set(pooled_ordered) & set(pooled_permuted):
                pooled_values.setdefault(unit, []).append(pooled_ordered[unit] - pooled_permuted[unit])
            order_seed_effects.append(witness["ordered_accuracy"] - witness["permuted_accuracy"])
            order_resource_checks.append(witness.get("labels_unchanged") is True
                                         and witness.get("timestep_multiset_preserved_per_example") is True
                                         and all((witness.get("resource_match") or {}).values()))
        units = [mean(v) for v in unit_values.values()]
        order_units = [mean(v) for v in order_values.values()]
        pooled_units = [mean(v) for v in pooled_values.values()]
        future, future_group = summarize(gains), summarize(units)
        checks["HARTH-preflight:seed_count"] = {r.get("seed") for r in rows} == set(expected_seeds)
        checks["HARTH-preflight:future_gain"] = (
            _close(future["mean"], expected["future_adaptation"]["mean"])
            and _close(future["lower_95_cb"], expected["future_adaptation"]["lower_95_cb"])
            and _close(future_group["lower_95_cb"], expected["future_adaptation"]["group_lower_95_cb"]))
        checks["HARTH-preflight:return_gain"] = (
            abs(round(mean(returns), 5) - expected["returning_context"]["mean"]) < 1e-4)
        boundary = expected["context_boundary"]
        checks["HARTH-preflight:boundary"] = (
            abs(round(lower_bound(shifts), 5) - boundary["distribution_shift_lower_95_cb"]) < 1e-4
            and abs(round(lower_bound(costs), 5) - boundary["retention_cost_lower_95_cb"]) < 1e-4)
        order_seed, order_group, pooled_group = (summarize(order_seed_effects), summarize(order_units),
                                                 summarize(pooled_units))
        order_expected = expected["temporal_order_permutation"]
        checks["HARTH-preflight:split_identity"] = bool(split_checks) and all(split_checks)
        checks["HARTH-preflight:order_resource_match"] = bool(order_resource_checks) and all(
            order_resource_checks)
        checks["HARTH-preflight:temporal_order_permutation"] = (
            order_seed["verdict"] == order_expected["seed_decision"]["verdict"]
            and _close(order_seed["mean"], order_expected["seed_decision"]["mean"])
            and _close(order_seed["lower_95_cb"], order_expected["seed_decision"]["lower_95_cb"])
            and _close(order_group["lower_95_cb"], order_expected["group_lower_95_cb"])
            and _close(pooled_group["lower_95_cb"], order_expected["pooled_group_lower_95_cb"])
            and _close(pooled_group["upper_95_cb"], order_expected["pooled_group_upper_95_cb"]))
        temporal_order_required = (order_seed["verdict"] == "positive"
                                   and order_group["lower_95_cb"] >= io.SESOI)
        pooled_equivalent = (pooled_group["lower_95_cb"] >= -io.EQUIVALENCE_MARGIN
                             and pooled_group["upper_95_cb"] <= io.EQUIVALENCE_MARGIN)
        checks["HARTH-preflight:order_equivalence_classifications"] = (
            temporal_order_required == bool(expected["checks"].get("temporal_order_required"))
            and pooled_equivalent == bool(expected["checks"].get("pooled_reader_order_invariant")))
        classification = ("preflight_pass" if all((
            len(rows) == len(expected_seeds) and all(r.get("all_checks_pass") for r in rows),
            lower_bound(shifts) >= 0.02, lower_bound(gains) >= 0.02, lower_bound(costs) < 0,
            future["verdict"] == "positive" and future_group["lower_95_cb"] >= io.SESOI,
            verdict(returns) == "positive", len(units) >= 2, temporal_order_required,
            pooled_equivalent)) else "preflight_failed")
        checks["HARTH-preflight:terminal_classification"] = classification == expected.get("classification")
        checks["HARTH-preflight:claim_ceiling"] = expected.get("claim_ceiling") == (
            "secondary natural bed with a declared synthetic covariate-shift admission task")
        recomputed["HARTH-preflight"] = {"future": future, "future_group": future_group,
                                         "return": summarize(returns), "order_seed": order_seed,
                                         "order_group": order_group, "pooled_group": pooled_group}
    if io.exists("MOP_HYBRID_ADAPTATION_RESULT.json"):
        hybrid = io.load("MOP_HYBRID_ADAPTATION_RESULT.json")
        result = hybrid.get("result") or {}
        classifications = []
        for bedname, expected in (result.get("per_bed") or {}).items():
            rows = [json.loads(p.read_text()) for p in sorted((io.RUNS / "hybrid").glob(
                f"{bedname}_*.json"))]
            checks[f"hybrid:{bedname}:exact_seed_set"] = {r.get("seed") for r in rows} == set(expected_seeds)
            gain = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                    - r["arms"]["head_only"]["future_acquisition_B"]["accuracy"] for r in rows]
            noise = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                     - r["arms"]["head_plus_state_noise"]["future_acquisition_B"]["accuracy"] for r in rows]
            gain_units, noise_units = {}, {}
            metric_checks, split_checks = [], []
            for row in rows:
                split_checks.append(row.get("units") == _partition_units(
                    bedname, int(row["seed"]), offset=110_000, eval_fraction=0.25))
                hybrid_units = row["arms"]["head_plus_state"]["future_acquisition_B"]["per_unit_accuracy"]
                head_units = row["arms"]["head_only"]["future_acquisition_B"]["per_unit_accuracy"]
                noisy_units = row["arms"]["head_plus_state_noise"]["future_acquisition_B"]["per_unit_accuracy"]
                for unit in set(hybrid_units) & set(head_units) & set(noisy_units):
                    gain_units.setdefault(unit, []).append(hybrid_units[unit] - head_units[unit])
                    noise_units.setdefault(unit, []).append(hybrid_units[unit] - noisy_units[unit])
                weights = _context_weights(bedname, row["units"])
                for arm in row["arms"].values():
                    for phase, group in (("future_acquisition_B", "B_eval"),
                                         ("return_retention_A", "A_eval")):
                        values, inventory = _unit_inventory(arm[phase].get("per_unit_accuracy"))
                        metric = _weighted_metric(values, weights[group])
                        metric_checks.append(inventory["all_pass"] and metric is not None
                                             and _close(metric, arm[phase].get("accuracy")))
            gain_seed, noise_seed = summarize(gain), summarize(noise)
            gain_group, noise_group = summarize([mean(v) for v in gain_units.values()]), summarize(
                [mean(v) for v in noise_units.values()])
            gain_mine = {"mean": gain_seed["mean"], "lower_95_cb": gain_seed["lower_95_cb"],
                         "upper_95_cb": gain_seed["upper_95_cb"], "verdict": gain_seed["verdict"],
                         "group_mean": gain_group["mean"],
                         "group_lower_95_cb": gain_group["lower_95_cb"],
                         "group_upper_95_cb": gain_group["upper_95_cb"],
                         "group_heterogeneity": gain_group["heterogeneity"]}
            noise_mine = {"mean": noise_seed["mean"], "lower_95_cb": noise_seed["lower_95_cb"],
                          "upper_95_cb": noise_seed["upper_95_cb"], "verdict": noise_seed["verdict"],
                          "group_mean": noise_group["mean"],
                          "group_lower_95_cb": noise_group["lower_95_cb"],
                          "group_upper_95_cb": noise_group["upper_95_cb"],
                          "group_heterogeneity": noise_group["heterogeneity"]}
            checks[f"hybrid:{bedname}:hybrid_vs_head"] = len(rows) == len(expected_seeds) and _effect_matches(
                gain_mine, expected["hybrid_minus_head"], require_group=False)
            checks[f"hybrid:{bedname}:hybrid_vs_noise"] = len(rows) == len(expected_seeds) and _effect_matches(
                noise_mine, expected["hybrid_minus_head_noise"], require_group=False)
            checks[f"hybrid:{bedname}:metrics_reconstructed"] = bool(metric_checks) and all(metric_checks)
            checks[f"hybrid:{bedname}:split_identity"] = bool(split_checks) and all(split_checks)
            shifts = [r["before_adaptation"]["A"]["accuracy"]
                      - r["before_adaptation"]["B"]["accuracy"] for r in rows]
            costs = [r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]
                     - r["before_adaptation"]["A"]["accuracy"] for r in rows]
            acquisitions = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                            - r["before_adaptation"]["B"]["accuracy"] for r in rows]
            floors = [r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]
                      >= r["before_adaptation"]["A"]["accuracy"] - io.SESOI for r in rows]
            supported = (gain_seed["verdict"] == noise_seed["verdict"] == "positive"
                         and gain_group["lower_95_cb"] >= io.SESOI
                         and noise_group["lower_95_cb"] >= io.SESOI and all(floors)
                         and lower_bound(shifts) >= 0.02 and lower_bound(acquisitions) >= 0.02
                         and lower_bound(costs) < 0)
            classifications.append(supported)
            checks[f"hybrid:{bedname}:terminal_classification"] = supported == bool(expected.get("supported"))
            recomputed[f"hybrid:{bedname}"] = {"hybrid_minus_head": gain_mine,
                                                "hybrid_minus_noise": noise_mine}
        if result:
            expected_class = "hybrid_supported" if classifications and all(classifications) \
                else "hybrid_not_supported_under_floors"
            checks["hybrid:terminal_classification"] = bool(hybrid.get("experiment_terminal")) \
                and result.get("classification") == expected_class
            checks["hybrid:claim_ceiling"] = result.get("claim_ceiling") == (
                "early acquisition and return retention on two controlled shifted contexts")
    return {"role": "C scientific verifier", "checks": checks, "mismatches": mismatches,
            "n_checks": len(checks), "all_pass": all(checks.values()) and not mismatches,
            "recomputed": recomputed,
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
