"""Role B instrumentation auditor and Role C independent scientific verifier.

Role B never looks at an outcome. Role C recomputes every effect with its own arithmetic and its own t table
and forms its verdicts before comparing. File reading is shared on purpose; scientific logic is not.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time

import numpy as np
import torch

from fastforge import engine as E
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io

T95 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}
SESOI = 0.05
OPTIMIZATION_SEEDS = {0, 1, 2}
CORRECTED_CELLS = {Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large")),
                   Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large",
                                       readout="mlp_strong"))}
BASE_CONVERGENCE_GRID = (400, 800, 1600, 3200)
EXTENDED_CONVERGENCE_GRID = (6400, 12800)
CONVERGENCE_SEEDS = {0, 1, 2}
_PARAMETER_CACHE: dict[tuple, dict] = {}


def _expected_convergence_specs() -> list[dict]:
    """Rebuild the sealed 76 identity inventory without importing the E2 run controller."""
    specs = [dict(Fx.REFERENCE, family=f) for f in ("gru", "lstm", "mgu", "pooled", "tcn")]
    specs += [dict(Fx.REFERENCE, family="histmlp", history_k=20),
              dict(Fx.REFERENCE, tier="large"),
              dict(Fx.REFERENCE, family="pooled", tier="large", readout="mlp_strong"),
              dict(Fx.REFERENCE, family="histmlp", history_k="full_window"),
              dict(Fx.REFERENCE, reset="horizon_45"), dict(Fx.REFERENCE, reset="horizon_90")]
    for family in ("gru", "lstm", "mgu", "pooled", "histmlp", "tcn"):
        for tier in A.CAPACITY_TIERS:
            candidate = dict(Fx.REFERENCE, family=family, tier=tier)
            if Fx.cell_name(**candidate) not in {Fx.cell_name(**s) for s in specs}:
                specs.append(candidate)
    for group in ("architecture", "readout", "horizon", "reset", "capacity_by_horizon", "history",
                  "capacity_by_readout"):
        for candidate in Fx.sweep_cells()[group]:
            if Fx.cell_name(**candidate) not in {Fx.cell_name(**s) for s in specs}:
                specs.append(candidate)
    return specs


def _result_hash_valid(document: dict) -> bool:
    return (document.get("program") == io.PROGRAM
            and isinstance(document.get("source_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", document["source_commit"]) is not None
            and isinstance(document.get("source_tree_oid"), str)
            and re.fullmatch(r"[0-9a-f]{40}", document["source_tree_oid"]) is not None
            and document.get("result_hash_version") == "canonical_json_v2"
            and document.get("result_sha256") == io.sha_obj(
                {k: v for k, v in document.items() if k != "result_sha256"}))


def _parameter_count(bed: str, spec: dict) -> dict:
    key = (bed, *sorted(spec.items()))
    if key not in _PARAMETER_CACHE:
        model = Fx.build_cell(B.splits(bed, 0), seed=0, **spec)[0]
        _PARAMETER_CACHE[key] = A.count(model)
    return _PARAMETER_CACHE[key]


def _curve_receipt_checks(document: dict, *, bed: str, spec: dict,
                          grid: tuple[int, ...], scientific: bool = True) -> dict[str, bool]:
    cell = Fx.cell_name(**spec)
    curve = {int(k): v for k, v in (document.get("curve") or {}).items()}
    spread = {int(k): v for k, v in (document.get("seed_spread") or {}).items()}
    counts = document.get("parameter_count") or {}
    rebuilt = _parameter_count(bed, spec)
    band = A.TIER_RANGE.get(spec.get("tier"), (1, 0))
    checks = {
        "identity": document.get("bed") == bed and document.get("spec") == spec
        and document.get("cell") == cell,
        "exact_grid": set(curve) == set(spread) == set(grid),
        "exact_seeds": set(map(int, document.get("seeds") or [])) == CONVERGENCE_SEEDS,
        "parameter_inventory": counts == rebuilt and counts.get("total") == counts.get("core", 0)
        + counts.get("readout", 0),
        "tier_band": band[0] <= counts.get("core", -1) <= band[1],
        "canonical_result_hash": _result_hash_valid(document),
    }
    if scientific:
        finite = all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in curve.values()) \
            and all(math.isfinite(float(v)) and float(v) >= 0 for v in spread.values())
        plateau = _plateau(curve)
        checks["finite_scores"] = len(curve) == len(grid) and finite
        checks["plateau_reconstructed"] = document.get("classification") == plateau["classification"] \
            and document.get("selected_checkpoint") == plateau["selected_checkpoint"] \
            and _close(document.get("second_half_movement"), plateau["second_half_movement"]) \
            and _close(document.get("residual_slope"), plateau["residual_slope"]) \
            and document.get("converged") is plateau["all_pass"]
    records = document.get("arm_records")
    declared_scores = document.get("seed_scores")
    if isinstance(records, dict):
        record_checks = []
        for budget in grid:
            rows = records.get(str(budget), records.get(budget, []))
            structural = ({int(row.get("seed", -1)) for row in rows} == CONVERGENCE_SEEDS
                          and len(rows) == len(CONVERGENCE_SEEDS)
                          and all(int(row.get("updates", -1)) == budget
                                  and len(str(row.get("checkpoint_sha", ""))) == 64 for row in rows))
            if scientific:
                scores = [float(row.get("score")) for row in rows]
                shown = (declared_scores or {}).get(str(budget), (declared_scores or {}).get(budget, []))
                structural = (structural and len(shown) == len(scores)
                              and all(_close(a, b) for a, b in zip(shown, scores))
                              and _close(mean(scores), curve.get(budget))
                              and _close(sd(scores), spread.get(budget)))
            record_checks.append(structural)
        checks["raw_arm_records"] = bool(record_checks) and all(record_checks)
    else:
        checks["raw_arm_records"] = False
    return checks


def _convergence_audit(bed: str, *, scientific: bool = True) -> dict:
    """Reconstruct the aggregate from exact base, extension and correction receipts."""
    specs = _expected_convergence_specs()
    expected = {Fx.cell_name(**spec): spec for spec in specs}
    checks: dict[str, bool] = {"exact_76_unique_identities": len(specs) == len(expected) == 76}
    checks["exact_base_file_inventory"] = {p.name for p in (io.RUNS / "e2_converge").glob(
        f"cshard_{bed}_*.json")} == {f"cshard_{bed}_{i}.json" for i in range(76)}
    checks["exact_extended_file_inventory"] = {p.name for p in (
        io.RUNS / "e2_converge_extended").glob(f"xshard_{bed}_*.json")} == {
            f"xshard_{bed}_{i}.json" for i in range(76)}
    sources = {}
    for index, spec in enumerate(specs):
        cell = Fx.cell_name(**spec)
        base_path = io.RUNS / "e2_converge" / f"cshard_{bed}_{index}.json"
        ext_path = io.RUNS / "e2_converge_extended" / f"xshard_{bed}_{index}.json"
        base = json.loads(base_path.read_text()) if base_path.is_file() else {}
        extended = json.loads(ext_path.read_text()) if ext_path.is_file() else {}
        base_checks = _curve_receipt_checks(base, bed=bed, spec=spec, grid=BASE_CONVERGENCE_GRID,
                                            scientific=scientific) \
            if base else {"present": False}
        ext_checks = _curve_receipt_checks(extended, bed=bed, spec=spec,
                                           grid=BASE_CONVERGENCE_GRID + EXTENDED_CONVERGENCE_GRID,
                                           scientific=scientific) \
            if extended else {"present": False}
        binding = extended.get("extends") or {}
        ext_checks["base_hash_binding"] = bool(base) and binding == {
            "path": base_path.relative_to(io.ROOT).as_posix(), "sha256": io.sha_file(base_path),
            "grid": list(BASE_CONVERGENCE_GRID)}
        checks[f"base:{index}:{cell}"] = all(base_checks.values())
        checks[f"extended:{index}:{cell}"] = all(ext_checks.values())
        sources[cell] = extended
    corrected_cell = Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large"))
    correction_path = io.RUNS / "e2_converge_corrections" / f"convergence_{bed}.json"
    correction = json.loads(correction_path.read_text()) if correction_path.is_file() else {}
    correction_checks = _curve_receipt_checks(
        correction, bed=bed, spec=expected[corrected_cell],
        grid=BASE_CONVERGENCE_GRID + EXTENDED_CONVERGENCE_GRID,
        scientific=scientific) if correction else {"present": False}
    correction_checks["supersedes_exact_identity"] = set(correction.get("supersedes") or []) == {
        f"e2_converge/cshard_{bed}_25.json", f"e2_converge_extended/xshard_{bed}_25.json"}
    checks[f"correction:{corrected_cell}"] = all(correction_checks.values())
    checks["exact_correction_file_inventory"] = {p.name for p in (
        io.RUNS / "e2_converge_corrections").glob(f"*{bed}*.json")} == {correction_path.name}
    sources[corrected_cell] = correction
    aggregate_path = io.RUNS / "e2_converge" / f"converge_{bed}.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.is_file() else {}
    configs = aggregate.get("configs") or {}
    checks["aggregate_canonical_result_hash"] = _result_hash_valid(aggregate)
    checks["aggregate_v4_schema"] = aggregate.get("schema") == "mop-e2-convergence/v4"
    checks["aggregate_identity_and_inventory"] = aggregate.get("bed") == bed and set(configs) == set(expected)
    expected_index = []
    for index, spec in enumerate(specs):
        cell = Fx.cell_name(**spec)
        path = (correction_path if cell == corrected_cell else
                io.RUNS / "e2_converge_extended" / f"xshard_{bed}_{index}.json")
        expected_index.append({"cell": cell, "path": path.relative_to(io.ROOT).as_posix(),
                               "sha256": io.sha_file(path) if path.is_file() else None})
    checks["aggregate_exact_shard_index"] = aggregate.get("shard_index") == expected_index
    checks["aggregate_exact_source_receipts"] = bool(configs) and all(
        configs.get(cell) == sources.get(cell) for cell in expected)
    checks["aggregate_grid"] = set(map(int, aggregate.get("grid") or [])) == set(
        BASE_CONVERGENCE_GRID + EXTENDED_CONVERGENCE_GRID)
    if scientific:
        classifications = {cell: _plateau((row or {}).get("curve") or {})["classification"]
                           for cell, row in sources.items()}
        checks["aggregate_classifications"] = all(
            configs.get(cell, {}).get("classification") == classification
            for cell, classification in classifications.items())
        unconverged = sorted(cell for cell, value in classifications.items() if value != "converged")
        load_bearing = [Fx.cell_name(**specs[i]) for i in (0, 2, 3, 4, 8, 9, 10)]
        checks["aggregate_terminal_summaries"] = (
            aggregate.get("unconverged") == unconverged
            and aggregate.get("all_converged") is (not unconverged)
            and aggregate.get("load_bearing_cells") == load_bearing
            and aggregate.get("load_bearing_unconverged") == [
                cell for cell in load_bearing if classifications.get(cell) != "converged"]
            and aggregate.get("load_bearing_all_converged") is all(
                classifications.get(cell) == "converged" for cell in load_bearing))
    return {"checks": checks, "all_pass": all(checks.values()), "configs": configs,
            "aggregate_path": aggregate_path, "aggregate_sha256": (
                io.sha_file(aggregate_path) if aggregate_path.is_file() else None)}


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


def _summary_effect(seed_effects: list[float], per_unit: dict[str, float]) -> dict:
    seed, group = summarize(seed_effects), summarize(list(per_unit.values()))
    achieved_mde = T95.get(len(seed_effects), 1.729) * sd(seed_effects) / math.sqrt(
        len(seed_effects)) if len(seed_effects) > 1 else math.inf
    return {"mean": seed["mean"], "lower_95_cb": seed["lower_95_cb"],
            "upper_95_cb": seed["upper_95_cb"], "verdict": seed["verdict"],
            "estimator_sufficient": achieved_mde <= SESOI or seed["verdict"] in ("positive", "harm"),
            "per_seed_effects": [round(float(v), 5) for v in seed_effects],
            "per_unit_effects": {str(k): round(float(v), 5) for k, v in sorted(per_unit.items())},
            "group_mean": group["mean"], "group_lower_95_cb": group["lower_95_cb"],
            "group_upper_95_cb": group["upper_95_cb"],
            "group_heterogeneity": group["heterogeneity"], "n_units": len(per_unit)}


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


def _tensor_sha(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
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
            key = (int(row["seed"]), row["cell"])
            existing = rows.get(key) or {}
            tier = (existing.get("spec") or {}).get("tier")
            band = A.TIER_RANGE.get(tier, (1, 0))
            if not existing or not band[0] <= (existing.get("params") or {}).get("core", -1) <= band[1]:
                rows[key] = row
    return list(rows.values())


def _principal_correction_binding(bed: str, aggregate_sha: str | None,
                                  selected: dict[str, int], seeds: set[int]) -> bool:
    paths = sorted((io.RUNS / "e2_principal_corrections").glob(f"capacity_{bed}_*.json"))
    expected_cells = CORRECTED_CELLS
    if len(paths) != len(seeds):
        return False
    for path in paths:
        document = json.loads(path.read_text())
        authority = document.get("convergence_authority") or {}
        rows = document.get("runs") or []
        if not (document.get("schema") == "mop-e2-capacity-tier-correction-shard/v2"
                and _result_hash_valid(document) and document.get("seed") in seeds
                and {row.get("cell") for row in rows} == expected_cells
                and authority.get("sha256") == aggregate_sha
                and authority.get("selected_checkpoints") == {
                    cell: selected.get(cell) for cell in expected_cells}
                and all(row.get("steps") == row.get("updates") == selected.get(row.get("cell"))
                        for row in rows)):
            return False
    return True


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
    achieved_mde = T95.get(len(seed_effects), 1.729) * sd(seed_effects) / math.sqrt(
        len(seed_effects)) if len(seed_effects) > 1 else float("inf")
    return {"mean": seed_summary["mean"], "lower_95_cb": seed_summary["lower_95_cb"],
            "upper_95_cb": seed_summary["upper_95_cb"], "verdict": seed_summary["verdict"],
            "estimator_sufficient": achieved_mde <= SESOI or seed_summary["verdict"] in ("positive", "harm"),
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
    pairs = (("mean", "mean"), ("lower_95_cb", "lower_95_cb"), ("upper_95_cb", "upper_95_cb"))
    ok = all(_close(mine[a], expected.get(b)) for a, b in pairs)
    ok = ok and mine["verdict"] == expected.get("verdict")
    if "estimator_sufficient" in expected:
        ok = ok and mine["estimator_sufficient"] == expected.get("estimator_sufficient")
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
    needed = ("mean", "lower_95_cb", "upper_95_cb", "group_mean",
              "group_lower_95_cb", "group_upper_95_cb")
    return not any(effect.get(field) is None for field in needed) \
        and abs(effect["mean"]) <= margin and effect["lower_95_cb"] >= -margin \
        and effect["upper_95_cb"] <= margin and abs(effect["group_mean"]) <= margin \
        and effect["group_lower_95_cb"] >= -margin and effect["group_upper_95_cb"] <= margin


def _equivalence_row_audit(effect: dict, sealed: dict) -> tuple[bool, bool]:
    """Return scientific equivalence and sealed agreement as deliberately separate facts."""
    seed_equivalent = _equivalent(effect, io.EQUIVALENCE_MARGIN)
    group_equivalent = effect.get("group_lower_95_cb", -math.inf) >= -io.EQUIVALENCE_MARGIN \
        and effect.get("group_upper_95_cb", math.inf) <= io.EQUIVALENCE_MARGIN
    actual_pass = seed_equivalent and group_equivalent
    sealed_matches = seed_equivalent == sealed.get("seed_equivalent") \
        and group_equivalent == sealed.get("group_equivalent") \
        and all(_close(effect[field], sealed.get(field)) for field in (
            "mean", "lower_95_cb", "group_lower_95_cb", "group_upper_95_cb"))
    return actual_pass, sealed_matches


def _plateau(curve: dict) -> dict:
    """Independent reconstruction of the strict convergence witness."""
    budgets = sorted(int(k) for k in curve)
    values = [float(curve.get(str(k), curve.get(k))) for k in budgets]
    if len(values) < 4:
        return {"classification": "insufficient_budget_grid", "selected_checkpoint": None,
                "second_half_movement": None, "residual_slope": None, "all_pass": False}
    half = values[len(values) // 2:]
    movement = max(half) - min(half)
    slope = (half[-1] - half[0]) / max(1, len(half) - 1)
    best = max(range(len(values)), key=lambda i: values[i])
    checks = (best != len(values) - 1 or values[-1] - max(values[:-1]) <= 0.01,
              movement <= 0.01, slope <= 0.002, max(values) - values[-1] <= 0.03)
    return {"classification": "converged" if all(checks) else "unconverged",
            "selected_checkpoint": budgets[best], "second_half_movement": round(movement, 5),
            "residual_slope": round(slope, 6), "all_pass": all(checks)}


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


def _independent_result_keys(sealed: dict, recomputed: dict) -> list[str]:
    """Refold the preregistered result vocabulary without calling the producer analysis."""
    beds = tuple(sealed.get("principal_beds") or ())
    if set(beds) != {"har_stream", "speech_stream"} or any(b not in recomputed for b in beds):
        return []

    def ready(d):
        return (bool(d) and d.get("estimator_sufficient") is True
                and (d.get("convergence") or {}).get("all_converged") is True
                and all(d.get(k) is not None for k in (
                    "mean", "lower_95_cb", "upper_95_cb", "group_mean",
                    "group_lower_95_cb", "group_upper_95_cb")))

    def positive(d):
        return ready(d) and d.get("verdict") == "positive" and d["group_lower_95_cb"] >= SESOI

    def equivalent(d):
        return ready(d) and _equivalent(d, io.EQUIVALENCE_MARGIN)

    def group(bed, name):
        return recomputed.get(bed, {}).get(name, {})

    keys = []
    matched = [group(b, "recurrent_versus_matched_history").get(
        "gru_vs_histmlp_kfull_window", {}) for b in beds]
    if all(positive(d) for d in matched):
        keys.append("recurrent_beats_matched_history")
    if all(equivalent(d) for d in matched):
        keys.append("matched_history_matches_recurrent")
    capacity = [group(b, "capacity") for b in beds]
    monotonic = []
    for table in capacity:
        medium, large = table.get("gru_medium_vs_small", {}), table.get("gru_large_vs_small", {})
        monotonic.append(positive(medium) and positive(large)
                         and medium["mean"] <= large["mean"] + 1e-9)
    if all(monotonic):
        keys.append("capacity_monotonic_and_large")
    elif all(table and all(equivalent(d) for d in table.values()) for table in capacity):
        keys.append("capacity_flat_or_saturating")
    horizon = [group(b, "horizon") for b in beds]
    reset = [group(b, "reset") for b in beds]

    def destructive(d):
        return ready(d) and d["group_upper_95_cb"] <= -SESOI

    horizon_gate = all(
        destructive(h.get("gru_h45_vs_full", {}))
        and any(destructive(r.get(k, {})) for k in ("misaligned_a", "misaligned_b"))
        and destructive(r.get("random_rate_matched", {})) for h, r in zip(horizon, reset, strict=True))
    horizon_focus = [[table.get(k, {}) for k in ("gru_h45_vs_full", "gru_h90_vs_full")]
                     for table in horizon]
    horizon_ready = all(ready(d) for rows in horizon_focus for d in rows)
    threshold_rows = []
    for table in horizon:
        threshold = next((h for h in (1, 2, 5, 10, 20, 45, 90)
                          if equivalent(table.get(f"gru_h{h}_vs_full", {}))), None)
        threshold_rows.append(table.get(f"gru_h{threshold}_vs_full", {}) if threshold is not None else {})
    if horizon_ready and horizon_gate and all(ready(d) and equivalent(d) for d in threshold_rows):
        keys.append("horizon_threshold_at_dependency_length")
    elif horizon_ready and all(equivalent(d) for rows in horizon_focus for d in rows):
        keys.append("horizon_flat")
    interactions = [group(b, "capacity_by_horizon") for b in beds]
    if all(table and all(ready(d) for d in table.values()) for table in interactions):
        if all(any(positive(d) for d in table.values()) for table in interactions):
            keys.append("capacity_helps_only_at_long_horizon")
        elif all(all(equivalent(d) for d in table.values()) for table in interactions):
            keys.append("capacity_and_horizon_independent")
    family_pass, family_ready = {}, {}
    for family in A.RECURRENT:
        effects = [group(b, "recurrence_versus_best_stateless").get(
            f"{family}_vs_{stateless}", {}) for b in beds for stateless in A.STATELESS]
        family_pass[family], family_ready[family] = all(map(positive, effects)), all(map(ready, effects))
    if set(family_pass) == set(A.RECURRENT) and all(family_pass.values()):
        keys.append("all_recurrent_families_agree")
    elif all(family_ready.values()) and any(family_pass.values()):
        keys.append("one_recurrent_family_dissents")
    optimization_ready = all((recomputed.get(f"optimization:{b}") or {}).get("receipt_valid") is True
                             and (recomputed.get(f"optimization:{b}") or {}).get("converged") is True
                             for b in beds)
    if optimization_ready and all(positive(d) for d in matched):
        keys.append("converged_everywhere_and_gap_remains")
    preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") if io.exists(
        "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {}
    third = group("harth_stream", "recurrent_versus_matched_history").get(
        "gru_vs_histmlp_kfull_window", {})
    if "harth_stream" in (preflight.get("selected") or []) and third:
        if positive(third):
            keys.append("third_bed_agrees")
        elif ready(third) and (_equivalent(third, io.EQUIVALENCE_MARGIN)
                               or third.get("group_upper_95_cb", math.inf) <= -SESOI):
            keys.append("third_bed_dissents")
    else:
        keys.append("third_bed_invalid")
    readouts = [group(b, "readout") for b in beds]
    if all(table and all(ready(d) for d in table.values()) for table in readouts):
        if all(any(positive(d) for d in table.values()) for table in readouts):
            keys.append("readout_capacity_reproduces_the_effect")
        elif all(all(equivalent(d) for d in table.values()) for table in readouts):
            keys.append("readout_capacity_flat")
    return keys


def _independent_hypothesis_fold(results: list[str]) -> dict:
    """Local immutable copy of the preregistered reducer, with no producer hypothesis import."""
    hypotheses = tuple(f"H{i}_{name}" for i, name in enumerate((
        "recurrence", "explicit_history", "capacity", "state_horizon", "optimization",
        "core_horizon_interaction", "architecture_family", "bed_specificity"), 1))
    mapping = {
        "recurrent_beats_matched_history": ((hypotheses[0],), (hypotheses[1],), (), (hypotheses[2], hypotheses[3])),
        "matched_history_matches_recurrent": ((hypotheses[1],), (hypotheses[0],), (hypotheses[0],), (hypotheses[3],)),
        "capacity_monotonic_and_large": ((hypotheses[2],), (hypotheses[0],), (), (hypotheses[5],)),
        "capacity_flat_or_saturating": ((), (hypotheses[2],), (hypotheses[2],), ()),
        "horizon_threshold_at_dependency_length": ((hypotheses[3],), (), (), (hypotheses[5],)),
        "horizon_flat": ((), (hypotheses[3], hypotheses[0]), (hypotheses[3],), ()),
        "capacity_helps_only_at_long_horizon": ((hypotheses[5],), (), (), (hypotheses[2],)),
        "capacity_and_horizon_independent": ((), (hypotheses[5],), (hypotheses[5],), ()),
        "all_recurrent_families_agree": ((), (hypotheses[6],), (hypotheses[6],), (hypotheses[0],)),
        "one_recurrent_family_dissents": ((hypotheses[6],), (hypotheses[0],), (), (hypotheses[4],)),
        "unconverged_arms_explain_the_gap": ((hypotheses[4],), (hypotheses[0], hypotheses[2]), (), ()),
        "converged_everywhere_and_gap_remains": ((), (hypotheses[4],), (hypotheses[4],), ()),
        "third_bed_agrees": ((), (hypotheses[7],), (hypotheses[7],), ()),
        "third_bed_dissents": ((hypotheses[7],), (), (), (hypotheses[0],)),
        "third_bed_invalid": ((), (), (), (hypotheses[7],)),
        "readout_capacity_reproduces_the_effect": ((hypotheses[2],), (hypotheses[0],), (), ()),
        "readout_capacity_flat": ((), (hypotheses[2],), (), ()),
    }
    state = {h: {kind: [] for kind in ("supports", "weakens", "closes", "unresolved")}
             for h in hypotheses}
    unknown = [result for result in results if result not in mapping]
    for result in results:
        for kind, named in zip(("supports", "weakens", "closes", "unresolved"),
                               mapping.get(result, ((), (), (), ())), strict=True):
            for hypothesis in named:
                state[hypothesis][kind].append(result)
    folded = {}
    for hypothesis, evidence in state.items():
        status = ("closed" if evidence["closes"] else "supported" if evidence["supports"]
                  and not evidence["weakens"] else "mixed" if evidence["supports"]
                  and evidence["weakens"] else "weakened" if evidence["weakens"] else
                  "unresolved" if evidence["unresolved"] else "open")
        folded[hypothesis] = {"state": status, "evidence": evidence}
    return {"hypotheses": folded, "unknown_result_keys": unknown,
            "observed_results": sorted(set(results))}


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
        checks[f"{bed}:original_capacity_defect_inventory_bound"] = len(invalid_originals) == int(
            (correction.get("original_invalid_receipts") or {}).get(bed, -1))
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
        checks[f"{bed}:principal_parameters_rebuilt_exactly"] = all(
            r.get("params") == _parameter_count(bed, r["spec"]) for r in runs)
        convergence = _convergence_audit(bed, scientific=False)
        checks[f"{bed}:exact_raw_convergence_authority"] = convergence["all_pass"]
        checkpoints = {cell: int(row["selected_checkpoint"])
                       for cell, row in convergence["configs"].items()
                       if row.get("selected_checkpoint") is not None}
        checks[f"{bed}:training_budget_matches_selected_checkpoint"] = bool(runs) and all(
            r["steps"] == r["updates"] == checkpoints.get(r["cell"]) for r in runs)
        principal_shards = [json.loads(p.read_text()) for p in sorted(
            (io.RUNS / "e2_principal").glob(f"{bed}_*.json"))]
        checks[f"{bed}:exact_principal_shard_inventory"] = len(principal_shards) == len(
            doc.get("seeds") or []) and all(_result_hash_valid(shard) for shard in principal_shards)
        expected_factorial = {Fx.cell_name(**spec) for spec in Fx.sweep_cells()["_all"]}
        checks[f"{bed}:principal_shards_bind_exact_convergence_hash"] = bool(principal_shards) and all(
            shard.get("schema") == "mop-e2-principal-shard/v2"
            and (shard.get("convergence_authority") or {}).get("sha256")
            == convergence["aggregate_sha256"]
            and (shard.get("convergence_authority") or {}).get("selected_checkpoints") == checkpoints
            and (shard.get("convergence_authority") or {}).get("all_factorial_cells_measured") is True
            and {row.get("cell") for row in shard.get("runs") or []} == expected_factorial
            for shard in principal_shards)
        checks[f"{bed}:principal_corrections_bind_exact_convergence_hash"] = _principal_correction_binding(
            bed, convergence["aggregate_sha256"], checkpoints, set(doc.get("seeds") or []))
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
        identity = noisy_trace.get("matched_noise_identity") or {}
        try:
            dtype = getattr(torch, str(identity["dtype"]).removeprefix("torch."))
            generator = torch.Generator(device="cpu").manual_seed(int(identity["rng_seed"]))
            regenerated = torch.randn(tuple(identity["shape"]), dtype=dtype, generator=generator)
            regenerated = regenerated * float(identity["learned_state_norm"]) / float(
                regenerated.norm() + 1e-9)
            direction = regenerated / (regenerated.norm() + 1e-12)
            expected_l2 = math.sqrt(max(0.0, float(identity["learned_state_norm"]) ** 2
                                        + float(identity["noise_norm"]) ** 2
                                        - 2 * float(identity["learned_state_norm"])
                                        * float(identity["noise_norm"])
                                        * float(identity["cosine_with_learned_state"])))
            learned_trace = (arms.get("head_plus_state") or {}).get("trace") or {}
            checks[f"{key}:noise_tensor_reconstructed"] = (
                _tensor_sha(regenerated) == identity.get("noise_tensor_sha256")
                and _tensor_sha(direction) == identity.get("noise_direction_sha256")
                and _close(float(regenerated.norm()), identity.get("noise_norm"), tol=1.1e-5))
            checks[f"{key}:learned_state_and_readout_bound"] = (
                identity.get("learned_state_sha256") == identity.get("regenerated_learned_state_sha256")
                == learned_trace.get("state_sha256_after")
                and identity.get("reference_learned_readout_group_sha256")
                == learned_trace.get("readout_group_sha256_after")
                == identity.get("readout_group_sha256_after")
                and identity.get("readout_matches_reference") is True)
            checks[f"{key}:noise_difference_witness"] = _close(
                expected_l2, identity.get("l2_difference_from_learned_state"), tol=1.1e-5)
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError):
            checks[f"{key}:noise_tensor_reconstructed"] = False
            checks[f"{key}:learned_state_and_readout_bound"] = False
            checks[f"{key}:noise_difference_witness"] = False
    hybrid_by_bed: dict[str, set] = {}
    for p in hybrid_paths:
        d = json.loads(p.read_text())
        hybrid_by_bed.setdefault(str(d.get("bed")), set()).add(d.get("seed"))
    for bed, seeds in hybrid_by_bed.items():
        checks[f"hybrid:{bed}:exact_seed_set"] = seeds == expected_seeds
    optimization_paths = sorted((io.RUNS / "e2_optimization_corrections").glob("optimization_*.json")) \
        if (io.RUNS / "e2_optimization_corrections").is_dir() else []
    for p in optimization_paths:
        d = json.loads(p.read_text())
        bed, tier = d.get("bed"), d.get("tier")
        key = f"optimization:{bed}:{tier}"
        spec = d.get("spec") or {}
        records = d.get("arm_records") or {}
        expected_budgets = set(map(int, d.get("regular_grid") or []))
        if tier == "small":
            expected_budgets.add(int((d.get("compute_match") or {}).get("small_steps", -1)))
        flattened = [(int(budget), row) for budget, rows in records.items() for row in rows]
        checks[f"{key}:same_family_capacity_only"] = (
            spec.get("family") == "gru" and spec.get("tier") == tier
            and Fx.cell_name(**spec) == d.get("cell"))
        expected_roles = ({"small_model_at_same_compute", "small_model_at_strict_selected_convergence"}
                          if tier == "small" else
                          {"large_model_at_same_update_count", "large_model_at_strict_selected_convergence"})
        checks[f"{key}:exact_role_inventory"] = set(d.get("four_contrast_roles") or {}) == expected_roles
        checks[f"{key}:capacity_band"] = tier in A.TIER_RANGE and A.TIER_RANGE[tier][0] <= int(
            (d.get("parameter_count") or {}).get("core", -1)) <= A.TIER_RANGE[tier][1]
        checks[f"{key}:exact_budget_inventory"] = set(map(int, records)) == expected_budgets
        checks[f"{key}:exact_seed_inventory"] = bool(flattened) and all(
            {int(row.get("seed")) for row in rows} == OPTIMIZATION_SEEDS for rows in records.values())
        checks[f"{key}:parameter_update_exposure"] = bool(flattened) and all(
            int(row.get("updates", -1)) == budget
            and int(row.get("trainable_param_count", -1)) == int((d.get("parameter_count") or {}).get("total", -2))
            and int(row.get("parameter_update_exposure", -1))
            == int(row.get("trainable_param_count", 0)) * budget for budget, row in flattened)
        checks[f"{key}:checkpoint_receipts"] = bool(flattened) and all(
            isinstance(row.get("checkpoint_sha"), str) and len(row["checkpoint_sha"]) == 64
            for _, row in flattened)
        match = d.get("compute_match") or {}
        checks[f"{key}:declared_compute_match"] = (
            match.get("large_parameter_updates", 0) > 0 and match.get("small_parameter_updates", 0) > 0
            and abs(float(match["large_parameter_updates"]) - float(match["small_parameter_updates"]))
            / float(match["large_parameter_updates"]) <= 0.0001)
    if optimization_paths:
        checks["optimization:exact_six_receipts"] = len(optimization_paths) == 6
        docs = [json.loads(p.read_text()) for p in optimization_paths]
        expected_cartesian = {(bed, tier) for bed in ("har_stream", "speech_stream", "harth_stream")
                              for tier in ("small", "large")}
        checks["optimization:exact_bed_tier_cartesian_inventory"] = {
            (d.get("bed"), d.get("tier")) for d in docs} == expected_cartesian
        checks["optimization:filenames_bind_identity"] = all(
            p.name == f"optimization_{d.get('bed')}_{d.get('tier')}.json"
            for p, d in zip(optimization_paths, docs, strict=True))
        for bed in ("har_stream", "speech_stream", "harth_stream"):
            pair = {d.get("tier"): d for d in docs if d.get("bed") == bed}
            pair_ok = set(pair) == {"small", "large"}
            if pair_ok:
                large, small = pair["large"], pair["small"]
                anchor = int(large.get("same_update_anchor", -1))
                match_budget = int((small.get("compute_match") or {}).get("small_steps", -1))
                large_rows = (large.get("arm_records") or {}).get(str(anchor), [])
                small_rows = (small.get("arm_records") or {}).get(str(match_budget), [])
                large_exposure = {int(r.get("seed")): int(r.get("parameter_update_exposure", -1))
                                  for r in large_rows}
                small_exposure = {int(r.get("seed")): int(r.get("parameter_update_exposure", -2))
                                  for r in small_rows}
                pair_ok = (set(large_exposure) == set(small_exposure) == OPTIMIZATION_SEEDS
                           and all(abs(large_exposure[s] - small_exposure[s]) / large_exposure[s] <= 0.0001
                                   for s in OPTIMIZATION_SEEDS)
                           and {k: v for k, v in large.get("spec", {}).items() if k != "tier"}
                           == {k: v for k, v in small.get("spec", {}).items() if k != "tier"})
            checks[f"optimization:{bed}:paired_actual_compute_and_spec"] = pair_ok
    if io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json"):
        owned = io.load("MOP_OWNED_TEMPORAL_CORE_V1.json")
        core = owned.get("core") or {}
        checkpoint_receipts = core.get("checkpoints") or {}
        valid_domains = set(core.get("valid_domains") or [])
        disk_beds = {p.stem.removeprefix("owned_temporal_core_v1_")
                     for p in (io.PROOF / "checkpoints").glob("owned_temporal_core_v1_*.pt")}
        checks["owned_core:exact_checkpoint_bed_inventory"] = (
            set(checkpoint_receipts) == valid_domains and disk_beds == valid_domains)
        for bed, receipt in checkpoint_receipts.items():
            path = io.ROOT / str(receipt.get("path"))
            key = f"owned_core:checkpoint:{bed}"
            checks[f"{key}:file_hash"] = path.is_file() and io.sha_file(path) == receipt.get("sha256")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                spec = payload.get("spec") or {}
                model = Fx.build_cell(B.splits(bed, int(payload.get("seed", 0))),
                                      seed=int(payload.get("seed", 0)), **spec)[0]
                restored = model.load_state_dict(payload.get("state_dict") or {}, strict=True)
                checks[f"{key}:payload_identity"] = (
                    payload.get("schema") == "mop-owned-temporal-core-checkpoint/v1"
                    and payload.get("bed") == bed and spec == (owned.get("selection") or {}).get(
                        "selected", {}).get("spec") and not restored.missing_keys
                    and not restored.unexpected_keys)
                checks[f"{key}:restored_parameter_inventory"] = (
                    A.count(model) == payload.get("params") == receipt.get("params"))
                checks[f"{key}:restored_checkpoint_hash"] = E.checkpoint_sha(model) == receipt.get(
                    "checkpoint_sha")
                selected_cell = ((owned.get("selection") or {}).get("selected") or {}).get("cell")
                principal = io.load("MOP_E2_PRINCIPAL_RESULT.json")
                expected_checkpoint = (((principal.get("per_bed") or {}).get(bed, {}).get(
                    "convergence") or {}).get("configs") or {}).get(selected_cell, {}).get(
                        "selected_checkpoint")
                training = payload.get("training_receipt") or {}
                checks[f"{key}:selected_budget_binding"] = (
                    isinstance(expected_checkpoint, int)
                    and payload.get("selected_checkpoint") == receipt.get("selected_checkpoint")
                    == expected_checkpoint and training.get("updates") == expected_checkpoint)
                checks[f"{key}:launch_provenance"] = (
                    payload.get("seed") == 0 and payload.get("source_commit") == owned.get("source_commit")
                    and payload.get("source_tree_oid") == owned.get("source_tree_oid")
                    and isinstance(payload.get("source_commit"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", payload["source_commit"]) is not None
                    and isinstance(payload.get("source_tree_oid"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", payload["source_tree_oid"]) is not None)
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                checks[f"{key}:payload_identity"] = False
                checks[f"{key}:restored_parameter_inventory"] = False
                checks[f"{key}:restored_checkpoint_hash"] = False
                checks[f"{key}:selected_budget_binding"] = False
                checks[f"{key}:launch_provenance"] = False
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
    owned, independently_admitted, independently_third_replicated = {}, False, False
    for bed, a in sealed["per_bed"].items():
        if a.get("status") == "no_runs":
            continue
        runs = _principal_runs(bed)
        by_cell, unit_cells = _effect_inputs(runs)
        weights = _test_unit_weights(bed)
        convergence_audit = _convergence_audit(bed)
        conv_configs = convergence_audit["configs"]

        def independent_convergence(cell: str) -> str:
            raw = conv_configs.get(cell) or conv_configs.get(cell.replace("|horizon_full|", "|none|"))
            return _plateau((raw or {}).get("curve") or {})["classification"] if raw else "not_measured"
        checks[f"{bed}:exact_raw_convergence_authority"] = convergence_audit["all_pass"]
        selected_checkpoints = {cell: int(row["selected_checkpoint"])
                                for cell, row in conv_configs.items()
                                if row.get("selected_checkpoint") is not None}
        checks[f"{bed}:exact_seed_set"] = {int(r["seed"]) for r in runs} == set(expected_seeds)
        checks[f"{bed}:every_cell_has_exact_seed_set"] = bool(by_cell) and all(
            set(rows) == set(expected_seeds) for rows in by_cell.values())
        receipt_identity, unit_identity, metrics = [], [], []
        for r in runs:
            values, inventory = _unit_inventory(r.get("per_unit_accuracy"))
            receipt_identity.append(r.get("bed") == bed and int(r.get("seed")) in expected_seeds
                                    and r.get("eval_on") == "test"
                                    and r.get("cell") == Fx.cell_name(**r.get("spec", {}))
                                    and r.get("steps") == r.get("updates")
                                    == selected_checkpoints.get(r.get("cell")))
            unit_identity.append(inventory["all_pass"] and set(values) == set(weights))
            metric = _weighted_metric(values, weights)
            metrics.append(metric is not None and _close(metric, r.get("accuracy")))
        checks[f"{bed}:receipt_bed_seed_cell_and_split_identity"] = bool(runs) and all(receipt_identity)
        checks[f"{bed}:exact_evaluation_unit_identity"] = bool(runs) and all(unit_identity)
        checks[f"{bed}:accuracy_reconstructed_from_per_unit_receipts"] = bool(runs) and all(metrics)
        expected_factorial = {Fx.cell_name(**spec) for spec in Fx.sweep_cells()["_all"]}
        principal_shards = [json.loads(p.read_text()) for p in sorted(
            (io.RUNS / "e2_principal").glob(f"{bed}_*.json"))]
        checks[f"{bed}:exact_principal_shard_inventory"] = len(principal_shards) == len(expected_seeds) \
            and all(_result_hash_valid(shard) for shard in principal_shards)
        checks[f"{bed}:principal_convergence_hash_binding"] = bool(principal_shards) and all(
            shard.get("schema") == "mop-e2-principal-shard/v2"
            and (shard.get("convergence_authority") or {}).get("sha256")
            == convergence_audit["aggregate_sha256"]
            and (shard.get("convergence_authority") or {}).get("selected_checkpoints")
            == selected_checkpoints
            and {row.get("cell") for row in shard.get("runs") or []} == expected_factorial
            for shard in principal_shards)
        checks[f"{bed}:principal_correction_convergence_hash_binding"] = _principal_correction_binding(
            bed, convergence_audit["aggregate_sha256"], selected_checkpoints, set(expected_seeds))
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
                if mine is not None:
                    parameter_sum = sum(int(by_cell[cell][expected_seeds[0]]["params"]["total"])
                                        for cell in components if cell in by_cell)
                    statuses = {cell: independent_convergence(cell) for cell in components}
                    all_converged = bool(statuses) and all(v == "converged" for v in statuses.values())
                    mine["component_parameter_sum"] = parameter_sum
                    mine["convergence"] = {"cells": statuses, "all_converged": all_converged,
                                           "classification": ("converged" if all_converged else
                                                              "provisional_unconverged_or_unmeasured")}
                    mine["cost_adjusted_effect_per_100k_parameters"] = (
                        round(float(mine["mean"]) * 100_000 / parameter_sum, 5)
                        if parameter_sum else None)
                    mine["component_floor_status"] = (
                        "passes" if all_converged and mine["group_lower_95_cb"] >= io.SESOI
                        else "provisional_or_below_floor")
                recomputed[bed][group][k] = mine
                ok = (_effect_matches(mine, d) and mine is not None
                      and mine["component_parameter_sum"] == d.get("component_parameter_sum")
                      and _close(mine["cost_adjusted_effect_per_100k_parameters"],
                                 d.get("cost_adjusted_effect_per_100k_parameters"))
                      and mine["component_floor_status"] == d.get("component_floor_status"))
                if group in {"core_by_readout", "core_by_capacity", "core_by_horizon",
                             "readout_by_capacity", "history_by_architecture", "capacity_by_horizon"}:
                    did = (len(components) == 4 and signs == [1, -1, -1, 1]
                           and d.get("estimand") == "difference_in_differences")
                    checks[f"{bed}:{group}:{k}:did_identity"] = did
                    ok = ok and did
                convergence = d.get("convergence") or {}
                declared = convergence.get("cells") or {}
                checks[f"{bed}:{group}:{k}:convergence_classification"] = (
                    bool(declared) and mine is not None and declared == mine["convergence"]["cells"]
                    and convergence.get("all_converged") == mine["convergence"]["all_converged"]
                    and convergence.get("classification") == (
                        "converged" if mine["convergence"]["all_converged"]
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
                recomputed[f"replication:{bed}:{key}"] = mine
                checks[f"{bed}:independent_replication:{key}"] = ok
                if not ok:
                    mismatches.append({"bed": bed, "contrast": key,
                                       "sealed": expected, "recomputed": mine})

    if io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json"):
        owned = io.load("MOP_OWNED_TEMPORAL_CORE_V1.json")
        selected_core = owned.get("selected") is True
        selection = owned.get("selection") or {}
        evidence = selection.get("equivalence_evidence") or {}
        principal_beds = list(sealed["principal_beds"])
        bed_inputs = {bed: _effect_inputs(_principal_runs(bed)) for bed in principal_beds}
        shared = set.intersection(*(set(cells) for cells, _ in bed_inputs.values())) if bed_inputs else set()
        shared = {cell for cell in shared if cell.split("|")[3] in ("none", "horizon_full")
                  or cell.split("|")[3].startswith("horizon_")}
        convergence_by_bed = {bed: _convergence_audit(bed)["configs"] for bed in principal_beds}
        shared = {cell for cell in shared if all(
            _plateau((convergence_by_bed[bed].get(cell) or {}).get("curve") or {})[
                "classification"] == "converged" for bed in principal_beds)}
        if not selected_core:
            shared = set()
        worst = {cell: min(mean([float(row["accuracy"]) for row in bed_inputs[bed][0][cell].values()])
                            for bed in principal_beds) for cell in shared}
        best = max(worst, key=worst.get) if worst else None
        checks["owned_core:best_cell_recomputed"] = best == selection.get("best_cell")
        checks["owned_core:exact_equivalence_candidate_inventory"] = set(evidence) == shared
        checks["owned_core:equivalence_evidence_present"] = not selection.get("selected") or bool(evidence)
        independently_equivalent = []
        for candidate in sorted(shared):
            expected = evidence.get(candidate) or {}
            per_bed, actual_pass, sealed_matches = {}, True, True
            for bed in principal_beds:
                by_cell, unit_cells = bed_inputs[bed]
                effect = _recompute_effect(by_cell, unit_cells, [candidate, best], [1, -1], expected_seeds)
                if effect is None:
                    actual_pass = False
                    sealed_matches = False
                    continue
                seed_equivalent = _equivalent(effect, io.EQUIVALENCE_MARGIN)
                group_equivalent = (effect["group_lower_95_cb"] >= -io.EQUIVALENCE_MARGIN
                                    and effect["group_upper_95_cb"] <= io.EQUIVALENCE_MARGIN)
                per_bed[bed] = {"seed_equivalent": seed_equivalent,
                                "group_equivalent": group_equivalent, **effect}
                sealed_bed = (expected.get("per_bed") or {}).get(bed, {})
                row_actual, row_matches = _equivalence_row_audit(effect, sealed_bed)
                actual_pass = actual_pass and row_actual
                sealed_matches = sealed_matches and row_matches
            sealed_matches = (sealed_matches and set((expected.get("per_bed") or {})) == set(principal_beds)
                              and expected.get("passes") is actual_pass
                              and expected.get("source") == "paired raw seed and independent-unit effects")
            checks[f"owned_core:equivalence:{candidate}:sealed_matches"] = sealed_matches
            if actual_pass:
                independently_equivalent.append(candidate)
        checks["owned_core:equivalent_region"] = sorted(independently_equivalent) == sorted(
            selection.get("equivalent_region") or [])
        simplicity = {"pooled": 0, "histmlp": 1, "tcn": 2, "mgu": 3, "gru": 4, "lstm": 5,
                      "ff_gru": 6}
        readout_simplicity = {"linear": 0, "mlp1": 1, "mlp_strong": 2}
        ordered = []
        replicated_cells = {Fx.cell_name(**dict(Fx.REFERENCE, family=family))
                            for family in ("gru", "mgu")}
        for cell in independently_equivalent:
            family, tier, readout, reset, history = cell.split("|")
            if cell not in replicated_cells:
                continue
            rows = [row for bed in principal_beds for row in bed_inputs[bed][0][cell].values()]
            spec = {"family": family, "tier": tier, "readout": readout, "reset": reset,
                    "history_k": history.removeprefix("h")}
            spec["history_k"] = int(spec["history_k"]) if spec["history_k"].isdigit() else spec["history_k"]
            params = _parameter_count(principal_beds[0], spec)
            checks[f"owned_core:parameters_rebuilt:{cell}"] = all(row.get("params") == params for row in rows)
            per_bed_wall = [mean([float(row.get("wall_seconds", math.inf))
                                  for row in bed_inputs[bed][0][cell].values()]) for bed in principal_beds]
            horizon = 192 if reset in ("none", "horizon_full") else int(reset.split("_")[1])
            ordered.append({"cell": cell, "params": int(params["total"]), "compute": max(per_bed_wall),
                            "horizon": horizon, "readout": readout_simplicity.get(readout, 9),
                            "architecture": simplicity.get(family, 9), "spec": spec})
        ordered.sort(key=lambda row: (row["params"], row["compute"], row["horizon"], row["readout"],
                                     row["architecture"]))
        sealed_order = [row.get("cell") for row in selection.get("ordered_candidates") or []]
        checks["owned_core:minimal_order_recomputed"] = sealed_order == [
            row["cell"] for row in ordered[:12]]
        selected = selection.get("selected") or {}
        checks["owned_core:selected_is_actual_minimum"] = ((not selected_core and not selected)
            or bool(ordered) and selected.get("cell") == ordered[0]["cell"])
        checks["owned_core:no_selection_is_explicit"] = selected_core or (
            bool(selection.get("reason")) and not owned.get("core"))
        checks["owned_core:claim_ceiling"] = owned.get("evidence_ceiling") == (
            "this is a substrate component with evidence on the beds named here. It does not establish a "
            "complete substrate architecture, continual plasticity, cross domain transfer, functional "
            "reorganization or activation") and owned.get("activation") is False
        preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") if io.exists(
            "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {}
        third_result = io.load("MOP_THIRD_TEMPORAL_BED_RESULT.json") if io.exists(
            "MOP_THIRD_TEMPORAL_BED_RESULT.json") else {}
        replication = io.load("MOP_E2_INDEPENDENT_REPLICATION.json") if io.exists(
            "MOP_E2_INDEPENDENT_REPLICATION.json") else {}
        admitted = isinstance(preflight.get("selected"), list) and "harth_stream" in preflight["selected"]
        third_effects = [recomputed.get(f"replication:harth_stream:{key}") or {}
                         for key in ("torch_gru_vs_full_history", "explicit_mgu_vs_full_history")]
        third_conv_path = io.RUNS / "e2_converge" / "converge_harth_stream.json"
        third_configs = (json.loads(third_conv_path.read_text()).get("configs") or {}) \
            if third_conv_path.is_file() else {}
        history_cell = "histmlp|small|linear|none|hfull_window"
        effect_cells = ("gru|small|linear|none|h1", "mgu|small|linear|none|h1")
        conv_pairs = [all(_plateau((third_configs.get(cell) or {}).get("curve") or {})[
            "classification"] == "converged" for cell in (effect_cell, history_cell))
                      for effect_cell in effect_cells]
        reproduces = all(d.get("verdict") == "positive" and d.get("group_lower_95_cb", -math.inf) >= SESOI
                         and converged for d, converged in zip(third_effects, conv_pairs, strict=True))
        independently_third_replicated = reproduces
        third_classification = ("invalid_secondary_bed" if not admitted else "replicated" if reproduces
                                else "valid_secondary_bed_did_not_reproduce_the_principal_effect")
        checks["third_bed:admission_binding"] = (
            third_result.get("admitted") == admitted
            and replication.get("third_bed_admitted") == admitted
            and third_result.get("classification") == third_classification
            and replication.get("third_bed_classification") == third_classification)
        checks["third_bed:claim_ceiling"] = third_result.get("claim_ceiling") == (
            "secondary natural bed only; it is not promoted to a principal adaptation bed")
        valid_domains = ((owned.get("core") or {}).get("valid_domains") or [])
        checks["owned_core:third_domain_license"] = ((not selected_core and not valid_domains) or
            ("harth_stream" in valid_domains) == (admitted and third_classification == "replicated"))
        checkpoint_receipts = ((owned.get("core") or {}).get("checkpoints") or {})
        disk_beds = {p.stem.removeprefix("owned_temporal_core_v1_")
                     for p in (io.PROOF / "checkpoints").glob("owned_temporal_core_v1_*.pt")}
        checks["owned_core:exact_checkpoint_bed_inventory"] = (
            set(checkpoint_receipts) == set(valid_domains) and disk_beds == set(valid_domains))

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
                    component_parameters = int((left or {}).get("component_parameter_sum") or 0) + int(
                        (right or {}).get("component_parameter_sum") or 0)
                    converged = all((row or {}).get("convergence", {}).get("all_converged")
                                    for row in (left, right))
                    adjusted = (seed_welch["mean"] * 100_000 / component_parameters
                                if component_parameters else None)
                    floor = ("passes" if converged and unit_welch["lower_95_cb"] >= io.SESOI
                             else "provisional_or_below_floor")
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
                          and component_parameters == expected.get("component_parameter_sum")
                          and _close(adjusted, expected.get("cost_adjusted_effect_per_100k_parameters"))
                          and converged == (expected.get("convergence") or {}).get("all_converged")
                          and floor == expected.get("component_floor_status")
                          and independent_verdict == expected.get("verdict"))
                checks[f"interaction:{report_key}:{key}"] = ok
                if not ok:
                    mismatches.append({"bed": "cross_bed", "contrast": f"{report_key}:{key}",
                                       "sealed": expected,
                                       "recomputed": {"seed": seed_welch, "group": unit_welch}})
        for bed, expected in (interactions.get("optimization_by_capacity") or {}).items():
            paths = {tier: io.RUNS / "e2_optimization_corrections" /
                     f"optimization_{bed}_{tier}.json" for tier in ("large", "small")}
            docs = {tier: json.loads(path.read_text()) for tier, path in paths.items()
                    if path.is_file()}
            budgets = expected.get("budgets") or {}

            def at(document, key, budget):
                table = document.get(key) or {}
                return table.get(str(budget), table.get(budget))

            if set(docs) == {"large", "small"} and set(budgets) == {
                    "large_same_update", "large_convergence", "small_same_compute", "small_convergence"}:
                large, small = docs["large"], docs["small"]
                arm_defs = {
                    "large_convergence": (large, budgets["large_convergence"],
                                          "large_model_at_strict_selected_convergence"),
                    "large_same_update": (large, budgets["large_same_update"],
                                          "large_model_at_same_update_count"),
                    "small_convergence": (small, budgets["small_convergence"],
                                          "small_model_at_strict_selected_convergence"),
                    "small_same_compute": (small, budgets["small_same_compute"],
                                           "small_model_at_same_compute")}
                records, receipt_checks = {}, {"canonical_launch_provenance": all(
                    _result_hash_valid(document) for document in docs.values())}
                for name, (doc, budget, role) in arm_defs.items():
                    rows = at(doc, "arm_records", budget) or []
                    ids = [int(row.get("seed", -1)) for row in rows]
                    records[name] = {int(row["seed"]): row for row in rows if "seed" in row}
                    role_row = (doc.get("four_contrast_roles") or {}).get(role) or {}
                    receipt_checks[f"{name}:seeds"] = set(ids) == OPTIMIZATION_SEEDS \
                        and len(ids) == len(set(ids)) == len(OPTIMIZATION_SEEDS)
                    receipt_checks[f"{name}:role"] = role_row.get("budget") == budget \
                        and role_row.get("records") == rows
                    receipt_checks[f"{name}:exposure"] = bool(rows) and all(
                        int(row.get("updates", -1)) == budget
                        and int(row.get("parameter_update_exposure", -1))
                        == int(row.get("trainable_param_count", 0)) * budget for row in rows)
                receipt_checks["roles"] = (
                    set(large.get("four_contrast_roles") or {}) == {
                        "large_model_at_same_update_count", "large_model_at_strict_selected_convergence"}
                    and set(small.get("four_contrast_roles") or {}) == {
                        "small_model_at_same_compute", "small_model_at_strict_selected_convergence"})
                receipt_checks["specs"] = (
                    large.get("bed") == small.get("bed") == bed
                    and large.get("spec", {}).get("family") == small.get("spec", {}).get("family") == "gru"
                    and large.get("spec", {}).get("tier") == "large"
                    and small.get("spec", {}).get("tier") == "small"
                    and {k: v for k, v in large.get("spec", {}).items() if k != "tier"}
                    == {k: v for k, v in small.get("spec", {}).items() if k != "tier"})
                receipt_checks["parameter_bands"] = all(
                    A.TIER_RANGE[tier][0] <= int((docs[tier].get("parameter_count") or {}).get("core", -1))
                    <= A.TIER_RANGE[tier][1] for tier in docs)
                for tier, doc in docs.items():
                    for budget in map(int, doc.get("regular_grid") or []):
                        rows = at(doc, "arm_records", budget) or []
                        scores = [float(row["score"]) for row in rows]
                        receipt_checks[f"{tier}:{budget}:curve"] = (
                            len(scores) == len(OPTIMIZATION_SEEDS)
                            and _close(mean(scores), at(doc, "curve", budget))
                            and _close(sd(scores), at(doc, "seed_spread", budget)))
                for seed in OPTIMIZATION_SEEDS:
                    inventories = [set(records.get(name, {}).get(seed, {}).get("per_unit_accuracy") or {})
                                   for name in arm_defs]
                    receipt_checks[f"units:{seed}"] = all(inventories) and all(
                        inv == inventories[0] for inv in inventories[1:])
                actual_errors = []
                for seed in OPTIMIZATION_SEEDS:
                    le = float(records.get("large_same_update", {}).get(seed, {}).get(
                        "parameter_update_exposure", 0))
                    se = float(records.get("small_same_compute", {}).get(seed, {}).get(
                        "parameter_update_exposure", 0))
                    actual_errors.append(abs(le - se) / le if le else math.inf)
                receipt_checks["compute_match"] = bool(actual_errors) and max(actual_errors) <= 0.0001
                receipt_valid = all(receipt_checks.values())
                ordered = ("large_convergence", "large_same_update",
                           "small_convergence", "small_same_compute")
                effects = [records[ordered[0]][seed]["score"] - records[ordered[1]][seed]["score"]
                           - records[ordered[2]][seed]["score"] + records[ordered[3]][seed]["score"]
                           for seed in sorted(OPTIMIZATION_SEEDS)] if receipt_valid else []
                per_unit: dict[str, list[float]] = {}
                if receipt_valid:
                    for seed in sorted(OPTIMIZATION_SEEDS):
                        for unit in records[ordered[0]][seed]["per_unit_accuracy"]:
                            per_unit.setdefault(unit, []).append(
                                records[ordered[0]][seed]["per_unit_accuracy"][unit]
                                - records[ordered[1]][seed]["per_unit_accuracy"][unit]
                                - records[ordered[2]][seed]["per_unit_accuracy"][unit]
                                + records[ordered[3]][seed]["per_unit_accuracy"][unit])
                units = [mean(values) for values in per_unit.values()]
                seed_summary, group_summary = summarize(effects), summarize(units)
                plateau = {tier: _plateau({int(k): v for k, v in doc.get("curve", {}).items()
                                           if int(k) in set(map(int, doc.get("regular_grid") or []))})
                           for tier, doc in docs.items()}
                converged = all(row["classification"] == "converged" for row in plateau.values())
                scientific_verdict = ("invalid_receipt" if not receipt_valid else
                                      "provisional_unconverged" if not converged else
                                      "positive_seed_only_group_floor_not_met"
                                      if seed_summary["verdict"] == "positive"
                                      and group_summary["lower_95_cb"] < io.SESOI
                                      else seed_summary["verdict"])
                match = small.get("compute_match") or {}
                large_records = list(records.get("large_same_update", {}).values())
                small_records = list(records.get("small_same_compute", {}).values())
                computed_large = {int(r["trainable_param_count"]) * int(r["updates"])
                                  for r in large_records}
                computed_small = {int(r["trainable_param_count"]) * int(r["updates"])
                                  for r in small_records}
                exposure_match = (len(computed_large) == len(computed_small) == 1
                                  and abs(next(iter(computed_large)) - next(iter(computed_small)))
                                  / next(iter(computed_large)) <= 0.0001)
                exposure_by_arm = {name: sum(float(row["parameter_update_exposure"])
                                             for row in records.get(name, {}).values()) for name in ordered}
                exposure_denominator = sum(exposure_by_arm.values())
                adjusted = (seed_summary["mean"] * 1_000_000_000 / exposure_denominator
                            if exposure_denominator else None)
                component_floor = ("passes" if receipt_valid and converged
                                   and seed_summary["verdict"] == "positive"
                                   and group_summary["lower_95_cb"] >= io.SESOI
                                   else "provisional_or_below_floor")
                roles = {**(large.get("four_contrast_roles") or {}),
                         **(small.get("four_contrast_roles") or {})}
                ok = (large.get("seeds") == small.get("seeds") == [0, 1, 2]
                      and large.get("spec", {}).get("family") == small.get("spec", {}).get("family") == "gru"
                      and large.get("spec", {}).get("tier") == "large"
                      and small.get("spec", {}).get("tier") == "small"
                      and set(roles) == {"large_model_at_same_update_count",
                                         "large_model_at_strict_selected_convergence",
                                         "small_model_at_same_compute",
                                         "small_model_at_strict_selected_convergence"}
                      and receipt_valid and exposure_match
                      and _close(match.get("relative_parameter_update_error"),
                                 abs(next(iter(computed_large)) - next(iter(computed_small)))
                                 / next(iter(computed_large)))
                      and all(plateau[tier]["classification"] == docs[tier].get("classification")
                              and plateau[tier]["selected_checkpoint"] == docs[tier].get(
                                  "selected_checkpoint") for tier in docs)
                      and expected.get("estimand") == "difference_in_differences"
                      and expected.get("formula_signs") == [1, -1, -1, 1]
                      and seed_summary["verdict"] == expected.get("raw_statistical_verdict")
                      and scientific_verdict == expected.get("verdict")
                      and _close(seed_summary["mean"], expected.get("mean"))
                      and _close(seed_summary["lower_95_cb"], expected.get("lower_95_cb"))
                      and _close(group_summary["mean"], expected.get("group_mean"))
                      and _close(group_summary["lower_95_cb"], expected.get("group_lower_95_cb"))
                      and _close(group_summary["upper_95_cb"], expected.get("group_upper_95_cb"))
                      and _close(group_summary["heterogeneity"], expected.get("group_heterogeneity"))
                      and exposure_by_arm == expected.get("parameter_update_exposure_by_arm")
                      and _close(exposure_denominator, expected.get(
                          "parameter_update_exposure_denominator"))
                      and _close(adjusted, expected.get(
                          "raw_cost_adjusted_effect_per_billion_parameter_updates"))
                      and ((converged and _close(adjusted, expected.get(
                          "cost_adjusted_effect_per_billion_parameter_updates")))
                           or (not converged and expected.get(
                               "cost_adjusted_effect_per_billion_parameter_updates") is None))
                      and expected.get("receipts_valid") is True
                      and all((expected.get("receipt_checks") or {}).values())
                      and component_floor == expected.get("component_floor_status"))
                recomputed[f"optimization:{bed}"] = {"seed": seed_summary, "group": group_summary,
                                                       "plateau": plateau,
                                                       "receipt_valid": receipt_valid,
                                                       "converged": converged,
                                                       "scientific_verdict": scientific_verdict}
            else:
                ok = expected.get("mean") is None and not docs
            checks[f"interaction:optimization_by_capacity:{bed}"] = ok
            if not ok:
                mismatches.append({"bed": bed, "contrast": "optimization_by_capacity",
                                   "sealed": expected, "recomputed": {
                                       "receipt_checks": receipt_checks if docs else {},
                                       "receipt_valid": receipt_valid if docs else False,
                                       "plateau": plateau if docs else {},
                                       "scientific_verdict": scientific_verdict if docs else None,
                                       "seed": seed_summary if docs else None,
                                       "group": group_summary if docs else None}})

    independent_keys = _independent_result_keys(sealed, recomputed)
    checks["derived:observed_result_keys"] = sorted(independent_keys) == sorted(
        sealed.get("observed_result_keys") or [])
    independent_fold = _independent_hypothesis_fold(independent_keys)
    sealed_fold = sealed.get("hypothesis_fold") or {}
    checks["derived:hypothesis_fold_identity"] = (
        independent_fold.get("observed_results") == sealed_fold.get("observed_results")
        and all({k: (sealed_fold.get("hypotheses") or {}).get(hypothesis, {}).get(k)
                 for k in ("state", "evidence")} == row
                for hypothesis, row in independent_fold.get("hypotheses", {}).items())
        and not independent_fold.get("unknown_result_keys"))
    recomputed["derived_result_keys"] = independent_keys

    correction = io.load("MOP_E2_CAPACITY_TIER_CORRECTION.json") if io.exists(
        "MOP_E2_CAPACITY_TIER_CORRECTION.json") else {}
    factorial = io.load("MOP_E2_FACTORIAL_AUTHORITY.json") if io.exists(
        "MOP_E2_FACTORIAL_AUTHORITY.json") else {}
    mutations = io.load("MOP_TEMPORAL_CORE_MUTATION_REPORT.json") if io.exists(
        "MOP_TEMPORAL_CORE_MUTATION_REPORT.json") else {}
    replication = io.load("MOP_E2_INDEPENDENT_REPLICATION.json") if io.exists(
        "MOP_E2_INDEPENDENT_REPLICATION.json") else {}
    current_verifier_agrees = all(checks.values()) and not mismatches
    current_implementations_agree = bool(replication.get("all_pass")) and all(
        value for key, value in checks.items() if "independent_replication" in key)
    verified_classifications = {}
    for identity, expected in (sealed.get("terminal_classification") or {}).items():
        bed, group, key = identity.split(":", 2)
        effect = recomputed.get(bed, {}).get(group, {}).get(key) or {}
        bed_valid = bool(((factorial.get("principal_beds") or {}).get(bed, {}).get("checks") or {}).get(
            "all_pass"))
        mine = _terminal_classification(
            effect, instrument_valid=bool(correction.get("all_pass")), bed_valid=bed_valid,
            verifier_agrees=False,
            mutations_rejected=bool(mutations.get("all_rejected")),
            implementations_agree=current_implementations_agree)
        checks[f"terminal:{identity}"] = mine == expected
        if mine != expected:
            mismatches.append({"bed": bed, "contrast": identity, "sealed": expected, "recomputed": mine})
        verified_classifications[identity] = _terminal_classification(
            effect, instrument_valid=bool(correction.get("all_pass")), bed_valid=bed_valid,
            verifier_agrees=current_verifier_agrees,
            mutations_rejected=bool(mutations.get("all_rejected")),
            implementations_agree=current_implementations_agree)

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
            unit_means = {unit: mean(values) for unit, values in unit_values.items()}
            mine = _summary_effect(effects, unit_means)
            sealed_effect = expected["shared_minus_domain_local"]
            inverse = verdict([-v for v in effects])
            classification = ("shared_component_supported" if seed_summary["verdict"] == "positive"
                              and group_summary["lower_95_cb"] >= io.SESOI and all(floors)
                              else "domain_local_component_supported" if inverse == "positive"
                              and inverse_group["lower_95_cb"] >= io.SESOI
                              else "shared_and_domain_local_inconclusive")
            e3_classes.append(classification)
            parameter_update_cost = int(round(mean([
                int(row["source_training"].get("trainable_param_count", 0))
                * int(row["source_training"].get("updates", 0))
                + 2 * int(row["target_training_match"].get("parameter_exposure_per_arm", 0))
                for row in rows])))
            adjusted = (mine["mean"] * 1_000_000 / parameter_update_cost
                        if parameter_update_cost else None)
            floor_status = {"name": "source_retention", "per_seed": floors,
                            "all_pass": all(floors), "margin": io.SESOI}
            ok = (len(rows) == len(expected_seeds) and _effect_matches(mine, sealed_effect)
                  and _close(inverse_group["lower_95_cb"],
                             sealed_effect.get("inverse_group_lower_95_cb"))
                  and (sealed_effect.get("cost_denominator") or {}).get(
                      "parameter_update_exposure") == parameter_update_cost
                  and _close(adjusted, sealed_effect.get(
                      "cost_adjusted_effect_per_million_parameter_updates"))
                  and floor_status == sealed_effect.get("component_floor_status")
                  and sealed_effect.get("bed_specific_effects") == {
                      target: {"mean": round(mean(effects), 5),
                               "per_seed_effects": [round(v, 5) for v in effects]}}
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
        future_mine = _summary_effect(gains, {u: mean(v) for u, v in unit_values.items()})
        return_mine = _summary_effect(returns, {u: mean(v) for u, v in {
            unit: [row["return_after_recovery"]["per_unit_accuracy"][unit]
                   - row["return_before_recovery"]["per_unit_accuracy"][unit]
                   for row in rows if unit in row["return_after_recovery"]["per_unit_accuracy"]
                   and unit in row["return_before_recovery"]["per_unit_accuracy"]]
            for unit in set().union(*(set(row["return_after_recovery"]["per_unit_accuracy"])
                                     & set(row["return_before_recovery"]["per_unit_accuracy"])
                                     for row in rows))}.items() if v})
        order_mine = _summary_effect(order_seed_effects, {u: mean(v) for u, v in order_values.items()})
        pooled_seed_effects = []
        for row in rows:
            witness = row["temporal_order_permutation"]
            ordered = witness.get("pooled_ordered_accuracy")
            permuted = witness.get("pooled_permuted_accuracy")
            if ordered is None or permuted is None:
                ordered = mean(list(witness["pooled_ordered_per_unit_accuracy"].values()))
                permuted = mean(list(witness["pooled_permuted_per_unit_accuracy"].values()))
            pooled_seed_effects.append(float(ordered) - float(permuted))
        pooled_mine = _summary_effect(pooled_seed_effects, {u: mean(v) for u, v in pooled_values.items()})
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
        future_floors = [row["after_B_adaptation"]["A"]["accuracy"]
                         >= row["before_adaptation"]["A"]["accuracy"] - io.SESOI for row in rows]
        return_floors = [row["return_after_recovery"]["accuracy"]
                         >= row["before_adaptation"]["A"]["accuracy"] - io.SESOI for row in rows]

        def third_contract(mine, sealed_effect, receipt_key, floor_status):
            exposure = int(round(mean([int(row["receipts"][receipt_key].get(
                "trainable_param_count", 0)) * int(row["receipts"][receipt_key].get("updates", 0))
                                       for row in rows])))
            adjusted = mine["mean"] * 1_000_000 / exposure if exposure else None
            return (_effect_matches(mine, sealed_effect)
                    and (sealed_effect.get("cost_denominator") or {}).get(
                        "parameter_update_exposure") == exposure
                    and _close(adjusted, sealed_effect.get(
                        "cost_adjusted_effect_per_million_parameter_updates"))
                    and sealed_effect.get("component_floor_status") == floor_status
                    and sealed_effect.get("bed_specific_effects") == {
                        "harth_stream": {"mean": mine["mean"],
                                         "per_seed_effects": mine["per_seed_effects"]}})

        checks["HARTH-preflight:future_full_effect_contract"] = third_contract(
            future_mine, expected["future_adaptation"], "adapt_B",
            {"name": "A_context_retention_during_B_adaptation", "per_seed": future_floors,
             "all_pass": all(future_floors), "margin": io.SESOI})
        checks["HARTH-preflight:return_full_effect_contract"] = third_contract(
            return_mine, expected["returning_context"], "recover_A_readout",
            {"name": "return_to_A_accuracy", "per_seed": return_floors,
             "all_pass": all(return_floors), "margin": io.SESOI})
        order_floor = {"name": "pooled_reader_order_invariance_control", "per_seed": [],
                       "all_pass": pooled_equivalent, "margin": io.EQUIVALENCE_MARGIN}
        checks["HARTH-preflight:order_full_effect_contract"] = third_contract(
            order_mine, expected["temporal_order_permutation"], "pretrain", order_floor)
        pooled_floor = {"name": "order_invariance_equivalence", "per_seed": [],
                        "all_pass": pooled_equivalent, "margin": io.EQUIVALENCE_MARGIN}
        checks["HARTH-preflight:pooled_full_effect_contract"] = third_contract(
            pooled_mine, expected["temporal_order_permutation"]["pooled_control_effect"],
            "pooled_control_pretrain", pooled_floor)
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
        preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") if io.exists(
            "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {}
        candidates = preflight.get("candidates") or {}
        harth_candidate = candidates.get("harth_stream") or {}
        load_bearing = [Fx.cell_name(**_expected_convergence_specs()[i]) for i in (0, 2, 3, 4, 8, 9, 10)]
        harth_configs = _convergence_audit("harth_stream")["configs"]
        load_bearing_converged = all(_plateau((harth_configs.get(cell) or {}).get("curve") or {})[
            "classification"] == "converged" for cell in load_bearing)
        split = B.splits("harth_stream", 0)
        main_units, tune_units, test_units = (set(split["units"][key])
                                             for key in ("main", "tune", "test"))
        units_valid = (not (main_units & tune_units or main_units & test_units or tune_units & test_units)
                       and len(main_units | tune_units | test_units) >= 4)
        scout_path = io.RUNS / "e2_scout" / "scout_harth_stream.json"
        scout_doc = json.loads(scout_path.read_text()) if scout_path.is_file() else {}
        scout_means = scout_doc.get("cell_means") or {}
        static_gap_measured = all(cell in scout_means for cell in (
            Fx.cell_name(**Fx.REFERENCE), Fx.cell_name(**dict(Fx.REFERENCE, family="pooled"))))
        instrumentation_complete = (bool(rows) and bool(harth_configs) and static_gap_measured
                                    and pooled_equivalent and bool(order_resource_checks)
                                    and all(order_resource_checks))
        boundary_valid = lower_bound(shifts) >= 0.02
        future_valid = future["verdict"] == "positive" and future_group["lower_95_cb"] >= io.SESOI
        return_valid = verdict(returns) == "positive"
        if not units_valid:
            expected_candidate_classification = "invalid_units"
        elif not instrumentation_complete:
            expected_candidate_classification = "preflight_incomplete"
        elif not temporal_order_required:
            expected_candidate_classification = "invalid_no_temporal_requirement"
        elif not load_bearing_converged:
            expected_candidate_classification = "invalid_instrumentation"
        elif not boundary_valid or not return_valid:
            expected_candidate_classification = "invalid_no_context_boundary"
        elif not future_valid:
            expected_candidate_classification = "invalid_no_headroom"
        else:
            expected_candidate_classification = "valid_secondary_bed"
        independently_admitted = (expected_candidate_classification == "valid_secondary_bed"
                                  and harth_candidate.get("identity") == B.identity("harth_stream"))
        independent_selected = ["harth_stream"] if independently_admitted else []
        checks["third_bed:exact_candidate_inventory"] = set(candidates) == {
            "harth_stream", "pamap2_stream"}
        checks["third_bed:candidate_classification_recomputed"] = harth_candidate.get(
            "classification") == expected_candidate_classification
        checks["third_bed:pamap2_unmeasured_classification"] = (candidates.get(
            "pamap2_stream") or {}).get("classification") == "invalid_instrumentation"
        checks["third_bed:exact_selected_set_recomputed"] = preflight.get("selected") == independent_selected
        third_result = io.load("MOP_THIRD_TEMPORAL_BED_RESULT.json") if io.exists(
            "MOP_THIRD_TEMPORAL_BED_RESULT.json") else {}
        replication = io.load("MOP_E2_INDEPENDENT_REPLICATION.json") if io.exists(
            "MOP_E2_INDEPENDENT_REPLICATION.json") else {}
        checks["third_bed:independent_admission_propagates"] = (
            third_result.get("admitted") is independently_admitted
            and replication.get("third_bed_admitted") is independently_admitted)
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
            gain_mine = _summary_effect(gain, {u: mean(v) for u, v in gain_units.items()})
            noise_mine = _summary_effect(noise, {u: mean(v) for u, v in noise_units.items()})
            floors = [r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]
                      >= r["before_adaptation"]["A"]["accuracy"] - io.SESOI for r in rows]
            floor_status = {"name": "return_retention_A", "per_seed": floors,
                            "all_pass": all(floors), "margin": io.SESOI}

            def successor_contract(expected_effect, mine, left, right):
                exposure = int(round(mean([
                    int(r["arms"][left]["trace"].get("parameter_update_exposure", 0))
                    + int(r["arms"][right]["trace"].get("parameter_update_exposure", 0)) for r in rows])))
                adjusted = mine["mean"] * 1_000_000 / exposure if exposure else None
                return (_effect_matches(mine, expected_effect)
                        and (expected_effect.get("cost_denominator") or {}).get(
                            "parameter_update_exposure") == exposure
                        and _close(adjusted, expected_effect.get(
                            "cost_adjusted_effect_per_million_parameter_updates"))
                        and expected_effect.get("component_floor_status") == floor_status
                        and expected_effect.get("bed_specific_effects") == {
                            bedname: {"mean": mine["mean"],
                                      "per_seed_effects": mine["per_seed_effects"]}})

            checks[f"hybrid:{bedname}:hybrid_vs_head"] = len(rows) == len(expected_seeds) \
                and successor_contract(expected["hybrid_minus_head"], gain_mine,
                                       "head_plus_state", "head_only")
            checks[f"hybrid:{bedname}:hybrid_vs_noise"] = len(rows) == len(expected_seeds) \
                and successor_contract(expected["hybrid_minus_head_noise"], noise_mine,
                                       "head_plus_state", "head_plus_state_noise")
            checks[f"hybrid:{bedname}:metrics_reconstructed"] = bool(metric_checks) and all(metric_checks)
            checks[f"hybrid:{bedname}:split_identity"] = bool(split_checks) and all(split_checks)
            shifts = [r["before_adaptation"]["A"]["accuracy"]
                      - r["before_adaptation"]["B"]["accuracy"] for r in rows]
            costs = [r["arms"]["head_plus_state"]["return_retention_A"]["accuracy"]
                     - r["before_adaptation"]["A"]["accuracy"] for r in rows]
            acquisitions = [r["arms"]["head_plus_state"]["future_acquisition_B"]["accuracy"]
                            - r["before_adaptation"]["B"]["accuracy"] for r in rows]
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
    if io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json"):
        queue = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json")
        gates = queue.get("gates") or {}
        checks["successor_queue:exact_candidate_inventory"] = {
            row.get("candidate_id") for row in gates.values()} == {"E3", "E5", "E6", "E7", "E8"} \
            and len(gates) == 5
        fold = independent_fold.get("hypotheses") or {}
        core_positive = bool(owned.get("selected")) and (fold.get("H1_recurrence") or {}).get(
            "state") == "supported"
        capacity = sealed.get("capacity_retention_or_interference") or {}
        capacity_values = capacity.get("per_independent_unit_effects") or []
        capacity_valid = capacity.get("measured") is True and capacity.get("estimand") in (
            "retention_loss_large_minus_small", "interference_cost_large_minus_small") \
            and len(capacity_values) >= 2
        capacity_lower = lower_bound([float(v) for v in capacity_values]) if capacity_valid else None
        capacity_opens = capacity_lower is not None and capacity_lower >= io.SESOI
        architecture_keys = set(recomputed.get("har_stream", {}).get("architecture", {})) \
            & set(recomputed.get("speech_stream", {}).get("architecture", {}))
        heterogeneity_bounds = []
        for key in architecture_keys:
            left = recomputed["har_stream"]["architecture"][key] or {}
            right = recomputed["speech_stream"]["architecture"][key] or {}
            ready = all(row.get("estimator_sufficient") is True
                        and (row.get("convergence") or {}).get("all_converged") is True
                        for row in (left, right))
            difference = _welch(list((left.get("per_unit_effects") or {}).values()),
                                list((right.get("per_unit_effects") or {}).values()))
            if difference and ready:
                heterogeneity_bounds.append(max(difference["lower_95_cb"],
                                                -difference["upper_95_cb"], 0.0))
        heterogeneity = max(heterogeneity_bounds, default=0.0)
        bed_heterogeneity = heterogeneity >= io.SESOI
        expected_opens = {
            "E3_shared_versus_local": core_positive and (capacity_opens or bed_heterogeneity),
            "E5_self_supervised": False,
            "hybrid_adaptation": bool(owned.get("selected")),
            "third_bed_replication": independently_admitted and independently_third_replicated,
            "minimal_core_cross_domain_transfer": False,
        }
        checks["successor_queue:gate_openings_recomputed"] = set(gates) == set(expected_opens) and all(
            gates[name].get("opens") is value for name, value in expected_opens.items())
        owned_params = int(((owned.get("selection") or {}).get("selected") or {}).get("params_total")
                           or (owned.get("core") or {}).get("owned_parameters") or 0)
        principal_beds = tuple(sealed.get("principal_beds") or ())
        core_lcbs = [recomputed.get(bed, {}).get("recurrent_versus_matched_history", {}).get(
            "gru_vs_histmlp_kfull_window", {}).get("group_lower_95_cb") for bed in principal_beds]
        hybrid_voi = max(0.0, min(map(float, core_lcbs))) if len(core_lcbs) == 2 and all(
            isinstance(v, (int, float)) for v in core_lcbs) else None
        third_lcbs = [recomputed.get(f"replication:harth_stream:{key}", {}).get("group_lower_95_cb")
                      for key in ("torch_gru_vs_full_history", "explicit_mgu_vs_full_history")]
        third_voi = min(map(float, third_lcbs)) if all(isinstance(v, (int, float))
                                                      for v in third_lcbs) else 0.0
        e3_voi = max(heterogeneity if bed_heterogeneity else 0.0,
                     float(capacity_lower) if capacity_opens else 0.0)
        costs = {
            "E3_shared_versus_local": 2 * len(expected_seeds) * 2 * Fx.STEPS * owned_params,
            "E5_self_supervised": 2 * len(expected_seeds) * 2 * Fx.STEPS * owned_params,
            "hybrid_adaptation": len(principal_beds or (1,)) * len(expected_seeds) * 3
            * (Fx.STEPS // 4) * owned_params,
            "third_bed_replication": len(expected_seeds) * Fx.STEPS * owned_params,
            "minimal_core_cross_domain_transfer": 2 * len(expected_seeds) * 2 * Fx.STEPS * owned_params,
        }
        vois = {"E3_shared_versus_local": e3_voi, "E5_self_supervised": 0.0,
                "hybrid_adaptation": hybrid_voi, "third_bed_replication": third_voi,
                "minimal_core_cross_domain_transfer": 0.0}
        complete = {
            "E3_shared_versus_local": owned_params > 0 and math.isfinite(e3_voi) and e3_voi >= 0,
            "E5_self_supervised": False,
            "hybrid_adaptation": owned_params > 0 and hybrid_voi is not None and hybrid_voi >= 0,
            "third_bed_replication": owned_params > 0 and len(third_lcbs) == 2
            and all(isinstance(v, (int, float)) for v in third_lcbs) and third_voi >= 0,
            "minimal_core_cross_domain_transfer": False,
        }
        for name in costs:
            rank = gates.get(name, {}).get("ranking") or {}
            priority = float(vois[name]) / costs[name] if complete[name] else None
            checks[f"successor_queue:{name}:numeric_ranking"] = (
                _close(rank.get("value_of_information"), vois[name])
                and rank.get("estimated_parameter_update_cost") == costs[name]
                and rank.get("complete") is complete[name]
                and ((priority is None and rank.get("priority_score") is None)
                     or _close(rank.get("priority_score"), priority, tol=1e-12)))
        legacy = {"third_bed_replication": 0, "E3_shared_versus_local": 1,
                  "hybrid_adaptation": 2, "E5_self_supervised": 3,
                  "minimal_core_cross_domain_transfer": 4}
        ranked = sorted((name for name, opened in expected_opens.items() if opened and complete[name]),
                        key=lambda name: (-float(vois[name]) / costs[name], legacy[name], name))
        checks["successor_queue:ranked_and_licensed_recomputed"] = (
            queue.get("opened") == [name for name, opened in expected_opens.items() if opened]
            and queue.get("ranked_opened") == ranked and queue.get("licensed_top_two") == ranked[:2])
    return {"role": "C scientific verifier", "checks": checks, "mismatches": mismatches,
            "preclassification_checks_pass": current_verifier_agrees,
            "verified_terminal_classification": verified_classifications,
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
