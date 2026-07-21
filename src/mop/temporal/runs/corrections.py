"""Append-only correction for large HistMLP receipts that missed their sealed parameter band."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal import witness as W
from mop.temporal.runs import e2

BEDS = ("har_stream", "speech_stream", "harth_stream")
LINEAR = dict(Fx.REFERENCE, family="histmlp", tier="large")
STRONG = dict(Fx.REFERENCE, family="histmlp", tier="large", readout="mlp_strong")
SPECS = (LINEAR, STRONG)
OPTIMIZATION_SPECS = {"small": dict(Fx.REFERENCE), "large": dict(Fx.REFERENCE, tier="large")}


def measure_curve(bedname: str, spec: dict, budgets: tuple[int, ...]) -> dict:
    curve, spread, seed_scores, unit_scores, wall, records = {}, {}, {}, {}, {}, {}
    params = None
    for steps in budgets:
        rows = [Fx.run_cell(B.splits(bedname, seed), spec, seed, "tune", steps=steps)
                for seed in e2.CONVERGENCE_SEEDS]
        values = [r["accuracy"] for r in rows]
        curve[steps] = float(np.mean(values))
        spread[steps] = round(float(np.std(values, ddof=1)), 5)
        seed_scores[steps] = [round(float(v), 5) for v in values]
        unit_scores[steps] = {str(seed): row["per_unit_accuracy"]
                              for seed, row in zip(e2.CONVERGENCE_SEEDS, rows, strict=True)}
        wall[steps] = [float(row.get("wall_seconds", 0)) for row in rows]
        params = rows[0]["params"]
        records[steps] = [{"seed": seed, "updates": row["updates"],
                           "trainable_param_count": row["params"]["total"],
                           "parameter_update_exposure": row["params"]["total"] * row["updates"],
                           "score": row["accuracy"], "per_unit_accuracy": row["per_unit_accuracy"],
                           "checkpoint_sha": row["checkpoint_sha_after"],
                           "wall_seconds": float(row.get("wall_seconds", 0))}
                          for seed, row in zip(e2.CONVERGENCE_SEEDS, rows, strict=True)]
    return {"curve": curve, "seed_spread": spread, "seed_scores": seed_scores,
            "per_unit_seed_scores": unit_scores, "wall_seconds_per_seed": wall,
            "parameter_count": params, "arm_records": records}


def principal_shard(bedname: str, seed: int) -> dict:
    t0 = time.time()
    sp = B.splits(bedname, seed)
    runs = [Fx.run_cell(sp, spec, seed, "test") for spec in SPECS]
    lo, hi = A.TIER_RANGE["large"]
    checks = {"two_affected_cells": len(runs) == 2,
              "large_core_band": all(lo <= r["params"]["core"] <= hi for r in runs),
              "exact_budget": all(r["steps"] == r["updates"] == Fx.STEPS for r in runs),
              "no_undeclared_changes": all(not r["undeclared_changes"] for r in runs)}
    doc = {"schema": "mop-e2-capacity-tier-correction-shard/v1", "bed": bedname, "seed": seed,
           "supersedes_for_analysis": [Fx.cell_name(**s) for s in SPECS], "runs": runs,
           "checks": checks, "all_checks_pass": all(checks.values()),
           "reason": "the original width search ceiling left the large HistMLP core below 600000 parameters",
           "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"capacity_{bedname}_{seed}.json", doc, "e2_principal_corrections")
    print(f"capacity correction {bedname} seed {seed}: {checks['large_core_band']}", flush=True)
    return doc


def convergence_shard(bedname: str) -> dict:
    t0 = time.time()
    measured = measure_curve(bedname, LINEAR, e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID)
    curve = measured["curve"]
    witness = W.plateau_validity(curve)
    lo, hi = A.TIER_RANGE["large"]
    doc = {"schema": "mop-e2-capacity-tier-correction-convergence/v2", "bed": bedname,
           "spec": LINEAR, "cell": Fx.cell_name(**LINEAR), "curve": curve,
           "seed_spread": measured["seed_spread"],
           "seed_scores": measured["seed_scores"],
           "per_unit_seed_scores": measured["per_unit_seed_scores"],
           "wall_seconds_per_seed": measured["wall_seconds_per_seed"],
           "seeds": list(e2.CONVERGENCE_SEEDS),
           "parameter_count": measured["parameter_count"],
           "parameter_band_valid": lo <= measured["parameter_count"]["core"] <= hi,
           "supersedes": [f"e2_converge/cshard_{bedname}_25.json",
                          f"e2_converge_extended/xshard_{bedname}_25.json"],
           **witness, "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"convergence_{bedname}.json", doc, "e2_converge_corrections")
    print(f"capacity convergence correction {bedname}: {doc['classification']}", flush=True)
    return doc


def optimization_shard(bedname: str, tier: str) -> dict:
    """Per seed GRU capacity curves plus the exact small-model compute match."""
    t0, spec = time.time(), OPTIMIZATION_SPECS[tier]
    sp = B.splits(bedname, 0)
    counts = {name: A.count(Fx.build_cell(sp, seed=0, **candidate)[0])["total"]
              for name, candidate in OPTIMIZATION_SPECS.items()}
    large_steps = min(e2.CONVERGENCE_GRID)
    small_steps = max(1, round(large_steps * counts["large"] / counts["small"]))
    budgets = e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID
    measured_budgets = tuple(sorted(set(budgets + ((small_steps,) if tier == "small" else ()))))
    measured = measure_curve(bedname, spec, measured_budgets)
    regular_curve = {k: measured["curve"][k] for k in budgets}
    witness = W.plateau_validity(regular_curve)
    large_compute, small_compute = large_steps * counts["large"], small_steps * counts["small"]
    selected = int(witness["selected_checkpoint"])
    role = ("large_model_at_same_update_count" if tier == "large" else
            "small_model_at_same_compute")
    design_checks = {"same_gru_family_differs_only_by_capacity_tier": all(
                  OPTIMIZATION_SPECS["small"][k] == OPTIMIZATION_SPECS["large"][k]
                  for k in OPTIMIZATION_SPECS["small"] if k != "tier"),
              "same_update_anchor_is_preregistered": large_steps in e2.CONVERGENCE_GRID,
              "compute_match_within_tolerance": abs(large_compute - small_compute) / large_compute <= 0.0001,
              "small_compute_match_within_feasible_budget": small_steps <= max(e2.EXTENDED_CONVERGENCE_GRID),
              "strict_selection_from_regular_curve": selected in regular_curve,
              "all_seed_arm_records_complete": all(len(measured["arm_records"][budget]) ==
                                                     len(e2.CONVERGENCE_SEEDS)
                                                     for budget in measured_budgets)}
    doc = {"schema": "mop-e2-optimization-capacity-correction/v1", "bed": bedname,
           "tier": tier, "spec": spec, "cell": Fx.cell_name(**spec), "seeds": list(e2.CONVERGENCE_SEEDS),
           **measured, "regular_grid": list(budgets), "same_update_anchor": large_steps,
           "compute_match": {"large_steps": large_steps, "large_parameter_updates": large_compute,
                             "small_steps": small_steps, "small_parameter_updates": small_compute,
                             "relative_parameter_update_error": round(
                                 abs(large_compute - small_compute) / large_compute, 8)},
           "four_contrast_roles": {
               role: {"budget": large_steps if tier == "large" else small_steps,
                      "records": measured["arm_records"][large_steps if tier == "large" else small_steps]},
               f"{tier}_model_at_strict_selected_convergence": {
                   "budget": selected, "records": measured["arm_records"][selected],
                   "classification": witness["classification"]}},
           "design_checks": design_checks, "all_checks_pass": all(design_checks.values()),
           **witness, "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"optimization_{bedname}_{tier}.json", doc, "e2_optimization_corrections")
    print(f"optimization correction {bedname} {tier}: {doc['classification']}", flush=True)
    return doc


def aggregate() -> dict:
    principal = [json.loads((io.RUNS / "e2_principal_corrections" /
                 f"capacity_{bed}_{seed}.json").read_text()) for bed in BEDS for seed in e2.PRINCIPAL_SEEDS]
    convergence = [json.loads((io.RUNS / "e2_converge_corrections" /
                   f"convergence_{bed}.json").read_text()) for bed in BEDS]
    optimization = [json.loads((io.RUNS / "e2_optimization_corrections" /
                    f"optimization_{bed}_{tier}.json").read_text())
                    for bed in BEDS for tier in OPTIMIZATION_SPECS]
    original_invalid = {}
    lo, hi = A.TIER_RANGE["large"]
    for bed in BEDS:
        count = 0
        for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
            count += sum(r["cell"] in {Fx.cell_name(**s) for s in SPECS}
                         and not lo <= r["params"]["core"] <= hi for r in json.loads(p.read_text())["runs"])
        original_invalid[bed] = count
    checks = {"all_24_principal_corrections": len(principal) == 24,
              "all_correction_receipts_valid": all(d["all_checks_pass"] for d in principal),
              "all_three_convergence_corrections": len(convergence) == 3,
              "all_corrected_parameter_bands_valid": all(d["parameter_band_valid"] for d in convergence),
              "all_six_optimization_corrections": len(optimization) == 6,
              "all_optimization_receipts_valid": all(d["all_checks_pass"] for d in optimization),
              "optimization_same_compute_measured": all(
                  d["compute_match"]["relative_parameter_update_error"] <= 0.0001 for d in optimization),
              "original_receipts_quarantined_not_deleted": all(v == 16 for v in original_invalid.values())}
    refreshed = {bed: e2.converge(bed) for bed in BEDS}
    checks["convergence_aggregates_refreshed_with_corrections"] = all(
        (d.get("configs") or {}).get(Fx.cell_name(**LINEAR), {}).get("parameter_band_valid") for d in refreshed.values())
    doc = {"schema": "mop-e2-capacity-tier-correction/v1", "affected_cells": [
           Fx.cell_name(**s) for s in SPECS], "original_invalid_receipts": original_invalid,
           "principal_shards": [{"bed": d["bed"], "seed": d["seed"]} for d in principal],
           "convergence": {d["bed"]: {"classification": d["classification"],
                                      "parameter_count": d["parameter_count"]} for d in convergence},
           "optimization": {f"{d['bed']}:{d['tier']}": {
               "classification": d["classification"], "compute_match": d["compute_match"],
               "parameter_count": d["parameter_count"]} for d in optimization},
           "refreshed_convergence_aggregates": {bed: {"all_converged": d["all_converged"],
                                                       "load_bearing_all_converged": d["load_bearing_all_converged"]}
                                                for bed, d in refreshed.items()},
           "checks": checks, "all_pass": all(checks.values()),
           "rule": "analysis replaces only the same bed, seed and cell; original receipts remain immutable"}
    io.seal("MOP_E2_CAPACITY_TIER_CORRECTION.json", doc)
    print(f"capacity correction aggregate: {doc['all_pass']}", flush=True)
    return doc


def run_all() -> dict:
    from mop.temporal.runs import supervisor

    principal = [f"capacity_{bed}_{seed}" for bed in BEDS for seed in e2.PRINCIPAL_SEEDS]
    convergence = [f"convergence_{bed}" for bed in BEDS]
    optimization = [f"optimization_{bed}_{tier}" for bed in BEDS for tier in OPTIMIZATION_SPECS]
    while True:
        pending_p = supervisor.recoverable_pending("e2_principal_corrections", principal)
        pending_c = supervisor.recoverable_pending("e2_converge_corrections", convergence)
        pending_o = supervisor.recoverable_pending("e2_optimization_corrections", optimization)
        if not pending_p and not pending_c and not pending_o:
            return aggregate()
        active = sum(supervisor.lock_active(p.stem.replace("_", ":", 1))
                     for p in supervisor.LOCKS.glob("*.json"))
        free = max(0, supervisor.CAP_LARGE - active)
        for name in pending_p + pending_c + pending_o:
            if free <= 0:
                break
            if name.startswith("capacity_"):
                bed_seed = name.removeprefix("capacity_")
                bedname, seed = bed_seed.rsplit("_", 1)
                args = ["principal", bedname, seed]
            elif name.startswith("convergence_"):
                args = ["convergence", name.removeprefix("convergence_")]
            else:
                bedname, tier = name.removeprefix("optimization_").rsplit("_", 1)
                args = ["optimization", bedname, tier]
            if supervisor.launch(args, "capacity_corrections.log", f"fix:{name}",
                                 module="mop.temporal.runs.corrections"):
                free -= 1
        time.sleep(5)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] == "all":
        run_all()
    elif argv[0] == "principal":
        principal_shard(argv[1], int(argv[2]))
    elif argv[0] == "convergence":
        convergence_shard(argv[1])
    elif argv[0] == "optimization":
        optimization_shard(argv[1], argv[2])
    elif argv[0] == "aggregate":
        aggregate()
    else:
        raise ValueError(argv)
    lock = os.environ.get("TEMPORAL_SHARD_LOCK")
    if lock:
        from pathlib import Path
        Path(lock).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
