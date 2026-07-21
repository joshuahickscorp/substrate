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
    curve, spread, seed_scores = {}, {}, {}
    for steps in e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID:
        values = [Fx.run_cell(B.splits(bedname, seed), LINEAR, seed, "tune", steps=steps)["accuracy"]
                  for seed in e2.CONVERGENCE_SEEDS]
        curve[steps] = float(np.mean(values))
        spread[steps] = round(float(np.std(values, ddof=1)), 5)
        seed_scores[steps] = [round(float(v), 5) for v in values]
    witness = W.plateau_validity(curve)
    probe = Fx.run_cell(B.splits(bedname, 0), LINEAR, 0, "tune", steps=1)
    lo, hi = A.TIER_RANGE["large"]
    doc = {"schema": "mop-e2-capacity-tier-correction-convergence/v1", "bed": bedname,
           "spec": LINEAR, "cell": Fx.cell_name(**LINEAR), "curve": curve, "seed_spread": spread,
           "seed_scores": seed_scores, "seeds": list(e2.CONVERGENCE_SEEDS),
           "parameter_count": probe["params"], "parameter_band_valid": lo <= probe["params"]["core"] <= hi,
           "supersedes": [f"e2_converge/cshard_{bedname}_25.json",
                          f"e2_converge_extended/xshard_{bedname}_25.json"],
           **witness, "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"convergence_{bedname}.json", doc, "e2_converge_corrections")
    print(f"capacity convergence correction {bedname}: {doc['classification']}", flush=True)
    return doc


def aggregate() -> dict:
    principal = [json.loads((io.RUNS / "e2_principal_corrections" /
                 f"capacity_{bed}_{seed}.json").read_text()) for bed in BEDS for seed in e2.PRINCIPAL_SEEDS]
    convergence = [json.loads((io.RUNS / "e2_converge_corrections" /
                   f"convergence_{bed}.json").read_text()) for bed in BEDS]
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
              "original_receipts_quarantined_not_deleted": all(v == 16 for v in original_invalid.values())}
    refreshed = {bed: e2.converge(bed) for bed in BEDS}
    checks["convergence_aggregates_refreshed_with_corrections"] = all(
        (d.get("configs") or {}).get(Fx.cell_name(**LINEAR), {}).get("parameter_band_valid") for d in refreshed.values())
    doc = {"schema": "mop-e2-capacity-tier-correction/v1", "affected_cells": [
           Fx.cell_name(**s) for s in SPECS], "original_invalid_receipts": original_invalid,
           "principal_shards": [{"bed": d["bed"], "seed": d["seed"]} for d in principal],
           "convergence": {d["bed"]: {"classification": d["classification"],
                                      "parameter_count": d["parameter_count"]} for d in convergence},
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
    while True:
        pending_p = supervisor.missing("e2_principal_corrections", principal)
        pending_c = supervisor.missing("e2_converge_corrections", convergence)
        if not pending_p and not pending_c:
            return aggregate()
        free = max(0, supervisor.CAP_LARGE - supervisor.workers())
        for name in pending_p + pending_c:
            if free <= 0:
                break
            if name.startswith("capacity_"):
                bed_seed = name.removeprefix("capacity_")
                bedname, seed = bed_seed.rsplit("_", 1)
                args = ["principal", bedname, seed]
            else:
                args = ["convergence", name.removeprefix("convergence_")]
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
