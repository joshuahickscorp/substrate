"""E2: calibration, scout, principal.
    calibration   synthetic worlds with a known generative truth. The analysis must name each one
    scout         four seeds on tuning units, to estimate variance and drop dominated cells
    principal     eight seeds, untouched group disjoint test split
Usage
    python -m mop.temporal.runs.e2 calibration
    python -m mop.temporal.runs.e2 scout <bed>
    python -m mop.temporal.runs.e2 principal <bed> <seed>
    python -m mop.temporal.runs.e2 converge <bed>
House style: no dashes.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import numpy as np
from mop.method import power
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
SCOUT_SEEDS = (0, 1, 2, 3)
PRINCIPAL_SEEDS = tuple(range(8))
MAX_SEEDS = 12
CONVERGENCE_GRID = (400, 800, 1600, 3200)
EXTENDED_CONVERGENCE_GRID = (6400, 12800)
CONVERGENCE_SEEDS = (0, 1, 2)
PREREG = power.preregistration(
    name="E2", independent_unit="natural unit of the bed", expected_sd=0.035, sesoi=io.SESOI,
    seeds=len(PRINCIPAL_SEEDS), units=9, max_seeds=MAX_SEEDS, futility=0.01, harm=0.05,
    continuation_rule=("the seed count is fixed at eight before the first principal cell, and a preregistered "
                       "maximum of twelve applies only to cells whose scout standard deviation exceeds 0.06"))
# ---------------------------------------------------------------- calibration
def _world(kind: str, seeds: int = 8, sd: float = 0.02, seed: int = 0) -> dict:
    """Synthetic cell scores whose generative truth is known. No training, this calibrates the analysis."""
    rng = np.random.default_rng(seed)
    base = 0.55
    def draw(mu):
        return [float(mu + rng.normal(0, sd)) for _ in range(seeds)]
    cells = {}
    fams = A.FAMILIES
    for f in fams:
        for t in A.CAPACITY_TIERS:
            for r in A.READOUTS:
                mu = base
                if kind == "pure_capacity":
                    mu += {"micro": -0.10, "small": 0.0, "medium": 0.10, "large": 0.18}[t]
                elif kind == "pure_readout":
                    mu += {"linear": 0.0, "mlp1": 0.09, "mlp_strong": 0.16}[r]
                elif kind in ("pure_recurrence", "core_horizon_interaction"):
                    mu += 0.20 if f in A.RECURRENT else 0.0
                elif kind == "explicit_history_sufficiency":
                    mu += 0.20 if f in A.RECURRENT or f == "histmlp" else 0.0
                elif kind == "optimization_artifact":
                    mu += 0.20 if f in A.RECURRENT else 0.0
                cells[AN.name(family=f, tier=t, readout=r)] = draw(mu)
    for f in ("gru", "mgu"):
        for h in (1, 2, 5, 10, 20, 45, 90, "full"):
            mu = base + (0.20 if kind in ("pure_recurrence", "optimization_artifact",
                                          "explicit_history_sufficiency", "core_horizon_interaction") else 0.0)
            if kind in ("pure_horizon", "core_horizon_interaction"):
                hv = 10**9 if h == "full" else int(h)
                mu = base + (0.22 if hv >= 45 else 0.0)
            cells[AN.name(family=f, reset=f"horizon_{h}")] = draw(mu)
    if kind == "core_horizon_interaction":
        for t in A.CAPACITY_TIERS:
            for h in (5, 45, "full"):
                hv = 10**9 if h == "full" else int(h)
                bonus = {"micro": 0.0, "small": 0.05, "medium": 0.12, "large": 0.20}[t] if hv >= 45 else 0.0
                cells[AN.name(tier=t, reset=f"horizon_{h}")] = draw(base + bonus)
    for k in (1, 2, 5, 10, 20, "full_window", "pooled_summary"):
        mu = base
        if kind == "explicit_history_sufficiency":
            kv = 10**9 if isinstance(k, str) else k
            mu += 0.20 if kv >= 20 else 0.0
        cells[AN.name(family="histmlp", history_k=k)] = draw(mu)
    for r in ("every_observation", "misaligned_a", "misaligned_b", "random_rate_matched",
              "block_shuffled", "true_boundary", "wrong_boundary"):
        mu = base + (0.20 if kind in ("pure_recurrence", "optimization_artifact",
                                      "explicit_history_sufficiency") else 0.0)
        if kind == "reset_alignment_artifact":
            mu = base + (0.25 if r == "true_boundary" else 0.0)
        cells[AN.name(reset=r)] = draw(mu)
    if kind == "no_effect":
        cells = {k: draw(base) for k in cells}
    return cells
CALIBRATION_WORLDS = {
    "pure_capacity": {"capacity"},
    "pure_readout": {"readout"},
    "pure_horizon": {"horizon"},
    "pure_recurrence": {"recurrence"},
    "explicit_history_sufficiency": {"explicit_history_sufficiency"},
    "core_horizon_interaction": {"core_horizon_interaction"},
    "optimization_artifact": {"recurrence"},
    "reset_alignment_artifact": {"reset_alignment_artifact"},
    "no_effect": {"no_effect"},
}
def calibration() -> dict:
    res = {}
    for world, expected in CALIBRATION_WORLDS.items():
        cells = _world(world)
        r = AN.recover(cells, PREREG)
        got = set(r["recovered"])
        res[world] = {
            "expected_contains": sorted(expected),
            "recovered": sorted(got),
            "pass": expected <= got,
            "findings": {k: v for k, v in r["findings"].items() if v},
        }
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    return res
# ---------------------------------------------------------------- scout and principal
def _series(runs: list[dict]) -> tuple[dict, dict]:
    cells: dict[str, list] = {}
    units: dict[str, dict] = {}
    for r in runs:
        cells.setdefault(r["cell"], []).append(r["accuracy"])
        acc = units.setdefault(r["cell"], {})
        for u, a in r["per_unit_accuracy"].items():
            acc.setdefault(u, []).append(a)
    return cells, {c: {u: float(np.mean(v)) for u, v in d.items()} for c, d in units.items()}
def scout_shard(bedname: str, seed: int) -> dict:
    """One bed, one seed, every cell. Sharding by seed is what lets the scout use the whole host."""
    t0 = time.time()
    sp = B.splits(bedname, seed)
    runs = [Fx.run_cell(sp, spec, seed, "tune", steps=Fx.STEPS // 2) for spec in Fx.sweep_cells()["_all"]]
    doc = {"bed": bedname, "seed": seed, "runs": runs, "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"shard_{bedname}_{seed}.json", doc, "e2_scout")
    print(f"E2 scout shard {bedname} seed {seed}: {len(runs)} cells in {doc['wall_seconds']}s", flush=True)
    return doc
def scout(bedname: str) -> dict:
    t0 = time.time()
    cells = Fx.sweep_cells()["_all"]
    runs = []
    for seed in SCOUT_SEEDS:
        p = io.RUNS / "e2_scout" / f"shard_{bedname}_{seed}.json"
        if p.is_file():
            runs.extend(json.loads(p.read_text())["runs"])
        else:
            runs.extend(scout_shard(bedname, seed)["runs"])
    series, units = _series(runs)
    sds = {c: float(np.std(v, ddof=1)) for c, v in series.items() if len(v) > 1}
    means = {c: float(np.mean(v)) for c, v in series.items()}
    best = max(means, key=means.get)
    dominated = [c for c, m in means.items() if m < means[best] - 0.35 and "horizon" not in c
                 and "boundary" not in c and "every_observation" not in c]
    doc = {
        "schema": "mop-e2-scout/v1",
        "bed": bedname,
        "seeds": list(SCOUT_SEEDS),
        "steps": Fx.STEPS // 2,
        "evaluated_on": "tuning units, the test split is untouched",
        "n_cells": len(cells),
        "cell_means": {k: round(v, 5) for k, v in means.items()},
        "cell_sds": {k: round(v, 5) for k, v in sds.items()},
        "best_cell": best,
        "high_variance_cells": sorted(c for c, s in sds.items() if s > 0.06),
        "clearly_dominated_cells": sorted(dominated),
        "median_sd": round(float(np.median(list(sds.values()))), 5),
        "instrumentation": {
            "undeclared_parameter_changes": sum(len(r["undeclared_changes"]) for r in runs),
            "reset_witnesses": {r["cell"]: r["reset_witness"]["classification"] for r in runs},
            "history_profiles": {r["cell"]: r["history_profile"] for r in runs},
        },
        "runs": runs,
        "not_scientific_evidence": True,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"scout_{bedname}.json", doc, "e2_scout")
    print(f"E2 scout {bedname}: {len(cells)} cells, best {best} {means[best]:.4f}, "
          f"median sd {doc['median_sd']}, dominated {len(dominated)}", flush=True)
    return doc
def principal(bedname: str, seed: int) -> dict:
    sp = B.splits(bedname, seed)
    cells = Fx.sweep_cells()["_all"]
    runs = [Fx.run_cell(sp, spec, seed, "test") for spec in cells]
    doc = {"bed": bedname, "seed": seed, "n_cells": len(runs), "runs": runs,
           "majority_rate": B.majority_rate(sp["test"][1]), "chance_rate": B.chance_rate(sp["classes"])}
    io.run_json(f"{bedname}_{seed}.json", doc, "e2_principal")
    print(f"E2 principal {bedname} seed {seed}: {len(runs)} cells, "
          f"best {max(r['accuracy'] for r in runs):.4f}", flush=True)
    return doc
CONVERGE_CONFIGS = [
    dict(Fx.REFERENCE, family=f) for f in ("gru", "lstm", "mgu", "pooled", "tcn")
] + [dict(Fx.REFERENCE, family="histmlp", history_k=20),
     dict(Fx.REFERENCE, tier="large"),
     dict(Fx.REFERENCE, family="pooled", tier="large", readout="mlp_strong"),
     dict(Fx.REFERENCE, family="histmlp", history_k="full_window"),
     dict(Fx.REFERENCE, reset="horizon_45"),
     dict(Fx.REFERENCE, reset="horizon_90")]
for family in ("gru", "lstm", "mgu", "pooled", "histmlp", "tcn"):
    for tier in A.CAPACITY_TIERS:
        spec = dict(Fx.REFERENCE, family=family, tier=tier)
        if Fx.cell_name(**spec) not in {Fx.cell_name(**c) for c in CONVERGE_CONFIGS}:
            CONVERGE_CONFIGS.append(spec)
for group in ("architecture", "readout", "horizon", "reset", "capacity_by_horizon", "history", "capacity_by_readout"):
    for spec in Fx.sweep_cells()[group]:
        if Fx.cell_name(**spec) not in {Fx.cell_name(**c) for c in CONVERGE_CONFIGS}:
            CONVERGE_CONFIGS.append(spec)
LOAD_BEARING_CONVERGENCE_CELLS = tuple(
    Fx.cell_name(**CONVERGE_CONFIGS[i]) for i in (0, 2, 3, 4, 8, 9, 10)
)
CORRECTED_CONVERGENCE_CELLS = {
    Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", tier="large")):
        "convergence_{bed}.json",
}
LEGACY_CONVERGENCE_CONFIG_COUNT = 66
BACKFILL_WORKERS = 16
def _parallel_backfill(bedname: str, command: str, indices: list[int]) -> None:
    """Fill appended convergence identities at the measured large class optimum."""
    for start in range(0, len(indices), BACKFILL_WORKERS):
        batch = [subprocess.Popen([sys.executable, "-m", "mop.temporal.runs.e2",
                                  command, bedname, str(idx)])
                 for idx in indices[start:start + BACKFILL_WORKERS]]
        failed = [proc.returncode for proc in batch if proc.wait() != 0]
        if failed:
            raise RuntimeError(f"{command} backfill failed on {bedname}: {failed}")
def _backfill_appended_convergence(bedname: str) -> None:
    """The live predecessor knew 66 identities; append later sealed cells without serial fallback."""
    appended = range(min(LEGACY_CONVERGENCE_CONFIG_COUNT, len(CONVERGE_CONFIGS)),
                     len(CONVERGE_CONFIGS))
    base = [idx for idx in appended if not (io.RUNS / "e2_converge" /
            f"cshard_{bedname}_{idx}.json").is_file()]
    if base:
        _parallel_backfill(bedname, "converge_shard", base)
    extended = [idx for idx in appended if not (io.RUNS / "e2_converge_extended" /
                f"xshard_{bedname}_{idx}.json").is_file()]
    if extended:
        _parallel_backfill(bedname, "extend_converge_shard", extended)
def converge_shard(bedname: str, idx: int) -> dict:
    from mop.temporal import witness as W
    t0 = time.time()
    spec = CONVERGE_CONFIGS[idx]
    curve, spread, parameter_count = {}, {}, None
    for steps in CONVERGENCE_GRID:
        rows = [Fx.run_cell(B.splits(bedname, s_), spec, s_, "tune", steps=steps)
                for s_ in CONVERGENCE_SEEDS]
        vals = [row["accuracy"] for row in rows]
        parameter_count = rows[0]["params"]
        curve[steps] = float(np.mean(vals))
        spread[steps] = round(float(np.std(vals, ddof=1)), 5)
    w = W.plateau_validity(curve)
    doc = {"bed": bedname, "spec": spec, "cell": Fx.cell_name(**spec), "curve": curve,
           "seed_spread": spread, "seeds": list(CONVERGENCE_SEEDS),
           "parameter_count": parameter_count, **w,
           "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"cshard_{bedname}_{idx}.json", doc, "e2_converge")
    print(f"E2 converge shard {bedname} {Fx.cell_name(**spec)}: {w['classification']} "
          f"movement {w.get('second_half_movement')} in {doc['wall_seconds']}s", flush=True)
    return doc
def extend_converge_shard(bedname: str, idx: int) -> dict:
    """Append larger budgets under a new shard identity, preserving the original curve receipt."""
    from mop.temporal import witness as W
    t0 = time.time()
    spec = CONVERGE_CONFIGS[idx]
    base_path = io.RUNS / "e2_converge" / f"cshard_{bedname}_{idx}.json"
    base = json.loads(base_path.read_text()) if base_path.is_file() else converge_shard(bedname, idx)
    expected_cell = Fx.cell_name(**spec)
    if base.get("cell") != expected_cell or Fx.cell_name(**base.get("spec", {})) != expected_cell:
        raise RuntimeError(f"base convergence identity mismatch at {base_path}: expected {expected_cell}")
    curve = {int(k): float(v) for k, v in base["curve"].items()}
    spread = {int(k): float(v) for k, v in base["seed_spread"].items()}
    parameter_count = base.get("parameter_count")
    for steps in EXTENDED_CONVERGENCE_GRID:
        rows = [Fx.run_cell(B.splits(bedname, s_), spec, s_, "tune", steps=steps)
                for s_ in CONVERGENCE_SEEDS]
        vals = [row["accuracy"] for row in rows]
        parameter_count = rows[0]["params"]
        curve[steps] = float(np.mean(vals))
        spread[steps] = round(float(np.std(vals, ddof=1)), 5)
    w = W.plateau_validity(curve)
    doc = {
        "schema": "mop-e2-extended-convergence-shard/v1",
        "bed": bedname,
        "spec": spec,
        "cell": expected_cell,
        "curve": curve,
        "seed_spread": spread,
        "seeds": list(CONVERGENCE_SEEDS),
        "parameter_count": parameter_count,
        "extends": {
            "path": base_path.relative_to(io.ROOT).as_posix(),
            "sha256": io.sha_file(base_path),
            "grid": base.get("budgets", list(CONVERGENCE_GRID)),
        },
        "authority_commit": io.commit(),
        **w,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"xshard_{bedname}_{idx}.json", doc, "e2_converge_extended")
    print(f"E2 extended convergence {bedname} {doc['cell']}: {w['classification']} "
          f"movement {w.get('second_half_movement')} in {doc['wall_seconds']}s", flush=True)
    return doc
def converge(bedname: str) -> dict:
    """Long budget curves for every load bearing configuration, judged by the strict plateau witness."""
    from mop.temporal import witness as W
    t0 = time.time()
    _backfill_appended_convergence(bedname)
    out = {}
    for idx, spec in enumerate(CONVERGE_CONFIGS):
        cell = Fx.cell_name(**spec)
        correction_name = CORRECTED_CONVERGENCE_CELLS.get(cell)
        corrected = (io.RUNS / "e2_converge_corrections" / correction_name.format(bed=bedname)
                     if correction_name else io.RUNS / "absent")
        p = corrected if corrected.is_file() else io.RUNS / "e2_converge_extended" / f"xshard_{bedname}_{idx}.json"
        p = p if p.is_file() else io.RUNS / "e2_converge" / f"cshard_{bedname}_{idx}.json"
        if p.is_file():
            d = json.loads(p.read_text())
            out[d["cell"]] = d
            continue
        d = converge_shard(bedname, idx)
        out[d["cell"]] = d
    grid = sorted({int(x) for d in out.values() for x in d.get("curve", {})})
    load = {k: out.get(k) for k in LOAD_BEARING_CONVERGENCE_CELLS}
    doc = {"schema": "mop-e2-convergence/v3", "bed": bedname, "grid": grid,
           "seeds_per_budget": list(CONVERGENCE_SEEDS),
           "configs": out,
           "all_converged": all(v["converged"] for v in out.values()),
           "unconverged": [k for k, v in out.items() if not v["converged"]],
           "load_bearing_cells": list(LOAD_BEARING_CONVERGENCE_CELLS),
           "load_bearing_all_converged": all(v and v["converged"] for v in load.values()),
           "load_bearing_unconverged": [k for k, v in load.items() if not v or not v["converged"]],
           "wall_seconds": round(time.time() - t0, 1)}
    io.run_json(f"converge_{bedname}.json", doc, "e2_converge")
    print(f"E2 convergence {bedname}: all converged {doc['all_converged']}, "
          f"unconverged {doc['unconverged']}", flush=True)
    return doc
def scout_result() -> dict:
    """Seal the scout reduction and a hash inventory of every completed raw scout shard."""
    beds, shards, failures = {}, [], []
    expected_cells = {Fx.cell_name(**s) for s in Fx.sweep_cells()["_all"]}
    for bed in ("har_stream", "speech_stream", "harth_stream"):
        aggregate = io.RUNS / "e2_scout" / f"scout_{bed}.json"
        if aggregate.is_file():
            d = json.loads(aggregate.read_text())
            beds[bed] = {k: d.get(k) for k in (
                "seeds", "steps", "evaluated_on", "n_cells", "best_cell", "high_variance_cells",
                "clearly_dominated_cells", "median_sd", "not_scientific_evidence")}
        for seed in SCOUT_SEEDS:
            p = io.RUNS / "e2_scout" / f"shard_{bed}_{seed}.json"
            checks = {"exists": p.is_file()}
            if p.is_file():
                d = json.loads(p.read_text())
                cells = [r.get("cell") for r in d.get("runs", [])]
                checks.update({
                    "bed_identity": d.get("bed") == bed and all(r.get("bed") == bed for r in d["runs"]),
                    "seed_identity": d.get("seed") == seed and all(r.get("seed") == seed for r in d["runs"]),
                    "factorial_identity": len(cells) == len(expected_cells) and set(cells) == expected_cells,
                    "receipts_complete": all(r.get("checkpoint_sha_after") and r.get("steps")
                                             and r.get("params") for r in d["runs"]),
                })
            checks["all_pass"] = all(checks.values())
            row = {"path": p.relative_to(io.ROOT).as_posix(), "bed": bed, "seed": seed,
                   "sha256": io.sha_file(p) if p.is_file() else None, "checks": checks}
            shards.append(row)
            if not checks["all_pass"]:
                failures.append(row["path"])
    doc = {"schema": "mop-e2-scout-result/v1", "beds": beds, "shards": shards,
           "n_expected_shards": 12, "n_completed_shards": sum(s["checks"]["all_pass"] for s in shards),
           "failures": failures, "all_pass": not failures,
           "reduction_rule": ("only clearly dominated, contrast redundant, instrument invalid, or resource "
                              "infeasible cells may be removed; this scout removed no selected factorial cell"),
           "not_principal_scientific_evidence": True}
    io.seal("MOP_E2_SCOUT_RESULT.json", doc)
    return doc
def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "calibration"
    if cmd == "calibration":
        r = calibration()
        io.seal("MOP_E2_CALIBRATION.json", {
            "schema": "mop-e2-calibration/v1",
            "rule": "a world whose generative truth the analysis cannot name is a world the analysis may not judge",
            "worlds": r,
            "all_pass": r["all_pass"],
            "n_worlds": len([k for k, v in r.items() if isinstance(v, dict)]),
        })
        print(f"E2 calibration: {len([k for k, v in r.items() if isinstance(v, dict)])} worlds, "
              f"all pass {r['all_pass']}", flush=True)
        for k, v in r.items():
            if isinstance(v, dict) and not v["pass"]:
                print(f"  MISS {k}: expected {v['expected_contains']} got {v['recovered']}", flush=True)
    elif cmd == "scout":
        scout(argv[1])
    elif cmd == "scout_shard":
        scout_shard(argv[1], int(argv[2]))
    elif cmd == "converge_shard":
        converge_shard(argv[1], int(argv[2]))
    elif cmd == "extend_converge_shard":
        extend_converge_shard(argv[1], int(argv[2]))
    elif cmd == "principal":
        principal(argv[1], int(argv[2]))
    elif cmd == "converge":
        converge(argv[1])
    elif cmd == "scout_result":
        scout_result()
    print(f"E2_{cmd.upper()}_DONE", flush=True)
    lock = os.environ.get("TEMPORAL_SHARD_LOCK")
    if lock:
        from pathlib import Path
        Path(lock).unlink(missing_ok=True)
if __name__ == "__main__":
    main()
