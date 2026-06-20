#!/usr/bin/env python
"""CPU-Now campaign driver. Runs the cost-ordered tiers T0..T3 on cpu, parallelized across
cores via the worker pool, checkpointed and resumable. T0-T2 run to completion at modest real
scale; T3 drains a wall-clock budget. Records per-run-unit timings and feeds the Studio cost
projection. Every result is tagged real-encoder or provisional. Tier E and R are never run.

Usage: python scripts/cpu_campaign.py [--budget 600] [--seeds 5] [--t3-budget 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from devsys.config import REPO_ROOT
from devsys.harness.cpu_pool import physical_cores, plan_pool, run_pool
from devsys.logging_utils import get_logger

log = get_logger("cpu_campaign")
OUT = REPO_ROOT / "runs" / "cpu_campaign"
CKPT = OUT / "ckpt"


def _save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- unit builders
def _e1_units(seeds, dim=64, samples=160):
    base = [
        f"experiment.stream.dim={dim}",
        f"experiment.stream.samples_per_task={samples}",
        "shell.buffer.index=brute",
        "shell.consolidation.method=ewc",
        "shell.consolidation.ewc_lambda=1000.0",
    ]
    units = []
    for inc in ("domain", "class", "task"):
        for s in range(seeds):
            units.append(
                {
                    "key": f"e1_{inc}_s{s}",
                    "overrides": [
                        "experiment=e1_baseline",
                        "device=cpu",
                        f"seed={s}",
                        f"experiment.stream.incremental={inc}",
                        *base,
                    ],
                }
            )
    return units


def _i4_units(seeds, hiddens=(32, 48, 64)):
    sl = "[" + ",".join(str(i) for i in range(seeds)) + "]"
    return [
        {
            "key": f"i4_h{h}",
            "overrides": [
                "experiment=i4_backprop_alts",
                "device=cpu",
                f"experiment.hidden={h}",
                f"experiment.seeds={sl}",
                "experiment.samples=400",
                "experiment.epochs=250",
            ],
        }
        for h in hiddens
    ]


def _e9_units(seeds, hiddens=(32, 64)):
    sl = "[" + ",".join(str(i) for i in range(seeds)) + "]"
    return [
        {
            "key": f"e9_h{h}",
            "overrides": [
                "experiment=e9_local",
                "device=cpu",
                f"experiment.hidden={h}",
                f"experiment.seeds={sl}",
                "experiment.samples=400",
                "experiment.passes=4",
            ],
        }
        for h in hiddens
    ]


def _t3_units(seeds=3):
    """Reduced grids: representative axis subsets, reduced seeds (recorded in DECISIONS)."""
    common = [
        "device=cpu",
        "experiment.stream.dim=64",
        "experiment.stream.samples_per_task=140",
        "shell.buffer.index=brute",
    ]
    units = []
    # E2 replay (tie_tol subset)
    for tt in (0.02, 0.05):
        units.append(
            {
                "key": f"t3_e2_tt{tt}",
                "overrides": ["experiment=e2_replay", "seed=0", f"experiment.tie_tol={tt}", *common],
            }
        )
    # E3 plasticity (schedule subset via shell.plasticity.schedule)
    for sch in ("soft", "hard"):
        units.append(
            {
                "key": f"t3_e3_{sch}",
                "overrides": [
                    "experiment=e3_plasticity",
                    "seed=0",
                    f"shell.plasticity.schedule={sch}",
                    *common,
                ],
            }
        )
    # E4 neuromod (signal-by-gate is internal; sweep noise_scale)
    for ns in (4.0, 6.0):
        units.append(
            {
                "key": f"t3_e4_ns{ns}",
                "overrides": [
                    "experiment=e4_neuromod",
                    "seed=0",
                    f"experiment.noise_scale={ns}",
                    "experiment.dim=40",
                    "experiment.steps=160",
                    "device=cpu",
                ],
            }
        )
    # E7 sparse (kwta_k subset)
    for k in (4, 8):
        units.append(
            {
                "key": f"t3_e7_k{k}",
                "overrides": [
                    "experiment=e7_sparse",
                    "seed=0",
                    f"experiment.kwta_k={k}",
                    "experiment.epochs_per_task=30",
                    *common,
                ],
            }
        )
    # consolidation comparison (E1 with si vs ewc vs both) - the sharp weight-space test
    for meth in ("si", "both"):
        units.append(
            {
                "key": f"t3_consol_{meth}",
                "overrides": [
                    "experiment=e1_baseline",
                    "seed=0",
                    f"shell.consolidation.method={meth}",
                    *common,
                ],
            }
        )
    return units


# --------------------------------------------------------------------------- tiers
def tier0(seeds: int) -> dict:
    log.info("T0: determinism (11A) + negative-result registry (11D)")
    res = {}
    try:
        from devsys.studies.determinism_study import determinism_study

        res["determinism_11A"] = determinism_study(reps=3, toy=True)
    except Exception as e:
        res["determinism_11A"] = {"error": repr(e)[:200]}
        log.warning("11A failed: %s", repr(e)[:160])
    try:
        from devsys.studies.negative_registry import negative_registry

        res["negative_registry_11D"] = negative_registry(toy=True)
    except Exception as e:
        res["negative_registry_11D"] = {"error": repr(e)[:200]}
        log.warning("11D failed: %s", repr(e)[:160])
    _save("t0.json", res)
    return res


def tier1(seeds: int) -> dict:
    log.info("T1: diagnostics battery + seed-variance (11B)")
    res = {}
    try:
        import subprocess

        py = str(REPO_ROOT / ".venv/bin/python")
        p = subprocess.run(
            [py, str(REPO_ROOT / "scripts/run_diagnostics.py"), "device=cpu", "+dim=64"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        res["diagnostics"] = (
            json.loads(p.stdout)
            if p.returncode == 0 and p.stdout.strip().startswith("{")
            else {"raw": p.stdout[-400:], "rc": p.returncode}
        )
    except Exception as e:
        res["diagnostics"] = {"error": repr(e)[:200]}
    try:
        from devsys.studies.seed_variance import seed_variance

        res["seed_variance_11B"] = seed_variance(max_seeds=min(8, max(5, seeds)), toy=True)
    except Exception as e:
        res["seed_variance_11B"] = {"error": repr(e)[:200]}
        log.warning("11B failed: %s", repr(e)[:160])
    _save("t1.json", res)
    return res


def tier2(seeds: int) -> dict:
    log.info("T2: I4/E9 full + E1 expanded + track8 (8B, 8C)")
    workers, threads = plan_pool("heavy")
    units = _i4_units(seeds) + _e9_units(seeds) + _e1_units(seeds)
    pool = run_pool(units, workers, threads, CKPT / "t2")
    res = {
        "pool": {k: {"seconds": v["seconds"], "cached": v.get("cached")} for k, v in pool.results.items()},
        "degraded": pool.degraded,
        "wall_seconds": pool.wall_seconds,
        "metrics": {k: v["metrics"] for k, v in pool.results.items()},
    }
    for mod, fn, kw in (
        ("assoc_memory", "assoc_memory", {"toy": True}),
        ("pc_depth", "pc_depth", {"toy": True}),
    ):
        try:
            m = __import__(f"devsys.studies.{mod}", fromlist=[fn])
            res[mod] = getattr(m, fn)(**kw)
        except Exception as e:
            res[mod] = {"error": repr(e)[:200]}
            log.warning("%s failed: %s", mod, repr(e)[:160])
    _save("t2.json", res)
    return res


def tier3(budget_s: float, seeds: int = 3) -> dict:
    log.info("T3: reduced grids, draining %.0fs budget", budget_s)
    workers, threads = plan_pool("heavy")
    units = _t3_units(seeds)
    # bound by budget: run in slices, stop when budget exceeded (checkpointing makes it resumable)
    t0 = time.perf_counter()
    done = []
    pool_results = {}
    for u in units:
        if time.perf_counter() - t0 > budget_s:
            log.info("T3 budget hit; %d/%d units done, rest checkpointed/resumable", len(done), len(units))
            break
        r = run_pool([u], workers, threads, CKPT / "t3")
        pool_results.update(
            {k: {"seconds": v["seconds"], "metrics": v["metrics"]} for k, v in r.results.items()}
        )
        done.append(u["key"])
    res = {"completed": done, "total_units": len(units), "results": pool_results, "budget_s": budget_s}
    _save("t3.json", res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="full seed count for T0-T2")
    ap.add_argument("--t3-budget", type=float, default=600.0, help="wall-clock seconds for T3")
    ap.add_argument("--tiers", default="0,1,2,3")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    OUT.mkdir(parents=True, exist_ok=True)
    log.info(
        "CPU campaign: %d physical cores, seeds=%d, t3_budget=%.0fs", physical_cores(), a.seeds, a.t3_budget
    )
    sel = set(a.tiers.split(","))
    summary = {"cores": physical_cores(), "seeds": a.seeds}
    timings = {}
    if "0" in sel:
        summary["t0"] = tier0(a.seeds)
    if "1" in sel:
        summary["t1"] = tier1(a.seeds)
    if "2" in sel:
        t2 = tier2(a.seeds)
        summary["t2"] = {k: v for k, v in t2.items() if k != "metrics"}
        for k, v in t2.get("pool", {}).items():
            timings[k.split("_s")[0].rsplit("_", 0)[0]] = v["seconds"]  # rough per-leg-type seconds
    if "3" in sel:
        summary["t3"] = tier3(a.t3_budget, seeds=3)
    # cost projection from measured timings
    try:
        from devsys.studies.cost_projection import cost_projection

        w, _ = plan_pool("heavy")
        summary["cost_projection"] = cost_projection(timings or {}, workers=w, full_seed=a.seeds)
    except Exception as e:
        summary["cost_projection"] = {"error": repr(e)[:200]}
    _save("summary.json", summary)
    print(
        json.dumps(
            {k: (v if not isinstance(v, dict) else "...") for k, v in summary.items()}, indent=2, default=str
        )
    )
    log.info("campaign summary written to %s", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
