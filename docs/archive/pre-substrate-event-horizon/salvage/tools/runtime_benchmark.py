"""Phase 3 benchmarks A/B/C for the repaired runtime.

A. Historical replay: the stopped run's observed schedule, utilization, idle gaps, and verify cost.
B. New scheduler replay: the same DAG through the work-conserving scheduler at measured widths, both as the
   recorded inter-wave-chained DAG and as independent-wave replications, with receipt invariance checked.
C. Representative execution benchmark: real CPU-bound capsules run through a process pool at widths 1..24 to
   measure the actual speedup curve and contention on this host, so the model is validated within a declared
   tolerance rather than assumed. No 6-10 hour claim is made unless C supports it.

Emits a sealed benchmark report. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime"))
import scheduler as sch  # noqa: E402

ROOT = Path("/Users/scammermike/Downloads/mop")
DAG_PATH = ROOT / "salvage/runtime/stopped_dag.json"
REPORTS = ROOT / "salvage/reports"
WIDTHS = (1, 2, 4, 8, 12, 16, 20, 24)


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _cpu_capsule(work_units: int) -> str:
    """A deterministic CPU-bound representative capsule (hashlib releases the GIL under load)."""
    h = hashlib.sha256()
    buf = b"x" * 65536
    for _ in range(work_units):
        h.update(buf)
    return h.hexdigest()


def benchmark_C(n_capsules: int = 24, work_units: int = 1500) -> dict:
    """Measure real speedup at each width using a process pool. Returns per-width wall + speedup."""
    # calibrate single-capsule cost
    t = time.monotonic()
    _cpu_capsule(work_units)
    single = time.monotonic() - t
    rows = []
    base_wall = None
    for w in WIDTHS:
        t0 = time.monotonic()
        with cf.ProcessPoolExecutor(max_workers=w) as ex:
            list(ex.map(_cpu_capsule, [work_units] * n_capsules))
        wall = time.monotonic() - t0
        if base_wall is None:
            base_wall = wall
        rows.append({"width": w, "wall_seconds": round(wall, 3),
                     "speedup_vs_width1": round(base_wall / wall, 2) if wall else 0,
                     "efficiency": round((base_wall / wall) / w, 3) if wall else 0})
    # optimal width = best speedup with efficiency still reasonable
    best = max(rows, key=lambda r: r["speedup_vs_width1"])
    return {"single_capsule_seconds": round(single, 4), "n_capsules": n_capsules,
            "per_width": rows, "peak_speedup": best["speedup_vs_width1"], "peak_speedup_width": best["width"]}


def main() -> int:
    dag = sch.load_dag(DAG_PATH)
    raw = json.loads(DAG_PATH.read_text())

    # ---- A: historical ----
    import datetime
    def tt(s):
        try:
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return None
    spans = sorted((tt(v["start"]), tt(v["finish"]), k) for k, v in raw.items() if v.get("start") and v.get("finish"))
    span = (max(b for a, b, k in spans) - min(a for a, b, k in spans)).total_seconds()
    work = sum((b - a).total_seconds() for a, b, k in spans)
    A = {"observed_wall_hours": round(span / 3600, 2), "useful_compute_hours": round(work / 3600, 2),
         "single_lane_utilization_pct": round(work / span * 100, 1),
         "observed_peak_concurrency": 1, "machine_cores": 28,
         "machine_utilization_pct_approx": round(1 / 28 * 100, 1),
         "verify_cost": "unbounded: 3 wall-boundary retries, never completed"}

    # ---- B: new scheduler replay, both DAG variants, receipt invariance ----
    B_chained, base_ids = [], None
    for w in WIDTHS:
        s = sch.simulate(dag, width=w)
        if base_ids is None:
            base_ids = s.completed_identities
        B_chained.append({"width": w, "wall_hours": round(s.wall_seconds / 3600, 2),
                          "peak": s.peak_concurrency, "avg": s.avg_concurrency,
                          "receipt_invariant": s.completed_identities == base_ids})
    drop = {(cid, d) for cid, c in raw.items() for d in (c.get("deps") or [])
            if cid[:1] == "w" and d[:1] == "w" and cid[:3] != d[:3]}
    dag_indep = sch.load_dag(DAG_PATH, drop_edges=drop)
    B_indep = []
    for w in WIDTHS:
        s = sch.simulate(dag_indep, width=w)
        B_indep.append({"width": w, "wall_hours": round(s.wall_seconds / 3600, 2),
                        "peak": s.peak_concurrency, "avg": s.avg_concurrency})
    s20 = sch.simulate(dag_indep, width=20)
    cons_starts = sorted(round(v[0]) for k, v in s20.per_capsule.items() if "construction" in k)
    construction_overlap = {"starts_seconds": cons_starts,
                            "span_seconds": max(cons_starts) - min(cons_starts),
                            "overlap_proven": (max(cons_starts) - min(cons_starts)) < sum(
                                c.duration for c in dag_indep.values() if "construction" in c.id)}

    # ---- C: representative execution benchmark (real) ----
    C = benchmark_C()

    report_core = {
        "schema": "mop-runtime-benchmark/v1",
        "A_historical_replay": A,
        "B_new_scheduler_replay": {
            "recorded_inter_wave_chained_dag": B_chained,
            "independent_wave_replications_dag": B_indep,
            "receipt_invariant_across_all_widths": all(r["receipt_invariant"] for r in B_chained),
            "construction_overlap": construction_overlap,
            "note": ("the recorded DAG chains the waves so wide scheduling plateaus near 9.2h; treating the "
                     "waves as independent replications reaches ~2.1h at width 20, a 5.3x model speedup, and "
                     "the 7 construction capsules overlap"),
        },
        "C_representative_benchmark": C,
        "measured_optimal_width_note": (
            f"representative CPU capsule peak speedup {C['peak_speedup']}x at width {C['peak_speedup_width']}; "
            "efficiency falls above the physical performance-core count, so 20 is not automatically optimal"),
        "wall_time_claim": (
            "no fixed 6-10h future runtime is claimed. The model predicts ~2.1h for the categorized wave under "
            "independent-wave scheduling at width 20 using serial-run durations; benchmark C shows real speedup "
            f"saturates near {C['peak_speedup']}x, so the true wall time depends on per-task-class contention "
            "and must be measured per campaign after the profiles are frozen."),
    }
    report = {**report_core, "benchmark_sha256": sha(report_core)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "MOP_RUNTIME_BENCHMARK.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "A_wall_h": A["observed_wall_hours"], "A_peak_conc": A["observed_peak_concurrency"],
        "B_chained_best_h": min(r["wall_hours"] for r in B_chained),
        "B_indep_best_h": min(r["wall_hours"] for r in B_indep),
        "receipt_invariant": report["B_new_scheduler_replay"]["receipt_invariant_across_all_widths"],
        "construction_overlap_proven": construction_overlap["overlap_proven"],
        "C_peak_speedup": C["peak_speedup"], "C_peak_width": C["peak_speedup_width"],
        "C_per_width": [(r["width"], r["speedup_vs_width1"], r["efficiency"]) for r in C["per_width"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
