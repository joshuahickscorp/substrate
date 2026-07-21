"""Test, coverage, resource and clean clone reports."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from mop.method import gate
from mop.temporal import io

ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": "src"}


def _env():
    import os
    return dict(os.environ, **ENV)


def tests_and_coverage() -> tuple[dict, dict]:
    t0 = time.time()
    out = io.RUNS / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    data = out / ".cov"
    r = subprocess.run([sys.executable, "-m", "coverage", "run", "--branch", f"--data-file={data}",
                        "--source=src/mop/temporal,src/mop/method", "-m", "pytest",
                        "tests/temporal", "tests/method", "-q"],
                       cwd=io.ROOT, env=_env(), capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "coverage", "json", f"--data-file={data}",
                    "-o", str(out / "coverage.json")], cwd=io.ROOT, env=_env(), capture_output=True)
    cov = json.loads((out / "coverage.json").read_text()) if (out / "coverage.json").is_file() else {}
    files = cov.get("files", {})

    def totals(pred):
        sel = {k: v for k, v in files.items() if pred(k)}
        st = sum(v["summary"]["num_statements"] for v in sel.values())
        cv = sum(v["summary"]["covered_lines"] for v in sel.values())
        nb = sum(v["summary"]["num_branches"] for v in sel.values())
        bc = sum(v["summary"]["covered_branches"] for v in sel.values())
        return {"files": sorted(sel), "statement": round(100.0 * cv / max(1, st), 1),
                "branch": round(100.0 * bc / max(1, nb), 1), "num_statements": st, "num_branches": nb}

    kernel = totals(lambda k: k.startswith("src/mop/method/") and "/runs/" not in k)
    critical = totals(lambda k: (k.startswith("src/mop/method/") and "/runs/" not in k)
                      or (k.startswith("src/mop/temporal/") and "/runs/" not in k))
    stages = totals(lambda k: "/runs/" in k)
    tr = {"schema": "mop-temporal-core-test-report/v1", "returncode": r.returncode,
          "passed": r.returncode == 0, "tail": (r.stdout or "").strip().splitlines()[-3:],
          "wall_seconds": round(time.time() - t0, 1)}
    cr = {"schema": "mop-temporal-core-coverage-report/v1",
          "method_kernel": kernel, "method_kernel_gate": gate.coverage_gate(
              kernel["statement"], kernel["branch"], scope=kernel["files"], excluded=stages["files"]),
          "active_critical_path": critical,
          "active_critical_path_gate": gate.coverage_gate(
              critical["statement"], critical["branch"], scope=critical["files"],
              excluded=stages["files"], target_statement=90.0, target_branch=80.0),
          "program_stages": stages,
          "scope_rule": ("the method kernel keeps its 92 and 82 targets. The active critical path, the kernel "
                         "plus the temporal core modules, carries 90 and 80. Program stages are drivers "
                         "verified by executing the program and their numbers are reported, not hidden")}
    return tr, cr


def resource_report(points=(1, 2, 4, 8, 12, 16, 20, 24)) -> dict:
    rows = []
    for tier in ("small", "large"):
        script = (
            "import numpy as np, torch, time, sys; sys.path.insert(0,'src');"
            "from mop.temporal import beds as B, factorial as Fx;"
            f"sp=B.splits('har_stream',0); t=time.time();"
            f"Fx.run_cell(sp, dict(Fx.REFERENCE, tier='{tier}'), 0, 'tune', steps=40); print(40/(time.time()-t))")
        for n in points:
            t0 = time.time()
            procs = [subprocess.Popen([sys.executable, "-c", script], cwd=io.ROOT, env=_env(),
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                     for _ in range(n)]
            rates = []
            for p in procs:
                o, _ = p.communicate()
                try:
                    rates.append(float(o.strip().splitlines()[-1]))
                except (ValueError, IndexError):
                    pass
            rows.append({"tier": tier, "workers": n, "aggregate_steps_per_second": round(sum(rates), 2),
                         "per_worker": round(sum(rates) / max(1, len(rates)), 2),
                         "wall_seconds": round(time.time() - t0, 1)})
            print(f"  {tier} workers={n} aggregate={rows[-1]['aggregate_steps_per_second']}", flush=True)
    best = {t: max((r for r in rows if r["tier"] == t), key=lambda r: r["aggregate_steps_per_second"])
            for t in ("small", "large")}
    return {"schema": "mop-temporal-core-resource-report/v1", "measurements": rows,
            "optimum_workers": {t: b["workers"] for t, b in best.items()},
            "optimum_aggregate": {t: b["aggregate_steps_per_second"] for t, b in best.items()},
            "note": ("the optimum is rebenchmarked per size tier because the previous optimum was measured on "
                     "one tier and need not hold for a larger core")}


def clean_clone() -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["git", "clone", "--quiet", "--depth", "1", str(io.ROOT), td + "/c"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"cloned": False, "error": r.stderr[-300:]}
        t = subprocess.run([sys.executable, "-m", "pytest", "tests/temporal", "-q"],
                           cwd=td + "/c", env=_env(), capture_output=True, text=True)
        proof = subprocess.run(["find", "proof/substrate/" + io.PROGRAM, "-name", "*.json"],
                               cwd=td + "/c", capture_output=True, text=True)
        return {"schema": "mop-temporal-core-clean-clone/v1", "cloned": True,
                "tests_returncode": t.returncode,
                "tests_tail": (t.stdout or "").strip().splitlines()[-3:],
                "proof_artifacts_present": len([x for x in proof.stdout.splitlines() if x]),
                "skips_expected": "corpora live outside the repository by design, so data bound tests skip"}


def main():
    t0 = time.time()
    tr, cr = tests_and_coverage()
    io.seal("MOP_TEMPORAL_CORE_TEST_REPORT.json", tr)
    io.seal("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json", cr)
    io.seal("MOP_TEMPORAL_CORE_RESOURCE_REPORT.json", resource_report())
    io.seal("MOP_TEMPORAL_CORE_CLEAN_CLONE.json", clean_clone())
    print(f"reports: tests {tr['passed']}, kernel {cr['method_kernel']['statement']}/"
          f"{cr['method_kernel']['branch']}, critical {cr['active_critical_path']['statement']}/"
          f"{cr['active_critical_path']['branch']} in {round(time.time() - t0, 1)}s", flush=True)
    print("REPORTS_DONE", flush=True)


if __name__ == "__main__":
    main()
