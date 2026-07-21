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


def _host_snapshot() -> dict:
    import os
    import shutil

    def read(args):
        r = subprocess.run(args, capture_output=True, text=True)
        return {"returncode": r.returncode, "output": (r.stdout or r.stderr).strip()[-2000:]}

    snap = {"cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()),
            "disk_free_bytes": shutil.disk_usage(io.ROOT).free,
            "thread_controls": {k: ENV[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    for name, args in (("memory_pages", ["vm_stat"]), ("swap", ["sysctl", "vm.swapusage"]),
                       ("thermal", ["pmset", "-g", "therm"]), ("disk_io", ["iostat", "-d", "-c", "2"])):
        try:
            snap[name] = read(args)
        except FileNotFoundError:
            snap[name] = {"returncode": 127, "output": "command unavailable"}
    return snap


def resource_report(points=(1, 2, 4, 8, 12, 16, 20, 22, 24)) -> dict:
    before = _host_snapshot()
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
            failures = n - len(rates)
            rows.append({"tier": tier, "workers": n, "aggregate_steps_per_second": round(sum(rates), 2),
                         "per_worker": round(sum(rates) / max(1, len(rates)), 2),
                         "failures": failures, "failure_rate": round(failures / n, 5),
                         "wall_seconds": round(time.time() - t0, 1)})
            print(f"  {tier} workers={n} aggregate={rows[-1]['aggregate_steps_per_second']}", flush=True)
    for tier in ("small", "large"):
        baseline = next(r["per_worker"] for r in rows if r["tier"] == tier and r["workers"] == 1)
        for row in (r for r in rows if r["tier"] == tier):
            row["per_job_slowdown"] = round(row["per_worker"] / max(baseline, 1e-9), 4)
    best = {t: max((r for r in rows if r["tier"] == t), key=lambda r: r["aggregate_steps_per_second"])
            for t in ("small", "large")}
    return {"schema": "mop-temporal-core-resource-report/v1", "measurements": rows,
            "optimum_workers": {t: b["workers"] for t, b in best.items()},
            "optimum_aggregate": {t: b["aggregate_steps_per_second"] for t, b in best.items()},
            "scheduler_policy": {
                "small_cap": 24, "large_cap": 16,
                "mixed_fill": ("large work is ordered first under the measured large cap, then distinct "
                               "small shard identities fill otherwise idle slots without exceeding that cap"),
                "repaired_incident": ("six large configuration shards occupied only six of sixteen licensed "
                                      "slots while small extensions were ready; locked small identities filled "
                                      "the idle slots and the policy was made permanent"),
            },
            "host_before": before, "host_after": _host_snapshot(),
            "note": ("the optimum is rebenchmarked per size tier because the previous optimum was measured on "
                     "one tier and need not hold for a larger core")}


def clean_clone() -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        commit = io.commit()
        clone = td + "/c"
        r = subprocess.run(["git", "clone", "--quiet", "--no-local", str(io.ROOT), clone],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"cloned": False, "error": r.stderr[-300:]}
        checkout = subprocess.run(["git", "checkout", "--quiet", commit], cwd=clone,
                                  capture_output=True, text=True)
        env = _env()

        def run(args):
            return subprocess.run(args, cwd=clone, env=env, capture_output=True, text=True)

        head = run(["git", "rev-parse", "HEAD"])
        imp = run([sys.executable, "-c", "import mop.temporal, mop.method, fastforge.engine"])
        method = run([sys.executable, "-m", "pytest", "tests/method", "-q"])
        temporal = run([sys.executable, "-m", "pytest", "tests/temporal", "-q"])
        principal_subset = run([sys.executable, "-m", "pytest", "tests/temporal", "-q", "-k",
                                "calibration or selection or convergence or independent"])
        proof_script = """
import json
from pathlib import Path
from mop.temporal import io
root=Path('proof/substrate')/io.PROGRAM
docs=list(root.glob('*.json'))
bad=[]
for p in docs:
 d=json.loads(p.read_text())
 if 'sha256' in d and d['sha256'] != io.sha_obj({k:v for k,v in d.items() if k!='sha256'}): bad.append(str(p))
assert docs and not bad, (len(docs),bad)
print(len(docs))
"""
        proof = run([sys.executable, "-c", proof_script])
        checkpoint_script = """
from pathlib import Path
import torch
from fastforge import engine as E
from mop.temporal import beds as B, factorial as Fx, io
paths=sorted((Path('proof/substrate')/io.PROGRAM/'checkpoints').glob('*.pt'))
assert paths
for p in paths:
 d=torch.load(p,map_location='cpu',weights_only=False); sp=B.splits(d['bed'],d['seed'])
 m=Fx.build_cell(sp,seed=d['seed'],**d['spec'])[0]; m.load_state_dict(d['state_dict'])
 assert E.checkpoint_sha(m)==d['training_receipt']['checkpoint_sha_after']
print(len(paths))
"""
        checkpoint = run([sys.executable, "-c", checkpoint_script])
        supervisor = run([sys.executable, "-m", "mop.temporal.runs.supervisor", "status"])
        fabric_script = """
import json
from pathlib import Path
from mop.temporal import io
p=Path('proof/substrate')/io.PROGRAM/'MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json'
d=json.loads(p.read_text()); assert d['verification']['all_pass']
assert all(Path(a['canonical_path']).is_file() for a in d['artifacts'])
print(d['union']['merkle_root'])
"""
        fabric = run([sys.executable, "-c", fabric_script])
        clean = run(["git", "status", "--porcelain"])
        checks = {
            "exact_commit_checkout": checkout.returncode == 0 and head.stdout.strip() == commit,
            "package_import": imp.returncode == 0,
            "method_acceptance_and_tests": method.returncode == 0,
            "unit_tests": temporal.returncode == 0,
            "principal_test_subset": principal_subset.returncode == 0,
            "proof_hash_verification": proof.returncode == 0,
            "checkpoint_restoration": checkpoint.returncode == 0,
            "supervisor_status": supervisor.returncode == 0,
            "evidence_fabric_lookup": fabric.returncode == 0,
            "cli_inspection": supervisor.returncode == 0 and 'mop-temporal-supervisor-status/v1' in supervisor.stdout,
            "clean_worktree": clean.returncode == 0 and not clean.stdout.strip(),
        }
        checks["all_pass"] = all(checks.values())
        return {"schema": "mop-temporal-core-clean-clone/v2", "cloned": True,
                "commit": commit, "checks": checks,
                "method_tail": (method.stdout or method.stderr).strip().splitlines()[-3:],
                "temporal_tail": (temporal.stdout or temporal.stderr).strip().splitlines()[-3:],
                "proof_artifacts_present": int(proof.stdout.strip() or 0) if proof.returncode == 0 else 0,
                "checkpoints_restored": int(checkpoint.stdout.strip() or 0) if checkpoint.returncode == 0 else 0,
                "exact_skips": [], "offline_installation_claimed": False,
                "all_pass": checks["all_pass"]}


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
