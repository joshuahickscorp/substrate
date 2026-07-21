"""Test, coverage, resource and clean clone reports."""

from __future__ import annotations

import json
from pathlib import Path
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
            "scheduler_policy": {"small_cap": 24, "large_cap": 16,
                                 "resource_classes_are_isolated": True,
                                 "reason": ("a locked mixed fill trial terminated every added small job "
                                            "without a receipt while all large jobs remained healthy")},
            "orchestration_incidents": [json.loads(p.read_text()) for p in sorted(
                (io.RUNS / "orchestration").glob("*.json"))],
            "host_before": before, "host_after": _host_snapshot(),
            "note": ("the optimum is rebenchmarked per size tier because the previous optimum was measured on "
                     "one tier and need not hold for a larger core")}


def verify_fabric_tree(root: Path) -> dict:
    """Verify the final manifest, source files and content addressed objects in a clean tree."""
    from mop.temporal.runs.fabric import FABRIC_NAME, _atomic_partial, merkle, sha_bytes

    root = root.resolve()
    proof_root = root / "proof" / "substrate" / io.PROGRAM
    fabric_path = proof_root / FABRIC_NAME
    doc = json.loads(fabric_path.read_text())
    assert isinstance(doc, dict)
    assert doc.get("sha256_version") == "canonical_json_v2"
    assert doc.get("sha256") == io.sha_obj({k: v for k, v in doc.items() if k != "sha256"})
    assert doc["verification"]["all_pass"] and doc["mutations"]["all_rejected"]
    assert all(doc["mutations"].get("mutation_application", {}).values())
    artifacts = doc["artifacts"]
    assert isinstance(artifacts, list) and all(isinstance(a, dict) for a in artifacts)
    ids = [a["logical_id"] for a in artifacts]
    assert len(ids) == len(set(ids)) == doc["union"]["count"]
    hashes = []
    for artifact in artifacts:
        original = root / artifact["original_path"]
        stored = root / artifact["canonical_path"]
        payload = original.read_bytes()
        content_hash = sha_bytes(payload)
        assert content_hash == artifact["content_hash"] == stored.name
        assert stored.is_file() and sha_bytes(stored.read_bytes()) == content_hash
        hashes.append(content_hash)
        if original.suffix != ".json":
            continue
        parsed = json.loads(payload)
        assert artifact["json_parse_valid"] and isinstance(parsed, dict)
        if artifact["set"] in ("temporal_core_raw_receipt",
                                "temporal_core_quarantined_receipt"):
            version, hash_key = parsed.get("result_hash_version"), "result_sha256"
        else:
            version, hash_key = parsed.get("sha256_version"), "sha256"
        if version == "canonical_json_v2":
            assert artifact["canonical_hash_valid"]
            assert parsed[hash_key] == io.sha_obj({k: v for k, v in parsed.items() if k != hash_key})
            assert artifact["legacy_whole_file_sha256"] is None
        else:
            assert version is None
            assert artifact["canonical_hash_valid"] is None
            assert artifact["legacy_whole_file_sha256"] == content_hash
    assert merkle(hashes) == doc["union"]["merkle_root"]
    actual_proof = {p.relative_to(root).as_posix() for p in proof_root.rglob("*")
                    if p.is_file() and p.resolve() != fabric_path.resolve() and not _atomic_partial(p)}
    indexed_proof = {a["logical_id"] for a in artifacts if a["set"] == "temporal_core_proof"}
    assert actual_proof == indexed_proof
    runs = root / "runs" / "substrate" / io.PROGRAM
    receipt_paths = [p for p in runs.rglob("*.json") if p.is_file()
                     and "locks" not in p.relative_to(runs).parts and ".partial." not in p.name
                     and not p.name.endswith(".partial.json")]
    actual_raw = {p.relative_to(root).as_posix() for p in receipt_paths
                  if "quarantine" not in p.relative_to(runs).parts}
    actual_quarantine = {p.relative_to(root).as_posix() for p in receipt_paths
                         if "quarantine" in p.relative_to(runs).parts}
    indexed_raw = {a["logical_id"] for a in artifacts
                   if a["set"] == "temporal_core_raw_receipt"}
    indexed_quarantine = {a["logical_id"] for a in artifacts
                          if a["set"] == "temporal_core_quarantined_receipt"}
    assert actual_raw == indexed_raw
    assert actual_quarantine == indexed_quarantine
    assert doc["union"]["proof_count"] == len(actual_proof)
    assert doc["union"]["raw_receipt_count"] == len(actual_raw)
    assert doc["union"]["quarantined_receipt_count"] == len(actual_quarantine)
    inherited = doc.get("extends") or {}
    loaded = {}
    for name in ("integrated", "method"):
        binding = inherited.get(name)
        assert isinstance(binding, dict)
        path = root / binding["path"]
        payload = path.read_bytes()
        assert sha_bytes(payload) == binding["whole_file_sha256"]
        parent = json.loads(payload)
        assert isinstance(parent, dict) and isinstance(parent.get("union"), dict)
        assert parent["union"].get("count") == binding["count"]
        assert parent["union"].get("merkle_root") == binding["merkle_root"]
        parent_artifacts = parent.get("artifacts")
        if isinstance(parent_artifacts, list):
            assert all(isinstance(a, dict) and isinstance(a.get("content_hash"), str)
                       for a in parent_artifacts)
            assert len(parent_artifacts) == parent["union"]["count"]
            assert merkle([a["content_hash"] for a in parent_artifacts]) == parent["union"]["merkle_root"]
            assert binding.get("artifact_manifest_valid")
        if binding.get("embedded_sha256") is not None:
            assert parent.get("sha256") == binding["embedded_sha256"]
            assert parent["sha256"] == io.sha_obj({k: v for k, v in parent.items() if k != "sha256"})
            assert binding["embedded_sha256_valid"]
        loaded[name] = parent
    root_chain = inherited.get("root_chain") or {}
    if root_chain.get("method_extends_integrated_applicable"):
        declared = (loaded["method"].get("extends") or {}).get("integrated") or {}
        assert declared.get("count") == (loaded["integrated"].get("union") or {}).get("count")
        assert declared.get("merkle_root") == (loaded["integrated"].get("union") or {}).get("merkle_root")
        assert root_chain.get("method_extends_integrated_verified")
    if root_chain.get("binding_results_method_root_applicable"):
        binding_result = json.loads((proof_root / "MOP_TEMPORAL_CORE_BINDING_RESULTS.json").read_text())
        assert isinstance(binding_result, dict)
        assert binding_result.get("evidence_fabric_root") == (loaded["method"].get("union") or {}).get(
            "merkle_root")
        assert root_chain.get("binding_results_method_root_verified")
    return {"artifacts": len(artifacts), "raw_receipts": len(actual_raw),
            "quarantined_receipts": len(actual_quarantine), "proof_artifacts": len(actual_proof),
            "merkle_root": doc["union"]["merkle_root"]}


def clean_clone(science_snapshot_commit: str | None = None) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        commit = science_snapshot_commit or io.commit()
        clone = td + "/c"
        r = subprocess.run(["git", "clone", "--quiet", "--no-local", str(io.ROOT), clone],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"cloned": False, "commit": commit, "science_snapshot_commit": commit,
                    "error": r.stderr[-300:]}
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
 if d.get('sha256_version') == 'canonical_json_v2' and d['sha256'] != io.sha_obj({k:v for k,v in d.items() if k!='sha256'}): bad.append(str(p))
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
        supervisor_script = """
import json
from mop.temporal import io
from mop.temporal.runs.supervisor import status
s=status(); q=io.load('MOP_EXPERIMENT_VALUE_QUEUE.json'); licensed=q.get('licensed_top_two') or []
required={'scout','convergence','extended_convergence','principal','principal_corrections','convergence_corrections','optimization_corrections','third_bed_preflight'}
if 'E3_shared_versus_local' in licensed: required.add('e3')
if 'hybrid_adaptation' in licensed: required.add('hybrid')
assert not s['stop_switch_active'] and not s['active_shards']
assert all(not s['missing'][k] and not s['invalid'][k] and not s['partial_receipts'][k] for k in required)
print(json.dumps(s))
"""
        supervisor = run([sys.executable, "-c", supervisor_script])
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
            "cli_inspection": supervisor.returncode == 0 and 'mop-temporal-supervisor-status/v1' in supervisor.stdout,
            "clean_worktree": clean.returncode == 0 and not clean.stdout.strip(),
        }
        checks["all_pass"] = all(checks.values())
        return {"schema": "mop-temporal-core-clean-clone/v2", "cloned": True,
                "commit": commit, "science_snapshot_commit": commit, "checks": checks,
                "provenance_rule": ("this commit contains the sealed science snapshot; evidence fabric and "
                                    "terminal metadata may be committed only as descendants"),
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
