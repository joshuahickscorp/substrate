"""Test, coverage, resource and clean clone reports."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from mop.method import gate
from mop.temporal import io

ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": "src"}


def _env():
    import os
    env = dict(os.environ, **ENV)
    env["MOP_TEMPORAL_DATA_ROOT"] = str(io.DATA_ROOT)
    env["MOP_DATA_ROOT"] = str(io.DATA_ROOT)
    return env


def tests_and_coverage() -> tuple[dict, dict]:
    t0 = time.time()
    out = io.RUNS / "coverage"
    out.mkdir(parents=True, exist_ok=True)
    data = out / ".cov"
    r = subprocess.run([sys.executable, "-m", "coverage", "run", "--branch", f"--data-file={data}",
                        "--source=src/mop/temporal,src/mop/method", "-m", "pytest",
                        "tests/temporal", "tests/method", "-q"],
                       cwd=io.ROOT, env=_env(), capture_output=True, text=True)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "coverage.json"
        subprocess.run([sys.executable, "-m", "coverage", "json", f"--data-file={data}",
                        "-o", str(raw)], cwd=io.ROOT, env=_env(), capture_output=True)
        cov = json.loads(raw.read_text()) if raw.is_file() else {}
    io.run_json("coverage.json", cov, "coverage")
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
    from mop.temporal.runs.fabric import FABRIC_NAME, _atomic_partial, _json_binding, merkle, sha_bytes

    root = root.resolve()
    proof_root = root / "proof" / "substrate" / io.PROGRAM
    fabric_path = proof_root / FABRIC_NAME
    doc = json.loads(fabric_path.read_text())
    assert isinstance(doc, dict)
    assert doc.get("sha256_version") == "canonical_json_v2"
    assert doc.get("sha256") == io.sha_obj({k: v for k, v in doc.items() if k != "sha256"})
    artifacts = doc["artifacts"]
    assert isinstance(artifacts, list) and all(isinstance(a, dict) for a in artifacts)
    ids = [a["logical_id"] for a in artifacts]
    assert len(ids) == len(set(ids)) == doc["union"]["count"]
    hashes, unique, duplicate_bytes = [], {}, 0
    allowed_sets = {"temporal_core_proof", "temporal_core_raw_receipt",
                    "temporal_core_quarantined_receipt"}
    for artifact in artifacts:
        assert set(artifact) == {"logical_id", "original_path", "canonical_path", "content_hash",
                                 "bytes", "set", "is_null", "pack", "json_parse_valid",
                                 "hash_version", "canonical_hash_valid", "legacy_whole_file_sha256"}
        logical = Path(artifact["logical_id"])
        assert not logical.is_absolute() and ".." not in logical.parts
        assert artifact["original_path"] == artifact["logical_id"]
        assert artifact["set"] in allowed_sets and artifact["pack"] == "temporal-core-v1"
        original = root / artifact["original_path"]
        stored = root / artifact["canonical_path"]
        payload = original.read_bytes()
        content_hash = sha_bytes(payload)
        assert artifact["canonical_path"] == f"integrated/evidence_store/{content_hash}"
        assert content_hash == artifact["content_hash"] == stored.name
        assert stored.is_file() and sha_bytes(stored.read_bytes()) == content_hash
        assert artifact["bytes"] == len(payload)
        hashes.append(content_hash)
        if content_hash in unique:
            duplicate_bytes += len(payload)
        else:
            unique[content_hash] = len(payload)
        if original.suffix != ".json":
            assert artifact["is_null"] is False
            assert all(artifact[k] is None for k in (
                "json_parse_valid", "hash_version", "canonical_hash_valid",
                "legacy_whole_file_sha256"))
            continue
        expected_binding = _json_binding(payload, artifact["set"] != "temporal_core_proof")
        assert artifact["is_null"] == expected_binding.pop("contains_null_evidence")
        assert all(artifact[k] == v for k, v in expected_binding.items())
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            assert artifact["set"] == "temporal_core_quarantined_receipt"
            assert artifact["json_parse_valid"] is False
            continue
        if not isinstance(parsed, dict):
            assert artifact["set"] == "temporal_core_quarantined_receipt"
            assert artifact["json_parse_valid"] is False
            continue
        assert artifact["json_parse_valid"]
        if artifact["set"] == "temporal_core_quarantined_receipt":
            continue
        if artifact["set"] == "temporal_core_raw_receipt":
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
    union = doc["union"]
    assert union["count"] == len(artifacts)
    assert union["unique_objects"] == len(unique)
    assert union["unique_bytes"] == sum(unique.values())
    assert union["duplicate_bytes_eliminated"] == duplicate_bytes
    assert merkle(hashes) == union["merkle_root"]
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
    assert union["proof_count"] == len(actual_proof)
    assert union["raw_receipt_count"] == len(actual_raw)
    assert union["quarantined_receipt_count"] == len(actual_quarantine)
    inherited = doc.get("extends") or {}
    assert set(inherited) == {"integrated", "method", "root_chain"}
    loaded = {}
    parent_paths = {
        "integrated": "integrated/MOP_EVIDENCE_FABRIC.json",
        "method": ("proof/method/mop-experimental-method-reformation-v1/"
                   "MOP_METHOD_EVIDENCE_FABRIC.json"),
    }
    for name, expected_path in parent_paths.items():
        binding = inherited.get(name)
        assert isinstance(binding, dict)
        assert set(binding) == {"path", "whole_file_sha256", "count", "merkle_root",
                                "embedded_sha256", "embedded_sha256_valid",
                                "artifact_manifest_valid"}
        assert binding["path"] == expected_path
        path = root / expected_path
        payload = path.read_bytes()
        assert sha_bytes(payload) == binding["whole_file_sha256"]
        parent = json.loads(payload)
        assert isinstance(parent, dict) and isinstance(parent.get("union"), dict)
        assert parent["union"].get("count") == binding["count"]
        assert parent["union"].get("merkle_root") == binding["merkle_root"]
        parent_artifacts = parent.get("artifacts")
        assert isinstance(parent_artifacts, list)
        assert all(isinstance(a, dict) and isinstance(a.get("logical_id"), str)
                   and isinstance(a.get("content_hash"), str)
                   and len(a["content_hash"]) == 64
                   for a in parent_artifacts)
        parent_ids = [a["logical_id"] for a in parent_artifacts]
        assert len(parent_ids) == len(set(parent_ids))
        assert len(parent_artifacts) == parent["union"]["count"]
        assert merkle([a["content_hash"] for a in parent_artifacts]) == parent["union"]["merkle_root"]
        assert binding.get("artifact_manifest_valid") is True
        assert isinstance(binding.get("embedded_sha256"), str)
        assert parent.get("sha256") == binding["embedded_sha256"]
        assert parent["sha256"] == io.sha_obj({k: v for k, v in parent.items() if k != "sha256"})
        assert binding["embedded_sha256_valid"] is True
        loaded[name] = parent
    root_chain = inherited.get("root_chain") or {}
    assert root_chain.get("method_extends_integrated_applicable")
    declared = (loaded["method"].get("extends") or {}).get("integrated") or {}
    assert declared.get("count") == (loaded["integrated"].get("union") or {}).get("count")
    assert declared.get("merkle_root") == (loaded["integrated"].get("union") or {}).get("merkle_root")
    assert root_chain.get("method_extends_integrated_verified")
    assert root_chain.get("binding_results_method_root_applicable")
    binding_result = json.loads((proof_root / "MOP_TEMPORAL_CORE_BINDING_RESULTS.json").read_text())
    assert isinstance(binding_result, dict)
    assert binding_result.get("evidence_fabric_root") == (loaded["method"].get("union") or {}).get(
        "merkle_root")
    assert root_chain.get("binding_results_method_root_verified")
    assert root_chain == {
        "method_extends_integrated_applicable": True,
        "method_extends_integrated_verified": True,
        "binding_results_method_root_applicable": True,
        "binding_results_method_root_verified": True,
    }

    scientific = [a for a in artifacts if a["set"] != "temporal_core_quarantined_receipt"]
    expected_checks = {
        "exact_byte_recovery": True,
        "old_path_lookup": True,
        "no_hidden_unindexed_proof": len(actual_proof) == len(indexed_proof),
        "no_hidden_unindexed_receipts": actual_raw == indexed_raw,
        "quarantined_receipts_indexed_but_excluded_from_claims": (
            actual_quarantine == indexed_quarantine),
        "no_duplicate_identity": len(ids) == len(set(ids)),
        "all_scientific_json_parse_valid": all(
            a["json_parse_valid"] is not False for a in scientific),
        "canonical_hashes_valid": all(a["canonical_hash_valid"] is not False for a in scientific),
        "legacy_json_bound_by_whole_file_hash": all(
            a["legacy_whole_file_sha256"] == a["content_hash"] for a in scientific
            if a["json_parse_valid"] and a["hash_version"] is None),
        "nulls_indexed": True,
        "inherited_fabrics_untouched": True,
        "inherited_fabric_embedded_hashes_valid": True,
        "inherited_fabric_artifact_roots_valid": True,
        "inherited_fabric_root_chain_valid": True,
    }
    expected_checks["all_pass"] = all(expected_checks.values())
    assert doc["verification"] == expected_checks and expected_checks["all_pass"]

    expected_ids = actual_proof | actual_raw | actual_quarantine

    def manifest_valid(candidate):
        candidate_ids = [a.get("logical_id") for a in candidate]
        candidate_hashes = [a.get("content_hash") for a in candidate]
        return (len(candidate_ids) == len(set(candidate_ids))
                and set(candidate_ids) == expected_ids
                and all(isinstance(h, str) and len(h) == 64 for h in candidate_hashes)
                and merkle(candidate_hashes) == union["merkle_root"])

    changed = json.loads(json.dumps(artifacts))
    proof_id = next(a["logical_id"] for a in artifacts if a["set"] == "temporal_core_proof")
    next(a for a in changed if a["logical_id"] == proof_id)["content_hash"] = sha_bytes(b"tampered")
    missing = [a for a in artifacts if a["logical_id"] != proof_id]
    duplicate = artifacts + [json.loads(json.dumps(artifacts[0]))]
    null_ids = [a["logical_id"] for a in artifacts if a["is_null"]]
    omitted_null = [a for a in artifacts if not null_ids or a["logical_id"] != null_ids[0]]
    expected_mutations = {
        "mutated_proof_rejected": not manifest_valid(changed),
        "missing_proof_rejected": not manifest_valid(missing),
        "duplicate_identity_rejected": not manifest_valid(duplicate),
        "omitted_null_evidence_rejected": bool(null_ids) and not manifest_valid(omitted_null),
        "omitted_null_mutation_applied": bool(null_ids),
        "mutation_application": {
            "mutated_proof": any(a["content_hash"] != b["content_hash"]
                                 for a, b in zip(artifacts, changed)),
            "missing_proof": len(missing) < len(artifacts),
            "duplicate_identity": len(duplicate) > len(artifacts),
            "omitted_null_evidence": bool(null_ids) and len(omitted_null) < len(artifacts),
        },
    }
    expected_mutations["all_rejected"] = (
        all(expected_mutations[k] for k in ("mutated_proof_rejected", "missing_proof_rejected",
                                             "duplicate_identity_rejected",
                                             "omitted_null_evidence_rejected"))
        and all(expected_mutations["mutation_application"].values()))
    assert doc["mutations"] == expected_mutations and expected_mutations["all_rejected"]
    return {"artifacts": len(artifacts), "raw_receipts": len(actual_raw),
            "quarantined_receipts": len(actual_quarantine), "proof_artifacts": len(actual_proof),
            "merkle_root": doc["union"]["merkle_root"]}


def verify_core_checkpoints(root: Path) -> int:
    import re
    import torch
    from fastforge import engine as E
    from mop.temporal import arch as A, beds as B, factorial as Fx
    proof = root / "proof" / "substrate" / io.PROGRAM
    core = json.loads((proof / "MOP_OWNED_TEMPORAL_CORE_V1.json").read_text())
    body, selected = core.get("core") or {}, core.get("selected")
    declared = body.get("checkpoints") or {}
    checkpoint_root = proof / "checkpoints"
    paths = sorted(checkpoint_root.glob("*.pt"))
    if selected is False:
        reason = (core.get("selection") or {}).get("reason")
        assert isinstance(reason, str) and reason and not declared and not paths
        return 0
    assert selected is True and isinstance(declared, dict) and declared
    assert set(declared) == set(body.get("valid_domains") or [])
    expected = {(root / row["path"]).resolve() for row in declared.values()}
    assert expected == {path.resolve() for path in paths}
    selected_spec = dict(core["selection"]["selected"]["spec"])
    selected_cell = core["selection"]["selected"]["cell"]
    principal = json.loads((proof / "MOP_E2_PRINCIPAL_RESULT.json").read_text())
    source_commit, source_tree = core.get("source_commit"), core.get("source_tree_oid")
    assert all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
               for value in (source_commit, source_tree))
    if str(selected_spec.get("history_k", "")).isdigit():
        selected_spec["history_k"] = int(selected_spec["history_k"])
    for bed, row in declared.items():
        path = (root / row["path"]).resolve()
        assert path.is_relative_to(checkpoint_root.resolve()) and io.sha_file(path) == row["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["schema"] == "mop-owned-temporal-core-checkpoint/v1"
        assert (payload["bed"] == bed and payload["seed"] == 0 and payload["spec"] == selected_spec
                and payload.get("source_commit") == source_commit
                and payload.get("source_tree_oid") == source_tree)
        model = Fx.build_cell(B.splits(bed, 0), seed=0, **selected_spec)[0]
        model.load_state_dict(payload["state_dict"], strict=True)
        receipt = payload["training_receipt"]
        expected_update = principal["per_bed"][bed]["convergence"]["configs"][selected_cell][
            "selected_checkpoint"]
        assert E.checkpoint_sha(model) == receipt["checkpoint_sha_after"] == row["checkpoint_sha"]
        assert payload["params"] == row["params"] == A.count(model)
        assert (payload["selected_checkpoint"] == row["selected_checkpoint"]
                == receipt["updates"] == expected_update)
    return len(paths)


def clean_clone(science_snapshot_commit: str | None = None, *, require_fabric: bool = False,
                terminal_evidence_commit: str | None = None) -> dict:
    """Verify either the committed science snapshot or a later terminal evidence commit.

    The first phase deliberately does not require a fabric that cannot yet include its clean-clone
    report.  The second phase checks out a descendant containing that report and the fabric, then
    requires an exact content-store lookup through :func:`verify_fabric_tree`.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        snapshot = science_snapshot_commit or io.commit()
        target = (terminal_evidence_commit or io.commit()) if require_fabric else snapshot
        clone = td + "/c"
        r = subprocess.run(["git", "clone", "--quiet", "--no-local", str(io.ROOT), clone],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"cloned": False, "commit": snapshot, "science_snapshot_commit": snapshot,
                    "terminal_evidence_commit": target if require_fabric else None,
                    "error": r.stderr[-300:]}
        checkout = subprocess.run(["git", "checkout", "--quiet", target], cwd=clone,
                                  capture_output=True, text=True)
        env = _env()

        def run(args):
            return subprocess.run(args, cwd=clone, env=env, capture_output=True, text=True)

        head = run(["git", "rev-parse", "HEAD"])
        ancestry = run(["git", "merge-base", "--is-ancestor", snapshot, target]) if require_fabric else None
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
from mop.temporal.runs.reports import verify_core_checkpoints
print(verify_core_checkpoints(Path.cwd()))
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
        fabric_lookup = run([sys.executable, "-c", (
            "import json; from pathlib import Path; "
            "from mop.temporal.runs.reports import verify_fabric_tree; "
            "print(json.dumps(verify_fabric_tree(Path.cwd()), sort_keys=True))")]) if require_fabric else None
        clean = run(["git", "status", "--porcelain"])
        checks = {
            "exact_commit_checkout": checkout.returncode == 0 and head.stdout.strip() == target,
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
        if require_fabric:
            checks["terminal_evidence_fabric_lookup"] = fabric_lookup.returncode == 0
            checks["terminal_evidence_descends_from_science_snapshot"] = ancestry.returncode == 0
        checks["all_pass"] = all(checks.values())
        return {"schema": "mop-temporal-core-clean-clone/v2", "cloned": True,
                "phase": "terminal_evidence" if require_fabric else "science_snapshot",
                "commit": snapshot, "science_snapshot_commit": snapshot,
                "terminal_evidence_commit": target if require_fabric else None,
                "validated_commit": target, "checks": checks,
                "provenance_rule": ("the science snapshot is immutable; terminal evidence and metadata must "
                                    "be committed descendants, and terminal status requires a clean-clone "
                                    "content-store lookup of the committed evidence fabric"),
                "method_tail": (method.stdout or method.stderr).strip().splitlines()[-3:],
                "temporal_tail": (temporal.stdout or temporal.stderr).strip().splitlines()[-3:],
                "proof_artifacts_present": int(proof.stdout.strip() or 0) if proof.returncode == 0 else 0,
                "checkpoints_restored": int(checkpoint.stdout.strip() or 0) if checkpoint.returncode == 0 else 0,
                "exact_skips": [], "offline_installation_claimed": False,
                "all_pass": checks["all_pass"]}


def terminal_clone_main() -> None:
    _require_clean_head()
    prior = io.load("MOP_TEMPORAL_CORE_CLEAN_CLONE.json")
    snapshot = prior.get("science_snapshot_commit")
    if not isinstance(snapshot, str) or len(snapshot) != 40:
        raise RuntimeError("terminal clone requires a sealed science-snapshot clean-clone report")
    report = clean_clone(snapshot, require_fabric=True, terminal_evidence_commit=io.commit())
    io.seal("MOP_TEMPORAL_CORE_CLEAN_CLONE.json", report)
    if not report.get("all_pass"):
        raise RuntimeError("terminal evidence clean-clone verification is red")
    print("TERMINAL_CLONE_DONE", flush=True)


def _require_clean_head() -> None:
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=io.ROOT, capture_output=True, text=True)
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("clean-clone phase requires all science and evidence bytes committed at HEAD")


def science_clone_main() -> None:
    _require_clean_head()
    report = clean_clone(io.commit())
    io.seal("MOP_TEMPORAL_CORE_CLEAN_CLONE.json", report)
    if not report.get("all_pass"):
        raise RuntimeError("science snapshot clean-clone verification is red")
    print("SCIENCE_CLONE_DONE", flush=True)


def main():
    t0 = time.time()
    tr, cr = tests_and_coverage()
    io.seal("MOP_TEMPORAL_CORE_TEST_REPORT.json", tr)
    io.seal("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json", cr)
    io.seal("MOP_TEMPORAL_CORE_RESOURCE_REPORT.json", resource_report())
    print(f"reports: tests {tr['passed']}, kernel {cr['method_kernel']['statement']}/"
          f"{cr['method_kernel']['branch']}, critical {cr['active_critical_path']['statement']}/"
          f"{cr['active_critical_path']['branch']} in {round(time.time() - t0, 1)}s", flush=True)
    print("REPORTS_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["terminal_clone"]:
        terminal_clone_main()
    elif sys.argv[1:] == ["science_clone"]:
        science_clone_main()
    else:
        main()
