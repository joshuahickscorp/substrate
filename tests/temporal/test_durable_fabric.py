import json

import pytest

from mop.temporal import io
from mop.temporal.runs import fabric, reports, synthesis


def _canonical(payload, receipt=False):
    doc = dict(payload)
    version = "result_hash_version" if receipt else "sha256_version"
    digest = "result_sha256" if receipt else "sha256"
    doc[version] = "canonical_json_v2"
    doc[digest] = io.sha_obj(doc)
    return doc


def _legacy_hashed(payload):
    doc = dict(payload)
    doc["sha256"] = io.sha_obj(doc)
    return doc


def _build_fabric(monkeypatch, tmp_path):
    proof = tmp_path / "proof" / "substrate" / io.PROGRAM
    runs = tmp_path / "runs" / "substrate" / io.PROGRAM
    store = tmp_path / "integrated" / "evidence_store"
    inherited = tmp_path / "integrated" / "MOP_EVIDENCE_FABRIC.json"
    method = (tmp_path / "proof" / "method" / "mop-experimental-method-reformation-v1"
              / "MOP_METHOD_EVIDENCE_FABRIC.json")
    proof.mkdir(parents=True)
    (runs / "stage").mkdir(parents=True)
    (runs / "quarantine" / "incident").mkdir(parents=True)
    inherited.parent.mkdir(parents=True)
    method.parent.mkdir(parents=True)
    inherited_root = fabric.merkle(["1" * 64])
    method_root = fabric.merkle(["2" * 64])
    inherited.write_text(json.dumps(_legacy_hashed({
        "artifacts": [{"logical_id": "integrated-parent", "content_hash": "1" * 64}],
        "union": {"count": 1, "merkle_root": inherited_root}})))
    method.write_text(json.dumps(_legacy_hashed({
        "extends": {"integrated": {"count": 1, "merkle_root": inherited_root}},
        "artifacts": [{"logical_id": "method-parent", "content_hash": "2" * 64}],
        "union": {"count": 1, "merkle_root": method_root}})))
    (proof / "MOP_TEMPORAL_CORE_BINDING_RESULTS.json").write_text(json.dumps(_canonical({
        "evidence_fabric_root": method_root})))
    (proof / "claim.json").write_text(json.dumps(_canonical({"claim": None})))
    (proof / ".claim.json.partial.55").write_text("partial")
    (runs / "stage" / "raw.json").write_text(json.dumps(_canonical({"bed": "a"}, receipt=True)))
    (runs / "quarantine" / "incident" / "bad.json").write_text(json.dumps(
        _canonical({"classification": "identity_mismatch"}, receipt=True)))
    (runs / "quarantine" / "incident" / "malformed.json").write_text("{")
    (runs / "quarantine" / "incident" / "future.json").write_text(json.dumps({
        "classification": "unknown_hash_version", "result_hash_version": "future",
        "result_sha256": "0" * 64}))
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "PROOF", proof)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "commit", lambda: "a" * 40)
    monkeypatch.setattr(fabric, "STORE", store)
    monkeypatch.setattr(fabric, "INHERITED", inherited)
    monkeypatch.setattr(fabric, "METHOD", method)
    fabric.main()
    return proof, runs, inherited, method


def test_fabric_recomputes_exact_paths_self_hash_inheritance_and_quarantine(monkeypatch, tmp_path):
    proof, _, inherited, _ = _build_fabric(monkeypatch, tmp_path)
    checked = reports.verify_fabric_tree(tmp_path)
    assert checked["raw_receipts"] == 1
    assert checked["quarantined_receipts"] == 3
    doc_path = proof / fabric.FABRIC_NAME
    doc = json.loads(doc_path.read_text())
    quarantine = [a for a in doc["artifacts"]
                  if a["set"] == "temporal_core_quarantined_receipt"]
    assert len(quarantine) == 3
    assert any(a["json_parse_valid"] is False for a in quarantine)
    assert any(a["canonical_hash_valid"] is False for a in quarantine)
    assert all(doc["mutations"]["mutation_application"].values())
    assert doc["extends"]["integrated"]["whole_file_sha256"] == io.sha_file(inherited)
    extra = proof / "unindexed.txt"
    extra.write_text("not in manifest")
    with pytest.raises(AssertionError):
        reports.verify_fabric_tree(tmp_path)
    extra.unlink()
    original = inherited.read_bytes()
    inherited.write_bytes(original + b"\n")
    with pytest.raises(AssertionError):
        reports.verify_fabric_tree(tmp_path)
    inherited.write_bytes(original)
    doc["mutations"]["mutation_application"]["mutated_proof"] = False
    doc["mutations"]["all_rejected"] = True
    doc["sha256"] = io.sha_obj({k: v for k, v in doc.items() if k != "sha256"})
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        reports.verify_fabric_tree(tmp_path)


def test_malformed_json_and_unbound_receipt_dependencies_are_rejected(monkeypatch, tmp_path):
    assert not fabric._json_binding(b"[]", receipt=True)["json_parse_valid"]
    runs = tmp_path / "runs"
    stage = runs / "stage"
    stage.mkdir(parents=True)
    base = stage / "base.json"
    base.write_text(json.dumps(_canonical({"seed": 1}, receipt=True)))
    extension = _canonical({
        "extends": {"path": "runs/stage/base.json"}, "seed": 1}, receipt=True)
    (stage / "missing_hash.json").write_text(json.dumps(extension))
    (stage / "top_level_list.json").write_text("[]")
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", runs)
    items = synthesis.receipt_items({
        "authority": "head", "commit": "head", "bed": None, "factor_levels": None,
        "arm": None, "seed": None, "implementation": None, "parameter_count": None,
        "training_budget": None, "checkpoint": None, "tests": True,
        "verification": True, "mutations": True, "tag": None})
    missing = items["receipt:runs/stage/missing_hash.json"]
    malformed = items["receipt:runs/stage/top_level_list.json"]
    assert missing["status"] == "incomplete"
    assert missing["classification"] == "dependency_hash_mismatch"
    assert missing["dependency_bindings"] == [{
        "item_id": "receipt:runs/stage/base.json", "path": "runs/stage/base.json",
        "sha256": None, "bound": False}]
    assert malformed["status"] == "incomplete" and malformed["classification"] == "invalid_json"


def test_terminal_dag_separates_snapshot_fabric_and_metadata():
    dependencies = synthesis.stage_dependencies()
    assert all(stage not in parents for stage, parents in dependencies.items())
    visited = set()

    def visit(stage, active):
        assert stage not in active
        if stage not in visited:
            for parent in dependencies[stage]:
                visit(parent, active | {stage})
            visited.add(stage)

    for stage in dependencies:
        visit(stage, set())
    preclone = set(synthesis.DELIVERABLE_STAGE) - synthesis.POST_SNAPSHOT_DELIVERABLES
    assert all("stage:preclone_deliverables" not in synthesis.deliverable_dependencies(name)
               for name in preclone)
    assert synthesis.deliverable_dependencies("MOP_TEMPORAL_CORE_CLEAN_CLONE.json") == [
        "stage:evidence_fabric"]
    assert synthesis.deliverable_dependencies("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json") == [
        "stage:preclone_deliverables"]
    assert synthesis.deliverable_dependencies("MOP_TEMPORAL_CORE_STATE.json") == [
        "stage:clean_clone"]
    assert synthesis.science_snapshot_binding(
        {"commit": "a" * 40, "science_snapshot_commit": "a" * 40}, "b" * 40,
        ancestor=True)["relationship"] == "terminal_metadata_descendant"
    assert not synthesis.science_snapshot_binding(
        {"commit": "a" * 40, "science_snapshot_commit": "c" * 40}, "b" * 40,
        ancestor=True)["bound"]
    terminal = {
        "schema": "mop-temporal-core-clean-clone/v2", "phase": "terminal_evidence",
        "commit": "a" * 40, "science_snapshot_commit": "a" * 40,
        "terminal_evidence_commit": "b" * 40, "validated_commit": "b" * 40,
        "checks": {"terminal_evidence_fabric_lookup": True,
                   "terminal_evidence_descends_from_science_snapshot": True, "all_pass": True},
        "all_pass": True,
    }
    assert synthesis.terminal_evidence_binding(
        terminal, "c" * 40,
        ancestor={"snapshot_to_evidence": True, "evidence_to_final": True})["bound"]
    terminal["checks"]["terminal_evidence_fabric_lookup"] = False
    assert not synthesis.terminal_evidence_binding(
        terminal, "c" * 40,
        ancestor={"snapshot_to_evidence": True, "evidence_to_final": True})["bound"]


def test_terminal_clean_clone_requires_independent_fabric_lookup(monkeypatch, tmp_path):
    snapshot, terminal = "a" * 40, "b" * 40
    fail_lookup = False

    def fake_run(args, **kwargs):
        nonlocal fail_lookup
        joined = " ".join(str(a) for a in args)
        stdout, returncode = "", 0
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = terminal + "\n"
        elif "verify_fabric_tree" in joined:
            stdout = '{"artifacts": 4}\n'
            returncode = int(fail_lookup)
        elif "root=Path('proof/substrate')" in joined or "paths=sorted" in joined:
            stdout = "1\n"
        elif "from mop.temporal.runs.supervisor import status" in joined:
            stdout = '{"schema":"mop-temporal-supervisor-status/v1"}\n'
        return reports.subprocess.CompletedProcess(args, returncode, stdout, "fabric lookup failed")

    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(reports.subprocess, "run", fake_run)
    result = reports.clean_clone(snapshot, require_fabric=True, terminal_evidence_commit=terminal)
    assert result["phase"] == "terminal_evidence"
    assert result["commit"] == result["science_snapshot_commit"] == snapshot
    assert result["validated_commit"] == result["terminal_evidence_commit"] == terminal
    assert result["checks"]["terminal_evidence_fabric_lookup"] and result["all_pass"]
    fail_lookup = True
    result = reports.clean_clone(snapshot, require_fabric=True, terminal_evidence_commit=terminal)
    assert not result["checks"]["terminal_evidence_fabric_lookup"]
    assert not result["all_pass"]
