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
        "artifacts": [{"content_hash": "1" * 64}],
        "union": {"count": 1, "merkle_root": inherited_root}})))
    method.write_text(json.dumps(_legacy_hashed({
        "extends": {"integrated": {"count": 1, "merkle_root": inherited_root}},
        "artifacts": [{"content_hash": "2" * 64}],
        "union": {"count": 1, "merkle_root": method_root}})))
    (proof / "MOP_TEMPORAL_CORE_BINDING_RESULTS.json").write_text(json.dumps(_canonical({
        "evidence_fabric_root": method_root})))
    (proof / "claim.json").write_text(json.dumps(_canonical({"claim": None})))
    (proof / ".claim.json.partial.55").write_text("partial")
    (runs / "stage" / "raw.json").write_text(json.dumps(_canonical({"bed": "a"}, receipt=True)))
    (runs / "quarantine" / "incident" / "bad.json").write_text(json.dumps(
        _canonical({"classification": "identity_mismatch"}, receipt=True)))
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
    assert checked["quarantined_receipts"] == 1
    doc_path = proof / fabric.FABRIC_NAME
    doc = json.loads(doc_path.read_text())
    quarantine = [a for a in doc["artifacts"]
                  if a["set"] == "temporal_core_quarantined_receipt"]
    assert len(quarantine) == 1
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
    doc["unbound_field"] = True
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
        "stage:preclone_deliverables"]
    assert synthesis.deliverable_dependencies("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json") == [
        "stage:clean_clone"]
    assert synthesis.science_snapshot_binding(
        {"commit": "a" * 40, "science_snapshot_commit": "a" * 40}, "b" * 40,
        ancestor=True)["relationship"] == "terminal_metadata_descendant"
    assert not synthesis.science_snapshot_binding(
        {"commit": "a" * 40, "science_snapshot_commit": "c" * 40}, "b" * 40,
        ancestor=True)["bound"]
