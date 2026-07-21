import json

import pytest

from mop.temporal import io
from mop.temporal.runs import fabric, reports, supervisor


def _canonical_doc(payload, receipt=False):
    doc = dict(payload)
    version = "result_hash_version" if receipt else "sha256_version"
    digest = "result_sha256" if receipt else "sha256"
    doc[version] = "canonical_json_v2"
    doc[digest] = io.sha_obj(doc)
    return doc


def test_fabric_indexes_and_binds_proof_and_raw_receipts(monkeypatch, tmp_path):
    proof = tmp_path / "proof" / "substrate" / io.PROGRAM
    runs = tmp_path / "runs" / "substrate" / io.PROGRAM
    store = tmp_path / "integrated" / "evidence_store"
    inherited = tmp_path / "integrated" / "MOP_EVIDENCE_FABRIC.json"
    method = tmp_path / "proof" / "method" / "mop-experimental-method-reformation-v1" / "MOP_METHOD_EVIDENCE_FABRIC.json"
    proof.mkdir(parents=True)
    (runs / "stage").mkdir(parents=True)
    (runs / "locks").mkdir()
    inherited.parent.mkdir(parents=True)
    method.parent.mkdir(parents=True)
    inherited.write_text(json.dumps({"union": {"count": 2}}))
    method.write_text(json.dumps({"union": {"count": 3, "merkle_root": "method"}}))
    (proof / "proof.json").write_text(json.dumps(_canonical_doc({"claim": None})))
    (runs / "stage" / "canonical.json").write_text(json.dumps(
        _canonical_doc({"bed": "har_stream"}, receipt=True)))
    (runs / "stage" / "legacy.json").write_text(json.dumps({"bed": "speech_stream"}))
    (runs / "locks" / "worker.json").write_text(json.dumps({"pid": 1}))
    (runs / "stage" / ".work.partial.7.json").write_text(json.dumps({"partial": True}))

    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "PROOF", proof)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "commit", lambda: "commit")
    monkeypatch.setattr(fabric, "STORE", store)
    monkeypatch.setattr(fabric, "INHERITED", inherited)
    monkeypatch.setattr(fabric, "METHOD", method)
    fabric.main()

    doc = json.loads((proof / "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json").read_text())
    assert doc["verification"]["all_pass"]
    assert doc["mutations"]["all_rejected"] and doc["mutations"]["omitted_null_mutation_applied"]
    assert doc["union"]["proof_count"] == 1 and doc["union"]["raw_receipt_count"] == 2
    raw = [a for a in doc["artifacts"] if a["set"] == "temporal_core_raw_receipt"]
    assert {a["original_path"].rsplit("/", 1)[-1] for a in raw} == {"canonical.json", "legacy.json"}
    legacy = next(a for a in raw if a["original_path"].endswith("legacy.json"))
    assert legacy["legacy_whole_file_sha256"] == legacy["content_hash"]
    assert reports.verify_fabric_tree(tmp_path)["raw_receipts"] == 2
    stored = tmp_path / doc["artifacts"][0]["canonical_path"]
    original_store_bytes = stored.read_bytes()
    stored.write_bytes(b"tampered")
    with pytest.raises(AssertionError):
        reports.verify_fabric_tree(tmp_path)
    stored.write_bytes(original_store_bytes)
    fabric_path = proof / "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"
    sealed_fabric = json.loads(fabric_path.read_text())
    sealed_fabric["union"]["merkle_root"] = "0" * 64
    fabric_path.write_text(json.dumps(sealed_fabric))
    with pytest.raises(AssertionError):
        reports.verify_fabric_tree(tmp_path)


def test_supervisor_status_exposes_all_correction_receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor, "workers", lambda: 0)
    status = supervisor.status()
    assert status["completed"]["principal_corrections"] == 0
    assert status["completed"]["convergence_corrections"] == 0
    assert len(status["missing"]["principal_corrections"]) == 24
    assert len(status["missing"]["convergence_corrections"]) == 3
    assert set(status["invalid"]) == set(status["missing"]) == set(status["partial_receipts"])


def test_supervisor_rejects_unknown_receipt_hash_versions(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "future.json").write_text(json.dumps({"result_hash_version": "future"}))
    assert supervisor.invalid("stage", ["future"]) == ["future"]
