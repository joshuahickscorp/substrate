import json

from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2, synthesis


def _common():
    return {"authority": "head", "bed": None, "factor_levels": None, "arm": None, "seed": None,
            "implementation": None, "parameter_count": None, "training_budget": None,
            "checkpoint": None, "tests": True, "verification": True, "mutations": True,
            "commit": "head", "tag": None}


def test_receipt_index_excludes_ephemeral_files_and_retains_invalid_evidence(monkeypatch, tmp_path):
    root, runs = tmp_path, tmp_path / "runs"
    stage, locks = runs / "stage", runs / "locks"
    stage.mkdir(parents=True)
    locks.mkdir()
    legacy = stage / "legacy.json"
    legacy.write_text(json.dumps({"bed": "b"}))
    canonical = {"result_hash_version": "canonical_json_v2", "seed": 1}
    canonical["result_sha256"] = io.sha_obj(canonical)
    (stage / "canonical.json").write_text(json.dumps(canonical))
    forged = dict(canonical, seed=2)
    (stage / "forged.json").write_text(json.dumps(forged))
    (stage / "malformed.json").write_text("{")
    (stage / ".work.partial.7.json").write_text("{}")
    (locks / "worker.json").write_text("{}")
    monkeypatch.setattr(io, "ROOT", root)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "PROOF", root / "proof")
    items = synthesis.receipt_items(_common())
    assert len(items) == 4
    assert not any("locks" in key or "partial" in key for key in items)
    legacy_item = next(v for k, v in items.items() if k.endswith("legacy.json"))
    assert legacy_item["authority"] is None
    assert legacy_item["status"] == "incomplete"
    assert legacy_item["receipt_integrity"] == "unauthorized_legacy_receipt"
    assert next(v for k, v in items.items() if k.endswith("canonical.json"))["status"] == "terminal"
    assert next(v for k, v in items.items() if k.endswith("forged.json"))["classification"] == "hash_mismatch"
    malformed_item = next(v for k, v in items.items() if k.endswith("malformed.json"))
    assert malformed_item["classification"] == "invalid_json"
    assert malformed_item["authority"] is None and malformed_item["custody_binding"] is None
    assert all(isinstance(v["dependencies"], list) for v in items.values())


def test_only_exactly_indexed_legacy_bytes_are_authorized(monkeypatch, tmp_path):
    runs, proof = tmp_path / "runs", tmp_path / "proof"
    stage = runs / "e2_principal"
    stage.mkdir(parents=True)
    proof.mkdir()
    receipt = stage / "bed_0.json"
    receipt.write_text(json.dumps({"bed": "bed", "seed": 0, "runs": []}))
    rel = "runs/e2_principal/bed_0.json"
    index = {"shard_index": [{"path": rel, "sha256": io.sha_file(receipt)}]}
    index["sha256"] = io.sha_obj(index)
    (proof / "MOP_E2_PRINCIPAL_RESULT.json").write_text(json.dumps(index))
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "PROOF", proof)
    item = synthesis.receipt_items(_common())[f"receipt:{rel}"]
    assert item["status"] == "terminal"
    assert item["receipt_integrity"] == "indexed_legacy_outer_sha256"
    receipt.write_text(json.dumps({"bed": "bed", "seed": 1, "runs": []}))
    item = synthesis.receipt_items(_common())[f"receipt:{rel}"]
    assert item["status"] == "incomplete"
    assert item["classification"] == "unauthorized_legacy_receipt"


def test_convergence_aggregate_authority_is_an_exact_dependency(monkeypatch, tmp_path):
    runs, proof = tmp_path / "runs", tmp_path / "proof"
    converge, principal = runs / "e2_converge", runs / "e2_principal"
    converge.mkdir(parents=True)
    principal.mkdir()
    proof.mkdir()
    authority = {"result_hash_version": "canonical_json_v2", "bed": "bed"}
    authority["result_sha256"] = io.sha_obj(authority)
    authority_path = converge / "converge_bed.json"
    authority_path.write_text(json.dumps(authority))
    rel = "runs/e2_converge/converge_bed.json"
    receipt = {"result_hash_version": "canonical_json_v2", "bed": "bed", "seed": 0,
               "convergence_authority": {"path": rel, "sha256": io.sha_file(authority_path)}}
    receipt["result_sha256"] = io.sha_obj(receipt)
    (principal / "bed_0.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "PROOF", proof)
    item = synthesis.receipt_items(_common())["receipt:runs/e2_principal/bed_0.json"]
    assert item["status"] == "terminal"
    assert item["dependencies"] == [f"receipt:{rel}"]
    assert item["dependency_bindings"][0]["bound"]
    receipt["convergence_authority"]["sha256"] = "0" * 64
    receipt["result_sha256"] = io.sha_obj({k: v for k, v in receipt.items() if k != "result_sha256"})
    (principal / "bed_0.json").write_text(json.dumps(receipt))
    item = synthesis.receipt_items(_common())["receipt:runs/e2_principal/bed_0.json"]
    assert item["status"] == "incomplete"
    assert item["classification"] == "dependency_hash_mismatch"


def test_principal_v2_cannot_omit_aggregate_authority(monkeypatch, tmp_path):
    runs, proof = tmp_path / "runs", tmp_path / "proof"
    stage = runs / "e2_principal"
    stage.mkdir(parents=True)
    proof.mkdir()
    receipt = {"schema": "mop-e2-principal-shard/v2",
               "result_hash_version": "canonical_json_v2", "bed": "bed", "seed": 0}
    receipt["result_sha256"] = io.sha_obj(receipt)
    (stage / "bed_0.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "PROOF", proof)
    item = synthesis.receipt_items(_common())["receipt:runs/e2_principal/bed_0.json"]
    assert item["status"] == "incomplete"
    assert item["classification"] == "missing_required_aggregate_dependency"


def test_convergence_v4_requires_exact_76_shard_dependency_rows():
    rows = [{"cell": Fx.cell_name(**spec), "path": f"runs/e2_converge/cshard_bed_{i}.json",
             "sha256": f"{i:064x}"} for i, spec in enumerate(e2.CONVERGE_CONFIGS)]
    doc = {"schema": "mop-e2-convergence/v4", "shard_index": rows}
    paths, hashes, shaped = synthesis._declared_dependencies(doc)
    assert shaped and len(paths) == len(hashes) == 76
    assert synthesis._required_dependencies_present(doc, paths)
    assert not synthesis._required_dependencies_present(
        {"schema": "mop-e2-convergence/v4", "shard_index": rows[:-1]}, paths[:-1])


def test_terminal_convergence_requires_every_selected_factorial_cell():
    cells = {Fx.cell_name(**spec): {"classification": "converged"}
             for spec in Fx.sweep_cells()["_all"]}
    doc = {"beds": ["bed"], "per_bed": {"bed": {"convergence": {
        "configs": cells, "grid": list(e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID)}}}}
    assert synthesis.convergence_is_terminal(doc)
    cells.pop(next(iter(cells)))
    assert not synthesis.convergence_is_terminal(doc)


def test_dependency_ready_is_not_an_alias_for_all_terminal():
    stages = {"authority": True, "science": False, "publish": False}
    deps = {"authority": [], "science": ["authority"], "publish": ["science"]}
    assert synthesis.ready_stages(stages, deps) == ["science"]
    stages["science"] = True
    assert synthesis.ready_stages(stages, deps) == ["publish"]
    stages["publish"] = True
    assert not synthesis.ready_stages(stages, deps)
