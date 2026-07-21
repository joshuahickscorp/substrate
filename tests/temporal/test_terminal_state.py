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
    items = synthesis.receipt_items(_common())
    assert len(items) == 4
    assert not any("locks" in key or "partial" in key for key in items)
    legacy_item = next(v for k, v in items.items() if k.endswith("legacy.json"))
    assert legacy_item["authority"] == synthesis.LEGACY_RECEIPT_AUTHORITY
    assert legacy_item["receipt_integrity"] == "legacy_outer_sha256"
    assert next(v for k, v in items.items() if k.endswith("canonical.json"))["status"] == "terminal"
    assert next(v for k, v in items.items() if k.endswith("forged.json"))["classification"] == "hash_mismatch"
    assert next(v for k, v in items.items() if k.endswith("malformed.json"))["classification"] == "invalid_json"
    assert all(isinstance(v["dependencies"], list) for v in items.values())


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
