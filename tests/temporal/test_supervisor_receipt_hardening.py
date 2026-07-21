import json
import os
import subprocess

from mop.temporal import io
from mop.temporal import factorial as Fx
from mop.temporal.runs import corrections, e2, supervisor


def canonical(payload):
    doc = dict(payload, program=io.PROGRAM, source_commit="a" * 40,
               source_tree_oid="b" * 40, result_hash_version="canonical_json_v2")
    doc["result_sha256"] = io.sha_obj(doc)
    return doc


def write_receipt(root, sub, name, doc):
    stage = root / sub
    stage.mkdir(parents=True, exist_ok=True)
    (stage / f"{name}.json").write_text(json.dumps(doc))


def test_canonical_receipts_require_immutable_commit_and_tree_binding():
    valid = canonical({"value": 1})
    assert supervisor._provenance_valid(valid)
    missing_tree = dict(valid)
    missing_tree.pop("source_tree_oid")
    missing_tree["result_sha256"] = io.sha_obj(
        {k: v for k, v in missing_tree.items() if k != "result_sha256"})
    assert not supervisor._provenance_valid(missing_tree)


def test_launch_environment_and_lock_bind_the_same_clean_source_tree(monkeypatch, tmp_path):
    commit, tree = "c" * 40, "d" * 40

    def git(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, tree + "\n", "")

    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "commit", lambda: commit)
    monkeypatch.setattr(supervisor.subprocess, "run", git)
    env, binding = supervisor.launch_environment()
    assert binding == {"source_commit": commit, "source_tree_oid": tree}
    assert env["MOP_LAUNCH_SOURCE_COMMIT"] == commit
    assert env["MOP_LAUNCH_SOURCE_TREE_OID"] == tree

    locks, logs = tmp_path / "locks", tmp_path / "logs"
    monkeypatch.setattr(supervisor, "LOCKS", locks)
    monkeypatch.setattr(supervisor, "LOGS", logs)
    monkeypatch.setattr(supervisor, "launch_environment", lambda: (env, binding))
    seen = {}

    class Child:
        pid = 12345

    def popen(args, **kwargs):
        seen.update(kwargs)
        return Child()

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    assert supervisor.launch(["principal", "har_stream", "0"], "worker.log", "p:har_stream_0")
    lock = json.loads((locks / "p_har_stream_0.json").read_text())
    assert lock["source_commit"] == commit and lock["source_tree_oid"] == tree
    assert seen["env"]["MOP_LAUNCH_SOURCE_COMMIT"] == commit


def test_all_scientific_stages_reject_empty_canonical_objects(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "RUNS", runs)
    cases = {
        "e2_scout": "shard_har_stream_0",
        "e2_principal": "har_stream_0",
        "e2_converge": "cshard_har_stream_0",
        "e2_converge_extended": "xshard_har_stream_0",
        "e2_principal_corrections": "capacity_har_stream_0",
        "e2_converge_corrections": "convergence_har_stream",
        "e2_optimization_corrections": "optimization_har_stream_small",
        "e3": "har_stream_to_speech_stream_0",
        "third_bed_preflight": "harth_preflight_0",
        "hybrid": "har_stream_0",
    }
    for sub, name in cases.items():
        write_receipt(runs, sub, name, canonical({}))
        assert supervisor.invalid(sub, [name]) == [name]


def test_extended_convergence_dependency_is_exact_and_cannot_escape_data_root(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "RUNS", runs)
    idx, bed = 0, "har_stream"
    spec = e2.CONVERGE_CONFIGS[idx]
    def raw(grid):
        return {"seed_scores": {str(k): [0.5] * len(e2.CONVERGENCE_SEEDS) for k in grid},
                "arm_records": {str(k): [{"seed": seed, "updates": k, "score": 0.5,
                    "checkpoint_sha": "f" * 64} for seed in e2.CONVERGENCE_SEEDS] for k in grid}}
    base = canonical({"bed": bed, "spec": spec, "cell": e2.Fx.cell_name(**spec),
                      "curve": {str(k): 0.5 for k in e2.CONVERGENCE_GRID},
                      "seeds": list(e2.CONVERGENCE_SEEDS), "selected_checkpoint": e2.CONVERGENCE_GRID[0],
                      "converged": True, **raw(e2.CONVERGENCE_GRID)})
    write_receipt(runs, "e2_converge", f"cshard_{bed}_{idx}", base)
    base_path = runs / "e2_converge" / f"cshard_{bed}_{idx}.json"
    extension = canonical({"bed": bed, "spec": spec, "cell": e2.Fx.cell_name(**spec),
        "curve": {str(k): 0.5 for k in e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID},
        "seeds": list(e2.CONVERGENCE_SEEDS), "selected_checkpoint": e2.CONVERGENCE_GRID[0],
        "converged": True, "extends": {"path": base_path.relative_to(tmp_path).as_posix(),
        "sha256": io.sha_file(base_path), "grid": list(e2.CONVERGENCE_GRID)},
        **raw(e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID)})
    name = f"xshard_{bed}_{idx}"
    write_receipt(runs, "e2_converge_extended", name, extension)
    assert supervisor.invalid("e2_converge_extended", [name]) == []
    extension["extends"]["path"] = "/tmp/untrusted.json"
    extension["result_sha256"] = io.sha_obj(
        {k: v for k, v in extension.items() if k != "result_sha256"})
    write_receipt(runs, "e2_converge_extended", name, extension)
    assert supervisor.invalid("e2_converge_extended", [name]) == [name]


def test_principal_and_correction_budgets_bind_exact_selected_convergence(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "RUNS", runs_root)
    bed, seed, budget = "har_stream", 0, int(e2.CONVERGENCE_GRID[0])
    cells = [Fx.cell_name(**spec) for spec in Fx.sweep_cells()["_all"]]
    convergence = canonical({"schema": "mop-e2-convergence/v4", "bed": bed,
        "shard_index": [{"cell": cell} for cell in cells], "configs": {
        cell: {"selected_checkpoint": budget} for cell in cells}})
    write_receipt(runs_root, "e2_converge", f"converge_{bed}", convergence)
    path = runs_root / "e2_converge" / f"converge_{bed}.json"

    def bound_doc(selected_cells, schema):
        return canonical({"schema": schema, "bed": bed, "seed": seed,
            "n_cells": len(selected_cells),
            "runs": [{"bed": bed, "seed": seed, "cell": cell,
                      "steps": budget, "updates": budget} for cell in selected_cells],
            "convergence_authority": {"path": path.relative_to(tmp_path).as_posix(),
                "sha256": io.sha_file(path),
                "selected_checkpoints": {cell: budget for cell in selected_cells}}})

    principal = bound_doc(cells, "mop-e2-principal-shard/v2")
    write_receipt(runs_root, "e2_principal", f"{bed}_{seed}", principal)
    assert supervisor.invalid("e2_principal", [f"{bed}_{seed}"]) == []
    principal["runs"][0]["steps"] = budget + 1
    principal["result_sha256"] = io.sha_obj(
        {k: v for k, v in principal.items() if k != "result_sha256"})
    write_receipt(runs_root, "e2_principal", f"{bed}_{seed}", principal)
    assert supervisor.invalid("e2_principal", [f"{bed}_{seed}"]) == [f"{bed}_{seed}"]

    correction_cells = [Fx.cell_name(**spec) for spec in corrections.SPECS]
    correction = bound_doc(correction_cells, "mop-e2-capacity-tier-correction-shard/v2")
    name = f"capacity_{bed}_{seed}"
    write_receipt(runs_root, "e2_principal_corrections", name, correction)
    assert supervisor.invalid("e2_principal_corrections", [name]) == []
    correction["convergence_authority"]["path"] = "/tmp/forged.json"
    correction["result_sha256"] = io.sha_obj(
        {k: v for k, v in correction.items() if k != "result_sha256"})
    write_receipt(runs_root, "e2_principal_corrections", name, correction)
    assert supervisor.invalid("e2_principal_corrections", [name]) == [name]


def test_legacy_path_requires_exact_hash_checked_normalization_authority(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "RUNS", runs)
    legacy = {"schema": "mop-temporal-orchestration-incident/v1",
              "source_commit": supervisor.LEGACY_AUTHORITY,
              "classification":
              "legacy_pre_persistence_object_hash_not_recomputable_after_json_normalization",
              "compatibility": "legacy receipts accepted only with whole-file hashes"}
    legacy["result_sha256"] = io.sha_obj(legacy)
    write_receipt(runs, "orchestration", supervisor.LEGACY_BINDING.removesuffix(".json"), legacy)
    write_receipt(runs, "stage", "legacy", {"parseable": True})
    assert supervisor.invalid("stage", ["legacy"]) == []
    legacy["compatibility"] = "unbound"
    write_receipt(runs, "orchestration", supervisor.LEGACY_BINDING.removesuffix(".json"), legacy)
    assert supervisor.invalid("stage", ["legacy"]) == ["legacy"]


def test_stale_partial_is_moved_append_only_with_bytes_and_incident(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "RUNS", runs)
    monkeypatch.setattr(supervisor, "_pid_alive", lambda pid: False)
    stage = runs / "e2_principal"
    stage.mkdir(parents=True)
    partial = stage / ".har_stream_0.json.partial.999999"
    partial.write_bytes(b"unfinished but preserved")
    os.utime(partial, (1, 1))
    assert supervisor.quarantine_stale_partials("e2_principal", stale_after=0) == [partial.name]
    assert not partial.exists()
    preserved = list((runs / "quarantine" / "stale_partials").iterdir())
    assert len(preserved) == 1 and preserved[0].read_bytes() == b"unfinished but preserved"
    incidents = list((runs / "orchestration").glob("quarantine_stale_partials_*.json"))
    assert len(incidents) == 1
    incident = json.loads(incidents[0].read_text())
    assert incident["preservation"].startswith("append only")


def test_invalid_receipt_recovery_preserves_original_bytes(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "RUNS", runs)
    source = runs / "e2_principal" / "har_stream_0.json"
    source.parent.mkdir(parents=True)
    original = b'{"old_fixed_budget":1200}'
    source.write_bytes(original)
    assert supervisor.recoverable_pending("e2_principal", ["har_stream_0"]) == ["har_stream_0"]
    assert not source.exists()
    quarantined = list((runs / "quarantine" / "invalid_receipts").iterdir())
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == original


def test_live_state_reconciliation_is_restartable_and_never_activates(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor, "workers", lambda: 0)
    monkeypatch.setattr(supervisor.io, "exists", lambda name: False)
    sealed = {}
    monkeypatch.setattr(supervisor.io, "seal", lambda name, doc: sealed.update({name: doc}))
    doc = supervisor.reconcile_live_state("test", supervisor.status())
    assert sealed["MOP_TEMPORAL_CORE_STATE.json"] == doc
    assert doc["schema"] == "mop-temporal-core-state/v2" and doc["activation"] is False
    assert doc["all_terminal"] is False and doc["resume"].endswith("exact receipt identities")
    assert "receipts:principal" in doc["items"] and "stage:e2_principal" in doc["items"]
