import copy
import json
import sys

import scripts.p5_traingrid_memory_probe as probe

P5_CORE_RUNTIME_SOURCES = (
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)


def test_traingrid_progress_is_atomic_resumable_and_identity_bound(monkeypatch, tmp_path):
    boundary = tmp_path / "boundary.json"
    boundary.write_text("{}\n")
    output = tmp_path / "trace.json"
    calls = []

    def fake_child(spec):
        calls.append(dict(spec))
        return {
            "cell": f"f{spec['frames']}_{spec['mechanism']}_b{spec['batch']}",
            "ok": True,
            "loss_finite": True,
            "peak_rss_gb": 0.5,
            "wall_seconds_step": 0.01,
            "memory_guard_exceeded": False,
        }

    monkeypatch.setattr(probe, "BOUNDARY_TRACE", boundary)
    monkeypatch.setattr(probe, "run_child", fake_child)
    monkeypatch.setattr(probe, "_free_disk_gb", lambda _root: 100.0)
    monkeypatch.setattr(
        sys,
        "argv",
        ["p5_traingrid_memory_probe.py", "--out", str(output), "--repeats", "1", "--seed", "7"],
    )
    assert probe.main() == 0
    assert len(calls) == 24
    progress_path = output.with_suffix(".json.progress.json")
    progress = json.loads(progress_path.read_text())
    assert progress["schema"] == probe.PROGRESS_SCHEMA
    assert progress["complete"] is True
    assert progress["completed_rows"] == 24
    source_bindings = probe._source_bindings()
    source_bindings_sha256 = probe._json_sha256(source_bindings)
    assert probe.SOURCE_PATHS[-2:] == P5_CORE_RUNTIME_SOURCES
    assert progress["identity"]["source_bindings"] == source_bindings
    assert progress["identity"]["source_bindings_sha256"] == source_bindings_sha256
    first_receipt = json.loads(output.read_text())
    assert first_receipt["atomic_progress"]["completed_rows"] == 24
    assert first_receipt["source_bindings"] == source_bindings
    assert first_receipt["source_bindings_sha256"] == source_bindings_sha256

    calls.clear()
    assert probe.main() == 0
    assert calls == []
    receipt = json.loads(output.read_text())
    assert all(row["resumed_from_atomic_progress"] for row in receipt["cells"])
    declared_digest = receipt.pop("payload_sha256")
    assert declared_digest == probe._json_sha256(receipt)

    for relative in P5_CORE_RUNTIME_SOURCES:
        mutated_bindings = copy.deepcopy(source_bindings)
        source_index = probe.SOURCE_PATHS.index(relative)
        mutated_bindings[source_index]["file_sha256"] = "0" * 64
        with monkeypatch.context() as source_drift:
            source_drift.setattr(
                probe,
                "_source_bindings",
                lambda bindings=mutated_bindings: bindings,
            )
            assert probe.main() == 1
    assert calls == []

    original_progress = json.loads(progress_path.read_text())
    tampered_progress = copy.deepcopy(original_progress)
    tampered_identity = tampered_progress["identity"]
    tampered_identity["source_bindings"][0]["file_sha256"] = "0" * 64
    tampered_identity["source_bindings_sha256"] = probe._json_sha256(tampered_identity["source_bindings"])
    tampered_progress["identity_sha256"] = probe._json_sha256(tampered_identity)
    probe._atomic_json(progress_path, tampered_progress)
    assert probe.main() == 1
    assert calls == []
    probe._atomic_json(progress_path, original_progress)

    monkeypatch.setattr(
        sys,
        "argv",
        ["p5_traingrid_memory_probe.py", "--out", str(output), "--repeats", "1", "--seed", "8"],
    )
    assert probe.main() == 1
    assert calls == []
