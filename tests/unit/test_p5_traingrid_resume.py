import json
import sys

import scripts.p5_traingrid_memory_probe as probe


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
    assert json.loads(output.read_text())["atomic_progress"]["completed_rows"] == 24

    calls.clear()
    assert probe.main() == 0
    assert calls == []
    receipt = json.loads(output.read_text())
    assert all(row["resumed_from_atomic_progress"] for row in receipt["cells"])

    monkeypatch.setattr(
        sys,
        "argv",
        ["p5_traingrid_memory_probe.py", "--out", str(output), "--repeats", "1", "--seed", "8"],
    )
    assert probe.main() == 1
    assert calls == []
