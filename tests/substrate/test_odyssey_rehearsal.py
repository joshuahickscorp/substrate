"""Honesty-focused tests for Odyssey G06–G09 rehearsal producers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from substrate import odyssey_authority as authority
from substrate import odyssey_rehearsal as rehearsal

ROOT = Path(__file__).resolve().parents[2]
PY_OUT = ROOT / "artifacts/substrate/odyssey7d/v1/rehearsal"


def _seal(gate_id: str, subject_path: Path, tmp_path: Path) -> dict:
    out = tmp_path / f"{gate_id}.gate.json"
    return authority.seal_machine_gate(ROOT, gate_id, subject_path, out)


def test_g08_subject_seals(tmp_path: Path) -> None:
    out = PY_OUT / "test_G08.subject.json"
    if out.exists():
        out.unlink()
    subject = rehearsal.run_g08(ROOT, out)
    assert subject["status"] == "pass"
    gate = _seal("G08", out, tmp_path)
    assert gate["status"] == "pass"
    assert gate["gate_id"] == "G08"


def test_g07_subject_seals(tmp_path: Path) -> None:
    out = PY_OUT / "test_G07.subject.json"
    if out.exists():
        out.unlink()
    subject = rehearsal.run_g07(ROOT, out)
    assert subject["status"] == "pass"
    assert subject["observed_total_private_growth_bytes"] == sum(row["durable_growth_bytes"] for row in subject["cell_observations"])
    assert subject["p95_private_growth_bytes"] >= subject["observed_total_private_growth_bytes"]
    gate = _seal("G07", out, tmp_path)
    assert gate["status"] == "pass"


def test_g09_subject_seals(tmp_path: Path) -> None:
    out = PY_OUT / "test_G09.subject.json"
    if out.exists():
        out.unlink()
    subject = rehearsal.run_g09(ROOT, out)
    assert subject["status"] == "pass"
    for row in subject["rehearsals"]:
        for role in ("candidate", "control"):
            arm = row["arms"][role]
            assert arm["pre_interrupt_state_sha256"] == arm["restored_state_sha256"]
    gate = _seal("G09", out, tmp_path)
    assert gate["status"] == "pass"


def test_g08_unexpected_decision_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_decision(resident_gib: float, *, critical_pressure: bool) -> str:
        return "admit_or_resume"

    monkeypatch.setattr(rehearsal, "_broker_decision", fake_decision)
    out = PY_OUT / "test_G08_refuse.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="real broker returned"):
        rehearsal.run_g08(ROOT, out)


def test_g07_free_space_floor_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rehearsal,
        "_free_bytes",
        lambda _path: rehearsal.DEVICE_FREE_FLOOR_BYTES - 1,
    )
    out = PY_OUT / "test_G07_floor.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="25 GiB device floor"):
        rehearsal.run_g07(ROOT, out)


def test_g07_cap_refusal_on_huge_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    real_work = rehearsal._g07_cell_work

    def huge_growth(root, cell_id, cell_base, *, free_samples):
        row = real_work(root, cell_id, cell_base, free_samples=free_samples)
        row["durable_growth_bytes"] = 3 * 1024**3  # 3 GiB measured stand-in
        return row

    monkeypatch.setattr(rehearsal, "_g07_cell_work", huge_growth)
    out = PY_OUT / "test_G07_cap.subject.json"
    if out.exists():
        out.unlink()
    # 8 * 3 GiB * 84 microcycles >> 120 GiB
    with pytest.raises(rehearsal.Refused, match="120 GiB cap"):
        rehearsal.run_g07(ROOT, out)


def test_g09_restore_mismatch_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    real_digest = rehearsal._state_digest
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        digest = real_digest(state)
        # After the pre-interrupt digest, poison the restored digest once.
        if calls["n"] == 2:
            return "0" * 64
        return digest

    monkeypatch.setattr(rehearsal, "_state_digest", flaky)
    out = PY_OUT / "test_G09_mismatch.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="restore mismatch"):
        rehearsal.run_g09(ROOT, out)


def test_g09_corrupt_delta_refuses_restore(tmp_path: Path) -> None:
    """A delta the production reader rejects must fail restore, not seal green."""
    arm = tmp_path / "arm"
    arm.mkdir()
    authority = rehearsal.digest({"rehearsal": "g09", "frontier": "A", "role": "candidate"})
    run_id = "g09-A-candidate"
    meta = rehearsal._g09_build_durable_chain(arm, authority_sha256=authority, run_id=run_id, frontier="A")
    delta = Path(meta["delta_path"])
    raw = json.loads(delta.read_text(encoding="utf-8"))
    # Keep self-digest valid but break the structural fields the worker checks.
    raw["completed_phase_count"] = 999
    raw.pop("sha256", None)
    raw["sha256"] = rehearsal.digest(raw)
    delta.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(rehearsal.Refused, match="restore refused by worker readers"):
        rehearsal._g09_state_from_durable(
            authority_sha256=authority,
            run_id=run_id,
            frontier_ids=["A"],
            phases=list(meta["phases"]),
            expected_phases=int(meta["expected_phases"]),
            events_path=Path(meta["events_path"]),
            checkpoints=Path(meta["checkpoints"]),
        )


def test_g08_measured_rss_over_cap_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rehearsal,
        "_measure_memory_pools",
        lambda: {
            "host_rss_bytes": rehearsal.RESIDENT_CAP_BYTES + 1,
            "host_rss_gib": 86.0,
            "pageout_bytes": 0,
            "thermal_pressure": "nominal",
            "critical_pressure": False,
            "vm_pressure_level": 0,
            "memory_pools_bytes": {
                "host": rehearsal.RESIDENT_CAP_BYTES + 1,
                "vm": 0,
                "container": 0,
                "model_service": 0,
                "broker": 0,
            },
            "memory_pools_gib": {
                "host": 86.0,
                "vm": 0.0,
                "container": 0.0,
                "model_service": 0.0,
                "broker": 0.0,
            },
            "lane_resident_bytes": {f: 0 for f in "ABCDEFGH"},
            "lane_resident_gib": {f: 0.0 for f in "ABCDEFGH"},
            "self_rss_bytes": 1,
            "model_service_bytes": 0,
        },
    )
    # Advance monotonic through the sealed 30s sampling interval without waiting.
    clock = {"t": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["t"] += float(seconds)

    monkeypatch.setattr(rehearsal.time, "sleep", fake_sleep)
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: clock["t"])
    out = PY_OUT / "test_G08_rss_cap.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="exceeds"):
        rehearsal.run_g08(ROOT, out)


def test_g07_transient_inflates_runtime_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inflating measured transients past free capacity must refuse."""
    real_work = rehearsal._g07_cell_work

    def huge_transient(root, cell_id, cell_base, *, free_samples):
        row = real_work(root, cell_id, cell_base, free_samples=free_samples)
        # 40 GiB per-lane transient * 8 slots blows any reasonable free floor.
        row["largest_transient_bytes"] = 40 * 1024**3
        return row

    monkeypatch.setattr(rehearsal, "_g07_cell_work", huge_transient)
    out = PY_OUT / "test_G07_transient_floor.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="runtime floor|dynamic capacity|minimum free"):
        rehearsal.run_g07(ROOT, out)


def test_g06_receipt_invariance_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    real_job = rehearsal._g06_cell_job

    def poison_unit(payload):
        row = real_job(payload)
        # Different unit digest per width makes invariance unsatisfiable.
        if int(payload["width"]) > 1:
            row["unit_receipt_sha256"] = "a" * 64
        return row

    monkeypatch.setattr(rehearsal, "_g06_cell_job", poison_unit)

    def fake_sub(job):
        class Proc:
            returncode = 0

            def communicate(self, timeout=None):
                result = poison_unit(job)
                path = Path(job["cell_base"]) / "job-result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result), encoding="utf-8")
                return ("", "")

            def kill(self):
                return None

        return Proc()

    monkeypatch.setattr(rehearsal, "_run_g06_cell_subprocess", fake_sub)
    out = PY_OUT / "test_G06_invariance.subject.json"
    if out.exists():
        out.unlink()
    with pytest.raises(rehearsal.Refused, match="receipt invariance"):
        rehearsal.run_g06(
            ROOT,
            out,
            widths=(1, 2),
            repetitions=1,
            launch=False,
            num_predict=8,
        )


def test_receipt_tamper_breaks_seal(tmp_path: Path) -> None:
    out = PY_OUT / "test_G08_tamper.subject.json"
    if out.exists():
        out.unlink()
    subject = rehearsal.run_g08(ROOT, out)
    # Tamper with the first referenced receipt file.
    ref = subject["observations"][0]["receipt_refs"][0]
    path = ROOT / ref["path"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["decision"] = "tampered"
    path.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(authority.Refused):
        _seal("G08", out, tmp_path)


def test_g06_reduced_non_launch_marks_subject(tmp_path: Path) -> None:
    out = PY_OUT / "test_G06_reduced.subject.json"
    if out.exists():
        out.unlink()
    subject = rehearsal.run_g06(
        ROOT,
        out,
        widths=(1,),
        repetitions=1,
        launch=False,
        num_predict=64,
    )
    assert subject.get("launch_subject") is False
    assert subject["non_launch_reduced_schedule"]["widths"] == [1]
    # Full-schedule validator must reject a reduced subject.
    with pytest.raises(authority.Refused):
        _seal("G06", out, tmp_path)


def test_g06_slowdown_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    real_limits = rehearsal._calibration_limits

    def tight_limits(root, frozen):
        limits = real_limits(root, frozen)
        limits["max_slowdown_ratio"] = 1.000001
        return limits

    monkeypatch.setattr(rehearsal, "_calibration_limits", tight_limits)

    # Force a second width so slowdown can exceed 1.0 after baseline exists.
    out = PY_OUT / "test_G06_slowdown.subject.json"
    if out.exists():
        out.unlink()

    real_job = rehearsal._g06_cell_job

    def slower(payload):
        row = real_job(payload)
        if int(payload["width"]) > 1:
            row["wall_seconds"] = float(row["wall_seconds"]) * 4.0
        return row

    monkeypatch.setattr(rehearsal, "_g06_cell_job", slower)

    # Run cell work in-process by short-circuiting subprocess collection.
    def fake_sub(job):
        class Proc:
            returncode = 0

            def communicate(self, timeout=None):
                result = slower(job)
                path = Path(job["cell_base"]) / "job-result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result), encoding="utf-8")
                return ("", "")

            def kill(self):
                return None

        return Proc()

    monkeypatch.setattr(rehearsal, "_run_g06_cell_subprocess", fake_sub)

    with pytest.raises(rehearsal.Refused, match="per_cell_slowdown_ratio"):
        rehearsal.run_g06(
            ROOT,
            out,
            widths=(1, 2),
            repetitions=1,
            launch=False,
            num_predict=8,
        )
