import json

from mop.studio.memory_envelope import SCHEMA as MEM_SCHEMA
from mop.studio.wave0_report import build_wave0_report, render_markdown, upsert_report_block


def _memory():
    return {
        "schema": MEM_SCHEMA,
        "process_rss_gb": {"start": 1.0, "end": 2.0, "peak": 2.5},
        "system_available_gb": {"start": 100.0, "end": 90.0, "peak": 88.0},
        "mps_driver_allocated_gb": {"start": 0.0, "end": 3.0, "peak": 3.5},
    }


def test_wave0_report_all_ok_when_receipts_green():
    report = build_wave0_report(
        transfer={"all_ok": True, "summary": {"passed": 28, "total": 28, "failed": 0}},
        daemon_state={"summary": {"success": 7, "running": 1}},
        encode_device={
            "winner": "cpu",
            "cpu_s_per_clip": 14.0,
            "mps": 99.0,
            "n_clips": 8,
            "memory_envelope": _memory(),
        },
        encode_schedule={"ok_to_launch": True, "blocked_reasons": [], "effective_clips": 1000},
    )
    assert report["all_ok"] is True
    assert report["memory_envelope"]["process_rss_peak_gb"] == 2.5
    assert report["memory_envelope"]["system_available_min_gb"] == 88.0
    md = render_markdown(report)
    assert "Studio Wave 0 Auto Receipt" in md
    assert "CPU 14.0 s/clip" in md


def test_wave0_report_marks_blocked_encode_incomplete():
    report = build_wave0_report(
        transfer={"all_ok": True, "summary": {"passed": 28, "total": 28}},
        daemon_state={"summary": {"success": 7}},
        encode_device={"winner": "blocked", "memory_envelope": _memory()},
        encode_schedule={"ok_to_launch": False, "blocked_reasons": ["no usable CPU or MPS speed"]},
    )
    assert report["all_ok"] is False
    assert "no usable" in render_markdown(report)


def test_upsert_report_block_is_idempotent(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# Report\n\n## Wave log\n")
    report = build_wave0_report(
        transfer={"all_ok": True, "summary": {"passed": 1, "total": 1}},
        daemon_state={"summary": {"success": 1}},
        encode_device={"winner": "mps", "cpu_s_per_clip": 4.0, "mps": 1.0, "memory_envelope": _memory()},
        encode_schedule={"ok_to_launch": True},
    )
    block = render_markdown(report)
    upsert_report_block(path, block)
    upsert_report_block(path, block)
    text = path.read_text()
    assert text.count("STUDIO-WAVE0-AUTO:START") == 1
    assert "## Wave log" in text
    assert json.dumps(report["schema"])
