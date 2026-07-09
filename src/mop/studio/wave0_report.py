"""Studio Wave-0 report synthesis.

The Studio Wave-0 run emits several receipts: transfer check, disk recovery, doctor, daemon state,
encode device, and encode schedule. This module turns them into one JSON summary plus a bounded
Markdown block for STUDIO_RUN_REPORT.md, so the report records actual launch status, s/clip, and
memory envelope without hand editing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "mop-studio-wave0-report/v1"
AUTO_START = "<!-- STUDIO-WAVE0-AUTO:START -->"
AUTO_END = "<!-- STUDIO-WAVE0-AUTO:END -->"


def build_wave0_report(
    *,
    transfer: dict[str, Any] | None,
    encode_device: dict[str, Any] | None,
    encode_schedule: dict[str, Any] | None,
    doctor: dict[str, Any] | None = None,
    disk_recovery: dict[str, Any] | None = None,
    daemon_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact Wave-0 summary from receipts."""
    memory = (encode_device or {}).get("memory_envelope") or (encode_schedule or {}).get("memory_envelope")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "launch_status": _launch_summary(doctor, disk_recovery),
        "transfer": _transfer_summary(transfer),
        "daemon": _daemon_summary(daemon_state),
        "encode": _encode_summary(encode_device, encode_schedule),
        "memory_envelope": _memory_summary(memory),
    }
    report["all_ok"] = _all_ok(report)
    return report


def load_json(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_json(report: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    """Render the bounded Markdown block inserted into STUDIO_RUN_REPORT.md."""
    encode = report["encode"]
    memory = report["memory_envelope"]
    daemon = report["daemon"]
    transfer = report["transfer"]
    launch = report["launch_status"]
    status = "COMPLETE" if report["all_ok"] else "INCOMPLETE"
    lines = [
        AUTO_START,
        "## Studio Wave 0 Auto Receipt",
        "",
        f"- Status: {status}.",
        (
            f"- Launch profile: {launch['profile']}; hardware {launch['hardware']['detail']}; "
            f"disk {launch['disk']['detail']}."
        ),
        (
            f"- MPS: {launch['mps']['detail']}; encoders: {launch['encoders']['detail']}; "
            f"cache path: {launch['cache_path']['detail']}."
        ),
        f"- Disk recovery: {_ok_word(launch['disk_recovery']['ok'])} ({launch['disk_recovery']['detail']}).",
        f"- Transfer check: {_ok_word(transfer['ok'])} ({transfer['detail']}).",
        f"- Daemon gates: {daemon['detail']}.",
        (
            f"- Encode winner: {encode['winner']}; CPU {encode['cpu_s_per_clip']} s/clip; "
            f"MPS {encode['mps_s_per_clip']} s/clip; schedule launch {_ok_word(encode['ok_to_launch'])}."
        ),
        (
            f"- Memory envelope: process RSS peak {memory['process_rss_peak_gb']} GB; "
            f"min system available {memory['system_available_min_gb']} GB; "
            f"MPS driver peak {memory['mps_driver_peak_gb']} GB."
        ),
    ]
    if encode["blocked_reasons"]:
        lines.append(f"- Blocked reasons: {'; '.join(encode['blocked_reasons'])}.")
    lines.extend(["", AUTO_END])
    return "\n".join(lines) + "\n"


def upsert_report_block(report_path: Path | str, block: str) -> None:
    """Insert or replace the auto block in STUDIO_RUN_REPORT.md."""
    path = Path(report_path)
    text = path.read_text() if path.exists() else "# STUDIO RUN REPORT\n\n"
    if AUTO_START in text and AUTO_END in text:
        before, rest = text.split(AUTO_START, 1)
        _, after = rest.split(AUTO_END, 1)
        text = before.rstrip() + "\n\n" + block.rstrip() + "\n" + after
    else:
        anchor = "## Wave log"
        if anchor in text:
            text = text.replace(anchor, block + "\n" + anchor, 1)
        else:
            text = text.rstrip() + "\n\n" + block
    path.write_text(text)


def _transfer_summary(transfer: dict[str, Any] | None) -> dict[str, Any]:
    if transfer is None:
        return {"ok": False, "detail": "missing transfer receipt"}
    summary = transfer.get("summary", {})
    return {
        "ok": bool(transfer.get("all_ok")),
        "detail": f"{summary.get('passed', 0)}/{summary.get('total', 0)} checks passed",
        "failed": int(summary.get("failed", 0)),
    }


def _launch_summary(
    doctor: dict[str, Any] | None,
    disk_recovery: dict[str, Any] | None,
) -> dict[str, Any]:
    hardware = _doctor_check(doctor, "apple_silicon")
    torch_check = _doctor_check(doctor, "torch")
    disk_space = _doctor_check(doctor, "disk_space")
    profile_floor = _doctor_check(doctor, "profile_floor")
    encoders = _doctor_check(doctor, "encoders")
    cache_path = _doctor_check(doctor, "cache_write")
    profile = _profile_from_doctor_or_disk(doctor, disk_recovery)
    disk = {
        "ok": disk_space["ok"] and profile_floor["ok"],
        "detail": f"{disk_space['detail']}; {profile_floor['detail']}",
        "disk_space": disk_space,
        "profile_floor": profile_floor,
    }
    disk_rec = _disk_recovery_summary(disk_recovery)
    ok = bool(
        (doctor or {}).get("all_ok")
        and hardware["ok"]
        and torch_check["ok"]
        and disk["ok"]
        and encoders["ok"]
        and cache_path["ok"]
        and disk_rec["ok"]
    )
    return {
        "ok": ok,
        "profile": profile,
        "hardware": hardware,
        "mps": torch_check,
        "disk": disk,
        "encoders": encoders,
        "cache_path": cache_path,
        "disk_recovery": disk_rec,
    }


def _doctor_check(doctor: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if doctor is None:
        return {"ok": False, "detail": "missing doctor receipt"}
    for check in doctor.get("checks", []):
        if check.get("name") == name:
            return {"ok": bool(check.get("ok")), "detail": str(check.get("detail", ""))}
    return {"ok": False, "detail": f"doctor check {name!r} missing"}


def _profile_from_doctor_or_disk(doctor: dict[str, Any] | None, disk_recovery: dict[str, Any] | None) -> str:
    profile = (disk_recovery or {}).get("profile")
    if isinstance(profile, dict) and profile.get("name"):
        return str(profile["name"])
    for check in (doctor or {}).get("checks", []):
        if check.get("name") == "profile_floor":
            detail = str(check.get("detail", ""))
            return detail.split(":", 1)[0] if ":" in detail else "unknown"
    return "missing"


def _disk_recovery_summary(disk_recovery: dict[str, Any] | None) -> dict[str, Any]:
    if disk_recovery is None:
        return {"ok": False, "detail": "missing disk recovery receipt"}
    free = disk_recovery.get("free_disk", {})
    summary = disk_recovery.get("summary", {})
    problems = disk_recovery.get("problems") or []
    detail = (
        f"{free.get('free_gb')} GB free vs floor {free.get('floor_gb')} GB; "
        f"{summary.get('safe_count', 0)} safe candidate(s), "
        f"{summary.get('blocked_count', 0)} protected candidate(s), "
        f"would delete {summary.get('would_delete_human', '0 B')}"
    )
    if problems:
        detail += f"; first problem: {problems[0]}"
    return {
        "ok": bool(disk_recovery.get("all_ok")),
        "detail": detail,
        "free_disk": free,
        "summary": summary,
        "problems": problems,
    }


def _daemon_summary(daemon: dict[str, Any] | None) -> dict[str, Any]:
    if daemon is None:
        return {"ok": False, "detail": "missing daemon state", "summary": {}}
    summary = daemon.get("summary", {})
    failed = int(summary.get("failed", 0)) + int(summary.get("blocked", 0))
    success = int(summary.get("success", 0))
    dry = int(summary.get("dry-run", 0))
    total = success + failed + dry
    return {
        "ok": failed == 0 and total > 0 and dry == 0,
        "detail": f"{success}/{total} succeeded, {dry} dry-run, {failed} failed-or-blocked",
        "summary": summary,
    }


def _encode_summary(
    encode_device: dict[str, Any] | None,
    encode_schedule: dict[str, Any] | None,
) -> dict[str, Any]:
    device = encode_device or {}
    schedule = encode_schedule or {}
    return {
        "ok": bool(schedule.get("ok_to_launch")) and device.get("winner") not in {None, "blocked"},
        "winner": device.get("winner", "missing"),
        "cpu_s_per_clip": device.get("cpu_s_per_clip"),
        "mps_s_per_clip": device.get("mps"),
        "n_clips": int(device.get("n_clips") or 0),
        "ok_to_launch": bool(schedule.get("ok_to_launch")),
        "blocked_reasons": list(schedule.get("blocked_reasons") or []),
        "effective_clips": schedule.get("effective_clips"),
        "winner_plan": schedule.get("winner"),
    }


def _memory_summary(memory: dict[str, Any] | None) -> dict[str, Any]:
    if memory is None:
        return {
            "ok": False,
            "schema": None,
            "process_rss_peak_gb": None,
            "system_available_min_gb": None,
            "mps_driver_peak_gb": None,
        }
    return {
        "ok": memory.get("schema") == "mop-memory-envelope/v1",
        "schema": memory.get("schema"),
        "process_rss_peak_gb": _peak(memory, "process_rss_gb"),
        "system_available_min_gb": _peak(memory, "system_available_gb"),
        "mps_driver_peak_gb": _peak(memory, "mps_driver_allocated_gb"),
    }


def _peak(memory: dict[str, Any], key: str) -> float | None:
    series = memory.get(key)
    if not isinstance(series, dict):
        return None
    value = series.get("peak")
    return float(value) if isinstance(value, int | float) else None


def _all_ok(report: dict[str, Any]) -> bool:
    return bool(
        report["launch_status"]["ok"]
        and report["transfer"]["ok"]
        and report["daemon"]["ok"]
        and report["encode"]["ok"]
        and report["memory_envelope"]["ok"]
    )


def _ok_word(ok: bool) -> str:
    return "ok" if ok else "not-ok"
