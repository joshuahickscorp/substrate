"""Long-run daemon for Studio jobs.

This is a small supervisor, not a scheduler of science. It runs a JSON list of commands under one
Profile, writes resumable state after every transition, emits heartbeat events, and stops cleanly on
disk-floor, command failure, or operator dry-run. Long Studio jobs should be boring in exactly this way.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .profiles import Profile, get_profile

SCHEMA = "mop-long-run-daemon/v1"
STATE_FILE = "daemon_state.json"


@dataclass(frozen=True)
class DaemonJob:
    """One supervised command."""

    job_id: str
    command: tuple[str, ...]
    cwd: str | None = None
    kind: str = "run"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id is required")
        if not self.command:
            raise ValueError(f"job {self.job_id} has an empty command")


Runner = Callable[[DaemonJob, Path], int]
DiskProbe = Callable[[], tuple[bool, float]]


def load_plan(path: Path | str) -> list[DaemonJob]:
    """Read a daemon plan."""
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"plan schema {raw.get('schema')!r} != {SCHEMA!r}")
    jobs = raw.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan must contain a non-empty jobs list")
    return [_job_from_obj(obj) for obj in jobs]


def write_plan_template(path: Path | str) -> dict[str, Any]:
    """Write a safe template plan with dry-run-friendly commands."""
    plan = {
        "schema": SCHEMA,
        "jobs": [
            {
                "id": "transfer_check",
                "cmd": [
                    "python",
                    "scripts/studio_transfer_check.py",
                    "--profile",
                    "studio-m1ultra",
                    "--out",
                    "runs/studio_wave0/transfer_check.json",
                ],
                "kind": "gate",
            },
            {
                "id": "doctor",
                "cmd": ["python", "scripts/studio_doctor.py", "--profile", "studio-m1ultra"],
                "kind": "gate",
            },
            {
                "id": "profiles",
                "cmd": ["python", "scripts/studio_pipeline.py", "profiles"],
                "kind": "gate",
            },
            {
                "id": "docs_gate",
                "cmd": ["python", "scripts/check_docs.py"],
                "kind": "gate",
            },
            {
                "id": "acceptance",
                "cmd": ["python", "scripts/acceptance.py"],
                "kind": "gate",
            },
            {
                "id": "dr1_smoke",
                "cmd": ["python", "scripts/studio/dr1_smoke.py"],
                "kind": "gate",
            },
            {
                "id": "encode_microbench",
                "cmd": [
                    "python",
                    "scripts/mop_encode_autoselect.py",
                    "--profile",
                    "studio-m1ultra",
                    "--planned-clips",
                    "1000",
                    "--n-clips",
                    "8",
                ],
                "kind": "microbench",
            },
            {
                "id": "wave0_report",
                "cmd": [
                    "python",
                    "scripts/studio_wave0_report.py",
                    "--daemon-state",
                    "runs/studio_wave0/daemon_state.json",
                    "--apply",
                ],
                "kind": "report",
            },
        ],
    }
    Path(path).write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def run_daemon(
    plan: Sequence[DaemonJob] | Path | str,
    *,
    out_dir: Path | str,
    profile_name: str = "studio-m1ultra",
    execute: bool = False,
    heartbeat_s: float = 300.0,
    poll_s: float = 5.0,
    disk_root: Path | str | None = None,
    runner: Runner | None = None,
    disk_probe: DiskProbe | None = None,
) -> dict[str, Any]:
    """Run or dry-run a daemon plan.

    When `execute` is false, every pending job is marked `dry-run` and no command starts. When true,
    jobs run sequentially and completed jobs are skipped on resume.
    """
    jobs = load_plan(plan) if isinstance(plan, str | Path) else list(plan)
    profile = get_profile(profile_name)
    root = Path(disk_root) if disk_root is not None else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    state = _load_state(out, profile, execute)

    for job in jobs:
        rec = state["jobs"].get(job.job_id, {})
        if rec.get("status") in {"success", "dry-run"}:
            _event(state, "resume-skip", job.job_id, f"already {rec['status']}")
            _write_state(out, state)
            continue

        ok, free_gb = _disk_ok(profile, root, disk_probe)
        if not ok:
            _set_job(state, job, "blocked", free_gb=free_gb, reason="free disk below profile floor")
            _event(state, "blocked", job.job_id, f"{free_gb:.1f} GB free")
            _write_state(out, state)
            break

        if not execute:
            _set_job(state, job, "dry-run", free_gb=free_gb, returncode=0)
            _event(state, "dry-run", job.job_id, "not executed")
            _write_state(out, state)
            continue

        _set_job(state, job, "running", free_gb=free_gb)
        _event(state, "start", job.job_id, "started")
        _write_state(out, state)
        rc = _run_job(job, out, heartbeat_s, poll_s, state, profile, root, runner, disk_probe)
        status = "success" if rc == 0 else "failed"
        _set_job(state, job, status, returncode=rc)
        _event(state, status, job.job_id, f"returncode {rc}")
        _write_state(out, state)
        if rc != 0:
            break

    state["updated_at"] = _now()
    state["summary"] = _summary(state)
    _write_state(out, state)
    return state


def _job_from_obj(obj: Any) -> DaemonJob:
    if not isinstance(obj, dict):
        raise ValueError("each job must be an object")
    raw_cmd = obj.get("cmd", obj.get("command"))
    if isinstance(raw_cmd, str):
        cmd = tuple(shlex.split(raw_cmd))
    elif isinstance(raw_cmd, list):
        cmd = tuple(str(x) for x in raw_cmd)
    else:
        raise ValueError(f"job {obj.get('id', obj.get('job_id'))!r} needs cmd as string or list")
    return DaemonJob(
        job_id=str(obj.get("id", obj.get("job_id", ""))),
        command=cmd,
        cwd=str(obj["cwd"]) if obj.get("cwd") is not None else None,
        kind=str(obj.get("kind", "run")),
        notes=str(obj.get("notes", "")),
    )


def _load_state(out_dir: Path, profile: Profile, execute: bool) -> dict[str, Any]:
    path = out_dir / STATE_FILE
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("schema") == SCHEMA:
            state["resumed_at"] = _now()
            return state
    return {
        "schema": SCHEMA,
        "started_at": _now(),
        "updated_at": _now(),
        "profile": profile.as_dict(),
        "execute": bool(execute),
        "jobs": {},
        "events": [],
    }


def _write_state(out_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    (out_dir / STATE_FILE).write_text(json.dumps(state, indent=2, default=str) + "\n")


def _set_job(state: dict[str, Any], job: DaemonJob, status: str, **extra: Any) -> None:
    rec = {
        "id": job.job_id,
        "kind": job.kind,
        "command": list(job.command),
        "cwd": job.cwd,
        "notes": job.notes,
        "status": status,
        "updated_at": _now(),
    }
    prior = state["jobs"].get(job.job_id, {})
    rec["started_at"] = prior.get("started_at", _now()) if status != "dry-run" else _now()
    if status in {"success", "failed", "blocked", "dry-run"}:
        rec["finished_at"] = _now()
    rec.update(extra)
    state["jobs"][job.job_id] = rec


def _event(state: dict[str, Any], event: str, job_id: str, detail: str) -> None:
    state["events"].append({"at": _now(), "event": event, "job_id": job_id, "detail": detail})


def _disk_ok(profile: Profile, root: Path | None, disk_probe: DiskProbe | None) -> tuple[bool, float]:
    return disk_probe() if disk_probe is not None else profile.free_disk_ok(root)


def _run_job(
    job: DaemonJob,
    out_dir: Path,
    heartbeat_s: float,
    poll_s: float,
    state: dict[str, Any],
    profile: Profile,
    disk_root: Path | None,
    runner: Runner | None,
    disk_probe: DiskProbe | None,
) -> int:
    if runner is not None:
        return int(runner(job, out_dir))

    stdout_path = out_dir / "logs" / f"{job.job_id}.stdout.log"
    stderr_path = out_dir / "logs" / f"{job.job_id}.stderr.log"
    cwd = Path(job.cwd) if job.cwd is not None else None
    with stdout_path.open("a") as so, stderr_path.open("a") as se:
        proc = subprocess.Popen(job.command, cwd=cwd, stdout=so, stderr=se, text=True)
        last_heartbeat = time.monotonic()
        while proc.poll() is None:
            time.sleep(max(0.1, poll_s))
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_s:
                ok, free_gb = _disk_ok(profile, disk_root, disk_probe)
                _event(state, "heartbeat", job.job_id, f"pid {proc.pid}, free_gb {free_gb:.1f}")
                state["jobs"][job.job_id]["last_free_gb"] = round(free_gb, 3)
                _write_state(out_dir, state)
                last_heartbeat = now
                if not ok:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return 98
        return int(proc.returncode or 0)


def _summary(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in state.get("jobs", {}).values():
        status = str(rec.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _now() -> str:
    return datetime.now(UTC).isoformat()
