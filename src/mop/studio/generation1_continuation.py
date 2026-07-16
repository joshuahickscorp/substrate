"""Crash-recoverable handoff between sealed Generation 1 programs."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ..config import REPO_ROOT
from .generation1_supervisor import (
    Generation1Refused,
    Program,
    atomic_write_json,
    canonical_sha256,
    load_program,
    read_status,
    sha256_file,
    start_detached,
)

PLAN_SCHEMA = "mop-generation1-continuation-plan/v1"
STATE_SCHEMA = "mop-generation1-continuation-state/v1"
STATUS_SCHEMA = "mop-generation1-continuation-status/v1"
STATE_FILE = "continuation_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "control/continuation.lock"
START_LOCK_FILE = "control/start.lock"
TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
PROGRAM_TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
MAX_JSON_BYTES = 4 * 1024 * 1024


class ContinuationRefused(RuntimeError):
    """A continuation plan or observed state failed closed."""


@dataclass(frozen=True, slots=True)
class ProgramReference:
    path: Path
    file_sha256: str
    program_sha256: str


@dataclass(frozen=True, slots=True)
class ContinuationPlan:
    path: Path
    file_sha256: str
    plan_sha256: str
    router_id: str
    out_dir: Path
    prerequisite: ProgramReference
    target: ProgramReference
    authorities: tuple[tuple[Path, str], ...]
    poll_seconds: float
    startup_ack_seconds: float


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationRefused(f"{label} is missing or not a regular file: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ContinuationRefused(f"{label} exceeds the size limit: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationRefused(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContinuationRefused(f"{label} must be a JSON object: {path}")
    return payload


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContinuationRefused(
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContinuationRefused(f"{label} must be a lowercase SHA-256 digest")
    return value


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContinuationRefused(f"{label} must be a nonempty repository-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise ContinuationRefused(f"{label} must be repository-relative")
    resolved = (REPO_ROOT / raw).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ContinuationRefused(f"{label} escapes the repository")
    return resolved


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise ContinuationRefused(f"{label} self-seal is invalid")


def _program_reference(value: object, label: str) -> ProgramReference:
    if not isinstance(value, Mapping):
        raise ContinuationRefused(f"{label} must be an object")
    _exact_keys(value, {"path", "file_sha256", "program_sha256"}, label)
    return ProgramReference(
        path=_repo_path(value["path"], f"{label}.path"),
        file_sha256=_digest(value["file_sha256"], f"{label}.file_sha256"),
        program_sha256=_digest(value["program_sha256"], f"{label}.program_sha256"),
    )


def load_plan(path: Path | str) -> ContinuationPlan:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ContinuationRefused("continuation plan must remain inside the repository")
    payload = _read_json(resolved, "continuation plan")
    _exact_keys(
        payload,
        {
            "schema",
            "router_id",
            "out_dir",
            "prerequisite",
            "target",
            "authorities",
            "control",
            "plan_sha256",
        },
        "continuation plan",
    )
    if payload["schema"] != PLAN_SCHEMA:
        raise ContinuationRefused("continuation plan schema is invalid")
    _validate_seal(payload, "plan_sha256", "continuation plan")
    router_id = payload["router_id"]
    if not isinstance(router_id, str) or not router_id:
        raise ContinuationRefused("router_id must be nonempty")
    out_dir = _repo_path(payload["out_dir"], "out_dir")
    raw_authorities = payload["authorities"]
    if not isinstance(raw_authorities, list) or not raw_authorities:
        raise ContinuationRefused("continuation authorities must be a nonempty list")
    authorities: list[tuple[Path, str]] = []
    for index, raw in enumerate(raw_authorities):
        label = f"authorities[{index}]"
        if not isinstance(raw, Mapping):
            raise ContinuationRefused(f"{label} must be an object")
        _exact_keys(raw, {"path", "sha256"}, label)
        authorities.append(
            (_repo_path(raw["path"], f"{label}.path"), _digest(raw["sha256"], f"{label}.sha256"))
        )
    control = payload["control"]
    if not isinstance(control, Mapping):
        raise ContinuationRefused("continuation control must be an object")
    _exact_keys(control, {"poll_seconds", "startup_ack_seconds"}, "continuation control")
    poll_seconds = control["poll_seconds"]
    startup_ack_seconds = control["startup_ack_seconds"]
    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, int | float)
        or not 1 <= float(poll_seconds) <= 3600
    ):
        raise ContinuationRefused("poll_seconds must be between 1 and 3600")
    if (
        isinstance(startup_ack_seconds, bool)
        or not isinstance(startup_ack_seconds, int | float)
        or not 1 <= float(startup_ack_seconds) <= 600
    ):
        raise ContinuationRefused("startup_ack_seconds must be between 1 and 600")
    plan = ContinuationPlan(
        path=resolved,
        file_sha256=sha256_file(resolved),
        plan_sha256=str(payload["plan_sha256"]),
        router_id=router_id,
        out_dir=out_dir,
        prerequisite=_program_reference(payload["prerequisite"], "prerequisite"),
        target=_program_reference(payload["target"], "target"),
        authorities=tuple(authorities),
        poll_seconds=float(poll_seconds),
        startup_ack_seconds=float(startup_ack_seconds),
    )
    validate_plan_authority(plan)
    return plan


def _load_bound_program(reference: ProgramReference, label: str) -> Program:
    if sha256_file(reference.path) != reference.file_sha256:
        raise ContinuationRefused(f"{label} program file hash drifted")
    try:
        program = load_program(reference.path, repo_root=REPO_ROOT)
    except (Generation1Refused, OSError, ValueError) as exc:
        raise ContinuationRefused(f"{label} program authority is invalid: {exc}") from exc
    if program.program_sha256 != reference.program_sha256:
        raise ContinuationRefused(f"{label} program digest drifted")
    return program


def validate_plan_authority(plan: ContinuationPlan) -> tuple[Program, Program]:
    for path, expected in plan.authorities:
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ContinuationRefused(f"continuation authority drifted: {path}")
    prerequisite = _load_bound_program(plan.prerequisite, "prerequisite")
    target = _load_bound_program(plan.target, "target")
    if prerequisite.program_id == target.program_id:
        raise ContinuationRefused("prerequisite and target programs must differ")
    return prerequisite, target


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise ContinuationRefused(f"continuation lock is already held: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _process_alive(identity: object) -> bool:
    if not isinstance(identity, Mapping):
        return False
    pid = identity.get("pid")
    created = identity.get("create_time")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return False
    if isinstance(created, bool) or not isinstance(created, int | float):
        return False
    try:
        process = psutil.Process(pid)
        return math.isclose(process.create_time(), float(created), rel_tol=0.0, abs_tol=0.01)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False


def _identity() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    implementation = Path(__file__).resolve()
    return {
        "pid": process.pid,
        "create_time": process.create_time(),
        "implementation_path": str(implementation.relative_to(REPO_ROOT.resolve())),
        "implementation_sha256": sha256_file(implementation),
    }


def _initial_state(plan: ContinuationPlan, *, execute: bool) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "router_id": plan.router_id,
        "plan": {
            "path": str(plan.path.relative_to(REPO_ROOT.resolve())),
            "file_sha256": plan.file_sha256,
            "plan_sha256": plan.plan_sha256,
        },
        "supervisor": _identity(),
        "execution_enabled": execute,
        "state": "starting",
        "prerequisite_state": None,
        "target_state": None,
        "prerequisite_start_requests": 0,
        "target_start_requests": 0,
        "started_at": _now(),
        "updated_at": _now(),
        "finished_at": None,
        "problems": [],
    }


def _load_state(plan: ContinuationPlan, *, execute: bool) -> dict[str, Any]:
    path = plan.out_dir / STATE_FILE
    if not path.exists():
        return _initial_state(plan, execute=execute)
    state = _read_json(path, "continuation state")
    _validate_seal(state, "state_sha256", "continuation state")
    if state.get("schema") != STATE_SCHEMA or state.get("router_id") != plan.router_id:
        raise ContinuationRefused("continuation state identity drifted")
    binding = state.get("plan")
    if not isinstance(binding, Mapping) or binding.get("plan_sha256") != plan.plan_sha256:
        raise ContinuationRefused("continuation state plan binding drifted")
    state.pop("state_sha256", None)
    state["supervisor"] = _identity()
    state["execution_enabled"] = execute
    return state


def _publish(plan: ContinuationPlan, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _now()
    state_core = dict(state)
    state_core.pop("state_sha256", None)
    sealed_state = {**state_core, "state_sha256": canonical_sha256(state_core)}
    atomic_write_json(plan.out_dir / STATE_FILE, sealed_state)
    state.clear()
    state.update(sealed_state)
    status_core = {
        "schema": STATUS_SCHEMA,
        "router_id": plan.router_id,
        "created_at": _now(),
        "plan": state["plan"],
        "supervisor": state["supervisor"],
        "execution_enabled": state["execution_enabled"],
        "state": state["state"],
        "prerequisite_state": state["prerequisite_state"],
        "target_state": state["target_state"],
        "prerequisite_start_requests": state["prerequisite_start_requests"],
        "target_start_requests": state["target_start_requests"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "problems": state["problems"],
    }
    status = {**status_core, "status_sha256": canonical_sha256(status_core)}
    atomic_write_json(plan.out_dir / STATUS_FILE, status)
    return status


def read_continuation_status(plan: ContinuationPlan) -> dict[str, Any]:
    status = _read_json(plan.out_dir / STATUS_FILE, "continuation status")
    _validate_seal(status, "status_sha256", "continuation status")
    if status.get("schema") != STATUS_SCHEMA or status.get("router_id") != plan.router_id:
        raise ContinuationRefused("continuation status identity drifted")
    return status


def _program_status(program: Program) -> dict[str, Any] | None:
    if not program.status_path.exists():
        return None
    try:
        return read_status(program)
    except (Generation1Refused, OSError, ValueError) as exc:
        raise ContinuationRefused(
            f"program status authority is invalid for {program.program_id}: {exc}"
        ) from exc


def _ensure_running(program: Program) -> dict[str, Any]:
    try:
        return start_detached(program, execute=True, use_caffeinate=True)
    except (Generation1Refused, OSError, ValueError) as exc:
        raise RuntimeError(f"could not start or recover {program.program_id}: {exc}") from exc


def _append_problem(state: dict[str, Any], problem: str) -> None:
    problems = list(state.get("problems") or [])
    if not problems or problems[-1] != problem:
        problems.append(problem)
    state["problems"] = problems[-32:]


def run_continuation(
    plan: ContinuationPlan,
    *,
    execute: bool,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    if not execute:
        raise ContinuationRefused("continuation execution requires explicit --execute")
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    cycles = 0
    with _FileLock(plan.out_dir / LOCK_FILE):
        state = _load_state(plan, execute=True)
        if state.get("state") in TERMINAL_STATES:
            return _publish(plan, state)
        while True:
            try:
                prerequisite, target = validate_plan_authority(plan)
                prerequisite_status = _program_status(prerequisite)
                if prerequisite_status is None or (
                    prerequisite_status.get("state") not in PROGRAM_TERMINAL_STATES
                    and not _process_alive(prerequisite_status.get("supervisor"))
                ):
                    state["prerequisite_start_requests"] += 1
                    _ensure_running(prerequisite)
                    prerequisite_status = _program_status(prerequisite)
                prerequisite_state = (
                    str(prerequisite_status.get("state")) if prerequisite_status is not None else "starting"
                )
                state["prerequisite_state"] = prerequisite_state
                if prerequisite_state in PROGRAM_TERMINAL_STATES and prerequisite_state != "complete":
                    state["state"] = "failure_hold"
                    state["finished_at"] = _now()
                    _append_problem(
                        state,
                        f"prerequisite program entered terminal state {prerequisite_state}",
                    )
                    return _publish(plan, state)
                if prerequisite_state != "complete":
                    state["state"] = "waiting_prerequisite"
                    state["target_state"] = None
                else:
                    target_status = _program_status(target)
                    if target_status is None or (
                        target_status.get("state") not in PROGRAM_TERMINAL_STATES
                        and not _process_alive(target_status.get("supervisor"))
                    ):
                        state["target_start_requests"] += 1
                        _ensure_running(target)
                        target_status = _program_status(target)
                    target_state = str(target_status.get("state")) if target_status else "starting"
                    state["target_state"] = target_state
                    if target_state == "complete":
                        state["state"] = "complete"
                        state["finished_at"] = _now()
                        return _publish(plan, state)
                    if target_state in PROGRAM_TERMINAL_STATES:
                        state["state"] = "failure_hold"
                        state["finished_at"] = _now()
                        _append_problem(state, f"target program entered terminal state {target_state}")
                        return _publish(plan, state)
                    state["state"] = "waiting_target"
            except ContinuationRefused as exc:
                state["state"] = "integrity_hold"
                state["finished_at"] = _now()
                _append_problem(state, f"{type(exc).__name__}: {exc}")
                return _publish(plan, state)
            except RuntimeError as exc:
                state["state"] = "retry_wait"
                _append_problem(state, f"{type(exc).__name__}: {exc}")
            status = _publish(plan, state)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return status
            time.sleep(plan.poll_seconds)


def start_continuation_detached(
    plan: ContinuationPlan,
    *,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise ContinuationRefused("detached continuation start requires explicit --execute")
    validate_plan_authority(plan)
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    with _FileLock(plan.out_dir / START_LOCK_FILE):
        status_path = plan.out_dir / STATUS_FILE
        if status_path.exists():
            status = read_continuation_status(plan)
            if status["state"] in TERMINAL_STATES:
                return {"already_terminal": True, "status": status}
            if _process_alive(status.get("supervisor")):
                return {"already_running": True, "status": status}
        entrypoint = REPO_ROOT / "scripts/mop_generation1_continuation.py"
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(entrypoint),
            "run",
            "--plan",
            str(plan.path),
            "--execute",
        ]
        caffeinate = shutil.which("caffeinate") if use_caffeinate else None
        launched = [caffeinate, "-ims", *command] if caffeinate else command
        stdout_path = plan.out_dir / "logs/continuation.stdout.log"
        stderr_path = plan.out_dir / "logs/continuation.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        prior_mtime = status_path.stat().st_mtime_ns if status_path.exists() else 0
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
        environment["PYTHONUNBUFFERED"] = "1"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                launched,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + plan.startup_ack_seconds
        acknowledged: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if status_path.exists() and status_path.stat().st_mtime_ns > prior_mtime:
                try:
                    acknowledged = read_continuation_status(plan)
                    break
                except (ContinuationRefused, OSError, ValueError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise ContinuationRefused(
                f"detached continuation did not acknowledge startup; inspect {stderr_path}"
            )
        return {
            "launched_pid": process.pid,
            "caffeinate": bool(caffeinate),
            "command": launched,
            "status": acknowledged,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "start"):
        child = subparsers.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--execute", action="store_true")
        if name == "start":
            child.add_argument("--no-caffeinate", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = load_plan(arguments.plan)
        if arguments.command == "run":
            payload = run_continuation(plan, execute=arguments.execute)
        elif arguments.command == "start":
            payload = start_continuation_detached(
                plan,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        else:
            payload = read_continuation_status(plan)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (ContinuationRefused, Generation1Refused, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "ContinuationPlan",
    "ContinuationRefused",
    "PLAN_SCHEMA",
    "load_plan",
    "main",
    "read_continuation_status",
    "run_continuation",
    "start_continuation_detached",
    "validate_plan_authority",
]


if __name__ == "__main__":
    raise SystemExit(main())
