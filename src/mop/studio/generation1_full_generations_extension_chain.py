"""Append-only waiter for the full-generations Generation 1 wave.

This parent never signals or restarts the predecessor.  It observes an exact,
clean, complete ``generation1-successor-categorized-batch-wave-v1`` generic-supervisor
snapshot, checks current host admission, and only then starts or resumes the
sealed full-generations wave program through the generic Generation 1 supervisor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studio import generation1_supervisor
from mop.studio.generation1_supervisor import (
    LOCK_FILE as GENERIC_LOCK_FILE,
)
from mop.studio.generation1_supervisor import (
    STATUS_SCHEMA as GENERIC_STATUS_SCHEMA,
)
from mop.studio.generation1_supervisor import (
    FileLock,
    Generation1Refused,
    Generation1Supervisor,
    Program,
    atomic_write_json,
    canonical_sha256,
    load_program,
    read_status,
    sha256_file,
    start_detached,
)

PROGRAM_ID = "generation1-full-generations-extension-chain-v1"
PARENT_LABEL = "mop-full-generations-extension-chain"
STATE_SCHEMA = "mop-generation1-full-generations-extension-chain-state/v1"
STATUS_SCHEMA = "mop-generation1-full-generations-extension-chain-status/v1"

PREDECESSOR_PROGRAM_ID = "generation1-successor-categorized-batch-wave-v1"
TARGET_PROGRAM_ID = "generation1-full-generations-wave-v1"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_PREDECESSOR_PROGRAM = (
    REPO_ROOT / "configs/campaign/generation1_successor_categorized_batch_wave_v1.json"
)
DEFAULT_PREDECESSOR_STATUS = (
    REPO_ROOT / "runs/generation1/generation1-successor-categorized-batch-wave-v1/current_status.json"
)
DEFAULT_TARGET_PROGRAM = REPO_ROOT / "configs/campaign/generation1_full_generations_wave_v1.json"

STATE_FILE = "extension_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "control/extension.lock"
START_LOCK_FILE = "control/start.lock"
POLL_SECONDS = 30.0
STARTUP_ACK_SECONDS = 300.0
PROCESS_TIME_TOLERANCE_SECONDS = 0.02
MAX_JSON_BYTES = 64 * 1024 * 1024
SNAPSHOT_ATTEMPTS = 5
SNAPSHOT_INTERVAL_SECONDS = 0.05
LOCK_ATTEMPTS = 41
LOCK_INTERVAL_SECONDS = 0.05

TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})
GENERIC_TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
GENERIC_STATES = frozenset(
    {
        "starting",
        "execution_disabled",
        "progressing",
        "resource_wait",
        "retry_wait",
        "running",
        "recovery_observe",
        "recovery_wait",
        *GENERIC_TERMINAL_STATES,
    }
)
EXTENSION_STATES = frozenset(
    {
        "starting",
        "waiting_predecessor",
        "waiting_host",
        "waiting_target",
        "retry_wait",
        *TERMINAL_STATES,
    }
)
ROW_STATES = frozenset(
    {
        "pending",
        "observed",
        "running",
        "adoption_wait",
        "launching",
        "retry_wait",
        "complete",
        "failure_hold",
    }
)
GENERIC_STATUS_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "program",
        "supervisor",
        "execution_enabled",
        "state",
        "queue_head_sha256",
        "next_injection_sequence",
        "accepted_injection_count",
        "current_capsule",
        "capsules",
        "last_admission",
        "lane_reservation",
        "problems",
        "status_sha256",
    }
)
GENERIC_CAPSULE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "priority",
        "depends_on",
        "capsule_sha256",
        "source",
        "status",
        "attempts",
        "child_pid",
        "child_create_time",
        "started_at",
        "finished_at",
        "returncode",
        "artifacts",
        "last_problem",
        "runtime",
    }
)
ROW_FIELDS = frozenset(
    {
        "status",
        "attempts",
        "returncode",
        "started_at",
        "finished_at",
        "artifacts",
        "last_problem",
        "process",
        "launch_requested_at",
        "launched_pid",
    }
)
STATE_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "updated_at",
        "finished_at",
        "execution_enabled",
        "status",
        "supervisor",
        "parent_implementation",
        "predecessor_program",
        "target_program",
        "capsules",
        "last_admission",
        "problems",
        "state_sha256",
    }
)
STATUS_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "updated_at",
        "finished_at",
        "execution_enabled",
        "state",
        "supervisor",
        "parent_implementation",
        "predecessor_program",
        "target_program",
        "capsules",
        "counts",
        "last_admission",
        "problems",
        "signals_allowed",
        "activation_allowed",
        "scientific_promotion",
        "status_sha256",
    }
)
SUPERVISOR_FIELDS = frozenset({"pid", "create_time", "implementation_path", "implementation_sha256"})
IMPLEMENTATION_PATH = Path(__file__).resolve()


class FullGenerationsExtensionRefused(RuntimeError):
    """The full-generations successor boundary could not be established exactly."""


IdentityProbe = Callable[[Mapping[str, Any]], str]
ProcessTable = Callable[[], Sequence["ProcessSnapshot"]]
ProgramLoader = Callable[..., Program]
StatusReader = Callable[[Program], Mapping[str, Any]]
SupervisorStarter = Callable[..., Mapping[str, Any]]
AdmissionProbe = Callable[[Program, Mapping[str, Any] | None], Mapping[str, Any]]
NowFn = Callable[[], dt.datetime]


class ProcessSnapshot:
    """Exact process identity used to prevent duplicate detached supervisors."""

    __slots__ = ("pid", "create_time", "pgid", "cwd", "label", "command")

    def __init__(
        self,
        *,
        pid: int,
        create_time: float,
        pgid: int,
        cwd: str,
        label: str,
        command: Sequence[str],
    ) -> None:
        self.pid = pid
        self.create_time = create_time
        self.pgid = pgid
        self.cwd = cwd
        self.label = label
        self.command = tuple(command)

    def identity(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "create_time": self.create_time,
            "pgid": self.pgid,
            "cwd": self.cwd,
            "label": self.label,
        }


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat()


def _parse_time(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.UTC)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise FullGenerationsExtensionRefused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or not 0 < stat.st_size <= MAX_JSON_BYTES:
        raise FullGenerationsExtensionRefused(f"{label} is not a safe bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise FullGenerationsExtensionRefused(f"{label} changed while being read: {path}")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullGenerationsExtensionRefused(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FullGenerationsExtensionRefused(f"{label} must be a JSON object: {path}")
    return payload


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise FullGenerationsExtensionRefused(f"{label} self-seal is invalid")


def _require_fields(value: object, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullGenerationsExtensionRefused(f"{label} must be an object")
    if set(value) != fields:
        raise FullGenerationsExtensionRefused(
            f"{label} fields drifted; missing={sorted(fields - set(value))}, "
            f"extra={sorted(set(value) - fields)}"
        )
    return value


def _dotted(payload: object, path: str) -> object:
    value = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _safe_current_file(repo_root: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise FullGenerationsExtensionRefused(f"{label} path must be repository-relative")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FullGenerationsExtensionRefused(f"{label} path is unsafe")
    candidate = repo_root / relative
    cursor = repo_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise FullGenerationsExtensionRefused(f"{label} path traverses a symlink")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo_root) or not resolved.is_file():
        raise FullGenerationsExtensionRefused(f"{label} is not a safe current file")
    return resolved


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _implementation_authority(repo_root: Path) -> dict[str, str]:
    return {
        "path": _relative_or_absolute(IMPLEMENTATION_PATH, repo_root),
        "sha256": sha256_file(IMPLEMENTATION_PATH),
    }


def _generic_supervisor_authority(repo_root: Path) -> dict[str, str]:
    implementation = Path(generation1_supervisor.__file__).resolve()
    return {
        "path": _relative_or_absolute(implementation, repo_root),
        "sha256": sha256_file(implementation),
    }


def _program_authority(
    path: Path,
    *,
    repo_root: Path,
    expected_program_id: str,
    signals_allowed: bool | None = None,
) -> dict[str, Any]:
    try:
        program = load_program(path, repo_root=repo_root)
    except (Generation1Refused, OSError, TypeError, ValueError) as exc:
        raise FullGenerationsExtensionRefused(
            f"{expected_program_id} manifest is invalid: {type(exc).__name__}: {exc}"
        ) from exc
    if program.program_id != expected_program_id:
        raise FullGenerationsExtensionRefused(f"{expected_program_id} manifest identity drifted")
    authority: dict[str, Any] = {
        "path": _relative_or_absolute(program.path, repo_root),
        "file_sha256": program.file_sha256,
        "program_sha256": program.program_sha256,
        "program_id": program.program_id,
        "status_path": _relative_or_absolute(program.status_path, repo_root),
    }
    if signals_allowed is not None:
        authority["signals_allowed"] = signals_allowed
    return authority


def _self_identity(repo_root: Path) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    authority = _implementation_authority(repo_root)
    return {
        "pid": process.pid,
        "create_time": process.create_time(),
        "implementation_path": authority["path"],
        "implementation_sha256": authority["sha256"],
    }


def _default_identity_probe(identity: Mapping[str, Any]) -> str:
    pid = identity.get("pid")
    created = identity.get("create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(created, bool)
        or not isinstance(created, int | float)
        or not math.isfinite(float(created))
        or float(created) <= 0
    ):
        return "unknown"
    try:
        process = psutil.Process(pid)
        if not math.isclose(
            process.create_time(),
            float(created),
            rel_tol=0.0,
            abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
        ):
            return "gone"
        if process.status() == psutil.STATUS_ZOMBIE:
            return "gone"
        return "alive" if process.is_running() else "gone"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "gone"
    except (psutil.AccessDenied, OSError, RuntimeError, ValueError):
        return "unknown"


def _default_process_table() -> tuple[ProcessSnapshot, ...]:
    rows: list[ProcessSnapshot] = []
    for process in psutil.process_iter():
        try:
            command = tuple(process.cmdline())
            if not command:
                continue
            pid = int(process.pid)
            rows.append(
                ProcessSnapshot(
                    pid=pid,
                    create_time=float(process.create_time()),
                    pgid=int(os.getpgid(pid)),
                    cwd=str(Path(process.cwd()).resolve()),
                    label=str(command[0]),
                    command=command,
                )
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, OSError, RuntimeError, ValueError):
            continue
    return tuple(rows)


def _snapshot_matches(snapshot: ProcessSnapshot, identity: Mapping[str, Any]) -> bool:
    pid = identity.get("pid")
    created = identity.get("create_time")
    return (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid == snapshot.pid
        and isinstance(created, int | float)
        and not isinstance(created, bool)
        and math.isfinite(float(created))
        and math.isclose(
            float(created),
            snapshot.create_time,
            rel_tol=0.0,
            abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
        )
    )


def _empty_row() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "returncode": None,
        "started_at": None,
        "finished_at": None,
        "artifacts": [],
        "last_problem": None,
        "process": None,
        "launch_requested_at": None,
        "launched_pid": None,
    }


def _artifact(repo_root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(repo_root)),
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "all_ok": True,
        "problems": [],
    }


def _program_binding(program: Program) -> dict[str, str]:
    return {
        "path": str(program.path),
        "file_sha256": program.file_sha256,
        "program_sha256": program.program_sha256,
    }


def _validate_completed_artifact(
    program: Program,
    capsule: Any,
    expectation: Any,
    report: object,
) -> None:
    payload_path = _safe_current_file(
        program.repo_root,
        expectation.path,
        f"{program.program_id} capsule {capsule.capsule_id} artifact",
    )
    payload = _read_json(
        payload_path,
        f"{program.program_id} capsule {capsule.capsule_id} artifact",
    )
    if payload.get("schema") != expectation.schema:
        raise FullGenerationsExtensionRefused(
            f"{program.program_id} capsule {capsule.capsule_id} artifact schema drifted"
        )
    for dotted, expected in expectation.fields:
        if _dotted(payload, dotted) != expected:
            raise FullGenerationsExtensionRefused(
                f"{program.program_id} capsule {capsule.capsule_id} artifact field {dotted} drifted"
            )
    if expectation.seal_field is not None:
        _validate_seal(
            payload,
            expectation.seal_field,
            f"{program.program_id} capsule {capsule.capsule_id} artifact",
        )
    expected_report = {
        "path": expectation.path,
        "sha256": sha256_file(payload_path),
        "schema": payload.get("schema"),
        "problems": [],
        "all_ok": True,
    }
    if report != expected_report:
        raise FullGenerationsExtensionRefused(
            f"{program.program_id} capsule {capsule.capsule_id} artifact report drifted"
        )


def validate_generic_status(
    program: Program,
    status: Mapping[str, Any],
    *,
    require_complete: bool = False,
) -> str:
    """Validate an exact zero-injection generic-supervisor status and artifacts."""

    _require_fields(status, GENERIC_STATUS_FIELDS, f"{program.program_id} status")
    _validate_seal(status, "status_sha256", f"{program.program_id} status")
    state = status.get("state")
    if (
        status.get("schema") != GENERIC_STATUS_SCHEMA
        or status.get("program_id") != program.program_id
        or status.get("program") != _program_binding(program)
        or not isinstance(state, str)
        or state not in GENERIC_STATES
        or (require_complete and state != "complete")
    ):
        raise FullGenerationsExtensionRefused(
            f"{program.program_id} status identity, program binding, or state drifted"
        )
    execution_enabled = status.get("execution_enabled")
    if execution_enabled is not True and not (state == "execution_disabled" and execution_enabled is False):
        raise FullGenerationsExtensionRefused(f"{program.program_id} execution authority drifted")
    supervisor = status.get("supervisor")
    implementation = _generic_supervisor_authority(program.repo_root)
    if (
        not isinstance(status.get("created_at"), str)
        or not isinstance(supervisor, Mapping)
        or set(supervisor) != SUPERVISOR_FIELDS
        or isinstance(supervisor.get("pid"), bool)
        or not isinstance(supervisor.get("pid"), int)
        or supervisor["pid"] <= 0
        or isinstance(supervisor.get("create_time"), bool)
        or not isinstance(supervisor.get("create_time"), int | float)
        or not math.isfinite(float(supervisor["create_time"]))
        or float(supervisor["create_time"]) <= 0
        or supervisor.get("implementation_path") != implementation["path"]
        or supervisor.get("implementation_sha256") != implementation["sha256"]
        or (
            status.get("lane_reservation") is not None
            and not isinstance(status.get("lane_reservation"), Mapping)
        )
        or (
            status.get("last_admission") is not None and not isinstance(status.get("last_admission"), Mapping)
        )
    ):
        raise FullGenerationsExtensionRefused(f"{program.program_id} status metadata drifted")
    problems = status.get("problems")
    if not isinstance(problems, list) or any(
        not isinstance(problem, str) or not problem for problem in problems
    ):
        raise FullGenerationsExtensionRefused(f"{program.program_id} problem inventory drifted")
    genesis = canonical_sha256(
        {
            "program_sha256": program.program_sha256,
            "base_capsules": [capsule.capsule_sha256 for capsule in program.capsules],
        }
    )
    if (
        status.get("queue_head_sha256") != genesis
        or status.get("accepted_injection_count") != 0
        or status.get("next_injection_sequence") != 1
    ):
        raise FullGenerationsExtensionRefused(f"{program.program_id} queue or injection authority drifted")
    expected_capsules = {capsule.capsule_id: capsule for capsule in program.capsules}
    capsules = status.get("capsules")
    if not isinstance(capsules, Mapping) or set(capsules) != set(expected_capsules):
        raise FullGenerationsExtensionRefused(f"{program.program_id} capsule inventory drifted")
    current_capsule = status.get("current_capsule")
    if current_capsule is not None and current_capsule not in expected_capsules:
        raise FullGenerationsExtensionRefused(f"{program.program_id} active capsule drifted")
    for capsule_id, capsule in expected_capsules.items():
        row = _require_fields(
            capsules[capsule_id],
            GENERIC_CAPSULE_FIELDS,
            f"{program.program_id} capsule {capsule_id}",
        )
        expected_identity = {
            "id": capsule.capsule_id,
            "kind": capsule.kind,
            "priority": capsule.priority,
            "depends_on": list(capsule.depends_on),
            "capsule_sha256": capsule.capsule_sha256,
            "source": "base",
        }
        if any(row.get(key) != expected for key, expected in expected_identity.items()):
            raise FullGenerationsExtensionRefused(
                f"{program.program_id} capsule {capsule_id} identity drifted"
            )
        row_state = row.get("status")
        attempts = row.get("attempts")
        if (
            row_state not in {"pending", "running", "complete", "failed"}
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or not isinstance(row.get("artifacts"), list)
            or not isinstance(row.get("runtime"), Mapping)
        ):
            raise FullGenerationsExtensionRefused(
                f"{program.program_id} capsule {capsule_id} receipt shape drifted"
            )
        if row_state == "complete":
            reports = row.get("artifacts")
            if (
                attempts <= 0
                or row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("last_problem") is not None
                or not isinstance(reports, list)
                or len(reports) != len(capsule.artifacts)
            ):
                raise FullGenerationsExtensionRefused(
                    f"{program.program_id} capsule {capsule_id} completion drifted"
                )
            for expectation, report in zip(capsule.artifacts, reports, strict=True):
                _validate_completed_artifact(program, capsule, expectation, report)
    if state == "complete" and (
        problems != []
        or status.get("current_capsule") is not None
        or status.get("lane_reservation") is not None
        or any(row.get("status") != "complete" for row in capsules.values())
    ):
        raise FullGenerationsExtensionRefused(f"completed {program.program_id} status is not clean")
    return state


def _stable_complete_generic_status(
    program: Program,
    *,
    acquire_lock: bool = True,
) -> dict[str, Any]:
    if acquire_lock:
        for attempt in range(LOCK_ATTEMPTS):
            try:
                with FileLock(program.program_root / GENERIC_LOCK_FILE):
                    return _stable_complete_generic_status(program, acquire_lock=False)
            except Generation1Refused as exc:
                if not isinstance(exc.__cause__, BlockingIOError):
                    raise FullGenerationsExtensionRefused(str(exc)) from exc
                if attempt + 1 < LOCK_ATTEMPTS:
                    time.sleep(LOCK_INTERVAL_SECONDS)
                    continue
                raise FullGenerationsExtensionRefused(
                    f"{program.program_id} supervisor lock remained held during completion replay"
                ) from exc
    for attempt in range(SNAPSHOT_ATTEMPTS):
        before = _read_json(program.status_path, f"{program.program_id} status")
        validate_generic_status(program, before, require_complete=True)
        after = _read_json(program.status_path, f"{program.program_id} status")
        validate_generic_status(program, after, require_complete=True)
        if before == after:
            return before
        if attempt + 1 < SNAPSHOT_ATTEMPTS:
            time.sleep(SNAPSHOT_INTERVAL_SECONDS)
    raise FullGenerationsExtensionRefused(f"{program.program_id} complete status did not become stable")


def _default_host_admission(
    program: Program,
    status: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    completed: set[str] = set()
    if status is not None:
        capsules = status.get("capsules")
        if isinstance(capsules, Mapping):
            completed = {
                capsule_id
                for capsule_id, row in capsules.items()
                if isinstance(row, Mapping) and row.get("status") == "complete"
            }
    candidate = next(
        (capsule for capsule in program.capsules if capsule.capsule_id not in completed),
        None,
    )
    if candidate is None:
        return {"allowed": True, "reason": "target already has no incomplete base capsule"}
    try:
        admission = Generation1Supervisor(
            program,
            execute=False,
        )._default_admission(candidate)  # noqa: SLF001 - exact generic admission boundary
    except (Generation1Refused, OSError, RuntimeError, ValueError) as exc:
        raise FullGenerationsExtensionRefused(
            f"full-generations host admission failed closed: {type(exc).__name__}: {exc}"
        ) from exc
    receipt = dict(admission.receipt)
    receipt["allowed"] = bool(admission.allowed)
    return receipt


def _admission_summary(
    receipt: Mapping[str, Any],
    *,
    checked_at: str,
) -> dict[str, Any]:
    allowed = receipt.get("allowed")
    aggregate = receipt.get("aggregate")
    denied: list[str] = []
    reason: object = receipt.get("reason")
    if isinstance(aggregate, Mapping):
        values = aggregate.get("denied_reasons")
        if isinstance(values, list):
            denied = [str(value) for value in values if str(value)]
        reason = aggregate.get("reason", reason)
    if allowed is not True and allowed is not False:
        raise FullGenerationsExtensionRefused("full-generations host admission returned no exact decision")
    return {
        "allowed": allowed,
        "checked_at": checked_at,
        "reason": str(reason or ("host admission allowed" if allowed else "host is busy")),
        "denied_reasons": denied,
    }


def _status_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    capsules = state["capsules"]
    completed = sum(row.get("status") == "complete" for row in capsules.values())
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "finished_at": state["finished_at"],
        "execution_enabled": state["execution_enabled"],
        "state": state["status"],
        "supervisor": state["supervisor"],
        "parent_implementation": state["parent_implementation"],
        "predecessor_program": state["predecessor_program"],
        "target_program": state["target_program"],
        "capsules": state["capsules"],
        "counts": {
            "complete": completed,
            "total": len(capsules),
            "remaining": len(capsules) - completed,
        },
        "last_admission": state["last_admission"],
        "problems": state["problems"],
        "signals_allowed": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": canonical_sha256(core)}


class FullGenerationsExtensionChain:
    """Observe categorized wave completion, then own the full-generations generic supervisor."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        predecessor_program_path: Path = DEFAULT_PREDECESSOR_PROGRAM,
        target_program_path: Path = DEFAULT_TARGET_PROGRAM,
        execute: bool = False,
        identity_probe_fn: IdentityProbe = _default_identity_probe,
        process_table_fn: ProcessTable = _default_process_table,
        program_loader_fn: ProgramLoader = load_program,
        status_reader_fn: StatusReader = read_status,
        target_starter_fn: SupervisorStarter = start_detached,
        host_admission_fn: AdmissionProbe = _default_host_admission,
        now_fn: NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.predecessor_program_path = predecessor_program_path.resolve()
        self.target_program_path = target_program_path.resolve()
        self.execute = execute
        self.identity_probe_fn = identity_probe_fn
        self.process_table_fn = process_table_fn
        self.program_loader_fn = program_loader_fn
        self.status_reader_fn = status_reader_fn
        self.target_starter_fn = target_starter_fn
        self.host_admission_fn = host_admission_fn
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.poll_seconds = poll_seconds
        for label, path in (
            ("full-generations extension root", self.root),
            ("predecessor program", self.predecessor_program_path),
            ("target program", self.target_program_path),
        ):
            if not path.is_relative_to(self.repo_root):
                raise FullGenerationsExtensionRefused(f"{label} must remain inside the repository")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def status_path(self) -> Path:
        return self.root / STATUS_FILE

    def _load_program(self, path: Path, expected_id: str) -> Program:
        try:
            program = self.program_loader_fn(path, repo_root=self.repo_root)
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise FullGenerationsExtensionRefused(
                f"{expected_id} program is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        if program.program_id != expected_id:
            raise FullGenerationsExtensionRefused(f"loaded {expected_id} program identity drifted")
        return program

    def _predecessor_authority(self) -> dict[str, Any]:
        return _program_authority(
            self.predecessor_program_path,
            repo_root=self.repo_root,
            expected_program_id=PREDECESSOR_PROGRAM_ID,
            signals_allowed=False,
        )

    def _target_authority(self) -> dict[str, Any]:
        return _program_authority(
            self.target_program_path,
            repo_root=self.repo_root,
            expected_program_id=TARGET_PROGRAM_ID,
        )

    def _validate_authorities(self, state: Mapping[str, Any]) -> None:
        if state.get("parent_implementation") != _implementation_authority(self.repo_root):
            raise FullGenerationsExtensionRefused(
                "full-generations extension implementation authority drifted"
            )
        if state.get("predecessor_program") != self._predecessor_authority():
            raise FullGenerationsExtensionRefused("full-generations extension predecessor authority drifted")
        if state.get("target_program") != self._target_authority():
            raise FullGenerationsExtensionRefused("full-generations extension target authority drifted")

    def _initial_state(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        return {
            "schema": STATE_SCHEMA,
            "program_id": PROGRAM_ID,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "execution_enabled": self.execute,
            "status": "starting",
            "supervisor": _self_identity(self.repo_root),
            "parent_implementation": _implementation_authority(self.repo_root),
            "predecessor_program": self._predecessor_authority(),
            "target_program": self._target_authority(),
            "capsules": {
                "categorized_batch_wave_v1": _empty_row(),
                "full_generations_wave_v1": _empty_row(),
            },
            "last_admission": None,
            "problems": [],
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        state = _read_json(self.state_path, "full-generations extension state")
        _validate_seal(state, "state_sha256", "full-generations extension state")
        if set(state) != STATE_FIELDS:
            raise FullGenerationsExtensionRefused("full-generations extension state fields drifted")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != PROGRAM_ID:
            raise FullGenerationsExtensionRefused("full-generations extension state identity drifted")
        capsules = state.get("capsules")
        if not isinstance(capsules, Mapping) or set(capsules) != {
            "categorized_batch_wave_v1",
            "full_generations_wave_v1",
        }:
            raise FullGenerationsExtensionRefused("full-generations extension capsule inventory drifted")
        self._validate_authorities(state)
        state.pop("state_sha256", None)
        state["execution_enabled"] = self.execute
        state["supervisor"] = _self_identity(self.repo_root)
        return state

    def _append_problem(self, problem: str) -> None:
        problems = list(self.state.get("problems") or [])
        if not problems or problems[-1] != problem:
            problems.append(problem)
        self.state["problems"] = problems[-64:]

    def _publish(self) -> dict[str, Any]:
        self.state["updated_at"] = _iso(self.now_fn())
        core = dict(self.state)
        core.pop("state_sha256", None)
        sealed = {**core, "state_sha256": canonical_sha256(core)}
        atomic_write_json(self.state_path, sealed)
        self.state = dict(sealed)
        status = _status_payload(self.state)
        atomic_write_json(self.status_path, status)
        return status

    def _complete_predecessor(self, *, acquire_lock: bool = True) -> tuple[Program, dict[str, Any]]:
        program = self._load_program(
            self.predecessor_program_path,
            PREDECESSOR_PROGRAM_ID,
        )
        return program, _stable_complete_generic_status(
            program,
            acquire_lock=acquire_lock,
        )

    def _reconcile_predecessor(self, row: dict[str, Any]) -> bool:
        program = self._load_program(
            self.predecessor_program_path,
            PREDECESSOR_PROGRAM_ID,
        )
        if not program.status_path.exists():
            if row.get("status") == "complete":
                raise FullGenerationsExtensionRefused(
                    "completed categorized wave predecessor evidence disappeared"
                )
            row["status"] = "pending"
            row["last_problem"] = "waiting for exact clean categorized-batch-wave-v1 completion"
            return False
        raw = _read_json(program.status_path, "categorized batch wave status")
        state = validate_generic_status(program, raw)
        if state != "complete":
            if row.get("status") == "complete":
                raise FullGenerationsExtensionRefused(
                    "completed categorized wave predecessor evidence regressed"
                )
            if state in UNSAFE_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"categorized batch wave entered {state}"
                return False
            row["status"] = "observed"
            row["started_at"] = row.get("started_at") or _iso(self.now_fn())
            row["last_problem"] = f"observed categorized batch wave state {state}; signals remain forbidden"
            return False
        complete = _stable_complete_generic_status(program)
        expected = [_artifact(self.repo_root, program.status_path, complete)]
        if row.get("status") == "complete":
            if (
                row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("artifacts") != expected
                or row.get("last_problem") is not None
            ):
                raise FullGenerationsExtensionRefused(
                    "completed categorized wave predecessor binding drifted"
                )
            return True
        row.update(
            {
                "status": "complete",
                "returncode": 0,
                "finished_at": _iso(self.now_fn()),
                "artifacts": expected,
                "last_problem": None,
            }
        )
        return True

    def _target_status(self, program: Program) -> dict[str, Any] | None:
        if not program.status_path.exists():
            return None
        try:
            status = dict(self.status_reader_fn(program))
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise FullGenerationsExtensionRefused(
                f"full-generations target status is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        validate_generic_status(program, status)
        return status

    def _visible_target_supervisors(self, program: Program) -> tuple[ProcessSnapshot, ...]:
        label = f"mop-supervisor:{TARGET_PROGRAM_ID}"
        visible = tuple(
            process
            for process in self.process_table_fn()
            if process.pid != os.getpid()
            and (process.label == label or _runs_generic_supervisor_command(process, program))
        )
        if any(
            process.cwd != str(self.repo_root)
            or process.pgid <= 0
            or process.create_time <= 0
            or not math.isfinite(process.create_time)
            for process in visible
        ):
            raise FullGenerationsExtensionRefused("visible full-generations supervisor has inexact metadata")
        if len(visible) > 1:
            raise FullGenerationsExtensionRefused("multiple full-generations generic supervisors are visible")
        return visible

    def _record_admission(
        self,
        program: Program,
        status: Mapping[str, Any] | None,
    ) -> bool:
        receipt = self.host_admission_fn(program, status)
        if not isinstance(receipt, Mapping):
            raise FullGenerationsExtensionRefused("full-generations host admission returned no receipt")
        summary = _admission_summary(
            receipt,
            checked_at=_iso(self.now_fn()),
        )
        self.state["last_admission"] = summary
        return bool(summary["allowed"])

    def _reconcile_target(self, row: dict[str, Any], program: Program) -> dict[str, Any]:
        status = self._target_status(program)
        visible = self._visible_target_supervisors(program)
        if row.get("status") == "complete":
            if status is None:
                raise FullGenerationsExtensionRefused("completed full-generations target status disappeared")
            complete = _stable_complete_generic_status(program)
            supervisor = complete.get("supervisor")
            if visible and (
                not isinstance(supervisor, Mapping) or not _snapshot_matches(visible[0], supervisor)
            ):
                raise FullGenerationsExtensionRefused(
                    "completed full-generations target conflicts with visible supervisor"
                )
            expected = [_artifact(self.repo_root, program.status_path, complete)]
            if (
                row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("artifacts") != expected
                or row.get("last_problem") is not None
            ):
                raise FullGenerationsExtensionRefused("completed full-generations target binding drifted")
            return row
        if status is not None:
            state = str(status["state"])
            if state == "complete":
                complete = _stable_complete_generic_status(program)
                supervisor = complete.get("supervisor")
                if visible and (
                    not isinstance(supervisor, Mapping) or not _snapshot_matches(visible[0], supervisor)
                ):
                    raise FullGenerationsExtensionRefused(
                        "completed full-generations target conflicts with visible supervisor"
                    )
                row.update(
                    {
                        "status": "complete",
                        "returncode": 0,
                        "finished_at": _iso(self.now_fn()),
                        "artifacts": [_artifact(self.repo_root, program.status_path, complete)],
                        "last_problem": None,
                    }
                )
                return row
            if state in UNSAFE_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"full-generations target entered {state}"
                return row
            supervisor = status.get("supervisor")
            if isinstance(supervisor, Mapping):
                identity_state = self.identity_probe_fn(supervisor)
                if identity_state == "unknown":
                    raise FullGenerationsExtensionRefused("full-generations supervisor identity is inexact")
                if identity_state == "alive":
                    if len(visible) != 1 or not _snapshot_matches(visible[0], supervisor):
                        raise FullGenerationsExtensionRefused(
                            "sealed full-generations supervisor conflicts with visible process"
                        )
                    row["status"] = "running"
                    row["process"] = dict(supervisor)
                    row["started_at"] = row.get("started_at") or _iso(self.now_fn())
                    row["last_problem"] = f"full-generations generic supervisor is in state {state}"
                    return row
        if visible:
            row["status"] = "adoption_wait"
            row["process"] = visible[0].identity()
            row["last_problem"] = (
                "exact full-generations supervisor is visible while sealed acknowledgement is pending"
            )
            return row
        if not self.execute:
            row["status"] = "pending"
            row["last_problem"] = "full-generations target start requires explicit --execute"
            return row

        predecessor = self._load_program(
            self.predecessor_program_path,
            PREDECESSOR_PROGRAM_ID,
        )
        with FileLock(predecessor.program_root / GENERIC_LOCK_FILE):
            before = _stable_complete_generic_status(predecessor, acquire_lock=False)
            if not self._record_admission(program, status):
                row["status"] = "pending"
                row["last_problem"] = f"host busy: {self.state['last_admission']['reason']}"
                return row
            row["status"] = "launching"
            row["attempts"] = int(row.get("attempts") or 0) + 1
            row["launch_requested_at"] = _iso(self.now_fn())
            row["last_problem"] = "sealed full-generations launch intent; replaying all gates"
            self.state["status"] = "waiting_target"
            self.state["finished_at"] = None
            self._publish()
            row = self.state["capsules"]["full_generations_wave_v1"]
            after = _stable_complete_generic_status(predecessor, acquire_lock=False)
            if before != after:
                raise FullGenerationsExtensionRefused(
                    "categorized wave predecessor changed across full-generations launch intent"
                )
            if not self._record_admission(program, status):
                row["status"] = "pending"
                row["last_problem"] = (
                    f"host became busy before launch: {self.state['last_admission']['reason']}"
                )
                return row
            try:
                launched = self.target_starter_fn(
                    program,
                    execute=True,
                    use_caffeinate=True,
                )
            except (Generation1Refused, OSError, RuntimeError, ValueError) as exc:
                row["status"] = "retry_wait"
                row["last_problem"] = (
                    f"full-generations startup acknowledgement pending: {type(exc).__name__}: {exc}"
                )
                return row
            if not isinstance(launched, Mapping):
                raise FullGenerationsExtensionRefused("full-generations generic starter returned no receipt")
            returned = launched.get("status")
            if not isinstance(returned, Mapping):
                row["status"] = "retry_wait"
                row["last_problem"] = "full-generations starter returned no sealed status acknowledgement"
                return row
            returned_state = validate_generic_status(program, returned)
            if returned_state in UNSAFE_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"full-generations target acknowledged unsafe state {returned_state}"
                return row
            launched_pid = launched.get("launched_pid")
            row["launched_pid"] = (
                launched_pid if isinstance(launched_pid, int) and not isinstance(launched_pid, bool) else None
            )
            row["status"] = "running"
            row["started_at"] = row.get("started_at") or _iso(self.now_fn())
            row["last_problem"] = "full-generations generic supervisor start accepted"
            supervisor = returned.get("supervisor")
            if isinstance(supervisor, Mapping):
                row["process"] = dict(supervisor)
            return row

    def _validate_terminal_artifacts(self, target_program: Program) -> None:
        predecessor = self.state["capsules"]["categorized_batch_wave_v1"]
        if predecessor.get("status") == "complete":
            self._reconcile_predecessor(predecessor)
        target = self.state["capsules"]["full_generations_wave_v1"]
        if target.get("status") == "complete":
            self._reconcile_target(target, target_program)

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_authorities(self.state)
            target_program = self._load_program(
                self.target_program_path,
                TARGET_PROGRAM_ID,
            )
            if self.state.get("status") in TERMINAL_STATES and self.state.get("status") != "complete":
                self._validate_terminal_artifacts(target_program)
                return self._publish()
            predecessor_row = self.state["capsules"]["categorized_batch_wave_v1"]
            predecessor_complete = self._reconcile_predecessor(predecessor_row)
            if predecessor_row.get("status") == "failure_hold":
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem(str(predecessor_row.get("last_problem")))
                return self._publish()
            if not predecessor_complete:
                self.state["status"] = "waiting_predecessor"
                return self._publish()
            target_row = self.state["capsules"]["full_generations_wave_v1"]
            target_row = self._reconcile_target(target_row, target_program)
            if target_row.get("status") == "complete":
                self.state["status"] = "complete"
                self.state["finished_at"] = _iso(self.now_fn())
            elif target_row.get("status") == "failure_hold":
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem(str(target_row.get("last_problem")))
            elif target_row.get("status") == "retry_wait":
                self.state["status"] = "retry_wait"
            elif (
                target_row.get("status") == "pending"
                and isinstance(self.state.get("last_admission"), Mapping)
                and self.state["last_admission"].get("allowed") is False
            ):
                self.state["status"] = "waiting_host"
            else:
                self.state["status"] = "waiting_target"
            return self._publish()
        except (
            FullGenerationsExtensionRefused,
            Generation1Refused,
            AttributeError,
            IndexError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self.state["status"] = "integrity_hold"
            self.state["finished_at"] = _iso(self.now_fn())
            self._append_problem(f"{type(exc).__name__}: {exc}")
            return self._publish()

    def run(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        with FileLock(self.root / LOCK_FILE):
            self.state = self._load_state()
            self.state["supervisor"] = _self_identity(self.repo_root)
            while True:
                status = self.tick()
                cycles += 1
                if status["state"] in TERMINAL_STATES:
                    return status
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                self.sleep_fn(self.poll_seconds)


def _validate_status_authorities(
    status: Mapping[str, Any],
    *,
    repo_root: Path,
    predecessor_program_path: Path,
    target_program_path: Path,
) -> None:
    if status.get("parent_implementation") != _implementation_authority(repo_root):
        raise FullGenerationsExtensionRefused(
            "full-generations extension sealed implementation authority drifted"
        )
    if status.get("predecessor_program") != _program_authority(
        predecessor_program_path,
        repo_root=repo_root,
        expected_program_id=PREDECESSOR_PROGRAM_ID,
        signals_allowed=False,
    ):
        raise FullGenerationsExtensionRefused(
            "full-generations extension sealed predecessor authority drifted"
        )
    if status.get("target_program") != _program_authority(
        target_program_path,
        repo_root=repo_root,
        expected_program_id=TARGET_PROGRAM_ID,
    ):
        raise FullGenerationsExtensionRefused("full-generations extension sealed target authority drifted")


def validate_full_generations_extension_status(
    status: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    predecessor_program_path: Path = DEFAULT_PREDECESSOR_PROGRAM,
    target_program_path: Path = DEFAULT_TARGET_PROGRAM,
) -> str:
    """Validate one exact full-generations-waiter acknowledgement."""

    repo_root = repo_root.resolve()
    predecessor_program_path = predecessor_program_path.resolve()
    target_program_path = target_program_path.resolve()
    _require_fields(status, STATUS_FIELDS, "full-generations extension status")
    _validate_seal(status, "status_sha256", "full-generations extension status")
    state = status.get("state")
    problems = status.get("problems")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("program_id") != PROGRAM_ID
        or status.get("signals_allowed") is not False
        or status.get("activation_allowed") is not False
        or status.get("scientific_promotion") is not False
        or not isinstance(status.get("execution_enabled"), bool)
        or not isinstance(state, str)
        or state not in EXTENSION_STATES
        or not isinstance(problems, list)
        or any(not isinstance(problem, str) or not problem for problem in problems)
    ):
        raise FullGenerationsExtensionRefused("full-generations extension status identity or safety drifted")
    if (state == "complete" or state not in TERMINAL_STATES) and (
        status.get("execution_enabled") is not True or problems != []
    ):
        raise FullGenerationsExtensionRefused("full-generations extension safe acknowledgement is not clean")
    _validate_status_authorities(
        status,
        repo_root=repo_root,
        predecessor_program_path=predecessor_program_path,
        target_program_path=target_program_path,
    )
    created = _parse_time(status.get("created_at"))
    updated = _parse_time(status.get("updated_at"))
    finished = _parse_time(status.get("finished_at"))
    if created is None or updated is None or created > updated:
        raise FullGenerationsExtensionRefused("full-generations extension timestamps drifted")
    if state in TERMINAL_STATES:
        if finished is None or finished > updated:
            raise FullGenerationsExtensionRefused("full-generations extension terminal timestamp drifted")
    elif status.get("finished_at") is not None:
        raise FullGenerationsExtensionRefused(
            "full-generations extension nonterminal status is marked finished"
        )
    supervisor = status.get("supervisor")
    implementation = _implementation_authority(repo_root)
    if (
        not isinstance(supervisor, Mapping)
        or set(supervisor) != SUPERVISOR_FIELDS
        or isinstance(supervisor.get("pid"), bool)
        or not isinstance(supervisor.get("pid"), int)
        or supervisor["pid"] <= 0
        or isinstance(supervisor.get("create_time"), bool)
        or not isinstance(supervisor.get("create_time"), int | float)
        or not math.isfinite(float(supervisor["create_time"]))
        or float(supervisor["create_time"]) <= 0
        or supervisor.get("implementation_path") != implementation["path"]
        or supervisor.get("implementation_sha256") != implementation["sha256"]
    ):
        raise FullGenerationsExtensionRefused("full-generations extension supervisor authority drifted")
    last_admission = status.get("last_admission")
    if last_admission is not None and (
        not isinstance(last_admission, Mapping)
        or set(last_admission) != {"allowed", "checked_at", "reason", "denied_reasons"}
        or not isinstance(last_admission.get("allowed"), bool)
        or _parse_time(last_admission.get("checked_at")) is None
        or not isinstance(last_admission.get("reason"), str)
        or not last_admission["reason"]
        or not isinstance(last_admission.get("denied_reasons"), list)
        or any(not isinstance(reason, str) or not reason for reason in last_admission["denied_reasons"])
    ):
        raise FullGenerationsExtensionRefused("full-generations extension admission summary drifted")
    capsules = status.get("capsules")
    expected_ids = {"categorized_batch_wave_v1", "full_generations_wave_v1"}
    if not isinstance(capsules, Mapping) or set(capsules) != expected_ids:
        raise FullGenerationsExtensionRefused("full-generations extension capsule inventory drifted")
    completed = 0
    for capsule_id, row in capsules.items():
        _require_fields(row, ROW_FIELDS, f"full-generations extension capsule {capsule_id}")
        row_state = row.get("status")
        attempts = row.get("attempts")
        artifacts = row.get("artifacts")
        returncode = row.get("returncode")
        started_at = _parse_time(row.get("started_at"))
        finished_at = _parse_time(row.get("finished_at"))
        launch_requested_at = _parse_time(row.get("launch_requested_at"))
        last_problem = row.get("last_problem")
        process = row.get("process")
        launched_pid = row.get("launched_pid")
        if (
            row_state not in ROW_STATES
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or not isinstance(artifacts, list)
            or any(not isinstance(report, Mapping) for report in artifacts)
            or (returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)))
            or (row.get("started_at") is not None and started_at is None)
            or (row.get("finished_at") is not None and finished_at is None)
            or (row.get("launch_requested_at") is not None and launch_requested_at is None)
            or (started_at is not None and finished_at is not None and started_at > finished_at)
            or (last_problem is not None and (not isinstance(last_problem, str) or not last_problem))
            or (process is not None and not isinstance(process, Mapping))
            or (
                launched_pid is not None
                and (isinstance(launched_pid, bool) or not isinstance(launched_pid, int) or launched_pid <= 0)
            )
        ):
            raise FullGenerationsExtensionRefused(
                f"full-generations extension capsule {capsule_id} receipt drifted"
            )
        if isinstance(process, Mapping):
            process_pid = process.get("pid")
            process_created = process.get("create_time")
            if (
                isinstance(process_pid, bool)
                or not isinstance(process_pid, int)
                or process_pid <= 0
                or isinstance(process_created, bool)
                or not isinstance(process_created, int | float)
                or not math.isfinite(float(process_created))
                or float(process_created) <= 0
            ):
                raise FullGenerationsExtensionRefused(
                    f"full-generations extension capsule {capsule_id} process identity drifted"
                )
        if row_state == "complete":
            completed += 1
            if returncode != 0 or finished_at is None or not artifacts or last_problem is not None:
                raise FullGenerationsExtensionRefused(
                    f"full-generations extension capsule {capsule_id} completion drifted"
                )
        elif (
            returncode is not None
            or artifacts
            or (row_state == "failure_hold" and (finished_at is None or not isinstance(last_problem, str)))
            or (row_state != "failure_hold" and finished_at is not None)
        ):
            raise FullGenerationsExtensionRefused(
                f"full-generations extension capsule {capsule_id} noncompletion drifted"
            )
    counts = {
        "complete": completed,
        "total": len(expected_ids),
        "remaining": len(expected_ids) - completed,
    }
    if status.get("counts") != counts:
        raise FullGenerationsExtensionRefused("full-generations extension counts drifted")
    predecessor_state = capsules["categorized_batch_wave_v1"].get("status")
    target_state = capsules["full_generations_wave_v1"].get("status")
    pristine_target = capsules["full_generations_wave_v1"] == _empty_row()
    all_complete = completed == 2
    coherent = True
    if state == "starting":
        coherent = all(row == _empty_row() for row in capsules.values())
    elif state == "waiting_predecessor":
        coherent = predecessor_state in {"pending", "observed"} and pristine_target
    elif state == "waiting_host":
        admission = status.get("last_admission")
        coherent = (
            predecessor_state == "complete"
            and target_state == "pending"
            and isinstance(admission, Mapping)
            and admission.get("allowed") is False
        )
    elif state == "waiting_target":
        coherent = (
            predecessor_state == "complete"
            and target_state
            in {
                "pending",
                "running",
                "adoption_wait",
                "launching",
            }
            and not (
                target_state == "pending"
                and isinstance(last_admission, Mapping)
                and last_admission.get("allowed") is False
            )
        )
    elif state == "retry_wait":
        coherent = predecessor_state == "complete" and target_state == "retry_wait"
    elif state == "complete":
        coherent = all_complete
    elif state == "failure_hold":
        coherent = "failure_hold" in {predecessor_state, target_state}
    if not coherent or (state not in TERMINAL_STATES and all_complete):
        raise FullGenerationsExtensionRefused(
            "full-generations extension state and capsule progression drifted"
        )
    completed_bindings = (
        (
            "categorized_batch_wave_v1",
            predecessor_program_path,
            PREDECESSOR_PROGRAM_ID,
        ),
        (
            "full_generations_wave_v1",
            target_program_path,
            TARGET_PROGRAM_ID,
        ),
    )
    for capsule_id, program_path, expected_program_id in completed_bindings:
        if capsules[capsule_id].get("status") != "complete":
            continue
        try:
            program = load_program(program_path, repo_root=repo_root)
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise FullGenerationsExtensionRefused(
                f"{expected_program_id} completion binding is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        if program.program_id != expected_program_id:
            raise FullGenerationsExtensionRefused(
                f"{expected_program_id} completion program identity drifted"
            )
        completed_status = _stable_complete_generic_status(program)
        expected_artifacts = [
            _artifact(
                repo_root,
                program.status_path,
                completed_status,
            )
        ]
        if capsules[capsule_id].get("artifacts") != expected_artifacts:
            raise FullGenerationsExtensionRefused(
                f"full-generations extension capsule {capsule_id} completion binding drifted"
            )
    return state


def _read_stable_state_status(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for attempt in range(SNAPSHOT_ATTEMPTS):
        before = _read_json(root / STATE_FILE, "full-generations extension state")
        first = _read_json(root / STATUS_FILE, "full-generations extension status")
        after = _read_json(root / STATE_FILE, "full-generations extension state")
        second = _read_json(root / STATUS_FILE, "full-generations extension status")
        try:
            projection = _status_payload(before)
        except (AttributeError, KeyError, TypeError, ValueError):
            projection = None
        if before == after and first == second and first == projection:
            return before, first
        if attempt + 1 < SNAPSHOT_ATTEMPTS:
            time.sleep(SNAPSHOT_INTERVAL_SECONDS)
    raise FullGenerationsExtensionRefused("full-generations extension state/status did not become coherent")


def read_validated_complete_full_generations_extension_status(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path = REPO_ROOT,
    predecessor_program_path: Path = DEFAULT_PREDECESSOR_PROGRAM,
    target_program_path: Path = DEFAULT_TARGET_PROGRAM,
    acquire_lock: bool = True,
) -> dict[str, Any]:
    """Replay one stable complete full-generations extension snapshot."""

    root = root.resolve()
    repo_root = repo_root.resolve()
    if not root.is_dir() or not root.is_relative_to(repo_root):
        raise FullGenerationsExtensionRefused(
            "full-generations extension root is absent or outside the repository"
        )
    if acquire_lock:
        for attempt in range(LOCK_ATTEMPTS):
            try:
                with FileLock(root / LOCK_FILE):
                    return read_validated_complete_full_generations_extension_status(
                        root=root,
                        repo_root=repo_root,
                        predecessor_program_path=predecessor_program_path,
                        target_program_path=target_program_path,
                        acquire_lock=False,
                    )
            except Generation1Refused as exc:
                if not isinstance(exc.__cause__, BlockingIOError):
                    raise
                if attempt + 1 < LOCK_ATTEMPTS:
                    time.sleep(LOCK_INTERVAL_SECONDS)
                    continue
                raise FullGenerationsExtensionRefused(
                    "full-generations extension lifetime lock remained held"
                ) from exc
    state, status = _read_stable_state_status(root)
    if set(state) != STATE_FIELDS:
        raise FullGenerationsExtensionRefused("full-generations extension complete state fields drifted")
    _validate_seal(state, "state_sha256", "full-generations extension state")
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("program_id") != PROGRAM_ID
        or state.get("status") != "complete"
        or state.get("execution_enabled") is not True
        or state.get("problems") != []
    ):
        raise FullGenerationsExtensionRefused("full-generations extension complete state boundary drifted")
    if (
        validate_full_generations_extension_status(
            status,
            repo_root=repo_root,
            predecessor_program_path=predecessor_program_path,
            target_program_path=target_program_path,
        )
        != "complete"
    ):
        raise FullGenerationsExtensionRefused("full-generations extension durable status is not complete")
    predecessor = load_program(predecessor_program_path, repo_root=repo_root)
    target = load_program(target_program_path, repo_root=repo_root)
    predecessor_status = _stable_complete_generic_status(predecessor)
    target_status = _stable_complete_generic_status(target)
    expected = {
        "categorized_batch_wave_v1": [_artifact(repo_root, predecessor.status_path, predecessor_status)],
        "full_generations_wave_v1": [_artifact(repo_root, target.status_path, target_status)],
    }
    for capsule_id, artifacts in expected.items():
        if state["capsules"][capsule_id].get("artifacts") != artifacts:
            raise FullGenerationsExtensionRefused(
                f"full-generations extension capsule {capsule_id} terminal binding drifted"
            )
    final_state, final_status = _read_stable_state_status(root)
    if final_state != state or final_status != status:
        raise FullGenerationsExtensionRefused("full-generations extension changed during terminal replay")
    return status


def read_full_generations_extension_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    status = _read_json(root.resolve() / STATUS_FILE, "full-generations extension status")
    _validate_seal(status, "status_sha256", "full-generations extension status")
    return status


def _runs_parent_command(process: ProcessSnapshot, entrypoint: Path) -> bool:
    if len(process.command) < 3 or process.command[2] != "run":
        return False
    expected_python = (REPO_ROOT / ".venv/bin/python").resolve()
    actual_entrypoint = Path(process.command[1])
    if not actual_entrypoint.is_absolute():
        actual_entrypoint = Path(process.cwd) / actual_entrypoint
    try:
        return (
            Path(process.command[0]).resolve() == expected_python
            and actual_entrypoint.resolve() == entrypoint.resolve()
        )
    except OSError:
        return False


def _runs_generic_supervisor_command(
    process: ProcessSnapshot,
    program: Program,
) -> bool:
    if (
        len(process.command) != 6
        or process.command[2] != "run"
        or process.command[3] != "--program"
        or process.command[5] != "--execute"
    ):
        return False
    expected_python = (program.repo_root / ".venv/bin/python").resolve()
    expected_entrypoint = (program.repo_root / "scripts/mop_generation1_campaign.py").resolve()
    actual_entrypoint = Path(process.command[1])
    actual_program = Path(process.command[4])
    if not actual_entrypoint.is_absolute():
        actual_entrypoint = Path(process.cwd) / actual_entrypoint
    if not actual_program.is_absolute():
        actual_program = Path(process.cwd) / actual_program
    try:
        return (
            Path(process.command[0]).resolve() == expected_python
            and actual_entrypoint.resolve() == expected_entrypoint
            and actual_program.resolve() == program.path.resolve()
        )
    except OSError:
        return False


def _visible_parent(entrypoint: Path) -> ProcessSnapshot | None:
    candidates = [
        process
        for process in _default_process_table()
        if process.pid != os.getpid()
        and (process.label == PARENT_LABEL or _runs_parent_command(process, entrypoint))
    ]
    if not candidates:
        return None
    if any(
        process.cwd != str(REPO_ROOT.resolve())
        or process.pgid <= 0
        or process.create_time <= 0
        or not math.isfinite(process.create_time)
        for process in candidates
    ):
        raise FullGenerationsExtensionRefused(
            "visible full-generations extension parent has inexact metadata"
        )
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise FullGenerationsExtensionRefused("multiple full-generations extension parents are visible")
    if labelled:
        return labelled[0]
    return next(
        (process for process in candidates if process.pid == process.pgid),
        candidates[0],
    )


def start_full_generations_extension_detached(
    *,
    root: Path = DEFAULT_ROOT,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    """Idempotently start the lightweight full-generations extension parent."""

    if not execute:
        raise FullGenerationsExtensionRefused(
            "detached full-generations extension start requires explicit --execute"
        )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_generation1_full_generations_extension_chain.py").resolve()
        status = None
        if (root / STATUS_FILE).is_file():
            status = read_full_generations_extension_status(root)
            validate_full_generations_extension_status(status)
        visible = _visible_parent(entrypoint)
        identity = status.get("supervisor") if status is not None else None
        identity_state = _default_identity_probe(identity) if isinstance(identity, Mapping) else "gone"
        if identity_state == "unknown":
            raise FullGenerationsExtensionRefused(
                "sealed full-generations extension parent identity is inexact"
            )
        if identity_state == "alive":
            if (
                visible is None
                or not isinstance(identity, Mapping)
                or not _snapshot_matches(visible, identity)
            ):
                raise FullGenerationsExtensionRefused(
                    "sealed full-generations extension identity conflicts with visible parent"
                )
            return {"already_running": True, "status": status}
        if visible is not None:
            return {
                "already_running": True,
                "status": status,
                "observed_process": visible.identity(),
            }
        if status is not None and status.get("state") in TERMINAL_STATES:
            if status.get("state") == "complete":
                status = read_validated_complete_full_generations_extension_status(root=root)
            return {"already_terminal": True, "status": status}

        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(entrypoint),
            "run",
            "--execute",
            "--root",
            str(root),
        ]
        caffeinate = shutil.which("caffeinate") if use_caffeinate else None
        launched = [caffeinate, "-ims", *command] if caffeinate else command
        stdout_path = root / "logs/parent.stdout.log"
        stderr_path = root / "logs/parent.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        status_path = root / STATUS_FILE
        prior_mtime = status_path.stat().st_mtime_ns if status_path.exists() else 0
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MOP_PROCESS_LABEL"] = PARENT_LABEL
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
        deadline = time.monotonic() + STARTUP_ACK_SECONDS
        acknowledged = None
        while time.monotonic() < deadline:
            if status_path.is_file() and status_path.stat().st_mtime_ns > prior_mtime:
                try:
                    acknowledged = read_full_generations_extension_status(root)
                    validate_full_generations_extension_status(acknowledged)
                    break
                except (FullGenerationsExtensionRefused, OSError, ValueError):
                    acknowledged = None
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise FullGenerationsExtensionRefused(
                f"detached full-generations extension did not acknowledge startup; inspect {stderr_path}"
            )
        visible = _visible_parent(entrypoint)
        supervisor = acknowledged.get("supervisor")
        if (
            visible is None
            or not isinstance(supervisor, Mapping)
            or not _snapshot_matches(visible, supervisor)
        ):
            raise FullGenerationsExtensionRefused(
                "detached full-generations extension acknowledgement conflicts with visible parent"
            )
        if _default_identity_probe(visible.identity()) != "alive":
            raise FullGenerationsExtensionRefused(
                "detached full-generations extension visible parent is not provably alive"
            )
        return {
            "launched_pid": process.pid,
            "caffeinate": bool(caffeinate),
            "command": launched,
            "status": acknowledged,
            "observed_process": visible.identity(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "start", "status"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name in {"run", "start"}:
            child.add_argument("--execute", action="store_true")
        if name == "run":
            child.add_argument("--once", action="store_true")
        if name == "start":
            child.add_argument("--no-caffeinate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            if not arguments.execute:
                raise FullGenerationsExtensionRefused(
                    "full-generations extension execution requires explicit --execute"
                )
            set_process_label(PARENT_LABEL)
            payload = FullGenerationsExtensionChain(
                root=arguments.root,
                execute=True,
            ).run(max_cycles=1 if arguments.once else None)
        elif arguments.command == "start":
            payload = start_full_generations_extension_detached(
                root=arguments.root,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        else:
            payload = read_full_generations_extension_status(arguments.root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if arguments.command == "run" and payload.get("state") not in {
            "complete",
            "drained",
        }:
            return 2
        return 0
    except (
        FullGenerationsExtensionRefused,
        Generation1Refused,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"all_ok": False, "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2


__all__ = [
    "FullGenerationsExtensionChain",
    "FullGenerationsExtensionRefused",
    "DEFAULT_PREDECESSOR_PROGRAM",
    "DEFAULT_ROOT",
    "DEFAULT_TARGET_PROGRAM",
    "PARENT_LABEL",
    "PREDECESSOR_PROGRAM_ID",
    "PROGRAM_ID",
    "ProcessSnapshot",
    "STATE_SCHEMA",
    "STATUS_SCHEMA",
    "TARGET_PROGRAM_ID",
    "build_parser",
    "main",
    "read_full_generations_extension_status",
    "read_validated_complete_full_generations_extension_status",
    "start_full_generations_extension_detached",
    "validate_full_generations_extension_status",
    "validate_generic_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
