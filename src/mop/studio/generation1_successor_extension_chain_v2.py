"""Durable waiter for the second Generation 1 successor horizon.

The extension chain is intentionally observation-only with respect to the live
``generation1-successor-evidence-chain-v5`` parent.  It binds its own
implementation and the exact sealed v2 horizon manifest, waits for the v5
chain to complete, and then starts or resumes the v2 program through the
generic Generation 1 supervisor.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies import generation1_successor_horizon_v2 as horizon_v2
from mop.studio import generation1_successor_chain_v5 as predecessor_v5
from mop.studio import generation1_supervisor
from mop.studio.generation1_supervisor import (
    STATUS_SCHEMA as GENERIC_STATUS_SCHEMA,
)
from mop.studio.generation1_supervisor import (
    FileLock,
    Generation1Refused,
    Program,
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
)
from mop.studio.generation1_supervisor import (
    read_status as read_generation1_status,
)
from mop.studio.generation1_supervisor import (
    start_detached as start_generation1_detached,
)

PROGRAM_ID = "generation1-successor-extension-chain-v2"
PARENT_LABEL = "mop-successor-extension-chain-v2"
STATE_SCHEMA = "mop-generation1-successor-extension-chain-state/v2"
STATUS_SCHEMA = "mop-generation1-successor-extension-chain-status/v2"

PREDECESSOR_PROGRAM_ID = "generation1-successor-evidence-chain-v5"
PREDECESSOR_STATUS_SCHEMA = "mop-generation1-successor-evidence-chain-status/v5"
TARGET_PROGRAM_ID = "generation1-successor-horizon-v2"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_PREDECESSOR_STATUS = (
    REPO_ROOT / "runs/generation1/generation1-successor-evidence-chain-v5/current_status.json"
)
DEFAULT_TARGET_PROGRAM = REPO_ROOT / "configs/campaign/generation1_successor_horizon_v2.json"

STATE_FILE = "extension_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "control/extension.lock"
START_LOCK_FILE = "control/start.lock"
POLL_SECONDS = 30.0
STARTUP_ACK_SECONDS = 30.0
PROCESS_TIME_TOLERANCE_SECONDS = 0.02
MAX_JSON_BYTES = 64 * 1024 * 1024
SNAPSHOT_ATTEMPTS = 5
SNAPSHOT_INTERVAL_SECONDS = 0.05
LOCK_ATTEMPTS = 41
LOCK_INTERVAL_SECONDS = 0.05

TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
PREDECESSOR_TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
TARGET_TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
PREDECESSOR_STATES = frozenset(
    {
        "starting",
        "waiting_legacy",
        "adoption_wait",
        "waiting_horizon",
        *PREDECESSOR_TERMINAL_STATES,
    }
)
PREDECESSOR_CAPSULE_IDS = (
    "legacy_d1",
    "legacy_successor_mechanics",
    "legacy_final",
    "successor_horizon",
)
PREDECESSOR_CAPSULE_STATES = frozenset(
    {
        "pending",
        "running",
        "adopted",
        "adoption_wait",
        "launching",
        "complete",
        "failure_hold",
    }
)
TARGET_STATES = frozenset(
    {
        "starting",
        "execution_disabled",
        "progressing",
        "resource_wait",
        "retry_wait",
        "running",
        "recovery_observe",
        "recovery_wait",
        *TARGET_TERMINAL_STATES,
    }
)
EXTENSION_STATES = frozenset(
    {
        "starting",
        "waiting_predecessor",
        "waiting_target",
        "retry_wait",
        *TERMINAL_STATES,
    }
)
PREDECESSOR_STATUS_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "updated_at",
        "supervisor",
        "execution_enabled",
        "state",
        "horizon_program",
        "capsules",
        "counts",
        "finished_at",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "supersedes",
        "status_sha256",
    }
)
PREDECESSOR_CAPSULE_FIELDS = frozenset(
    {
        "status",
        "attempts",
        "returncode",
        "started_at",
        "finished_at",
        "artifacts",
        "last_problem",
        "process",
        "adoption_receipts",
        "launch_requested_at",
        "launched_pid",
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
EXTENSION_STATUS_FIELDS = frozenset(
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
        "predecessor",
        "target_program",
        "capsules",
        "counts",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "status_sha256",
    }
)
EXTENSION_STATE_FIELDS = frozenset(
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
        "predecessor",
        "target_program",
        "capsules",
        "problems",
        "state_sha256",
    }
)
EXTENSION_CAPSULE_FIELDS = frozenset(
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
EXTENSION_CAPSULE_STATES = frozenset(
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
SUPERVISOR_FIELDS = frozenset(
    {
        "pid",
        "create_time",
        "implementation_path",
        "implementation_sha256",
    }
)
GENERIC_SUPERVISOR_FIELDS = frozenset(
    {
        "pid",
        "create_time",
        "implementation_path",
        "implementation_sha256",
    }
)
IMPLEMENTATION_PATH = Path(__file__).resolve()


class SuccessorExtensionRefused(RuntimeError):
    pass


IdentityProbe = Callable[[Mapping[str, Any]], str]
ProgramLoader = Callable[..., Program]
StatusReader = Callable[[Program], Mapping[str, Any]]
SupervisorStarter = Callable[..., Mapping[str, Any]]
PredecessorCompletionValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
PredecessorLaunchValidator = Callable[[], None]
NowFn = Callable[[], dt.datetime]


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    pid: int
    create_time: float
    pgid: int
    cwd: str
    label: str
    command: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "create_time": self.create_time,
            "pgid": self.pgid,
            "cwd": self.cwd,
            "label": self.label,
        }


ProcessTable = Callable[[], Sequence[ProcessSnapshot]]


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
        raise SuccessorExtensionRefused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or not 0 < stat.st_size <= MAX_JSON_BYTES:
        raise SuccessorExtensionRefused(f"{label} is not a safe bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise SuccessorExtensionRefused(f"{label} changed while being read: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuccessorExtensionRefused(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorExtensionRefused(f"{label} must be a JSON object: {path}")
    return value


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise SuccessorExtensionRefused(f"{label} self-seal is invalid")


def _require_exact_fields(
    value: object,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuccessorExtensionRefused(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise SuccessorExtensionRefused(
            f"{label} fields drifted; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _validate_problem_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(problem, str) or not problem for problem in value):
        raise SuccessorExtensionRefused(f"{label} problems inventory drifted")
    return value


def _dotted_value(payload: object, dotted: str) -> object:
    value = payload
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


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


def _self_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    implementation = IMPLEMENTATION_PATH.resolve()
    return {
        "pid": process.pid,
        "create_time": process.create_time(),
        "implementation_path": (
            str(implementation.relative_to(repo_root.resolve()))
            if implementation.is_relative_to(repo_root.resolve())
            else str(implementation)
        ),
        "implementation_sha256": sha256_file(implementation),
    }


def _implementation_authority(repo_root: Path) -> dict[str, Any]:
    implementation = IMPLEMENTATION_PATH.resolve()
    resolved_root = repo_root.resolve()
    return {
        "path": (
            str(implementation.relative_to(resolved_root))
            if implementation.is_relative_to(resolved_root)
            else str(implementation)
        ),
        "sha256": sha256_file(implementation),
    }


def _generic_supervisor_authority(repo_root: Path) -> dict[str, str]:
    implementation = Path(generation1_supervisor.__file__).resolve()
    resolved_root = repo_root.resolve()
    return {
        "path": (
            str(implementation.relative_to(resolved_root))
            if implementation.is_relative_to(resolved_root)
            else str(implementation)
        ),
        "sha256": sha256_file(implementation),
    }


def _target_program_authority(
    target_program_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    resolved_root = repo_root.resolve()
    resolved_path = target_program_path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise SuccessorExtensionRefused("successor horizon v2 manifest must remain inside the repository")
    payload = _read_json(resolved_path, "successor horizon v2 manifest")
    _validate_seal(payload, "program_sha256", "successor horizon v2 manifest")
    if (
        payload.get("schema") != "mop-generation1-program/v1"
        or payload.get("program_id") != TARGET_PROGRAM_ID
    ):
        raise SuccessorExtensionRefused("successor horizon v2 manifest identity drifted")
    return {
        "path": str(resolved_path.relative_to(resolved_root)),
        "file_sha256": sha256_file(resolved_path),
        "program_sha256": payload["program_sha256"],
        "program_id": payload["program_id"],
    }


def _predecessor_observation_binding(
    predecessor_status_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "program_id": PREDECESSOR_PROGRAM_ID,
        "status_schema": PREDECESSOR_STATUS_SCHEMA,
        "status_path": str(predecessor_status_path.resolve().relative_to(repo_root.resolve())),
        "signals_allowed": False,
    }


def _program_binding(program: Program) -> dict[str, str]:
    return {
        "path": str(program.path),
        "file_sha256": program.file_sha256,
        "program_sha256": program.program_sha256,
    }


def _snapshot_matches_identity(
    snapshot: ProcessSnapshot,
    identity: Mapping[str, Any],
) -> bool:
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


def _validate_extension_status_authorities(
    status: Mapping[str, Any],
    *,
    repo_root: Path,
    target_program_path: Path,
    predecessor_status_path: Path,
) -> None:
    if status.get("parent_implementation") != _implementation_authority(repo_root):
        raise SuccessorExtensionRefused("sealed successor extension implementation authority drifted")
    if status.get("target_program") != _target_program_authority(
        target_program_path,
        repo_root,
    ):
        raise SuccessorExtensionRefused("sealed successor extension target authority drifted")
    if status.get("predecessor") != _predecessor_observation_binding(
        predecessor_status_path,
        repo_root,
    ):
        raise SuccessorExtensionRefused("sealed successor extension predecessor authority drifted")


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


def _extension_status_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    capsules = state["capsules"]
    complete = sum(row.get("status") == "complete" for row in capsules.values())
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
        "predecessor": state["predecessor"],
        "target_program": state["target_program"],
        "capsules": capsules,
        "counts": {
            "complete": complete,
            "total": len(capsules),
            "remaining": len(capsules) - complete,
        },
        "problems": state["problems"],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": canonical_sha256(core)}


class SuccessorExtensionChain:

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        predecessor_status_path: Path = DEFAULT_PREDECESSOR_STATUS,
        target_program_path: Path = DEFAULT_TARGET_PROGRAM,
        execute: bool = False,
        identity_probe_fn: IdentityProbe = _default_identity_probe,
        process_table_fn: ProcessTable = _default_process_table,
        program_loader_fn: ProgramLoader = load_program,
        target_status_reader_fn: StatusReader = read_generation1_status,
        target_starter_fn: SupervisorStarter = start_generation1_detached,
        predecessor_completion_validator_fn: PredecessorCompletionValidator | None = None,
        predecessor_launch_validator_fn: PredecessorLaunchValidator | None = None,
        now_fn: NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.predecessor_status_path = predecessor_status_path.resolve()
        self.target_program_path = target_program_path.resolve()
        self.execute = execute
        self.identity_probe_fn = identity_probe_fn
        self.process_table_fn = process_table_fn
        self.program_loader_fn = program_loader_fn
        self.target_status_reader_fn = target_status_reader_fn
        self.target_starter_fn = target_starter_fn
        self.predecessor_completion_validator_fn = (
            predecessor_completion_validator_fn or self._validate_complete_predecessor_snapshot
        )
        self.predecessor_launch_validator_fn = (
            predecessor_launch_validator_fn or self._validate_predecessor_launch_authority
        )
        self._predecessor_lock_held = False
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.poll_seconds = poll_seconds
        for label, path in (
            ("extension root", self.root),
            ("predecessor status", self.predecessor_status_path),
            ("target program", self.target_program_path),
        ):
            if not path.is_relative_to(self.repo_root):
                raise SuccessorExtensionRefused(f"{label} must remain inside the repository")
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def status_path(self) -> Path:
        return self.root / STATUS_FILE

    def _identity(self) -> dict[str, Any]:
        return _self_identity(self.repo_root)

    def _implementation_authority(self) -> dict[str, Any]:
        return _implementation_authority(self.repo_root)

    def _target_authority(self) -> dict[str, Any]:
        return _target_program_authority(
            self.target_program_path,
            self.repo_root,
        )

    def _predecessor_binding(self) -> dict[str, Any]:
        return _predecessor_observation_binding(
            self.predecessor_status_path,
            self.repo_root,
        )

    def _validate_bound_authorities(self, state: Mapping[str, Any]) -> None:
        if state.get("parent_implementation") != self._implementation_authority():
            raise SuccessorExtensionRefused("successor extension implementation authority drifted")
        if state.get("target_program") != self._target_authority():
            raise SuccessorExtensionRefused("successor horizon v2 manifest authority drifted")
        if state.get("predecessor") != self._predecessor_binding():
            raise SuccessorExtensionRefused("predecessor observation binding drifted")

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
            "supervisor": self._identity(),
            "parent_implementation": self._implementation_authority(),
            "predecessor": self._predecessor_binding(),
            "target_program": self._target_authority(),
            "capsules": {
                "predecessor_chain_v5": _empty_row(),
                "successor_horizon_v2": _empty_row(),
            },
            "problems": [],
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        state = _read_json(self.state_path, "successor extension state")
        _validate_seal(state, "state_sha256", "successor extension state")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != PROGRAM_ID:
            raise SuccessorExtensionRefused("successor extension state identity drifted")
        capsules = state.get("capsules")
        if not isinstance(capsules, dict) or set(capsules) != {
            "predecessor_chain_v5",
            "successor_horizon_v2",
        }:
            raise SuccessorExtensionRefused("successor extension capsule inventory drifted")
        self._validate_bound_authorities(state)
        state.pop("state_sha256", None)
        state["execution_enabled"] = self.execute
        state["supervisor"] = self._identity()
        return state

    def _append_problem(self, problem: str) -> None:
        problems = list(self.state.get("problems") or [])
        if not problems or problems[-1] != problem:
            problems.append(problem)
        self.state["problems"] = problems[-64:]

    def _artifact(self, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(self.repo_root)),
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "all_ok": True,
            "problems": [],
        }

    def _current_artifact_payload(
        self,
        raw_path: object,
        label: str,
    ) -> tuple[Path, dict[str, Any]]:
        if not isinstance(raw_path, str) or not raw_path:
            raise SuccessorExtensionRefused(f"{label} path must be repository-relative")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SuccessorExtensionRefused(f"{label} path is unsafe: {raw_path}")
        path = self.repo_root / relative
        resolved = path.resolve()
        if not resolved.is_relative_to(self.repo_root) or path.is_symlink() or not resolved.is_file():
            raise SuccessorExtensionRefused(f"{label} path is not a safe current file: {raw_path}")
        return resolved, _read_json(resolved, label)

    def _validate_predecessor_artifact(
        self,
        report: object,
        capsule_id: str,
    ) -> None:
        value = _require_exact_fields(
            report,
            frozenset({"path", "sha256", "schema", "all_ok", "problems"}),
            f"v5 predecessor capsule {capsule_id} artifact report",
        )
        path, payload = self._current_artifact_payload(
            value["path"],
            f"v5 predecessor capsule {capsule_id} artifact",
        )
        if (
            value.get("sha256") != sha256_file(path)
            or not isinstance(value.get("schema"), str)
            or not value["schema"]
            or value.get("schema") != payload.get("schema")
            or value.get("all_ok") is not True
            or value.get("problems") != []
        ):
            raise SuccessorExtensionRefused(f"v5 predecessor capsule {capsule_id} artifact report drifted")

    def _validate_predecessor_status(self, value: Mapping[str, Any]) -> None:
        _require_exact_fields(value, PREDECESSOR_STATUS_FIELDS, "v5 predecessor status")
        state = value.get("state")
        if (
            value.get("schema") != PREDECESSOR_STATUS_SCHEMA
            or value.get("program_id") != PREDECESSOR_PROGRAM_ID
            or value.get("activation_allowed") is not False
            or value.get("scientific_promotion") is not False
            or value.get("supersedes") != "generation1-successor-evidence-chain-v4"
            or not isinstance(value.get("execution_enabled"), bool)
            or not isinstance(state, str)
            or state not in PREDECESSOR_STATES
        ):
            raise SuccessorExtensionRefused("v5 predecessor status identity, safety, or state drifted")
        if (
            not isinstance(value.get("created_at"), str)
            or not isinstance(value.get("updated_at"), str)
            or not isinstance(value.get("supervisor"), Mapping)
            or not isinstance(value.get("horizon_program"), Mapping)
        ):
            raise SuccessorExtensionRefused("v5 predecessor status metadata drifted")
        problems = _validate_problem_list(value.get("problems"), "v5 predecessor status")
        capsules = value.get("capsules")
        if not isinstance(capsules, Mapping) or set(capsules) != set(PREDECESSOR_CAPSULE_IDS):
            raise SuccessorExtensionRefused("v5 predecessor status capsule inventory drifted")
        completed = 0
        for capsule_id in PREDECESSOR_CAPSULE_IDS:
            row = _require_exact_fields(
                capsules[capsule_id],
                PREDECESSOR_CAPSULE_FIELDS,
                f"v5 predecessor capsule {capsule_id}",
            )
            row_state = row.get("status")
            if not isinstance(row_state, str) or row_state not in PREDECESSOR_CAPSULE_STATES:
                raise SuccessorExtensionRefused(f"v5 predecessor capsule {capsule_id} state drifted")
            attempts = row.get("attempts")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
                raise SuccessorExtensionRefused(f"v5 predecessor capsule {capsule_id} attempts drifted")
            if row_state == "complete":
                completed += 1
        expected_counts = {
            "complete": completed,
            "total": len(PREDECESSOR_CAPSULE_IDS),
            "remaining": len(PREDECESSOR_CAPSULE_IDS) - completed,
        }
        if value.get("counts") != expected_counts:
            raise SuccessorExtensionRefused("v5 predecessor status counts drifted")
        if state != "complete":
            return
        if (
            problems != []
            or value.get("execution_enabled") is not True
            or not isinstance(value.get("finished_at"), str)
        ):
            raise SuccessorExtensionRefused("completed v5 predecessor status is not clean")
        for capsule_id in PREDECESSOR_CAPSULE_IDS:
            row = capsules[capsule_id]
            artifacts = row.get("artifacts")
            if (
                row.get("status") != "complete"
                or row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or not isinstance(artifacts, list)
                or not artifacts
            ):
                raise SuccessorExtensionRefused(
                    f"completed v5 predecessor capsule {capsule_id} receipt drifted"
                )
            for report in artifacts:
                self._validate_predecessor_artifact(report, capsule_id)

    def _validate_complete_predecessor_snapshot(
        self,
        _status: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        horizon_program_path = self.repo_root / "configs/campaign/generation1_successor_horizon_v1.json"
        try:
            return predecessor_v5.read_validated_complete_chain_status(
                root=self.predecessor_status_path.parent,
                repo_root=self.repo_root,
                horizon_program_path=horizon_program_path,
                acquire_lock=not self._predecessor_lock_held,
            )
        except (
            predecessor_v5.SuccessorChainRefused,
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
            raise SuccessorExtensionRefused(
                f"v5 predecessor complete snapshot is invalid: {type(exc).__name__}: {exc}"
            ) from exc

    def _validate_predecessor_launch_authority(self) -> None:
        try:
            before = dict(self._validate_complete_predecessor_snapshot({}))
            admission = horizon_v2.build_admission(
                parent_result_path=self.repo_root / "proof/GENERATION1_SUCCESSOR_HORIZON.json",
                parent_verification_path=(
                    self.repo_root / "proof/GENERATION1_SUCCESSOR_HORIZON.verification.json"
                ),
                parent_report_receipt_path=(
                    self.repo_root / "runs/generation1/generation1-successor-horizon-v1/report_receipt.json"
                ),
            )
            horizon_v2.validate_admission(admission)
            after = dict(self._validate_complete_predecessor_snapshot(before))
        except (
            SuccessorExtensionRefused,
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
            raise SuccessorExtensionRefused(
                f"v2 prelaunch predecessor authority replay failed: {type(exc).__name__}: {exc}"
            ) from exc
        if after != before:
            raise SuccessorExtensionRefused("v5 predecessor changed during v2 prelaunch authority replay")

    def _publish(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        self.state["updated_at"] = now
        core = dict(self.state)
        core.pop("state_sha256", None)
        sealed_state = {**core, "state_sha256": canonical_sha256(core)}
        atomic_write_json(self.state_path, sealed_state)
        self.state = dict(sealed_state)
        status = _extension_status_payload(self.state)
        atomic_write_json(self.status_path, status)
        return status

    def _predecessor_status(self) -> dict[str, Any] | None:
        if not self.predecessor_status_path.exists():
            return None
        value = _read_json(self.predecessor_status_path, "v5 predecessor status")
        _validate_seal(value, "status_sha256", "v5 predecessor status")
        self._validate_predecessor_status(value)
        if value.get("state") == "complete":
            validated = self.predecessor_completion_validator_fn(value)
            if not isinstance(validated, Mapping):
                raise SuccessorExtensionRefused(
                    "v5 predecessor completion validator returned no authoritative status"
                )
            value = dict(validated)
            _validate_seal(value, "status_sha256", "v5 predecessor status")
            self._validate_predecessor_status(value)
            if value.get("state") != "complete":
                raise SuccessorExtensionRefused(
                    "v5 predecessor changed state during complete snapshot validation"
                )
        return value

    def _reconcile_predecessor(self, row: dict[str, Any]) -> bool:
        status = self._predecessor_status()
        if row.get("status") == "complete":
            if status is None or status.get("state") != "complete" or status.get("problems") != []:
                raise SuccessorExtensionRefused("completed predecessor evidence disappeared or regressed")
            expected_artifacts = [self._artifact(self.predecessor_status_path, status)]
            if (
                row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("artifacts") != expected_artifacts
            ):
                raise SuccessorExtensionRefused("completed predecessor evidence binding drifted")
            return True
        if status is None:
            row["status"] = "pending"
            row["last_problem"] = "waiting for the sealed v5 chain status"
            return False
        state = str(status.get("state"))
        row["last_problem"] = f"observed v5 chain state {state}; signals remain forbidden"
        if state == "complete":
            if status.get("problems") != []:
                raise SuccessorExtensionRefused("completed predecessor status is not clean")
            row["status"] = "complete"
            row["returncode"] = 0
            row["finished_at"] = row.get("finished_at") or _iso(self.now_fn())
            row["artifacts"] = [self._artifact(self.predecessor_status_path, status)]
            row["last_problem"] = None
            return True
        if state in PREDECESSOR_TERMINAL_STATES:
            row["status"] = "failure_hold"
            row["finished_at"] = _iso(self.now_fn())
            return False
        row["status"] = "observed"
        if row.get("started_at") is None:
            row["started_at"] = _iso(self.now_fn())
        return False

    def _load_target_program(self) -> Program:
        try:
            program = self.program_loader_fn(
                self.target_program_path,
                repo_root=self.repo_root,
            )
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise SuccessorExtensionRefused(
                f"successor horizon v2 program is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        bound = self.state.get("target_program")
        if (
            program.program_id != TARGET_PROGRAM_ID
            or not isinstance(bound, Mapping)
            or program.program_sha256 != bound.get("program_sha256")
        ):
            raise SuccessorExtensionRefused("loaded successor horizon v2 authority drifted")
        return program

    def _target_status(self, program: Program) -> Mapping[str, Any] | None:
        if not program.status_path.exists():
            return None
        try:
            status = self.target_status_reader_fn(program)
        except (Generation1Refused, OSError, ValueError) as exc:
            raise SuccessorExtensionRefused(
                f"successor horizon v2 status is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        _validate_seal(status, "status_sha256", "successor horizon v2 status")
        _require_exact_fields(status, GENERIC_STATUS_FIELDS, "successor horizon v2 status")
        state = status.get("state")
        if (
            status.get("schema") != GENERIC_STATUS_SCHEMA
            or status.get("program_id") != TARGET_PROGRAM_ID
            or status.get("program") != _program_binding(program)
            or not isinstance(state, str)
            or state not in TARGET_STATES
        ):
            raise SuccessorExtensionRefused(
                "successor horizon v2 status identity, program binding, or state drifted"
            )
        execution_enabled = status.get("execution_enabled")
        if execution_enabled is not True and not (
            state == "execution_disabled" and execution_enabled is False
        ):
            raise SuccessorExtensionRefused("successor horizon v2 status execution authority drifted")
        supervisor = status.get("supervisor")
        implementation = _generic_supervisor_authority(self.repo_root)
        if (
            not isinstance(status.get("created_at"), str)
            or not isinstance(supervisor, Mapping)
            or set(supervisor) != GENERIC_SUPERVISOR_FIELDS
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
        ):
            raise SuccessorExtensionRefused("successor horizon v2 status metadata drifted")
        _validate_problem_list(status.get("problems"), "successor horizon v2 status")
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
            raise SuccessorExtensionRefused(
                "successor horizon v2 status queue or injection authority drifted"
            )
        if status.get("last_admission") is not None and not isinstance(
            status.get("last_admission"),
            Mapping,
        ):
            raise SuccessorExtensionRefused("successor horizon v2 status admission evidence drifted")
        capsules = status.get("capsules")
        expected_capsules = {capsule.capsule_id: capsule for capsule in program.capsules}
        if not isinstance(capsules, Mapping) or set(capsules) != set(expected_capsules):
            raise SuccessorExtensionRefused("successor horizon v2 status capsule inventory drifted")
        current_capsule = status.get("current_capsule")
        if current_capsule is not None and current_capsule not in expected_capsules:
            raise SuccessorExtensionRefused("successor horizon v2 active capsule drifted")
        for capsule_id, capsule in expected_capsules.items():
            row = _require_exact_fields(
                capsules[capsule_id],
                GENERIC_CAPSULE_FIELDS,
                f"successor horizon v2 capsule {capsule_id}",
            )
            identity = {
                "id": capsule.capsule_id,
                "kind": capsule.kind,
                "priority": capsule.priority,
                "depends_on": list(capsule.depends_on),
                "capsule_sha256": capsule.capsule_sha256,
                "source": "base",
            }
            if any(row.get(key) != expected for key, expected in identity.items()):
                raise SuccessorExtensionRefused(f"successor horizon v2 capsule {capsule_id} identity drifted")
            row_state = row.get("status")
            if row_state not in {"pending", "running", "complete", "failed"}:
                raise SuccessorExtensionRefused(f"successor horizon v2 capsule {capsule_id} state drifted")
            attempts = row.get("attempts")
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
                or not isinstance(row.get("artifacts"), list)
                or not isinstance(row.get("runtime"), Mapping)
            ):
                raise SuccessorExtensionRefused(
                    f"successor horizon v2 capsule {capsule_id} receipt shape drifted"
                )
            if row_state == "complete":
                self._validate_completed_target_capsule(capsule, row)
        if state == "complete" and (
            status.get("problems") != []
            or status.get("current_capsule") is not None
            or status.get("lane_reservation") is not None
            or any(row.get("status") != "complete" for row in capsules.values())
        ):
            raise SuccessorExtensionRefused("completed successor horizon v2 status is not clean")
        return status

    def _validate_completed_target_capsule(
        self,
        capsule: Any,
        row: Mapping[str, Any],
    ) -> None:
        attempts = row.get("attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts <= 0
            or row.get("returncode") != 0
            or not isinstance(row.get("finished_at"), str)
            or row.get("last_problem") is not None
        ):
            raise SuccessorExtensionRefused(
                f"completed successor horizon v2 capsule {capsule.capsule_id} receipt drifted"
            )
        reports = row.get("artifacts")
        if not isinstance(reports, list) or len(reports) != len(capsule.artifacts):
            raise SuccessorExtensionRefused(
                f"completed successor horizon v2 capsule {capsule.capsule_id} artifact inventory drifted"
            )
        for expectation, report in zip(capsule.artifacts, reports, strict=True):
            path, payload = self._current_artifact_payload(
                expectation.path,
                f"successor horizon v2 capsule {capsule.capsule_id} artifact",
            )
            if payload.get("schema") != expectation.schema:
                raise SuccessorExtensionRefused(
                    f"successor horizon v2 capsule {capsule.capsule_id} artifact schema drifted"
                )
            for dotted, expected in expectation.fields:
                if _dotted_value(payload, dotted) != expected:
                    raise SuccessorExtensionRefused(
                        f"successor horizon v2 capsule {capsule.capsule_id} artifact field {dotted} drifted"
                    )
            if expectation.seal_field is not None:
                _validate_seal(
                    payload,
                    expectation.seal_field,
                    f"successor horizon v2 capsule {capsule.capsule_id} artifact",
                )
            expected_report = {
                "path": expectation.path,
                "sha256": sha256_file(path),
                "schema": payload.get("schema"),
                "problems": [],
                "all_ok": True,
            }
            if report != expected_report:
                raise SuccessorExtensionRefused(
                    f"successor horizon v2 capsule {capsule.capsule_id} artifact report drifted"
                )

    def _visible_target_supervisors(self) -> tuple[ProcessSnapshot, ...]:
        expected_label = f"mop-supervisor:{TARGET_PROGRAM_ID}"
        rows = tuple(
            row for row in self.process_table_fn() if row.pid != os.getpid() and row.label == expected_label
        )
        if any(
            row.cwd != str(self.repo_root)
            or row.pgid <= 0
            or not math.isfinite(row.create_time)
            or row.create_time <= 0
            for row in rows
        ):
            raise SuccessorExtensionRefused("visible successor horizon v2 supervisor has inexact metadata")
        if len(rows) > 1:
            raise SuccessorExtensionRefused("multiple successor horizon v2 supervisors are visible")
        return rows

    def _reconcile_target(
        self,
        row: dict[str, Any],
        program: Program,
    ) -> dict[str, Any]:
        status = self._target_status(program)
        visible = self._visible_target_supervisors()
        if row.get("status") == "complete":
            if status is None or status.get("state") != "complete" or status.get("problems") != []:
                raise SuccessorExtensionRefused(
                    "completed successor horizon v2 evidence disappeared or regressed"
                )
            expected_artifacts = [self._artifact(program.status_path, status)]
            if (
                row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("artifacts") != expected_artifacts
            ):
                raise SuccessorExtensionRefused("completed successor horizon v2 evidence binding drifted")
            return row
        if status is not None:
            state = str(status.get("state"))
            if state == "complete":
                if status.get("problems") != []:
                    raise SuccessorExtensionRefused("completed successor horizon v2 status is not clean")
                if visible:
                    supervisor = status.get("supervisor")
                    if not isinstance(supervisor, Mapping) or not _snapshot_matches_identity(
                        visible[0], supervisor
                    ):
                        raise SuccessorExtensionRefused(
                            "completed successor horizon v2 status conflicts "
                            "with the visible exact supervisor"
                        )
                row["status"] = "complete"
                row["returncode"] = 0
                row["finished_at"] = row.get("finished_at") or _iso(self.now_fn())
                row["artifacts"] = [self._artifact(program.status_path, status)]
                row["last_problem"] = None
                return row
            if state in TARGET_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"successor horizon v2 entered {state}"
                return row
            supervisor = status.get("supervisor")
            if isinstance(supervisor, Mapping):
                identity_state = self.identity_probe_fn(supervisor)
                if identity_state == "unknown":
                    raise SuccessorExtensionRefused("successor horizon v2 supervisor identity is inexact")
                if identity_state == "alive":
                    if len(visible) != 1 or not _snapshot_matches_identity(visible[0], supervisor):
                        raise SuccessorExtensionRefused(
                            "sealed successor horizon v2 supervisor conflicts "
                            "with the visible exact supervisor"
                        )
                    row["status"] = "running"
                    row["process"] = dict(supervisor)
                    row["last_problem"] = f"generic supervisor is running in state {state}"
                    return row

        if visible:
            row["status"] = "adoption_wait"
            row["process"] = visible[0].identity()
            row["last_problem"] = (
                "exact v2 supervisor is visible while sealed status acknowledgement is pending"
            )
            return row
        if not self.execute:
            row["status"] = "pending"
            row["last_problem"] = "successor horizon v2 start requires explicit --execute"
            return row

        predecessor_lock = self.predecessor_status_path.parent / predecessor_v5.LOCK_FILE
        with FileLock(predecessor_lock):
            self._predecessor_lock_held = True
            try:
                self.predecessor_launch_validator_fn()
                row["status"] = "launching"
                row["attempts"] = int(row.get("attempts") or 0) + 1
                row["launch_requested_at"] = _iso(self.now_fn())
                row["last_problem"] = "requesting idempotent generic supervisor start"
                self.state["status"] = "waiting_target"
                self.state["finished_at"] = None
                self._publish()
                row = self.state["capsules"]["successor_horizon_v2"]
                self.predecessor_launch_validator_fn()
                try:
                    launched = self.target_starter_fn(
                        program,
                        execute=True,
                        use_caffeinate=True,
                    )
                except (Generation1Refused, OSError, RuntimeError, ValueError) as exc:
                    row["status"] = "retry_wait"
                    row["last_problem"] = f"v2 startup acknowledgement pending: {type(exc).__name__}: {exc}"
                    return row
                launched_pid = launched.get("launched_pid")
                row["launched_pid"] = launched_pid if isinstance(launched_pid, int) else None
                returned_status = launched.get("status")
                row["status"] = "running"
                row["started_at"] = row.get("started_at") or _iso(self.now_fn())
                row["last_problem"] = "generic Generation 1 supervisor start accepted"
                if isinstance(returned_status, Mapping):
                    supervisor = returned_status.get("supervisor")
                    if isinstance(supervisor, Mapping):
                        row["process"] = dict(supervisor)
                return row
            finally:
                self._predecessor_lock_held = False

    def _validate_terminal_artifacts(self, program: Program) -> None:
        predecessor = self.state["capsules"]["predecessor_chain_v5"]
        if predecessor.get("status") == "complete":
            self._reconcile_predecessor(predecessor)
        target = self.state["capsules"]["successor_horizon_v2"]
        if target.get("status") == "complete":
            self._reconcile_target(target, program)

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_bound_authorities(self.state)
            target_program = self._load_target_program()
            if self.state.get("status") in TERMINAL_STATES and self.state.get("status") != "complete":
                self._validate_terminal_artifacts(target_program)
                return self._publish()
            predecessor = self.state["capsules"]["predecessor_chain_v5"]
            predecessor_complete = self._reconcile_predecessor(predecessor)
            if predecessor.get("status") == "failure_hold":
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem(str(predecessor.get("last_problem") or "v5 predecessor failed"))
                return self._publish()
            if not predecessor_complete:
                self.state["status"] = "waiting_predecessor"
                return self._publish()

            target = self.state["capsules"]["successor_horizon_v2"]
            target = self._reconcile_target(target, target_program)
            if target.get("status") == "complete":
                self.state["status"] = "complete"
                self.state["finished_at"] = _iso(self.now_fn())
            elif target.get("status") == "failure_hold":
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem(str(target.get("last_problem")))
            elif target.get("status") == "retry_wait":
                self.state["status"] = "retry_wait"
            else:
                self.state["status"] = "waiting_target"
            return self._publish()
        except (
            SuccessorExtensionRefused,
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
            self.state["supervisor"] = self._identity()
            while True:
                status = self.tick()
                cycles += 1
                if status["state"] in TERMINAL_STATES:
                    return status
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                self.sleep_fn(self.poll_seconds)


def validate_extension_status(
    status: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    target_program_path: Path = DEFAULT_TARGET_PROGRAM,
    predecessor_status_path: Path = DEFAULT_PREDECESSOR_STATUS,
) -> str:

    repo_root = repo_root.resolve()
    target_program_path = target_program_path.resolve()
    predecessor_status_path = predecessor_status_path.resolve()
    if set(status) != EXTENSION_STATUS_FIELDS:
        raise SuccessorExtensionRefused("successor extension status fields drifted")
    _validate_seal(status, "status_sha256", "successor extension status")
    state = status.get("state")
    problems = status.get("problems")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("program_id") != PROGRAM_ID
        or status.get("activation_allowed") is not False
        or status.get("scientific_promotion") is not False
        or not isinstance(status.get("execution_enabled"), bool)
        or not isinstance(state, str)
        or state not in EXTENSION_STATES
        or not isinstance(problems, list)
        or any(not isinstance(problem, str) or not problem for problem in problems)
    ):
        raise SuccessorExtensionRefused("successor extension status identity or state drifted")
    if (state == "complete" or state not in TERMINAL_STATES) and (
        status.get("execution_enabled") is not True or problems != []
    ):
        raise SuccessorExtensionRefused("successor extension safe acknowledgement is not clean")
    _validate_extension_status_authorities(
        status,
        repo_root=repo_root,
        target_program_path=target_program_path,
        predecessor_status_path=predecessor_status_path,
    )
    created_at = _parse_time(status.get("created_at"))
    updated_at = _parse_time(status.get("updated_at"))
    finished_at = _parse_time(status.get("finished_at"))
    if created_at is None or updated_at is None or created_at > updated_at:
        raise SuccessorExtensionRefused("successor extension status timestamps drifted")
    if state in TERMINAL_STATES:
        if finished_at is None or finished_at > updated_at:
            raise SuccessorExtensionRefused("successor extension terminal timestamp drifted")
    elif status.get("finished_at") is not None:
        raise SuccessorExtensionRefused("successor extension nonterminal status is marked finished")
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
        raise SuccessorExtensionRefused("successor extension supervisor authority drifted")
    capsules = status.get("capsules")
    expected_ids = {"predecessor_chain_v5", "successor_horizon_v2"}
    if not isinstance(capsules, Mapping) or set(capsules) != expected_ids:
        raise SuccessorExtensionRefused("successor extension status capsule inventory drifted")
    complete = 0
    for capsule_id, raw_row in capsules.items():
        if not isinstance(raw_row, Mapping) or set(raw_row) != EXTENSION_CAPSULE_FIELDS:
            raise SuccessorExtensionRefused(f"successor extension capsule {capsule_id} fields drifted")
        row_state = raw_row.get("status")
        attempts = raw_row.get("attempts")
        returncode = raw_row.get("returncode")
        started_at = _parse_time(raw_row.get("started_at"))
        finished_at = _parse_time(raw_row.get("finished_at"))
        launch_requested_at = _parse_time(raw_row.get("launch_requested_at"))
        artifacts = raw_row.get("artifacts")
        last_problem = raw_row.get("last_problem")
        process = raw_row.get("process")
        launched_pid = raw_row.get("launched_pid")
        if (
            not isinstance(row_state, str)
            or row_state not in EXTENSION_CAPSULE_STATES
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or (returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)))
            or (raw_row.get("started_at") is not None and started_at is None)
            or (raw_row.get("finished_at") is not None and finished_at is None)
            or (raw_row.get("launch_requested_at") is not None and launch_requested_at is None)
            or (started_at is not None and finished_at is not None and started_at > finished_at)
            or not isinstance(artifacts, list)
            or any(not isinstance(report, Mapping) for report in artifacts)
            or (last_problem is not None and (not isinstance(last_problem, str) or not last_problem))
            or (process is not None and not isinstance(process, Mapping))
            or (
                launched_pid is not None
                and (isinstance(launched_pid, bool) or not isinstance(launched_pid, int) or launched_pid <= 0)
            )
        ):
            raise SuccessorExtensionRefused(f"successor extension capsule {capsule_id} receipt shape drifted")
        if isinstance(process, Mapping):
            process_pid = process.get("pid")
            process_create_time = process.get("create_time")
            if (
                isinstance(process_pid, bool)
                or not isinstance(process_pid, int)
                or process_pid <= 0
                or isinstance(process_create_time, bool)
                or not isinstance(process_create_time, int | float)
                or not math.isfinite(float(process_create_time))
                or float(process_create_time) <= 0
            ):
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} process identity drifted"
                )
        if row_state == "complete":
            complete += 1
            if returncode != 0 or finished_at is None or not artifacts or last_problem is not None:
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} completion drifted"
                )
        elif (
            returncode is not None
            or artifacts != []
            or (row_state == "failure_hold" and (finished_at is None or not isinstance(last_problem, str)))
            or (row_state != "failure_hold" and finished_at is not None)
        ):
            raise SuccessorExtensionRefused(f"successor extension capsule {capsule_id} noncompletion drifted")
    expected_counts = {
        "complete": complete,
        "total": len(expected_ids),
        "remaining": len(expected_ids) - complete,
    }
    if status.get("counts") != expected_counts:
        raise SuccessorExtensionRefused("successor extension status counts drifted")
    predecessor_state = capsules["predecessor_chain_v5"].get("status")
    target_state = capsules["successor_horizon_v2"].get("status")
    all_complete = complete == len(expected_ids)
    pristine_target = capsules["successor_horizon_v2"] == _empty_row()
    coherent = True
    if state == "starting":
        coherent = all(row == _empty_row() for row in capsules.values())
    elif state == "waiting_predecessor":
        coherent = predecessor_state in {"pending", "observed"} and pristine_target
    elif state == "waiting_target":
        coherent = predecessor_state == "complete" and target_state in {
            "pending",
            "running",
            "adoption_wait",
            "launching",
        }
    elif state == "retry_wait":
        coherent = predecessor_state == "complete" and target_state == "retry_wait"
    elif state == "complete":
        coherent = all_complete
    elif state == "failure_hold":
        coherent = "failure_hold" in {predecessor_state, target_state}
    if not coherent or (state not in TERMINAL_STATES and all_complete):
        raise SuccessorExtensionRefused("successor extension state and capsule progression drifted")
    if state not in TERMINAL_STATES and any(row.get("status") == "failure_hold" for row in capsules.values()):
        raise SuccessorExtensionRefused("successor extension safe status contains a failed capsule")
    return state


def _read_stable_extension_state_status(
    state_path: Path,
    status_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for attempt in range(SNAPSHOT_ATTEMPTS):
        before = _read_json(state_path, "successor extension state")
        first_status = _read_json(status_path, "successor extension status")
        after = _read_json(state_path, "successor extension state")
        second_status = _read_json(status_path, "successor extension status")
        try:
            projected = _extension_status_payload(before)
        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ):
            projected = None
        if before == after and first_status == second_status and first_status == projected:
            return before, first_status
        if attempt + 1 < SNAPSHOT_ATTEMPTS:
            time.sleep(SNAPSHOT_INTERVAL_SECONDS)
    raise SuccessorExtensionRefused("successor extension state/status snapshot did not become coherent")


def _validate_extension_bound_artifact_hashes(
    repo_root: Path,
    capsules: Mapping[str, Any],
) -> None:
    for capsule_id, row in capsules.items():
        if not isinstance(row, Mapping):
            raise SuccessorExtensionRefused(f"successor extension capsule {capsule_id} is not an object")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list):
            raise SuccessorExtensionRefused(f"successor extension capsule {capsule_id} artifacts drifted")
        for report in artifacts:
            if not isinstance(report, Mapping):
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} artifact report drifted"
                )
            raw_path = report.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} artifact path drifted"
                )
            relative = Path(raw_path)
            candidate = repo_root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} artifact path is unsafe"
                )
            path = candidate.resolve()
            if not path.is_relative_to(repo_root) or report.get("sha256") != sha256_file(path):
                raise SuccessorExtensionRefused(
                    f"successor extension capsule {capsule_id} final artifact hash drifted"
                )


def read_validated_complete_extension_status(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path = REPO_ROOT,
    target_program_path: Path | None = None,
    predecessor_status_path: Path | None = None,
    acquire_lock: bool = True,
) -> dict[str, Any]:

    root = root.resolve()
    repo_root = repo_root.resolve()
    if not root.is_dir() or not root.is_relative_to(repo_root):
        raise SuccessorExtensionRefused("successor extension root is absent or outside the repository")
    if acquire_lock:
        for attempt in range(LOCK_ATTEMPTS):
            try:
                with FileLock(root / LOCK_FILE):
                    return read_validated_complete_extension_status(
                        root=root,
                        repo_root=repo_root,
                        target_program_path=target_program_path,
                        predecessor_status_path=predecessor_status_path,
                        acquire_lock=False,
                    )
            except Generation1Refused as exc:
                if not isinstance(exc.__cause__, BlockingIOError):
                    raise
                if attempt + 1 < LOCK_ATTEMPTS:
                    time.sleep(LOCK_INTERVAL_SECONDS)
                    continue
                raise SuccessorExtensionRefused(
                    "successor extension lifetime lock remained held while validating a complete snapshot"
                ) from exc
        raise SuccessorExtensionRefused("successor extension lifetime lock acquisition exhausted")
    if target_program_path is None:
        relative = DEFAULT_TARGET_PROGRAM.resolve().relative_to(REPO_ROOT.resolve())
        target_program_path = repo_root / relative
    if predecessor_status_path is None:
        relative = DEFAULT_PREDECESSOR_STATUS.resolve().relative_to(REPO_ROOT.resolve())
        predecessor_status_path = repo_root / relative
    target_program_path = target_program_path.resolve()
    predecessor_status_path = predecessor_status_path.resolve()
    state_path = root / STATE_FILE
    status_path = root / STATUS_FILE
    stable_state, stable_status = _read_stable_extension_state_status(
        state_path,
        status_path,
    )
    if set(stable_state) != EXTENSION_STATE_FIELDS:
        raise SuccessorExtensionRefused("successor extension complete state fields drifted")
    _validate_seal(
        stable_state,
        "state_sha256",
        "successor extension state",
    )
    if (
        stable_state.get("schema") != STATE_SCHEMA
        or stable_state.get("program_id") != PROGRAM_ID
        or stable_state.get("status") != "complete"
        or stable_state.get("execution_enabled") is not True
        or stable_state.get("problems") != []
    ):
        raise SuccessorExtensionRefused("successor extension complete state boundary drifted")
    validated_state = validate_extension_status(
        stable_status,
        repo_root=repo_root,
        target_program_path=target_program_path,
        predecessor_status_path=predecessor_status_path,
    )
    if validated_state != "complete":
        raise SuccessorExtensionRefused("successor extension durable status is not complete")
    owner = SuccessorExtensionChain(
        root=root,
        repo_root=repo_root,
        predecessor_status_path=predecessor_status_path,
        target_program_path=target_program_path,
        execute=True,
    )
    state_core = copy.deepcopy(dict(stable_state))
    state_core.pop("state_sha256", None)
    owner.state = state_core
    target_program = owner._load_target_program()
    before_capsules = copy.deepcopy(owner.state["capsules"])
    owner._validate_terminal_artifacts(target_program)
    if owner.state["capsules"] != before_capsules:
        raise SuccessorExtensionRefused(
            "successor extension evidence required reconciliation during terminal replay"
        )
    current_state, current_status = _read_stable_extension_state_status(
        state_path,
        status_path,
    )
    if current_state != stable_state or current_status != stable_status:
        raise SuccessorExtensionRefused("successor extension state/status changed during terminal replay")
    owner._validate_terminal_artifacts(target_program)
    if owner.state["capsules"] != before_capsules:
        raise SuccessorExtensionRefused(
            "successor extension evidence required reconciliation during final replay"
        )
    final_state, final_status = _read_stable_extension_state_status(
        state_path,
        status_path,
    )
    if final_state != stable_state or final_status != stable_status:
        raise SuccessorExtensionRefused(
            "successor extension state/status changed during final evidence replay"
        )
    _validate_extension_bound_artifact_hashes(
        repo_root,
        stable_state["capsules"],
    )
    return stable_status


def read_extension_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    status = _read_json(root.resolve() / STATUS_FILE, "successor extension status")
    _validate_seal(status, "status_sha256", "successor extension status")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("program_id") != PROGRAM_ID
        or status.get("activation_allowed") is not False
        or status.get("scientific_promotion") is not False
    ):
        raise SuccessorExtensionRefused("successor extension status identity drifted")
    return status


def _process_identity_alive(identity: object) -> bool:
    return isinstance(identity, Mapping) and _default_identity_probe(identity) == "alive"


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


def _visible_parent_process(entrypoint: Path) -> ProcessSnapshot | None:
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
        or not math.isfinite(process.create_time)
        or process.create_time <= 0
        for process in candidates
    ):
        raise SuccessorExtensionRefused("visible successor extension parent has inexact metadata")
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise SuccessorExtensionRefused("multiple successor extension parent processes are visible")
    if labelled:
        return labelled[0]
    return next(
        (process for process in candidates if process.pid == process.pgid),
        candidates[0],
    )


def _revalidate_terminal_extension(root: Path) -> dict[str, Any]:
    with FileLock(root / LOCK_FILE):
        owner = SuccessorExtensionChain(
            root=root,
            repo_root=REPO_ROOT,
            predecessor_status_path=DEFAULT_PREDECESSOR_STATUS,
            target_program_path=DEFAULT_TARGET_PROGRAM,
            execute=True,
        )
        if owner.state.get("status") not in TERMINAL_STATES:
            raise SuccessorExtensionRefused(
                "sealed terminal status disagrees with the durable successor extension state"
            )
        owner.state["supervisor"] = owner._identity()
        return owner.tick()


def start_extension_detached(
    *,
    root: Path = DEFAULT_ROOT,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise SuccessorExtensionRefused("detached successor extension start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_generation1_successor_extension_chain_v2.py").resolve()
        status_path = root / STATUS_FILE
        status = None
        if status_path.is_file():
            status = read_extension_status(root)
            _validate_extension_status_authorities(
                status,
                repo_root=REPO_ROOT,
                target_program_path=DEFAULT_TARGET_PROGRAM,
                predecessor_status_path=DEFAULT_PREDECESSOR_STATUS,
            )
        visible = _visible_parent_process(entrypoint)
        status_identity = status.get("supervisor") if status is not None else None
        if _process_identity_alive(status_identity):
            if visible is None:
                raise SuccessorExtensionRefused(
                    "sealed successor extension identity has no exact visible parent"
                )
            if (
                not isinstance(status_identity, Mapping)
                or status_identity.get("pid") != visible.pid
                or not math.isclose(
                    float(status_identity.get("create_time", -1)),
                    visible.create_time,
                    rel_tol=0.0,
                    abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
                )
            ):
                raise SuccessorExtensionRefused("sealed extension identity conflicts with the visible parent")
            if status is None:
                raise SuccessorExtensionRefused("live successor extension identity lacks a sealed status")
            try:
                validate_extension_status(
                    status,
                    repo_root=REPO_ROOT,
                    target_program_path=DEFAULT_TARGET_PROGRAM,
                    predecessor_status_path=DEFAULT_PREDECESSOR_STATUS,
                )
            except (SuccessorExtensionRefused, OSError, ValueError):
                return {
                    "already_running": True,
                    "status": status,
                    "observed_process": visible.identity(),
                }
            return {"already_running": True, "status": status}
        if visible is not None:
            return {
                "already_running": True,
                "status": status,
                "observed_process": visible.identity(),
            }
        if status is not None and status.get("state") in TERMINAL_STATES:
            return {
                "already_terminal": True,
                "status": _revalidate_terminal_extension(root),
            }

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
                    acknowledged = read_extension_status(root)
                    validate_extension_status(
                        acknowledged,
                        repo_root=REPO_ROOT,
                        target_program_path=DEFAULT_TARGET_PROGRAM,
                        predecessor_status_path=DEFAULT_PREDECESSOR_STATUS,
                    )
                    break
                except (SuccessorExtensionRefused, OSError, ValueError):
                    acknowledged = None
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise SuccessorExtensionRefused(
                f"detached extension did not acknowledge startup; inspect {stderr_path}"
            )
        _validate_extension_status_authorities(
            acknowledged,
            repo_root=REPO_ROOT,
            target_program_path=DEFAULT_TARGET_PROGRAM,
            predecessor_status_path=DEFAULT_PREDECESSOR_STATUS,
        )
        return {
            "launched_pid": process.pid,
            "caffeinate": bool(caffeinate),
            "command": launched,
            "status": acknowledged,
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
                raise SuccessorExtensionRefused("successor extension execution requires explicit --execute")
            set_process_label(PARENT_LABEL)
            payload = SuccessorExtensionChain(
                root=arguments.root,
                execute=True,
            ).run(max_cycles=1 if arguments.once else None)
        elif arguments.command == "start":
            payload = start_extension_detached(
                root=arguments.root,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        else:
            payload = read_extension_status(arguments.root)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if arguments.command == "run" and payload.get("state") not in {
            "complete",
            "drained",
        }:
            return 2
        return 0
    except (
        SuccessorExtensionRefused,
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
    "PARENT_LABEL",
    "PREDECESSOR_PROGRAM_ID",
    "PREDECESSOR_STATUS_SCHEMA",
    "PROGRAM_ID",
    "ProcessSnapshot",
    "STATE_SCHEMA",
    "STATUS_SCHEMA",
    "SuccessorExtensionChain",
    "SuccessorExtensionRefused",
    "TARGET_PROGRAM_ID",
    "build_parser",
    "main",
    "read_extension_status",
    "read_validated_complete_extension_status",
    "start_extension_detached",
    "validate_extension_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
