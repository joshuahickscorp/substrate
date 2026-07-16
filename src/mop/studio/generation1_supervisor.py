"""Deterministic detached supervision for the Generation 1 result corpus.

The supervisor is intentionally a thin control plane.  A sealed program manifest
declares exact command capsules and their resource envelopes.  One capsule runs at
a time, while a capsule may use its own deterministic worker pool.  Exploratory
injections are append-only, self-hashed files and are consumed only between
capsules, so they can never mutate a running command or a claim-bearing base plan.

The operator does not need to poll.  Resource refusals are retried by the detached
supervisor, and status is an atomic sealed snapshot that can be read on demand.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ..config import REPO_ROOT
from ..process_labels import set_process_label
from . import local_throttle as local_throttle_runtime
from .local_throttle import (
    TaskDeclaration,
    ThrottlePolicy,
    ThrottleRefused,
    active_lanes,
    aggregate_admission,
    collect_host_telemetry,
    evaluate_task,
    is_taskpolicy_coexistence_command,
    load_policy,
)

PROGRAM_SCHEMA = "mop-generation1-program/v1"
CAPSULE_SCHEMA = "mop-generation1-capsule/v1"
INJECTION_SCHEMA = "mop-generation1-injection/v1"
STATE_SCHEMA = "mop-generation1-state/v1"
STATUS_SCHEMA = "mop-generation1-status/v1"
CONTROL_SCHEMA = "mop-generation1-control/v1"
INJECTION_RECEIPT_SCHEMA = "mop-generation1-injection-receipt/v1"

STATE_FILE = "program_state.json"
STATUS_FILE = "current_status.json"
CONTROL_FILE = "control/stop-request.json"
LOCK_FILE = "control/supervisor.lock"
START_LOCK_FILE = "control/start.lock"
TERMINAL_STATES = frozenset({"complete", "drained", "failure_hold", "integrity_hold"})
CAPSULE_KINDS = frozenset({"corpus", "aggregate", "verifier", "exploratory"})
INJECTABLE_KIND = "exploratory"
LANES = frozenset({"heavy", "cpu", "network", "light"})
ACCELERATORS = frozenset({"none", "mps"})
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CAPSULE_ATTEMPTS = 3
MAX_RUNTIME_EVENTS = 256
PROCESS_IDENTITY_TOLERANCE_SECONDS = 0.01


class Generation1Refused(RuntimeError):
    """Fail-closed manifest, injection, execution, or recovery refusal."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Generation1Refused(f"{label} must be an object")
    if set(value) != expected:
        raise Generation1Refused(
            f"{label} fields drifted; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise Generation1Refused(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise Generation1Refused(f"{label} is invalid")
    return value


def _safe_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Generation1Refused(f"{label} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise Generation1Refused(f"{label} must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise Generation1Refused(f"{label} escapes the repository")
    return path


def _read_json(path: Path, label: str, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise Generation1Refused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise Generation1Refused(f"{label} must be a regular non-symlink file")
    if stat.st_size <= 0 or stat.st_size > maximum_bytes:
        raise Generation1Refused(f"{label} byte envelope is invalid")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise Generation1Refused(f"{label} changed while being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Generation1Refused(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise Generation1Refused(f"{label} must be an object")
    return value


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: canonical_sha256(core)}


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise Generation1Refused(f"{label} self-seal mismatch")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path, "immutable receipt")
        if existing != dict(payload):
            raise Generation1Refused(f"immutable receipt drifted: {path}")
        return
    atomic_write_json(path, payload)


def _dotted(payload: object, path: str) -> object:
    value = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


@dataclass(frozen=True, slots=True)
class FileAuthority:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactExpectation:
    path: str
    schema: str
    fields: tuple[tuple[str, Any], ...]
    seal_field: str | None


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    lane: str
    accelerator: str
    cpu_cores: int
    estimated_unified_memory_gb: float
    estimated_mps_gb: float
    resource_basis: str
    forecast_write_gb: float
    atomic_write_gb: float
    wall_minutes: int
    process_marker: str


@dataclass(frozen=True, slots=True)
class Capsule:
    capsule_id: str
    kind: str
    priority: int
    depends_on: tuple[str, ...]
    command: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    resources: ResourceEnvelope
    artifacts: tuple[ArtifactExpectation, ...]
    authorities: tuple[FileAuthority, ...]
    capsule_sha256: str

    def task_declaration(self) -> TaskDeclaration:
        coexistence = is_taskpolicy_coexistence_command(self.command)
        return TaskDeclaration(
            task_id=self.capsule_id,
            lane=self.resources.lane,
            accelerator=self.resources.accelerator,
            cpu_cores=self.resources.cpu_cores,
            estimated_unified_memory_gb=self.resources.estimated_unified_memory_gb,
            estimated_mps_gb=self.resources.estimated_mps_gb,
            resource_basis=self.resources.resource_basis,
            forecast_write_gb=self.resources.forecast_write_gb,
            atomic_write_gb=self.resources.atomic_write_gb,
            wall_minutes=self.resources.wall_minutes,
            pause_safe=coexistence,
            atomic_checkpoints=coexistence,
            checkpoint_globs=(tuple(artifact.path for artifact in self.artifacts) if coexistence else ()),
            restart_exit_codes=(),
            command=self.command,
            requires_empty_lanes=True,
        )


@dataclass(frozen=True, slots=True)
class Program:
    path: Path
    file_sha256: str
    program_sha256: str
    program_id: str
    repo_root: Path
    program_root: Path
    policy: FileAuthority
    authorities: tuple[FileAuthority, ...]
    inbox: Path
    receipt_root: Path
    throttle_state_root: Path
    admission_samples: int
    admission_interval_seconds: float
    resource_retry_seconds: float
    startup_ack_seconds: float
    capsules: tuple[Capsule, ...]

    @property
    def state_path(self) -> Path:
        return self.program_root / STATE_FILE

    @property
    def status_path(self) -> Path:
        return self.program_root / STATUS_FILE


def _parse_authority(raw: object, root: Path, label: str, *, require_file: bool) -> FileAuthority:
    value = _exact_keys(raw, {"path", "sha256"}, label)
    path = _safe_path(root, value["path"], f"{label}.path")
    digest = _digest(value["sha256"], f"{label}.sha256")
    if require_file:
        if not path.is_file() or path.is_symlink():
            raise Generation1Refused(f"{label} is missing or not a regular file")
        if sha256_file(path) != digest:
            raise Generation1Refused(f"{label} hash drifted")
    return FileAuthority(str(value["path"]), digest)


def _parse_artifact(raw: object, root: Path, label: str) -> ArtifactExpectation:
    value = _exact_keys(raw, {"path", "schema", "fields", "seal_field"}, label)
    _safe_path(root, value["path"], f"{label}.path")
    schema = value["schema"]
    fields = value["fields"]
    seal = value["seal_field"]
    if not isinstance(schema, str) or not schema:
        raise Generation1Refused(f"{label}.schema must be nonempty")
    if not isinstance(fields, Mapping):
        raise Generation1Refused(f"{label}.fields must be an object")
    if seal is not None and (not isinstance(seal, str) or not seal):
        raise Generation1Refused(f"{label}.seal_field must be null or nonempty")
    return ArtifactExpectation(
        path=str(value["path"]),
        schema=schema,
        fields=tuple((str(key), expected) for key, expected in fields.items()),
        seal_field=seal,
    )


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Generation1Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise Generation1Refused(f"{label} must be finite and nonnegative")
    return number


def _parse_resources(raw: object, label: str) -> ResourceEnvelope:
    value = _exact_keys(
        raw,
        {
            "lane",
            "accelerator",
            "cpu_cores",
            "estimated_unified_memory_gb",
            "estimated_mps_gb",
            "resource_basis",
            "forecast_write_gb",
            "atomic_write_gb",
            "wall_minutes",
            "process_marker",
        },
        label,
    )
    lane = str(value["lane"])
    accelerator = str(value["accelerator"])
    cores = value["cpu_cores"]
    wall = value["wall_minutes"]
    marker = value["process_marker"]
    if lane not in LANES or accelerator not in ACCELERATORS:
        raise Generation1Refused(f"{label} lane or accelerator is invalid")
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise Generation1Refused(f"{label}.cpu_cores must be positive")
    if isinstance(wall, bool) or not isinstance(wall, int) or wall < 1:
        raise Generation1Refused(f"{label}.wall_minutes must be positive")
    if not isinstance(value["resource_basis"], str) or not value["resource_basis"].strip():
        raise Generation1Refused(f"{label}.resource_basis must be nonempty")
    if not isinstance(marker, str) or not marker.strip():
        raise Generation1Refused(f"{label}.process_marker must be nonempty")
    memory = _finite_nonnegative(value["estimated_unified_memory_gb"], f"{label}.memory")
    mps = _finite_nonnegative(value["estimated_mps_gb"], f"{label}.mps")
    if accelerator == "none" and mps != 0:
        raise Generation1Refused(f"{label}.estimated_mps_gb must be zero without MPS")
    return ResourceEnvelope(
        lane=lane,
        accelerator=accelerator,
        cpu_cores=cores,
        estimated_unified_memory_gb=memory,
        estimated_mps_gb=mps,
        resource_basis=str(value["resource_basis"]),
        forecast_write_gb=_finite_nonnegative(value["forecast_write_gb"], f"{label}.write"),
        atomic_write_gb=_finite_nonnegative(value["atomic_write_gb"], f"{label}.atomic_write"),
        wall_minutes=wall,
        process_marker=marker,
    )


def _parse_capsule(raw: object, root: Path, label: str, *, injectable: bool) -> Capsule:
    value = _exact_keys(
        raw,
        {
            "schema",
            "id",
            "kind",
            "priority",
            "depends_on",
            "command",
            "cwd",
            "environment",
            "resources",
            "artifacts",
            "authorities",
            "capsule_sha256",
        },
        label,
    )
    if value["schema"] != CAPSULE_SCHEMA:
        raise Generation1Refused(f"{label} schema must be {CAPSULE_SCHEMA}")
    _validate_seal(value, "capsule_sha256", label)
    capsule_id = _identifier(value["id"], f"{label}.id")
    kind = str(value["kind"])
    if kind not in CAPSULE_KINDS or (injectable and kind != INJECTABLE_KIND):
        raise Generation1Refused(f"{label}.kind is not permitted")
    priority = value["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        raise Generation1Refused(f"{label}.priority must be a nonnegative integer")
    dependencies = value["depends_on"]
    command = value["command"]
    environment = value["environment"]
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise Generation1Refused(f"{label}.depends_on must be a list of ids")
    if len(dependencies) != len(set(dependencies)) or capsule_id in dependencies:
        raise Generation1Refused(f"{label}.depends_on is duplicated or self-referential")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise Generation1Refused(f"{label}.command must be a nonempty argv list")
    cwd = str(value["cwd"])
    cwd_path = _safe_path(root, cwd, f"{label}.cwd")
    if not cwd_path.is_dir():
        raise Generation1Refused(f"{label}.cwd is not a directory")
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not key or "=" in key or not isinstance(item, str)
        for key, item in environment.items()
    ):
        raise Generation1Refused(f"{label}.environment must map variable names to strings")
    artifacts_raw = value["artifacts"]
    authorities_raw = value["authorities"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise Generation1Refused(f"{label}.artifacts must be nonempty")
    if not isinstance(authorities_raw, list) or not authorities_raw:
        raise Generation1Refused(f"{label}.authorities must be nonempty")
    resources = _parse_resources(value["resources"], f"{label}.resources")
    if resources.process_marker not in " ".join(command):
        raise Generation1Refused(f"{label} command does not contain its process marker")
    taskpolicy_command = is_taskpolicy_coexistence_command(command)
    if Path(command[0]).name not in {"python", "python3"} and not taskpolicy_command:
        raise Generation1Refused(
            f"{label} command must use an explicit Python interpreter or the pinned taskpolicy wrapper"
        )
    artifacts = tuple(
        _parse_artifact(item, root, f"{label}.artifacts[{index}]") for index, item in enumerate(artifacts_raw)
    )
    authorities = tuple(
        _parse_authority(item, root, f"{label}.authorities[{index}]", require_file=True)
        for index, item in enumerate(authorities_raw)
    )
    authority_labels = {
        label for authority in authorities for label in (authority.path, Path(authority.path).name)
    }
    if resources.process_marker not in authority_labels:
        raise Generation1Refused(f"{label} process marker is not a file authority")
    return Capsule(
        capsule_id=capsule_id,
        kind=kind,
        priority=priority,
        depends_on=tuple(dependencies),
        command=tuple(command),
        cwd=cwd,
        environment=tuple(sorted((str(key), str(item)) for key, item in environment.items())),
        resources=resources,
        artifacts=artifacts,
        authorities=authorities,
        capsule_sha256=_digest(value["capsule_sha256"], f"{label}.capsule_sha256"),
    )


def _validate_capsules(capsules: Sequence[Capsule]) -> None:
    by_id = {capsule.capsule_id: capsule for capsule in capsules}
    if len(by_id) != len(capsules):
        raise Generation1Refused("capsule ids are duplicated")
    artifact_paths = [artifact.path for capsule in capsules for artifact in capsule.artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise Generation1Refused("capsule artifact paths are duplicated")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(capsule_id: str) -> None:
        if capsule_id in visiting:
            raise Generation1Refused(f"capsule dependency cycle at {capsule_id}")
        if capsule_id in visited:
            return
        visiting.add(capsule_id)
        capsule = by_id[capsule_id]
        for dependency in capsule.depends_on:
            if dependency not in by_id:
                raise Generation1Refused(f"capsule {capsule_id} has missing dependency {dependency}")
            visit(dependency)
        visiting.remove(capsule_id)
        visited.add(capsule_id)

    for capsule_id in sorted(by_id):
        visit(capsule_id)


def load_program(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> Program:
    root = Path(repo_root).resolve()
    source = Path(path).resolve()
    if not source.is_relative_to(root):
        raise Generation1Refused("program manifest must be inside the repository")
    raw = _read_json(source, "Generation 1 program manifest")
    value = _exact_keys(
        raw,
        {
            "schema",
            "program_id",
            "program_root",
            "policy",
            "authorities",
            "injection",
            "control",
            "capsules",
            "program_sha256",
        },
        "program manifest",
    )
    if value["schema"] != PROGRAM_SCHEMA:
        raise Generation1Refused(f"program schema must be {PROGRAM_SCHEMA}")
    _validate_seal(value, "program_sha256", "program manifest")
    program_id = _identifier(value["program_id"], "program_id")
    program_root = _safe_path(root, value["program_root"], "program_root")
    policy = _parse_authority(value["policy"], root, "policy", require_file=True)
    authorities_raw = value["authorities"]
    if not isinstance(authorities_raw, list) or not authorities_raw:
        raise Generation1Refused("program authorities must be nonempty")
    authorities = tuple(
        _parse_authority(item, root, f"program authorities[{index}]", require_file=True)
        for index, item in enumerate(authorities_raw)
    )
    injection = _exact_keys(value["injection"], {"inbox", "receipt_root"}, "injection")
    control = _exact_keys(
        value["control"],
        {
            "throttle_state_root",
            "admission_samples",
            "admission_interval_seconds",
            "resource_retry_seconds",
            "startup_ack_seconds",
        },
        "control",
    )
    admission_samples = control["admission_samples"]
    if isinstance(admission_samples, bool) or not isinstance(admission_samples, int) or admission_samples < 1:
        raise Generation1Refused("control.admission_samples must be positive")

    def positive_number(name: str) -> float:
        number = control[name]
        if isinstance(number, bool) or not isinstance(number, int | float) or float(number) <= 0:
            raise Generation1Refused(f"control.{name} must be positive")
        return float(number)

    capsules_raw = value["capsules"]
    if not isinstance(capsules_raw, list) or not capsules_raw:
        raise Generation1Refused("program capsules must be nonempty")
    capsules = tuple(
        _parse_capsule(item, root, f"program capsules[{index}]", injectable=False)
        for index, item in enumerate(capsules_raw)
    )
    _validate_capsules(capsules)
    return Program(
        path=source,
        file_sha256=sha256_file(source),
        program_sha256=_digest(value["program_sha256"], "program_sha256"),
        program_id=program_id,
        repo_root=root,
        program_root=program_root,
        policy=policy,
        authorities=authorities,
        inbox=_safe_path(root, injection["inbox"], "injection.inbox"),
        receipt_root=_safe_path(root, injection["receipt_root"], "injection.receipt_root"),
        throttle_state_root=_safe_path(root, control["throttle_state_root"], "control.throttle_state_root"),
        admission_samples=admission_samples,
        admission_interval_seconds=positive_number("admission_interval_seconds"),
        resource_retry_seconds=positive_number("resource_retry_seconds"),
        startup_ack_seconds=positive_number("startup_ack_seconds"),
        capsules=capsules,
    )


def _empty_runtime_summary() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "peak_process_tree_rss_bytes": 0,
        "minimum_memory_available_gb": None,
        "minimum_memory_available_percent": None,
        "minimum_memory_pressure_free_percent": None,
        "maximum_swap_used_gb": None,
        "minimum_disk_free_gb": None,
        "thermal_statuses": [],
        "power_sources": [],
        "reservation_count": 0,
        "last_reservation": None,
        "resource_stop_count": 0,
        "retry_count": 0,
        "failure_attempt_count": 0,
        "safety_state": "not-started",
        "last_sample": None,
        "event_count": 0,
        "events_dropped": 0,
        "events": [],
    }


def _capsule_row(capsule: Capsule, source: str) -> dict[str, Any]:
    return {
        "id": capsule.capsule_id,
        "kind": capsule.kind,
        "priority": capsule.priority,
        "depends_on": list(capsule.depends_on),
        "capsule_sha256": capsule.capsule_sha256,
        "source": source,
        "status": "pending",
        "attempts": 0,
        "child_pid": None,
        "child_create_time": None,
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "artifacts": [],
        "last_problem": None,
        "runtime": _empty_runtime_summary(),
    }


def _genesis_head(program: Program) -> str:
    return canonical_sha256(
        {
            "program_sha256": program.program_sha256,
            "base_capsules": [capsule.capsule_sha256 for capsule in program.capsules],
        }
    )


@dataclass(frozen=True, slots=True)
class Admission:
    allowed: bool
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    child_pid: int | None
    child_create_time: float | None
    started_at: str
    finished_at: str
    stdout_path: str
    stderr_path: str
    runtime_problem: str | None = None
    resource_deferred: bool = False


AdmissionProbe = Callable[[Capsule], Admission]
StartCallback = Callable[[int, float | None], None]
Runner = Callable[[Capsule, Path, Path, Mapping[str, str], StartCallback], CommandResult]
LaneReserver = Callable[[Capsule, int | None, float | None], Mapping[str, Any]]
LaneReleaser = Callable[[Capsule], None]


def _process_identity_state(pid: object, create_time: object) -> str:
    """Return alive, gone, or unknown without treating a reused PID as the recorded child."""

    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(create_time, bool)
        or not isinstance(create_time, int | float)
    ):
        return "unknown"
    try:
        process = psutil.Process(pid)
        observed_create_time = process.create_time()
        if not math.isclose(
            observed_create_time,
            float(create_time),
            rel_tol=0.0,
            abs_tol=PROCESS_IDENTITY_TOLERANCE_SECONDS,
        ):
            return "gone"
        if process.status() == psutil.STATUS_ZOMBIE:
            return "gone"
        return "alive" if process.is_running() else "gone"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "gone"
    except (psutil.AccessDenied, OSError, ValueError):
        return "unknown"


def _process_tree_rss_bytes(pid: int, create_time: float) -> int | None:
    """Measure one exact process tree, returning None when identity cannot be proven."""

    identity = _process_identity_state(pid, create_time)
    if identity == "gone":
        return 0
    if identity != "alive":
        return None
    try:
        process = psutil.Process(pid)
        values = [int(process.memory_info().rss)]
        for child in process.children(recursive=True):
            try:
                values.append(int(child.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return sum(values)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 0
    except (psutil.AccessDenied, OSError, ValueError):
        return None


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise Generation1Refused(f"Generation 1 lock is already held: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class Generation1Supervisor:
    def __init__(
        self,
        program: Program,
        *,
        execute: bool = False,
        admission_probe: AdmissionProbe | None = None,
        runner: Runner | None = None,
        lane_reserver: LaneReserver | None = None,
        lane_releaser: LaneReleaser | None = None,
        now_fn: Callable[[], datetime] = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.program = program
        self.execute = execute
        self.admission_probe = admission_probe or self._default_admission
        self.runner = runner or self._default_runner
        self.lane_reserver = lane_reserver or self._default_lane_reserver
        self.lane_releaser = lane_releaser or self._default_lane_releaser
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.loaded_implementation_sha256 = sha256_file(Path(__file__))
        self._reserved_capsule_id: str | None = None
        self.state = self._load_or_create_state()

    def _identity(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        implementation_path = Path(__file__).resolve()
        try:
            implementation_label = str(implementation_path.relative_to(self.program.repo_root))
        except ValueError:  # test repositories may use a temporary root
            implementation_label = str(implementation_path)
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "implementation_path": implementation_label,
            "implementation_sha256": self.loaded_implementation_sha256,
        }

    @property
    def _lane_run_id(self) -> str:
        return f"generation1-{self.program.program_id}"

    def _load_live_policy(self) -> ThrottlePolicy:
        self._validate_file_authority(self.program.policy, "policy")
        try:
            return load_policy(_safe_path(self.program.repo_root, self.program.policy.path, "policy"))
        except ThrottleRefused as exc:
            raise Generation1Refused(f"live throttle policy is invalid: {exc}") from exc

    def _append_runtime_event(
        self,
        capsule: Capsule,
        event: str,
        **payload: Any,
    ) -> None:
        row = self.state["capsules"][capsule.capsule_id]
        runtime = dict(row.get("runtime") or _empty_runtime_summary())
        sequence = int(runtime.get("event_count") or 0) + 1
        events = list(runtime.get("events") or [])
        events.append(
            {
                "sequence": sequence,
                "at": _iso(self.now_fn()),
                "event": event,
                **payload,
            }
        )
        if len(events) > MAX_RUNTIME_EVENTS:
            dropped = len(events) - MAX_RUNTIME_EVENTS
            events = events[-MAX_RUNTIME_EVENTS:]
            runtime["events_dropped"] = int(runtime.get("events_dropped") or 0) + dropped
        runtime["event_count"] = sequence
        runtime["events"] = events
        row["runtime"] = runtime

    def _default_lane_reserver(
        self,
        capsule: Capsule,
        observed_child_pid: int | None,
        observed_child_create_time: float | None,
    ) -> Mapping[str, Any]:
        policy = self._load_live_policy()
        excluded_pids: set[int] = set()
        excluded_groups: set[int] = set()
        if observed_child_pid is not None:
            if _process_identity_state(observed_child_pid, observed_child_create_time) != "alive":
                raise Generation1Refused("cannot reserve a recovery lane for an inexact child")
            excluded_pids.add(observed_child_pid)
            excluded_groups.add(observed_child_pid)
        telemetry = collect_host_telemetry(
            policy,
            disk_root=self.program.repo_root,
            excluded_pids=excluded_pids,
            excluded_process_groups=excluded_groups,
        )
        task = capsule.task_declaration()
        try:
            decision = local_throttle_runtime._reserve_registry(  # noqa: SLF001
                self.program.throttle_state_root,
                self._lane_run_id,
                task,
                telemetry,
                policy,
            )
            if observed_child_pid is not None:
                local_throttle_runtime._update_registry(  # noqa: SLF001
                    self.program.throttle_state_root,
                    self._lane_run_id,
                    local_throttle_runtime._registry_row(  # noqa: SLF001
                        task,
                        self._lane_run_id,
                        observed_child_pid,
                        "recovery-observe",
                    ),
                )
        except ThrottleRefused as exc:
            raise Generation1Refused(f"atomic Generation1 lane reservation refused: {exc}") from exc
        return {
            "reserved": True,
            "run_id": self._lane_run_id,
            "capsule_id": capsule.capsule_id,
            "observed_child_pid": observed_child_pid,
            "observed_child_create_time": observed_child_create_time,
            "decision": decision,
        }

    def _default_lane_releaser(self, _capsule: Capsule) -> None:
        try:
            local_throttle_runtime._update_registry(  # noqa: SLF001
                self.program.throttle_state_root,
                self._lane_run_id,
                None,
            )
        except ThrottleRefused as exc:
            raise Generation1Refused(f"atomic Generation1 lane release failed: {exc}") from exc

    def _reserve_lane(
        self,
        capsule: Capsule,
        child_pid: int | None = None,
        child_create_time: float | None = None,
    ) -> None:
        if self._reserved_capsule_id is not None:
            if self._reserved_capsule_id != capsule.capsule_id:
                raise Generation1Refused(
                    f"Generation1 lane is already reserved for {self._reserved_capsule_id}"
                )
            return
        receipt = dict(self.lane_reserver(capsule, child_pid, child_create_time))
        self._reserved_capsule_id = capsule.capsule_id
        self.state["lane_reservation"] = {
            "capsule_id": capsule.capsule_id,
            "reserved_at": _iso(self.now_fn()),
            "child_pid": child_pid,
            "child_create_time": child_create_time,
            "receipt": receipt,
        }
        runtime = dict(self.state["capsules"][capsule.capsule_id].get("runtime") or _empty_runtime_summary())
        runtime["reservation_count"] = int(runtime.get("reservation_count") or 0) + 1
        runtime["last_reservation"] = dict(self.state["lane_reservation"])
        self.state["capsules"][capsule.capsule_id]["runtime"] = runtime
        self._append_runtime_event(
            capsule,
            "lane-reserved",
            child_pid=child_pid,
            recovery=child_pid is not None,
        )

    def _release_lane(self, capsule: Capsule) -> None:
        if self._reserved_capsule_id is None:
            return
        if self._reserved_capsule_id != capsule.capsule_id:
            raise Generation1Refused(
                f"cannot release {self._reserved_capsule_id} as capsule {capsule.capsule_id}"
            )
        self.lane_releaser(capsule)
        self._append_runtime_event(capsule, "lane-released")
        self._reserved_capsule_id = None
        self.state["lane_reservation"] = None

    def _sample_runtime(
        self,
        capsule: Capsule,
        child_pid: int,
        child_create_time: float,
    ) -> dict[str, Any]:
        policy = self._load_live_policy()
        telemetry = collect_host_telemetry(
            policy,
            disk_root=self.program.repo_root,
            excluded_pids={child_pid},
            excluded_process_groups={child_pid},
        )
        other_active = [
            row
            for row in active_lanes(self.program.throttle_state_root)
            if row.get("run_id") != self._lane_run_id
        ]
        decision = evaluate_task(
            capsule.task_declaration(),
            telemetry,
            policy,
            active=other_active,
            task_already_active=True,
            evidence_root=self.program.repo_root,
        )
        identity_state = _process_identity_state(child_pid, child_create_time)
        process_tree_rss = _process_tree_rss_bytes(child_pid, child_create_time)
        declared_rss = int(capsule.resources.estimated_unified_memory_gb * 1_000_000_000)
        rss_ok = process_tree_rss is not None and (
            identity_state == "gone" or process_tree_rss <= declared_rss
        )
        identity_ok = identity_state != "unknown"
        denied_reasons = list(decision.get("denied_reasons") or [])
        if not identity_ok:
            denied_reasons.append("recorded child identity cannot be verified")
        if not rss_ok:
            denied_reasons.append("owned process-tree RSS exceeds its declaration or cannot be measured")
        memory = telemetry.get("memory") or {}
        pressure = memory.get("pressure") or {}
        swap = telemetry.get("swap") or {}
        disk = telemetry.get("disk") or {}
        thermal = telemetry.get("thermal") or {}
        power = telemetry.get("power") or {}
        allowed = bool(decision.get("allowed")) and identity_ok and rss_ok
        critical = (
            bool(decision.get("critical"))
            or not identity_ok
            or (
                process_tree_rss is not None and identity_state == "alive" and process_tree_rss > declared_rss
            )
        )
        return {
            "created_at": _iso(self.now_fn()),
            "child": {
                "pid": child_pid,
                "create_time": child_create_time,
                "identity_state": identity_state,
            },
            "host": {
                "memory_available_gb": float(memory.get("available_bytes") or 0) / 1e9,
                "memory_available_percent": memory.get("available_percent"),
                "memory_pressure_available": bool(pressure.get("available")),
                "memory_pressure_free_percent": pressure.get("free_percent"),
                "swap_used_gb": swap.get("used_gb"),
                "disk_free_gb": disk.get("free_gb"),
                "thermal_available": bool(thermal.get("available")),
                "thermal_status": thermal.get("status"),
                "power_available": bool(power.get("available")),
                "power_source": power.get("source"),
                "power_on_ac": power.get("on_ac"),
                "missing_required_telemetry": list(telemetry.get("missing_required_telemetry") or []),
            },
            "process_tree_rss_bytes": process_tree_rss,
            "declared_process_tree_rss_bytes": declared_rss,
            "allowed": allowed,
            "critical": critical,
            "denied_reasons": denied_reasons,
            "disk_forecast": decision.get("disk_forecast"),
        }

    def _record_runtime_sample(
        self,
        capsule: Capsule,
        sample: Mapping[str, Any],
        *,
        mode: str,
    ) -> dict[str, Any]:
        row = self.state["capsules"][capsule.capsule_id]
        runtime = dict(row.get("runtime") or {})
        runtime["sample_count"] = int(runtime.get("sample_count") or 0) + 1
        rss = sample.get("process_tree_rss_bytes")
        peak = int(runtime.get("peak_process_tree_rss_bytes") or 0)
        if isinstance(rss, int) and not isinstance(rss, bool):
            peak = max(peak, rss)
        runtime["peak_process_tree_rss_bytes"] = peak
        host = sample.get("host")
        if isinstance(host, Mapping):
            for runtime_key, host_key in (
                ("minimum_memory_available_gb", "memory_available_gb"),
                ("minimum_memory_available_percent", "memory_available_percent"),
                (
                    "minimum_memory_pressure_free_percent",
                    "memory_pressure_free_percent",
                ),
                ("minimum_disk_free_gb", "disk_free_gb"),
            ):
                observed = host.get(host_key)
                prior = runtime.get(runtime_key)
                if isinstance(observed, int | float) and not isinstance(observed, bool):
                    runtime[runtime_key] = (
                        float(observed)
                        if not isinstance(prior, int | float) or isinstance(prior, bool)
                        else min(float(prior), float(observed))
                    )
            swap = host.get("swap_used_gb")
            prior_swap = runtime.get("maximum_swap_used_gb")
            if isinstance(swap, int | float) and not isinstance(swap, bool):
                runtime["maximum_swap_used_gb"] = (
                    float(swap)
                    if not isinstance(prior_swap, int | float) or isinstance(prior_swap, bool)
                    else max(float(prior_swap), float(swap))
                )
            thermal_status = host.get("thermal_status")
            thermal_statuses = list(runtime.get("thermal_statuses") or [])
            if isinstance(thermal_status, str) and thermal_status not in thermal_statuses:
                thermal_statuses.append(thermal_status)
            runtime["thermal_statuses"] = thermal_statuses
            power_source = host.get("power_source")
            power_sources = list(runtime.get("power_sources") or [])
            if isinstance(power_source, str) and power_source not in power_sources:
                power_sources.append(power_source)
            runtime["power_sources"] = power_sources
        runtime["safety_state"] = f"{mode}-safe" if bool(sample.get("allowed")) else f"{mode}-unsafe"
        runtime["last_sample"] = dict(sample)
        row["runtime"] = runtime
        return self._publish()

    def _record_resource_stop(
        self,
        capsule: Capsule,
        *,
        reason: str,
        wall_boundary: bool,
    ) -> None:
        row = self.state["capsules"][capsule.capsule_id]
        runtime = dict(row.get("runtime") or _empty_runtime_summary())
        runtime["resource_stop_count"] = int(runtime.get("resource_stop_count") or 0) + 1
        runtime["safety_state"] = "wall-stop" if wall_boundary else "resource-stop"
        row["runtime"] = runtime
        self._append_runtime_event(
            capsule,
            "wall-boundary-stop" if wall_boundary else "runtime-resource-stop",
            reason=reason,
        )
        self._publish()

    @staticmethod
    def _signal_exact_owned_group(
        process: subprocess.Popen[Any],
        child_create_time: float | None,
        requested_signal: signal.Signals,
    ) -> None:
        if process.poll() is not None:
            return
        if _process_identity_state(process.pid, child_create_time) != "alive":
            raise Generation1Refused("owned child identity is not exact; no process signal was sent")
        try:
            if os.getpgid(process.pid) != process.pid:
                raise Generation1Refused(
                    "owned child is not the leader of its isolated process group; no signal sent"
                )
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            return
        except Generation1Refused:
            raise
        except (PermissionError, OSError) as exc:
            raise Generation1Refused(
                f"could not signal exact owned process group: {type(exc).__name__}: {exc}"
            ) from exc

    def _stop_exact_owned_process(
        self,
        process: subprocess.Popen[Any],
        child_create_time: float | None,
        *,
        wall_boundary: bool,
        grace_seconds: float,
    ) -> int:
        signals = (
            (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
            if wall_boundary
            else (signal.SIGTERM, signal.SIGKILL)
        )
        for requested_signal in signals:
            if process.poll() is not None:
                return int(process.returncode)
            self._signal_exact_owned_group(process, child_create_time, requested_signal)
            try:
                return process.wait(timeout=max(0.1, grace_seconds))
            except subprocess.TimeoutExpired:
                continue
        raise Generation1Refused("exact owned process group remained alive after bounded shutdown")

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.program.state_path.is_file():
            state = _read_json(self.program.state_path, "Generation 1 state")
            _validate_seal(state, "state_sha256", "Generation 1 state")
            if state.get("schema") != STATE_SCHEMA:
                raise Generation1Refused("Generation 1 state schema drifted")
            if state.get("program") != {
                "path": str(self.program.path),
                "file_sha256": self.program.file_sha256,
                "program_sha256": self.program.program_sha256,
            }:
                raise Generation1Refused("Generation 1 state program authority drifted")
            state["supervisor"] = self._identity()
            state["execution_enabled"] = self.execute
            state["lane_reservation"] = None
            rows = state.get("capsules")
            if isinstance(rows, Mapping):
                for row in rows.values():
                    if isinstance(row, dict):
                        runtime = _empty_runtime_summary()
                        existing_runtime = row.get("runtime")
                        if isinstance(existing_runtime, Mapping):
                            runtime.update(existing_runtime)
                        row["runtime"] = runtime
            return state
        now = _iso(self.now_fn())
        return {
            "schema": STATE_SCHEMA,
            "program_id": self.program.program_id,
            "program": {
                "path": str(self.program.path),
                "file_sha256": self.program.file_sha256,
                "program_sha256": self.program.program_sha256,
            },
            "supervisor": self._identity(),
            "execution_enabled": self.execute,
            "status": "starting",
            "queue_head_sha256": _genesis_head(self.program),
            "next_injection_sequence": 1,
            "accepted_injections": [],
            "processed_injection_files": [],
            "capsules": {
                capsule.capsule_id: _capsule_row(capsule, "base") for capsule in self.program.capsules
            },
            "current_capsule": None,
            "last_admission": None,
            "lane_reservation": None,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "problems": [],
        }

    def _all_capsules(self) -> dict[str, Capsule]:
        capsules = {capsule.capsule_id: capsule for capsule in self.program.capsules}
        for accepted in self.state.get("accepted_injections", []):
            path = _safe_path(self.program.repo_root, accepted["path"], "accepted injection path")
            injection = self._load_injection(path)
            for capsule in injection["capsules"]:
                capsules[capsule.capsule_id] = capsule
        _validate_capsules(tuple(capsules.values()))
        return capsules

    def _validate_state_capsules(self, capsules: Mapping[str, Capsule]) -> None:
        rows = self.state.get("capsules")
        if not isinstance(rows, Mapping) or set(rows) != set(capsules):
            raise Generation1Refused("state capsule inventory drifted")
        valid_statuses = {"pending", "running", "complete", "failed"}
        for capsule_id, capsule in capsules.items():
            row = rows[capsule_id]
            if not isinstance(row, Mapping):
                raise Generation1Refused(f"state capsule {capsule_id} is invalid")
            expected = {
                "id": capsule.capsule_id,
                "kind": capsule.kind,
                "priority": capsule.priority,
                "depends_on": list(capsule.depends_on),
                "capsule_sha256": capsule.capsule_sha256,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise Generation1Refused(f"state capsule {capsule_id} authority drifted")
            if row.get("status") not in valid_statuses:
                raise Generation1Refused(f"state capsule {capsule_id} status drifted")

    def _hold(self, status: str, problem: str) -> None:
        self.state["status"] = status
        if problem not in self.state["problems"]:
            self.state["problems"].append(problem)
        self.state["finished_at"] = _iso(self.now_fn())

    def _validate_program_authority(self) -> None:
        if sha256_file(self.program.path) != self.program.file_sha256:
            raise Generation1Refused("program manifest file drifted")
        if sha256_file(Path(__file__)) != self.loaded_implementation_sha256:
            raise Generation1Refused("loaded supervisor implementation drifted on disk")
        self._validate_file_authority(self.program.policy, "policy")
        for index, authority in enumerate(self.program.authorities):
            self._validate_file_authority(authority, f"program authority {index}")

    def _validate_file_authority(self, authority: FileAuthority, label: str) -> None:
        path = _safe_path(self.program.repo_root, authority.path, label)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != authority.sha256:
            raise Generation1Refused(f"{label} hash drifted")

    def _validate_capsule_authorities(self, capsule: Capsule) -> None:
        for index, authority in enumerate(capsule.authorities):
            self._validate_file_authority(authority, f"capsule {capsule.capsule_id} authority {index}")

    def _load_injection(self, path: Path) -> dict[str, Any]:
        raw = _read_json(path, "Generation 1 injection")
        value = _exact_keys(
            raw,
            {
                "schema",
                "program_id",
                "injection_id",
                "sequence",
                "created_at",
                "action",
                "expected_queue_head_sha256",
                "capsules",
                "reason",
                "injection_sha256",
            },
            "injection",
        )
        if value["schema"] != INJECTION_SCHEMA or value["action"] != "append-capsules":
            raise Generation1Refused("injection schema or action is invalid")
        _validate_seal(value, "injection_sha256", "injection")
        injection_id = _identifier(value["injection_id"], "injection_id")
        if value["program_id"] != self.program.program_id:
            raise Generation1Refused("injection program id drifted")
        sequence = value["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise Generation1Refused("injection sequence must be positive")
        if not isinstance(value["created_at"], str) or not isinstance(value["reason"], str):
            raise Generation1Refused("injection created_at and reason must be strings")
        expected_head = _digest(value["expected_queue_head_sha256"], "expected queue head")
        rows = value["capsules"]
        if not isinstance(rows, list) or not rows:
            raise Generation1Refused("injection capsules must be nonempty")
        capsules = tuple(
            _parse_capsule(item, self.program.repo_root, f"injection capsules[{index}]", injectable=True)
            for index, item in enumerate(rows)
        )
        return {
            "id": injection_id,
            "sequence": sequence,
            "expected_head": expected_head,
            "sha256": _digest(value["injection_sha256"], "injection_sha256"),
            "capsules": capsules,
        }

    def _validate_accepted_chain(self) -> None:
        head = _genesis_head(self.program)
        next_sequence = 1
        seen_ids: set[str] = set()
        capsules = {capsule.capsule_id: capsule for capsule in self.program.capsules}
        for accepted in self.state.get("accepted_injections", []):
            path = _safe_path(self.program.repo_root, accepted.get("path"), "accepted injection")
            if sha256_file(path) != accepted.get("file_sha256"):
                raise Generation1Refused("accepted injection file changed")
            injection = self._load_injection(path)
            if (
                injection["id"] in seen_ids
                or injection["sequence"] != next_sequence
                or injection["expected_head"] != head
                or injection["sha256"] != accepted.get("injection_sha256")
                or accepted.get("previous_head_sha256") != head
            ):
                raise Generation1Refused("accepted injection chain drifted")
            for capsule in injection["capsules"]:
                if capsule.capsule_id in capsules:
                    raise Generation1Refused("accepted injection capsule id collided")
                capsules[capsule.capsule_id] = capsule
            new_head = canonical_sha256(
                {
                    "previous_head_sha256": head,
                    "injection_sha256": injection["sha256"],
                    "capsules": [capsule.capsule_sha256 for capsule in injection["capsules"]],
                }
            )
            if accepted.get("new_head_sha256") != new_head:
                raise Generation1Refused("accepted injection queue head drifted")
            head = new_head
            next_sequence += 1
            seen_ids.add(injection["id"])
        _validate_capsules(tuple(capsules.values()))
        if self.state.get("queue_head_sha256") != head:
            raise Generation1Refused("state queue head drifted")
        if self.state.get("next_injection_sequence") != next_sequence:
            raise Generation1Refused("state injection sequence drifted")
        for processed in self.state.get("processed_injection_files", []):
            path = _safe_path(self.program.repo_root, processed.get("path"), "processed injection")
            if not path.is_file() or sha256_file(path) != processed.get("file_sha256"):
                raise Generation1Refused("processed injection file changed")

    def _record_injection_receipt(
        self,
        *,
        path: Path,
        file_sha256: str,
        accepted: bool,
        injection_id: str,
        problems: Sequence[str],
        previous_head: str,
        new_head: str,
    ) -> None:
        core = {
            "schema": INJECTION_RECEIPT_SCHEMA,
            "program_id": self.program.program_id,
            "created_at": _iso(self.now_fn()),
            "injection_id": injection_id,
            "path": str(path.relative_to(self.program.repo_root)),
            "file_sha256": file_sha256,
            "accepted": accepted,
            "problems": list(problems),
            "previous_head_sha256": previous_head,
            "new_head_sha256": new_head,
            "scientific_promotion": False,
        }
        receipt = _sealed(core, "receipt_sha256")
        directory = "accepted" if accepted else "rejected"
        write_immutable_json(
            self.program.receipt_root / directory / f"{injection_id}-{file_sha256[:12]}.json",
            receipt,
        )

    def _consume_injections(self) -> None:
        self.program.inbox.mkdir(parents=True, exist_ok=True)
        processed_paths = {row["path"] for row in self.state.get("processed_injection_files", [])}
        for path in sorted(self.program.inbox.glob("*.json"), key=lambda item: item.name):
            relative = str(path.resolve().relative_to(self.program.repo_root))
            if relative in processed_paths:
                continue
            file_digest = sha256_file(path)
            prior_head = str(self.state["queue_head_sha256"])
            injection_id = f"invalid-{file_digest[:12]}"
            problems: list[str] = []
            accepted = False
            try:
                injection = self._load_injection(path)
                injection_id = injection["id"]
                if injection["sequence"] != self.state["next_injection_sequence"]:
                    raise Generation1Refused("injection sequence is not the exact next sequence")
                if injection["expected_head"] != prior_head:
                    raise Generation1Refused("injection expected queue head is stale")
                existing = self._all_capsules()
                for capsule in injection["capsules"]:
                    if capsule.capsule_id in existing:
                        raise Generation1Refused("injection capsule id already exists")
                    existing[capsule.capsule_id] = capsule
                _validate_capsules(tuple(existing.values()))
                new_head = canonical_sha256(
                    {
                        "previous_head_sha256": prior_head,
                        "injection_sha256": injection["sha256"],
                        "capsules": [capsule.capsule_sha256 for capsule in injection["capsules"]],
                    }
                )
                self.state["accepted_injections"].append(
                    {
                        "id": injection_id,
                        "sequence": injection["sequence"],
                        "path": relative,
                        "file_sha256": file_digest,
                        "injection_sha256": injection["sha256"],
                        "previous_head_sha256": prior_head,
                        "new_head_sha256": new_head,
                    }
                )
                for capsule in injection["capsules"]:
                    self.state["capsules"][capsule.capsule_id] = _capsule_row(
                        capsule, f"injection:{injection_id}"
                    )
                self.state["queue_head_sha256"] = new_head
                self.state["next_injection_sequence"] += 1
                accepted = True
            except (Generation1Refused, OSError, ValueError) as exc:
                new_head = prior_head
                problems.append(f"{type(exc).__name__}: {exc}")
            self.state["processed_injection_files"].append(
                {
                    "path": relative,
                    "file_sha256": file_digest,
                    "accepted": accepted,
                    "injection_id": injection_id,
                }
            )
            processed_paths.add(relative)
            self._record_injection_receipt(
                path=path,
                file_sha256=file_digest,
                accepted=accepted,
                injection_id=injection_id,
                problems=problems,
                previous_head=prior_head,
                new_head=new_head,
            )

    def _default_admission(self, capsule: Capsule) -> Admission:
        policy = load_policy(_safe_path(self.program.repo_root, self.program.policy.path, "policy"))
        if capsule.resources.wall_minutes > int(policy.limits["hard_wall_minutes"]):
            return Admission(
                False,
                {"allowed": False, "denied_reasons": ["capsule exceeds policy hard wall"]},
            )
        if capsule.resources.process_marker not in policy.monitor.get("known_heavy_markers", []):
            return Admission(
                False,
                {
                    "allowed": False,
                    "denied_reasons": ["capsule process marker is absent from known-heavy policy"],
                },
            )
        task = capsule.task_declaration()
        declaration_problems = task.validate(int(policy.limits["hard_wall_minutes"]))
        if declaration_problems:
            return Admission(
                False,
                {
                    "allowed": False,
                    "denied_reasons": [
                        f"invalid capsule resource declaration: {problem}" for problem in declaration_problems
                    ],
                },
            )
        active = active_lanes(self.program.throttle_state_root)
        snapshots: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for index in range(self.program.admission_samples):
            snapshot = collect_host_telemetry(policy, disk_root=self.program.repo_root)
            decision = evaluate_task(task, snapshot, policy, active=active)
            snapshots.append(snapshot)
            decisions.append(decision)
            if index + 1 < self.program.admission_samples:
                self.sleep_fn(self.program.admission_interval_seconds)
        aggregate = aggregate_admission(decisions, self.program.admission_samples)
        return Admission(
            bool(aggregate["allowed"]),
            {
                "allowed": bool(aggregate["allowed"]),
                "aggregate": aggregate,
                "decisions": decisions,
                "telemetry": snapshots,
            },
        )

    def _default_runner(
        self,
        capsule: Capsule,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str],
        on_start: StartCallback,
    ) -> CommandResult:
        started_at = _iso(self.now_fn())
        policy = self._load_live_policy()
        monitor_interval = max(0.05, float(policy.monitor["sample_interval_seconds"]))
        graceful_stop = max(0.1, float(policy.monitor["graceful_stop_seconds"]))
        pause_bad_samples = max(1, int(policy.monitor["pause_bad_samples"]))
        deadline = time.monotonic() + capsule.resources.wall_minutes * 60.0
        runtime_problem: str | None = None
        resource_deferred = False
        bad_samples = 0
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                capsule.command,
                cwd=_safe_path(self.program.repo_root, capsule.cwd, "capsule cwd"),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=dict(environment),
                start_new_session=True,
                close_fds=True,
            )
            try:
                create_time = psutil.Process(process.pid).create_time()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                create_time = None
            on_start(process.pid, create_time)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    runtime_problem = "capsule exceeded its declared wall boundary"
                    resource_deferred = capsule.task_declaration().pause_safe
                    self._record_resource_stop(
                        capsule,
                        reason=runtime_problem,
                        wall_boundary=True,
                    )
                    returncode = self._stop_exact_owned_process(
                        process,
                        create_time,
                        wall_boundary=True,
                        grace_seconds=graceful_stop,
                    )
                    break
                try:
                    returncode = process.wait(timeout=min(monitor_interval, remaining))
                    break
                except subprocess.TimeoutExpired:
                    try:
                        if create_time is None:
                            raise Generation1Refused(
                                "owned child create time is unavailable; runtime identity is inexact"
                            )
                        sample = self._sample_runtime(capsule, process.pid, create_time)
                    except (Generation1Refused, OSError, ValueError) as exc:
                        rss = (
                            _process_tree_rss_bytes(process.pid, create_time)
                            if create_time is not None
                            else None
                        )
                        sample = {
                            "created_at": _iso(self.now_fn()),
                            "child": {
                                "pid": process.pid,
                                "create_time": create_time,
                                "identity_state": _process_identity_state(process.pid, create_time),
                            },
                            "host": {"missing_required_telemetry": ["runtime-monitor"]},
                            "process_tree_rss_bytes": rss,
                            "declared_process_tree_rss_bytes": int(
                                capsule.resources.estimated_unified_memory_gb * 1_000_000_000
                            ),
                            "allowed": False,
                            "critical": True,
                            "denied_reasons": [
                                f"runtime telemetry failed closed: {type(exc).__name__}: {exc}"
                            ],
                            "disk_forecast": None,
                        }
                    self._record_runtime_sample(capsule, sample, mode="owned")
                    if bool(sample.get("allowed")):
                        bad_samples = 0
                        continue
                    bad_samples += 1
                    if not bool(sample.get("critical")) and bad_samples < pause_bad_samples:
                        continue
                    runtime_problem = (
                        "; ".join(str(value) for value in sample.get("denied_reasons") or [])
                        or "runtime safety gate refused continued execution"
                    )
                    self._record_resource_stop(
                        capsule,
                        reason=runtime_problem,
                        wall_boundary=False,
                    )
                    resource_deferred = capsule.task_declaration().pause_safe
                    returncode = self._stop_exact_owned_process(
                        process,
                        create_time,
                        wall_boundary=False,
                        grace_seconds=graceful_stop,
                    )
                    break
        return CommandResult(
            returncode=returncode,
            child_pid=process.pid,
            child_create_time=create_time,
            started_at=started_at,
            finished_at=_iso(self.now_fn()),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            runtime_problem=runtime_problem,
            resource_deferred=resource_deferred,
        )

    def _artifact_reports(self, capsule: Capsule) -> tuple[list[dict[str, Any]], list[str]]:
        reports: list[dict[str, Any]] = []
        problems: list[str] = []
        for expectation in capsule.artifacts:
            path = _safe_path(self.program.repo_root, expectation.path, "artifact")
            item_problems: list[str] = []
            payload: dict[str, Any] = {}
            try:
                payload = _read_json(path, f"capsule {capsule.capsule_id} artifact")
            except Generation1Refused as exc:
                item_problems.append(str(exc))
            if payload and payload.get("schema") != expectation.schema:
                item_problems.append("artifact schema drifted")
            for dotted, expected in expectation.fields:
                if _dotted(payload, dotted) != expected:
                    item_problems.append(f"{dotted}={_dotted(payload, dotted)!r}, expected {expected!r}")
            if payload and expectation.seal_field is not None:
                try:
                    _validate_seal(payload, expectation.seal_field, "capsule artifact")
                except Generation1Refused as exc:
                    item_problems.append(str(exc))
            report = {
                "path": expectation.path,
                "sha256": sha256_file(path) if path.is_file() else None,
                "schema": payload.get("schema"),
                "problems": item_problems,
                "all_ok": not item_problems,
            }
            reports.append(report)
            problems.extend(f"{expectation.path}: {problem}" for problem in item_problems)
        return reports, problems

    def _validate_completed_artifacts(self, capsules: Mapping[str, Capsule]) -> None:
        for capsule_id, row in self.state["capsules"].items():
            if row.get("status") != "complete":
                continue
            capsule = capsules[capsule_id]
            reports, problems = self._artifact_reports(capsule)
            if problems or reports != row.get("artifacts"):
                raise Generation1Refused(f"completed capsule {capsule_id} artifact drifted")

    def _control_requested(self) -> bool:
        path = self.program.program_root / CONTROL_FILE
        if not path.is_file():
            return False
        control = _read_json(path, "Generation 1 control")
        _validate_seal(control, "control_sha256", "Generation 1 control")
        if control.get("schema") != CONTROL_SCHEMA or control.get("action") != "drain":
            raise Generation1Refused("Generation 1 control is invalid")
        return True

    def _ready_capsules(self, capsules: Mapping[str, Capsule]) -> list[Capsule]:
        ready = [
            capsule
            for capsule in capsules.values()
            if self.state["capsules"][capsule.capsule_id]["status"] == "pending"
            and all(self.state["capsules"][item]["status"] == "complete" for item in capsule.depends_on)
        ]
        return sorted(ready, key=lambda capsule: (capsule.priority, capsule.capsule_id))

    def _retry_or_hold(
        self,
        *,
        capsule: Capsule,
        row: dict[str, Any],
        problem: str,
        hold_problem: str,
        resource_deferred: bool = False,
    ) -> dict[str, Any]:
        """Retry resumable command/artifact failures before failing closed.

        A seed capsule is itself resumable: successful experiment attempts remain immutable and
        the next invocation runs only missing classes.  Bounded supervisor retries therefore
        recover transient worker exits without weakening scientific gates or creating an
        unbounded failure loop.
        """

        row["last_problem"] = problem
        self.state["current_capsule"] = None
        runtime = dict(row.get("runtime") or _empty_runtime_summary())
        if resource_deferred:
            runtime["retry_count"] = int(runtime.get("retry_count") or 0) + 1
            row["runtime"] = runtime
            row["status"] = "pending"
            self.state["status"] = "resource_wait"
            self._append_runtime_event(
                capsule,
                "capsule-resource-deferral-scheduled",
                attempt=int(row["attempts"]),
                problem=problem,
            )
            return self._publish()
        failure_attempts = int(runtime.get("failure_attempt_count") or 0) + 1
        runtime["failure_attempt_count"] = failure_attempts
        if failure_attempts < MAX_CAPSULE_ATTEMPTS:
            runtime["retry_count"] = int(runtime.get("retry_count") or 0) + 1
            row["runtime"] = runtime
            row["status"] = "pending"
            self.state["status"] = "retry_wait"
            self._append_runtime_event(
                capsule,
                "capsule-retry-scheduled",
                attempt=int(row["attempts"]),
                problem=problem,
            )
            return self._publish()
        row["runtime"] = runtime
        row["status"] = "failed"
        self._append_runtime_event(
            capsule,
            "capsule-retries-exhausted",
            attempt=int(row["attempts"]),
            problem=problem,
        )
        self._hold("failure_hold", hold_problem)
        return self._publish()

    def _publish(self) -> dict[str, Any]:
        self.state["updated_at"] = _iso(self.now_fn())
        sealed_state = _sealed(self.state, "state_sha256")
        atomic_write_json(self.program.state_path, sealed_state)
        self.state = sealed_state
        status_core = {
            "schema": STATUS_SCHEMA,
            "program_id": self.program.program_id,
            "created_at": _iso(self.now_fn()),
            "program": self.state["program"],
            "supervisor": self.state["supervisor"],
            "execution_enabled": self.execute,
            "state": self.state["status"],
            "queue_head_sha256": self.state["queue_head_sha256"],
            "next_injection_sequence": self.state["next_injection_sequence"],
            "accepted_injection_count": len(self.state["accepted_injections"]),
            "current_capsule": self.state["current_capsule"],
            "capsules": self.state["capsules"],
            "last_admission": self.state["last_admission"],
            "lane_reservation": self.state.get("lane_reservation"),
            "problems": self.state["problems"],
        }
        status = _sealed(status_core, "status_sha256")
        atomic_write_json(self.program.status_path, status)
        return status

    def _reconcile_running_capsule(
        self,
        capsules: Mapping[str, Capsule],
    ) -> dict[str, Any] | None:
        running = [
            capsule_id for capsule_id, row in self.state["capsules"].items() if row.get("status") == "running"
        ]
        if not running:
            return None
        if len(running) != 1:
            raise Generation1Refused(f"multiple running capsules are recorded: {running}")
        capsule = capsules[running[0]]
        row = self.state["capsules"][capsule.capsule_id]
        self.state["current_capsule"] = capsule.capsule_id
        identity_state = _process_identity_state(
            row.get("child_pid"),
            row.get("child_create_time"),
        )
        if identity_state == "unknown":
            self._hold(
                "integrity_hold",
                (
                    f"capsule {capsule.capsule_id} has an inexact recorded child identity; "
                    "no retry and no process signal are permitted"
                ),
            )
            return self._publish()
        if identity_state == "gone":
            try:
                self._release_lane(capsule)
            except (Generation1Refused, OSError, ValueError) as exc:
                self._hold(
                    "integrity_hold",
                    f"recovery lane release failed closed: {type(exc).__name__}: {exc}",
                )
                return self._publish()
            runtime = dict(row.get("runtime") or {})
            runtime["safety_state"] = "recovery-child-gone"
            row["runtime"] = runtime
            row["finished_at"] = _iso(self.now_fn())
            self._append_runtime_event(
                capsule,
                "recovery-child-gone",
                child_pid=row.get("child_pid"),
                attempt=int(row["attempts"]),
            )
            return self._retry_or_hold(
                capsule=capsule,
                row=row,
                problem="recorded child is gone after supervisor recovery",
                hold_problem=(
                    f"capsule {capsule.capsule_id} exhausted {MAX_CAPSULE_ATTEMPTS} crash-recovery attempts"
                ),
            )

        child_pid = int(row["child_pid"])
        child_create_time = float(row["child_create_time"])
        try:
            self._reserve_lane(capsule, child_pid, child_create_time)
        except (Generation1Refused, OSError, ValueError) as exc:
            self.state["status"] = "recovery_wait"
            row["last_problem"] = (
                f"exact child remains alive; atomic recovery lane unavailable: {type(exc).__name__}: {exc}"
            )
            return self._publish()
        self.state["status"] = "recovery_observe"
        row["last_problem"] = "exact recorded child remains alive; observation only"
        try:
            sample = self._sample_runtime(capsule, child_pid, child_create_time)
        except (Generation1Refused, OSError, ValueError) as exc:
            sample = {
                "created_at": _iso(self.now_fn()),
                "child": {
                    "pid": child_pid,
                    "create_time": child_create_time,
                    "identity_state": _process_identity_state(child_pid, child_create_time),
                },
                "host": {"missing_required_telemetry": ["recovery-runtime-monitor"]},
                "process_tree_rss_bytes": _process_tree_rss_bytes(child_pid, child_create_time),
                "declared_process_tree_rss_bytes": int(
                    capsule.resources.estimated_unified_memory_gb * 1_000_000_000
                ),
                "allowed": False,
                "critical": True,
                "denied_reasons": [f"recovery telemetry failed closed: {type(exc).__name__}: {exc}"],
                "disk_forecast": None,
            }
        if not bool(sample.get("allowed")):
            row["last_problem"] = (
                "recovered exact child is unsafe but observation-only; no process signal sent: "
                + "; ".join(str(value) for value in sample.get("denied_reasons") or [])
            )
        return self._record_runtime_sample(capsule, sample, mode="recovery-observe")

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_program_authority()
            self._validate_accepted_chain()
            capsules = self._all_capsules()
            self._validate_state_capsules(capsules)
            recovery_status = self._reconcile_running_capsule(capsules)
            if recovery_status is not None:
                return recovery_status
            if self._control_requested():
                self.state["status"] = "drained"
                self.state["finished_at"] = _iso(self.now_fn())
                return self._publish()
            self._consume_injections()
            capsules = self._all_capsules()
            self._validate_state_capsules(capsules)
            self._validate_completed_artifacts(capsules)
        except (Generation1Refused, OSError, ValueError) as exc:
            self._hold("integrity_hold", f"{type(exc).__name__}: {exc}")
            return self._publish()
        if all(row["status"] == "complete" for row in self.state["capsules"].values()):
            self.state["status"] = "complete"
            self.state["current_capsule"] = None
            self.state["finished_at"] = _iso(self.now_fn())
            return self._publish()
        ready = self._ready_capsules(capsules)
        if not ready:
            self._hold("integrity_hold", "pending capsules have no satisfiable deterministic route")
            return self._publish()
        capsule = ready[0]
        row = self.state["capsules"][capsule.capsule_id]
        self.state["current_capsule"] = capsule.capsule_id
        try:
            self._validate_capsule_authorities(capsule)
            admission = self.admission_probe(capsule)
        except (Generation1Refused, OSError, ValueError) as exc:
            self._hold("integrity_hold", f"capsule admission failed closed: {type(exc).__name__}: {exc}")
            return self._publish()
        self.state["last_admission"] = dict(admission.receipt)
        if not admission.allowed:
            self.state["status"] = "resource_wait"
            row["last_problem"] = "resource admission refused"
            return self._publish()
        if not self.execute:
            self.state["status"] = "execution_disabled"
            row["last_problem"] = "execution requires explicit --execute"
            return self._publish()
        try:
            self._reserve_lane(capsule)
        except (Generation1Refused, OSError, ValueError) as exc:
            self.state["status"] = "resource_wait"
            row["last_problem"] = (
                f"atomic Generation1 lane reservation unavailable: {type(exc).__name__}: {exc}"
            )
            return self._publish()
        row["status"] = "running"
        row["attempts"] += 1
        row["started_at"] = _iso(self.now_fn())
        row["finished_at"] = None
        row["returncode"] = None
        row["child_pid"] = None
        row["child_create_time"] = None
        row["artifacts"] = []
        row["last_problem"] = None
        runtime = dict(row.get("runtime") or {})
        runtime["safety_state"] = "awaiting-child"
        row["runtime"] = runtime
        self.state["status"] = "running"
        stdout_path = self.program.program_root / "logs" / f"{capsule.capsule_id}.stdout.log"
        stderr_path = self.program.program_root / "logs" / f"{capsule.capsule_id}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        python_paths = [str(self.program.repo_root / "src"), str(self.program.repo_root)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["PYTHONUNBUFFERED"] = "1"
        environment.update(dict(capsule.environment))
        environment["MOP_PROCESS_LABEL"] = f"mop-capsule:{capsule.capsule_id}"
        child_started = False

        def on_start(pid: int, create_time: float | None) -> None:
            nonlocal child_started
            child_started = True
            row["child_pid"] = pid
            row["child_create_time"] = create_time
            reservation = self.state.get("lane_reservation")
            if isinstance(reservation, dict):
                reservation["child_pid"] = pid
                reservation["child_create_time"] = create_time
            self._append_runtime_event(
                capsule,
                "child-started",
                pid=pid,
                create_time=create_time,
                attempt=int(row["attempts"]),
            )
            self._publish()

        # Persist the consumed attempt before Popen. A crash in the launch handshake therefore
        # fails closed instead of making an unrecorded duplicate launch possible.
        self._publish()
        try:
            result = self.runner(capsule, stdout_path, stderr_path, environment, on_start)
        except Exception as exc:  # a command runner failure is operational, not a scientific null
            problem = f"runner failed: {type(exc).__name__}: {exc}"
            identity_state = _process_identity_state(row.get("child_pid"), row.get("child_create_time"))
            if child_started and identity_state == "alive":
                row["last_problem"] = problem
                self.state["status"] = "recovery_observe"
                return self._publish()
            if child_started and identity_state == "unknown":
                row["last_problem"] = problem
                self._hold(
                    "integrity_hold",
                    (
                        f"capsule {capsule.capsule_id} runner failed with an inexact live-child "
                        "boundary; no process signal or retry is permitted"
                    ),
                )
                return self._publish()
            try:
                self._release_lane(capsule)
            except (Generation1Refused, OSError, ValueError) as release_exc:
                self._hold(
                    "integrity_hold",
                    f"runner-failure lane release failed: {type(release_exc).__name__}: {release_exc}",
                )
                return self._publish()
            row["finished_at"] = _iso(self.now_fn())
            return self._retry_or_hold(
                capsule=capsule,
                row=row,
                problem=problem,
                hold_problem=(
                    f"capsule {capsule.capsule_id} runner failed after {MAX_CAPSULE_ATTEMPTS} attempts"
                ),
            )
        row["child_pid"] = result.child_pid
        row["child_create_time"] = result.child_create_time
        row["returncode"] = result.returncode
        row["finished_at"] = result.finished_at
        identity_state = _process_identity_state(result.child_pid, result.child_create_time)
        if identity_state == "alive":
            row["last_problem"] = "runner returned while its exact child remained alive"
            self.state["status"] = "recovery_observe"
            return self._publish()
        if identity_state == "unknown":
            self._hold(
                "integrity_hold",
                (
                    f"capsule {capsule.capsule_id} runner returned across an inexact child "
                    "boundary; no release, signal, or retry is permitted"
                ),
            )
            return self._publish()
        try:
            self._release_lane(capsule)
        except (Generation1Refused, OSError, ValueError) as exc:
            self._hold(
                "integrity_hold",
                f"capsule lane release failed closed: {type(exc).__name__}: {exc}",
            )
            return self._publish()
        if result.returncode != 0:
            return self._retry_or_hold(
                capsule=capsule,
                row=row,
                problem=result.runtime_problem or f"command returned {result.returncode}",
                hold_problem=(
                    f"capsule {capsule.capsule_id} command failed after {MAX_CAPSULE_ATTEMPTS} attempts"
                ),
                resource_deferred=result.resource_deferred,
            )
        reports, problems = self._artifact_reports(capsule)
        row["artifacts"] = reports
        if problems:
            return self._retry_or_hold(
                capsule=capsule,
                row=row,
                problem="; ".join(problems),
                hold_problem=(
                    f"capsule {capsule.capsule_id} artifact validation failed after "
                    f"{MAX_CAPSULE_ATTEMPTS} attempts"
                ),
            )
        row["status"] = "complete"
        row["last_problem"] = None
        self._append_runtime_event(
            capsule,
            "capsule-complete",
            attempt=int(row["attempts"]),
            returncode=result.returncode,
        )
        self.state["current_capsule"] = None
        self.state["status"] = "progressing"
        return self._publish()

    def run(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        with FileLock(self.program.program_root / LOCK_FILE):
            self.state["supervisor"] = self._identity()
            while True:
                status = self.tick()
                cycles += 1
                if status["state"] in TERMINAL_STATES:
                    return status
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                if status["state"] in {"resource_wait", "retry_wait", "recovery_wait"}:
                    self.sleep_fn(self.program.resource_retry_seconds)
                elif status["state"] == "recovery_observe":
                    self.sleep_fn(self.program.admission_interval_seconds)


def read_status(program: Program) -> dict[str, Any]:
    status = _read_json(program.status_path, "Generation 1 status")
    if status.get("schema") != STATUS_SCHEMA or status.get("program_id") != program.program_id:
        raise Generation1Refused("Generation 1 status identity drifted")
    _validate_seal(status, "status_sha256", "Generation 1 status")
    return status


def request_drain(program: Program, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise Generation1Refused("drain reason must be nonempty")
    core = {
        "schema": CONTROL_SCHEMA,
        "program_id": program.program_id,
        "action": "drain",
        "created_at": _iso(_now()),
        "reason": reason,
        "requester_pid": os.getpid(),
    }
    payload = _sealed(core, "control_sha256")
    atomic_write_json(program.program_root / CONTROL_FILE, payload)
    return payload


def submit_injection(program: Program, request_path: Path | str) -> dict[str, Any]:
    source = Path(request_path).resolve()
    # Parse with a temporary supervisor-free validator so submission never mutates program state.
    raw = _read_json(source, "injection request")
    value = _exact_keys(
        raw,
        {
            "schema",
            "program_id",
            "injection_id",
            "sequence",
            "created_at",
            "action",
            "expected_queue_head_sha256",
            "capsules",
            "reason",
            "injection_sha256",
        },
        "injection request",
    )
    if value["schema"] != INJECTION_SCHEMA or value["program_id"] != program.program_id:
        raise Generation1Refused("injection request does not target this program")
    _validate_seal(value, "injection_sha256", "injection request")
    injection_id = _identifier(value["injection_id"], "injection_id")
    target = program.inbox / f"{injection_id}.json"
    if target.exists():
        if _read_json(target, "existing injection") != raw:
            raise Generation1Refused("injection id already exists with different bytes")
    else:
        atomic_write_json(target, raw)
    return {
        "submitted": True,
        "program_id": program.program_id,
        "injection_id": injection_id,
        "path": str(target),
        "file_sha256": sha256_file(target),
        "accepted_at": "next capsule boundary",
    }


def _process_identity_alive(identity: object) -> bool:
    if not isinstance(identity, Mapping):
        return False
    return _process_identity_state(identity.get("pid"), identity.get("create_time")) == "alive"


def start_detached(
    program: Program,
    *,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise Generation1Refused("detached start requires explicit --execute")
    program.program_root.mkdir(parents=True, exist_ok=True)
    with FileLock(program.program_root / START_LOCK_FILE):
        if program.status_path.is_file():
            status = read_status(program)
            if status["state"] in TERMINAL_STATES and status["state"] != "complete":
                return {"already_terminal": True, "status": status}
            if _process_identity_alive(status.get("supervisor")):
                return {"already_running": True, "status": status}
        entrypoint = program.repo_root / "scripts/mop_generation1_campaign.py"
        command = [
            str(program.repo_root / ".venv/bin/python"),
            str(entrypoint),
            "run",
            "--program",
            str(program.path),
            "--execute",
        ]
        caffeinate = shutil.which("caffeinate") if use_caffeinate else None
        launched = [caffeinate, "-ims", *command] if caffeinate else command
        stdout_path = program.program_root / "logs/supervisor.stdout.log"
        stderr_path = program.program_root / "logs/supervisor.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        prior_mtime = program.status_path.stat().st_mtime_ns if program.status_path.exists() else 0
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(program.repo_root / "src"), str(program.repo_root)])
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MOP_PROCESS_LABEL"] = f"mop-supervisor:{program.program_id}"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                launched,
                cwd=program.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + program.startup_ack_seconds
        acknowledged: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if program.status_path.is_file() and program.status_path.stat().st_mtime_ns > prior_mtime:
                try:
                    acknowledged = read_status(program)
                    break
                except (Generation1Refused, OSError, json.JSONDecodeError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise Generation1Refused(
                f"detached supervisor did not acknowledge startup; inspect {stderr_path}"
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
    for name in ("validate", "run", "start", "status", "stop", "inject"):
        command = subparsers.add_parser(name)
        command.add_argument("--program", type=Path, required=True)
        if name in {"run", "start"}:
            command.add_argument("--execute", action="store_true")
        if name == "run":
            command.add_argument("--once", action="store_true")
        if name == "start":
            command.add_argument("--no-caffeinate", action="store_true")
        if name == "stop":
            command.add_argument("--reason", default="operator requested drain")
        if name == "inject":
            command.add_argument("--request", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        program = load_program(arguments.program)
        if arguments.command == "run":
            set_process_label(f"mop-supervisor:{program.program_id}")
        if arguments.command == "validate":
            payload = {
                "valid": True,
                "schema": PROGRAM_SCHEMA,
                "program_id": program.program_id,
                "program_file_sha256": program.file_sha256,
                "program_sha256": program.program_sha256,
                "capsules": [capsule.capsule_id for capsule in program.capsules],
                "execution_authorized": False,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if arguments.command == "run":
            status = Generation1Supervisor(program, execute=arguments.execute).run(
                max_cycles=1 if arguments.once else None
            )
            print(json.dumps(status, indent=2, sort_keys=True))
            return 0 if status["state"] in {"complete", "drained"} else 2
        if arguments.command == "start":
            payload = start_detached(
                program,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        elif arguments.command == "status":
            payload = read_status(program)
        elif arguments.command == "stop":
            payload = request_drain(program, arguments.reason)
        elif arguments.command == "inject":
            payload = submit_injection(program, arguments.request)
        else:  # pragma: no cover
            raise AssertionError(arguments.command)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (Generation1Refused, OSError, ValueError) as exc:
        print(json.dumps({"all_ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "Admission",
    "ArtifactExpectation",
    "CAPSULE_SCHEMA",
    "Capsule",
    "CommandResult",
    "Generation1Refused",
    "Generation1Supervisor",
    "INJECTION_SCHEMA",
    "PROGRAM_SCHEMA",
    "Program",
    "STATUS_SCHEMA",
    "atomic_write_json",
    "build_parser",
    "canonical_sha256",
    "load_program",
    "main",
    "read_status",
    "request_drain",
    "start_detached",
    "submit_injection",
]
