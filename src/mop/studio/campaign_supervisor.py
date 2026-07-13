"""Detached, receipt-governed campaign supervision for the local Mac Studio.

This module is a control plane around :mod:`mop.studio.local_throttle`.  It never
launches experiment commands directly and never signals a discovered governor or
experiment process.  The local throttle remains the sole lane allocator and process
owner; this supervisor supplies durable campaign ordering, observation, retries, and
operator controls across governor legs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ..config import REPO_ROOT
from .local_throttle import (
    IMPLEMENTATION_PATH as THROTTLE_IMPLEMENTATION_PATH,
)
from .local_throttle import (
    PROGRESS_AUTHORITY_SCHEMA,
    TASKPOLICY_COEXISTENCE_PREFIX,
    TaskDeclaration,
    ThrottlePolicy,
    active_lanes,
    collect_host_telemetry,
    dry_run_decision,
    load_policy,
)

PLAN_SCHEMA = "mop-throttle-campaign-plan/v1"
STATE_SCHEMA = "mop-throttle-campaign-state/v1"
STATUS_SCHEMA = "mop-throttle-campaign-status/v1"
CONTROL_SCHEMA = "mop-throttle-campaign-control/v1"
EVENT_SCHEMA = "mop-throttle-campaign-event/v1"
MIGRATION_MARKER_SCHEMA = "mop-campaign-migration-ready/v1"
DRIFT_PROBLEM = "live throttle policy or implementation drifted from the campaign authority"
STATE_FILE = "campaign_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "campaign.lock"
START_LOCK_FILE = "start.lock"
EVENTS_FILE = "events.jsonl"
CONTROL_FILE = "control/stop-request.json"
HOURLY_DIR = "hourly"
TERMINAL_STATES = frozenset(
    {
        "complete",
        "drained",
        "failure_hold",
        "integrity_hold",
        "migration_restart_required",
        "policy_drift_hold",
    }
)
STEP_KINDS = frozenset({"throttle-task", "marker"})
STEP_TERMINAL = frozenset({"complete"})
TASK_OUTPUT_FLAGS = frozenset({"--out", "--output", "--verification-out"})
NATIVE_ARTIFACT_SEAL_FIELDS = {
    "mop-edcm1-receipt/v3": "receipt_sha256",
    "mop-edcm1-verification-artifact/v1": "verification_artifact_sha256",
    "mop-escs-substrate-preflight-report/v1": "report_sha256",
    "mop-escs-x0-receipt/v1": "receipt_sha256",
    "mop-escs-x0-verification/v1": "verification_sha256",
}
RUN_ACTIVE_STATES = frozenset({"launching", "running", "paused"})
RUN_RESUMABLE_STATES = frozenset(
    {"resumable-wall-boundary", "resumable-invocation-boundary"}
)
RUN_ADMISSION_STATES = frozenset({"admission-refused", "atomic-reservation-refused"})
RUN_FAILURE_STATES = frozenset(
    {
        "failed",
        "failed-dynamic-safety-stop",
        "failed-completion-authority",
        "failed-progress-authority",
        "atomic-reservation-refused",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
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


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _safe_repo_path(value: str, label: str) -> Path:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    target = (REPO_ROOT / relative).resolve()
    if not target.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{label} escapes the repository")
    return target


def _declared_task_output(task: TaskDeclaration, label: str) -> str:
    matches = [(index, value) for index, value in enumerate(task.command) if value in TASK_OUTPUT_FLAGS]
    if len(matches) != 1:
        raise ValueError(
            f"{label} command must declare exactly one output authority across {sorted(TASK_OUTPUT_FLAGS)}"
        )
    index, flag = matches[0]
    if index + 1 >= len(task.command):
        raise ValueError(f"{label} command {flag} is missing its output target")
    output = task.command[index + 1]
    if output.startswith("-"):
        raise ValueError(f"{label} command {flag} is missing its output target")
    _safe_repo_path(output, f"{label} command {flag} target")
    return output


def _native_artifact_seal_problems(payload: Mapping[str, Any], schema: str) -> list[str]:
    seal_field = NATIVE_ARTIFACT_SEAL_FIELDS.get(schema)
    if seal_field is None:
        return []
    core = dict(payload)
    declared = core.pop(seal_field, None)
    if (
        not isinstance(declared, str)
        or re.fullmatch(r"[0-9a-f]{64}", declared) is None
        or canonical_sha256(core) != declared
    ):
        return [f"artifact {schema} native {seal_field} self-seal mismatch"]
    return []


def _admission_denied_reasons(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    admission_value = receipt.get("admission")
    admission = admission_value if isinstance(admission_value, Mapping) else {}
    reasons: list[str] = []

    def add_reason(value: object) -> None:
        reason = str(value)
        if reason and reason not in reasons:
            reasons.append(reason)

    declared = admission.get("denied_reasons")
    for value in declared if isinstance(declared, list) else []:
        add_reason(value)
    if not reasons:
        decisions = receipt.get("decisions")
        for decision in decisions if isinstance(decisions, list) else []:
            if not isinstance(decision, Mapping) or decision.get("allowed") is True:
                continue
            denied = decision.get("denied_reasons")
            for value in denied if isinstance(denied, list) else []:
                add_reason(value)
    if not reasons and admission.get("allowed") is not True and admission.get("reason"):
        add_reason(admission["reason"])
    if receipt.get("status") == "atomic-reservation-refused" and receipt.get("reservation_error"):
        add_reason(receipt["reservation_error"])
    return tuple(reasons)


def _campaign_lock_path(plan: CampaignPlan) -> Path:
    return plan.campaign_root / ".campaign_locks" / f"{plan.campaign_id}.{LOCK_FILE}"


def _campaign_start_lock_path(plan: CampaignPlan) -> Path:
    return plan.campaign_root / ".campaign_locks" / f"{plan.campaign_id}.{START_LOCK_FILE}"


def _exact_keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")


def _dotted_value(payload: Any, dotted: str) -> Any:
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _sealed(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(digest_field, None)
    document = dict(core)
    document[digest_field] = canonical_sha256(core)
    return document


def _validate_seal(payload: Mapping[str, Any], digest_field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(digest_field, None)
    if not isinstance(declared, str) or canonical_sha256(core) != declared:
        raise ValueError(f"{label} self-hash mismatch")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically publish one immutable snapshot, returning false if it exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not an object")
    return loaded


@dataclass(frozen=True)
class ArtifactExpectation:
    path: str
    schema: str
    fields: tuple[tuple[str, Any], ...]

    @classmethod
    def from_raw(cls, raw: Any, label: str) -> ArtifactExpectation:
        _exact_keys(raw, {"path", "schema", "fields"}, label)
        fields = raw.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError(f"{label}.fields must be a nonempty object")
        path = str(raw.get("path", ""))
        _safe_repo_path(path, f"{label}.path")
        schema = str(raw.get("schema", ""))
        if not schema:
            raise ValueError(f"{label}.schema must be nonempty")
        return cls(path, schema, tuple((str(key), value) for key, value in fields.items()))


@dataclass(frozen=True)
class CampaignStep:
    step_id: str
    kind: str
    depends_on: tuple[str, ...]
    task_id: str | None
    artifact: ArtifactExpectation
    max_failures: int
    max_no_progress_legs: int


@dataclass(frozen=True)
class CampaignPlan:
    path: Path
    sha256: str
    campaign_id: str
    policy_path: Path
    state_root: Path
    campaign_root: Path
    poll_seconds: float
    hourly_status_seconds: float
    base_backoff_seconds: float
    max_backoff_seconds: float
    admission_samples: int
    admission_interval_seconds: float
    launch_grace_seconds: float
    steps: tuple[CampaignStep, ...]
    policy: ThrottlePolicy

    @property
    def out_dir(self) -> Path:
        return self.campaign_root / self.campaign_id


def load_campaign_plan(path: Path | str) -> CampaignPlan:
    source = Path(path).resolve()
    raw_bytes = source.read_bytes()
    raw = json.loads(raw_bytes)
    _exact_keys(
        raw,
        {
            "schema",
            "campaign_id",
            "policy_path",
            "state_root",
            "campaign_root",
            "poll_seconds",
            "hourly_status_seconds",
            "base_backoff_seconds",
            "max_backoff_seconds",
            "admission_samples",
            "admission_interval_seconds",
            "launch_grace_seconds",
            "steps",
        },
        "campaign plan",
    )
    if raw.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"campaign plan schema must be {PLAN_SCHEMA}")
    campaign_id = str(raw.get("campaign_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", campaign_id):
        raise ValueError("campaign_id must contain only letters, digits, dot, underscore, or hyphen")
    policy_path = _safe_repo_path(str(raw.get("policy_path", "")), "policy_path")
    state_root = _safe_repo_path(str(raw.get("state_root", "")), "state_root")
    campaign_root = _safe_repo_path(str(raw.get("campaign_root", "")), "campaign_root")
    policy = load_policy(policy_path)
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("campaign plan steps must be a nonempty list")
    steps: list[CampaignStep] = []
    seen_ids: set[str] = set()
    seen_tasks: set[str] = set()
    for index, value in enumerate(steps_raw):
        label = f"campaign step {index}"
        _exact_keys(
            value,
            {
                "id",
                "kind",
                "depends_on",
                "task",
                "artifact",
                "max_failures",
                "max_no_progress_legs",
            },
            label,
        )
        step_id = str(value.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", step_id) or step_id in seen_ids:
            raise ValueError(f"{label}.id is invalid or duplicated")
        seen_ids.add(step_id)
        kind = str(value.get("kind", ""))
        if kind not in STEP_KINDS:
            raise ValueError(f"{label}.kind must be one of {sorted(STEP_KINDS)}")
        dependencies = value.get("depends_on")
        if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise ValueError(f"{label}.depends_on must be a list of step ids")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"{label}.depends_on contains duplicates")
        task_value = value.get("task")
        task_id = str(task_value) if task_value is not None else None
        task: TaskDeclaration | None = None
        if kind == "throttle-task":
            if task_id not in policy.tasks:
                raise ValueError(f"{label}.task is not a configured throttle task")
            if task_id in seen_tasks:
                raise ValueError(f"throttle task {task_id!r} is duplicated in the campaign")
            seen_tasks.add(task_id)
            task = policy.task(task_id)
        elif task_id is not None:
            raise ValueError(f"{label}.task must be null for a marker")
        artifact = ArtifactExpectation.from_raw(value.get("artifact"), f"{label}.artifact")
        if task is not None:
            declared_output = _declared_task_output(task, f"{label}.task")
            if artifact.path != declared_output:
                raise ValueError(
                    f"{label}.artifact path must equal task {task.task_id!r} declared output "
                    f"{declared_output!r}"
                )
        max_failures = int(value.get("max_failures", 0))
        max_no_progress = int(value.get("max_no_progress_legs", 0))
        if max_failures < 1 or max_no_progress < 1:
            raise ValueError(f"{label} failure budgets must be positive")
        steps.append(
            CampaignStep(
                step_id,
                kind,
                tuple(dependencies),
                task_id,
                artifact,
                max_failures,
                max_no_progress,
            )
        )
    _validate_step_dag(steps)

    def positive_number(name: str) -> float:
        number = float(raw.get(name, 0))
        if not number > 0:
            raise ValueError(f"{name} must be positive")
        return number

    admission_samples = int(raw.get("admission_samples", 0))
    if admission_samples < 1:
        raise ValueError("admission_samples must be positive")
    base_backoff = positive_number("base_backoff_seconds")
    max_backoff = positive_number("max_backoff_seconds")
    if max_backoff < base_backoff:
        raise ValueError("max_backoff_seconds must be at least base_backoff_seconds")
    return CampaignPlan(
        path=source,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        campaign_id=campaign_id,
        policy_path=policy_path,
        state_root=state_root,
        campaign_root=campaign_root,
        poll_seconds=positive_number("poll_seconds"),
        hourly_status_seconds=positive_number("hourly_status_seconds"),
        base_backoff_seconds=base_backoff,
        max_backoff_seconds=max_backoff,
        admission_samples=admission_samples,
        admission_interval_seconds=positive_number("admission_interval_seconds"),
        launch_grace_seconds=positive_number("launch_grace_seconds"),
        steps=tuple(steps),
        policy=policy,
    )


def _validate_step_dag(steps: Sequence[CampaignStep]) -> None:
    by_id = {step.step_id: step for step in steps}
    for step in steps:
        missing = sorted(set(step.depends_on) - set(by_id))
        if missing:
            raise ValueError(f"step {step.step_id!r} has missing dependencies {missing}")
        if step.step_id in step.depends_on:
            raise ValueError(f"step {step.step_id!r} depends on itself")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError(f"campaign dependency cycle at {step_id!r}")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in by_id[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step in steps:
        visit(step.step_id)


class CampaignLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> CampaignLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"campaign lock is already held: {self.path}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class ObservedRun:
    task_id: str | None
    run_id: str
    scheduler_pid: int
    child_pid: int | None
    status: str
    command: tuple[str, ...]
    receipt_path: str
    scheduler_create_time: float | None = None
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionResult:
    allowed: bool
    reasons: tuple[str, ...] = ()
    receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LaunchResult:
    scheduler_pid: int
    command: tuple[str, ...]
    stdout_path: str
    stderr_path: str


@dataclass(frozen=True)
class CompletionResult:
    complete: bool
    artifact_sha256: str | None = None
    governor_receipt_path: str | None = None
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunOutcome:
    status: str
    final_returncode: int | None
    checkpoint_sha256: str | None
    receipt_path: str
    admission_denied_reasons: tuple[str, ...] = ()


ActiveProbe = Callable[[], list[ObservedRun]]
AdmissionProbe = Callable[[str], AdmissionResult]
Launcher = Callable[[str, str, Path], LaunchResult]
IdentityProbe = Callable[[], tuple[str, str]]
CompletionProbe = Callable[[CampaignStep], CompletionResult]
OutcomeProbe = Callable[[str], RunOutcome | None]
TelemetryProbe = Callable[[], Mapping[str, Any]]


def _registered_child_command_matches(
    declared_command: tuple[str, ...],
    observed_command: tuple[str, ...],
) -> bool:
    """Match a child before or after the exact pinned taskpolicy exec chain."""
    if observed_command == declared_command:
        return True
    prefix_size = len(TASKPOLICY_COEXISTENCE_PREFIX)
    if declared_command[:prefix_size] != TASKPOLICY_COEXISTENCE_PREFIX:
        return False
    post_exec_command = declared_command[prefix_size:]
    return bool(post_exec_command) and observed_command == post_exec_command


def probe_active_runs(policy: ThrottlePolicy, state_root: Path) -> list[ObservedRun]:
    commands: dict[tuple[str, ...], list[str]] = {}
    for task_id, task in policy.tasks.items():
        commands.setdefault(task.command, []).append(task_id)
    observations: list[ObservedRun] = []
    for row in active_lanes(state_root):
        row_status = str(row.get("status", "unknown"))
        command = tuple(str(value) for value in row.get("command", []))
        candidates = commands.get(command, [])
        run_id = str(row.get("run_id", ""))
        receipt_path = state_root / run_id / "run_receipt.json"
        problems: list[str] = []
        receipt: dict[str, Any] = {}
        observed_task_id = candidates[0] if len(candidates) == 1 else None
        if len(candidates) != 1:
            problems.append(f"active command maps to {len(candidates)} policy tasks")
        if receipt_path.is_file():
            try:
                receipt = read_json(receipt_path)
                receipt_task = _dotted_value(receipt, "task.task_id")
                if observed_task_id is not None and receipt_task != observed_task_id:
                    problems.append("active registry/receipt task mismatch")
                if receipt.get("run_id") != run_id:
                    problems.append("active registry/receipt run id mismatch")
                if receipt.get("mode") != "execute":
                    problems.append("active receipt is not an executing governor receipt")
                if receipt.get("policy") != {"path": str(policy.path), "sha256": policy.sha256}:
                    problems.append("active receipt policy binding mismatch")
                expected_implementation = {
                    "path": str(THROTTLE_IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(THROTTLE_IMPLEMENTATION_PATH),
                }
                if receipt.get("implementation") != expected_implementation:
                    problems.append("active receipt governor binding mismatch")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                problems.append(f"active receipt unreadable: {type(exc).__name__}")
        elif row_status != "launching":
            problems.append("active run receipt is missing")
        scheduler_create_time: float | None = None
        scheduler_pid = int(row.get("scheduler_pid") or 0)
        child_pid = int(row["child_pid"]) if row.get("child_pid") else None
        try:
            scheduler = psutil.Process(scheduler_pid)
            scheduler_create_time = scheduler.create_time()
            scheduler_command = scheduler.cmdline()
            if not any(value.endswith("scripts/local_execution_throttle.py") for value in scheduler_command):
                problems.append("scheduler pid does not name the throttle wrapper")
            for option, expected in (("--run-id", run_id), ("--task", observed_task_id)):
                if expected is None:
                    continue
                try:
                    observed = scheduler_command[scheduler_command.index(option) + 1]
                except (ValueError, IndexError):
                    observed = None
                if observed != expected:
                    problems.append(f"scheduler command {option} mismatch")
            started_at = _parse_iso(receipt.get("started_at"))
            if started_at is not None and scheduler_create_time > started_at.timestamp() + 1.0:
                problems.append("scheduler birth time is newer than its run receipt")
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # The registry snapshot raced with a normal scheduler exit. Omitting
            # the stale observation lets receipt reconciliation own the outcome.
            continue
        except (psutil.AccessDenied, OSError):
            problems.append("scheduler process identity is unavailable")
        if child_pid is not None:
            try:
                child = psutil.Process(child_pid)
                if child.ppid() != scheduler_pid:
                    problems.append("registered child is not owned by scheduler pid")
                if not _registered_child_command_matches(command, tuple(child.cmdline())):
                    problems.append("registered child command differs from task declaration")
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # A just-finished child may precede the scheduler's terminal
                # registry update by a few instructions.
                pass
            except (psutil.AccessDenied, OSError):
                problems.append("registered child process identity is unavailable")
        observations.append(
            ObservedRun(
                task_id=observed_task_id,
                run_id=run_id,
                scheduler_pid=scheduler_pid,
                child_pid=child_pid,
                status=row_status,
                command=command,
                receipt_path=str(receipt_path),
                scheduler_create_time=scheduler_create_time,
                problems=tuple(problems),
            )
        )
    return observations


def _artifact_report(expectation: ArtifactExpectation) -> tuple[dict[str, Any], list[str]]:
    path = _safe_repo_path(expectation.path, "artifact path")
    problems: list[str] = []
    payload: dict[str, Any] = {}
    if not path.is_file():
        return payload, ["artifact is missing"]
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"artifact is unreadable: {type(exc).__name__}"]
    if payload.get("schema") != expectation.schema:
        problems.append(f"artifact schema {payload.get('schema')!r} != {expectation.schema!r}")
    problems.extend(_native_artifact_seal_problems(payload, expectation.schema))
    for dotted, expected in expectation.fields:
        observed = _dotted_value(payload, dotted)
        if observed != expected:
            problems.append(f"{dotted}={observed!r}, expected {expected!r}")
    return payload, problems


def _governor_receipt_valid(
    receipt: Mapping[str, Any],
    *,
    task_id: str,
    artifact_path: str,
    artifact_sha256: str,
    policy: ThrottlePolicy,
) -> list[str]:
    problems: list[str] = []
    task = policy.task(task_id)
    expected_task = json.loads(canonical_bytes(asdict(task)))
    expected_command = list(task.command)
    expected_command_sha = hashlib.sha256(
        json.dumps(expected_command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    core = dict(receipt)
    declared = core.pop("payload_sha256", None)
    if not isinstance(declared, str) or canonical_sha256(core) != declared:
        problems.append("governor receipt self-hash mismatch")
    if receipt.get("schema") != "mop-local-throttle-receipt/v1":
        problems.append("governor receipt schema mismatch")
    if receipt.get("status") != "complete" or receipt.get("final_returncode") != 0:
        problems.append("governor run is not complete")
    if _dotted_value(receipt, "admission.allowed") is not True:
        problems.append("governor admission authority is not allowed")
    if _dotted_value(receipt, "task.task_id") != task_id:
        problems.append("governor task id mismatch")
    if receipt.get("task") != expected_task:
        problems.append("governor task declaration mismatch")
    if receipt.get("policy") != {"path": str(policy.path), "sha256": policy.sha256}:
        problems.append("governor policy binding mismatch")
    expected_implementation = {
        "path": str(THROTTLE_IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(THROTTLE_IMPLEMENTATION_PATH),
    }
    if receipt.get("implementation") != expected_implementation:
        problems.append("governor implementation binding mismatch")
    completion = receipt.get("completion_authority")
    if not isinstance(completion, dict):
        problems.append("governor completion authority is missing")
    else:
        if completion.get("schema") != "mop-local-throttle-completion-authority/v1":
            problems.append("completion authority schema mismatch")
        if completion.get("task_id") != task_id:
            problems.append("completion task id mismatch")
        if completion.get("task") != expected_task:
            problems.append("completion task declaration mismatch")
        if completion.get("command") != expected_command:
            problems.append("completion command mismatch")
        if completion.get("command_sha256") != expected_command_sha:
            problems.append("completion command hash mismatch")
        if completion.get("policy") != receipt.get("policy"):
            problems.append("completion policy binding mismatch")
        if completion.get("implementation") != receipt.get("implementation"):
            problems.append("completion implementation binding mismatch")
        if completion.get("returncode") != 0:
            problems.append("completion return code mismatch")
        if completion.get("output") != {"path": artifact_path, "sha256": artifact_sha256}:
            problems.append("completion output binding mismatch")
        if completion.get("final_checkpoint_aggregate_sha256") != _dotted_value(
            receipt, "final_checkpoint.aggregate_sha256"
        ):
            problems.append("completion checkpoint authority mismatch")
        if completion.get("owned_child_active") is not False:
            problems.append("completion still reports an owned child")
    rows = _dotted_value(receipt, "final_checkpoint.files")
    matched = [
        row
        for row in rows or []
        if isinstance(row, dict) and row.get("path") == artifact_path and row.get("sha256") == artifact_sha256
    ]
    if len(matched) != 1:
        problems.append("artifact is absent from final checkpoint authority")
    decisions = receipt.get("decisions")
    for requirement in task.prerequisites:
        prerequisite_path = _safe_repo_path(requirement.path, "task prerequisite path")
        if not prerequisite_path.is_file():
            problems.append(f"current prerequisite is missing: {requirement.path}")
            continue
        expected_sha = sha256_file(prerequisite_path)
        bound_in_every_sample = bool(decisions)
        for decision in decisions or []:
            gates = decision.get("gates") if isinstance(decision, dict) else None
            gate = next(
                (
                    value
                    for value in gates or []
                    if isinstance(value, dict) and value.get("name") == "receipt_prerequisites"
                ),
                None,
            )
            observed = gate.get("observed") if isinstance(gate, dict) else None
            rows = [
                value
                for value in observed or []
                if isinstance(value, dict) and value.get("path") == requirement.path
            ]
            if (
                len(rows) != 1
                or rows[0].get("sha256") != expected_sha
                or rows[0].get("schema") != requirement.schema
                or rows[0].get("all_ok") is not True
            ):
                bound_in_every_sample = False
                break
        if not bound_in_every_sample:
            problems.append(f"governor admission did not bind current prerequisite: {requirement.path}")
    return problems


def probe_completion(step: CampaignStep, policy: ThrottlePolicy, state_root: Path) -> CompletionResult:
    artifact_path = _safe_repo_path(step.artifact.path, "artifact path")
    before_sha = sha256_file(artifact_path) if artifact_path.is_file() else None
    _payload, problems = _artifact_report(step.artifact)
    if problems:
        return CompletionResult(False, problems=tuple(problems))
    try:
        artifact_sha = sha256_file(artifact_path)
    except OSError:
        return CompletionResult(False, problems=("artifact changed during validation",))
    if before_sha != artifact_sha:
        return CompletionResult(False, problems=("artifact changed during validation",))
    if step.kind == "marker":
        try:
            _validate_seal(_payload, "marker_sha256", "migration marker")
        except ValueError as exc:
            return CompletionResult(False, artifact_sha256=artifact_sha, problems=(str(exc),))
        return CompletionResult(True, artifact_sha256=artifact_sha)
    assert step.task_id is not None
    reports: list[tuple[float, Path, list[str]]] = []
    for receipt_path in state_root.glob("*/run_receipt.json"):
        try:
            receipt = read_json(receipt_path)
            if _dotted_value(receipt, "task.task_id") != step.task_id:
                continue
            receipt_problems = _governor_receipt_valid(
                receipt,
                task_id=step.task_id,
                artifact_path=step.artifact.path,
                artifact_sha256=artifact_sha,
                policy=policy,
            )
            reports.append((receipt_path.stat().st_mtime, receipt_path, receipt_problems))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reports.append((0.0, receipt_path, [f"governor receipt unreadable: {type(exc).__name__}"]))
    valid = sorted((row for row in reports if not row[2]), key=lambda row: row[0], reverse=True)
    if not valid:
        details = [problem for _, _, row_problems in reports for problem in row_problems]
        return CompletionResult(
            False,
            artifact_sha256=artifact_sha,
            problems=tuple(details or ["no sealed governor completion receipt"]),
        )
    return CompletionResult(
        True,
        artifact_sha256=artifact_sha,
        governor_receipt_path=str(valid[0][1]),
    )


def probe_run_outcome(run_id: str, state_root: Path) -> RunOutcome | None:
    path = state_root / run_id / "run_receipt.json"
    if not path.is_file():
        return None
    try:
        receipt = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return RunOutcome("receipt-invalid", None, None, str(path))
    status = str(receipt.get("status", "unknown"))
    if status in RUN_RESUMABLE_STATES:
        core = dict(receipt)
        declared = core.pop("payload_sha256", None)
        progress = receipt.get("progress_authority")
        checkpoint = _dotted_value(receipt, "final_checkpoint.aggregate_sha256")
        if (
            not isinstance(declared, str)
            or canonical_sha256(core) != declared
            or not isinstance(progress, Mapping)
            or progress.get("schema") != PROGRESS_AUTHORITY_SCHEMA
            or progress.get("task_id") != _dotted_value(receipt, "task.task_id")
            or progress.get("task") != receipt.get("task")
            or progress.get("command") != _dotted_value(receipt, "task.command")
            or progress.get("policy") != receipt.get("policy")
            or progress.get("implementation") != receipt.get("implementation")
            or progress.get("task_policy_authority") != receipt.get("task_policy_authority")
            or progress.get("returncode") != receipt.get("final_returncode")
            or progress.get("final_checkpoint_aggregate_sha256") != checkpoint
            or progress.get("owned_child_active") is not False
            or progress.get("child_resource") != receipt.get("child_resource")
        ):
            return RunOutcome("receipt-invalid", None, None, str(path))
    return RunOutcome(
        status=status,
        final_returncode=(
            int(receipt["final_returncode"]) if isinstance(receipt.get("final_returncode"), int) else None
        ),
        checkpoint_sha256=(
            str(_dotted_value(receipt, "final_checkpoint.aggregate_sha256"))
            if _dotted_value(receipt, "final_checkpoint.aggregate_sha256") is not None
            else None
        ),
        receipt_path=str(path),
        admission_denied_reasons=_admission_denied_reasons(receipt),
    )


class CampaignSupervisor:
    def __init__(
        self,
        plan: CampaignPlan,
        *,
        out_dir: Path | None = None,
        execute: bool = False,
        active_probe: ActiveProbe | None = None,
        admission_probe: AdmissionProbe | None = None,
        launcher: Launcher | None = None,
        identity_probe: IdentityProbe | None = None,
        completion_probe: CompletionProbe | None = None,
        outcome_probe: OutcomeProbe | None = None,
        telemetry_probe: TelemetryProbe | None = None,
        now_fn: Callable[[], datetime] = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.plan = plan
        self.policy = plan.policy
        self.out_dir = (out_dir or plan.out_dir).resolve()
        self.execute = execute
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.active_probe = active_probe or (lambda: probe_active_runs(self.policy, self.plan.state_root))
        self.admission_probe = admission_probe or self._default_admission_probe
        self.launcher = launcher or self._default_launcher
        self.identity_probe = identity_probe or self._default_identity_probe
        self.completion_probe = completion_probe or (
            lambda step: probe_completion(step, self.policy, self.plan.state_root)
        )
        self.outcome_probe = outcome_probe or (lambda run_id: probe_run_outcome(run_id, self.plan.state_root))
        self.telemetry_probe = telemetry_probe or (
            lambda: collect_host_telemetry(self.policy, disk_root=REPO_ROOT)
        )
        self.state_path = self.out_dir / STATE_FILE
        self.status_path = self.out_dir / STATUS_FILE
        self.events_path = self.out_dir / EVENTS_FILE
        self.control_path = self.out_dir / CONTROL_FILE
        self.lock_path = _campaign_lock_path(plan)
        # This captures the implementation actually imported by this process.  A
        # later on-disk change can be authorized at a transition marker, but cannot
        # safely be treated as loaded until a fresh supervisor process starts.
        self.loaded_throttle_sha256 = sha256_file(THROTTLE_IMPLEMENTATION_PATH)
        self.loaded_supervisor_sha256 = sha256_file(Path(__file__))
        self.pinned_policy_sha256 = self.policy.sha256
        self.pinned_throttle_sha256 = self.loaded_throttle_sha256
        self.state = self._load_or_create_state()
        self._last_active: list[ObservedRun] = []
        self._last_telemetry: Mapping[str, Any] = {}
        self._last_hour_key: str | None = None

    def _load_or_create_state(self) -> dict[str, Any]:
        now = self.now_fn()
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("schema") != STATE_SCHEMA:
                raise ValueError("campaign state schema mismatch")
            _validate_seal(state, "state_sha256", "campaign state")
            if state.get("campaign_id") != self.plan.campaign_id:
                raise ValueError("campaign state id mismatch")
            if state.get("plan_sha256") != self.plan.sha256:
                raise ValueError("campaign plan drift requires a new campaign id")
            state_policy = _dotted_value(state, "policy.sha256")
            state_throttle = _dotted_value(state, "throttle_implementation.sha256")
            if not isinstance(state_policy, str) or not isinstance(state_throttle, str):
                raise ValueError("campaign state authority binding is missing")
            self.pinned_policy_sha256 = state_policy
            self.pinned_throttle_sha256 = state_throttle
            state.setdefault("pending_baseline_transition", None)
            state.setdefault("baseline_transitions", [])
            for row in state.get("steps", {}).values():
                if isinstance(row, dict):
                    row.setdefault("scheduler_pid", None)
                    row.setdefault("scheduler_create_time", None)
                    row.setdefault("child_pid", None)
            state["supervisor"] = self._supervisor_identity()
            state["execution_enabled"] = bool(self.execute)
            state["resumed_at"] = _iso(now)
            return state
        steps = {
            step.step_id: {
                "id": step.step_id,
                "kind": step.kind,
                "task_id": step.task_id,
                "depends_on": list(step.depends_on),
                "status": "pending",
                "run_id": None,
                "adopted": False,
                "scheduler_pid": None,
                "scheduler_create_time": None,
                "child_pid": None,
                "leg": 0,
                "failures": 0,
                "admission_refusals": 0,
                "no_progress_legs": 0,
                "missing_receipt_polls": 0,
                "last_checkpoint_sha256": None,
                "artifact_sha256": None,
                "governor_receipt_path": None,
                "backoff_until": None,
                "last_problem": None,
                "launched_at": None,
                "updated_at": _iso(now),
            }
            for step in self.plan.steps
        }
        return {
            "schema": STATE_SCHEMA,
            "campaign_id": self.plan.campaign_id,
            "plan": {"path": str(self.plan.path), "sha256": self.plan.sha256},
            "plan_sha256": self.plan.sha256,
            "policy": {"path": str(self.policy.path), "sha256": self.pinned_policy_sha256},
            "throttle_implementation": {
                "path": str(THROTTLE_IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
                "sha256": self.pinned_throttle_sha256,
            },
            "supervisor": self._supervisor_identity(),
            "execution_enabled": bool(self.execute),
            "started_at": _iso(now),
            "updated_at": _iso(now),
            "status": "starting",
            "drain_requested": False,
            "current_step": None,
            "steps": steps,
            "launches": [],
            "problems": [],
            "pending_baseline_transition": None,
            "baseline_transitions": [],
        }

    def _supervisor_identity(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        return {
            "pid": os.getpid(),
            "create_time": process.create_time(),
            "implementation_path": str(Path(__file__).relative_to(REPO_ROOT)),
            "implementation_sha256": self.loaded_supervisor_sha256,
            "loaded_throttle_sha256": self.loaded_throttle_sha256,
        }

    def _default_identity_probe(self) -> tuple[str, str]:
        return load_policy(self.plan.policy_path).sha256, sha256_file(THROTTLE_IMPLEMENTATION_PATH)

    def _default_admission_probe(self, task_id: str) -> AdmissionResult:
        receipt = dry_run_decision(
            self.policy.task(task_id),
            self.policy,
            samples=self.plan.admission_samples,
            interval_seconds=self.plan.admission_interval_seconds,
            state_root=self.plan.state_root,
            disk_root=REPO_ROOT,
        )
        admission = receipt.get("admission") or {}
        reasons = _admission_denied_reasons(receipt)
        return AdmissionResult(bool(admission.get("allowed")), reasons, receipt)

    def _default_launcher(self, task_id: str, run_id: str, summary_path: Path) -> LaunchResult:
        stdout_path = self.out_dir / "launches" / f"{run_id}.scheduler.stdout.log"
        stderr_path = self.out_dir / "launches" / f"{run_id}.scheduler.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "scripts/local_execution_throttle.py"),
            "--policy",
            str(self.plan.policy_path),
            "run",
            "--task",
            task_id,
            "--run-id",
            run_id,
            "--execute",
            "--out",
            str(summary_path),
        )
        environment = dict(os.environ)
        python_path = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        environment["PYTHONUNBUFFERED"] = "1"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        return LaunchResult(process.pid, command, str(stdout_path), str(stderr_path))

    def _event(self, name: str, **payload: Any) -> None:
        core = {
            "schema": EVENT_SCHEMA,
            "at": _iso(self.now_fn()),
            "campaign_id": self.plan.campaign_id,
            "event": name,
            **payload,
        }
        event = dict(core)
        event["event_sha256"] = canonical_sha256(core)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("ab") as handle:
            handle.write(canonical_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _check_drain_request(self) -> bool:
        if not self.control_path.is_file():
            return True
        try:
            control = read_json(self.control_path)
            if control.get("schema") != CONTROL_SCHEMA:
                raise ValueError("campaign control schema mismatch")
            _validate_seal(control, "control_sha256", "campaign control")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._hold("integrity_hold", f"campaign control is invalid: {exc}")
            return False
        if control.get("action") == "drain":
            if not self.state.get("drain_requested"):
                self._event("drain-requested", reason=control.get("reason"))
            self.state["drain_requested"] = True
        else:
            self._hold("integrity_hold", "campaign control action is invalid")
            return False
        return True

    def _hold(self, status: str, problem: str) -> None:
        if self.state.get("status") != status or problem not in self.state.get("problems", []):
            self._event(status, problem=problem)
        self.state["status"] = status
        if problem not in self.state["problems"]:
            self.state["problems"].append(problem)

    def _mark_complete(self, step: CampaignStep, result: CompletionResult) -> None:
        row = self.state["steps"][step.step_id]
        if row["status"] == "complete":
            return
        row.update(
            {
                "status": "complete",
                "artifact_sha256": result.artifact_sha256,
                "governor_receipt_path": result.governor_receipt_path,
                "last_problem": None,
                "updated_at": _iso(self.now_fn()),
            }
        )
        self._event(
            "step-complete",
            step_id=step.step_id,
            task_id=step.task_id,
            artifact_sha256=result.artifact_sha256,
        )

    def _dependencies_complete(self, step: CampaignStep) -> bool:
        return all(
            self.state["steps"][dependency]["status"] in STEP_TERMINAL for dependency in step.depends_on
        )

    def _in_backoff(self, row: Mapping[str, Any]) -> bool:
        deadline = _parse_iso(row.get("backoff_until"))
        return deadline is not None and self.now_fn() < deadline

    def _backoff(self, step: CampaignStep, reason: str, *, admission: bool) -> None:
        row = self.state["steps"][step.step_id]
        key = "admission_refusals" if admission else "failures"
        row[key] = int(row.get(key, 0)) + 1
        exponent = max(0, int(row[key]) - 1)
        seconds = min(
            self.plan.max_backoff_seconds,
            self.plan.base_backoff_seconds * (2**exponent),
        )
        deadline = datetime.fromtimestamp(self.now_fn().timestamp() + seconds, tz=UTC)
        row.update(
            {
                "status": "backoff",
                "backoff_until": _iso(deadline),
                "last_problem": reason,
                "updated_at": _iso(self.now_fn()),
            }
        )
        self._event(
            "step-backoff",
            step_id=step.step_id,
            reason=reason,
            seconds=seconds,
            admission=admission,
        )

    def _reconcile_run(self, step: CampaignStep, active: Sequence[ObservedRun]) -> None:
        row = self.state["steps"][step.step_id]
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        matching = [run for run in active if run.run_id == run_id]
        if matching:
            run = matching[0]
            if run.task_id != step.task_id:
                self._hold("integrity_hold", f"active run {run_id} changed task identity")
                return
            expected_pid = row.get("scheduler_pid")
            expected_birth = row.get("scheduler_create_time")
            if expected_pid is not None and expected_pid != run.scheduler_pid:
                self._hold("integrity_hold", f"active run {run_id} changed scheduler pid")
                return
            if expected_birth is not None and expected_birth != run.scheduler_create_time:
                self._hold("integrity_hold", f"active run {run_id} changed scheduler birth identity")
                return
            if run.problems:
                self._hold("integrity_hold", f"active run {run_id}: {'; '.join(run.problems)}")
            row.update(
                {
                    "status": "observing",
                    "scheduler_pid": run.scheduler_pid,
                    "scheduler_create_time": run.scheduler_create_time,
                    "child_pid": run.child_pid,
                    "updated_at": _iso(self.now_fn()),
                }
            )
            return
        launched_at = _parse_iso(row.get("launched_at"))
        outcome = self.outcome_probe(run_id)
        if outcome is None:
            if (
                launched_at is not None
                and (self.now_fn() - launched_at).total_seconds() < self.plan.launch_grace_seconds
            ):
                return
            row["missing_receipt_polls"] = int(row.get("missing_receipt_polls", 0)) + 1
            if row["missing_receipt_polls"] >= 3:
                self._hold("integrity_hold", f"run {run_id} disappeared without a receipt")
            return
        row["missing_receipt_polls"] = 0
        if outcome.status in RUN_ACTIVE_STATES:
            self._hold("integrity_hold", f"run {run_id} receipt is active but scheduler is absent")
            return
        if outcome.status in RUN_RESUMABLE_STATES:
            checkpoint = outcome.checkpoint_sha256
            if checkpoint is None or checkpoint == row.get("last_checkpoint_sha256"):
                row["no_progress_legs"] = int(row.get("no_progress_legs", 0)) + 1
            else:
                row["no_progress_legs"] = 0
                row["last_checkpoint_sha256"] = checkpoint
            if row["no_progress_legs"] >= step.max_no_progress_legs:
                self._hold("failure_hold", f"step {step.step_id} exhausted no-progress leg budget")
                return
            row.update(
                {
                    "status": "pending",
                    "run_id": None,
                    "adopted": False,
                    "scheduler_pid": None,
                    "scheduler_create_time": None,
                    "child_pid": None,
                    "backoff_until": None,
                    "last_problem": outcome.status,
                    "updated_at": _iso(self.now_fn()),
                }
            )
            self._event(
                "resumable-leg",
                step_id=step.step_id,
                run_id=run_id,
                checkpoint_sha256=checkpoint,
            )
            return
        if outcome.status in RUN_ADMISSION_STATES:
            row.update(
                {
                    "run_id": None,
                    "adopted": False,
                    "scheduler_pid": None,
                    "scheduler_create_time": None,
                    "child_pid": None,
                }
            )
            reason = "; ".join(outcome.admission_denied_reasons) or outcome.status
            self._backoff(step, reason, admission=True)
            return
        if outcome.status == "complete":
            self._hold(
                "integrity_hold",
                f"run {run_id} completed without a valid terminal artifact/provenance join",
            )
            return
        if outcome.status == "receipt-invalid":
            self._hold("integrity_hold", f"run {run_id} receipt is invalid")
            return
        if outcome.status in RUN_FAILURE_STATES or outcome.final_returncode not in (None, 0):
            failures = int(row.get("failures", 0)) + 1
            row.update(
                {
                    "run_id": None,
                    "adopted": False,
                    "scheduler_pid": None,
                    "scheduler_create_time": None,
                    "child_pid": None,
                }
            )
            if failures >= step.max_failures:
                row["failures"] = failures
                self._hold("failure_hold", f"step {step.step_id} exhausted failure budget")
                return
            self._backoff(step, f"run failed: {outcome.status}", admission=False)
            return
        self._hold("integrity_hold", f"run {run_id} has unknown terminal status {outcome.status!r}")

    def _adopt(self, step: CampaignStep, run: ObservedRun) -> None:
        row = self.state["steps"][step.step_id]
        if row.get("run_id") == run.run_id and row.get("status") == "observing":
            return
        row.update(
            {
                "status": "observing",
                "run_id": run.run_id,
                "adopted": True,
                "scheduler_pid": run.scheduler_pid,
                "scheduler_create_time": run.scheduler_create_time,
                "child_pid": run.child_pid,
                "launched_at": _iso(self.now_fn()),
                "last_problem": None,
                "updated_at": _iso(self.now_fn()),
            }
        )
        self._event(
            "active-run-adopted",
            step_id=step.step_id,
            task_id=step.task_id,
            run_id=run.run_id,
            scheduler_pid=run.scheduler_pid,
            child_pid=run.child_pid,
        )

    def _launch(self, step: CampaignStep) -> None:
        assert step.task_id is not None
        row = self.state["steps"][step.step_id]
        row["leg"] = int(row.get("leg", 0)) + 1
        stamp = self.now_fn().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{self.plan.campaign_id}-{step.task_id}-{stamp}-leg{row['leg']:02d}"
        summary = self.out_dir / "launches" / f"{run_id}.governor.json"
        result = self.launcher(step.task_id, run_id, summary)
        try:
            scheduler_create_time = psutil.Process(result.scheduler_pid).create_time()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
            scheduler_create_time = None
        row.update(
            {
                "status": "launched",
                "run_id": run_id,
                "adopted": False,
                "scheduler_pid": result.scheduler_pid,
                "scheduler_create_time": scheduler_create_time,
                "child_pid": None,
                "launched_at": _iso(self.now_fn()),
                "backoff_until": None,
                "last_problem": None,
                "updated_at": _iso(self.now_fn()),
            }
        )
        self.state["launches"].append(
            {
                "step_id": step.step_id,
                "task_id": step.task_id,
                "run_id": run_id,
                "scheduler_pid": result.scheduler_pid,
                "command": list(result.command),
                "stdout": result.stdout_path,
                "stderr": result.stderr_path,
                "at": _iso(self.now_fn()),
            }
        )
        self._event(
            "governor-launched",
            step_id=step.step_id,
            task_id=step.task_id,
            run_id=run_id,
            scheduler_pid=result.scheduler_pid,
        )

    def _transition_completion(
        self,
        step: CampaignStep,
        *,
        current_policy: str,
        current_throttle: str,
        active: Sequence[ObservedRun],
    ) -> CompletionResult:
        marker_path = _safe_repo_path(step.artifact.path, "migration marker path")
        before_sha = sha256_file(marker_path) if marker_path.is_file() else None
        payload, problems = _artifact_report(step.artifact)
        if problems:
            return CompletionResult(False, problems=tuple(problems))
        try:
            _exact_keys(
                payload,
                {
                    "schema",
                    "campaign_id",
                    "plan_sha256",
                    "ready",
                    "expected_old",
                    "expected_new",
                    "created_at",
                    "reason",
                    "marker_sha256",
                },
                "migration marker",
            )
            _validate_seal(payload, "marker_sha256", "migration marker")
            for label in ("expected_old", "expected_new"):
                _exact_keys(
                    payload.get(label),
                    {"policy_sha256", "governor_sha256"},
                    f"migration marker {label}",
                )
        except ValueError as exc:
            return CompletionResult(False, problems=(str(exc),))
        old = payload["expected_old"]
        new = payload["expected_new"]
        checks = {
            "campaign id": (payload.get("campaign_id"), self.plan.campaign_id),
            "plan hash": (payload.get("plan_sha256"), self.plan.sha256),
            "old policy hash": (old.get("policy_sha256"), self.pinned_policy_sha256),
            "old governor hash": (old.get("governor_sha256"), self.pinned_throttle_sha256),
            "new policy hash": (new.get("policy_sha256"), current_policy),
            "new governor hash": (new.get("governor_sha256"), current_throttle),
        }
        mismatches = [
            f"{label} mismatch" for label, (observed, expected) in checks.items() if observed != expected
        ]
        if payload.get("ready") is not True:
            mismatches.append("ready is not true")
        for authority in (old, new):
            for name in ("policy_sha256", "governor_sha256"):
                value = authority.get(name)
                if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                    mismatches.append(f"{name} is not a lowercase sha256")
        if active:
            mismatches.append("active governor lanes prevent baseline transition")
        try:
            artifact_sha = sha256_file(marker_path)
        except OSError:
            return CompletionResult(False, problems=("migration marker changed during validation",))
        if before_sha != artifact_sha:
            return CompletionResult(False, problems=("migration marker changed during validation",))
        if mismatches:
            return CompletionResult(
                False,
                artifact_sha256=artifact_sha,
                problems=tuple(mismatches),
            )

        marker_sha = str(payload["marker_sha256"])
        if current_throttle != self.loaded_throttle_sha256:
            pending = {
                "marker_path": step.artifact.path,
                "marker_sha256": marker_sha,
                "expected_old": dict(old),
                "expected_new": dict(new),
                "authorized_at": _iso(self.now_fn()),
            }
            if self.state.get("pending_baseline_transition") != pending:
                self.state["pending_baseline_transition"] = pending
                self._event(
                    "baseline-transition-restart-required",
                    step_id=step.step_id,
                    marker_sha256=marker_sha,
                )
            self.state["status"] = "migration_restart_required"
            return CompletionResult(
                False,
                artifact_sha256=artifact_sha,
                problems=("restart required to load the authorized governor implementation",),
            )

        # A policy-only migration can be reloaded in-process.  A governor-code
        # migration reaches here only in the fresh process whose imported hash is
        # exactly the marker's expected-new hash.
        replacement_policy = load_policy(self.plan.policy_path)
        if replacement_policy.sha256 != current_policy:
            return CompletionResult(False, problems=("policy changed while adopting baseline",))
        transition = {
            "step_id": step.step_id,
            "marker_path": step.artifact.path,
            "marker_sha256": marker_sha,
            "old": dict(old),
            "new": dict(new),
            "adopted_at": _iso(self.now_fn()),
        }
        recorded = self.state.setdefault("baseline_transitions", [])
        if not any(row.get("marker_sha256") == marker_sha for row in recorded):
            recorded.append(transition)
            self._event(
                "baseline-transition-adopted",
                step_id=step.step_id,
                marker_sha256=marker_sha,
                old=dict(old),
                new=dict(new),
            )
        self.policy = replacement_policy
        self.pinned_policy_sha256 = current_policy
        self.pinned_throttle_sha256 = current_throttle
        self.state["policy"] = {
            "path": str(replacement_policy.path),
            "sha256": current_policy,
        }
        self.state["throttle_implementation"] = {
            "path": str(THROTTLE_IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
            "sha256": current_throttle,
        }
        self.state["pending_baseline_transition"] = None
        self.state["problems"] = [
            problem for problem in self.state.get("problems", []) if problem != DRIFT_PROBLEM
        ]
        return CompletionResult(True, artifact_sha256=artifact_sha)

    def tick(self) -> dict[str, Any]:
        if not self._check_drain_request():
            return self._publish()
        try:
            plan_sha = sha256_file(self.plan.path)
            supervisor_sha = sha256_file(Path(__file__))
        except OSError as exc:
            self._hold("integrity_hold", f"campaign authority is unreadable: {type(exc).__name__}")
            return self._publish()
        if plan_sha != self.plan.sha256 or supervisor_sha != self.loaded_supervisor_sha256:
            self._hold("integrity_hold", "campaign plan or supervisor implementation drifted")
            return self._publish()
        try:
            current_policy, current_throttle = self.identity_probe()
            active = self.active_probe()
        except Exception as exc:  # external observation failure must fail closed
            self._hold("integrity_hold", f"campaign observation failed: {type(exc).__name__}: {exc}")
            return self._publish()
        self._last_active = active
        drifted = (
            current_policy != self.pinned_policy_sha256 or current_throttle != self.pinned_throttle_sha256
        )
        if drifted:
            for step in self.plan.steps:
                row = self.state["steps"][step.step_id]
                if (
                    step.kind == "marker"
                    and row["status"] != "complete"
                    and self._dependencies_complete(step)
                ):
                    transition = self._transition_completion(
                        step,
                        current_policy=current_policy,
                        current_throttle=current_throttle,
                        active=active,
                    )
                    if transition.complete:
                        self._mark_complete(step, transition)
                    if self.state.get("status") == "migration_restart_required":
                        return self._publish()
                    drifted = (
                        current_policy != self.pinned_policy_sha256
                        or current_throttle != self.pinned_throttle_sha256
                    )
                    if not drifted:
                        break
            if drifted:
                self._hold("policy_drift_hold", DRIFT_PROBLEM)
                return self._publish()
        for run in active:
            if run.problems:
                self._hold("integrity_hold", f"active run {run.run_id}: {'; '.join(run.problems)}")
                return self._publish()

        for step in self.plan.steps:
            row = self.state["steps"][step.step_id]
            if row["status"] == "complete" or not self._dependencies_complete(step):
                continue
            if step.task_id is not None and any(run.task_id == step.task_id for run in active):
                continue
            completion = (
                self._transition_completion(
                    step,
                    current_policy=current_policy,
                    current_throttle=current_throttle,
                    active=active,
                )
                if step.kind == "marker"
                else self.completion_probe(step)
            )
            if completion.complete:
                self._mark_complete(step, completion)
            if self.state.get("status") == "migration_restart_required":
                return self._publish()

        for step in self.plan.steps:
            row = self.state["steps"][step.step_id]
            if step.kind == "throttle-task" and row["status"] in {
                "observing",
                "launched",
            }:
                self._reconcile_run(step, active)
                if self.state["status"] in TERMINAL_STATES:
                    return self._publish()

        if all(self.state["steps"][step.step_id]["status"] == "complete" for step in self.plan.steps):
            self.state["status"] = "complete"
            self.state["current_step"] = None
            return self._publish()

        if self.state.get("drain_requested"):
            self.state["status"] = "draining" if active else "drained"
            return self._publish()

        ready = [
            step
            for step in self.plan.steps
            if self.state["steps"][step.step_id]["status"] != "complete" and self._dependencies_complete(step)
        ]
        if not ready:
            self.state["status"] = "waiting_dependencies"
            self.state["current_step"] = None
            return self._publish()

        for step in ready:
            row = self.state["steps"][step.step_id]
            self.state["current_step"] = step.step_id
            if step.kind == "marker":
                row["status"] = "waiting_marker"
                row["last_problem"] = f"waiting for {step.artifact.path}"
                row["updated_at"] = _iso(self.now_fn())
                self.state["status"] = "waiting_marker"
                return self._publish()
            matching = [run for run in active if run.task_id == step.task_id]
            if len(matching) > 1:
                self._hold(
                    "integrity_hold",
                    f"multiple active governors match campaign step {step.step_id}",
                )
                return self._publish()
            if matching:
                self._adopt(step, matching[0])
                self.state["status"] = "observing_existing"
                return self._publish()
            if row["status"] in {"observing", "launched"}:
                self.state["status"] = "observing"
                return self._publish()
            if self._in_backoff(row):
                self.state["status"] = "backoff"
                return self._publish()
            if not self.execute:
                row["status"] = "execution_disabled"
                self.state["status"] = "execution_disabled"
                return self._publish()
            admission = self.admission_probe(str(step.task_id))
            if not admission.allowed:
                self._backoff(
                    step,
                    "; ".join(admission.reasons) or "governor admission refused",
                    admission=True,
                )
                self.state["status"] = "backoff"
                return self._publish()
            self._launch(step)
            self.state["status"] = "running"
            return self._publish()
        return self._publish()

    def _status_payload(self) -> dict[str, Any]:
        now = self.now_fn()
        active = [asdict(run) for run in self._last_active]
        current_step_id = self.state.get("current_step")
        current_step = self.state["steps"].get(current_step_id) if isinstance(current_step_id, str) else None
        payload = {
            "schema": STATUS_SCHEMA,
            "campaign_id": self.plan.campaign_id,
            "created_at": _iso(now),
            "plan": {"path": str(self.plan.path), "sha256": self.plan.sha256},
            "policy": {"path": str(self.policy.path), "sha256": self.pinned_policy_sha256},
            "throttle_implementation_sha256": self.pinned_throttle_sha256,
            "supervisor": self.state["supervisor"],
            "state": self.state.get("status"),
            "execution_enabled": self.execute,
            "drain_requested": bool(self.state.get("drain_requested")),
            "current_step": current_step,
            "steps": self.state["steps"],
            "active_lanes": active,
            "telemetry": dict(self._last_telemetry),
            "problems": list(self.state.get("problems", [])),
        }
        return _sealed(payload, "status_sha256")

    def _publish(self) -> dict[str, Any]:
        now = self.now_fn()
        hour_key = now.strftime("%Y%m%dT%H0000Z")
        if not self._last_telemetry or hour_key != self._last_hour_key:
            try:
                self._last_telemetry = self.telemetry_probe()
            except Exception as exc:  # status telemetry must not kill an owned run
                self._last_telemetry = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._last_hour_key = hour_key
        self.state["updated_at"] = _iso(now)
        sealed_state = _sealed(self.state, "state_sha256")
        atomic_write_json(self.state_path, sealed_state)
        self.state = sealed_state
        status = self._status_payload()
        atomic_write_json(self.status_path, status)
        hourly = self.out_dir / HOURLY_DIR / f"{hour_key}.json"
        write_immutable_json(hourly, status)
        return status

    def run(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        with CampaignLock(self.lock_path):
            self._event("supervisor-start", execute=self.execute)
            self.state["status"] = "supervisor_started"
            self._publish()
            while True:
                status = self.tick()
                cycles += 1
                if str(status.get("state")) in TERMINAL_STATES:
                    self._event("supervisor-stop", state=status.get("state"))
                    return status
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                self.sleep_fn(self.plan.poll_seconds)


def request_drain(out_dir: Path, reason: str) -> dict[str, Any]:
    core = {
        "schema": CONTROL_SCHEMA,
        "action": "drain",
        "requested_at": _iso(_now()),
        "reason": reason,
        "requester_pid": os.getpid(),
    }
    control = _sealed(core, "control_sha256")
    atomic_write_json(out_dir / CONTROL_FILE, control)
    return control


def create_transition_marker(
    plan: CampaignPlan,
    *,
    out_dir: Path,
    reason: str,
    expected_new_policy_sha256: str | None = None,
    expected_new_governor_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the one immutable receipt allowed to authorize baseline drift."""

    state_path = out_dir / STATE_FILE
    if not state_path.is_file():
        raise FileNotFoundError(f"campaign state does not exist: {state_path}")
    state = read_json(state_path)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("campaign state schema mismatch")
    _validate_seal(state, "state_sha256", "campaign state")
    if state.get("campaign_id") != plan.campaign_id or state.get("plan_sha256") != plan.sha256:
        raise ValueError("campaign state is not bound to this plan")
    markers = [step for step in plan.steps if step.kind == "marker"]
    if len(markers) != 1:
        raise ValueError("mark-ready requires exactly one marker step")
    marker = markers[0]
    if not all(state["steps"][dependency]["status"] == "complete" for dependency in marker.depends_on):
        raise RuntimeError("marker dependencies are not complete")
    if active_lanes(plan.state_root):
        raise RuntimeError("cannot authorize a baseline transition while a governor lane is active")
    old_policy = _dotted_value(state, "policy.sha256")
    old_governor = _dotted_value(state, "throttle_implementation.sha256")
    if not isinstance(old_policy, str) or not isinstance(old_governor, str):
        raise ValueError("campaign state baseline authority is missing")
    current_policy = expected_new_policy_sha256 or load_policy(plan.policy_path).sha256
    current_governor = expected_new_governor_sha256 or sha256_file(THROTTLE_IMPLEMENTATION_PATH)
    for label, value in (
        ("expected new policy", current_policy),
        ("expected new governor", current_governor),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} must be a lowercase sha256")
    core = {
        "schema": MIGRATION_MARKER_SCHEMA,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.sha256,
        "ready": True,
        "expected_old": {
            "policy_sha256": old_policy,
            "governor_sha256": old_governor,
        },
        "expected_new": {
            "policy_sha256": current_policy,
            "governor_sha256": current_governor,
        },
        "created_at": _iso(_now()),
        "reason": reason,
    }
    marker_payload = _sealed(core, "marker_sha256")
    marker_path = _safe_repo_path(marker.artifact.path, "migration marker path")
    if not write_immutable_json(marker_path, marker_payload):
        raise FileExistsError(f"migration marker is immutable and already exists: {marker_path}")
    return marker_payload


def read_campaign_status(out_dir: Path) -> dict[str, Any]:
    path = out_dir / STATUS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"campaign status does not exist: {path}")
    status = read_json(path)
    if status.get("schema") != STATUS_SCHEMA:
        raise ValueError("campaign status schema mismatch")
    _validate_seal(status, "status_sha256", "campaign status")
    return status


def _campaign_out_dir(plan: CampaignPlan, override: Path | None) -> Path:
    return (override or plan.out_dir).resolve()


def start_detached(
    plan: CampaignPlan,
    *,
    out_dir: Path,
    execute: bool,
    use_caffeinate: bool = True,
    acknowledgement_seconds: float = 15.0,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start_lock = _campaign_start_lock_path(plan)
    with CampaignLock(start_lock):
        try:
            with CampaignLock(_campaign_lock_path(plan)):
                pass
        except RuntimeError as exc:
            raise RuntimeError("campaign supervisor is already running") from exc
        stdout_path = out_dir / "logs/supervisor.stdout.log"
        stderr_path = out_dir / "logs/supervisor.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        entrypoint = REPO_ROOT / "scripts/mop_campaign.py"
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(entrypoint),
            "run",
            "--plan",
            str(plan.path),
            "--out-dir",
            str(out_dir),
        ]
        if execute:
            command.append("--execute")
        caffeinate = shutil.which("caffeinate") if use_caffeinate else None
        launched_command = [caffeinate, "-ims", *command] if caffeinate else command
        environment = dict(os.environ)
        values = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        if environment.get("PYTHONPATH"):
            values.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(values)
        environment["PYTHONUNBUFFERED"] = "1"
        before_mtime = (out_dir / STATUS_FILE).stat().st_mtime_ns if (out_dir / STATUS_FILE).exists() else 0
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                launched_command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + acknowledgement_seconds
        acknowledged: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            status_path = out_dir / STATUS_FILE
            if status_path.is_file() and status_path.stat().st_mtime_ns > before_mtime:
                try:
                    acknowledged = read_campaign_status(out_dir)
                    break
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise RuntimeError(f"detached supervisor did not acknowledge startup; inspect {stderr_path}")
        return {
            "launched_pid": process.pid,
            "caffeinate": bool(caffeinate),
            "command": launched_command,
            "status": acknowledged,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a campaign plan without executing")
    validate.add_argument("--plan", type=Path, required=True)
    run = subparsers.add_parser("run", help="run the foreground supervisor")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out-dir", type=Path)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--once", action="store_true", help="publish one observation and exit")
    start = subparsers.add_parser("start", help="start a detached supervisor")
    start.add_argument("--plan", type=Path, required=True)
    start.add_argument("--out-dir", type=Path)
    start.add_argument("--execute", action="store_true")
    start.add_argument("--no-caffeinate", action="store_true")
    status = subparsers.add_parser("status", help="read the latest sealed status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--out-dir", type=Path)
    stop = subparsers.add_parser("stop", help="request a non-signaling campaign drain")
    stop.add_argument("--plan", type=Path, required=True)
    stop.add_argument("--out-dir", type=Path)
    stop.add_argument("--reason", default="operator requested drain")
    marker = subparsers.add_parser(
        "mark-ready",
        help="seal the controlled old-to-new baseline transition marker",
    )
    marker.add_argument("--plan", type=Path, required=True)
    marker.add_argument("--out-dir", type=Path)
    marker.add_argument("--reason", required=True)
    marker.add_argument("--expected-new-policy-sha256")
    marker.add_argument("--expected-new-governor-sha256")
    wait = subparsers.add_parser("wait", help="wait for a terminal campaign state")
    wait.add_argument("--plan", type=Path, required=True)
    wait.add_argument("--out-dir", type=Path)
    wait.add_argument("--timeout-seconds", type=float, default=0.0)
    wait.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    plan = load_campaign_plan(arguments.plan)
    if arguments.command == "validate":
        payload = {
            "valid": True,
            "schema": PLAN_SCHEMA,
            "campaign_id": plan.campaign_id,
            "plan_sha256": plan.sha256,
            "policy_sha256": plan.policy.sha256,
            "steps": [step.step_id for step in plan.steps],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    out_dir = _campaign_out_dir(plan, arguments.out_dir)
    if arguments.command == "run":
        supervisor = CampaignSupervisor(plan, out_dir=out_dir, execute=arguments.execute)
        status = supervisor.run(max_cycles=1 if arguments.once else None)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status.get("state") in {"complete", "drained"} else 2
    if arguments.command == "start":
        result = start_detached(
            plan,
            out_dir=out_dir,
            execute=arguments.execute,
            use_caffeinate=not arguments.no_caffeinate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "status":
        print(json.dumps(read_campaign_status(out_dir), indent=2, sort_keys=True))
        return 0
    if arguments.command == "stop":
        print(json.dumps(request_drain(out_dir, arguments.reason), indent=2, sort_keys=True))
        return 0
    if arguments.command == "mark-ready":
        marker = create_transition_marker(
            plan,
            out_dir=out_dir,
            reason=arguments.reason,
            expected_new_policy_sha256=arguments.expected_new_policy_sha256,
            expected_new_governor_sha256=arguments.expected_new_governor_sha256,
        )
        print(json.dumps(marker, indent=2, sort_keys=True))
        return 0
    if arguments.command == "wait":
        started = time.monotonic()
        while True:
            status = read_campaign_status(out_dir)
            if status.get("state") in TERMINAL_STATES:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 0 if status.get("state") in {"complete", "drained"} else 2
            if arguments.timeout_seconds > 0 and time.monotonic() - started >= arguments.timeout_seconds:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 2
            time.sleep(max(0.1, arguments.poll_seconds))
    raise AssertionError(arguments.command)


__all__ = [
    "AdmissionResult",
    "ArtifactExpectation",
    "CampaignLock",
    "CampaignPlan",
    "CampaignStep",
    "CampaignSupervisor",
    "CompletionResult",
    "LaunchResult",
    "ObservedRun",
    "RunOutcome",
    "atomic_write_json",
    "build_parser",
    "canonical_sha256",
    "create_transition_marker",
    "load_campaign_plan",
    "main",
    "probe_active_runs",
    "probe_completion",
    "read_campaign_status",
    "request_drain",
    "start_detached",
]
