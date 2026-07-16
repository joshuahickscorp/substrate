"""Durable adopting parent for the Generation 1 successor evidence chain.

The parent adopts three already-running legacy queues by exact process
identity, restarts only a demonstrably absent queue, and never signals an
adopted process.  After the consolidated result is valid it starts and
monitors the bounded successor horizon through the generic Generation 1
supervisor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
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
from mop.studies import generation1_c3_d1_frozen_queue as d1_queue
from mop.studies import generation1_consolidated_final_campaign as final_campaign
from mop.studies import generation1_successor_mechanics_queue as mechanics_queue
from mop.studio.generation1_supervisor import (
    FileLock,
    Generation1Refused,
    atomic_write_json,
    canonical_sha256,
    load_program,
    sha256_file,
    write_immutable_json,
)
from mop.studio.generation1_supervisor import (
    read_status as read_generation1_status,
)
from mop.studio.generation1_supervisor import (
    start_detached as start_generation1_detached,
)

CHAIN_ID = "generation1-successor-evidence-chain-v3"
PARENT_LABEL = "mop-successor-evidence-chain"
STATE_SCHEMA = "mop-generation1-successor-evidence-chain-state/v3"
STATUS_SCHEMA = "mop-generation1-successor-evidence-chain-status/v3"
ADOPTION_SCHEMA = "mop-generation1-legacy-adoption-receipt/v1"
CONTROL_SCHEMA = "mop-generation1-successor-evidence-chain-control/v1"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / CHAIN_ID
DEFAULT_HORIZON_PROGRAM = REPO_ROOT / "configs/campaign/generation1_successor_horizon_v1.json"
STATE_FILE = "chain_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "control/chain.lock"
START_LOCK_FILE = "control/start.lock"
CONTROL_FILE = "control/request.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
POLL_SECONDS = 30.0
STARTUP_ACK_SECONDS = 30.0
LAUNCH_GRACE_SECONDS = 60.0
PROCESS_TIME_TOLERANCE_SECONDS = 0.02

TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
LEGACY_FAILURE_STATES = frozenset({"failure_hold", "failed", "stopped", "integrity_hold"})
HORIZON_TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})


class SuccessorChainRefused(RuntimeError):
    """The adopting parent could not establish an exact safe boundary."""


ResultValidator = Callable[[Mapping[str, Any]], None]
ProcessTable = Callable[[], Sequence["ProcessSnapshot"]]
IdentityProbe = Callable[[Mapping[str, Any]], str]
PopenFactory = Callable[..., Any]
SupervisorStarter = Callable[..., Mapping[str, Any]]
NowFn = Callable[[], dt.datetime]


@dataclass(frozen=True, slots=True)
class LegacySpec:
    stage_id: str
    program_id: str
    process_label: str
    child_label_prefixes: tuple[str, ...]
    status_path: Path
    status_schema: str
    result_path: Path
    result_schema: str
    restart_command: tuple[str, ...]
    result_validator: ResultValidator


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    pid: int
    create_time: float
    pgid: int
    cwd: str
    label: str
    command: tuple[str, ...]
    ppid: int | None = None

    def identity(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "create_time": self.create_time,
            "pgid": self.pgid,
            "cwd": self.cwd,
            "label": self.label,
        }


def legacy_specs(repo_root: Path = REPO_ROOT) -> tuple[LegacySpec, ...]:
    root = repo_root.resolve()
    python = str(root / ".venv/bin/python")
    return (
        LegacySpec(
            stage_id="legacy_d1",
            program_id=d1_queue.PROGRAM_ID,
            process_label="mop-d1-frozen-queue",
            child_label_prefixes=("mop-d1-prod-", "mop-d1-chal-"),
            status_path=root / "runs/generation1" / d1_queue.PROGRAM_ID / "current_status.json",
            status_schema=d1_queue.STATUS_SCHEMA,
            result_path=root / "proof/GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.json",
            result_schema=d1_queue.RESULT_SCHEMA,
            restart_command=(python, "scripts/generation1_c3/d1_frozen_queue.py"),
            result_validator=d1_queue.validate_aggregate,
        ),
        LegacySpec(
            stage_id="legacy_successor_mechanics",
            program_id=mechanics_queue.PROGRAM_ID,
            process_label="mop-successor-mechanics-queue",
            child_label_prefixes=("mop-g1-",),
            status_path=(root / "runs/generation1" / mechanics_queue.PROGRAM_ID / "current_status.json"),
            status_schema=mechanics_queue.STATUS_SCHEMA,
            result_path=root / "proof/GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json",
            result_schema=mechanics_queue.RESULT_SCHEMA,
            restart_command=(python, "scripts/generation1_c3/successor_mechanics_queue.py"),
            result_validator=mechanics_queue.validate_aggregate,
        ),
        LegacySpec(
            stage_id="legacy_final",
            program_id=final_campaign.PROGRAM_ID,
            process_label="mop-final-campaign",
            child_label_prefixes=("mop-final-",),
            status_path=root / "runs/generation1" / final_campaign.PROGRAM_ID / "current_status.json",
            status_schema=final_campaign.STATUS_SCHEMA,
            result_path=root / "proof/GENERATION1_CONSOLIDATED_FINAL_RESULT.json",
            result_schema=final_campaign.RESULT_SCHEMA,
            restart_command=(python, "scripts/generation1_c3/consolidated_final_campaign.py"),
            result_validator=final_campaign.validate_result,
        ),
    )


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
        raise SuccessorChainRefused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or not 0 < stat.st_size <= MAX_JSON_BYTES:
        raise SuccessorChainRefused(f"{label} is not a safe bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise SuccessorChainRefused(f"{label} changed while being read: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuccessorChainRefused(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SuccessorChainRefused(f"{label} must be a JSON object: {path}")
    return value


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise SuccessorChainRefused(f"{label} self-seal is invalid")


def _default_process_table() -> tuple[ProcessSnapshot, ...]:
    snapshots: list[ProcessSnapshot] = []
    for process in psutil.process_iter():
        try:
            command = tuple(process.cmdline())
            if not command:
                continue
            pid = int(process.pid)
            snapshots.append(
                ProcessSnapshot(
                    pid=pid,
                    create_time=float(process.create_time()),
                    pgid=int(os.getpgid(pid)),
                    cwd=str(Path(process.cwd()).resolve()),
                    label=str(command[0]),
                    command=command,
                    ppid=int(process.ppid()),
                )
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, OSError, RuntimeError, ValueError):
            # An inaccessible unrelated process cannot safely be classified as
            # one of our labelled queues. A previously adopted exact PID is
            # probed separately and fails closed if it becomes inaccessible.
            continue
    return tuple(snapshots)


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


def _self_identity(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    implementation = Path(__file__).resolve()
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


def _empty_capsule() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "returncode": None,
        "started_at": None,
        "finished_at": None,
        "artifacts": [],
        "last_problem": None,
        "process": None,
        "adoption_receipts": [],
        "launch_requested_at": None,
        "launched_pid": None,
    }


class SuccessorEvidenceChain:
    """One durable parent for legacy adoption and the bounded horizon."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        horizon_program_path: Path = DEFAULT_HORIZON_PROGRAM,
        specs: Sequence[LegacySpec] | None = None,
        execute: bool = False,
        process_table_fn: ProcessTable = _default_process_table,
        identity_probe_fn: IdentityProbe = _default_identity_probe,
        popen_fn: PopenFactory = subprocess.Popen,
        supervisor_start_fn: SupervisorStarter = start_generation1_detached,
        now_fn: NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.horizon_program_path = horizon_program_path.resolve()
        self.specs = tuple(specs) if specs is not None else legacy_specs(self.repo_root)
        self.execute = execute
        self.process_table_fn = process_table_fn
        self.identity_probe_fn = identity_probe_fn
        self.popen_fn = popen_fn
        self.supervisor_start_fn = supervisor_start_fn
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.poll_seconds = poll_seconds
        if not self.root.is_relative_to(self.repo_root):
            raise SuccessorChainRefused("chain root must remain inside the repository")
        if not self.horizon_program_path.is_relative_to(self.repo_root):
            raise SuccessorChainRefused("horizon program must remain inside the repository")
        if not self.specs or len({spec.stage_id for spec in self.specs}) != len(self.specs):
            raise SuccessorChainRefused("legacy stage ids must be nonempty and unique")
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
        implementation = Path(__file__).resolve()
        return {
            "path": (
                str(implementation.relative_to(self.repo_root))
                if implementation.is_relative_to(self.repo_root)
                else str(implementation)
            ),
            "sha256": sha256_file(implementation),
        }

    def _horizon_authority(self) -> dict[str, Any]:
        payload = _read_json(self.horizon_program_path, "successor horizon program manifest")
        _validate_seal(payload, "program_sha256", "successor horizon program manifest")
        if payload.get("schema") != "mop-generation1-program/v1" or payload.get("program_id") != (
            "generation1-successor-horizon-v1"
        ):
            raise SuccessorChainRefused("successor horizon program manifest identity drifted")
        return {
            "path": str(self.horizon_program_path.relative_to(self.repo_root)),
            "file_sha256": sha256_file(self.horizon_program_path),
            "program_sha256": payload["program_sha256"],
            "program_id": payload["program_id"],
        }

    def _validate_bound_authorities(self, state: Mapping[str, Any]) -> None:
        if state.get("parent_implementation") != self._implementation_authority():
            raise SuccessorChainRefused("successor chain parent implementation authority drifted")
        if state.get("horizon_program") != self._horizon_authority():
            raise SuccessorChainRefused("successor horizon program authority drifted")

    def _initial_state(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        capsules = {spec.stage_id: _empty_capsule() for spec in self.specs}
        capsules["successor_horizon"] = _empty_capsule()
        return {
            "schema": STATE_SCHEMA,
            "program_id": CHAIN_ID,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "execution_enabled": self.execute,
            "status": "starting",
            "supervisor": self._identity(),
            "parent_implementation": self._implementation_authority(),
            "horizon_program": self._horizon_authority(),
            "capsules": capsules,
            "problems": [],
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        state = _read_json(self.state_path, "successor chain state")
        _validate_seal(state, "state_sha256", "successor chain state")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != CHAIN_ID:
            raise SuccessorChainRefused("successor chain state identity drifted")
        expected_capsules = {spec.stage_id for spec in self.specs} | {"successor_horizon"}
        capsules = state.get("capsules")
        if not isinstance(capsules, dict) or set(capsules) != expected_capsules:
            raise SuccessorChainRefused("successor chain capsule inventory drifted")
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

    def _publish(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        self.state["updated_at"] = now
        core = dict(self.state)
        core.pop("state_sha256", None)
        sealed_state = {**core, "state_sha256": canonical_sha256(core)}
        atomic_write_json(self.state_path, sealed_state)
        self.state = dict(sealed_state)
        capsules = self.state["capsules"]
        complete = sum(row.get("status") == "complete" for row in capsules.values())
        status_core = {
            "schema": STATUS_SCHEMA,
            "program_id": CHAIN_ID,
            "created_at": self.state["created_at"],
            "updated_at": now,
            "supervisor": self.state["supervisor"],
            "execution_enabled": self.state["execution_enabled"],
            "state": self.state["status"],
            "horizon_program": self.state["horizon_program"],
            "capsules": capsules,
            "counts": {
                "complete": complete,
                "total": len(capsules),
                "remaining": len(capsules) - complete,
            },
            "finished_at": self.state["finished_at"],
            "problems": self.state["problems"],
            "activation_allowed": False,
            "scientific_promotion": False,
        }
        status = {**status_core, "status_sha256": canonical_sha256(status_core)}
        atomic_write_json(self.status_path, status)
        return status

    def _validated_status(self, spec: LegacySpec) -> dict[str, Any] | None:
        if not spec.status_path.exists():
            return None
        value = _read_json(spec.status_path, f"{spec.stage_id} status")
        _validate_seal(value, "status_sha256", f"{spec.stage_id} status")
        if (
            value.get("schema") != spec.status_schema
            or value.get("program_id") != spec.program_id
            or value.get("activation_allowed") is not False
            or value.get("scientific_promotion") is not False
        ):
            raise SuccessorChainRefused(f"{spec.stage_id} status identity or safety drifted")
        return value

    def _validated_result(self, spec: LegacySpec) -> dict[str, Any] | None:
        if not spec.result_path.exists():
            return None
        value = _read_json(spec.result_path, f"{spec.stage_id} result")
        if (
            value.get("schema") != spec.result_schema
            or value.get("program_id") != spec.program_id
            or value.get("complete") is not True
            or value.get("problems") != []
            or value.get("activation_allowed") is not False
            or value.get("scientific_promotion") is not False
        ):
            raise SuccessorChainRefused(f"{spec.stage_id} result identity or safety drifted")
        try:
            spec.result_validator(value)
        except (Generation1Refused, OSError, ValueError) as exc:
            raise SuccessorChainRefused(
                f"{spec.stage_id} terminal result is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        return value

    def _artifact(self, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path.resolve().relative_to(self.repo_root)),
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "all_ok": True,
            "problems": [],
        }

    def _command_residual(self, spec: LegacySpec, process: ProcessSnapshot) -> bool:
        expected_script = (self.repo_root / spec.restart_command[1]).resolve()
        for argument in process.command[1:]:
            if not argument or argument.startswith("-"):
                continue
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = Path(process.cwd) / candidate
            try:
                if candidate.resolve() == expected_script:
                    return True
            except OSError:
                continue
        return False

    def _resource_tracker(self, process: ProcessSnapshot) -> bool:
        if len(process.command) != 3 or process.command[1] != "-c":
            return False
        expected_python = (self.repo_root / ".venv/bin/python").resolve()
        try:
            interpreter_matches = Path(process.command[0]).resolve() == expected_python
        except OSError:
            return False
        return (
            interpreter_matches
            and re.fullmatch(
                r"from multiprocessing\.resource_tracker import main;main\([0-9]+\)",
                process.command[2],
            )
            is not None
        )

    def _spawn_worker(self, process: ProcessSnapshot) -> bool:
        if (
            len(process.command) != 4
            or process.command[1] != "-c"
            or process.command[3] != "--multiprocessing-fork"
        ):
            return False
        expected_python = (self.repo_root / ".venv/bin/python").resolve()
        try:
            interpreter_matches = Path(process.command[0]).resolve() == expected_python
        except OSError:
            return False
        return (
            interpreter_matches
            and re.fullmatch(
                (
                    r"from multiprocessing\.spawn import spawn_main; ?"
                    r"spawn_main\(tracker_fd=[0-9]+, pipe_handle=[0-9]+\)"
                ),
                process.command[2],
            )
            is not None
        )

    def _matching_processes(
        self,
        spec: LegacySpec,
        processes: Sequence[ProcessSnapshot],
    ) -> tuple[list[ProcessSnapshot], list[ProcessSnapshot]]:
        labelled = [process for process in processes if process.label == spec.process_label]
        for process in labelled:
            if process.cwd != str(self.repo_root) or process.pgid != process.pid:
                raise SuccessorChainRefused(
                    f"{spec.stage_id} exact label has invalid cwd or is not its process-group leader"
                )
            if not math.isfinite(process.create_time) or process.create_time <= 0:
                raise SuccessorChainRefused(f"{spec.stage_id} exact process create time is invalid")
        if len(labelled) > 1:
            raise SuccessorChainRefused(
                f"{spec.stage_id} has multiple exact parent processes: "
                + ", ".join(str(process.pid) for process in labelled)
            )
        worker_residuals = [
            process
            for process in processes
            if process not in labelled
            and any(process.label.startswith(prefix) for prefix in spec.child_label_prefixes)
        ]
        command_residuals = [
            process
            for process in processes
            if process not in labelled
            and process not in worker_residuals
            and self._command_residual(spec, process)
        ]
        trackers = [process for process in processes if self._resource_tracker(process)]
        spawn_workers = [
            process
            for process in processes
            if process not in labelled
            and process not in worker_residuals
            and process not in command_residuals
            and self._spawn_worker(process)
        ]
        residuals = [*worker_residuals, *command_residuals]
        if any(process.cwd != str(self.repo_root) for process in residuals):
            raise SuccessorChainRefused(f"{spec.stage_id} residual worker or restart command has invalid cwd")
        if labelled:
            parent = labelled[0]
            misgrouped_spawn_workers = [
                process
                for process in spawn_workers
                if process.ppid == parent.pid and process.pgid != parent.pid
            ]
            if misgrouped_spawn_workers:
                raise SuccessorChainRefused(
                    f"{spec.stage_id} direct spawn child escaped the exact parent process group: "
                    + ", ".join(str(process.pid) for process in misgrouped_spawn_workers)
                )
            owned_spawn_workers = [
                process for process in spawn_workers if process.pgid == parent.pid
            ]
            group_members = [
                process for process in processes if process.pid != parent.pid and process.pgid == parent.pid
            ]
            allowed_group_members = [
                process
                for process in group_members
                if process in worker_residuals
                or process in trackers
                or (process in owned_spawn_workers and process.ppid == parent.pid)
            ]
            unexpected = [process for process in group_members if process not in allowed_group_members]
            if unexpected:
                raise SuccessorChainRefused(
                    f"{spec.stage_id} process group has unexpected members: "
                    + ", ".join(str(process.pid) for process in unexpected)
                )
            if any(process.cwd != str(self.repo_root) for process in allowed_group_members):
                raise SuccessorChainRefused(f"{spec.stage_id} allowed process-group member has invalid cwd")
            foreign = [process for process in residuals if process.pgid != parent.pid]
            if foreign:
                raise SuccessorChainRefused(
                    f"{spec.stage_id} has residual workers outside the exact parent process group"
                )
        else:
            exact_owner_pgids = {
                process.pid
                for process in processes
                if process.label in {candidate.process_label for candidate in self.specs}
                and process.cwd == str(self.repo_root)
                and process.pgid == process.pid
                and math.isfinite(process.create_time)
                and process.create_time > 0
            }
            residuals.extend(
                process
                for process in spawn_workers
                if process.pgid not in exact_owner_pgids
            )
            residuals.extend(
                process
                for process in trackers
                if process.cwd == str(self.repo_root) and process.pgid not in exact_owner_pgids
            )
            if any(process.cwd != str(self.repo_root) for process in residuals):
                raise SuccessorChainRefused(
                    f"{spec.stage_id} residual worker or restart command has invalid cwd"
                )
        return labelled, residuals

    def _write_adoption_receipt(
        self,
        spec: LegacySpec,
        process: ProcessSnapshot,
        status: Mapping[str, Any] | None,
        row: dict[str, Any],
    ) -> None:
        identity = process.identity()
        identity_sha = canonical_sha256(identity)[:16]
        path = self.root / "adoptions" / spec.stage_id / f"{process.pid}-{identity_sha}.json"
        if path.exists():
            receipt = _read_json(path, f"{spec.stage_id} adoption receipt")
            _validate_seal(receipt, "receipt_sha256", f"{spec.stage_id} adoption receipt")
            if (
                receipt.get("schema") != ADOPTION_SCHEMA
                or receipt.get("chain_id") != CHAIN_ID
                or receipt.get("stage_id") != spec.stage_id
                or receipt.get("program_id") != spec.program_id
                or receipt.get("process") != identity
                or _parse_time(receipt.get("adopted_at")) is None
                or receipt.get("policy")
                != {
                    "observe_only": True,
                    "signals_allowed": False,
                    "restart_only_after_exact_absence": True,
                }
            ):
                raise SuccessorChainRefused(f"{spec.stage_id} adoption receipt identity drifted")
        else:
            observed_status = None
            if status is not None:
                observed_status = {
                    "path": str(spec.status_path.relative_to(self.repo_root)),
                    "file_sha256": sha256_file(spec.status_path),
                    "status_sha256": status.get("status_sha256"),
                    "state": status.get("state"),
                }
            core = {
                "schema": ADOPTION_SCHEMA,
                "chain_id": CHAIN_ID,
                "stage_id": spec.stage_id,
                "program_id": spec.program_id,
                "adopted_at": _iso(self.now_fn()),
                "process": identity,
                "observed_status": observed_status,
                "policy": {
                    "observe_only": True,
                    "signals_allowed": False,
                    "restart_only_after_exact_absence": True,
                },
            }
            receipt = {**core, "receipt_sha256": canonical_sha256(core)}
            write_immutable_json(path, receipt)
        receipts = list(row.get("adoption_receipts") or [])
        relative = str(path.relative_to(self.repo_root))
        if relative not in receipts:
            receipts.append(relative)
        row["adoption_receipts"] = receipts
        row["process"] = identity
        if row.get("started_at") is None:
            row["started_at"] = receipt["adopted_at"]

    def _mark_complete(
        self,
        row: dict[str, Any],
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        row["status"] = "complete"
        row["returncode"] = 0
        row["finished_at"] = _iso(self.now_fn())
        row["artifacts"] = [self._artifact(path, payload)]
        row["last_problem"] = None

    def _launch_legacy(self, spec: LegacySpec, row: dict[str, Any]) -> None:
        if not self.execute:
            row["status"] = "pending"
            row["last_problem"] = "legacy restart requires explicit --execute"
            return
        requested_at = _iso(self.now_fn())
        row["status"] = "launching"
        row["attempts"] = int(row.get("attempts") or 0) + 1
        row["launch_requested_at"] = requested_at
        row["last_problem"] = "restart intent persisted before process creation"
        self._publish()

        stdout_path = self.root / "logs" / f"{spec.stage_id}.stdout.log"
        stderr_path = self.root / "logs" / f"{spec.stage_id}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(self.repo_root / "src"), str(self.repo_root)])
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MOP_PROCESS_LABEL"] = spec.process_label
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = self.popen_fn(
                list(spec.restart_command),
                cwd=self.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
        row["launched_pid"] = int(process.pid)
        row["last_problem"] = "restart launched; awaiting exact process adoption"
        self._publish()

    def _launch_intent_in_grace(self, row: Mapping[str, Any]) -> bool:
        requested = _parse_time(row.get("launch_requested_at"))
        if requested is None:
            return False
        return (self.now_fn().astimezone(dt.UTC) - requested).total_seconds() < LAUNCH_GRACE_SECONDS

    def _reconcile_legacy(
        self,
        spec: LegacySpec,
        row: dict[str, Any],
        processes: Sequence[ProcessSnapshot],
    ) -> None:
        labelled, residuals = self._matching_processes(spec, processes)
        result = self._validated_result(spec)
        status = self._validated_status(spec) if result is None else None

        if labelled:
            process = labelled[0]
            current = row.get("process")
            identity = process.identity()
            if isinstance(current, Mapping) and any(
                current.get(key) != value for key, value in identity.items()
            ):
                prior_state = self.identity_probe_fn(current)
                if prior_state != "gone":
                    raise SuccessorChainRefused(
                        f"{spec.stage_id} prior adopted identity is {prior_state} while a different "
                        "exact parent is visible"
                    )
                row["process"] = None
            self._write_adoption_receipt(spec, process, status, row)
            if result is not None:
                self._mark_complete(row, spec.result_path, result)
                return
            if status is not None and status.get("state") in LEGACY_FAILURE_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"legacy program entered {status.get('state')}"
                return
            row["status"] = "adopted"
            row["returncode"] = None
            row["finished_at"] = None
            row["last_problem"] = "exact process adopted for observation only"
            return

        if result is not None:
            self._mark_complete(row, spec.result_path, result)
            return

        current = row.get("process")
        if isinstance(current, Mapping):
            identity_state = self.identity_probe_fn(current)
            if identity_state == "unknown":
                raise SuccessorChainRefused(f"{spec.stage_id} previously adopted identity is now inexact")
            if identity_state == "alive":
                raise SuccessorChainRefused(
                    f"{spec.stage_id} adopted PID is alive but exact process metadata is unavailable"
                )
            row["process"] = None

        status = status if status is not None else self._validated_status(spec)
        if status is not None and status.get("state") in LEGACY_FAILURE_STATES:
            row["status"] = "failure_hold"
            row["finished_at"] = _iso(self.now_fn())
            row["last_problem"] = f"legacy program entered {status.get('state')}"
            return
        if residuals:
            row["status"] = "adoption_wait"
            row["last_problem"] = (
                "parent absent while residual worker or restart command remains: "
                + ", ".join(str(process.pid) for process in residuals)
            )
            return
        if row.get("status") in {"launching", "adoption_wait"} and self._launch_intent_in_grace(row):
            row["status"] = "adoption_wait"
            row["last_problem"] = "sealed restart intent remains inside startup grace"
            return
        self._launch_legacy(spec, row)

    def _horizon_process_present(self, processes: Sequence[ProcessSnapshot]) -> bool:
        label = "mop-supervisor:generation1-successor-horizon-v1"
        matches = [process for process in processes if process.label == label]
        if any(process.cwd != str(self.repo_root) for process in matches):
            raise SuccessorChainRefused("successor horizon supervisor has invalid cwd")
        if len(matches) > 1:
            raise SuccessorChainRefused("multiple successor horizon supervisors are visible")
        return bool(matches)

    def _horizon_status_artifacts(self, program: Any, status: Mapping[str, Any]) -> list[dict[str, Any]]:
        artifacts = [self._artifact(program.status_path, status)]
        capsules = status.get("capsules")
        if isinstance(capsules, Mapping):
            for capsule in capsules.values():
                if not isinstance(capsule, Mapping):
                    continue
                for artifact in capsule.get("artifacts", []):
                    if isinstance(artifact, Mapping) and artifact.get("all_ok") is True:
                        artifacts.append(dict(artifact))
        return artifacts

    def _reconcile_horizon(
        self,
        row: dict[str, Any],
        processes: Sequence[ProcessSnapshot],
    ) -> None:
        try:
            program = load_program(self.horizon_program_path, repo_root=self.repo_root)
        except (Generation1Refused, OSError, ValueError) as exc:
            raise SuccessorChainRefused(
                f"successor horizon program is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        if program.program_id != "generation1-successor-horizon-v1":
            raise SuccessorChainRefused("successor horizon program id drifted")
        bound_program = self.state.get("horizon_program")
        if not isinstance(bound_program, Mapping) or program.program_sha256 != bound_program.get(
            "program_sha256"
        ):
            raise SuccessorChainRefused("loaded successor horizon program authority drifted")

        status = None
        if program.status_path.is_file():
            try:
                status = read_generation1_status(program)
            except (Generation1Refused, OSError, ValueError) as exc:
                raise SuccessorChainRefused(
                    f"successor horizon status is invalid: {type(exc).__name__}: {exc}"
                ) from exc
        if status is not None:
            state = str(status.get("state"))
            if state == "complete":
                row["status"] = "complete"
                row["returncode"] = 0
                row["finished_at"] = _iso(self.now_fn())
                row["artifacts"] = self._horizon_status_artifacts(program, status)
                row["last_problem"] = None
                return
            if state in HORIZON_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"successor horizon entered {state}"
                return
            supervisor = status.get("supervisor")
            if isinstance(supervisor, Mapping):
                identity_state = self.identity_probe_fn(supervisor)
                if identity_state == "unknown":
                    raise SuccessorChainRefused("successor horizon supervisor identity is inexact")
                if identity_state == "alive":
                    row["status"] = "running"
                    row["process"] = dict(supervisor)
                    row["last_problem"] = "generic Generation 1 supervisor is running"
                    return

        current = row.get("process")
        if isinstance(current, Mapping):
            identity_state = self.identity_probe_fn(current)
            if identity_state == "unknown":
                raise SuccessorChainRefused("successor horizon supervisor identity is inexact")
            if identity_state == "alive":
                row["status"] = "adoption_wait"
                row["last_problem"] = "exact horizon supervisor is alive; awaiting status"
                return
            row["process"] = None

        if self._horizon_process_present(processes):
            row["status"] = "adoption_wait"
            row["last_problem"] = "horizon process is visible while status acknowledgement is pending"
            return
        if row.get("status") in {"launching", "adoption_wait"} and self._launch_intent_in_grace(row):
            row["status"] = "adoption_wait"
            row["last_problem"] = "sealed horizon start intent remains inside startup grace"
            return
        if not self.execute:
            row["status"] = "pending"
            row["last_problem"] = "horizon start requires explicit --execute"
            return

        row["status"] = "launching"
        row["attempts"] = int(row.get("attempts") or 0) + 1
        row["launch_requested_at"] = _iso(self.now_fn())
        row["last_problem"] = "horizon start intent persisted before supervisor request"
        self._publish()
        try:
            launched = self.supervisor_start_fn(program, execute=True, use_caffeinate=True)
        except (Generation1Refused, OSError, RuntimeError, ValueError) as exc:
            row["status"] = "adoption_wait"
            row["last_problem"] = f"horizon startup acknowledgement pending: {type(exc).__name__}: {exc}"
            return
        launched_pid = launched.get("launched_pid")
        row["launched_pid"] = launched_pid if isinstance(launched_pid, int) else None
        returned_status = launched.get("status")
        row["status"] = "running"
        row["last_problem"] = "generic Generation 1 supervisor start accepted"
        if isinstance(returned_status, Mapping) and returned_status.get("supervisor") is not None:
            row["process"] = dict(returned_status["supervisor"])

    def _control_requested(self) -> bool:
        path = self.root / CONTROL_FILE
        if not path.exists():
            return False
        value = _read_json(path, "successor chain control")
        _validate_seal(value, "control_sha256", "successor chain control")
        if (
            value.get("schema") != CONTROL_SCHEMA
            or value.get("program_id") != CHAIN_ID
            or value.get("action") != "drain"
        ):
            raise SuccessorChainRefused("successor chain control identity drifted")
        return True

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_bound_authorities(self.state)
            if self.state.get("status") in TERMINAL_STATES:
                return self._publish()
            if self._control_requested():
                self.state["status"] = "drained"
                self.state["finished_at"] = _iso(self.now_fn())
                return self._publish()
            processes = tuple(self.process_table_fn())
            for spec in self.specs:
                row = self.state["capsules"][spec.stage_id]
                self._reconcile_legacy(spec, row, processes)

            legacy_rows = [self.state["capsules"][spec.stage_id] for spec in self.specs]
            if any(row.get("status") == "failure_hold" for row in legacy_rows):
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem("a legacy prerequisite entered a terminal failure state")
                return self._publish()
            if any(row.get("status") == "adoption_wait" for row in legacy_rows):
                self.state["status"] = "adoption_wait"
                return self._publish()
            if not all(row.get("status") == "complete" for row in legacy_rows):
                self.state["status"] = "waiting_legacy"
                return self._publish()

            horizon = self.state["capsules"]["successor_horizon"]
            self._reconcile_horizon(horizon, processes)
            if horizon.get("status") == "complete":
                self.state["status"] = "complete"
                self.state["finished_at"] = _iso(self.now_fn())
            elif horizon.get("status") == "failure_hold":
                self.state["status"] = "failure_hold"
                self.state["finished_at"] = _iso(self.now_fn())
                self._append_problem(str(horizon.get("last_problem")))
            elif horizon.get("status") == "adoption_wait":
                self.state["status"] = "adoption_wait"
            else:
                self.state["status"] = "waiting_horizon"
            return self._publish()
        except (SuccessorChainRefused, Generation1Refused, OSError, RuntimeError, ValueError) as exc:
            self.state["status"] = "integrity_hold"
            self.state["finished_at"] = _iso(self.now_fn())
            self._append_problem(f"{type(exc).__name__}: {exc}")
            return self._publish()

    def run(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        cycles = 0
        with FileLock(self.root / LOCK_FILE):
            self.state["supervisor"] = self._identity()
            while True:
                status = self.tick()
                cycles += 1
                if status["state"] in TERMINAL_STATES:
                    return status
                if max_cycles is not None and cycles >= max_cycles:
                    return status
                self.sleep_fn(self.poll_seconds)


def read_chain_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = root.resolve() / STATUS_FILE
    status = _read_json(path, "successor chain status")
    _validate_seal(status, "status_sha256", "successor chain status")
    if status.get("schema") != STATUS_SCHEMA or status.get("program_id") != CHAIN_ID:
        raise SuccessorChainRefused("successor chain status identity drifted")
    return status


def request_stop(root: Path = DEFAULT_ROOT, reason: str = "operator requested drain") -> dict[str, Any]:
    if not reason.strip():
        raise SuccessorChainRefused("drain reason must be nonempty")
    root = root.resolve()
    core = {
        "schema": CONTROL_SCHEMA,
        "program_id": CHAIN_ID,
        "action": "drain",
        "created_at": _iso(_now()),
        "requester_pid": os.getpid(),
        "reason": reason,
        "adopted_process_action": "none",
    }
    payload = {**core, "control_sha256": canonical_sha256(core)}
    atomic_write_json(root / CONTROL_FILE, payload)
    return payload


def _process_identity_alive(identity: object) -> bool:
    return isinstance(identity, Mapping) and _default_identity_probe(identity) == "alive"


def _runs_parent_command(process: ProcessSnapshot, entrypoint: Path) -> bool:
    for index, argument in enumerate(process.command[:-1]):
        if not argument or argument.startswith("-"):
            continue
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = Path(process.cwd) / candidate
        try:
            matches = candidate.resolve() == entrypoint
        except OSError:
            matches = False
        if matches and process.command[index + 1] == "run":
            return True
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
    for process in candidates:
        if (
            process.cwd != str(REPO_ROOT.resolve())
            or process.pgid <= 0
            or not math.isfinite(process.create_time)
            or process.create_time <= 0
        ):
            raise SuccessorChainRefused("visible successor chain parent has inexact process metadata")
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise SuccessorChainRefused("multiple successor chain parent processes are visible")
    if labelled:
        return labelled[0]
    return next((process for process in candidates if process.pid == process.pgid), candidates[0])


def start_chain_detached(
    *,
    root: Path = DEFAULT_ROOT,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise SuccessorChainRefused("detached successor chain start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_generation1_successor_chain.py").resolve()
        status_path = root / STATUS_FILE
        status = None
        if status_path.is_file():
            status = read_chain_status(root)
            if status.get("state") in TERMINAL_STATES:
                return {"already_terminal": True, "status": status}
        visible = _visible_parent_process(entrypoint)
        status_identity = status.get("supervisor") if status is not None else None
        if _process_identity_alive(status_identity):
            if visible is not None and (
                not isinstance(status_identity, Mapping)
                or status_identity.get("pid") != visible.pid
                or not math.isclose(
                    float(status_identity.get("create_time", -1)),
                    visible.create_time,
                    rel_tol=0.0,
                    abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
                )
            ):
                raise SuccessorChainRefused(
                    "sealed successor chain identity conflicts with the visible parent"
                )
            return {"already_running": True, "status": status}
        if visible is not None:
            return {
                "already_running": True,
                "status": status,
                "observed_process": visible.identity(),
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
                    acknowledged = read_chain_status(root)
                    break
                except (SuccessorChainRefused, OSError, ValueError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise SuccessorChainRefused(
                f"detached successor chain did not acknowledge startup; inspect {stderr_path}"
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
    for name in ("run", "start", "status", "stop"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        if name in {"run", "start"}:
            child.add_argument("--execute", action="store_true")
        if name == "run":
            child.add_argument("--once", action="store_true")
        if name == "start":
            child.add_argument("--no-caffeinate", action="store_true")
        if name == "stop":
            child.add_argument("--reason", default="operator requested drain")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "run":
            if not arguments.execute:
                raise SuccessorChainRefused("successor chain execution requires explicit --execute")
            set_process_label(PARENT_LABEL)
            payload = SuccessorEvidenceChain(root=arguments.root, execute=True).run(
                max_cycles=1 if arguments.once else None
            )
        elif arguments.command == "start":
            payload = start_chain_detached(
                root=arguments.root,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        elif arguments.command == "status":
            payload = read_chain_status(arguments.root)
        else:
            payload = request_stop(arguments.root, arguments.reason)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if arguments.command == "run" and payload.get("state") not in {"complete", "drained"}:
            return 2
        return 0
    except (SuccessorChainRefused, Generation1Refused, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"all_ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "ADOPTION_SCHEMA",
    "CHAIN_ID",
    "CONTROL_SCHEMA",
    "LegacySpec",
    "PARENT_LABEL",
    "ProcessSnapshot",
    "STATE_SCHEMA",
    "STATUS_SCHEMA",
    "SuccessorChainRefused",
    "SuccessorEvidenceChain",
    "build_parser",
    "legacy_specs",
    "main",
    "read_chain_status",
    "request_stop",
    "start_chain_detached",
]


if __name__ == "__main__":
    raise SystemExit(main())
