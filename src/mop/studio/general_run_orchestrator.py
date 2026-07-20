"""Single-process "General Run" orchestrator for the Generation 1 chain.

This net-new, append-only module replaces the four-parent waiter relay (the v7
adopter, the ext-v4 waiter, the categorized waiter, and the full-generations
waiter) with ONE tabula-rasa control loop named ``general-run`` (no version
suffix).  One process sequences the whole chain stage by stage:

    observe_legacy -> run_horizon_v1 -> run_horizon_v2
                   -> run_categorized_wave -> run_full_generations_wave

The individual sealed compute programs keep their sealed ids and still run as
their own generic supervisor + workers when active; only the relay layer is
consolidated here.  The orchestrator is observe-only toward every legacy and
already-running process, never signals anything, and never double-launches a
program that another parent (or a prior incarnation of this orchestrator) has
already started cleanly.

observe_legacy is churn-immune.  Every non-census prerequisite is complete iff
its sealed validated terminal result exists (a pure file read, never a process
walk).  The consolidated-final census is the one prerequisite that can still be
running here, and its ``mop-final-*`` worker pool churns faster than any bounded
resnapshot, so the orchestrator polls ONLY the census parent's pinned identity
(pid, create_time, label): while that exact identity is alive the stage waits,
and the moment its pid is free the census is done and the stage advances.  The
create_time match guards pid reuse, so a recycled pid carrying a different
create_time is never mistaken for the census and can neither spuriously advance
nor spuriously fail the stage.  No child worker is ever sampled and no process
title is ever stabilized, so the census worker churn can never hold the stage.

Each compute stage OBSERVES before it launches: it adopts an already-running
``mop-supervisor:<id>`` (by label or exact generic supervisor command) or a
clean sealed status instead of relaunching, and only starts the sealed program
through the generic supervisor ``start_detached`` (itself idempotent) when
nothing is cleanly running.  A stage advances only when its generic status is
clean-complete under the stable double-read ``_stable_complete_generic_status``.
The advisory result-aware reprofiler is consulted at each compute stage entry
and its recommended worker count is recorded in the status; the consult is
strictly advisory and never changes a sealed program.

On restart, ``_load_state`` re-reads the sealed state and resumes at the exact
stage; a live supervisor is re-adopted, so no program is ever double-launched
and no progress is lost.  The orchestrator ships its own drain control file so
future cutovers have the clean lever the three ext waiters never had.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studio import generation1_successor_chain_v7 as legacy_chain
from mop.studio.generation1_categorized_batch_extension_chain import (
    _admission_summary,
    _default_host_admission,
    _runs_generic_supervisor_command,
    _snapshot_matches,
    validate_generic_status,
)
from mop.studio.generation1_categorized_batch_extension_chain import (
    _stable_complete_generic_status as stable_complete_generic_status,
)
from mop.studio.generation1_result_aware_reprofiler import (
    DEFAULT_HOST_CORES,
    DEFAULT_MEMORY_GB,
    DEFAULT_PER_CAPSULE_MEM_CAP_GB,
    ReprofileRefused,
    recommended_worker_count,
)
from mop.studio.generation1_result_aware_reprofiler import (
    SCHEMA as REPROFILE_SCHEMA,
)
from mop.studio.generation1_successor_chain_v7 import (
    ProcessSnapshot,
    SuccessorChainRefused,
)
from mop.studio.generation1_supervisor import (
    LOCK_FILE as GENERIC_LOCK_FILE,
)
from mop.studio.generation1_supervisor import (
    FileLock,
    Generation1Refused,
    Program,
    atomic_write_json,
    canonical_sha256,
    load_program,
    read_status,
    sha256_file,
    start_detached,
)
from mop.studio.mop_label_scheme import ROLE_ADOPTER, mop_label

PROGRAM_ID = "general-run"
PARENT_LABEL = mop_label("general-run", ROLE_ADOPTER)
STATE_SCHEMA = "mop-general-run-state/v1"
STATUS_SCHEMA = "mop-general-run-status/v1"
CONTROL_SCHEMA = "mop-general-run-control/v1"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_HORIZON_PROGRAM = legacy_chain.DEFAULT_HORIZON_PROGRAM

STATE_FILE = "program_state.json"
STATUS_FILE = "current_status.json"
LOCK_FILE = "control/general-run.lock"
START_LOCK_FILE = "control/start.lock"
CONTROL_FILE = "control/request.json"

POLL_SECONDS = 30.0
STARTUP_ACK_SECONDS = 300.0
PROCESS_TIME_TOLERANCE_SECONDS = 0.02
MAX_JSON_BYTES = 64 * 1024 * 1024

FINAL_CENSUS_PID = 67790
FINAL_CENSUS_CREATE_TIME = 1784160329.97262
FINAL_CENSUS_LABEL = "mop-final-campaign"
FINAL_CENSUS_CREATE_TIME_TOLERANCE_SECONDS = 2.0

STAGE_OBSERVE_LEGACY = "observe_legacy"
STAGE_HORIZON_V1 = "run_horizon_v1"
STAGE_HORIZON_V2 = "run_horizon_v2"
STAGE_CATEGORIZED_WAVE = "run_categorized_wave"
STAGE_FULL_GENERATIONS_WAVE = "run_full_generations_wave"
STAGES = (
    STAGE_OBSERVE_LEGACY,
    STAGE_HORIZON_V1,
    STAGE_HORIZON_V2,
    STAGE_CATEGORIZED_WAVE,
    STAGE_FULL_GENERATIONS_WAVE,
)

TERMINAL_STATES = frozenset({"complete", "failure_hold", "integrity_hold", "drained"})
UNSAFE_TERMINAL_STATES = frozenset({"failure_hold", "integrity_hold", "drained"})
STATUS_STATES = frozenset({"starting", "waiting_host", *STAGES, *TERMINAL_STATES})
COMPUTE_ROW_STATES = frozenset(
    {"pending", "observed", "running", "adoption_wait", "launching", "retry_wait", "complete", "failure_hold"}
)

IMPLEMENTATION_PATH = Path(__file__).resolve()
SUPERVISOR_FIELDS = frozenset({"pid", "create_time", "implementation_path", "implementation_sha256"})
STATE_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "created_at",
        "updated_at",
        "finished_at",
        "execution_enabled",
        "status",
        "stage",
        "supervisor",
        "parent_implementation",
        "legacy_capsules",
        "compute_capsules",
        "reprofile",
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
        "stage",
        "supervisor",
        "parent_implementation",
        "capsules",
        "counts",
        "reprofile",
        "last_admission",
        "problems",
        "signals_allowed",
        "activation_allowed",
        "scientific_promotion",
        "status_sha256",
    }
)


class GeneralRunRefused(RuntimeError):
    pass


IdentityProbe = Callable[[Mapping[str, Any]], str]
ProcessTable = Callable[[], Sequence[ProcessSnapshot]]
ProgramLoader = Callable[..., Program]
StatusReader = Callable[[Program], Mapping[str, Any]]
SupervisorStarter = Callable[..., Mapping[str, Any]]
AdmissionProbe = Callable[[Program, Mapping[str, Any] | None], Mapping[str, Any]]
Reprofiler = Callable[[], Mapping[str, Any]]
NowFn = Callable[[], dt.datetime]


@dataclass(frozen=True, slots=True)
class ComputeStage:

    stage_id: str
    program_id: str
    manifest_rel: str
    predecessor_program_id: str | None
    enforce_admission: bool


COMPUTE_STAGES: tuple[ComputeStage, ...] = (
    ComputeStage(
        stage_id=STAGE_HORIZON_V1,
        program_id="generation1-successor-horizon-v1",
        manifest_rel="configs/campaign/generation1_successor_horizon_v1.json",
        predecessor_program_id=None,
        enforce_admission=False,
    ),
    ComputeStage(
        stage_id=STAGE_HORIZON_V2,
        program_id="generation1-successor-horizon-v2",
        manifest_rel="configs/campaign/generation1_successor_horizon_v2.json",
        predecessor_program_id="generation1-successor-horizon-v1",
        enforce_admission=False,
    ),
    ComputeStage(
        stage_id=STAGE_CATEGORIZED_WAVE,
        program_id="generation1-successor-categorized-batch-wave-v1",
        manifest_rel="configs/campaign/generation1_successor_categorized_batch_wave_v1.json",
        predecessor_program_id="generation1-successor-horizon-v2",
        enforce_admission=True,
    ),
    ComputeStage(
        stage_id=STAGE_FULL_GENERATIONS_WAVE,
        program_id="generation1-full-generations-wave-v1",
        manifest_rel="configs/campaign/generation1_full_generations_wave_v1.json",
        predecessor_program_id="generation1-successor-categorized-batch-wave-v1",
        enforce_admission=True,
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
        raise GeneralRunRefused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file() or not 0 < stat.st_size <= MAX_JSON_BYTES:
        raise GeneralRunRefused(f"{label} is not a safe bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise GeneralRunRefused(f"{label} changed while being read: {path}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeneralRunRefused(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GeneralRunRefused(f"{label} must be a JSON object: {path}")
    return value


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise GeneralRunRefused(f"{label} self-seal is invalid")


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
    return legacy_chain._default_identity_probe(identity)


def _default_process_table() -> tuple[ProcessSnapshot, ...]:
    return legacy_chain._default_process_table()


def _default_executable_probe(process: ProcessSnapshot) -> str | None:
    return legacy_chain._default_executable_probe(process)


def _default_reprofiler() -> dict[str, Any]:

    worker_math = recommended_worker_count(
        DEFAULT_HOST_CORES,
        DEFAULT_MEMORY_GB,
        DEFAULT_PER_CAPSULE_MEM_CAP_GB,
    )
    return {
        "schema": REPROFILE_SCHEMA,
        "advisory": True,
        "recommended_workers": worker_math["recommended_workers"],
        "binding_constraint": worker_math["binding_constraint"],
    }


def _empty_compute_row() -> dict[str, Any]:
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


class GeneralRunOrchestrator:

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        horizon_program_path: Path = DEFAULT_HORIZON_PROGRAM,
        legacy_specs: Sequence[Any] | None = None,
        execute: bool = False,
        identity_probe_fn: IdentityProbe = _default_identity_probe,
        process_table_fn: ProcessTable = _default_process_table,
        executable_probe_fn: Callable[[ProcessSnapshot], str | None] = _default_executable_probe,
        program_loader_fn: ProgramLoader = load_program,
        status_reader_fn: StatusReader = read_status,
        target_starter_fn: SupervisorStarter = start_detached,
        host_admission_fn: AdmissionProbe = _default_host_admission,
        reprofiler_fn: Reprofiler = _default_reprofiler,
        now_fn: NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.horizon_program_path = horizon_program_path.resolve()
        self.execute = execute
        self.identity_probe_fn = identity_probe_fn
        self.process_table_fn = process_table_fn
        self.executable_probe_fn = executable_probe_fn
        self.program_loader_fn = program_loader_fn
        self.status_reader_fn = status_reader_fn
        self.target_starter_fn = target_starter_fn
        self.host_admission_fn = host_admission_fn
        self.reprofiler_fn = reprofiler_fn
        self.now_fn = now_fn
        self.sleep_fn = sleep_fn
        self.poll_seconds = poll_seconds
        if not self.root.is_relative_to(self.repo_root):
            raise GeneralRunRefused("general-run root must remain inside the repository")
        self.root.mkdir(parents=True, exist_ok=True)
        self._stage_by_id = {stage.stage_id: stage for stage in COMPUTE_STAGES}
        self._manifest_paths = {
            stage.stage_id: (self.repo_root / stage.manifest_rel).resolve() for stage in COMPUTE_STAGES
        }
        self._manifest_by_program_id = {
            stage.program_id: (self.repo_root / stage.manifest_rel).resolve() for stage in COMPUTE_STAGES
        }
        self._legacy_owner = legacy_chain.SuccessorEvidenceChain(
            root=self.root,
            repo_root=self.repo_root,
            horizon_program_path=self.horizon_program_path,
            specs=tuple(legacy_specs) if legacy_specs is not None else None,
            execute=False,
            process_table_fn=self.process_table_fn,
            executable_probe_fn=self.executable_probe_fn,
            identity_probe_fn=self.identity_probe_fn,
            now_fn=self.now_fn,
            sleep_fn=self.sleep_fn,
        )
        self._legacy_specs = self._legacy_owner.specs
        self.state = self._load_state()


    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def status_path(self) -> Path:
        return self.root / STATUS_FILE

    def _iso_now(self) -> str:
        return _iso(self.now_fn())

    def _identity(self) -> dict[str, Any]:
        return _self_identity(self.repo_root)

    def _implementation_authority(self) -> dict[str, str]:
        return _implementation_authority(self.repo_root)

    def _initial_legacy_capsules(self) -> dict[str, Any]:
        return {spec.stage_id: legacy_chain.v3._empty_capsule() for spec in self._legacy_specs}

    def _initial_state(self) -> dict[str, Any]:
        now = self._iso_now()
        return {
            "schema": STATE_SCHEMA,
            "program_id": PROGRAM_ID,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "execution_enabled": self.execute,
            "status": "starting",
            "stage": STAGE_OBSERVE_LEGACY,
            "supervisor": self._identity(),
            "parent_implementation": self._implementation_authority(),
            "legacy_capsules": self._initial_legacy_capsules(),
            "compute_capsules": {stage.stage_id: _empty_compute_row() for stage in COMPUTE_STAGES},
            "reprofile": None,
            "last_admission": None,
            "problems": [],
        }

    def _validate_bound_authorities(self, state: Mapping[str, Any]) -> None:
        if state.get("parent_implementation") != self._implementation_authority():
            raise GeneralRunRefused("general-run parent implementation authority drifted")

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        state = _read_json(self.state_path, "general-run state")
        _validate_seal(state, "state_sha256", "general-run state")
        if set(state) != STATE_FIELDS:
            raise GeneralRunRefused("general-run state fields drifted")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != PROGRAM_ID:
            raise GeneralRunRefused("general-run state identity drifted")
        legacy = state.get("legacy_capsules")
        compute = state.get("compute_capsules")
        if not isinstance(legacy, Mapping) or set(legacy) != {spec.stage_id for spec in self._legacy_specs}:
            raise GeneralRunRefused("general-run legacy capsule inventory drifted")
        if not isinstance(compute, Mapping) or set(compute) != {stage.stage_id for stage in COMPUTE_STAGES}:
            raise GeneralRunRefused("general-run compute capsule inventory drifted")
        if state.get("stage") not in STAGES:
            raise GeneralRunRefused("general-run stage pointer drifted")
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

    def _counts(self) -> dict[str, int]:
        legacy = self.state["legacy_capsules"]
        compute = self.state["compute_capsules"]
        legacy_complete = sum(row.get("status") == "complete" for row in legacy.values())
        compute_complete = sum(row.get("status") == "complete" for row in compute.values())
        stage = self.state.get("stage")
        stage_index = STAGES.index(stage) if stage in STAGES else 0
        return {
            "legacy_complete": legacy_complete,
            "legacy_total": len(legacy),
            "compute_complete": compute_complete,
            "compute_total": len(compute),
            "stage_index": stage_index,
            "stage_total": len(STAGES),
        }

    def _status_payload(self) -> dict[str, Any]:
        capsules = {**self.state["legacy_capsules"], **self.state["compute_capsules"]}
        core = {
            "schema": STATUS_SCHEMA,
            "program_id": PROGRAM_ID,
            "created_at": self.state["created_at"],
            "updated_at": self.state["updated_at"],
            "finished_at": self.state["finished_at"],
            "execution_enabled": self.state["execution_enabled"],
            "state": self.state["status"],
            "stage": self.state["stage"],
            "supervisor": self.state["supervisor"],
            "parent_implementation": self.state["parent_implementation"],
            "capsules": capsules,
            "counts": self._counts(),
            "reprofile": self.state["reprofile"],
            "last_admission": self.state["last_admission"],
            "problems": self.state["problems"],
            "signals_allowed": False,
            "activation_allowed": False,
            "scientific_promotion": False,
        }
        return {**core, "status_sha256": canonical_sha256(core)}

    def _publish(self) -> dict[str, Any]:
        self.state["updated_at"] = self._iso_now()
        core = dict(self.state)
        core.pop("state_sha256", None)
        sealed = {**core, "state_sha256": canonical_sha256(core)}
        atomic_write_json(self.state_path, sealed)
        self.state = dict(sealed)
        status = self._status_payload()
        atomic_write_json(self.status_path, status)
        return status


    def _is_final_census(self, process: ProcessSnapshot) -> bool:
        return (
            process.pid == FINAL_CENSUS_PID
            and process.label == FINAL_CENSUS_LABEL
            and math.isfinite(process.create_time)
            and math.isclose(
                process.create_time,
                FINAL_CENSUS_CREATE_TIME,
                rel_tol=0.0,
                abs_tol=FINAL_CENSUS_CREATE_TIME_TOLERANCE_SECONDS,
            )
        )

    def _observe_final_census(self, row: dict[str, Any], processes: Sequence[ProcessSnapshot]) -> None:
        on_pid = [process for process in processes if process.pid == FINAL_CENSUS_PID]
        census = next((process for process in on_pid if self._is_final_census(process)), None)
        if census is not None:
            row["status"] = "adopted"
            row["process"] = census.identity()
            row["last_problem"] = "pinned consolidated-final census parent alive; awaiting its exit"
            return
        if on_pid:
            row["status"] = "adoption_wait"
            row["process"] = None
            row["last_problem"] = (
                f"census pid {FINAL_CENSUS_PID} is held by a non-census process; "
                "holding without mistaking a reused pid for the census"
            )
            return
        row["status"] = "complete"
        row["returncode"] = 0
        row["finished_at"] = self._iso_now()
        row["process"] = None
        row["artifacts"] = []
        row["last_problem"] = None

    def _tick_observe_legacy(self) -> dict[str, Any]:
        processes = tuple(self.process_table_fn())
        for spec in self._legacy_specs:
            row = self.state["legacy_capsules"][spec.stage_id]
            if spec.process_label == FINAL_CENSUS_LABEL:
                self._observe_final_census(row, processes)
                continue
            result = self._legacy_owner._validated_result(spec)
            if result is not None:
                self._legacy_owner._mark_complete(row, spec.result_path, result)
        rows = [self.state["legacy_capsules"][spec.stage_id] for spec in self._legacy_specs]
        if all(row.get("status") == "complete" for row in rows):
            self.state["stage"] = STAGE_HORIZON_V1
            self.state["status"] = STAGE_HORIZON_V1
            return self._publish()
        self.state["status"] = STAGE_OBSERVE_LEGACY
        return self._publish()


    def _load_compute_program(self, path: Path, expected_id: str) -> Program:
        try:
            program = self.program_loader_fn(path, repo_root=self.repo_root)
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise GeneralRunRefused(
                f"{expected_id} program is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        if program.program_id != expected_id:
            raise GeneralRunRefused(f"loaded {expected_id} program identity drifted")
        return program

    def _load_predecessor(self, stage: ComputeStage) -> Program | None:
        if stage.predecessor_program_id is None:
            return None
        path = self._manifest_by_program_id[stage.predecessor_program_id]
        return self._load_compute_program(path, stage.predecessor_program_id)

    def _artifact(self, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "path": str(path.resolve().relative_to(self.repo_root)),
            "sha256": sha256_file(path),
            "schema": payload.get("schema"),
            "all_ok": True,
            "problems": [],
        }

    def _compute_status(self, program: Program) -> dict[str, Any] | None:
        if not program.status_path.exists():
            return None
        try:
            status = dict(self.status_reader_fn(program))
        except (Generation1Refused, OSError, TypeError, ValueError) as exc:
            raise GeneralRunRefused(
                f"{program.program_id} status is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        validate_generic_status(program, status)
        return status

    def _visible_supervisors(
        self,
        program: Program,
        processes: Sequence[ProcessSnapshot],
    ) -> tuple[ProcessSnapshot, ...]:
        label = f"mop-supervisor:{program.program_id}"
        visible = tuple(
            process
            for process in processes
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
            raise GeneralRunRefused(f"visible {program.program_id} supervisor has inexact metadata")
        if len(visible) > 1:
            raise GeneralRunRefused(f"multiple {program.program_id} generic supervisors are visible")
        return visible

    def _record_reprofile(self, stage: ComputeStage) -> None:
        try:
            consult = self.reprofiler_fn()
            if not isinstance(consult, Mapping):
                raise GeneralRunRefused("reprofiler consult returned no mapping")
            workers = consult.get("recommended_workers")
            record: dict[str, Any] = {
                "stage": stage.stage_id,
                "program_id": stage.program_id,
                "recommended_workers": (
                    int(workers) if isinstance(workers, int) and not isinstance(workers, bool) else None
                ),
                "schema": consult.get("schema"),
                "advisory": True,
                "note": "advisory reprofile consult only; never changes a sealed program",
                "checked_at": self._iso_now(),
            }
        except (ReprofileRefused, GeneralRunRefused, OSError, RuntimeError, TypeError, ValueError):
            record = {
                "stage": stage.stage_id,
                "program_id": stage.program_id,
                "recommended_workers": None,
                "schema": None,
                "advisory": True,
                "note": "advisory reprofile consult unavailable; launch unaffected",
                "checked_at": self._iso_now(),
            }
        self.state["reprofile"] = record

    def _record_admission(self, program: Program, status: Mapping[str, Any] | None) -> bool:
        receipt = self.host_admission_fn(program, status)
        if not isinstance(receipt, Mapping):
            raise GeneralRunRefused("host admission returned no receipt")
        summary = _admission_summary(receipt, checked_at=self._iso_now())
        self.state["last_admission"] = summary
        return bool(summary["allowed"])

    def _persist_launch_intent(self, stage: ComputeStage, row: dict[str, Any]) -> dict[str, Any]:
        row["status"] = "launching"
        row["attempts"] = int(row.get("attempts") or 0) + 1
        row["launch_requested_at"] = self._iso_now()
        row["last_problem"] = "generic supervisor launch intent persisted before start"
        self.state["status"] = stage.stage_id
        self.state["finished_at"] = None
        self._publish()
        return self.state["compute_capsules"][stage.stage_id]

    def _invoke_starter(
        self,
        stage: ComputeStage,
        row: dict[str, Any],
        program: Program,
    ) -> dict[str, Any]:
        try:
            launched = self.target_starter_fn(program, execute=True, use_caffeinate=True)
        except (Generation1Refused, OSError, RuntimeError, ValueError) as exc:
            row["status"] = "retry_wait"
            row["last_problem"] = (
                f"{stage.program_id} startup acknowledgement pending: {type(exc).__name__}: {exc}"
            )
            return row
        if not isinstance(launched, Mapping):
            raise GeneralRunRefused(f"{stage.program_id} generic starter returned no receipt")
        returned = launched.get("status")
        if not isinstance(returned, Mapping):
            row["status"] = "retry_wait"
            row["last_problem"] = f"{stage.program_id} starter returned no sealed status acknowledgement"
            return row
        returned_state = validate_generic_status(program, returned)
        if returned_state in UNSAFE_TERMINAL_STATES:
            row["status"] = "failure_hold"
            row["finished_at"] = self._iso_now()
            row["last_problem"] = f"{stage.program_id} acknowledged unsafe state {returned_state}"
            return row
        launched_pid = launched.get("launched_pid")
        row["launched_pid"] = (
            launched_pid if isinstance(launched_pid, int) and not isinstance(launched_pid, bool) else None
        )
        row["status"] = "running"
        row["started_at"] = row.get("started_at") or self._iso_now()
        row["last_problem"] = f"{stage.program_id} generic supervisor start accepted"
        supervisor = returned.get("supervisor")
        if isinstance(supervisor, Mapping):
            row["process"] = dict(supervisor)
        return row

    def _launch_compute(
        self,
        stage: ComputeStage,
        row: dict[str, Any],
        program: Program,
    ) -> dict[str, Any]:
        predecessor = self._load_predecessor(stage)
        if predecessor is None:
            row = self._persist_launch_intent(stage, row)
            return self._invoke_starter(stage, row, program)
        with FileLock(predecessor.program_root / GENERIC_LOCK_FILE):
            before = stable_complete_generic_status(predecessor, acquire_lock=False)
            row = self._persist_launch_intent(stage, row)
            after = stable_complete_generic_status(predecessor, acquire_lock=False)
            if before != after:
                raise GeneralRunRefused(
                    f"{stage.predecessor_program_id} predecessor changed across "
                    f"{stage.stage_id} launch intent"
                )
            return self._invoke_starter(stage, row, program)

    def _mark_compute_complete(
        self,
        program: Program,
        row: dict[str, Any],
        visible: Sequence[ProcessSnapshot],
    ) -> None:
        complete = stable_complete_generic_status(program)
        supervisor = complete.get("supervisor")
        if visible and (not isinstance(supervisor, Mapping) or not _snapshot_matches(visible[0], supervisor)):
            raise GeneralRunRefused(f"completed {program.program_id} conflicts with visible supervisor")
        expected = [self._artifact(program.status_path, complete)]
        if row.get("status") == "complete":
            if (
                row.get("returncode") != 0
                or not isinstance(row.get("finished_at"), str)
                or row.get("artifacts") != expected
                or row.get("last_problem") is not None
            ):
                raise GeneralRunRefused(f"completed {program.program_id} binding drifted")
            return
        row.update(
            {
                "status": "complete",
                "returncode": 0,
                "finished_at": self._iso_now(),
                "artifacts": expected,
                "last_problem": None,
            }
        )
        if isinstance(supervisor, Mapping):
            row["process"] = dict(supervisor)

    def _reconcile_compute(
        self,
        stage: ComputeStage,
        row: dict[str, Any],
        program: Program,
        processes: Sequence[ProcessSnapshot],
    ) -> dict[str, Any]:
        status = self._compute_status(program)
        visible = self._visible_supervisors(program, processes)
        if row.get("status") == "complete":
            self._mark_compute_complete(program, row, visible)
            return row
        if status is not None:
            state = str(status["state"])
            if state == "complete":
                self._mark_compute_complete(program, row, visible)
                return row
            if state in UNSAFE_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = self._iso_now()
                row["last_problem"] = f"{stage.program_id} entered {state}"
                return row
            supervisor = status.get("supervisor")
            if isinstance(supervisor, Mapping):
                identity_state = self.identity_probe_fn(supervisor)
                if identity_state == "unknown":
                    raise GeneralRunRefused(f"{stage.program_id} supervisor identity is inexact")
                if identity_state == "alive":
                    if len(visible) != 1 or not _snapshot_matches(visible[0], supervisor):
                        raise GeneralRunRefused(
                            f"sealed {stage.program_id} supervisor conflicts with visible process"
                        )
                    row["status"] = "running"
                    row["process"] = dict(supervisor)
                    row["started_at"] = row.get("started_at") or self._iso_now()
                    row["last_problem"] = f"{stage.program_id} generic supervisor is in state {state}"
                    return row
        if visible:
            row["status"] = "adoption_wait"
            row["process"] = visible[0].identity()
            row["last_problem"] = (
                f"exact {stage.program_id} supervisor is visible while sealed acknowledgement is pending"
            )
            return row
        if not self.execute:
            row["status"] = "pending"
            row["last_problem"] = f"{stage.program_id} start requires explicit --execute"
            return row
        self._record_reprofile(stage)
        if stage.enforce_admission and not self._record_admission(program, status):
            row["status"] = "pending"
            row["last_problem"] = f"host busy: {self.state['last_admission']['reason']}"
            return row
        return self._launch_compute(stage, row, program)

    def _tick_compute(self, stage_id: str) -> dict[str, Any]:
        stage = self._stage_by_id[stage_id]
        program = self._load_compute_program(self._manifest_paths[stage_id], stage.program_id)
        processes = tuple(self.process_table_fn())
        row = self.state["compute_capsules"][stage_id]
        row = self._reconcile_compute(stage, row, program, processes)
        self.state["compute_capsules"][stage_id] = row
        row_state = row.get("status")
        if row_state == "complete":
            next_index = STAGES.index(stage_id) + 1
            if next_index >= len(STAGES):
                self.state["status"] = "complete"
                self.state["finished_at"] = self._iso_now()
            else:
                self.state["stage"] = STAGES[next_index]
                self.state["status"] = STAGES[next_index]
        elif row_state == "failure_hold":
            self.state["status"] = "failure_hold"
            self.state["finished_at"] = self._iso_now()
            self._append_problem(str(row.get("last_problem")))
        elif (
            row_state == "pending"
            and stage.enforce_admission
            and isinstance(self.state.get("last_admission"), Mapping)
            and self.state["last_admission"].get("allowed") is False
        ):
            self.state["status"] = "waiting_host"
        else:
            self.state["status"] = stage_id
        return self._publish()

    def _revalidate_complete(self) -> None:
        stage = COMPUTE_STAGES[-1]
        program = self._load_compute_program(self._manifest_paths[stage.stage_id], stage.program_id)
        row = self.state["compute_capsules"][stage.stage_id]
        if row.get("status") != "complete":
            raise GeneralRunRefused("general-run marked complete without a complete final wave")
        self._mark_compute_complete(program, row, ())


    def _control_requested(self) -> bool:
        path = self.root / CONTROL_FILE
        if not path.exists():
            return False
        value = _read_json(path, "general-run control")
        _validate_seal(value, "control_sha256", "general-run control")
        if (
            value.get("schema") != CONTROL_SCHEMA
            or value.get("program_id") != PROGRAM_ID
            or value.get("action") != "drain"
        ):
            raise GeneralRunRefused("general-run control identity drifted")
        return True

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_bound_authorities(self.state)
            state = self.state.get("status")
            stage = self.state.get("stage")
            if state == "complete":
                self._revalidate_complete()
                return self._publish()
            if state in TERMINAL_STATES:
                return self._publish()
            if self._control_requested():
                self.state["status"] = "drained"
                self.state["finished_at"] = self._iso_now()
                return self._publish()
            if stage == STAGE_OBSERVE_LEGACY:
                return self._tick_observe_legacy()
            return self._tick_compute(stage)
        except (
            GeneralRunRefused,
            SuccessorChainRefused,
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
            self.state["finished_at"] = self._iso_now()
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


def validate_general_run_status(status: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> str:

    repo_root = repo_root.resolve()
    if not isinstance(status, Mapping) or set(status) != STATUS_FIELDS:
        raise GeneralRunRefused("general-run status fields drifted")
    _validate_seal(status, "status_sha256", "general-run status")
    state = status.get("state")
    stage = status.get("stage")
    problems = status.get("problems")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("program_id") != PROGRAM_ID
        or status.get("signals_allowed") is not False
        or status.get("activation_allowed") is not False
        or status.get("scientific_promotion") is not False
        or not isinstance(status.get("execution_enabled"), bool)
        or not isinstance(state, str)
        or state not in STATUS_STATES
        or stage not in STAGES
        or not isinstance(problems, list)
        or any(not isinstance(problem, str) or not problem for problem in problems)
    ):
        raise GeneralRunRefused("general-run status identity or safety drifted")
    if (state == "complete" or state not in TERMINAL_STATES) and (
        status.get("execution_enabled") is not True or problems != []
    ):
        raise GeneralRunRefused("general-run safe acknowledgement is not clean")
    if status.get("parent_implementation") != _implementation_authority(repo_root):
        raise GeneralRunRefused("general-run status implementation authority drifted")
    created = _parse_time(status.get("created_at"))
    updated = _parse_time(status.get("updated_at"))
    finished = _parse_time(status.get("finished_at"))
    if created is None or updated is None or created > updated:
        raise GeneralRunRefused("general-run status timestamps drifted")
    if state in TERMINAL_STATES:
        if finished is None or finished > updated:
            raise GeneralRunRefused("general-run terminal timestamp drifted")
    elif status.get("finished_at") is not None:
        raise GeneralRunRefused("general-run nonterminal status is marked finished")
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
        raise GeneralRunRefused("general-run supervisor authority drifted")
    capsules = status.get("capsules")
    if not isinstance(capsules, Mapping):
        raise GeneralRunRefused("general-run status capsule inventory drifted")
    for capsule_id, row in capsules.items():
        if not isinstance(row, Mapping) or row.get("status") not in (
            COMPUTE_ROW_STATES | {"adopted", "adoption_wait", "launching"}
        ):
            raise GeneralRunRefused(f"general-run capsule {capsule_id} receipt drifted")
    counts = status.get("counts")
    if not isinstance(counts, Mapping) or counts.get("stage_total") != len(STAGES):
        raise GeneralRunRefused("general-run status counts drifted")
    return state


def read_general_run_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = root.resolve() / STATUS_FILE
    status = _read_json(path, "general-run status")
    _validate_seal(status, "status_sha256", "general-run status")
    if status.get("schema") != STATUS_SCHEMA or status.get("program_id") != PROGRAM_ID:
        raise GeneralRunRefused("general-run status identity drifted")
    return status


def read_validated_complete_general_run_status(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:

    status = read_general_run_status(root)
    state = validate_general_run_status(status, repo_root=repo_root)
    if state != "complete":
        raise GeneralRunRefused("general-run durable status is not complete")
    return status


def request_stop(root: Path = DEFAULT_ROOT, reason: str = "operator requested drain") -> dict[str, Any]:
    if not reason.strip():
        raise GeneralRunRefused("drain reason must be nonempty")
    root = root.resolve()
    (root / "control").mkdir(parents=True, exist_ok=True)
    core = {
        "schema": CONTROL_SCHEMA,
        "program_id": PROGRAM_ID,
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


def _exact_parent_command(process: ProcessSnapshot, entrypoint: Path) -> bool:
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
        and (process.label == PARENT_LABEL or _exact_parent_command(process, entrypoint))
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
            raise GeneralRunRefused("visible general-run parent has inexact metadata")
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise GeneralRunRefused("multiple general-run parent processes are visible")
    if labelled:
        return labelled[0]
    return next((process for process in candidates if process.pid == process.pgid), candidates[0])


def start_general_run_detached(
    *,
    root: Path = DEFAULT_ROOT,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise GeneralRunRefused("detached general-run start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_general_run.py").resolve()
        status_path = root / STATUS_FILE
        status = read_general_run_status(root) if status_path.is_file() else None
        visible = _visible_parent_process(entrypoint)
        identity = status.get("supervisor") if status is not None else None
        if _process_identity_alive(identity):
            if visible is None:
                raise GeneralRunRefused("sealed general-run identity has no exact visible parent")
            if (
                not isinstance(identity, Mapping)
                or identity.get("pid") != visible.pid
                or not math.isclose(
                    float(identity.get("create_time", -1)),
                    visible.create_time,
                    rel_tol=0.0,
                    abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
                )
            ):
                raise GeneralRunRefused("sealed general-run identity conflicts with visible parent")
            return {"already_running": True, "status": status}
        if visible is not None:
            return {"already_running": True, "status": status, "observed_process": visible.identity()}
        if status is not None and status.get("state") in TERMINAL_STATES:
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
                    acknowledged = read_general_run_status(root)
                    validate_general_run_status(acknowledged)
                    break
                except (GeneralRunRefused, OSError, ValueError):
                    acknowledged = None
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise GeneralRunRefused(
                f"detached general-run did not acknowledge startup; inspect {stderr_path}"
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
                raise GeneralRunRefused("general-run execution requires explicit --execute")
            set_process_label(PARENT_LABEL)
            payload = GeneralRunOrchestrator(root=arguments.root, execute=True).run(
                max_cycles=1 if arguments.once else None
            )
        elif arguments.command == "start":
            payload = start_general_run_detached(
                root=arguments.root,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        elif arguments.command == "status":
            payload = read_general_run_status(arguments.root)
        else:
            payload = request_stop(arguments.root, arguments.reason)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if arguments.command == "run" and payload.get("state") not in {"complete", "drained"}:
            return 2
        return 0
    except (
        GeneralRunRefused,
        Generation1Refused,
        SuccessorChainRefused,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"all_ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "COMPUTE_STAGES",
    "CONTROL_SCHEMA",
    "DEFAULT_ROOT",
    "GeneralRunOrchestrator",
    "GeneralRunRefused",
    "PARENT_LABEL",
    "PROGRAM_ID",
    "STAGES",
    "STATE_SCHEMA",
    "STATUS_SCHEMA",
    "build_parser",
    "main",
    "read_general_run_status",
    "read_validated_complete_general_run_status",
    "request_stop",
    "start_general_run_detached",
    "validate_general_run_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
