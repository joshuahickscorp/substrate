"""Append-only v5 adopter for the Generation 1 successor evidence chain.

Version 5 supersedes the immutable v4 integrity hold. It preserves the
observe-only legacy adoption and unchanged v1 horizon launch while adding a
bounded, fail-closed re-snapshot around the exact multiprocessing process-title
transition reproduced in the consolidated final campaign.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies import generation1_successor_horizon_v2_verify as horizon_v2_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics_queue
from mop.studio import generation1_successor_chain as v3
from mop.studio import generation1_successor_chain_v4 as predecessor_v4
from mop.studio import generation1_supervisor
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

CHAIN_ID = "generation1-successor-evidence-chain-v5"
PARENT_LABEL = "mop-successor-evidence-chain-v5"
STATE_SCHEMA = "mop-generation1-successor-evidence-chain-state/v5"
STATUS_SCHEMA = "mop-generation1-successor-evidence-chain-status/v5"
ADOPTION_SCHEMA = "mop-generation1-legacy-adoption-receipt/v3"
CONTROL_SCHEMA = "mop-generation1-successor-evidence-chain-control/v3"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / CHAIN_ID
DEFAULT_HORIZON_PROGRAM = v3.DEFAULT_HORIZON_PROGRAM
STATE_FILE = v3.STATE_FILE
STATUS_FILE = v3.STATUS_FILE
LOCK_FILE = v3.LOCK_FILE
START_LOCK_FILE = v3.START_LOCK_FILE
CONTROL_FILE = v3.CONTROL_FILE
STARTUP_ACK_SECONDS = v3.STARTUP_ACK_SECONDS
PROCESS_TIME_TOLERANCE_SECONDS = v3.PROCESS_TIME_TOLERANCE_SECONDS
SNAPSHOT_ATTEMPTS = 5
SNAPSHOT_INTERVAL_SECONDS = 0.05
LOCK_ATTEMPTS = 41
LOCK_INTERVAL_SECONDS = 0.05
PROCESS_TRANSITION_ATTEMPTS = 41
PROCESS_TRANSITION_INTERVAL_SECONDS = 0.05
TERMINAL_STATES = v3.TERMINAL_STATES
HORIZON_TERMINAL_STATES = v3.HORIZON_TERMINAL_STATES
HORIZON_STATUS_STATES = frozenset(
    {
        "starting",
        "execution_disabled",
        "progressing",
        "running",
        "resource_wait",
        "retry_wait",
        "recovery_wait",
        "recovery_observe",
        "complete",
        "drained",
        "failure_hold",
        "integrity_hold",
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
        "horizon_program",
        "capsules",
        "problems",
        "state_sha256",
    }
)
CAPSULE_FIELDS = frozenset(v3._empty_capsule())
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
STATUS_FIELDS = frozenset(
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
CHAIN_STATUS_STATES = frozenset(
    {
        "starting",
        "waiting_legacy",
        "adoption_wait",
        "waiting_horizon",
        *TERMINAL_STATES,
    }
)
CAPSULE_STATES = frozenset(
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
ADOPTION_FIELDS = frozenset(
    {
        "schema",
        "chain_id",
        "stage_id",
        "program_id",
        "adopted_at",
        "process",
        "observed_status",
        "policy",
        "receipt_sha256",
    }
)
ADOPTED_PROCESS_FIELDS = frozenset(
    {
        "pid",
        "create_time",
        "pgid",
        "cwd",
        "label",
    }
)
OBSERVED_STATUS_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "status_sha256",
        "state",
    }
)
LEGACY_OBSERVED_STATUS_STATES = frozenset(
    {
        "waiting",
        "running",
        "complete",
        "failure_hold",
    }
)

LegacySpec = v3.LegacySpec
ProcessSnapshot = v3.ProcessSnapshot
ProcessTable = v3.ProcessTable
SuccessorChainRefused = v3.SuccessorChainRefused
_default_identity_probe = v3._default_identity_probe
_default_process_table = v3._default_process_table
_iso = v3._iso
_now = v3._now
_parse_time = v3._parse_time
_read_json = v3._read_json
_validate_seal = v3._validate_seal


ExecutableProbe = Callable[[ProcessSnapshot], str | None]


def _default_executable_probe(process: ProcessSnapshot) -> str | None:
    try:
        live = psutil.Process(process.pid)
        if not math.isclose(
            live.create_time(),
            process.create_time,
            rel_tol=0.0,
            abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
        ):
            return None
        return str(Path(live.exe()).resolve())
    except (
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
        OSError,
        RuntimeError,
        ValueError,
    ):
        return None


def legacy_specs(repo_root: Path = REPO_ROOT) -> tuple[LegacySpec, ...]:

    mechanics_prefixes = tuple(f"mop-{lane.lane_id.lower()}-" for lane in mechanics_queue.LANES)
    return tuple(
        replace(spec, child_label_prefixes=mechanics_prefixes)
        if spec.stage_id == "legacy_successor_mechanics"
        else spec
        for spec in v3.legacy_specs(repo_root)
    )


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    return (
        str(resolved.relative_to(repo_root.resolve()))
        if resolved.is_relative_to(repo_root.resolve())
        else str(resolved)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _status_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    capsules = state["capsules"]
    complete = sum(row.get("status") == "complete" for row in capsules.values())
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": CHAIN_ID,
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "supervisor": state["supervisor"],
        "execution_enabled": state["execution_enabled"],
        "state": state["status"],
        "horizon_program": state["horizon_program"],
        "capsules": capsules,
        "counts": {
            "complete": complete,
            "total": len(capsules),
            "remaining": len(capsules) - complete,
        },
        "finished_at": state["finished_at"],
        "problems": state["problems"],
        "activation_allowed": False,
        "scientific_promotion": False,
        "supersedes": "generation1-successor-evidence-chain-v4",
    }
    return {**core, "status_sha256": canonical_sha256(core)}


def _implementation_authority(repo_root: Path) -> dict[str, Any]:
    implementation = Path(__file__).resolve()
    base = Path(v3.__file__).resolve()
    predecessor = Path(predecessor_v4.__file__).resolve()
    return {
        "path": _relative_or_absolute(implementation, repo_root),
        "sha256": sha256_file(implementation),
        "superseded_predecessor": {
            "path": _relative_or_absolute(predecessor, repo_root),
            "sha256": sha256_file(predecessor),
        },
        "inherited_base": {
            "path": _relative_or_absolute(base, repo_root),
            "sha256": sha256_file(base),
        },
    }


def _generic_supervisor_authority(repo_root: Path) -> dict[str, str]:
    implementation = Path(generation1_supervisor.__file__).resolve()
    return {
        "path": _relative_or_absolute(implementation, repo_root),
        "sha256": sha256_file(implementation),
    }


def _horizon_authority(
    horizon_program_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    payload = _read_json(horizon_program_path, "successor horizon program manifest")
    _validate_seal(payload, "program_sha256", "successor horizon program manifest")
    if (
        payload.get("schema") != "mop-generation1-program/v1"
        or payload.get("program_id") != "generation1-successor-horizon-v1"
    ):
        raise SuccessorChainRefused("successor horizon program manifest identity drifted")
    return {
        "path": str(horizon_program_path.resolve().relative_to(repo_root.resolve())),
        "file_sha256": sha256_file(horizon_program_path),
        "program_sha256": payload["program_sha256"],
        "program_id": payload["program_id"],
    }


class SuccessorEvidenceChain(v3.SuccessorEvidenceChain):

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        horizon_program_path: Path = DEFAULT_HORIZON_PROGRAM,
        specs: Sequence[LegacySpec] | None = None,
        execute: bool = False,
        process_table_fn: ProcessTable = _default_process_table,
        executable_probe_fn: ExecutableProbe = _default_executable_probe,
        identity_probe_fn: v3.IdentityProbe = _default_identity_probe,
        popen_fn: v3.PopenFactory = subprocess.Popen,
        supervisor_start_fn: v3.SupervisorStarter = start_generation1_detached,
        now_fn: v3.NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = v3.POLL_SECONDS,
    ) -> None:
        self.executable_probe_fn = executable_probe_fn
        super().__init__(
            root=root,
            repo_root=repo_root,
            horizon_program_path=horizon_program_path,
            specs=tuple(specs) if specs is not None else legacy_specs(repo_root),
            execute=execute,
            process_table_fn=process_table_fn,
            identity_probe_fn=identity_probe_fn,
            popen_fn=popen_fn,
            supervisor_start_fn=supervisor_start_fn,
            now_fn=now_fn,
            sleep_fn=sleep_fn,
            poll_seconds=poll_seconds,
        )

    def _identity(self) -> dict[str, Any]:
        process = psutil.Process(os.getpid())
        implementation = Path(__file__).resolve()
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "implementation_path": _relative_or_absolute(implementation, self.repo_root),
            "implementation_sha256": sha256_file(implementation),
        }

    def _implementation_authority(self) -> dict[str, Any]:
        return _implementation_authority(self.repo_root)

    def _horizon_authority(self) -> dict[str, Any]:
        return _horizon_authority(self.horizon_program_path, self.repo_root)

    def _launch_legacy(
        self,
        spec: LegacySpec,
        row: dict[str, Any],
    ) -> None:
        if self.execute:
            legacy_rows = [self.state["capsules"][candidate.stage_id] for candidate in self.specs]
            if any(
                candidate is not row and candidate.get("status") == "failure_hold"
                for candidate in legacy_rows
            ):
                row["status"] = "pending"
                row["last_problem"] = "restart blocked by another failed legacy prerequisite"
                return
            if self.state["capsules"]["successor_horizon"].get("status") != "pending":
                raise SuccessorChainRefused(
                    "legacy prerequisite regressed after successor horizon activity began"
                )
            self.state["status"] = (
                "adoption_wait"
                if any(
                    candidate is not row and candidate.get("status") == "adoption_wait"
                    for candidate in legacy_rows
                )
                else "waiting_legacy"
            )
            self.state["finished_at"] = None
        super()._launch_legacy(spec, row)

    def _initial_state(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        capsules = {spec.stage_id: v3._empty_capsule() for spec in self.specs}
        capsules["successor_horizon"] = v3._empty_capsule()
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
        state = _read_json(self.state_path, "successor chain v5 state")
        _validate_seal(state, "state_sha256", "successor chain v5 state")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != CHAIN_ID:
            raise SuccessorChainRefused("successor chain v5 state identity drifted")
        expected_capsules = {spec.stage_id for spec in self.specs} | {"successor_horizon"}
        capsules = state.get("capsules")
        if not isinstance(capsules, dict) or set(capsules) != expected_capsules:
            raise SuccessorChainRefused("successor chain v5 capsule inventory drifted")
        self._validate_bound_authorities(state)
        state.pop("state_sha256", None)
        state["execution_enabled"] = self.execute
        state["supervisor"] = self._identity()
        return state

    def _publish(self) -> dict[str, Any]:
        now = _iso(self.now_fn())
        self.state["updated_at"] = now
        core = dict(self.state)
        core.pop("state_sha256", None)
        sealed_state = {**core, "state_sha256": canonical_sha256(core)}
        atomic_write_json(self.state_path, sealed_state)
        self.state = dict(sealed_state)
        status = _status_payload(self.state)
        atomic_write_json(self.status_path, status)
        return status

    def _command_residual(self, spec: LegacySpec, process: ProcessSnapshot) -> bool:

        if len(process.command) != 2 or len(spec.restart_command) != 2:
            return False
        expected_executable = Path(spec.restart_command[0])
        actual_executable = Path(process.command[0])
        expected_script = self.repo_root / spec.restart_command[1]
        actual_script = Path(process.command[1])
        if not actual_script.is_absolute():
            actual_script = Path(process.cwd) / actual_script
        try:
            return (
                actual_executable.resolve() == expected_executable.resolve()
                and actual_script.resolve() == expected_script.resolve()
            )
        except OSError:
            return False

    def _exact_spawn_title_transition(
        self,
        process: ProcessSnapshot,
        parent: ProcessSnapshot,
    ) -> bool:

        if (
            process.ppid != parent.pid
            or process.pgid != parent.pid
            or process.cwd != str(self.repo_root)
            or not math.isfinite(process.create_time)
            or process.create_time <= 0
            or len(process.command) != 4
        ):
            return False
        expected_python = self.repo_root / ".venv/bin/python"
        executable = self.executable_probe_fn(process)
        if executable is None:
            return False
        try:
            if Path(executable).resolve() != expected_python.resolve():
                return False
        except OSError:
            return False
        joined_spawn = re.compile(
            rf"{re.escape(str(expected_python))} -c "
            r"from multiprocessing\.spawn import spawn_main; ?"
            r"spawn_main\(tracker_fd=[0-9]+, pipe_handle=[0-9]+\) "
            r"--multiprocessing-fork"
        )
        return joined_spawn.fullmatch(process.command[0]) is not None

    def _snapshot_provisionals(
        self,
        processes: Sequence[ProcessSnapshot],
    ) -> dict[str, tuple[ProcessSnapshot, tuple[ProcessSnapshot, ...]]]:

        provisional: dict[str, tuple[ProcessSnapshot, tuple[ProcessSnapshot, ...]]] = {}
        by_pid = {process.pid: process for process in processes}
        if len(by_pid) != len(processes):
            raise SuccessorChainRefused("process snapshot contains duplicate PIDs")
        for spec in self.specs:
            try:
                self._matching_processes(spec, processes)
            except SuccessorChainRefused as exc:
                prefix = f"{spec.stage_id} process group has unexpected members: "
                message = str(exc)
                if not message.startswith(prefix):
                    raise
                raw_pids = message.removeprefix(prefix).split(", ")
                try:
                    pids = tuple(int(raw_pid) for raw_pid in raw_pids)
                except ValueError:
                    raise SuccessorChainRefused(
                        f"{spec.stage_id} unexpected process inventory is malformed"
                    ) from exc
                parents = [process for process in processes if process.label == spec.process_label]
                candidates = tuple(by_pid[pid] for pid in pids if pid in by_pid)
                if (
                    len(parents) != 1
                    or len(candidates) != len(pids)
                    or any(
                        not self._exact_spawn_title_transition(candidate, parents[0])
                        for candidate in candidates
                    )
                ):
                    raise
                provisional[spec.stage_id] = (parents[0], candidates)
        return provisional

    @staticmethod
    def _same_snapshot_identity(
        process: ProcessSnapshot,
        identity: Mapping[str, Any],
    ) -> bool:
        created = identity.get("create_time")
        return (
            process.pid == identity.get("pid")
            and isinstance(created, int | float)
            and not isinstance(created, bool)
            and math.isclose(
                process.create_time,
                float(created),
                rel_tol=0.0,
                abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
            )
            and process.pgid == identity.get("pgid")
            and process.cwd == identity.get("cwd")
            and process.label == identity.get("label")
        )

    def _stable_process_table(self) -> tuple[ProcessSnapshot, ...]:

        current = tuple(self.process_table_fn())
        provisional = self._snapshot_provisionals(current)
        if not provisional:
            return current

        parent_bindings = {
            stage_id: parent.identity()
            for stage_id, (parent, _candidates) in provisional.items()
        }
        candidate_bindings: dict[tuple[int, float], tuple[str, dict[str, Any]]] = {
            (
                candidate.pid,
                candidate.create_time,
            ): (
                stage_id,
                {
                    "pid": candidate.pid,
                    "create_time": candidate.create_time,
                    "pgid": candidate.pgid,
                    "cwd": candidate.cwd,
                    "ppid": candidate.ppid,
                },
            )
            for stage_id, (_parent, candidates) in provisional.items()
            for candidate in candidates
        }
        clean_samples = 0
        for _attempt in range(PROCESS_TRANSITION_ATTEMPTS):
            self.sleep_fn(PROCESS_TRANSITION_INTERVAL_SECONDS)
            current = tuple(self.process_table_fn())
            by_pid = {process.pid: process for process in current}
            if len(by_pid) != len(current):
                raise SuccessorChainRefused("process snapshot contains duplicate PIDs")

            for stage_id, identity in parent_bindings.items():
                spec = next(spec for spec in self.specs if spec.stage_id == stage_id)
                parents = [process for process in current if process.label == spec.process_label]
                if len(parents) != 1 or not self._same_snapshot_identity(parents[0], identity):
                    raise SuccessorChainRefused(
                        f"{stage_id} exact parent changed during process-title stabilization"
                    )

            for (pid, create_time), (stage_id, identity) in candidate_bindings.items():
                visible = by_pid.get(pid)
                if visible is None:
                    identity_state = self.identity_probe_fn(identity)
                    if identity_state != "gone":
                        raise SuccessorChainRefused(
                            f"provisional process {pid} disappeared without exact gone identity"
                        )
                    continue
                if not math.isclose(
                    visible.create_time,
                    create_time,
                    rel_tol=0.0,
                    abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
                ):
                    raise SuccessorChainRefused(
                        f"provisional process {pid} changed identity during stabilization"
                    )
                if (
                    visible.pgid != identity.get("pgid")
                    or visible.cwd != identity.get("cwd")
                    or visible.ppid != identity.get("ppid")
                ):
                    raise SuccessorChainRefused(
                        f"provisional process {pid} changed ownership boundary during stabilization"
                    )
                spec = next(spec for spec in self.specs if spec.stage_id == stage_id)
                parent = next(
                    process for process in current if process.label == spec.process_label
                )
                resolved_worker = any(
                    visible.label.startswith(prefix) for prefix in spec.child_label_prefixes
                )
                if not (
                    resolved_worker
                    or self._spawn_worker(visible)
                    or self._exact_spawn_title_transition(visible, parent)
                ):
                    raise SuccessorChainRefused(
                        f"provisional process {pid} resolved to an unauthorized process class"
                    )

            provisional = self._snapshot_provisionals(current)
            for stage_id, (parent, candidates) in provisional.items():
                bound = parent_bindings.get(stage_id)
                if bound is None:
                    parent_bindings[stage_id] = parent.identity()
                elif not self._same_snapshot_identity(parent, bound):
                    raise SuccessorChainRefused(
                        f"{stage_id} exact parent changed during process-title stabilization"
                    )
                for candidate in candidates:
                    key = (candidate.pid, candidate.create_time)
                    prior = candidate_bindings.get(key)
                    if prior is None:
                        candidate_bindings[key] = (
                            stage_id,
                            {
                                "pid": candidate.pid,
                                "create_time": candidate.create_time,
                                "pgid": candidate.pgid,
                                "cwd": candidate.cwd,
                                "ppid": candidate.ppid,
                            },
                        )
                    elif prior[0] != stage_id or any(
                        prior[1].get(field) != getattr(candidate, field)
                        for field in ("pid", "create_time", "pgid", "cwd", "ppid")
                    ):
                        raise SuccessorChainRefused(
                            f"provisional process {candidate.pid} boundary changed during stabilization"
                        )

            if provisional:
                clean_samples = 0
                continue
            clean_samples += 1
            if clean_samples == 2:
                return current

        unresolved = sorted(
            {
                candidate.pid
                for _parent, candidates in provisional.values()
                for candidate in candidates
            }
        )
        suffix = ", ".join(str(pid) for pid in unresolved) if unresolved else "unknown"
        raise SuccessorChainRefused(
            f"process-title transition did not stabilize within the bounded window: {suffix}"
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

        command_residuals = [
            process
            for process in processes
            if process not in labelled and self._command_residual(spec, process)
        ]
        trackers = [process for process in processes if self._resource_tracker(process)]
        spawn_workers = [
            process
            for process in processes
            if process not in labelled and process not in command_residuals and self._spawn_worker(process)
        ]
        parent = labelled[0] if labelled else None
        worker_candidates = [
            process
            for process in processes
            if process not in labelled
            and process not in command_residuals
            and any(process.label.startswith(prefix) for prefix in spec.child_label_prefixes)
        ]
        if parent is not None:
            worker_residuals = [
                process
                for process in worker_candidates
                if process.cwd == str(self.repo_root)
                and process.pgid == parent.pid
                and process.ppid == parent.pid
            ]
            inexact_worker_candidates = [
                process
                for process in worker_candidates
                if process not in worker_residuals
                and (
                    process.cwd == str(self.repo_root)
                    or process.pgid == parent.pid
                    or process.ppid == parent.pid
                )
            ]
            if inexact_worker_candidates:
                raise SuccessorChainRefused(
                    f"{spec.stage_id} matching worker label is outside the exact "
                    "repository-cwd, parent-PGID, and parent-PPID boundary: "
                    + ", ".join(str(process.pid) for process in inexact_worker_candidates)
                )
        else:
            worker_residuals = [
                process for process in worker_candidates if process.cwd == str(self.repo_root)
            ]
        residuals = [*worker_residuals, *command_residuals]
        if any(process.cwd != str(self.repo_root) for process in residuals):
            raise SuccessorChainRefused(f"{spec.stage_id} residual worker or restart command has invalid cwd")

        if parent is not None:
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
                process
                for process in spawn_workers
                if process.pgid == parent.pid and process.ppid == parent.pid
            ]
            group_members = [
                process for process in processes if process.pid != parent.pid and process.pgid == parent.pid
            ]
            allowed_group_members = [
                process
                for process in group_members
                if process in worker_residuals or process in trackers or process in owned_spawn_workers
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
                    f"{spec.stage_id} has residual restart commands or owned workers "
                    "outside the exact parent process group"
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
                if process.cwd == str(self.repo_root) and process.pgid not in exact_owner_pgids
            )
            residuals.extend(
                process
                for process in trackers
                if process.cwd == str(self.repo_root) and process.pgid not in exact_owner_pgids
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
        policy = {
            "observe_only": True,
            "signals_allowed": False,
            "restart_only_after_exact_absence": True,
            "restart_command_match": "exact-executable-and-two-argv-shape",
            "process_title_transition": "exact-spawn-rewrite-bounded-resnapshot",
        }
        if path.exists():
            receipt = _read_json(path, f"{spec.stage_id} v5 adoption receipt")
            _validate_seal(receipt, "receipt_sha256", f"{spec.stage_id} v5 adoption receipt")
            if (
                receipt.get("schema") != ADOPTION_SCHEMA
                or receipt.get("chain_id") != CHAIN_ID
                or receipt.get("stage_id") != spec.stage_id
                or receipt.get("program_id") != spec.program_id
                or receipt.get("process") != identity
                or _parse_time(receipt.get("adopted_at")) is None
                or receipt.get("policy") != policy
            ):
                raise SuccessorChainRefused(f"{spec.stage_id} v5 adoption receipt identity drifted")
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
                "policy": policy,
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

    def _validated_horizon_status(self, program: Any) -> dict[str, Any] | None:
        if not program.status_path.is_file():
            return None
        try:
            status = read_generation1_status(program)
        except (Generation1Refused, OSError, ValueError) as exc:
            raise SuccessorChainRefused(
                f"successor horizon status is invalid: {type(exc).__name__}: {exc}"
            ) from exc
        expected_program = {
            "path": str(program.path),
            "file_sha256": program.file_sha256,
            "program_sha256": program.program_sha256,
        }
        if status.get("program") != expected_program:
            raise SuccessorChainRefused("successor horizon status program authority drifted")
        state = status.get("state")
        if not isinstance(state, str) or state not in HORIZON_STATUS_STATES:
            raise SuccessorChainRefused("successor horizon status state vocabulary drifted")
        problems = status.get("problems")
        supervisor = status.get("supervisor")
        implementation = _generic_supervisor_authority(self.repo_root)
        if not isinstance(problems, list) or any(
            not isinstance(problem, str) or not problem for problem in problems
        ):
            raise SuccessorChainRefused("successor horizon status problems inventory drifted")
        if (
            not isinstance(supervisor, Mapping)
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
        ):
            raise SuccessorChainRefused("successor horizon status supervisor authority drifted")
        if status.get("accepted_injection_count") != 0 or status.get("next_injection_sequence") != 1:
            raise SuccessorChainRefused("successor horizon status admitted an unauthorized injection")
        capsules = status.get("capsules")
        expected_capsules = {capsule.capsule_id: capsule for capsule in program.capsules}
        if not isinstance(capsules, Mapping) or set(capsules) != set(expected_capsules):
            raise SuccessorChainRefused("successor horizon status capsule inventory drifted")
        for capsule_id, capsule in expected_capsules.items():
            row = capsules[capsule_id]
            if not isinstance(row, Mapping):
                raise SuccessorChainRefused(f"successor horizon status capsule {capsule_id} is invalid")
            expected = {
                "id": capsule.capsule_id,
                "kind": capsule.kind,
                "priority": capsule.priority,
                "depends_on": list(capsule.depends_on),
                "capsule_sha256": capsule.capsule_sha256,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise SuccessorChainRefused(
                    f"successor horizon status capsule {capsule_id} authority drifted"
                )
            if row.get("status") not in {"pending", "running", "complete", "failed"}:
                raise SuccessorChainRefused(f"successor horizon status capsule {capsule_id} state drifted")
        return status

    def _validated_complete_horizon_artifacts(
        self,
        program: Any,
        status: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if status.get("problems") != []:
            raise SuccessorChainRefused("successor horizon complete status contains problems")
        try:
            horizon_v2_verify._validate_parent_supervisor_status(program, status)
        except ValueError as exc:
            raise SuccessorChainRefused(
                f"successor horizon complete supervisor evidence is invalid: {exc}"
            ) from exc
        capsules = status["capsules"]
        if any(not isinstance(row, Mapping) or row.get("status") != "complete" for row in capsules.values()):
            raise SuccessorChainRefused("successor horizon complete status has incomplete capsules")
        artifacts = self._horizon_status_artifacts(program, status)
        if not artifacts:
            raise SuccessorChainRefused("successor horizon complete status has no artifacts")
        for artifact in artifacts:
            raw_path = artifact.get("path")
            digest = artifact.get("sha256")
            if (
                not isinstance(raw_path, str)
                or not raw_path
                or not isinstance(digest, str)
                or len(digest) != 64
                or artifact.get("all_ok") is not True
                or artifact.get("problems") != []
            ):
                raise SuccessorChainRefused("successor horizon complete artifact report is invalid")
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise SuccessorChainRefused(f"successor horizon complete artifact path is unsafe: {raw_path}")
            path = self.repo_root / relative
            resolved = path.resolve()
            if (
                not resolved.is_relative_to(self.repo_root)
                or path.is_symlink()
                or not resolved.is_file()
                or sha256_file(resolved) != digest
            ):
                raise SuccessorChainRefused(f"successor horizon complete artifact drifted: {raw_path}")
        return artifacts

    def _exact_live_horizon_supervisor(
        self,
        supervisor: Mapping[str, Any],
        processes: Sequence[ProcessSnapshot],
    ) -> bool:
        label = "mop-supervisor:generation1-successor-horizon-v1"
        matches = [process for process in processes if process.label == label]
        if any(process.cwd != str(self.repo_root) for process in matches):
            raise SuccessorChainRefused("successor horizon supervisor has invalid cwd")
        if len(matches) > 1:
            raise SuccessorChainRefused("multiple successor horizon supervisors are visible")
        identity_state = self.identity_probe_fn(supervisor)
        if identity_state == "unknown":
            raise SuccessorChainRefused("successor horizon supervisor identity is inexact")
        if identity_state == "alive":
            if len(matches) != 1:
                raise SuccessorChainRefused("live successor horizon supervisor lacks one exact visible label")
            visible = matches[0]
            created = supervisor.get("create_time")
            if (
                supervisor.get("pid") != visible.pid
                or isinstance(created, bool)
                or not isinstance(created, int | float)
                or not math.isclose(
                    float(created),
                    visible.create_time,
                    rel_tol=0.0,
                    abs_tol=PROCESS_TIME_TOLERANCE_SECONDS,
                )
            ):
                raise SuccessorChainRefused(
                    "sealed successor horizon supervisor conflicts with visible identity"
                )
            return True
        if matches:
            raise SuccessorChainRefused("visible successor horizon supervisor conflicts with sealed identity")
        return False

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
        if not isinstance(bound_program, Mapping) or (
            program.program_sha256 != bound_program.get("program_sha256")
            or program.file_sha256 != bound_program.get("file_sha256")
            or str(program.path.relative_to(self.repo_root)) != bound_program.get("path")
        ):
            raise SuccessorChainRefused("loaded successor horizon program authority drifted")

        was_complete = row.get("status") == "complete"
        status = self._validated_horizon_status(program)
        if was_complete and status is None:
            raise SuccessorChainRefused("completed successor horizon status disappeared")
        if status is not None:
            state = status["state"]
            if state == "complete":
                artifacts = self._validated_complete_horizon_artifacts(program, status)
                if was_complete and (row.get("returncode") != 0 or row.get("artifacts") != artifacts):
                    raise SuccessorChainRefused(
                        "completed successor horizon receipt or artifact inventory drifted"
                    )
                row["status"] = "complete"
                row["returncode"] = 0
                if row.get("finished_at") is None:
                    row["finished_at"] = _iso(self.now_fn())
                row["artifacts"] = artifacts
                row["last_problem"] = None
                row["process"] = dict(status["supervisor"])
                return
            if was_complete:
                raise SuccessorChainRefused(f"completed successor horizon status regressed to {state}")
            if state in HORIZON_TERMINAL_STATES:
                row["status"] = "failure_hold"
                row["finished_at"] = _iso(self.now_fn())
                row["last_problem"] = f"successor horizon entered {state}"
                return
            supervisor = status.get("supervisor")
            if not isinstance(supervisor, Mapping):
                raise SuccessorChainRefused("successor horizon status supervisor is missing")
            if self._exact_live_horizon_supervisor(supervisor, processes):
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
        self.state["status"] = "waiting_horizon"
        self.state["finished_at"] = None
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

    def _validate_completed_chain_artifacts(
        self,
        processes: Sequence[ProcessSnapshot],
    ) -> None:
        for spec in self.specs:
            row = self.state["capsules"][spec.stage_id]
            result = self._validated_result(spec)
            expected_artifacts = [self._artifact(spec.result_path, result)] if result is not None else None
            if (
                result is None
                or row.get("status") != "complete"
                or row.get("returncode") != 0
                or row.get("artifacts") != expected_artifacts
            ):
                raise SuccessorChainRefused(
                    f"completed {spec.stage_id} result or artifact inventory disappeared or drifted"
                )
        horizon = self.state["capsules"]["successor_horizon"]
        if horizon.get("status") != "complete":
            raise SuccessorChainRefused("completed chain lost its successor horizon receipt")
        self._reconcile_horizon(horizon, processes)

    def tick(self) -> dict[str, Any]:
        try:
            self._validate_bound_authorities(self.state)
            state = self.state.get("status")
            if state == "complete":
                self._validate_completed_chain_artifacts(self._stable_process_table())
                return self._publish()
            if state in TERMINAL_STATES:
                return self._publish()
            if self._control_requested():
                self.state["status"] = "drained"
                self.state["finished_at"] = _iso(self.now_fn())
                return self._publish()

            processes = self._stable_process_table()
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
        except (
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

    def _control_requested(self) -> bool:
        path = self.root / CONTROL_FILE
        if not path.exists():
            return False
        value = _read_json(path, "successor chain v5 control")
        _validate_seal(value, "control_sha256", "successor chain v5 control")
        if (
            value.get("schema") != CONTROL_SCHEMA
            or value.get("program_id") != CHAIN_ID
            or value.get("action") != "drain"
        ):
            raise SuccessorChainRefused("successor chain v5 control identity drifted")
        return True


def validate_chain_status(
    status: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    horizon_program_path: Path | None = None,
    specs: Sequence[LegacySpec] | None = None,
) -> str:

    repo_root = repo_root.resolve()
    if horizon_program_path is None:
        relative = DEFAULT_HORIZON_PROGRAM.resolve().relative_to(REPO_ROOT.resolve())
        horizon_program_path = repo_root / relative
    horizon_program_path = horizon_program_path.resolve()
    configured_specs = tuple(specs) if specs is not None else legacy_specs(repo_root)
    if set(status) != STATUS_FIELDS:
        raise SuccessorChainRefused("successor chain v5 status fields drifted")
    _validate_seal(status, "status_sha256", "successor chain v5 status")
    state = status.get("state")
    problems = status.get("problems")
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("program_id") != CHAIN_ID
        or status.get("activation_allowed") is not False
        or status.get("scientific_promotion") is not False
        or status.get("supersedes") != "generation1-successor-evidence-chain-v4"
        or not isinstance(status.get("execution_enabled"), bool)
        or not isinstance(state, str)
        or state not in CHAIN_STATUS_STATES
        or not isinstance(problems, list)
        or any(not isinstance(problem, str) or not problem for problem in problems)
        or status.get("horizon_program") != _horizon_authority(horizon_program_path, repo_root)
    ):
        raise SuccessorChainRefused("successor chain v5 status identity or authority drifted")
    if (state == "complete" or state not in TERMINAL_STATES) and (
        status.get("execution_enabled") is not True or problems != []
    ):
        raise SuccessorChainRefused("successor chain v5 safe acknowledgement is not clean")
    created_at = _parse_time(status.get("created_at"))
    updated_at = _parse_time(status.get("updated_at"))
    finished_at = _parse_time(status.get("finished_at"))
    if created_at is None or updated_at is None or created_at > updated_at:
        raise SuccessorChainRefused("successor chain v5 status timestamps drifted")
    if state in TERMINAL_STATES:
        if finished_at is None or finished_at > updated_at:
            raise SuccessorChainRefused("successor chain v5 terminal timestamp drifted")
    elif status.get("finished_at") is not None:
        raise SuccessorChainRefused("successor chain v5 nonterminal status is marked finished")
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
        raise SuccessorChainRefused("successor chain v5 supervisor authority drifted")
    capsules = status.get("capsules")
    expected_ids = {spec.stage_id for spec in configured_specs} | {"successor_horizon"}
    if not isinstance(capsules, Mapping) or set(capsules) != expected_ids:
        raise SuccessorChainRefused("successor chain v5 status capsule inventory drifted")
    complete = 0
    for capsule_id, raw_row in capsules.items():
        if not isinstance(raw_row, Mapping) or set(raw_row) != CAPSULE_FIELDS:
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} fields drifted")
        row_state = raw_row.get("status")
        attempts = raw_row.get("attempts")
        returncode = raw_row.get("returncode")
        started_at = _parse_time(raw_row.get("started_at"))
        finished_at = _parse_time(raw_row.get("finished_at"))
        launch_requested_at = _parse_time(raw_row.get("launch_requested_at"))
        artifacts = raw_row.get("artifacts")
        adoption_receipts = raw_row.get("adoption_receipts")
        last_problem = raw_row.get("last_problem")
        process = raw_row.get("process")
        launched_pid = raw_row.get("launched_pid")
        if (
            not isinstance(row_state, str)
            or row_state not in CAPSULE_STATES
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
            or not isinstance(adoption_receipts, list)
            or any(not isinstance(receipt, str) or not receipt for receipt in adoption_receipts)
            or (last_problem is not None and (not isinstance(last_problem, str) or not last_problem))
            or (process is not None and not isinstance(process, Mapping))
            or (
                launched_pid is not None
                and (isinstance(launched_pid, bool) or not isinstance(launched_pid, int) or launched_pid <= 0)
            )
        ):
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} receipt shape drifted")
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
                raise SuccessorChainRefused(
                    f"successor chain v5 capsule {capsule_id} process identity drifted"
                )
        if row_state == "complete":
            complete += 1
            if returncode != 0 or finished_at is None or not artifacts or last_problem is not None:
                raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} completion drifted")
        elif (
            returncode is not None
            or artifacts != []
            or (row_state == "failure_hold" and (finished_at is None or not isinstance(last_problem, str)))
            or (row_state != "failure_hold" and finished_at is not None)
        ):
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} noncompletion drifted")
    expected_counts = {
        "complete": complete,
        "total": len(expected_ids),
        "remaining": len(expected_ids) - complete,
    }
    if status.get("counts") != expected_counts:
        raise SuccessorChainRefused("successor chain v5 status counts drifted")
    legacy_states = [capsules[spec.stage_id].get("status") for spec in configured_specs]
    horizon_state = capsules["successor_horizon"].get("status")
    all_legacy_complete = all(value == "complete" for value in legacy_states)
    all_complete = complete == len(expected_ids)
    pristine_horizon = capsules["successor_horizon"] == v3._empty_capsule()
    coherent = True
    if state == "starting":
        coherent = all(row == v3._empty_capsule() for row in capsules.values())
    elif state == "waiting_legacy":
        coherent = (
            not all_legacy_complete
            and pristine_horizon
            and all(value in {"pending", "launching", "adopted", "complete"} for value in legacy_states)
        )
    elif state == "adoption_wait":
        coherent = (
            pristine_horizon
            and any(value == "adoption_wait" for value in legacy_states)
            and all(
                value in {"pending", "launching", "adopted", "adoption_wait", "complete"}
                for value in legacy_states
            )
        ) or (all_legacy_complete and horizon_state == "adoption_wait")
    elif state == "waiting_horizon":
        coherent = all_legacy_complete and horizon_state in {"pending", "running", "launching"}
    elif state == "complete":
        coherent = all_complete
    elif state == "failure_hold":
        coherent = any(row.get("status") == "failure_hold" for row in capsules.values())
    if not coherent or (state not in TERMINAL_STATES and all_complete):
        raise SuccessorChainRefused("successor chain v5 state and capsule progression drifted")
    if state not in TERMINAL_STATES and any(row.get("status") == "failure_hold" for row in capsules.values()):
        raise SuccessorChainRefused("successor chain v5 safe status contains a failed capsule")
    return state


def _validate_complete_state_shape(
    owner: SuccessorEvidenceChain,
    state: Mapping[str, Any],
) -> None:
    if set(state) != STATE_FIELDS:
        raise SuccessorChainRefused("successor chain v5 state fields drifted")
    _validate_seal(state, "state_sha256", "successor chain v5 state")
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("program_id") != CHAIN_ID
        or state.get("execution_enabled") is not True
        or state.get("status") != "complete"
        or state.get("problems") != []
    ):
        raise SuccessorChainRefused("successor chain v5 complete state boundary drifted")
    created_at = _parse_time(state.get("created_at"))
    updated_at = _parse_time(state.get("updated_at"))
    finished_at = _parse_time(state.get("finished_at"))
    if (
        created_at is None
        or updated_at is None
        or finished_at is None
        or not created_at <= finished_at <= updated_at
    ):
        raise SuccessorChainRefused("successor chain v5 complete state timestamps drifted")
    owner._validate_bound_authorities(state)
    supervisor = state.get("supervisor")
    implementation = owner._implementation_authority()
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
        raise SuccessorChainRefused("successor chain v5 supervisor authority drifted")
    capsules = state.get("capsules")
    expected_ids = {spec.stage_id for spec in owner.specs} | {"successor_horizon"}
    if not isinstance(capsules, Mapping) or set(capsules) != expected_ids:
        raise SuccessorChainRefused("successor chain v5 complete capsule inventory drifted")
    for capsule_id, raw_row in capsules.items():
        if not isinstance(raw_row, Mapping) or set(raw_row) != CAPSULE_FIELDS:
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} fields drifted")
        attempts = raw_row.get("attempts")
        if (
            raw_row.get("status") != "complete"
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or raw_row.get("returncode") != 0
            or _parse_time(raw_row.get("finished_at")) is None
            or not isinstance(raw_row.get("artifacts"), list)
            or raw_row.get("last_problem") is not None
            or (raw_row.get("started_at") is not None and _parse_time(raw_row.get("started_at")) is None)
            or (
                raw_row.get("launch_requested_at") is not None
                and _parse_time(raw_row.get("launch_requested_at")) is None
            )
            or (raw_row.get("process") is not None and not isinstance(raw_row.get("process"), Mapping))
            or not isinstance(raw_row.get("adoption_receipts"), list)
            or any(
                not isinstance(receipt, str) or not receipt
                for receipt in raw_row.get("adoption_receipts", [])
            )
            or (
                raw_row.get("launched_pid") is not None
                and (
                    isinstance(raw_row.get("launched_pid"), bool)
                    or not isinstance(raw_row.get("launched_pid"), int)
                    or raw_row["launched_pid"] <= 0
                )
            )
        ):
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} completion receipt drifted")
        if capsule_id == "successor_horizon" and raw_row.get("adoption_receipts") != []:
            raise SuccessorChainRefused("successor chain v5 horizon capsule has unexpected adoption receipts")
    for spec in owner.specs:
        row = capsules[spec.stage_id]
        receipt_paths = row["adoption_receipts"]
        if len(receipt_paths) != len(set(receipt_paths)):
            raise SuccessorChainRefused(
                f"successor chain v5 {spec.stage_id} adoption receipt inventory contains duplicates"
            )
        validated_processes: list[Mapping[str, Any]] = []
        resolved_receipts: set[Path] = set()
        receipt_files: set[tuple[int, int]] = set()
        receipt_directory = owner.root / "adoptions" / spec.stage_id
        receipt_root = receipt_directory.resolve()
        for raw_path in receipt_paths:
            relative = Path(raw_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt path is unsafe"
                )
            candidate = owner.repo_root / relative
            lexical = owner.repo_root
            symlink_component = False
            for component in relative.parts:
                lexical /= component
                if lexical.is_symlink():
                    symlink_component = True
                    break
            if (
                candidate.parent != receipt_directory
                or symlink_component
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt is missing "
                    "or is not a canonical regular non-symlink file"
                )
            path = candidate.resolve()
            if not path.is_relative_to(receipt_root):
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt path drifted"
                )
            file_stat = candidate.stat()
            file_identity = (file_stat.st_dev, file_stat.st_ino)
            if path in resolved_receipts or file_identity in receipt_files:
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt resolves to a duplicate file"
                )
            resolved_receipts.add(path)
            receipt_files.add(file_identity)
            receipt = _read_json(path, f"{spec.stage_id} v5 adoption receipt")
            if set(receipt) != ADOPTION_FIELDS:
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt fields drifted"
                )
            _validate_seal(
                receipt,
                "receipt_sha256",
                f"{spec.stage_id} v5 adoption receipt",
            )
            process = receipt.get("process")
            observed_status = receipt.get("observed_status")
            policy = {
                "observe_only": True,
                "signals_allowed": False,
                "restart_only_after_exact_absence": True,
                "restart_command_match": "exact-executable-and-two-argv-shape",
                "process_title_transition": "exact-spawn-rewrite-bounded-resnapshot",
            }
            if (
                receipt.get("schema") != ADOPTION_SCHEMA
                or receipt.get("chain_id") != CHAIN_ID
                or receipt.get("stage_id") != spec.stage_id
                or receipt.get("program_id") != spec.program_id
                or _parse_time(receipt.get("adopted_at")) is None
                or not isinstance(process, Mapping)
                or set(process) != ADOPTED_PROCESS_FIELDS
                or receipt.get("policy") != policy
            ):
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt authority drifted"
                )
            pid = process.get("pid")
            create_time = process.get("create_time")
            pgid = process.get("pgid")
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or isinstance(create_time, bool)
                or not isinstance(create_time, int | float)
                or not math.isfinite(float(create_time))
                or float(create_time) <= 0
                or isinstance(pgid, bool)
                or not isinstance(pgid, int)
                or pgid != pid
                or process.get("cwd") != str(owner.repo_root)
                or process.get("label") != spec.process_label
            ):
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adopted process identity drifted"
                )
            if observed_status is not None and (
                not isinstance(observed_status, Mapping)
                or set(observed_status) != OBSERVED_STATUS_FIELDS
                or observed_status.get("path") != str(spec.status_path.relative_to(owner.repo_root))
                or not _is_sha256(observed_status.get("file_sha256"))
                or not _is_sha256(observed_status.get("status_sha256"))
                or not isinstance(observed_status.get("state"), str)
                or observed_status["state"] not in LEGACY_OBSERVED_STATUS_STATES
            ):
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} observed status authority drifted"
                )
            identity_sha = canonical_sha256(process)[:16]
            if candidate.name != f"{process.get('pid')}-{identity_sha}.json":
                raise SuccessorChainRefused(
                    f"successor chain v5 {spec.stage_id} adoption receipt filename drifted"
                )
            validated_processes.append(process)
        if not validated_processes and row.get("process") is not None:
            raise SuccessorChainRefused(
                f"successor chain v5 {spec.stage_id} has process identity without an adoption receipt"
            )
        if validated_processes and row.get("process") != validated_processes[-1]:
            raise SuccessorChainRefused(f"successor chain v5 {spec.stage_id} adopted process history drifted")


def _read_stable_state_status(
    state_path: Path,
    status_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for attempt in range(SNAPSHOT_ATTEMPTS):
        before = _read_json(state_path, "successor chain v5 state")
        first_status = _read_json(status_path, "successor chain v5 status")
        after = _read_json(state_path, "successor chain v5 state")
        second_status = _read_json(status_path, "successor chain v5 status")
        try:
            projected = _status_payload(before)
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
    raise SuccessorChainRefused("successor chain v5 state/status snapshot did not become coherent")


def _validate_bound_artifact_hashes(
    repo_root: Path,
    capsules: Mapping[str, Any],
) -> None:
    for capsule_id, row in capsules.items():
        if not isinstance(row, Mapping):
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} is not an object")
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list):
            raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} artifacts drifted")
        for report in artifacts:
            if not isinstance(report, Mapping):
                raise SuccessorChainRefused(
                    f"successor chain v5 capsule {capsule_id} artifact report drifted"
                )
            raw_path = report.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise SuccessorChainRefused(f"successor chain v5 capsule {capsule_id} artifact path drifted")
            relative = Path(raw_path)
            candidate = repo_root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                raise SuccessorChainRefused(
                    f"successor chain v5 capsule {capsule_id} artifact path is unsafe"
                )
            path = candidate.resolve()
            if not path.is_relative_to(repo_root) or report.get("sha256") != sha256_file(path):
                raise SuccessorChainRefused(
                    f"successor chain v5 capsule {capsule_id} final artifact hash drifted"
                )


def read_validated_complete_chain_status(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path = REPO_ROOT,
    horizon_program_path: Path | None = None,
    specs: Sequence[LegacySpec] | None = None,
    acquire_lock: bool = True,
) -> dict[str, Any]:

    root = root.resolve()
    repo_root = repo_root.resolve()
    if not root.is_dir() or not root.is_relative_to(repo_root):
        raise SuccessorChainRefused("successor chain v5 root is absent or outside the repository")
    if acquire_lock:
        for attempt in range(LOCK_ATTEMPTS):
            try:
                with FileLock(root / LOCK_FILE):
                    return read_validated_complete_chain_status(
                        root=root,
                        repo_root=repo_root,
                        horizon_program_path=horizon_program_path,
                        specs=specs,
                        acquire_lock=False,
                    )
            except Generation1Refused as exc:
                if not isinstance(exc.__cause__, BlockingIOError):
                    raise
                if attempt + 1 < LOCK_ATTEMPTS:
                    time.sleep(LOCK_INTERVAL_SECONDS)
                    continue
                raise SuccessorChainRefused(
                    "successor chain v5 lifetime lock remained held while validating a complete snapshot"
                ) from exc
        raise SuccessorChainRefused("successor chain v5 lifetime lock acquisition exhausted")
    if horizon_program_path is None:
        relative = DEFAULT_HORIZON_PROGRAM.resolve().relative_to(REPO_ROOT.resolve())
        horizon_program_path = repo_root / relative
    horizon_program_path = horizon_program_path.resolve()
    state_path = root / STATE_FILE
    status_path = root / STATUS_FILE
    stable_state, stable_status = _read_stable_state_status(state_path, status_path)
    _validate_seal(stable_status, "status_sha256", "successor chain v5 status")
    configured_specs = tuple(specs) if specs is not None else legacy_specs(repo_root)
    owner = SuccessorEvidenceChain(
        root=root,
        repo_root=repo_root,
        horizon_program_path=horizon_program_path,
        specs=configured_specs,
        execute=True,
        process_table_fn=lambda: (),
    )
    _validate_complete_state_shape(owner, stable_state)
    validated_state = validate_chain_status(
        stable_status,
        repo_root=repo_root,
        horizon_program_path=horizon_program_path,
        specs=configured_specs,
    )
    if validated_state != "complete":
        raise SuccessorChainRefused("successor chain v5 durable status is not complete")
    state_core = copy.deepcopy(dict(stable_state))
    state_core.pop("state_sha256", None)
    owner.state = state_core
    before_capsules = copy.deepcopy(owner.state["capsules"])
    owner._validate_completed_chain_artifacts(())
    if owner.state["capsules"] != before_capsules:
        raise SuccessorChainRefused(
            "successor chain v5 evidence required reconciliation during terminal replay"
        )
    current_state, current_status = _read_stable_state_status(state_path, status_path)
    if current_state != stable_state or current_status != stable_status:
        raise SuccessorChainRefused("successor chain v5 state/status changed during terminal replay")
    owner._validate_completed_chain_artifacts(())
    if owner.state["capsules"] != before_capsules:
        raise SuccessorChainRefused("successor chain v5 evidence required reconciliation during final replay")
    final_state, final_status = _read_stable_state_status(state_path, status_path)
    if final_state != stable_state or final_status != stable_status:
        raise SuccessorChainRefused("successor chain v5 state/status changed during final evidence replay")
    _validate_bound_artifact_hashes(
        repo_root,
        stable_state["capsules"],
    )
    return stable_status


def read_chain_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = root.resolve() / STATUS_FILE
    status = _read_json(path, "successor chain v5 status")
    _validate_seal(status, "status_sha256", "successor chain v5 status")
    if status.get("schema") != STATUS_SCHEMA or status.get("program_id") != CHAIN_ID:
        raise SuccessorChainRefused("successor chain v5 status identity drifted")
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
            raise SuccessorChainRefused("visible successor chain v5 parent has inexact metadata")
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise SuccessorChainRefused("multiple successor chain v5 parent processes are visible")
    if labelled:
        return labelled[0]
    return next((process for process in candidates if process.pid == process.pgid), candidates[0])


def _revalidate_terminal_chain(root: Path) -> dict[str, Any]:
    with FileLock(root / LOCK_FILE):
        owner = SuccessorEvidenceChain(
            root=root,
            repo_root=REPO_ROOT,
            horizon_program_path=DEFAULT_HORIZON_PROGRAM,
            execute=True,
        )
        if owner.state.get("status") not in TERMINAL_STATES:
            raise SuccessorChainRefused(
                "sealed terminal status disagrees with the durable successor chain v5 state"
            )
        owner.state["supervisor"] = owner._identity()
        return owner.tick()


def start_chain_detached(
    *,
    root: Path = DEFAULT_ROOT,
    execute: bool,
    use_caffeinate: bool = True,
) -> dict[str, Any]:
    if not execute:
        raise SuccessorChainRefused("detached successor chain v5 start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_generation1_successor_chain_v5.py").resolve()
        status_path = root / STATUS_FILE
        status = None
        if status_path.is_file():
            status = read_chain_status(root)
        visible = _visible_parent_process(entrypoint)
        status_identity = status.get("supervisor") if status is not None else None
        if _process_identity_alive(status_identity):
            if visible is None:
                raise SuccessorChainRefused("sealed successor chain v5 identity has no exact visible parent")
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
                raise SuccessorChainRefused(
                    "sealed successor chain v5 identity conflicts with visible parent"
                )
            if status is None:
                raise SuccessorChainRefused("live successor chain v5 identity lacks a sealed status")
            try:
                validate_chain_status(
                    status,
                    repo_root=REPO_ROOT,
                    horizon_program_path=DEFAULT_HORIZON_PROGRAM,
                    specs=legacy_specs(REPO_ROOT),
                )
            except (SuccessorChainRefused, OSError, ValueError):
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
                "status": _revalidate_terminal_chain(root),
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
                    validate_chain_status(
                        acknowledged,
                        repo_root=REPO_ROOT,
                        horizon_program_path=DEFAULT_HORIZON_PROGRAM,
                        specs=legacy_specs(REPO_ROOT),
                    )
                    break
                except (SuccessorChainRefused, OSError, ValueError):
                    acknowledged = None
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise SuccessorChainRefused(
                f"detached successor chain v5 did not acknowledge startup; inspect {stderr_path}"
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
                raise SuccessorChainRefused("successor chain v5 execution requires explicit --execute")
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
    "DEFAULT_ROOT",
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
    "read_validated_complete_chain_status",
    "request_stop",
    "start_chain_detached",
    "validate_chain_status",
]


if __name__ == "__main__":
    raise SystemExit(main())
