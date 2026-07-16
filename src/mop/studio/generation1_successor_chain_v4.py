"""Append-only v4 adopter for the Generation 1 successor evidence chain.

Version 4 supersedes the immutable v3 integrity hold.  It preserves v3's
observe-only legacy adoption and unchanged v1 horizon launch while tightening
unlabelled command recognition to the exact executable and argv positions
that the adopter itself launches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from mop.studio import generation1_successor_chain as v3
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

CHAIN_ID = "generation1-successor-evidence-chain-v4"
PARENT_LABEL = "mop-successor-evidence-chain-v4"
STATE_SCHEMA = "mop-generation1-successor-evidence-chain-state/v4"
STATUS_SCHEMA = "mop-generation1-successor-evidence-chain-status/v4"
ADOPTION_SCHEMA = "mop-generation1-legacy-adoption-receipt/v2"
CONTROL_SCHEMA = "mop-generation1-successor-evidence-chain-control/v2"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / CHAIN_ID
DEFAULT_HORIZON_PROGRAM = v3.DEFAULT_HORIZON_PROGRAM
STATE_FILE = v3.STATE_FILE
STATUS_FILE = v3.STATUS_FILE
LOCK_FILE = v3.LOCK_FILE
START_LOCK_FILE = v3.START_LOCK_FILE
CONTROL_FILE = v3.CONTROL_FILE
STARTUP_ACK_SECONDS = v3.STARTUP_ACK_SECONDS
PROCESS_TIME_TOLERANCE_SECONDS = v3.PROCESS_TIME_TOLERANCE_SECONDS
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

LegacySpec = v3.LegacySpec
ProcessSnapshot = v3.ProcessSnapshot
SuccessorChainRefused = v3.SuccessorChainRefused
_default_identity_probe = v3._default_identity_probe
_default_process_table = v3._default_process_table
_iso = v3._iso
_now = v3._now
_parse_time = v3._parse_time
_read_json = v3._read_json
_validate_seal = v3._validate_seal


def legacy_specs(repo_root: Path = REPO_ROOT) -> tuple[LegacySpec, ...]:
    """Return v3 authorities with exact mechanics worker-label families."""

    mechanics_prefixes = tuple(f"mop-{lane.lane_id.lower()}-" for lane in v3.mechanics_queue.LANES)
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


class SuccessorEvidenceChain(v3.SuccessorEvidenceChain):
    """v4 owner with a fresh durable root and exact command-shape adoption."""

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ROOT,
        repo_root: Path = REPO_ROOT,
        horizon_program_path: Path = DEFAULT_HORIZON_PROGRAM,
        specs: Sequence[LegacySpec] | None = None,
        execute: bool = False,
        process_table_fn: v3.ProcessTable = _default_process_table,
        identity_probe_fn: v3.IdentityProbe = _default_identity_probe,
        popen_fn: v3.PopenFactory = subprocess.Popen,
        supervisor_start_fn: v3.SupervisorStarter = v3.start_generation1_detached,
        now_fn: v3.NowFn = _now,
        sleep_fn: Callable[[float], None] = time.sleep,
        poll_seconds: float = v3.POLL_SECONDS,
    ) -> None:
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
        implementation = Path(__file__).resolve()
        base = Path(v3.__file__).resolve()
        return {
            "path": _relative_or_absolute(implementation, self.repo_root),
            "sha256": sha256_file(implementation),
            "inherited_base": {
                "path": _relative_or_absolute(base, self.repo_root),
                "sha256": sha256_file(base),
            },
        }

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
        state = _read_json(self.state_path, "successor chain v4 state")
        _validate_seal(state, "state_sha256", "successor chain v4 state")
        if state.get("schema") != STATE_SCHEMA or state.get("program_id") != CHAIN_ID:
            raise SuccessorChainRefused("successor chain v4 state identity drifted")
        expected_capsules = {spec.stage_id for spec in self.specs} | {"successor_horizon"}
        capsules = state.get("capsules")
        if not isinstance(capsules, dict) or set(capsules) != expected_capsules:
            raise SuccessorChainRefused("successor chain v4 capsule inventory drifted")
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
            "supersedes": "generation1-successor-evidence-chain-v3",
        }
        status = {**status_core, "status_sha256": canonical_sha256(status_core)}
        atomic_write_json(self.status_path, status)
        return status

    def _command_residual(self, spec: LegacySpec, process: ProcessSnapshot) -> bool:
        """Recognize only the exact two-argv restart shape emitted by this parent."""

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

    def _matching_processes(
        self,
        spec: LegacySpec,
        processes: Sequence[ProcessSnapshot],
    ) -> tuple[list[ProcessSnapshot], list[ProcessSnapshot]]:
        """Bind child labels only inside the exact owning parent boundary."""

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
        }
        if path.exists():
            receipt = _read_json(path, f"{spec.stage_id} v4 adoption receipt")
            _validate_seal(receipt, "receipt_sha256", f"{spec.stage_id} v4 adoption receipt")
            if (
                receipt.get("schema") != ADOPTION_SCHEMA
                or receipt.get("chain_id") != CHAIN_ID
                or receipt.get("stage_id") != spec.stage_id
                or receipt.get("program_id") != spec.program_id
                or receipt.get("process") != identity
                or _parse_time(receipt.get("adopted_at")) is None
                or receipt.get("policy") != policy
            ):
                raise SuccessorChainRefused(f"{spec.stage_id} v4 adoption receipt identity drifted")
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
        if not isinstance(problems, list) or any(
            not isinstance(problem, str) or not problem for problem in problems
        ):
            raise SuccessorChainRefused("successor horizon status problems inventory drifted")
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
        if self.state.get("status") != "complete":
            return super().tick()
        try:
            self._validate_bound_authorities(self.state)
            self._validate_completed_chain_artifacts(tuple(self.process_table_fn()))
            return self._publish()
        except (
            SuccessorChainRefused,
            Generation1Refused,
            OSError,
            RuntimeError,
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
        value = _read_json(path, "successor chain v4 control")
        _validate_seal(value, "control_sha256", "successor chain v4 control")
        if (
            value.get("schema") != CONTROL_SCHEMA
            or value.get("program_id") != CHAIN_ID
            or value.get("action") != "drain"
        ):
            raise SuccessorChainRefused("successor chain v4 control identity drifted")
        return True


def read_chain_status(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = root.resolve() / STATUS_FILE
    status = _read_json(path, "successor chain v4 status")
    _validate_seal(status, "status_sha256", "successor chain v4 status")
    if status.get("schema") != STATUS_SCHEMA or status.get("program_id") != CHAIN_ID:
        raise SuccessorChainRefused("successor chain v4 status identity drifted")
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
            raise SuccessorChainRefused("visible successor chain v4 parent has inexact metadata")
    labelled = [process for process in candidates if process.label == PARENT_LABEL]
    groups = {process.pgid for process in candidates}
    if len(labelled) > 1 or len(groups) > 1:
        raise SuccessorChainRefused("multiple successor chain v4 parent processes are visible")
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
                "sealed terminal status disagrees with the durable successor chain v4 state"
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
        raise SuccessorChainRefused("detached successor chain v4 start requires explicit --execute")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / START_LOCK_FILE):
        entrypoint = (REPO_ROOT / "scripts/mop_generation1_successor_chain_v4.py").resolve()
        status_path = root / STATUS_FILE
        status = None
        if status_path.is_file():
            status = read_chain_status(root)
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
                    "sealed successor chain v4 identity conflicts with visible parent"
                )
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
                    break
                except (SuccessorChainRefused, OSError, ValueError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise SuccessorChainRefused(
                f"detached successor chain v4 did not acknowledge startup; inspect {stderr_path}"
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
                raise SuccessorChainRefused("successor chain v4 execution requires explicit --execute")
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
    "request_stop",
    "start_chain_detached",
]


if __name__ == "__main__":
    raise SystemExit(main())
