"""Host-aware local execution throttle for long, resumable experiment legs.

The throttle owns only processes that it launches. It observes user applications, including
Blender, but never signals them. One heavy lane is the invariant. A second lane may only be a
declared CPU, network, or light task, and only after stricter telemetry gates pass.

P4 and P5 keep their hashed three-hour scientific configs. Their exit code 2 is a durable resume
boundary, so this supervisor may repeat the exact command inside one five-hour operational leg.
"""

from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from ..config import REPO_ROOT
from .profiles import get_profile

POLICY_SCHEMA = "mop-local-execution-throttle-policy/v1"
TELEMETRY_SCHEMA = "mop-local-host-telemetry/v1"
DECISION_SCHEMA = "mop-local-throttle-decision/v1"
RECEIPT_SCHEMA = "mop-local-throttle-receipt/v1"
REGISTRY_SCHEMA = "mop-local-throttle-active-registry/v1"
LANES = frozenset({"heavy", "cpu", "network", "light"})
ACCELERATORS = frozenset({"none", "mps"})
SECOND_LANES = frozenset({"cpu", "network", "light"})
DEFAULT_POLICY = REPO_ROOT / "configs/local_execution_throttle.yaml"
DEFAULT_STATE_ROOT = REPO_ROOT / "runs/local_throttle"
IMPLEMENTATION_PATH = Path(__file__).resolve()
P6_PREFLIGHT = REPO_ROOT / "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json"
P6_RUN_CONFIG = REPO_ROOT / "configs/experiment/continual_million_event_rungs.yaml"


class ThrottleRefused(RuntimeError):
    """Fail-closed configuration or admission refusal."""


@dataclass(frozen=True)
class ReceiptRequirement:
    path: str
    schema: str
    fields: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ReceiptRequirement:
        fields = raw.get("fields")
        return cls(
            path=str(raw.get("path", "")),
            schema=str(raw.get("schema", "")),
            fields=(
                tuple((str(key), value) for key, value in fields.items()) if isinstance(fields, dict) else ()
            ),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts:
            problems.append("receipt path must be a nonempty repository-relative path")
        if not self.schema:
            problems.append("receipt schema must be nonempty")
        if not self.fields:
            problems.append("receipt fields must be a nonempty mapping")
        return problems


@dataclass(frozen=True)
class TaskDeclaration:
    task_id: str
    lane: str
    accelerator: str
    cpu_cores: int
    estimated_unified_memory_gb: float | None
    estimated_mps_gb: float
    resource_basis: str
    forecast_write_gb: float
    atomic_write_gb: float
    wall_minutes: int
    pause_safe: bool
    atomic_checkpoints: bool
    checkpoint_globs: tuple[str, ...]
    restart_exit_codes: tuple[int, ...]
    command: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    prerequisites: tuple[ReceiptRequirement, ...] = ()
    resource_probe: bool = False
    requires_empty_lanes: bool = False
    resource_receipt_path: str | None = None
    resource_receipt_schema: str | None = None
    resource_receipt_rung: int | None = None

    @classmethod
    def from_mapping(cls, task_id: str, raw: dict[str, Any]) -> TaskDeclaration:
        memory_value = raw.get("estimated_unified_memory_gb")
        prerequisites_raw = raw.get("prerequisites", [])
        return cls(
            task_id=task_id,
            lane=str(raw.get("lane", "")),
            accelerator=str(raw.get("accelerator", "none")),
            cpu_cores=int(raw.get("cpu_cores", 0)),
            estimated_unified_memory_gb=(None if memory_value is None else float(memory_value)),
            estimated_mps_gb=float(raw.get("estimated_mps_gb", -1)),
            resource_basis=str(raw.get("resource_basis", "")),
            forecast_write_gb=float(raw.get("forecast_write_gb", -1)),
            atomic_write_gb=float(raw.get("atomic_write_gb", -1)),
            wall_minutes=int(raw.get("wall_minutes", 0)),
            pause_safe=bool(raw.get("pause_safe", False)),
            atomic_checkpoints=bool(raw.get("atomic_checkpoints", False)),
            checkpoint_globs=tuple(str(value) for value in raw.get("checkpoint_globs", [])),
            restart_exit_codes=tuple(int(value) for value in raw.get("restart_exit_codes", [])),
            command=tuple(str(value) for value in raw.get("command", [])),
            depends_on=tuple(str(value) for value in raw.get("depends_on", [])),
            prerequisites=tuple(
                ReceiptRequirement.from_mapping(value)
                for value in prerequisites_raw
                if isinstance(value, dict)
            ),
            resource_probe=bool(raw.get("resource_probe", False)),
            requires_empty_lanes=bool(raw.get("requires_empty_lanes", False)),
            resource_receipt_path=(
                str(raw["resource_receipt_path"]) if raw.get("resource_receipt_path") is not None else None
            ),
            resource_receipt_schema=(
                str(raw["resource_receipt_schema"])
                if raw.get("resource_receipt_schema") is not None
                else None
            ),
            resource_receipt_rung=(
                int(raw["resource_receipt_rung"]) if raw.get("resource_receipt_rung") is not None else None
            ),
        )

    def validate(self, hard_wall_minutes: int) -> list[str]:
        problems: list[str] = []
        if not self.task_id:
            problems.append("task id is empty")
        if self.lane not in LANES:
            problems.append(f"lane {self.lane!r} is not one of {sorted(LANES)}")
        if self.accelerator not in ACCELERATORS:
            problems.append(f"accelerator {self.accelerator!r} is not one of {sorted(ACCELERATORS)}")
        if self.cpu_cores < 1:
            problems.append("cpu_cores must be at least one")
        for field in ("estimated_mps_gb", "forecast_write_gb", "atomic_write_gb"):
            if not math.isfinite(float(getattr(self, field))) or float(getattr(self, field)) < 0:
                problems.append(f"{field} must be finite and nonnegative")
        if self.estimated_unified_memory_gb is not None and (
            not math.isfinite(self.estimated_unified_memory_gb) or self.estimated_unified_memory_gb < 0
        ):
            problems.append("estimated_unified_memory_gb must be finite and nonnegative")
        if self.estimated_unified_memory_gb is None and not (
            self.resource_probe or self.resource_receipt_path
        ):
            problems.append(
                "unmeasured memory is allowed only for an exclusive resource probe or measured receipt"
            )
        if self.accelerator == "none" and self.estimated_mps_gb != 0:
            problems.append("a non-MPS task must declare estimated_mps_gb=0")
        if not self.resource_basis:
            problems.append("resource_basis must explain the declaration")
        if not (1 <= self.wall_minutes <= hard_wall_minutes):
            problems.append(f"wall_minutes must be in [1, {hard_wall_minutes}]")
        if not self.command or any(not part for part in self.command):
            problems.append("command must be a non-empty argv list")
        if self.pause_safe and (not self.atomic_checkpoints or not self.checkpoint_globs):
            problems.append("pause_safe tasks require atomic_checkpoints and checkpoint_globs")
        if self.lane == "heavy" and not self.pause_safe:
            problems.append("heavy tasks must be pause_safe with an atomic resume authority")
        if len(self.restart_exit_codes) != len(set(self.restart_exit_codes)):
            problems.append("restart_exit_codes must be unique")
        if len(self.depends_on) != len(set(self.depends_on)):
            problems.append("depends_on task ids must be unique")
        for requirement in self.prerequisites:
            problems.extend(
                f"prerequisite {requirement.path}: {problem}" for problem in requirement.validate()
            )
        if self.resource_probe:
            if not self.requires_empty_lanes:
                problems.append("an unmeasured resource probe must require an empty scheduler")
            if self.resource_receipt_path is not None:
                problems.append("a resource probe cannot consume a prior resource receipt")
        receipt_fields = (
            self.resource_receipt_path,
            self.resource_receipt_schema,
            self.resource_receipt_rung,
        )
        if any(value is not None for value in receipt_fields) and not all(
            value is not None for value in receipt_fields
        ):
            problems.append("resource receipt path, schema, and rung must be declared together")
        if self.resource_receipt_path is not None:
            path = Path(self.resource_receipt_path)
            if path.is_absolute() or ".." in path.parts:
                problems.append("resource receipt path must be repository-relative")
        return problems


@dataclass(frozen=True)
class ThrottlePolicy:
    path: Path
    profile_name: str
    limits: dict[str, Any]
    monitor: dict[str, Any]
    thresholds: dict[str, dict[str, float]]
    tasks: dict[str, TaskDeclaration]
    execution_order: dict[str, tuple[str, ...]]
    sha256: str

    def task(self, task_id: str) -> TaskDeclaration:
        if task_id not in self.tasks:
            raise ThrottleRefused(f"unknown task {task_id!r}; choose from {sorted(self.tasks)}")
        return self.tasks[task_id]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _p6_resource_evidence() -> dict[str, Any]:
    """Read the immutable 384-event receipt used to derive P6 disk declarations."""

    config = yaml.safe_load(P6_RUN_CONFIG.read_text())
    if not isinstance(config, dict) or config.get("schema") != "mop-continual-progressive-rungs-config/v1":
        raise ThrottleRefused("P6 progressive run config schema drift")
    source = config.get("source_preflight")
    if not isinstance(source, dict):
        raise ThrottleRefused("P6 progressive run config lacks source_preflight")
    if _sha256_file(P6_PREFLIGHT) != source.get("file_sha256"):
        raise ThrottleRefused("P6 384-event preflight file hash drift")
    receipt = json.loads(P6_PREFLIGHT.read_text())
    if not isinstance(receipt, dict) or receipt.get("schema") != "mop-continual-million-event-preflight/v1":
        raise ThrottleRefused("P6 384-event preflight schema drift")
    without_digest = dict(receipt)
    declared_digest = without_digest.pop("payload_sha256", None)
    if _canonical_sha256(without_digest) != declared_digest or declared_digest != source.get(
        "payload_sha256"
    ):
        raise ThrottleRefused("P6 384-event preflight payload digest drift")
    resources = [
        arm.get("result", {}).get("metrics", {}).get("resources", {})
        for schedule in receipt.get("schedules", [])
        if isinstance(schedule, dict)
        for arm in schedule.get("arms", [])
        if isinstance(arm, dict)
    ]
    if not resources:
        raise ThrottleRefused("P6 384-event preflight has no resource rows")
    observed = {
        "events": int(receipt.get("resource_envelope", {}).get("configured_stream_events", 0)),
        "stream_bytes": max(int(row.get("stream_disk_bytes", 0)) for row in resources),
        "state_bytes": max(int(row.get("checkpoint_state_bytes", 0)) for row in resources),
    }
    expected = {
        "events": int(source.get("observed_events_per_stream", 0)),
        "stream_bytes": int(source.get("observed_stream_disk_bytes", 0)),
        "state_bytes": int(source.get("observed_max_checkpoint_state_bytes", 0)),
    }
    if observed != expected or min(observed.values()) <= 0:
        raise ThrottleRefused(f"P6 resource evidence drift: {observed} != {expected}")
    projection = config.get("resource_projection")
    multiplier = float(projection.get("evidence_multiplier", 0.0)) if isinstance(projection, dict) else 0.0
    if multiplier < 1.0:
        raise ThrottleRefused("P6 resource evidence multiplier must be at least one")
    return {
        **observed,
        "multiplier": multiplier,
        "preflight_file_sha256": _sha256_file(P6_PREFLIGHT),
        "preflight_payload_sha256": declared_digest,
        "config_sha256": _sha256_file(P6_RUN_CONFIG),
    }


def _p6_write_projection(
    evidence: dict[str, Any],
    *,
    rung: int,
    seeds: int,
    schedules: int,
    arms: int,
) -> tuple[float, float]:
    per_stream = math.ceil(int(evidence["stream_bytes"]) * rung / int(evidence["events"]))
    forecast_bytes = (
        per_stream * seeds * schedules + int(evidence["state_bytes"]) * seeds * schedules * arms
    ) * float(evidence["multiplier"])
    atomic_bytes = int(evidence["state_bytes"]) * float(evidence["multiplier"])
    return forecast_bytes / 1e9, atomic_bytes / 1e9


def _p6_checkpoint_globs(root: str, proof: str, *, probe: bool = False) -> tuple[str, ...]:
    if probe:
        return (
            f"{root}/streams/seed_20260710/abrupt/chunk_*.bin",
            f"{root}/streams/seed_20260710/abrupt/manifest.json",
            f"{root}/checkpoints/seed_20260710/abrupt/replay.json",
            f"{root}/progress.json",
            proof,
        )
    return (
        f"{root}/streams/seed_*/abrupt/chunk_*.bin",
        f"{root}/streams/seed_*/abrupt/manifest.json",
        f"{root}/streams/seed_*/gradual/chunk_*.bin",
        f"{root}/streams/seed_*/gradual/manifest.json",
        f"{root}/checkpoints/seed_*/abrupt/replay.json",
        f"{root}/checkpoints/seed_*/abrupt/no-replay.json",
        f"{root}/checkpoints/seed_*/abrupt/fresh-init.json",
        f"{root}/checkpoints/seed_*/gradual/replay.json",
        f"{root}/checkpoints/seed_*/gradual/no-replay.json",
        f"{root}/checkpoints/seed_*/gradual/fresh-init.json",
        f"{root}/progress.json",
        proof,
    )


def load_policy(path: Path | str = DEFAULT_POLICY) -> ThrottlePolicy:
    policy_path = Path(path).resolve()
    raw = yaml.safe_load(policy_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != POLICY_SCHEMA:
        raise ThrottleRefused(f"policy schema must be {POLICY_SCHEMA}")
    profile_name = str(raw.get("profile", ""))
    profile = get_profile(profile_name)
    limits = dict(raw.get("limits") or {})
    monitor = dict(raw.get("monitor") or {})
    thresholds = dict(raw.get("thresholds") or {})
    hard_wall = int(limits.get("hard_wall_minutes", 0))
    problems: list[str] = []
    if hard_wall != profile.max_wall_min:
        problems.append(f"hard wall {hard_wall} does not equal profile max_wall_min {profile.max_wall_min}")
    if float(limits.get("disk_floor_gb", -1)) != profile.min_free_disk_gb:
        problems.append(f"disk floor must equal profile floor {profile.min_free_disk_gb:g} GB")
    if int(limits.get("maximum_heavy_lanes", -1)) != 1:
        problems.append("maximum_heavy_lanes must remain one")
    if int(limits.get("maximum_active_lanes", -1)) != 2:
        problems.append("maximum_active_lanes must remain two")
    for name in ("first_lane", "second_lane"):
        if not isinstance(thresholds.get(name), dict):
            problems.append(f"thresholds.{name} must be a mapping")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, dict) or not tasks_raw:
        problems.append("tasks must be a non-empty mapping")
        tasks_raw = {}
    tasks: dict[str, TaskDeclaration] = {}
    for task_id, value in tasks_raw.items():
        if not isinstance(value, dict):
            problems.append(f"task {task_id}: declaration is not a mapping")
            continue
        if not isinstance(value.get("depends_on", []), list):
            problems.append(f"task {task_id}: depends_on must be a list")
            continue
        prerequisites_value = value.get("prerequisites", [])
        if not isinstance(prerequisites_value, list) or any(
            not isinstance(requirement, dict) for requirement in prerequisites_value
        ):
            problems.append(f"task {task_id}: prerequisites must be a list of mappings")
            continue
        task = TaskDeclaration.from_mapping(str(task_id), value)
        tasks[task.task_id] = task
        problems.extend(f"task {task.task_id}: {problem}" for problem in task.validate(hard_wall))
    order_raw = raw.get("execution_order") or {}
    if not isinstance(order_raw, dict):
        problems.append("execution_order must be a mapping")
        order_raw = {}
    execution_order = {
        str(key): tuple(str(value) for value in values)
        for key, values in order_raw.items()
        if isinstance(values, list)
    }
    expected_p5_order = ("p5smoke_cpu", "p5_traingrid_memory_probe_cpu", "p5pilot_cpu")
    if execution_order.get("p5_cpu") != expected_p5_order:
        problems.append(f"execution_order.p5_cpu must be {list(expected_p5_order)}")
    expected_p6_order = (
        "p6_10k_resource_probe_cpu",
        "p6_10k_replication_cpu",
        "p6_100k_replication_cpu",
        "p6_1m_replication_cpu",
    )
    if execution_order.get("p6_cpu") != expected_p6_order:
        problems.append(f"execution_order.p6_cpu must be {list(expected_p6_order)}")
    expected_p6_dependencies = {
        expected_p6_order[0]: (),
        expected_p6_order[1]: (expected_p6_order[0],),
        expected_p6_order[2]: (expected_p6_order[1],),
        expected_p6_order[3]: (expected_p6_order[2],),
    }
    for sequence, task_ids in execution_order.items():
        missing_tasks = [task_id for task_id in task_ids if task_id not in tasks]
        if missing_tasks:
            problems.append(f"execution_order.{sequence} names missing tasks {missing_tasks}")
    for task_id, task in tasks.items():
        for dependency in task.depends_on:
            if dependency == task_id:
                problems.append(f"task {task_id}: cannot depend on itself")
                continue
            if dependency not in tasks:
                problems.append(f"task {task_id}: dependency {dependency!r} is missing")
                continue
            ordered = any(
                dependency in sequence
                and task_id in sequence
                and sequence.index(dependency) < sequence.index(task_id)
                for sequence in execution_order.values()
            )
            if not ordered:
                problems.append(
                    f"task {task_id}: dependency {dependency!r} must precede it in one execution_order"
                )
    for task_id, expected_dependencies in expected_p6_dependencies.items():
        p6_task = tasks.get(task_id)
        if p6_task is not None and p6_task.depends_on != expected_dependencies:
            problems.append(f"task {task_id}: P6 dependency chain drifted")
    expected_commands: dict[str, tuple[str, ...]] = {
        "p4_resume_cpu": (
            ".venv/bin/python",
            "scripts/p4_capability_density.py",
            "--profile",
            "p4screen",
            "--device",
            "cpu",
            "--run-dir",
            "runs/p4_screen/p4screen",
            "--out",
            "proof/P4_CAPABILITY_DENSITY_SCREEN.json",
        ),
        "p4_resume_mps": (
            ".venv/bin/python",
            "scripts/p4_capability_density.py",
            "--profile",
            "p4screen",
            "--device",
            "mps",
            "--run-dir",
            "runs/p4_screen/p4screen_mps_clean",
            "--out",
            "proof/P4_CAPABILITY_DENSITY_SCREEN_MPS_CLEAN.json",
        ),
        "p5smoke_cpu": (
            ".venv/bin/python",
            "scripts/p5_context_capability.py",
            "--profile",
            "p5smoke",
            "--device",
            "cpu",
            "--run-dir",
            "runs/p5_context/p5smoke",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
        ),
        "p5_traingrid_memory_probe_cpu": (
            ".venv/bin/python",
            "scripts/p5_traingrid_memory_probe.py",
            "--out",
            "proof/P5_TRAINGRID_MEMORY_TRACE.json",
            "--repeats",
            "3",
            "--seed",
            "0",
        ),
        "p5pilot_cpu": (
            ".venv/bin/python",
            "scripts/p5_context_capability.py",
            "--profile",
            "p5pilot",
            "--device",
            "cpu",
            "--run-dir",
            "runs/p5_context/p5pilot",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
        ),
        "p6_10k_resource_probe_cpu": (
            ".venv/bin/python",
            "scripts/continual_million_event_rung.py",
            "--config",
            "configs/experiment/continual_million_event_rungs.yaml",
            "--rung",
            "10000",
            "--work-root",
            "runs/continual_million_event/rung_010000_probe",
            "--out",
            "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
            "--resource-probe",
            "--seed-count",
            "1",
            "--schedules",
            "abrupt",
            "--arms",
            "replay",
        ),
        "p6_10k_replication_cpu": (
            ".venv/bin/python",
            "scripts/continual_million_event_rung.py",
            "--config",
            "configs/experiment/continual_million_event_rungs.yaml",
            "--rung",
            "10000",
            "--work-root",
            "runs/continual_million_event/rung_010000",
            "--out",
            "proof/P6_CONTINUAL_10K.json",
        ),
        "p6_100k_replication_cpu": (
            ".venv/bin/python",
            "scripts/continual_million_event_rung.py",
            "--config",
            "configs/experiment/continual_million_event_rungs.yaml",
            "--rung",
            "100000",
            "--work-root",
            "runs/continual_million_event/rung_100000",
            "--out",
            "proof/P6_CONTINUAL_100K.json",
        ),
        "p6_1m_replication_cpu": (
            ".venv/bin/python",
            "scripts/continual_million_event_rung.py",
            "--config",
            "configs/experiment/continual_million_event_rungs.yaml",
            "--rung",
            "1000000",
            "--work-root",
            "runs/continual_million_event/rung_1000000",
            "--out",
            "proof/P6_CONTINUAL_1M.json",
        ),
    }
    for task_id, expected in expected_commands.items():
        canonical_task = tasks.get(task_id)
        if canonical_task is None:
            problems.append(f"required canonical task {task_id} is missing")
        elif canonical_task.command != expected:
            problems.append(f"task {task_id}: canonical command drifted")
    for task_id in ("p4_resume_cpu", "p4_resume_mps", "p5smoke_cpu", "p5pilot_cpu"):
        canonical_task = tasks.get(task_id)
        if canonical_task is not None and canonical_task.restart_exit_codes != (2,):
            problems.append(f"task {task_id}: exit code 2 must remain the resume boundary")
    try:
        p6_evidence = _p6_resource_evidence()
    except (OSError, ValueError, json.JSONDecodeError, ThrottleRefused) as exc:
        problems.append(f"P6 resource evidence invalid: {exc}")
        p6_evidence = None
    p6_projection_inputs = {
        "p6_10k_resource_probe_cpu": (10_000, 1, 1, 1),
        "p6_10k_replication_cpu": (10_000, 5, 2, 3),
        "p6_100k_replication_cpu": (100_000, 5, 2, 3),
        "p6_1m_replication_cpu": (1_000_000, 5, 2, 3),
    }
    p6_expected_globs = {
        "p6_10k_resource_probe_cpu": _p6_checkpoint_globs(
            "runs/continual_million_event/rung_010000_probe",
            "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
            probe=True,
        ),
        "p6_10k_replication_cpu": _p6_checkpoint_globs(
            "runs/continual_million_event/rung_010000",
            "proof/P6_CONTINUAL_10K.json",
        ),
        "p6_100k_replication_cpu": _p6_checkpoint_globs(
            "runs/continual_million_event/rung_100000",
            "proof/P6_CONTINUAL_100K.json",
        ),
        "p6_1m_replication_cpu": _p6_checkpoint_globs(
            "runs/continual_million_event/rung_1000000",
            "proof/P6_CONTINUAL_1M.json",
        ),
    }
    if p6_evidence is not None:
        runner_sha256 = _sha256_file(REPO_ROOT / "scripts/continual_million_event_rung.py")
        for task_id, (rung, seeds, schedules, arms) in p6_projection_inputs.items():
            p6_task = tasks.get(task_id)
            if p6_task is None:
                continue
            expected_forecast, expected_atomic = _p6_write_projection(
                p6_evidence,
                rung=rung,
                seeds=seeds,
                schedules=schedules,
                arms=arms,
            )
            if not math.isclose(p6_task.forecast_write_gb, expected_forecast, abs_tol=1e-12):
                problems.append(
                    f"task {task_id}: forecast_write_gb is not derived from the 384-event receipt"
                )
            if not math.isclose(p6_task.atomic_write_gb, expected_atomic, abs_tol=1e-12):
                problems.append(f"task {task_id}: atomic_write_gb is not derived from the 384-event receipt")
            if p6_task.estimated_unified_memory_gb is not None:
                problems.append(f"task {task_id}: P6 memory must come from a measured rung receipt")
            if p6_task.lane != "cpu" or p6_task.cpu_cores != 1 or not p6_task.requires_empty_lanes:
                problems.append(f"task {task_id}: P6 must be an exclusive single-core CPU lane")
            if str(p6_evidence["preflight_payload_sha256"]) not in p6_task.resource_basis:
                problems.append(f"task {task_id}: resource_basis must bind the 384-event payload")
            if p6_task.checkpoint_globs != p6_expected_globs[task_id]:
                problems.append(f"task {task_id}: atomic checkpoint globs drifted")
            if task_id != "p6_10k_resource_probe_cpu":
                expected_identity_fields = {
                    "identity.config_sha256": p6_evidence["config_sha256"],
                    "identity.runner_sha256": runner_sha256,
                    "identity.source_preflight_file_sha256": p6_evidence["preflight_file_sha256"],
                }
                if not any(
                    all(
                        dict(requirement.fields).get(field) == expected
                        for field, expected in expected_identity_fields.items()
                    )
                    for requirement in p6_task.prerequisites
                ):
                    problems.append(
                        f"task {task_id}: prerequisite must bind runner, config, and source hashes"
                    )
    if (
        tasks.get("p6_10k_resource_probe_cpu") is not None
        and not tasks["p6_10k_resource_probe_cpu"].resource_probe
    ):
        problems.append("P6 10k resource probe must declare resource_probe=true")
    for task_id in (
        "p6_10k_replication_cpu",
        "p6_100k_replication_cpu",
        "p6_1m_replication_cpu",
    ):
        dependent_task = tasks.get(task_id)
        if dependent_task is not None and (
            dependent_task.resource_probe
            or not dependent_task.resource_receipt_path
            or not dependent_task.resource_receipt_schema
            or dependent_task.resource_receipt_rung is None
        ):
            problems.append(f"task {task_id}: prior-rung resource receipt is required")
    p6_expected_resource_receipts = {
        "p6_10k_replication_cpu": ("proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json", 10_000),
        "p6_100k_replication_cpu": ("proof/P6_CONTINUAL_10K.json", 10_000),
        "p6_1m_replication_cpu": ("proof/P6_CONTINUAL_100K.json", 100_000),
    }
    for task_id, (path, rung) in p6_expected_resource_receipts.items():
        dependent_task = tasks.get(task_id)
        if dependent_task is not None and (
            dependent_task.resource_receipt_path != path
            or dependent_task.resource_receipt_schema != "mop-continual-progressive-rung/v1"
            or dependent_task.resource_receipt_rung != rung
        ):
            problems.append(f"task {task_id}: resource receipt dependency drifted")
    if problems:
        raise ThrottleRefused("; ".join(problems))
    return ThrottlePolicy(
        path=policy_path,
        profile_name=profile_name,
        limits=limits,
        monitor=monitor,
        thresholds={key: {str(k): float(v) for k, v in value.items()} for key, value in thresholds.items()},
        tasks=tasks,
        execution_order=execution_order,
        sha256=_sha256_file(policy_path),
    )


def _capture(command: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _memory_pressure() -> dict[str, Any]:
    ok, output = _capture(["memory_pressure", "-Q"])
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%", output)
    return {
        "available": bool(ok and match),
        "free_percent": float(match.group(1)) if match else None,
        "raw": output,
    }


def _thermal() -> dict[str, Any]:
    ok, output = _capture(["pmset", "-g", "therm"])
    normal_text = (
        "No thermal warning level has been recorded" in output
        and "No performance warning level has been recorded" in output
    )
    limits = [int(value) for value in re.findall(r"(?:CPU|GPU)_Speed_Limit\s*=\s*([0-9]+)", output)]
    if ok and (normal_text or (limits and min(limits) >= 100)):
        status = "normal"
    elif ok and limits:
        status = "degraded"
    else:
        status = "unknown"
    return {"available": bool(ok and status != "unknown"), "status": status, "raw": output}


def _power() -> dict[str, Any]:
    ok, output = _capture(["pmset", "-g", "batt"])
    source_match = re.search(r"Now drawing from '([^']+)'", output)
    percent_match = re.search(r"\b([0-9]{1,3})%;", output)
    source = source_match.group(1) if source_match else None
    return {
        "available": bool(ok and source),
        "source": source,
        "on_ac": bool(source and "AC" in source.upper()),
        "battery_percent": int(percent_match.group(1)) if percent_match else None,
        "raw": output,
    }


def _frontmost_app() -> dict[str, Any]:
    ok, front = _capture(["lsappinfo", "front"])
    if not ok or not front:
        return {"available": False, "name": None}
    info_ok, info = _capture(["lsappinfo", "info", "-only", "name", front])
    match = re.search(r'"LSDisplayName"="([^"]+)"', info)
    return {"available": bool(info_ok and match), "name": match.group(1) if match else None}


def _processes(policy: ThrottlePolicy, excluded_pids: set[int]) -> dict[str, Any]:
    foreground_markers = [str(value).lower() for value in policy.monitor.get("foreground_markers", [])]
    heavy_markers = [str(value).lower() for value in policy.monitor.get("known_heavy_markers", [])]
    foreground: list[dict[str, Any]] = []
    unmanaged_heavy: list[dict[str, Any]] = []
    inaccessible = 0
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "username"]):
        try:
            info = process.info
            pid = int(info["pid"])
            if pid in excluded_pids or pid in {os.getpid(), os.getppid()}:
                continue
            argv = [str(value) for value in (info.get("cmdline") or [])]
            command = " ".join(argv)
            searchable = f"{info.get('name') or ''} {command}".lower()
            # Foreground applications are identified from the executable, never arbitrary
            # arguments. A curl URL or output filename containing "blender" is not Blender.
            foreground_searchable = f"{info.get('name') or ''} {argv[0] if argv else ''}".lower()
            row = {
                "pid": pid,
                "ppid": info.get("ppid"),
                "name": info.get("name"),
                "command": command,
                "username": info.get("username"),
            }
            if any(marker in foreground_searchable for marker in foreground_markers):
                foreground.append(row)
            if any(marker in searchable for marker in heavy_markers):
                unmanaged_heavy.append(row)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            inaccessible += 1
    return {
        "available": True,
        "foreground_resource_processes": sorted(foreground, key=lambda row: row["pid"]),
        "unmanaged_known_heavy": sorted(unmanaged_heavy, key=lambda row: row["pid"]),
        "inaccessible_processes": inaccessible,
        "frontmost_application": _frontmost_app(),
    }


def _mps() -> dict[str, Any]:
    try:
        import torch

        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        if not available:
            return {
                "telemetry_available": False,
                "built": built,
                "available": False,
                "scope": "scheduler-process",
            }
        current = int(torch.mps.current_allocated_memory())
        driver = int(torch.mps.driver_allocated_memory())
        recommended = int(torch.mps.recommended_max_memory())
        return {
            "telemetry_available": recommended > 0,
            "built": built,
            "available": available,
            "current_allocated_bytes": current,
            "driver_allocated_bytes": driver,
            "recommended_working_set_bytes": recommended,
            "declared_headroom_bytes": max(0, recommended - driver),
            "scope": "scheduler-process plus declared child peak; no global per-process Metal API",
        }
    except (ImportError, RuntimeError, AttributeError, OSError) as exc:
        return {
            "telemetry_available": False,
            "built": False,
            "available": False,
            "scope": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_host_telemetry(
    policy: ThrottlePolicy,
    *,
    disk_root: Path | str = REPO_ROOT,
    excluded_pids: set[int] | None = None,
) -> dict[str, Any]:
    """Collect one complete host snapshot. Missing required probes remain explicit."""

    missing: list[str] = []
    cpu: dict[str, Any]
    memory: dict[str, Any]
    swap: dict[str, Any]
    disk: dict[str, Any]
    processes: dict[str, Any]
    try:
        logical_cpus = int(psutil.cpu_count(logical=True) or 0)
        load1, load5, load15 = os.getloadavg()
        cpu_fraction = float(psutil.cpu_percent(interval=0.05) / 100.0)
        cpu = {
            "available": logical_cpus > 0,
            "logical_cpus": logical_cpus,
            "load_1m": load1,
            "load_5m": load5,
            "load_15m": load15,
            "load_1m_per_logical_cpu": load1 / logical_cpus if logical_cpus else None,
            "utilization_fraction": cpu_fraction,
        }
    except (OSError, ValueError) as exc:
        cpu = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not cpu.get("available"):
        missing.append("cpu")
    try:
        virtual = psutil.virtual_memory()
        memory = {
            "available": True,
            "total_bytes": int(virtual.total),
            "available_bytes": int(virtual.available),
            "available_percent": float(100.0 - virtual.percent),
            "pressure": _memory_pressure(),
        }
    except (OSError, ValueError) as exc:
        memory = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not memory.get("available"):
        missing.append("memory")
    elif policy.monitor.get("require_memory_pressure", True) and not memory["pressure"].get("available"):
        missing.append("memory_pressure")
    try:
        swap_raw = psutil.swap_memory()
        swap = {
            "available": True,
            "total_bytes": int(swap_raw.total),
            "used_bytes": int(swap_raw.used),
            "free_bytes": int(swap_raw.free),
            "used_gb": float(swap_raw.used / 1e9),
            "percent": float(swap_raw.percent),
        }
    except (OSError, ValueError) as exc:
        swap = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not swap.get("available"):
        missing.append("swap")
    try:
        usage = shutil.disk_usage(disk_root)
        disk = {
            "available": True,
            "root": str(Path(disk_root).resolve()),
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "free_gb": float(usage.free / 1e9),
        }
    except OSError as exc:
        disk = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not disk.get("available"):
        missing.append("disk")
    try:
        processes = _processes(policy, excluded_pids or set())
    except (OSError, ValueError, psutil.Error) as exc:
        processes = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not processes.get("available"):
        missing.append("processes")
    thermal = _thermal()
    if policy.monitor.get("require_thermal", True) and not thermal.get("available"):
        missing.append("thermal")
    power = _power()
    if policy.monitor.get("require_power", True) and not power.get("available"):
        missing.append("power")
    return {
        "schema": TELEMETRY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "cpu": cpu,
        "memory": memory,
        "swap": swap,
        "disk": disk,
        "processes": processes,
        "mps": _mps(),
        "thermal": thermal,
        "power": power,
        "missing_required_telemetry": sorted(set(missing)),
        "all_required_available": not missing,
    }


def _gate(
    gates: list[dict[str, Any]],
    name: str,
    ok: bool,
    observed: Any,
    limit: Any,
    reason: str,
    *,
    critical: bool = False,
) -> None:
    gates.append(
        {
            "name": name,
            "ok": bool(ok),
            "observed": observed,
            "limit": limit,
            "reason": reason,
            "critical": bool(critical and not ok),
        }
    )


def _dotted_value(payload: Any, dotted: str) -> Any:
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _receipt_requirement_report(
    requirement: ReceiptRequirement,
    evidence_root: Path,
) -> dict[str, Any]:
    path = (evidence_root / requirement.path).resolve()
    problems: list[str] = []
    payload: dict[str, Any] = {}
    if not path.is_relative_to(evidence_root.resolve()):
        problems.append("receipt resolves outside evidence root")
    elif not path.is_file():
        problems.append("receipt is missing")
    else:
        try:
            loaded = json.loads(path.read_text())
            payload = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"receipt is unreadable: {type(exc).__name__}")
    if payload and payload.get("schema") != requirement.schema:
        problems.append(f"schema {payload.get('schema')!r} != {requirement.schema!r}")
    if payload and requirement.schema == "mop-continual-progressive-rung/v1":
        receipt_payload = dict(payload)
        declared_digest = receipt_payload.pop("payload_sha256", None)
        if _canonical_sha256(receipt_payload) != declared_digest:
            problems.append("progressive continual receipt payload digest drift")
    observations: dict[str, Any] = {}
    for field, expected in requirement.fields:
        observed = _dotted_value(payload, field)
        observations[field] = observed
        if observed != expected:
            problems.append(f"{field}={observed!r}, expected {expected!r}")
    return {
        "path": requirement.path,
        "schema": payload.get("schema"),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "observations": observations,
        "problems": problems,
        "all_ok": not problems,
    }


def _task_resource_memory(
    task: TaskDeclaration,
    policy: ThrottlePolicy,
    evidence_root: Path,
) -> tuple[float, dict[str, Any]]:
    if task.estimated_unified_memory_gb is not None:
        return task.estimated_unified_memory_gb, {
            "mode": "declared",
            "all_ok": True,
            "effective_unified_memory_gb": task.estimated_unified_memory_gb,
        }
    if task.resource_probe:
        return 0.0, {
            "mode": "exclusive-unmeasured-probe",
            "all_ok": True,
            "effective_unified_memory_gb": None,
            "safety": "no concurrent lane plus live pressure, swap, thermal, and power gates",
        }
    path = (evidence_root / str(task.resource_receipt_path)).resolve()
    problems: list[str] = []
    payload: dict[str, Any] = {}
    if not path.is_relative_to(evidence_root.resolve()):
        problems.append("resource receipt resolves outside evidence root")
    elif not path.is_file():
        problems.append("resource receipt is missing")
    else:
        try:
            loaded = json.loads(path.read_text())
            payload = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"resource receipt is unreadable: {type(exc).__name__}")
    if payload and payload.get("schema") != task.resource_receipt_schema:
        problems.append("resource receipt schema drift")
    if payload and task.resource_receipt_schema == "mop-continual-progressive-rung/v1":
        receipt_payload = dict(payload)
        declared_digest = receipt_payload.pop("payload_sha256", None)
        if _canonical_sha256(receipt_payload) != declared_digest:
            problems.append("resource receipt payload digest drift")
    if payload and payload.get("rung") != task.resource_receipt_rung:
        problems.append("resource receipt rung drift")
    if payload and payload.get("all_mechanics_ok") is not True:
        problems.append("resource receipt mechanics did not complete")
    measurement = payload.get("resource_measurement") if isinstance(payload, dict) else None
    rss_bytes = measurement.get("max_rss_bytes") if isinstance(measurement, dict) else None
    if not isinstance(rss_bytes, int) or rss_bytes <= 0:
        problems.append("resource receipt lacks a positive measured max_rss_bytes")
        rss_bytes = 0
    if isinstance(measurement, dict) and measurement.get("measured_after_complete") is not True:
        problems.append("resource receipt was not measured after a complete rung/probe")
    uncertainty = float(policy.limits["forecast_uncertainty_fraction"])
    effective = rss_bytes / 1e9 * (1.0 + uncertainty)
    return effective, {
        "mode": "measured-prior-rung",
        "path": str(task.resource_receipt_path),
        "schema": payload.get("schema"),
        "rung": payload.get("rung"),
        "max_rss_bytes": rss_bytes or None,
        "uncertainty_fraction": uncertainty,
        "effective_unified_memory_gb": effective if not problems else None,
        "problems": problems,
        "all_ok": not problems,
    }


def evaluate_task(
    task: TaskDeclaration,
    telemetry: dict[str, Any],
    policy: ThrottlePolicy,
    *,
    active: list[dict[str, Any]] | None = None,
    task_already_active: bool = False,
    evidence_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate one candidate against one host snapshot and declared active lanes."""

    active = list(active or [])
    evidence_path = Path(evidence_root).resolve()
    gates: list[dict[str, Any]] = []
    missing = list(telemetry.get("missing_required_telemetry") or [])
    _gate(gates, "required_telemetry", not missing, missing, [], "all required probes must report")
    prerequisite_reports = [
        _receipt_requirement_report(requirement, evidence_path) for requirement in task.prerequisites
    ]
    _gate(
        gates,
        "receipt_prerequisites",
        all(report["all_ok"] for report in prerequisite_reports),
        prerequisite_reports,
        "every declared prior-rung completion receipt must match exact fields",
        "P6 and other dependent tasks fail closed until immutable prior receipts pass",
    )
    effective_task_memory, resource_report = _task_resource_memory(task, policy, evidence_path)
    _gate(
        gates,
        "resource_measurement",
        bool(resource_report["all_ok"]),
        resource_report,
        "declared peak, exclusive probe, or measured prior-rung max RSS",
        "unknown memory never becomes an ordinary task estimate",
    )
    maximum_lanes = int(policy.limits["maximum_active_lanes"])
    _gate(
        gates,
        "lane_count",
        len(active) < maximum_lanes,
        len(active),
        f"fewer than {maximum_lanes} active lanes",
        "at most two scheduler-owned lanes",
    )
    active_exclusive = [row.get("run_id") for row in active if row.get("requires_empty_lanes")]
    _gate(
        gates,
        "exclusive_lane",
        (not task.requires_empty_lanes or not active) and not active_exclusive,
        {"active_count": len(active), "active_exclusive": active_exclusive},
        {"required_empty": task.requires_empty_lanes},
        "exclusive measurement and progressive P6 lanes never overlap P4, P5, or another task",
    )
    active_heavy = sum(1 for row in active if row.get("lane") == "heavy")
    _gate(
        gates,
        "one_heavy",
        task.lane != "heavy" or active_heavy == 0,
        active_heavy,
        1,
        "a second heavy lane is never admitted",
    )
    _gate(
        gates,
        "second_lane_kind",
        not active or task.lane in SECOND_LANES,
        task.lane,
        sorted(SECOND_LANES),
        "only CPU, network, or light work may be the second lane",
    )
    processes = telemetry.get("processes") or {}
    unmanaged = list(processes.get("unmanaged_known_heavy") or [])
    _gate(
        gates,
        "unmanaged_heavy_process",
        not unmanaged,
        [{"pid": row.get("pid"), "name": row.get("name")} for row in unmanaged],
        [],
        "unknown heavy work has no resource declaration, so admission fails closed",
    )
    foreground = list(processes.get("foreground_resource_processes") or [])
    _gate(
        gates,
        "foreground_second_lane",
        not active or not foreground,
        [{"pid": row.get("pid"), "name": row.get("name")} for row in foreground],
        [],
        "Blender or another declared foreground workload permits one experiment lane, never two",
    )
    tier_name = "second_lane" if active else "first_lane"
    thresholds = policy.thresholds[tier_name]
    cpu = telemetry.get("cpu") or {}
    load = cpu.get("load_1m_per_logical_cpu")
    load_max = thresholds["maximum_load_per_logical_cpu"]
    _gate(
        gates,
        "cpu_load",
        task_already_active or (isinstance(load, int | float) and float(load) <= load_max),
        load,
        "admission-only" if task_already_active else load_max,
        (
            "owned task CPU load is expected at runtime; memory, swap, thermal, and disk remain gated"
            if task_already_active
            else f"{tier_name} normalized one-minute load ceiling"
        ),
    )
    utilization = cpu.get("utilization_fraction")
    utilization_max = thresholds["maximum_cpu_utilization_fraction"]
    _gate(
        gates,
        "cpu_utilization",
        task_already_active
        or (isinstance(utilization, int | float) and float(utilization) <= utilization_max),
        utilization,
        "admission-only" if task_already_active else utilization_max,
        (
            "owned task CPU utilization is expected at runtime; pressure gates remain active"
            if task_already_active
            else f"{tier_name} instantaneous CPU ceiling"
        ),
    )
    logical = int(cpu.get("logical_cpus") or 0)
    declared_cores = task.cpu_cores + sum(int(row.get("cpu_cores") or 0) for row in active)
    _gate(
        gates,
        "declared_cpu_cores",
        logical > 0 and declared_cores <= logical,
        declared_cores,
        logical,
        "declared concurrent core demand cannot exceed measured logical CPUs",
    )
    memory = telemetry.get("memory") or {}
    memory_percent = memory.get("available_percent")
    pressure_percent = (memory.get("pressure") or {}).get("free_percent")
    effective_percent = (
        min(float(memory_percent), float(pressure_percent))
        if isinstance(memory_percent, int | float) and isinstance(pressure_percent, int | float)
        else None
    )
    memory_min = thresholds["minimum_memory_available_percent"]
    _gate(
        gates,
        "memory_pressure",
        effective_percent is not None and effective_percent >= memory_min,
        effective_percent,
        memory_min,
        f"{tier_name} available-memory floor from both VM and memory_pressure",
        critical=effective_percent is not None and effective_percent < 10.0,
    )
    available_gb = float(memory.get("available_bytes") or 0) / 1e9
    memory_headroom = float(policy.limits["minimum_unified_memory_headroom_gb"])
    required_available_gb = memory_headroom + (0.0 if task_already_active else effective_task_memory)
    _gate(
        gates,
        "candidate_memory_headroom",
        available_gb >= required_available_gb,
        available_gb,
        required_available_gb,
        (
            "running task retains the minimum unified-memory headroom"
            if task_already_active
            else "measured available unified memory covers candidate peak plus headroom"
        ),
    )
    total_gb = float(memory.get("total_bytes") or 0) / 1e9
    declared_memory = effective_task_memory + sum(
        float(row.get("estimated_unified_memory_gb") or 0) for row in active
    )
    _gate(
        gates,
        "declared_memory_budget",
        total_gb >= declared_memory + memory_headroom,
        declared_memory,
        max(0.0, total_gb - memory_headroom),
        "all declared lane peaks fit measured unified memory with headroom",
    )
    swap_used = (telemetry.get("swap") or {}).get("used_gb")
    swap_max = thresholds["maximum_swap_used_gb"]
    _gate(
        gates,
        "swap",
        isinstance(swap_used, int | float) and float(swap_used) <= swap_max,
        swap_used,
        swap_max,
        f"{tier_name} swap-use ceiling",
    )
    thermal = telemetry.get("thermal") or {}
    _gate(
        gates,
        "thermal",
        thermal.get("status") == "normal",
        thermal.get("status"),
        "normal",
        "thermal and performance pressure must be normal",
        critical=thermal.get("status") == "critical",
    )
    power = telemetry.get("power") or {}
    power_ok = bool(power.get("on_ac")) or task.lane in {"network", "light"}
    _gate(
        gates,
        "power",
        power_ok,
        power.get("source"),
        "AC Power for heavy/CPU work",
        "battery power blocks compute-heavy admission",
    )
    free_gb = float((telemetry.get("disk") or {}).get("free_gb") or 0.0)
    uncertainty = float(policy.limits["forecast_uncertainty_fraction"])
    reserve = max(
        float(policy.limits["minimum_write_reserve_gb"]),
        task.forecast_write_gb * (1.0 + uncertainty) + task.atomic_write_gb,
    )
    reserve += sum(
        float(row.get("forecast_write_gb") or 0) * (1.0 + uncertainty)
        + float(row.get("atomic_write_gb") or 0)
        for row in active
    )
    floor = float(policy.limits["disk_floor_gb"])
    projected = free_gb - reserve
    _gate(
        gates,
        "forecasted_disk",
        projected >= floor,
        {"free_gb": free_gb, "reserve_gb": reserve, "projected_free_gb": projected},
        floor,
        "40 GB floor is preserved after declared writes, atomic temp space, and uncertainty",
        critical=projected < floor,
    )
    if task.accelerator == "mps":
        mps = telemetry.get("mps") or {}
        mps_available = bool(mps.get("available") and mps.get("telemetry_available"))
        _gate(
            gates,
            "mps_telemetry",
            mps_available,
            {"available": mps.get("available"), "scope": mps.get("scope")},
            True,
            "MPS availability and working-set telemetry are required for an MPS task",
        )
        mps_free_gb = float(mps.get("declared_headroom_bytes") or 0) / 1e9
        required_mps = task.estimated_mps_gb + float(policy.limits["minimum_mps_headroom_gb"])
        _gate(
            gates,
            "mps_working_set",
            mps_available and mps_free_gb >= required_mps,
            mps_free_gb,
            required_mps,
            "recommended MPS working set covers declared peak plus headroom",
        )
        active_mps = sum(1 for row in active if row.get("accelerator") == "mps")
        _gate(
            gates,
            "single_mps_owner",
            active_mps == 0,
            active_mps,
            0,
            "global per-process MPS allocation is unavailable, so MPS is never shared",
        )
    reasons = [str(gate["reason"]) for gate in gates if not gate["ok"]]
    return {
        "schema": DECISION_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "task_id": task.task_id,
        "threshold_tier": tier_name,
        "active_lanes": active,
        "gates": gates,
        "allowed": not reasons,
        "critical": any(gate["critical"] for gate in gates),
        "denied_reasons": reasons,
        "disk_forecast": {
            "free_gb": free_gb,
            "reserved_write_gb": reserve,
            "projected_free_gb": projected,
            "floor_gb": floor,
        },
        "user_process_policy": "observed only; scheduler signals only its own process group",
    }


def aggregate_admission(decisions: list[dict[str, Any]], required_good: int) -> dict[str, Any]:
    tail = decisions[-required_good:] if required_good > 0 else []
    allowed = len(tail) == required_good and all(bool(value.get("allowed")) for value in tail)
    return {
        "allowed": allowed,
        "required_consecutive_good_samples": required_good,
        "samples_observed": len(decisions),
        "consecutive_good_samples": _trailing_count(decisions, True),
        "consecutive_bad_samples": _trailing_count(decisions, False),
        "reason": (
            "admission hysteresis satisfied"
            if allowed
            else "admission requires the configured consecutive healthy samples"
        ),
    }


def _trailing_count(decisions: list[dict[str, Any]], wanted: bool) -> int:
    count = 0
    for decision in reversed(decisions):
        if bool(decision.get("allowed")) != wanted:
            break
        count += 1
    return count


def hysteresis_transition(
    status: str,
    decision: dict[str, Any],
    *,
    good_count: int,
    bad_count: int,
    last_transition_monotonic: float,
    now_monotonic: float,
    policy: ThrottlePolicy,
) -> dict[str, Any]:
    """Pure pause/resume state transition used by the live supervisor and unit tests."""

    allowed = bool(decision.get("allowed"))
    good = good_count + 1 if allowed else 0
    bad = 0 if allowed else bad_count + 1
    action = "none"
    new_status = status
    cooldown = float(policy.monitor["cooldown_seconds"])
    if status == "running" and (
        bool(decision.get("critical")) or bad >= int(policy.monitor["pause_bad_samples"])
    ):
        action = "pause"
        new_status = "paused"
        good = 0
        bad = 0
    elif (
        status == "paused"
        and good >= int(policy.monitor["resume_good_samples"])
        and now_monotonic - last_transition_monotonic >= cooldown
    ):
        action = "resume"
        new_status = "running"
        good = 0
        bad = 0
    return {"status": new_status, "action": action, "good_count": good, "bad_count": bad}


def checkpoint_snapshot(task: TaskDeclaration, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Hash atomically published checkpoint files. Temporary files are never resume authorities."""

    root = Path(repo_root).resolve()
    paths: set[Path] = set()
    for pattern in task.checkpoint_globs:
        for value in glob.glob(str(root / pattern), recursive=True):
            path = Path(value)
            if path.is_file() and not path.name.endswith(".tmp"):
                paths.add(path.resolve())
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "atomic_publications_only": task.atomic_checkpoints,
        "files": rows,
        "file_count": len(rows),
        "aggregate_sha256": digest,
    }


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def _registry_paths(state_root: Path) -> tuple[Path, Path]:
    return state_root / "active.lock", state_root / "active.json"


def _load_active_unlocked(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": REGISTRY_SCHEMA, "updated_at": None, "runs": {}}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ThrottleRefused(f"active registry is unreadable: {type(exc).__name__}: {exc}") from exc
    if payload.get("schema") != REGISTRY_SCHEMA or not isinstance(payload.get("runs"), dict):
        raise ThrottleRefused("active registry schema is invalid")
    return payload


def active_lanes(state_root: Path | str = DEFAULT_STATE_ROOT) -> list[dict[str, Any]]:
    root = Path(state_root)
    _, path = _registry_paths(root)
    payload = _load_active_unlocked(path)
    return [
        row
        for row in payload["runs"].values()
        if isinstance(row, dict) and _pid_alive(int(row.get("scheduler_pid") or 0))
    ]


def _update_registry(state_root: Path, run_id: str, row: dict[str, Any] | None) -> None:
    state_root.mkdir(parents=True, exist_ok=True)
    lock_path, registry_path = _registry_paths(state_root)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = _load_active_unlocked(registry_path)
        payload["runs"] = {
            key: value
            for key, value in payload["runs"].items()
            if isinstance(value, dict) and _pid_alive(int(value.get("scheduler_pid") or 0))
        }
        if row is None:
            payload["runs"].pop(run_id, None)
        else:
            payload["runs"][run_id] = row
        payload["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(registry_path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _reserve_registry(
    state_root: Path,
    run_id: str,
    task: TaskDeclaration,
    telemetry: dict[str, Any],
    policy: ThrottlePolicy,
) -> dict[str, Any]:
    """Atomically recheck lane admission and reserve a slot, closing concurrent-launch races."""

    state_root.mkdir(parents=True, exist_ok=True)
    lock_path, registry_path = _registry_paths(state_root)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = _load_active_unlocked(registry_path)
        payload["runs"] = {
            key: value
            for key, value in payload["runs"].items()
            if isinstance(value, dict) and _pid_alive(int(value.get("scheduler_pid") or 0))
        }
        if run_id in payload["runs"]:
            raise ThrottleRefused(f"run_id {run_id!r} is already active")
        active = list(payload["runs"].values())
        decision = evaluate_task(task, telemetry, policy, active=active)
        if not decision["allowed"]:
            raise ThrottleRefused(
                "atomic launch reservation denied: " + "; ".join(decision["denied_reasons"])
            )
        payload["runs"][run_id] = _registry_row(task, run_id, None, "launching")
        payload["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(registry_path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return decision


def dry_run_decision(
    task: TaskDeclaration,
    policy: ThrottlePolicy,
    *,
    samples: int | None = None,
    interval_seconds: float | None = None,
    state_root: Path | str = DEFAULT_STATE_ROOT,
    disk_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    count = int(samples or policy.monitor["admission_good_samples"])
    interval = float(
        policy.monitor["sample_interval_seconds"] if interval_seconds is None else interval_seconds
    )
    snapshots: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    active = active_lanes(state_root)
    for index in range(count):
        snapshot = collect_host_telemetry(policy, disk_root=disk_root)
        decision = evaluate_task(task, snapshot, policy, active=active)
        snapshots.append(snapshot)
        decisions.append(decision)
        if index + 1 < count and interval > 0:
            time.sleep(interval)
    admission = aggregate_admission(decisions, int(policy.monitor["admission_good_samples"]))
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "dry-run",
        "policy": {"path": str(policy.path), "sha256": policy.sha256},
        "implementation": {
            "path": str(IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(IMPLEMENTATION_PATH),
        },
        "profile": get_profile(policy.profile_name).as_dict(),
        "task": asdict(task),
        "active_lanes": active,
        "telemetry_samples": snapshots,
        "decisions": decisions,
        "admission": admission,
        "command_executed": False,
    }


def _event(receipt: dict[str, Any], name: str, **payload: Any) -> None:
    receipt["events"].append({"at": datetime.now(UTC).isoformat(), "event": name, **payload})


def _registry_row(task: TaskDeclaration, run_id: str, child_pid: int | None, status: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "scheduler_pid": os.getpid(),
        "child_pid": child_pid,
        "status": status,
        "lane": task.lane,
        "accelerator": task.accelerator,
        "cpu_cores": task.cpu_cores,
        "estimated_unified_memory_gb": task.estimated_unified_memory_gb,
        "estimated_mps_gb": task.estimated_mps_gb,
        "forecast_write_gb": task.forecast_write_gb,
        "atomic_write_gb": task.atomic_write_gb,
        "requires_empty_lanes": task.requires_empty_lanes,
        "resource_probe": task.resource_probe,
        "command": list(task.command),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _signal_owned_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    """Signal only the new session created for this supervisor-owned child."""

    if process.poll() is None:
        os.killpg(process.pid, sig)


def _stop_at_wall(
    process: subprocess.Popen[Any],
    task: TaskDeclaration,
    receipt: dict[str, Any],
    receipt_path: Path,
    grace_seconds: float,
) -> int | None:
    _signal_owned_group(process, signal.SIGSTOP)
    snapshot = checkpoint_snapshot(task)
    _event(
        receipt,
        "wall-boundary-pause",
        child_pid=process.pid,
        checkpoint=snapshot,
        note="latest atomic checkpoint is the resume authority; work since it may replay",
    )
    _atomic_json(receipt_path, receipt)
    _signal_owned_group(process, signal.SIGCONT)
    _signal_owned_group(process, signal.SIGINT)
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        _signal_owned_group(process, signal.SIGTERM)
        try:
            return process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_owned_group(process, signal.SIGSTOP)
            _event(
                receipt,
                "managed-child-stopped",
                child_pid=process.pid,
                reason="did not exit after SIGINT and SIGTERM; SIGKILL is intentionally forbidden",
            )
            _atomic_json(receipt_path, receipt)
            return None


def run_task(
    task: TaskDeclaration,
    policy: ThrottlePolicy,
    *,
    run_id: str,
    state_root: Path | str = DEFAULT_STATE_ROOT,
    disk_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Run a declared task with dynamic pause/resume and identical-command restart boundaries."""

    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ThrottleRefused("run_id must contain only letters, digits, dot, underscore, or hyphen")
    root = Path(state_root)
    run_dir = root / run_id
    receipt_path = run_dir / "run_receipt.json"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    admission_receipt = dry_run_decision(
        task,
        policy,
        interval_seconds=min(1.0, float(policy.monitor["sample_interval_seconds"])),
        state_root=root,
        disk_root=disk_root,
    )
    if not admission_receipt["admission"]["allowed"]:
        admission_receipt["mode"] = "execute-refused"
        _atomic_json(receipt_path, admission_receipt)
        raise ThrottleRefused("host admission denied; inspect " + str(receipt_path))
    receipt: dict[str, Any] = {
        **admission_receipt,
        "mode": "execute",
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "events": [],
        "invocations": [],
        "command_executed": False,
        "ownership": "only child process groups created by this scheduler may receive signals",
    }
    try:
        reservation = _reserve_registry(
            root,
            run_id,
            task,
            admission_receipt["telemetry_samples"][-1],
            policy,
        )
    except ThrottleRefused as exc:
        receipt["status"] = "atomic-reservation-refused"
        receipt["reservation_error"] = str(exc)
        _atomic_json(receipt_path, receipt)
        raise
    receipt["atomic_reservation_decision"] = reservation
    receipt["command_executed"] = True
    _atomic_json(receipt_path, receipt)
    started = time.monotonic()
    deadline = started + task.wall_minutes * 60.0
    invocation = 0
    process: subprocess.Popen[Any] | None = None
    paused = False
    good_count = 0
    bad_count = 0
    last_transition = started
    final_returncode: int | None = None
    try:
        while time.monotonic() < deadline:
            invocation += 1
            stdout_path = logs / f"invocation_{invocation:02d}.stdout.log"
            stderr_path = logs / f"invocation_{invocation:02d}.stderr.log"
            with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
                process = subprocess.Popen(
                    task.command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
                invocation_row = {
                    "index": invocation,
                    "pid": process.pid,
                    "started_at": datetime.now(UTC).isoformat(),
                    "command": list(task.command),
                    "command_sha256": hashlib.sha256(
                        json.dumps(list(task.command), separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                }
                receipt["invocations"].append(invocation_row)
                _event(receipt, "invocation-start", index=invocation, child_pid=process.pid)
                _update_registry(root, run_id, _registry_row(task, run_id, process.pid, "running"))
                _atomic_json(receipt_path, receipt)
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(max(0.1, float(policy.monitor["sample_interval_seconds"])))
                    other_active = [row for row in active_lanes(root) if row.get("run_id") != run_id]
                    telemetry = collect_host_telemetry(
                        policy,
                        disk_root=disk_root,
                        excluded_pids={process.pid},
                    )
                    decision = evaluate_task(
                        task,
                        telemetry,
                        policy,
                        active=other_active,
                        task_already_active=True,
                    )
                    transition = hysteresis_transition(
                        "paused" if paused else "running",
                        decision,
                        good_count=good_count,
                        bad_count=bad_count,
                        last_transition_monotonic=last_transition,
                        now_monotonic=time.monotonic(),
                        policy=policy,
                    )
                    good_count = int(transition["good_count"])
                    bad_count = int(transition["bad_count"])
                    if transition["action"] == "pause" and task.pause_safe:
                        _signal_owned_group(process, signal.SIGSTOP)
                        paused = True
                        last_transition = time.monotonic()
                        checkpoint = checkpoint_snapshot(task)
                        _event(
                            receipt,
                            "dynamic-pause",
                            decision=decision,
                            checkpoint=checkpoint,
                            note="SIGSTOP targets this scheduler-owned process group only",
                        )
                        _update_registry(root, run_id, _registry_row(task, run_id, process.pid, "paused"))
                    elif transition["action"] == "pause":
                        _event(
                            receipt,
                            "pause-required-but-unsupported",
                            decision=decision,
                            note=(
                                "task declaration is not pause-safe; no user or foreign process was signaled"
                            ),
                        )
                    elif transition["action"] == "resume" and paused:
                        _signal_owned_group(process, signal.SIGCONT)
                        paused = False
                        last_transition = time.monotonic()
                        _event(receipt, "dynamic-resume", decision=decision)
                        _update_registry(root, run_id, _registry_row(task, run_id, process.pid, "running"))
                    receipt["last_telemetry"] = telemetry
                    receipt["last_decision"] = decision
                    _atomic_json(receipt_path, receipt)
                if time.monotonic() >= deadline and process.poll() is None:
                    if paused:
                        _signal_owned_group(process, signal.SIGCONT)
                        paused = False
                    final_returncode = _stop_at_wall(
                        process,
                        task,
                        receipt,
                        receipt_path,
                        float(policy.monitor["graceful_stop_seconds"]),
                    )
                    invocation_row["returncode"] = final_returncode
                    invocation_row["finished_at"] = datetime.now(UTC).isoformat()
                    receipt["status"] = "resumable-wall-boundary"
                    break
                final_returncode = process.poll()
                invocation_row["returncode"] = final_returncode
                invocation_row["finished_at"] = datetime.now(UTC).isoformat()
                invocation_row["checkpoint_after"] = checkpoint_snapshot(task)
                _event(
                    receipt,
                    "invocation-exit",
                    index=invocation,
                    returncode=final_returncode,
                )
                _atomic_json(receipt_path, receipt)
            if final_returncode == 0:
                receipt["status"] = "complete"
                break
            if final_returncode not in task.restart_exit_codes:
                receipt["status"] = "failed"
                break
            if time.monotonic() >= deadline:
                receipt["status"] = "resumable-wall-boundary"
                break
            _event(
                receipt,
                "identical-command-resume",
                prior_returncode=final_returncode,
                command=list(task.command),
            )
            _atomic_json(receipt_path, receipt)
        if receipt["status"] == "running":
            receipt["status"] = "resumable-wall-boundary"
    finally:
        if process is not None and paused and process.poll() is None:
            _signal_owned_group(process, signal.SIGCONT)
        if process is None or process.poll() is not None:
            _update_registry(root, run_id, None)
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["final_returncode"] = final_returncode
        receipt["final_checkpoint"] = checkpoint_snapshot(task)
        _atomic_json(receipt_path, receipt)
    return receipt


def write_receipt(path: Path | str, payload: dict[str, Any]) -> None:
    _atomic_json(Path(path), payload)
