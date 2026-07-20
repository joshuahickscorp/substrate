
from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import yaml

from ..config import REPO_ROOT
from ..studies.continual_million_event_verify import (
    TIE_RULE as P6_TIE_RULE,
)
from ..studies.continual_million_event_verify import (
    audit_rung_semantics,
)
from ..studies.continual_million_event_verify import (
    build_verification_receipt as build_p6_verification_receipt,
)
from . import external_coexistence as coexistence
from . import task_policy_authority as task_policy
from .profiles import get_profile

POLICY_SCHEMA = "mop-local-execution-throttle-policy/v1"
TELEMETRY_SCHEMA = "mop-local-host-telemetry/v1"
DECISION_SCHEMA = "mop-local-throttle-decision/v1"
RECEIPT_SCHEMA = "mop-local-throttle-receipt/v1"
COMPLETION_AUTHORITY_SCHEMA = "mop-local-throttle-completion-authority/v1"
PROGRESS_AUTHORITY_SCHEMA = "mop-local-throttle-progress-authority/v1"
REGISTRY_SCHEMA = "mop-local-throttle-active-registry/v1"
PAYLOAD_DIGEST_REQUIRED_SCHEMAS = frozenset(
    {
        "mop-continual-progressive-rung/v1",
        "mop-continual-progressive-rung-independent-verifier/v1",
        "mop-p5-context-screen/v1",
        "mop-p5-context-fresh-training-challenge/v1",
        "mop-p5-context-independent-verifier/v1",
        "mop-p5-traingrid-memory-trace/v1",
    }
)
LANES = frozenset({"heavy", "cpu", "network", "light"})
ACCELERATORS = frozenset({"none", "mps"})
SECOND_LANES = frozenset({"cpu", "network", "light"})
EXTERNAL_COEXISTENCE_PROFILE = "hawking_serial_cpu_v1"
EXTERNAL_COEXISTENCE_TASKS = frozenset(
    {
        "edcm1_official_cpu",
        "edcm1_verify_cpu",
        "escs_x0_official_cpu",
        "escs_x0_verify_cpu",
    }
)
SEED_BOUNDARY_TASKS = frozenset({"edcm1_official_cpu", "escs_x0_official_cpu"})
TASKPOLICY_COEXISTENCE_PREFIX = (
    "/usr/sbin/taskpolicy",
    "-b",
    "-d",
    "throttle",
    "-c",
    "background",
    "-m",
    "4096",
    "-P",
    "kill",
    "/usr/bin/env",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "VECLIB_MAXIMUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)
TASKPOLICY_COEXISTENCE_CAP_GB = 4096 * 1024 * 1024 / 1e9
DEFAULT_POLICY = REPO_ROOT / "configs/local_execution_throttle.yaml"
DEFAULT_STATE_ROOT = REPO_ROOT / "runs/local_throttle"
IMPLEMENTATION_PATH = Path(__file__).resolve()
TASK_POLICY_HELPER_PATH = Path(task_policy.__file__).resolve()
TASK_POLICY_HELPER_SHA256 = "db6adcd470c11195b842e7ddc27bb1a1b1b03942425c1d126240cbf7641d8c88"
EXTERNAL_COEXISTENCE_HELPER_PATH = Path(coexistence.__file__).resolve()
EXTERNAL_COEXISTENCE_HELPER_SHA256 = "2e9a3a5ce05268ee3b947151a2bf9261457c93241994742ecf535ac24e723a19"
HAWKING_ROOT = Path.home() / "Downloads/hawking"
HAWKING_PYTHON = Path(
    "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
)
HAWKING_PYTHON_ARGV0 = Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12")
OUTPUT_AUTHORITY_FLAGS = ("--out", "--output", "--verification-out")
COMPATIBLE_GOVERNOR_IMPLEMENTATION_SHA256 = frozenset(
    {
        "73ffca97b312bdb7971bcfffb441fb4b204a2a26f8c9964a50e4e7debe00f3f7",
        "bd7dd790460adc7760620007c691e3c345e89d9630c303abf40de59b924fddfb",
        "a1a8d4e3d6ca23d50808e6657eaa6c68eadffa4d8f29d81f98da49b4eb014d40",
    }
)
LEGACY_POLICY_BASELINE_BINDINGS = (
    (
        REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V0.json",
        "228916fc01f99796179aa7d9402fb46c974c602c28951d0a0e981e2678582f37",
        "73ffca97b312bdb7971bcfffb441fb4b204a2a26f8c9964a50e4e7debe00f3f7",
    ),
    (
        REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V1.json",
        "6203b72f8e12bedecc3f7aa460c7d6e9a9827c28a1c052d7928c1fe8c6ca65a6",
        "bd7dd790460adc7760620007c691e3c345e89d9630c303abf40de59b924fddfb",
    ),
    (
        REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V2.json",
        "4c34619ea2a7bdcde9526f9c86245cc990c6c6aaaad8e59abb1c06f37d7fd5ce",
        "a1a8d4e3d6ca23d50808e6657eaa6c68eadffa4d8f29d81f98da49b4eb014d40",
    ),
)
ESCS_PREFLIGHT_SCHEMA = "mop-escs-substrate-preflight-report/v1"
EDCM_RECEIPT_SCHEMA = "mop-edcm1-receipt/v3"
EDCM_VERIFICATION_SCHEMA = "mop-edcm1-verification-artifact/v1"
X0_RECEIPT_SCHEMA = "mop-escs-x0-receipt/v1"
X0_VERIFICATION_SCHEMA = "mop-escs-x0-verification/v1"
EDCM_RECEIPT_PATH = "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"
EDCM_VERIFICATION_PATH = "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json"
X0_RECEIPT_PATH = "proof/ESCS_X0_EVENT_FORMATION.json"
X0_VERIFICATION_PATH = "proof/ESCS_X0_EVENT_FORMATION.verification.json"
NATIVE_SEAL_FIELDS = {
    ESCS_PREFLIGHT_SCHEMA: "report_sha256",
    EDCM_RECEIPT_SCHEMA: "receipt_sha256",
    EDCM_VERIFICATION_SCHEMA: "verification_artifact_sha256",
    X0_RECEIPT_SCHEMA: "receipt_sha256",
    X0_VERIFICATION_SCHEMA: "verification_sha256",
}
P5_SCREEN_SCHEMA = "mop-p5-context-screen/v1"
P5_GRID_SCHEMA = "mop-p5-traingrid-memory-trace/v1"
P5_CHALLENGE_SCHEMA = "mop-p5-context-fresh-training-challenge/v1"
P5_VERIFIER_SCHEMA = "mop-p5-context-independent-verifier/v1"
P5_CONFIG_PATH = REPO_ROOT / "configs/experiment/mop_p5_context_capability.yaml"
P5_BOUNDARY_TRACE = REPO_ROOT / "proof/P5_MEMORY_BOUNDARY_TRACE.json"
P5_SOURCE_PATHS = (
    "configs/experiment/mop_p5_context_capability.yaml",
    "scripts/p5_context_capability.py",
    "src/mop/substrate/p5_context.py",
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)
P5_GRID_SOURCE_PATHS = (
    "configs/experiment/mop_p5_context_capability.yaml",
    "scripts/p5_traingrid_memory_probe.py",
    "scripts/p5_context_capability.py",
    "src/mop/substrate/p5_context.py",
    "src/mop/substrate/custom_workbench.py",
    "src/mop/substrate/p4_screen.py",
)
P5_CLAIM_SCOPE = (
    "exact-versus-factorized context pilot on deterministic programmatic video; "
    "not natural-video, memory-rung, or general-capability evidence"
)
P5_EVIDENCE_CLASS = "R1 independently recomputed programmatic pilot evidence"
P5_CHALLENGE_SOURCE_PATHS = P5_SOURCE_PATHS + (
    "scripts/p5_context_fresh_challenge.py",
    "src/mop/studies/p5_context_challenge.py",
    "src/mop/studies/p5_context_verify.py",
)
P5_VERIFIER_SOURCE_PATHS = P5_SOURCE_PATHS + (
    "scripts/verify_p5_context_capability.py",
    "src/mop/studies/p5_context_verify.py",
)
P5_FRAME_COUNTS = (64, 32, 16)
P5_PRIMARY_FRAMES = (64, 32)
P5_MECHANISMS = ("exact_global", "window_local", "recurrent", "hierarchical_pooled")
P5_FRESH_SEEDS = (5101, 5102, 5103)
P5_TRAINABILITY_MARGIN = 0.05
P5_CEILING_CHANCE_OFFSET = 0.05
P5_CEILING_UPPER = 0.95
P5_BASE_MUTATION_IDS = frozenset(
    {
        "incomplete-pilot",
        "all-ok-false",
        "source-hash-drift",
        "config-binding-drift",
        "confirmatory-promotion",
        "raw-score-mutation",
        "cached-seed-source-drift",
        "checkpoint-source-drift",
        "matched-compute-drift",
        "threshold-tie-promotion",
        "sealed-profile-config-mismatch",
        "fresh-challenge-hint-flip",
        "ceilinged-contrast-promotion",
        "missing-seed-result-artifact",
        "missing-arm-receipt-artifact",
        "missing-checkpoint-artifact",
        "checkpoint-file-hash-drift",
        "seed-selection-drift",
    }
)
P5_CHALLENGE_MUTATION_IDS = frozenset(
    {
        "fresh-seed-overlap",
        "fresh-run-drop",
        "fresh-confirmatory-promotion",
        "challenge-shape-omission",
        "fresh-trainability-gate-fabrication",
    }
)
P6_PREFLIGHT = REPO_ROOT / "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json"
P6_RUN_CONFIG = REPO_ROOT / "configs/experiment/continual_million_event_rungs.yaml"
P6_RUNG_SCHEMA = "mop-continual-progressive-rung/v1"
P6_VERIFIER_SCHEMA = "mop-continual-progressive-rung-independent-verifier/v1"
P6_CLAIM_SCOPE = "disk-backed programmatic continual-stream mechanics only; no capability claim"
P6_MINIMUM_SANE_RSS_BYTES = 16 * 1024 * 1024
P6_VERIFIER_IMPLEMENTATION_PATHS = (
    "src/mop/studies/continual_million_event_verify.py",
    "scripts/verify_continual_million_event_rung.py",
    "scripts/continual_million_event_rung.py",
    "configs/experiment/continual_million_event_rungs.yaml",
    "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json",
)
GOVERNED_PROVENANCE_SCHEMAS = frozenset(
    {
        P5_SCREEN_SCHEMA,
        P5_GRID_SCHEMA,
        P5_CHALLENGE_SCHEMA,
        P5_VERIFIER_SCHEMA,
        P6_RUNG_SCHEMA,
        P6_VERIFIER_SCHEMA,
        EDCM_RECEIPT_SCHEMA,
        EDCM_VERIFICATION_SCHEMA,
        X0_RECEIPT_SCHEMA,
        X0_VERIFICATION_SCHEMA,
    }
)


class ThrottleRefused(RuntimeError):
    pass


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


def _external_coexistence_task_problems(task: TaskDeclaration) -> list[str]:
    problems: list[str] = []
    if task.task_id not in EXTERNAL_COEXISTENCE_TASKS:
        problems.append("task id is outside the exact reviewed coexistence set")
    if task.lane != "cpu" or task.accelerator != "none" or task.cpu_cores != 1:
        problems.append("coexistence is restricted to one CPU core")
    if task.estimated_unified_memory_gb != TASKPOLICY_COEXISTENCE_CAP_GB:
        problems.append("task must declare its exact reviewed taskpolicy memory cap")
    if task.resource_probe:
        problems.append("kernel-bounded coexistence tasks are declared, not unmeasured, probes")
    if not task.requires_empty_lanes:
        problems.append("coexistence task must remain exclusive within the MOP scheduler")
    if not task.pause_safe or not task.atomic_checkpoints or not task.checkpoint_globs:
        problems.append("coexistence task must retain an atomic pause/replay authority")
    if task.command[: len(TASKPOLICY_COEXISTENCE_PREFIX)] != TASKPOLICY_COEXISTENCE_PREFIX:
        problems.append("task must use its pinned lower-priority taskpolicy wrapper")
    return problems


def is_taskpolicy_coexistence_command(command: Sequence[str]) -> bool:
    value = tuple(command)
    return value[: len(TASKPOLICY_COEXISTENCE_PREFIX)] == TASKPOLICY_COEXISTENCE_PREFIX


def _effective_external_task_cores(task: TaskDeclaration) -> int:
    return task.cpu_cores


def _is_external_coexistence_task(task: TaskDeclaration) -> bool:
    return not _external_coexistence_task_problems(task)


def _is_seed_boundary_task(task: TaskDeclaration) -> bool:
    return task.task_id in SEED_BOUNDARY_TASKS and _is_external_coexistence_task(task)


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


def _json_value(value: Any) -> Any:

    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _assert_task_policy_helper_pin() -> None:
    try:
        observed = _sha256_file(TASK_POLICY_HELPER_PATH)
    except OSError as exc:
        raise ThrottleRefused(f"task-policy helper is unreadable: {type(exc).__name__}") from exc
    if observed != TASK_POLICY_HELPER_SHA256:
        raise ThrottleRefused("task-policy helper implementation drifted from the governor-pinned authority")


def _assert_external_coexistence_helper_pin() -> None:
    try:
        observed = _sha256_file(EXTERNAL_COEXISTENCE_HELPER_PATH)
    except OSError as exc:
        raise ThrottleRefused(f"external-coexistence helper is unreadable: {type(exc).__name__}") from exc
    if observed != EXTERNAL_COEXISTENCE_HELPER_SHA256:
        raise ThrottleRefused(
            "external-coexistence helper implementation drifted from the governor-pinned authority"
        )


def _hawking_coexistence_profile() -> coexistence.HawkingSerialCPUProfile:
    _assert_external_coexistence_helper_pin()
    return coexistence.HawkingSerialCPUProfile.create(
        root=HAWKING_ROOT,
        python_executable=HAWKING_PYTHON,
        expected_uid=os.getuid(),
    )


def _hawking_v5_coexistence_profile() -> coexistence.HawkingV5UltraCPUProfile:
    _assert_external_coexistence_helper_pin()
    return coexistence.HawkingV5UltraCPUProfile.create(
        root=HAWKING_ROOT,
        python_executable=HAWKING_PYTHON,
        python_argv0=HAWKING_PYTHON_ARGV0,
        expected_uid=os.getuid(),
    )


def _task_policy_context(policy: ThrottlePolicy, task: TaskDeclaration) -> dict[str, Any]:
    safety = task_policy.build_policy_safety_contract(
        profile=get_profile(policy.profile_name).as_dict(),
        limits=policy.limits,
        monitor=policy.monitor,
        thresholds=policy.thresholds,
    )
    return {
        "policy_schema": POLICY_SCHEMA,
        "policy_path": str(policy.path),
        "full_policy_sha256": policy.sha256,
        "profile_name": policy.profile_name,
        "safety_contract": safety,
        "foreground_markers": tuple(str(value) for value in policy.monitor["foreground_markers"]),
        "known_heavy_markers": tuple(str(value) for value in policy.monitor["known_heavy_markers"]),
        "task_id": task.task_id,
        "task_payload": _json_value(asdict(task)),
    }


def _build_task_policy_authority(
    policy: ThrottlePolicy,
    task: TaskDeclaration,
) -> dict[str, Any]:
    _assert_task_policy_helper_pin()
    return task_policy.build_task_policy_authority(**_task_policy_context(policy, task))


def _load_legacy_policy_baselines() -> tuple[dict[str, Any], ...]:
    _assert_task_policy_helper_pin()
    manifests: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, str]] = set()
    for path, expected_manifest_sha256, expected_governor_sha256 in LEGACY_POLICY_BASELINE_BINDINGS:
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ThrottleRefused(
                f"reviewed legacy baseline {path.name} is unreadable: {type(exc).__name__}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ThrottleRefused(f"reviewed legacy baseline {path.name} is not an object")
        problems = task_policy.policy_baseline_manifest_problems(loaded)
        if problems:
            raise ThrottleRefused(f"reviewed legacy baseline {path.name} is invalid: {'; '.join(problems)}")
        implementation = loaded.get("governor_implementation")
        if (
            loaded.get("manifest_sha256") != expected_manifest_sha256
            or not isinstance(implementation, dict)
            or implementation.get("sha256") != expected_governor_sha256
        ):
            raise ThrottleRefused(f"reviewed legacy baseline {path.name} binding drifted")
        policy_binding = loaded.get("policy")
        if not isinstance(policy_binding, dict):
            raise ThrottleRefused(f"reviewed legacy baseline {path.name} policy binding is invalid")
        identity = (
            str(policy_binding.get("path")),
            str(policy_binding.get("sha256")),
            str(implementation.get("path")),
            str(implementation.get("sha256")),
        )
        if identity in identities:
            raise ThrottleRefused("reviewed legacy policy baselines contain an ambiguous identity")
        identities.add(identity)
        manifests.append(loaded)
    return tuple(manifests)


def _task_output_path(task: TaskDeclaration) -> str | None:

    indexes = [index for index, value in enumerate(task.command) if value in OUTPUT_AUTHORITY_FLAGS]
    if not indexes:
        return None
    if len(indexes) != 1 or indexes[0] + 1 >= len(task.command):
        raise ThrottleRefused(
            f"task {task.task_id}: command must declare exactly one output-authority target"
        )
    value = task.command[indexes[0] + 1]
    path = Path(value)
    if not value or value.startswith("-") or path.is_absolute() or ".." in path.parts:
        raise ThrottleRefused(f"task {task.task_id}: output-authority target must be repository-relative")
    return value


def _requires_completion_provenance(task: TaskDeclaration) -> bool:
    return task.task_id.startswith(("p5", "p6", "edcm1_", "escs_x0_"))


def _command_sha256(command: tuple[str, ...] | list[str]) -> str:
    return _canonical_sha256(list(command))


def _rusage_children_peak_rss_bytes() -> int:

    raw = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _process_tree_rss_bytes(pid: int) -> int:

    try:
        process = psutil.Process(pid)
        descendants = process.children(recursive=True)
        values = [int(process.memory_info().rss)]
        for child in descendants:
            try:
                values.append(int(child.memory_info().rss))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return sum(values)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return 0


def _p6_source_live_binding_authority(receipt: dict[str, Any]) -> dict[str, Any]:

    def live_path(value: object) -> tuple[str, Path]:
        if not isinstance(value, str) or not value.strip():
            raise ThrottleRefused("P6 source binding path is missing")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ThrottleRefused("P6 source binding path escapes the repository")
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()):
            raise ThrottleRefused("P6 source binding path escapes the repository")
        return value, path

    config = receipt.get("config")
    if not isinstance(config, dict):
        raise ThrottleRefused("P6 preflight config binding is missing")
    config_name, config_path = live_path(config.get("path"))
    config_sha256 = _sha256_file(config_path)
    config_payload = config.get("payload")
    if config_sha256 != config.get("sha256"):
        raise ThrottleRefused("P6 preflight live config hash drift")
    if not isinstance(config_payload, dict) or _canonical_sha256(config_payload) != config.get(
        "profile_sha256"
    ):
        raise ThrottleRefused("P6 preflight embedded config digest drift")
    if yaml.safe_load(config_path.read_text()) != config_payload:
        raise ThrottleRefused("P6 preflight live config payload drift")

    implementation = receipt.get("implementation")
    if not isinstance(implementation, list) or not implementation:
        raise ThrottleRefused("P6 preflight implementation bindings are missing")
    implementation_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in implementation:
        if not isinstance(row, dict):
            raise ThrottleRefused("P6 preflight implementation binding is invalid")
        name, path = live_path(row.get("path"))
        if name in seen:
            raise ThrottleRefused("P6 preflight implementation binding is duplicated")
        sha256 = _sha256_file(path)
        if sha256 != row.get("sha256"):
            raise ThrottleRefused(f"P6 preflight live implementation drift: {name}")
        seen.add(name)
        implementation_rows.append({"path": name, "sha256": sha256})
    if config_name not in seen:
        raise ThrottleRefused("P6 preflight config is absent from implementation bindings")

    wave = receipt.get("wave_e0")
    if not isinstance(wave, dict):
        raise ThrottleRefused("P6 preflight Wave E0 binding is missing")
    wave_name, wave_path = live_path(wave.get("path"))
    wave_sha256 = _sha256_file(wave_path)
    if wave_sha256 != wave.get("sha256"):
        raise ThrottleRefused("P6 preflight live Wave E0 hash drift")
    authority: dict[str, Any] = {
        "config": {
            "path": config_name,
            "sha256": config_sha256,
            "payload_sha256": str(config["profile_sha256"]),
        },
        "implementation": implementation_rows,
        "wave_e0": {"path": wave_name, "sha256": wave_sha256},
    }
    authority["bindings_sha256"] = _canonical_sha256(authority)
    return authority


def _p6_resource_evidence() -> dict[str, Any]:

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
    live_authority = _p6_source_live_binding_authority(receipt)
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
    profile = config.get("profile")
    if not isinstance(profile, dict):
        raise ThrottleRefused("P6 progressive run config profile is missing")
    minimum_chunk_events = int(profile.get("minimum_chunk_events", 0))
    chunks_per_stream = int(profile.get("chunks_per_stream", 0))
    if minimum_chunk_events <= 0 or chunks_per_stream <= 0:
        raise ThrottleRefused("P6 chunk projection parameters must be positive")
    return {
        **observed,
        "multiplier": multiplier,
        "minimum_chunk_events": minimum_chunk_events,
        "chunks_per_stream": chunks_per_stream,
        "preflight_file_sha256": _sha256_file(P6_PREFLIGHT),
        "preflight_payload_sha256": declared_digest,
        "preflight_live_bindings_sha256": live_authority["bindings_sha256"],
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
    cell_count = seeds * schedules * arms
    chunk_events = max(
        int(evidence["minimum_chunk_events"]),
        rung // int(evidence["chunks_per_stream"]),
    )
    stream_chunk_bytes = math.ceil(
        int(evidence["stream_bytes"]) * chunk_events / int(evidence["events"]) * float(evidence["multiplier"])
    )
    checkpoint_bytes = math.ceil(int(evidence["state_bytes"]) * float(evidence["multiplier"]))
    progress_bytes = math.ceil(
        (int(evidence["state_bytes"]) * cell_count + 16 * 1024) * float(evidence["multiplier"])
    )
    proof_bytes = math.ceil(
        (int(evidence["state_bytes"]) * (cell_count + 1) + 32 * 1024) * float(evidence["multiplier"])
    )
    atomic_bytes = max(stream_chunk_bytes, checkpoint_bytes, progress_bytes, proof_bytes)
    return forecast_bytes / 1e9, atomic_bytes / 1e9


def _p6_verifier_atomic_write_projection(
    evidence: dict[str, Any], *, seeds: int, schedules: int, arms: int
) -> float:

    cell_count = seeds * schedules * arms
    verifier_bytes = math.ceil(
        (int(evidence["state_bytes"]) * cell_count * 2 + 64 * 1024) * float(evidence["multiplier"])
    )
    return verifier_bytes / 1e9


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


def _safe_evidence_path(value: object, evidence_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence path must be repository-relative")
    root = evidence_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("evidence path resolves outside evidence root")
    return path


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _p5_live_bindings(paths: tuple[str, ...]) -> list[dict[str, str]]:
    return [{"path": path, "file_sha256": _sha256_file(REPO_ROOT / path)} for path in paths]


def _p5_resolved_config(profile: str) -> dict[str, Any]:
    raw = yaml.safe_load(P5_CONFIG_PATH.read_text())
    if not isinstance(raw, dict):
        raise ValueError("live P5 config is not a mapping")
    config = json.loads(json.dumps(raw))
    profiles = config.pop("profiles", None)
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        raise ValueError(f"live P5 profile {profile!r} is missing")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("live P5 training config is missing")
    config["training"] = {**training, **profiles[profile]}
    config["profile"] = profile
    return config


def _p5_evidence_path(value: object, evidence_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("P5 evidence path is missing")
    root = evidence_root.resolve()
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError("P5 evidence path resolves outside evidence root")
    return path


def _p5_read_json(path: Path, label: str) -> dict[str, Any]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} is not an object")
    return loaded


def _p5_payload_digest_ok(payload: dict[str, Any]) -> bool:
    core = dict(payload)
    declared = core.pop("payload_sha256", None)
    return _is_sha256(declared) and _canonical_sha256(core) == declared


def _p5_strict_patterns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sesoi = payload.get("sesoi")
    if isinstance(sesoi, bool) or not isinstance(sesoi, int | float) or not math.isfinite(float(sesoi)):
        raise ValueError("P5 SESOI is invalid")
    threshold = float(sesoi)
    patterns: list[dict[str, Any]] = []
    fields = {64: "primary_contrasts_f64", 32: "secondary_contrasts_f32"}
    for frames, field in fields.items():
        frame_summary = _dotted_value(payload, f"frames.f{frames}")
        if not isinstance(frame_summary, dict) or frame_summary.get("off_ceiling") is not True:
            continue
        contrasts = payload.get(field)
        if not isinstance(contrasts, dict):
            continue
        for registered_mechanism in P5_MECHANISMS:
            if registered_mechanism == "exact_global":
                continue
            key = f"exact_minus_{registered_mechanism}"
            row = contrasts.get(key)
            if not isinstance(row, dict):
                continue
            count, lo, hi = row.get("n"), row.get("lo"), row.get("hi")
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 2
                or isinstance(lo, bool)
                or not isinstance(lo, int | float)
                or isinstance(hi, bool)
                or not isinstance(hi, int | float)
                or not math.isfinite(float(lo))
                or not math.isfinite(float(hi))
            ):
                continue
            direction: str | None = None
            if float(lo) > threshold:
                direction = "exact-over-factorized"
            elif float(hi) < -threshold:
                direction = "factorized-over-exact"
            if direction is None:
                continue
            mechanism = str(key).removeprefix("exact_minus_")
            patterns.append(
                {
                    "id": f"f{frames}-exact-minus-{mechanism}",
                    "frames": frames,
                    "mechanism": mechanism,
                    "direction": direction,
                    "primary_ci": dict(row),
                }
            )
    return patterns


def _p5_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} is not a finite probability")
    return number


def _p5_arm_scores(
    cell: dict[str, Any], *, frames: int, seed: int, mechanism: str
) -> tuple[float, float, float]:
    seed_results = cell.get("seed_results")
    unit = seed_results.get(str(seed)) if isinstance(seed_results, dict) else None
    mechanisms = unit.get("mechanisms") if isinstance(unit, dict) else None
    arm = mechanisms.get(mechanism) if isinstance(mechanisms, dict) else None
    if not isinstance(arm, dict):
        raise ValueError(f"P5 f{frames} seed {seed} {mechanism} arm is missing")
    trained = _p5_probability(
        _dotted_value(arm, "evaluation.heldout_combo_score"),
        f"P5 f{frames} seed {seed} {mechanism} trained score",
    )
    frozen = _p5_probability(
        _dotted_value(arm, "frozen.evaluation.heldout_combo_score"),
        f"P5 f{frames} seed {seed} {mechanism} frozen score",
    )
    chance = _p5_probability(
        _dotted_value(arm, "evaluation.chance"),
        f"P5 f{frames} seed {seed} {mechanism} chance",
    )
    return trained, frozen, chance


def _p5_seed_selection_problems(
    payload: dict[str, Any],
    cells: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:

    problems: list[str] = []
    seeds = [int(value) for value in payload.get("seeds", [])]
    if not seeds:
        return ["P5 seed selection has no configured seeds"]
    from ..studies.p5_context_verify import (
        EXPECTED_TRANSFORMER_PARAMETERS,
        FLOP_MATCH_TOLERANCE,
        RECURRENT_PARAMETER_DEFICIT,
        _expected_flops_per_step,
        _expected_match,
    )

    first_seed = seeds[0]
    trained_f64, frozen_f64, _ = _p5_arm_scores(
        cells[64], frames=64, seed=first_seed, mechanism="exact_global"
    )
    trainability_failed = trained_f64 - frozen_f64 <= P5_TRAINABILITY_MARGIN
    gate = payload.get("trainability_gate")
    if not isinstance(gate, dict) or gate.get("failed") is not trainability_failed:
        problems.append("P5 trainability gate does not independently recompute for seed selection")

    futility_margin = float(config["screen"]["futility_margin"])
    off_ceiling: dict[str, bool] = {}
    truncation: dict[str, dict[str, Any] | None] = {}
    for frames in P5_FRAME_COUNTS:
        cell = cells[frames]
        mechanisms = cell.get("mechanisms")
        if not isinstance(mechanisms, list) or "exact_global" not in mechanisms:
            raise ValueError(f"P5 f{frames} exact_global mechanism is missing")
        trained, _, chance = _p5_arm_scores(
            cell,
            frames=frames,
            seed=first_seed,
            mechanism="exact_global",
        )
        difficulty = cell.get("difficulty_calibration")
        clears_floor = isinstance(difficulty, dict) and difficulty.get("clears_floor") is True
        off = bool(clears_floor and chance + P5_CEILING_CHANCE_OFFSET <= trained <= P5_CEILING_UPPER)
        off_ceiling[f"f{frames}"] = off

        futility_evidence: dict[str, Any] | None = None
        if not trainability_failed and off and len(seeds) > 3:
            first_three = seeds[:3]
            seed_results = cell.get("seed_results")
            complete_three = isinstance(seed_results, dict) and all(
                isinstance(seed_results.get(str(seed)), dict)
                and seed_results[str(seed)].get("complete") is True
                for seed in first_three
            )
            if complete_three:
                deltas: dict[str, float] = {}
                for mechanism in mechanisms:
                    if mechanism == "exact_global":
                        continue
                    values = []
                    for seed in first_three:
                        exact, _, _ = _p5_arm_scores(
                            cell,
                            frames=frames,
                            seed=seed,
                            mechanism="exact_global",
                        )
                        factorized, _, _ = _p5_arm_scores(
                            cell,
                            frames=frames,
                            seed=seed,
                            mechanism=str(mechanism),
                        )
                        values.append(exact - factorized)
                    deltas[str(mechanism)] = sum(values) / len(values)
                if (
                    deltas
                    and all(value <= 0.0 for value in deltas.values())
                    and all(abs(value) < futility_margin for value in deltas.values())
                ):
                    futility_evidence = {
                        "paired_mean_deltas": deltas,
                        "futility_margin": futility_margin,
                        "seeds_kept": first_three,
                    }
        truncation[f"f{frames}"] = futility_evidence
        if trainability_failed or (len(seeds) > 1 and not off):
            expected = seeds[:1]
        elif futility_evidence is not None:
            expected = seeds[:3]
        else:
            expected = seeds
        if cell.get("expected_seeds") != expected:
            problems.append(f"P5 f{frames} exact licensed seed set drift")
        seed_results = cell.get("seed_results")
        if not isinstance(seed_results, dict) or set(seed_results) != {str(seed) for seed in expected}:
            problems.append(f"P5 f{frames} seed result coverage exceeds or omits licensed seeds")
        if cell.get("seeds_completed") != len(expected):
            problems.append(f"P5 f{frames} completed seed count drift")

        expected_parameters = {
            str(mechanism): (
                EXPECTED_TRANSFORMER_PARAMETERS[frames] - RECURRENT_PARAMETER_DEFICIT
                if mechanism == "recurrent"
                else EXPECTED_TRANSFORMER_PARAMETERS[frames]
            )
            for mechanism in mechanisms
        }
        parameter_block = cell.get("parameters")
        recurrent_deviation = RECURRENT_PARAMETER_DEFICIT / EXPECTED_TRANSFORMER_PARAMETERS[frames]
        if (
            not isinstance(parameter_block, dict)
            or parameter_block.get("frames") != frames
            or parameter_block.get("parameters") != expected_parameters
            or parameter_block.get("tolerance_fraction") != 0.005
            or isinstance(parameter_block.get("recurrent_fractional_deviation"), bool)
            or not isinstance(parameter_block.get("recurrent_fractional_deviation"), int | float)
            or not math.isclose(
                float(parameter_block["recurrent_fractional_deviation"]),
                recurrent_deviation,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            problems.append(f"P5 f{frames} parameter-matching control drift")

        dense_steps = int(config["training"]["dense_steps"])
        checkpoint_every = int(config["training"]["checkpoint_every"])
        batch_size = int(config["training"]["batch_size"])
        expected_flops = {
            str(mechanism): _expected_flops_per_step(
                frames,
                str(mechanism),
                batch_size,
            )
            for mechanism in mechanisms
        }
        dense_flops = expected_flops["exact_global"]
        expected_matches = {
            mechanism: _expected_match(
                dense_steps,
                dense_flops,
                flops,
                checkpoint_every,
                exact=mechanism == "exact_global",
            )
            for mechanism, flops in expected_flops.items()
        }
        compute = cell.get("compute")
        compute_rows = compute.get("per_mechanism") if isinstance(compute, dict) else None
        if (
            not isinstance(compute, dict)
            or compute.get("dense_reference_steps") != dense_steps
            or compute.get("dense_flops_per_step") != dense_flops
            or not isinstance(compute_rows, dict)
            or set(compute_rows) != set(expected_flops)
        ):
            problems.append(f"P5 f{frames} dense compute reference drift")
        else:
            for mechanism, flops in expected_flops.items():
                row = compute_rows[mechanism]
                matched = row.get("matched") if isinstance(row, dict) else None
                if (
                    not isinstance(row, dict)
                    or row.get("estimated_flops_per_step") != flops
                    or not isinstance(matched, dict)
                    or any(matched.get(key) != value for key, value in expected_matches[mechanism].items())
                    or matched.get("tolerance_fraction") != FLOP_MATCH_TOLERANCE
                    or row.get("estimated_total_flops_completed_seeds")
                    != expected_matches[mechanism]["arm_total_flops"] * len(expected)
                ):
                    problems.append(f"P5 f{frames} {mechanism} matched-compute control drift")

        trained_by_mechanism: dict[str, list[float]] = {}
        frozen_by_mechanism: dict[str, list[float]] = {}
        for mechanism in mechanisms:
            mechanism_name = str(mechanism)
            scored = [
                _p5_arm_scores(
                    cell,
                    frames=frames,
                    seed=seed,
                    mechanism=mechanism_name,
                )
                for seed in expected
            ]
            trained_by_mechanism[mechanism_name] = [row[0] for row in scored]
            frozen_by_mechanism[mechanism_name] = [row[1] for row in scored]
        expected_scores = {
            mechanism: _p5_paired_ci(values) for mechanism, values in trained_by_mechanism.items()
        }
        expected_frozen_scores = {
            mechanism: _p5_paired_ci(values) for mechanism, values in frozen_by_mechanism.items()
        }
        expected_contrasts: dict[str, dict[str, Any]] = {}
        exact_values = trained_by_mechanism["exact_global"]
        sesoi = float(payload["sesoi"])
        for mechanism, values in trained_by_mechanism.items():
            if mechanism == "exact_global":
                continue
            ci = _p5_paired_ci([left - right for left, right in zip(exact_values, values, strict=True)])
            if int(ci["n"]) < 2:
                classification = "undetermined"
            elif float(ci["lo"]) > sesoi:
                classification = "meaningful_positive"
            elif float(ci["hi"]) < -sesoi:
                classification = "meaningful_negative"
            elif float(ci["lo"]) >= -sesoi and float(ci["hi"]) <= sesoi:
                classification = "bounded_within_sesoi"
            else:
                classification = "undetermined"
            expected_contrasts[f"exact_minus_{mechanism}"] = {
                **ci,
                "classification": classification,
            }
        if cell.get("scores") != expected_scores:
            problems.append(f"P5 f{frames} score aggregates do not independently recompute")
        if cell.get("frozen_scores") != expected_frozen_scores:
            problems.append(f"P5 f{frames} frozen score aggregates do not independently recompute")
        if cell.get("paired_contrasts") != expected_contrasts:
            problems.append(f"P5 f{frames} paired contrasts do not independently recompute")
        if cell.get("off_ceiling") is not off:
            problems.append(f"P5 f{frames} off-ceiling decision drift")
        expected_staged_out = bool(len(seeds) > 1 and not off)
        if cell.get("staged_out") is not expected_staged_out:
            problems.append(f"P5 f{frames} staged-out decision drift")
        if cell.get("futility_truncated") is not (futility_evidence is not None):
            problems.append(f"P5 f{frames} futility truncation decision drift")
        if cell.get("futility_evidence") != futility_evidence:
            problems.append(f"P5 f{frames} futility evidence drift")

    expected_staging = {
        "off_ceiling": off_ceiling,
        "futility_truncated": truncation,
    }
    if payload.get("staging") != expected_staging:
        problems.append("P5 top-level staging authority drift")
    expected_curve = {
        mechanism: {
            f"f{frames}": cells[frames]["scores"][mechanism]
            for frames in sorted(P5_FRAME_COUNTS)
            if mechanism in cells[frames].get("scores", {})
        }
        for mechanism in P5_MECHANISMS
        if any(mechanism in cells[frames].get("scores", {}) for frames in P5_FRAME_COUNTS)
    }
    if payload.get("context_response_curve") != expected_curve:
        problems.append("P5 context response curve drifted from raw cell scores")
    return problems


def _p5_paired_ci(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = sum(values) / count if values else 0.0
    if count < 2:
        return {"n": count, "mean": mean, "lo": mean, "hi": mean, "half": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    half = 1.96 * math.sqrt(variance) / math.sqrt(count)
    return {"n": count, "mean": mean, "lo": mean - half, "hi": mean + half, "half": half}


def _p5_canonical_challenge_patterns(
    primary: dict[str, Any], challenge: dict[str, Any], evidence_root: Path
) -> list[dict[str, Any]]:

    runs = challenge.get("training_runs")
    if not isinstance(runs, list):
        raise ValueError("P5 challenge training runs are missing")
    by_seed = {int(row["seed"]): row for row in runs if isinstance(row, dict)}
    if set(by_seed) != set(P5_FRESH_SEEDS):
        raise ValueError("P5 challenge training seed coverage drift")
    sesoi = float(primary["sesoi"])
    result: list[dict[str, Any]] = []
    for pattern in _p5_strict_patterns(primary):
        frames = int(pattern["frames"])
        mechanism = str(pattern["mechanism"])
        units: list[dict[str, Any]] = []
        for seed in P5_FRESH_SEEDS:
            bindings = by_seed[seed].get("cell_receipts")
            if not isinstance(bindings, dict):
                raise ValueError(f"P5 challenge seed {seed} cell bindings are missing")
            target_binding = bindings.get(f"f{frames}")
            trainability_binding = bindings.get("f64")
            if not isinstance(target_binding, dict) or not isinstance(trainability_binding, dict):
                raise ValueError(f"P5 challenge seed {seed} required cell binding is missing")
            target = _p5_read_json(
                _p5_evidence_path(target_binding.get("path"), evidence_root),
                f"P5 challenge seed {seed} f{frames} cell",
            )
            f64 = _p5_read_json(
                _p5_evidence_path(trainability_binding.get("path"), evidence_root),
                f"P5 challenge seed {seed} f64 cell",
            )
            exact, _, chance = _p5_arm_scores(
                target,
                frames=frames,
                seed=seed,
                mechanism="exact_global",
            )
            factorized, _, _ = _p5_arm_scores(
                target,
                frames=frames,
                seed=seed,
                mechanism=mechanism,
            )
            trained_f64, frozen_f64, _ = _p5_arm_scores(
                f64,
                frames=64,
                seed=seed,
                mechanism="exact_global",
            )
            trainability_delta = trained_f64 - frozen_f64
            difficulty = target.get("difficulty_calibration")
            off_ceiling = bool(
                isinstance(difficulty, dict)
                and difficulty.get("clears_floor") is True
                and chance + P5_CEILING_CHANCE_OFFSET <= exact <= P5_CEILING_UPPER
            )
            units.append(
                {
                    "seed": seed,
                    "delta": exact - factorized,
                    "trainability_delta": trainability_delta,
                    "trainability_ok": trainability_delta > P5_TRAINABILITY_MARGIN,
                    "off_ceiling": off_ceiling,
                }
            )
        fresh_ci = _p5_paired_ci([float(unit["delta"]) for unit in units])
        same_direction = (
            float(fresh_ci["lo"]) > sesoi
            if pattern["direction"] == "exact-over-factorized"
            else float(fresh_ci["hi"]) < -sesoi
        )
        verified = bool(
            same_direction
            and all(unit["trainability_ok"] is True for unit in units)
            and all(unit["off_ceiling"] is True for unit in units)
        )
        result.append(
            {
                "id": pattern["id"],
                "direction": pattern["direction"],
                "fresh_training_units": units,
                "fresh_ci": fresh_ci,
                "tie_is_null": True,
                "strict_direction_reproduced": same_direction,
                "programmatic_pattern_verified": verified,
                "scientific_promotion_allowed": False,
                "outcome": "favorable-programmatic-only" if verified else "null",
            }
        )
    return result


def _p5_frame_summary(cell: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "complete",
        "off_ceiling",
        "staged_out",
        "futility_truncated",
        "seeds_completed",
        "scores",
        "paired_contrasts",
        "all_ok",
    )
    return {field: cell.get(field) for field in fields}


def _p5_normalize_cached_seed(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if normalized.get("resumed_from_complete_receipt") is True:
        normalized.pop("resumed_from_complete_receipt")
    return normalized


def _p5_seed_artifact_problems(
    *,
    seed: dict[str, Any],
    seed_dir: Path,
    mechanisms: list[str],
    config_sha: str,
    registry_sha: str,
    source_sha: str,
    checkpoint_sha: str,
) -> list[str]:

    problems: list[str] = []
    try:
        seed_path = seed_dir / "seed_result.json"
        persisted_seed = _p5_read_json(seed_path, "P5 durable seed result")
        if _p5_normalize_cached_seed(seed) != _p5_normalize_cached_seed(persisted_seed):
            problems.append("durable seed result differs from its cell receipt")
        seed_value = seed.get("seed")
        if (
            not isinstance(seed_value, int)
            or isinstance(seed_value, bool)
            or seed.get("config_sha256") != config_sha
            or seed.get("registry_sha256") != registry_sha
            or seed.get("source_bindings_sha256") != source_sha
            or seed.get("checkpoint_requirements_sha256") != checkpoint_sha
        ):
            problems.append("durable seed identity drift")
        embedded_arms = seed.get("mechanisms")
        if not isinstance(embedded_arms, dict) or set(embedded_arms) != set(mechanisms):
            problems.append("durable seed arm coverage drift")
            return problems
        for mechanism in mechanisms:
            embedded = embedded_arms[mechanism]
            if not isinstance(embedded, dict):
                problems.append(f"{mechanism} embedded arm is invalid")
                continue
            arm_dir = seed_dir / mechanism
            arm_receipt = _p5_read_json(arm_dir / "arm_receipt.json", f"P5 {mechanism} arm receipt")
            checkpoint_path = arm_dir / "checkpoint.pt"
            matched = embedded.get("matched")
            training = embedded.get("training")
            requested_steps = matched.get("steps") if isinstance(matched, dict) else None
            checkpoint = arm_receipt.get("checkpoint")
            if (
                not checkpoint_path.is_file()
                or not isinstance(checkpoint, dict)
                or checkpoint.get("sha256") != _sha256_file(checkpoint_path)
                or arm_receipt.get("schema") != "mop-custom-substrate-arm/v1"
                or arm_receipt.get("objective") != "predictive"
                or arm_receipt.get("seed") != seed_value
                or arm_receipt.get("complete") is not True
                or arm_receipt.get("config_sha256") != config_sha
                or arm_receipt.get("data_sha256") != seed.get("data_sha256")
                or arm_receipt.get("requirements_sha256") != checkpoint_sha
                or arm_receipt.get("initial_state_sha256") != embedded.get("initial_state_sha256")
                or arm_receipt.get("requested_steps") != requested_steps
                or not isinstance(arm_receipt.get("completed_steps"), int)
                or arm_receipt["completed_steps"] < requested_steps
                or not isinstance(training, dict)
                or training.get("complete") is not True
                or training.get("requirements_sha256") != checkpoint_sha
                or training.get("completed_steps") != arm_receipt.get("completed_steps")
                or training.get("final_state_sha256") != arm_receipt.get("final_state_sha256")
            ):
                problems.append(f"{mechanism} durable arm or checkpoint identity drift")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        problems.append(f"durable seed artifact invalid: {exc}")
    return problems


def _p5_tensor_state_sha256(state: object, label: str) -> str:

    import torch

    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} is not a nonempty tensor state")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} contains a non-tensor entry")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(
                list(tensor.shape),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _p5_artifact_evidence_problems(
    evidence: object,
    *,
    expected_seeds: set[str],
    evidence_root: Path,
    label: str,
) -> list[str]:

    problems: list[str] = []
    if not isinstance(evidence, dict) or set(evidence) != expected_seeds:
        return [f"{label} seed artifact evidence coverage drift"]
    for seed, seed_evidence in evidence.items():
        if not isinstance(seed_evidence, dict):
            problems.append(f"{label} seed {seed} artifact evidence invalid")
            continue
        seed_result = seed_evidence.get("seed_result")
        arms = seed_evidence.get("arms")
        if not isinstance(seed_result, dict) or not isinstance(arms, dict) or set(arms) != set(P5_MECHANISMS):
            problems.append(f"{label} seed {seed} artifact evidence shape drift")
            continue
        bindings: list[tuple[str, object]] = [("seed_result", seed_result)]
        for mechanism, arm in arms.items():
            if not isinstance(arm, dict):
                problems.append(f"{label} seed {seed} {mechanism} artifact evidence invalid")
                continue
            bindings.extend(
                (
                    (f"{mechanism} arm receipt", arm.get("arm_receipt")),
                    (f"{mechanism} checkpoint", arm.get("checkpoint")),
                )
            )
        for binding_label, binding in bindings:
            if not isinstance(binding, dict):
                problems.append(f"{label} seed {seed} {binding_label} binding missing")
                continue
            try:
                path = _p5_evidence_path(binding.get("path"), evidence_root)
                if binding.get("sha256") != _sha256_file(path):
                    problems.append(f"{label} seed {seed} {binding_label} file hash drift")
                if binding_label.endswith("checkpoint") and (
                    not _is_sha256(binding.get("model_state_sha256"))
                    or not _is_sha256(binding.get("target_state_sha256"))
                ):
                    problems.append(f"{label} seed {seed} {binding_label} state hashes invalid")
            except (OSError, TypeError, ValueError) as exc:
                problems.append(f"{label} seed {seed} {binding_label} invalid: {exc}")
        try:
            seed_path = _p5_evidence_path(seed_result.get("path"), evidence_root)
            if seed_path.name != "seed_result.json" or seed_path.parent.name != f"seed_{seed}":
                raise ValueError("seed result path is not canonical")
            durable_seed = _p5_read_json(seed_path, f"{label} seed {seed} durable result")
            durable_arms = durable_seed.get("mechanisms")
            if not isinstance(durable_arms, dict) or set(durable_arms) != set(P5_MECHANISMS):
                raise ValueError("durable seed arm coverage drift")
            for mechanism, arm in arms.items():
                if not isinstance(arm, dict):
                    continue
                arm_binding = arm.get("arm_receipt")
                checkpoint_binding = arm.get("checkpoint")
                if not isinstance(arm_binding, dict) or not isinstance(checkpoint_binding, dict):
                    raise ValueError(f"{mechanism} durable binding is missing")
                expected_arm_path = seed_path.parent / mechanism / "arm_receipt.json"
                expected_checkpoint_path = seed_path.parent / mechanism / "checkpoint.pt"
                arm_path = _p5_evidence_path(arm_binding.get("path"), evidence_root)
                checkpoint_path = _p5_evidence_path(checkpoint_binding.get("path"), evidence_root)
                if arm_path != expected_arm_path or checkpoint_path != expected_checkpoint_path:
                    raise ValueError(f"{mechanism} artifact path is not canonical")
                arm_receipt = _p5_read_json(arm_path, f"{label} seed {seed} {mechanism} arm")
                try:
                    import torch

                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                except Exception as exc:
                    raise ValueError(f"{mechanism} checkpoint is unreadable: {exc}") from exc
                if not isinstance(checkpoint, dict):
                    raise ValueError(f"{mechanism} checkpoint is not an object")
                model_sha = _p5_tensor_state_sha256(
                    checkpoint.get("model"),
                    f"{label} seed {seed} {mechanism} checkpoint model",
                )
                target_sha = _p5_tensor_state_sha256(
                    checkpoint.get("target"),
                    f"{label} seed {seed} {mechanism} checkpoint target",
                )
                durable_arm = durable_arms[mechanism]
                training = durable_arm.get("training") if isinstance(durable_arm, dict) else None
                if (
                    checkpoint_binding.get("model_state_sha256") != model_sha
                    or checkpoint_binding.get("target_state_sha256") != target_sha
                    or not isinstance(training, dict)
                    or training.get("final_state_sha256") != model_sha
                    or arm_receipt.get("final_state_sha256") != model_sha
                    or arm_receipt.get("target_state_sha256") != target_sha
                ):
                    problems.append(f"{label} seed {seed} {mechanism} checkpoint state hash drift")
        except (KeyError, OSError, TypeError, ValueError) as exc:
            problems.append(f"{label} seed {seed} checkpoint state authority invalid: {exc}")
    return problems


def _p5_screen_authority_problems(
    payload: dict[str, Any], evidence_root: Path, *, validate_ancestors: bool = True
) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P5_SCREEN_SCHEMA or not _p5_payload_digest_ok(payload):
            problems.append("P5 screen schema or payload digest drift")
        profile = payload.get("profile")
        if profile not in {"p5smoke", "p5pilot"}:
            raise ValueError("P5 screen profile is not canonical")
        config = _p5_resolved_config(str(profile))
        bindings = _p5_live_bindings(P5_SOURCE_PATHS)
        bindings_sha = _canonical_sha256(bindings)
        if payload.get("source_bindings") != bindings:
            problems.append("P5 screen live source bindings drift")
        if payload.get("source_bindings_sha256") != bindings_sha:
            problems.append("P5 screen aggregate source digest drift")
        if payload.get("config_sha256") != _canonical_sha256(config):
            problems.append("P5 screen resolved config digest drift")
        cells_config = config.get("cells")
        if not isinstance(cells_config, list):
            raise ValueError("P5 screen cell registry is missing")
        registry_sha = _canonical_sha256(cells_config)
        checkpoint_sha = _canonical_sha256(
            {"registry_sha256": registry_sha, "source_bindings_sha256": bindings_sha}
        )
        if payload.get("cell_registry_sha256") != registry_sha:
            problems.append("P5 screen cell registry digest drift")
        if payload.get("checkpoint_requirements_sha256") != checkpoint_sha:
            problems.append("P5 screen checkpoint requirements digest drift")
        expected_serial = [f"f{row['frames']}_{row['mechanism']}" for row in cells_config]
        if payload.get("serial_order") != expected_serial:
            problems.append("P5 screen serial order drift")
        expected_seeds = [int(value) for value in config["training"]["seeds"]]
        if payload.get("seeds") != expected_seeds:
            problems.append("P5 screen configured seed set drift")
        if payload.get("complete") is not True or payload.get("all_ok") is not True:
            problems.append("P5 screen is incomplete or all_ok false")
        if payload.get("resumable") is not False or payload.get("problems") != []:
            problems.append("P5 screen retains resumability or problems")
        if (
            any(
                payload.get(flag) is not False
                for flag in (
                    "stopped_for_wall_budget",
                    "stopped_for_disk_floor",
                    "stopped_for_required_arm_refusal",
                )
            )
            or payload.get("required_arm_failure") is not None
        ):
            problems.append("P5 screen retains an operational or arm stop")
        promotion = payload.get("promotion")
        if not isinstance(promotion, dict) or (
            promotion.get("confirmatory_promotable") is not False
            or promotion.get("refused_by_construction") is not True
            or promotion.get("category_9_possible") is not False
        ):
            problems.append("P5 screen promotion refusal drift")

        gate = payload.get("trainability_gate")
        if not isinstance(gate, dict) or gate.get("applies") is not True or gate.get("evaluated") is not True:
            problems.append("P5 screen trainability gate is incomplete")
            gate = gate if isinstance(gate, dict) else {}
        terminal = payload.get("terminal_scientific_stop") is True
        gate_failed = gate.get("failed") is True
        if payload.get("trainability_gate_failed") is not gate_failed or terminal is not gate_failed:
            problems.append("P5 screen terminal state and trainability gate disagree")
        expected_status = "terminal-scientific-null" if terminal else "complete"
        expected_reason = "f64-trainability-gate-null" if terminal else None
        expected_outcome = "null" if terminal else "clears-margin"
        if (
            payload.get("execution_status") != expected_status
            or payload.get("terminal_stop_reason") != expected_reason
            or gate.get("outcome") != expected_outcome
        ):
            problems.append("P5 screen terminal status contract drift")

        patterns = _p5_strict_patterns(payload)
        hint = bool(patterns)
        if payload.get("fresh_challenge_required") is not hint:
            problems.append("P5 screen fresh challenge authorization hint drift")
        if terminal and hint:
            problems.append("P5 terminal null cannot authorize a fresh challenge")

        run_dir_value = "runs/p5_context/p5smoke" if profile == "p5smoke" else "runs/p5_context/p5pilot"
        run_dir = _p5_evidence_path(run_dir_value, evidence_root)
        raw_path = run_dir / "p5_context_receipt.json"
        raw = _p5_read_json(raw_path, "P5 raw screen receipt")
        if raw != payload:
            problems.append("P5 published screen differs from its raw run receipt")
        resolved = _p5_read_json(run_dir / "resolved_config.json", "P5 resolved config")
        if resolved != config:
            problems.append("P5 raw resolved config differs from live profile config")

        by_frame: dict[int, list[str]] = {frames: [] for frames in P5_FRAME_COUNTS}
        for row in cells_config:
            by_frame[int(row["frames"])].append(str(row["mechanism"]))
        top_frames = payload.get("frames")
        if not isinstance(top_frames, dict) or set(top_frames) != {
            f"f{frames}" for frames in P5_FRAME_COUNTS
        }:
            problems.append("P5 screen frame coverage drift")
            top_frames = top_frames if isinstance(top_frames, dict) else {}
        loaded_cells: dict[int, dict[str, Any]] = {}
        for frames in P5_FRAME_COUNTS:
            cell = _p5_read_json(
                run_dir / "frames" / f"f{frames}" / "cell_receipt.json",
                f"P5 f{frames} cell receipt",
            )
            loaded_cells[frames] = cell
            if (
                cell.get("schema") != "mop-p5-context-cell/v1"
                or cell.get("frames") != frames
                or cell.get("mechanisms") != by_frame[frames]
                or cell.get("complete") is not True
                or cell.get("all_ok") is not True
                or cell.get("problems") != []
            ):
                problems.append(f"P5 f{frames} cell contract drift")
            if top_frames.get(f"f{frames}") != _p5_frame_summary(cell):
                problems.append(f"P5 f{frames} top summary drift")
            cell_seed_values = cell.get("expected_seeds")
            seed_results = cell.get("seed_results")
            if (
                not isinstance(cell_seed_values, list)
                or not cell_seed_values
                or not set(int(value) for value in cell_seed_values) <= set(expected_seeds)
                or not isinstance(seed_results, dict)
                or set(seed_results) != {str(int(value)) for value in cell_seed_values}
            ):
                problems.append(f"P5 f{frames} seed coverage drift")
                continue
            for seed_value in cell_seed_values:
                seed = seed_results[str(int(seed_value))]
                if (
                    not isinstance(seed, dict)
                    or seed.get("schema") != "mop-p5-context-seed/v1"
                    or seed.get("complete") is not True
                    or seed.get("config_sha256") != payload.get("config_sha256")
                    or seed.get("registry_sha256") != registry_sha
                    or seed.get("source_bindings_sha256") != bindings_sha
                    or seed.get("checkpoint_requirements_sha256") != checkpoint_sha
                ):
                    problems.append(f"P5 f{frames} seed {seed_value} identity drift")
                    continue
                mechanisms = seed.get("mechanisms")
                if not isinstance(mechanisms, dict) or set(mechanisms) != set(by_frame[frames]):
                    problems.append(f"P5 f{frames} seed {seed_value} mechanism coverage drift")
                    continue
                if any(
                    not isinstance(arm, dict)
                    or arm.get("training", {}).get("complete") is not True
                    or arm.get("training", {}).get("requirements_sha256") != checkpoint_sha
                    or arm.get("matched", {}).get("matched_ok") is not True
                    for arm in mechanisms.values()
                ):
                    problems.append(f"P5 f{frames} seed {seed_value} arm identity drift")
                problems.extend(
                    f"P5 f{frames} seed {seed_value}: {problem}"
                    for problem in _p5_seed_artifact_problems(
                        seed=seed,
                        seed_dir=run_dir / "frames" / f"f{frames}" / f"seed_{seed_value}",
                        mechanisms=by_frame[frames],
                        config_sha=payload["config_sha256"],
                        registry_sha=registry_sha,
                        source_sha=bindings_sha,
                        checkpoint_sha=checkpoint_sha,
                    )
                )
        problems.extend(_p5_seed_selection_problems(payload, loaded_cells, config))
        if payload.get("primary_contrasts_f64") != loaded_cells[64].get("paired_contrasts"):
            problems.append("P5 f64 primary contrast binding drift")
        if payload.get("secondary_contrasts_f32") != loaded_cells[32].get("paired_contrasts"):
            problems.append("P5 f32 secondary contrast binding drift")

        if profile == "p5pilot" and validate_ancestors:
            smoke_path = _p5_evidence_path("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", evidence_root)
            smoke = _p5_read_json(smoke_path, "P5 smoke ancestor")
            problems.extend(
                f"P5 pilot smoke ancestor: {problem}"
                for problem in _p5_screen_authority_problems(smoke, evidence_root, validate_ancestors=False)
            )
            if (
                smoke.get("profile") != "p5smoke"
                or smoke.get("execution_status") != "complete"
                or smoke.get("terminal_scientific_stop") is not False
                or smoke.get("trainability_gate_failed") is not False
                or smoke.get("fresh_challenge_required") is not False
            ):
                problems.append("P5 pilot smoke ancestor did not clear the smoke gate")
            grid_path = _p5_evidence_path("proof/P5_TRAINGRID_MEMORY_TRACE.json", evidence_root)
            grid = _p5_read_json(grid_path, "P5 training-grid ancestor")
            problems.extend(
                f"P5 pilot grid ancestor: {problem}"
                for problem in _p5_grid_authority_problems(grid, evidence_root)
            )
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"P5 screen authority invalid: {exc}")
    return problems


def _p5_grid_authority_problems(payload: dict[str, Any], evidence_root: Path) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P5_GRID_SCHEMA or not _p5_payload_digest_ok(payload):
            problems.append("P5 training-grid schema or payload digest drift")
        bindings = _p5_live_bindings(P5_GRID_SOURCE_PATHS)
        bindings_sha = _canonical_sha256(bindings)
        if payload.get("source_bindings") != bindings:
            problems.append("P5 training-grid live source bindings drift")
        if payload.get("source_bindings_sha256") != bindings_sha:
            problems.append("P5 training-grid aggregate source digest drift")
        if payload.get("all_ok") is not True:
            problems.append("P5 training-grid all_ok is false")
        boundary_sha = _sha256_file(P5_BOUNDARY_TRACE)
        if payload.get("cited_boundary_trace") != {
            "path": "proof/P5_MEMORY_BOUNDARY_TRACE.json",
            "sha256": boundary_sha,
        }:
            problems.append("P5 training-grid boundary trace binding drift")
        config = _p5_resolved_config("p5pilot")
        expected_cells = config["cells"]
        expected_grid_config = {
            "cells": expected_cells,
            "batch_rows": [4, 1],
            "repeats": 3,
            "seed": 0,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "child_memory_guard_gb": 12.0,
            "device": "cpu",
        }
        if payload.get("config") != expected_grid_config:
            problems.append("P5 training-grid exact config drift")
        claim = payload.get("claim_boundary")
        if not isinstance(claim, dict) or (
            claim.get("mechanics_only") is not True
            or claim.get("moves_no_category") is not True
            or claim.get("naive_formula_is_diagnostic_only") is not True
        ):
            problems.append("P5 training-grid claim boundary drift")
        if payload.get("problems") not in (None, []) or payload.get("scientific_promotion") not in (
            None,
            False,
        ):
            problems.append("P5 training-grid retains problems or promotion")
        progress_binding = payload.get("atomic_progress")
        if not isinstance(progress_binding, dict):
            raise ValueError("P5 training-grid progress binding is missing")
        progress_path = _p5_evidence_path(progress_binding.get("path"), evidence_root)
        progress = _p5_read_json(progress_path, "P5 training-grid progress")
        identity = progress.get("identity")
        expected_identity = {
            "script_sha256": _sha256_file(REPO_ROOT / "scripts/p5_traingrid_memory_probe.py"),
            "source_bindings": bindings,
            "source_bindings_sha256": bindings_sha,
            "boundary_trace_sha256": boundary_sha,
            "cells": expected_cells,
            "batch_rows": [4, 1],
            "repeats": 3,
            "seed": 0,
            "mask_ratio": 0.5,
            "ema_decay": 0.99,
            "child_memory_guard_gb": 12.0,
            "device": "cpu",
        }
        rows = progress.get("rows")
        expected_row_keys = {
            f"f{item['frames']}:{item['mechanism']}:b{batch}:r{repeat}"
            for item in expected_cells
            for batch in (4, 1)
            for repeat in range(3)
        }
        if (
            progress_binding.get("sha256") != _sha256_file(progress_path)
            or progress_binding.get("identity_sha256") != _canonical_sha256(expected_identity)
            or progress_binding.get("completed_rows") != 72
            or progress.get("schema") != "mop-p5-traingrid-memory-progress/v1"
            or identity != expected_identity
            or progress.get("identity_sha256") != _canonical_sha256(expected_identity)
            or progress.get("complete") is not True
            or progress.get("completed_rows") != 72
            or not isinstance(rows, dict)
            or set(rows) != expected_row_keys
        ):
            problems.append("P5 training-grid live progress identity drift")
        elif any(
            not isinstance(row, dict)
            or row.get("ok") is not True
            or row.get("loss_finite") is not True
            or row.get("memory_guard_exceeded") is not False
            for row in rows.values()
        ):
            problems.append("P5 training-grid progress row validity drift")
        receipt_rows = payload.get("cells")
        if (
            not isinstance(receipt_rows, list)
            or len(receipt_rows) != 72
            or any(
                not isinstance(row, dict)
                or row.get("ok") is not True
                or row.get("loss_finite") is not True
                or row.get("memory_guard_exceeded") is not False
                for row in (receipt_rows if isinstance(receipt_rows, list) else [])
            )
        ):
            problems.append("P5 training-grid final row validity drift")
        elif isinstance(rows, dict):
            joined: dict[str, dict[str, Any]] = {}
            for row in receipt_rows:
                key = f"f{row.get('frames')}:{row.get('mechanism')}:b{row.get('batch')}:r{row.get('repeat')}"
                normalized = dict(row)
                if normalized.get("resumed_from_atomic_progress") is True:
                    normalized.pop("resumed_from_atomic_progress")
                if key in joined or key not in expected_row_keys:
                    problems.append("P5 training-grid final row coordinate drift")
                    break
                joined[key] = normalized
            if set(joined) != expected_row_keys or any(
                joined.get(key) != rows.get(key) for key in expected_row_keys
            ):
                problems.append("P5 training-grid final receipt does not join atomic progress")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"P5 training-grid authority invalid: {exc}")
    return problems


def _p5_challenge_seed_config(seed: int) -> dict[str, Any]:
    config = _p5_resolved_config("p5pilot")
    config["profile"] = f"p5fresh-seed-{seed}"
    config["training"]["seeds"] = [seed]
    return config


def _p5_challenge_authority_problems(payload: dict[str, Any], evidence_root: Path) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P5_CHALLENGE_SCHEMA or not _p5_payload_digest_ok(payload):
            problems.append("P5 fresh challenge schema or payload digest drift")
        if (
            payload.get("complete") is not True
            or payload.get("all_ok") is not True
            or payload.get("verification_ready") is not True
            or payload.get("resumable") is not False
            or payload.get("problems") != []
        ):
            problems.append("P5 fresh challenge is incomplete, resumable, or reports problems")
        if payload.get("source_bindings") != _p5_live_bindings(P5_CHALLENGE_SOURCE_PATHS):
            problems.append("P5 fresh challenge live source bindings drift")
        if payload.get("scientific_promotion") is not False or payload.get("promotion") != {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "scientific_capability_claim": False,
        }:
            problems.append("P5 fresh challenge promotion refusal drift")

        primary_path = _p5_evidence_path("proof/P5_CONTEXT_CAPABILITY_PILOT.json", evidence_root)
        primary = _p5_read_json(primary_path, "P5 challenge primary")
        problems.extend(
            f"P5 challenge primary: {problem}"
            for problem in _p5_screen_authority_problems(primary, evidence_root)
        )
        if primary.get("fresh_challenge_required") is not True:
            problems.append("P5 fresh challenge primary does not authorize challenge execution")
        primary_binding = payload.get("primary_receipt")
        if not isinstance(primary_binding, dict) or (
            primary_binding.get("path") != "proof/P5_CONTEXT_CAPABILITY_PILOT.json"
            or primary_binding.get("sha256") != _sha256_file(primary_path)
            or primary_binding.get("payload_sha256") != primary.get("payload_sha256")
        ):
            problems.append("P5 fresh challenge primary proof binding drift")
        expected_patterns = [
            {key: value for key, value in pattern.items() if key != "primary_ci"}
            for pattern in _p5_strict_patterns(primary)
        ]
        if not expected_patterns or payload.get("patterns") != expected_patterns:
            problems.append("P5 fresh challenge pattern authorization drift")
        if (
            payload.get("fresh_training_seeds") != list(P5_FRESH_SEEDS)
            or payload.get("fresh_seeds_disjoint_from_primary") is not True
        ):
            problems.append("P5 fresh challenge disjoint seed contract drift")
        if set(P5_FRESH_SEEDS) & set(int(value) for value in primary.get("seeds", [])):
            problems.append("P5 fresh challenge seeds overlap the primary")
        controls = payload.get("controls")
        if not isinstance(controls, dict) or any(
            controls.get(field) is not True
            for field in (
                "shared_primary_training_contract",
                "matched_parameter_and_flop_contract",
                "same_initialization_frozen_control",
                "f64_trainability_gate",
                "threshold_tie_is_null",
                "isolated_full_surface_seed_subruns",
            )
        ):
            problems.append("P5 fresh challenge controls drift")
        run_dir_value = payload.get("run_dir")
        run_dir = _p5_evidence_path(run_dir_value, evidence_root)
        run_dir_relative = str(run_dir.relative_to(evidence_root))
        expected_checkpoint_globs = [
            f"{run_dir_relative}/seed_*/frames/f*/seed_*/*/checkpoint.pt",
            f"{run_dir_relative}/seed_*/frames/f*/seed_*/*/arm_receipt.json",
            f"{run_dir_relative}/seed_*/frames/f*/seed_*/seed_result.json",
            f"{run_dir_relative}/seed_*/frames/f*/cell_receipt.json",
            f"{run_dir_relative}/seed_*/p5_context_receipt.json",
            f"{run_dir_relative}/seed_*/resolved_config.json",
        ]
        if payload.get("checkpoint_globs") != expected_checkpoint_globs:
            problems.append("P5 fresh challenge checkpoint glob authority drift")
        rows = payload.get("training_runs")
        if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(P5_FRESH_SEEDS):
            problems.append("P5 fresh challenge run coverage drift")
            rows = rows if isinstance(rows, list) else []
        expected_sources = _p5_live_bindings(P5_SOURCE_PATHS)
        expected_source_sha = _canonical_sha256(expected_sources)
        for row in rows:
            if not isinstance(row, dict):
                problems.append("P5 fresh challenge run row is not an object")
                continue
            seed = int(row.get("seed", -1))
            if (
                row.get("complete") is not True
                or row.get("all_ok") is not True
                or row.get("resumable") is not False
                or row.get("problems") != []
            ):
                problems.append(f"P5 fresh challenge seed {seed} is not complete and valid")
            raw_binding = row.get("raw_receipt")
            config_binding = row.get("resolved_config")
            cell_bindings = row.get("cell_receipts")
            if (
                not isinstance(raw_binding, dict)
                or not isinstance(config_binding, dict)
                or not isinstance(cell_bindings, dict)
                or set(cell_bindings) != {f"f{frames}" for frames in P5_FRAME_COUNTS}
            ):
                problems.append(f"P5 fresh challenge seed {seed} artifact bindings drift")
                continue
            raw_path = _p5_evidence_path(raw_binding.get("path"), evidence_root)
            config_path = _p5_evidence_path(config_binding.get("path"), evidence_root)
            raw = _p5_read_json(raw_path, f"P5 fresh seed {seed} raw receipt")
            resolved = _p5_read_json(config_path, f"P5 fresh seed {seed} config")
            expected_config = _p5_challenge_seed_config(seed)
            registry_sha = _canonical_sha256(expected_config["cells"])
            checkpoint_sha = _canonical_sha256(
                {
                    "registry_sha256": registry_sha,
                    "source_bindings_sha256": expected_source_sha,
                }
            )
            terminal = raw.get("terminal_scientific_stop") is True
            expected_status = "terminal-scientific-null" if terminal else "complete"
            expected_reason = "f64-trainability-gate-null" if terminal else None
            gate = raw.get("trainability_gate")
            if (
                raw_binding.get("sha256") != _sha256_file(raw_path)
                or raw_binding.get("payload_sha256") != raw.get("payload_sha256")
                or not _p5_payload_digest_ok(raw)
                or config_binding.get("sha256") != _sha256_file(config_path)
                or resolved != expected_config
                or raw.get("config_sha256") != _canonical_sha256(expected_config)
                or raw.get("cell_registry_sha256") != registry_sha
                or raw.get("source_bindings") != expected_sources
                or raw.get("source_bindings_sha256") != expected_source_sha
                or raw.get("checkpoint_requirements_sha256") != checkpoint_sha
                or raw.get("schema") != P5_SCREEN_SCHEMA
                or raw.get("profile") != f"p5fresh-seed-{seed}"
                or raw.get("seeds") != [seed]
                or raw.get("serial_order")
                != [f"f{item['frames']}_{item['mechanism']}" for item in expected_config["cells"]]
                or raw.get("complete") is not True
                or raw.get("all_ok") is not True
                or raw.get("resumable") is not False
                or raw.get("problems") != []
                or raw.get("execution_status") != expected_status
                or raw.get("terminal_stop_reason") != expected_reason
                or raw.get("trainability_gate_failed") is not terminal
                or not isinstance(gate, dict)
                or gate.get("applies") is not True
                or gate.get("evaluated") is not True
                or gate.get("failed") is not terminal
                or gate.get("outcome") != ("null" if terminal else "clears-margin")
                or raw.get("fresh_challenge_required") is not False
            ):
                problems.append(f"P5 fresh challenge seed {seed} raw authority drift")
            by_frame: dict[int, list[str]] = {frames: [] for frames in P5_FRAME_COUNTS}
            for item in expected_config["cells"]:
                by_frame[int(item["frames"])].append(str(item["mechanism"]))
            for frames in P5_FRAME_COUNTS:
                binding = cell_bindings[f"f{frames}"]
                if not isinstance(binding, dict):
                    problems.append(f"P5 fresh seed {seed} f{frames} binding drift")
                    continue
                cell_path = _p5_evidence_path(binding.get("path"), evidence_root)
                cell = _p5_read_json(cell_path, f"P5 fresh seed {seed} f{frames} cell")
                seed_payloads = cell.get("seed_results")
                unit = seed_payloads.get(str(seed)) if isinstance(seed_payloads, dict) else None
                if (
                    binding.get("sha256") != _sha256_file(cell_path)
                    or cell.get("complete") is not True
                    or cell.get("all_ok") is not True
                    or cell.get("problems") != []
                    or raw.get("frames", {}).get(f"f{frames}") != _p5_frame_summary(cell)
                    or not isinstance(unit, dict)
                    or unit.get("source_bindings_sha256") != expected_source_sha
                    or unit.get("checkpoint_requirements_sha256") != checkpoint_sha
                    or any(
                        arm.get("training", {}).get("requirements_sha256") != checkpoint_sha
                        for arm in unit.get("mechanisms", {}).values()
                        if isinstance(arm, dict)
                    )
                ):
                    problems.append(f"P5 fresh seed {seed} f{frames} cell authority drift")
                if isinstance(unit, dict):
                    problems.extend(
                        f"P5 fresh seed {seed} f{frames}: {problem}"
                        for problem in _p5_seed_artifact_problems(
                            seed=unit,
                            seed_dir=cell_path.parent / f"seed_{seed}",
                            mechanisms=by_frame[frames],
                            config_sha=str(raw.get("config_sha256")),
                            registry_sha=registry_sha,
                            source_sha=expected_source_sha,
                            checkpoint_sha=checkpoint_sha,
                        )
                    )
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"P5 fresh challenge authority invalid: {exc}")
    return problems


def _p5_verifier_authority_problems(payload: dict[str, Any], evidence_root: Path) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P5_VERIFIER_SCHEMA or not _p5_payload_digest_ok(payload):
            problems.append("P5 verifier schema or payload digest drift")
        if payload.get("claim_scope") != P5_CLAIM_SCOPE or payload.get("evidence_class") != P5_EVIDENCE_CLASS:
            problems.append("P5 verifier claim scope or evidence class drift")
        if (
            payload.get("verification_complete") is not True
            or payload.get("all_ok") is not True
            or payload.get("prerequisite_ready") is not True
            or payload.get("problems") != []
            or payload.get("all_controls_passed") is not True
            or payload.get("all_mutations_rejected") is not True
        ):
            problems.append("P5 verifier is incomplete, not ready, or reports problems")
        if payload.get("source_bindings") != _p5_live_bindings(P5_VERIFIER_SOURCE_PATHS):
            problems.append("P5 verifier live source bindings drift")
        if payload.get("scientific_promotion") is not False or payload.get("promotion") != {
            "confirmatory_promotable": False,
            "refused_by_construction": True,
            "scientific_capability_claim": False,
        }:
            problems.append("P5 verifier promotion refusal drift")
        controls = payload.get("controls")
        if not isinstance(controls, dict) or any(
            controls.get(field) is not True
            for field in (
                "same_initialization_frozen_control",
                "matched_parameter_and_flop_contract",
                "difficulty_calibration_checked",
                "seed_arm_checkpoint_artifacts_exactly_joined",
                "raw_per_seed_contrasts_independently_recomputed",
                "fresh_disjoint_training_for_every_primary_pattern",
                "threshold_tie_is_null",
                "confirmatory_promotion_refused",
            )
        ):
            problems.append("P5 verifier control contract drift")
        independence = payload.get("independence")
        expected_independence = {
            "imports_p5_training_or_evaluator": False,
            "raw_seed_score_recompute": True,
            "checkpoint_files_opened_with_weights_only": True,
            "checkpoint_model_and_target_state_hashes_recomputed": True,
            "heldout_metrics_reexecuted_from_checkpoint": False,
            "fresh_training_required_for_each_primary_pattern": True,
            "fresh_training_seeds": list(P5_FRESH_SEEDS),
            "fresh_seeds_disjoint_from_primary": True,
        }
        if independence != expected_independence:
            problems.append("P5 verifier independence authority drift")
        expected_metric_limit = (
            "checkpoint model and target states, identities, completed steps, and compute are "
            "independently hashed and joined; heldout scores are recomputed from durable per-seed "
            "receipts but are not re-evaluated from model checkpoints"
        )
        if payload.get("metric_recomputation_limit") != expected_metric_limit:
            problems.append("P5 verifier metric recomputation limit drift")
        outcome_contract = payload.get("outcome_contract")
        if outcome_contract != {
            "allowed": ["mechanics", "null", "favorable-programmatic-only"],
            "tie_is_null": True,
            "programmatic_only": True,
            "confirmatory_promotable": False,
            "scientific_capability_claim": False,
        }:
            problems.append("P5 verifier outcome contract drift")
        classification = payload.get("classification")
        if (
            classification not in {"null", "favorable-programmatic-only"}
            or payload.get("outcome") != classification
        ):
            problems.append("P5 verifier classification is not a verified null or favorable pattern")

        profile = payload.get("primary_profile")
        if profile not in {"p5smoke", "p5pilot"}:
            raise ValueError("P5 verifier primary profile is invalid")
        expected_primary_relative = (
            "proof/P5_CONTEXT_CAPABILITY_SMOKE.json"
            if profile == "p5smoke"
            else "proof/P5_CONTEXT_CAPABILITY_PILOT.json"
        )
        expected_run_relative = (
            "runs/p5_context/p5smoke/p5_context_receipt.json"
            if profile == "p5smoke"
            else "runs/p5_context/p5pilot/p5_context_receipt.json"
        )
        primary_path = _p5_evidence_path(expected_primary_relative, evidence_root)
        run_path = _p5_evidence_path(expected_run_relative, evidence_root)
        primary = _p5_read_json(primary_path, "P5 verifier primary")
        raw = _p5_read_json(run_path, "P5 verifier raw primary")
        problems.extend(
            f"P5 verifier primary: {problem}"
            for problem in _p5_screen_authority_problems(primary, evidence_root)
        )
        primary_binding = payload.get("primary_receipt")
        raw_binding = payload.get("primary_run_receipt")
        if not isinstance(primary_binding, dict) or (
            primary_binding.get("path") != expected_primary_relative
            or primary_binding.get("sha256") != _sha256_file(primary_path)
            or primary_binding.get("payload_sha256") != primary.get("payload_sha256")
        ):
            problems.append("P5 verifier primary proof join drift")
        if not isinstance(raw_binding, dict) or (
            raw_binding.get("path") != expected_run_relative
            or raw_binding.get("sha256") != _sha256_file(run_path)
            or raw_binding.get("exactly_matches_published") is not True
            or raw != primary
        ):
            problems.append("P5 verifier raw primary join drift")
        config_binding = payload.get("config")
        live_config = _p5_resolved_config(str(profile))
        if not isinstance(config_binding, dict) or (
            config_binding.get("path") != "configs/experiment/mop_p5_context_capability.yaml"
            or config_binding.get("sha256") != _sha256_file(P5_CONFIG_PATH)
            or config_binding.get("resolved_sha256") != _canonical_sha256(live_config)
        ):
            problems.append("P5 verifier live config join drift")
        patterns = _p5_strict_patterns(primary)
        hint = bool(patterns)
        expected_primary_outcome = "favorable-programmatic-only" if hint else "null"
        if payload.get("primary_outcome") != expected_primary_outcome:
            problems.append("P5 verifier primary outcome drift")
        if (
            payload.get("fresh_challenge_required") is not hint
            or primary.get("fresh_challenge_required") is not hint
        ):
            problems.append("P5 verifier fresh challenge authorization drift")
        if payload.get("primary_patterns") != patterns:
            problems.append("P5 verifier primary pattern binding drift")
        expected_primary_off_ceiling = {
            f"f{frames}": _dotted_value(primary, f"frames.f{frames}.off_ceiling")
            for frames in P5_PRIMARY_FRAMES
        }
        if (
            not isinstance(controls, dict)
            or controls.get("primary_off_ceiling") != expected_primary_off_ceiling
        ):
            problems.append("P5 verifier primary off-ceiling control drift")
        terminal_null = primary.get("terminal_scientific_stop") is True
        if payload.get("terminal_null") is not terminal_null:
            problems.append("P5 verifier terminal-null binding drift")
        expected_nonterminal_support = terminal_null or any(
            value is True for value in expected_primary_off_ceiling.values()
        )
        if (
            not isinstance(controls, dict)
            or controls.get("nonterminal_outcome_has_off_ceiling_multiunit_support")
            is not expected_nonterminal_support
        ):
            problems.append("P5 verifier nonterminal off-ceiling support drift")

        artifact_evidence = payload.get("artifact_evidence")
        cell_receipt_evidence = payload.get("cell_receipt_evidence")
        primary_cell_evidence = (
            cell_receipt_evidence.get("primary") if isinstance(cell_receipt_evidence, dict) else None
        )
        if not isinstance(primary_cell_evidence, dict) or set(primary_cell_evidence) != {
            f"f{frames}" for frames in P5_FRAME_COUNTS
        }:
            problems.append("P5 verifier primary cell receipt evidence coverage drift")
        else:
            for frames in P5_FRAME_COUNTS:
                expected_relative = f"runs/p5_context/{profile}/frames/f{frames}/cell_receipt.json"
                expected_path = _p5_evidence_path(expected_relative, evidence_root)
                if primary_cell_evidence[f"f{frames}"] != {
                    "path": expected_relative,
                    "sha256": _sha256_file(expected_path),
                }:
                    problems.append(f"P5 verifier primary f{frames} cell receipt binding drift")
        primary_artifacts = artifact_evidence.get("primary") if isinstance(artifact_evidence, dict) else None
        if not isinstance(primary_artifacts, dict) or set(primary_artifacts) != {
            f"f{frames}" for frames in P5_FRAME_COUNTS
        }:
            problems.append("P5 verifier primary artifact evidence frame coverage drift")
        else:
            for frames in P5_FRAME_COUNTS:
                primary_cell = _p5_read_json(
                    run_path.parent / "frames" / f"f{frames}" / "cell_receipt.json",
                    f"P5 verifier primary f{frames} cell",
                )
                cell_seeds = primary_cell.get("expected_seeds")
                expected_primary_seeds = (
                    {str(int(value)) for value in cell_seeds} if isinstance(cell_seeds, list) else set()
                )
                problems.extend(
                    _p5_artifact_evidence_problems(
                        primary_artifacts[f"f{frames}"],
                        expected_seeds=expected_primary_seeds,
                        evidence_root=evidence_root,
                        label=f"P5 verifier primary f{frames}",
                    )
                )
        if profile == "p5smoke" and (
            primary.get("terminal_scientific_stop") is not True
            or primary.get("trainability_gate_failed") is not True
            or primary.get("execution_status") != "terminal-scientific-null"
            or hint
            or classification != "null"
        ):
            problems.append("P5 verifier smoke branch is not the canonical terminal null")

        mutation_rows = payload.get("mutation_tests")
        mutation_map = (
            {
                str(row.get("id")): row
                for row in mutation_rows
                if isinstance(row, dict) and isinstance(row.get("id"), str)
            }
            if isinstance(mutation_rows, list)
            else {}
        )
        required_mutations = set(P5_BASE_MUTATION_IDS)
        if hint:
            required_mutations.update(P5_CHALLENGE_MUTATION_IDS)
        if (
            not isinstance(mutation_rows, list)
            or not mutation_rows
            or len(mutation_rows) != len(mutation_map)
            or len(mutation_rows) != len(required_mutations)
            or set(mutation_map) != required_mutations
            or any(row.get("rejected") is not True for row in mutation_map.values())
        ):
            problems.append("P5 verifier required mutation rejection set drift")

        verified_patterns = payload.get("verified_patterns")
        challenge_binding = payload.get("fresh_challenge")
        if hint:
            challenge_path = _p5_evidence_path(
                "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json", evidence_root
            )
            challenge = _p5_read_json(challenge_path, "P5 verifier fresh challenge")
            problems.extend(
                f"P5 verifier challenge: {problem}"
                for problem in _p5_challenge_authority_problems(challenge, evidence_root)
            )
            if not isinstance(challenge_binding, dict) or (
                challenge_binding.get("path") != "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
                or challenge_binding.get("sha256") != _sha256_file(challenge_path)
                or challenge_binding.get("payload_sha256") != challenge.get("payload_sha256")
            ):
                problems.append("P5 verifier fresh challenge join drift")
            expected_fresh_cells = {
                str(row["seed"]): row["cell_receipts"]
                for row in challenge.get("training_runs", [])
                if isinstance(row, dict)
            }
            fresh_cell_evidence = (
                cell_receipt_evidence.get("fresh_challenge")
                if isinstance(cell_receipt_evidence, dict)
                else None
            )
            if fresh_cell_evidence != expected_fresh_cells:
                problems.append("P5 verifier fresh cell receipt evidence drift")
            per_pattern = (
                challenge_binding.get("per_pattern") if isinstance(challenge_binding, dict) else None
            )
            canonical_patterns = _p5_canonical_challenge_patterns(
                primary,
                challenge,
                evidence_root,
            )
            if per_pattern != canonical_patterns:
                problems.append("P5 verifier fresh pattern canonical rebuild drift")
            fresh_artifacts = (
                artifact_evidence.get("fresh_challenge") if isinstance(artifact_evidence, dict) else None
            )
            if not isinstance(fresh_artifacts, dict) or set(fresh_artifacts) != {
                str(seed) for seed in P5_FRESH_SEEDS
            }:
                problems.append("P5 verifier fresh artifact evidence seed coverage drift")
            else:
                for seed in P5_FRESH_SEEDS:
                    by_frame = fresh_artifacts[str(seed)]
                    if not isinstance(by_frame, dict) or set(by_frame) != {
                        f"f{frames}" for frames in P5_FRAME_COUNTS
                    }:
                        problems.append(f"P5 verifier fresh seed {seed} artifact frame coverage drift")
                        continue
                    for frames in P5_FRAME_COUNTS:
                        problems.extend(
                            _p5_artifact_evidence_problems(
                                by_frame[f"f{frames}"],
                                expected_seeds={str(seed)},
                                evidence_root=evidence_root,
                                label=f"P5 verifier fresh seed {seed} f{frames}",
                            )
                        )
            expected_verified = [
                row for row in canonical_patterns if row["programmatic_pattern_verified"] is True
            ]
            expected_classification = "favorable-programmatic-only" if expected_verified else "null"
            if classification != expected_classification:
                problems.append("P5 verifier classification canonical rebuild drift")
            if verified_patterns != expected_verified:
                problems.append("P5 verifier verified pattern canonical rebuild drift")
        else:
            if classification != "null" or payload.get("outcome") != "null":
                problems.append("P5 no-pattern verdict must remain a canonical null")
            if (
                challenge_binding is not None
                or verified_patterns not in ([], None)
                or not isinstance(artifact_evidence, dict)
                or artifact_evidence.get("fresh_challenge") is not None
            ):
                problems.append("P5 no-pattern verdict retains a fresh challenge or verified pattern")
            if (
                not isinstance(cell_receipt_evidence, dict)
                or cell_receipt_evidence.get("fresh_challenge") is not None
            ):
                problems.append("P5 no-pattern verdict retains fresh cell receipt evidence")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"P5 verifier authority invalid: {exc}")
    return problems


def _p5_schema_authority_problems(schema: str, payload: dict[str, Any], evidence_root: Path) -> list[str]:
    if schema == P5_SCREEN_SCHEMA:
        return _p5_screen_authority_problems(payload, evidence_root)
    if schema == P5_GRID_SCHEMA:
        return _p5_grid_authority_problems(payload, evidence_root)
    if schema == P5_CHALLENGE_SCHEMA:
        return _p5_challenge_authority_problems(payload, evidence_root)
    if schema == P5_VERIFIER_SCHEMA:
        return _p5_verifier_authority_problems(payload, evidence_root)
    return []


def _p6_expected_plan(config: dict[str, Any], *, rung: int, mode: str) -> dict[str, Any]:
    replication = config["replication"]
    profile = config["profile"]
    all_seeds = [int(value) for value in replication["seeds"]]
    if mode == "resource-probe":
        seeds = all_seeds[:1]
        schedules = ["abrupt"]
        arms = ["replay"]
    elif mode == "replication":
        seeds = all_seeds
        schedules = [str(value) for value in replication["schedules"]]
        arms = [str(value) for value in replication["arms"]]
    else:
        raise ValueError(f"unsupported P6 rung mode {mode!r}")
    cells = [
        {"seed": seed, "schedule": schedule, "arm": arm}
        for seed in seeds
        for schedule in schedules
        for arm in arms
    ]
    return {
        "mode": mode,
        "rung": rung,
        "seeds": seeds,
        "schedules": schedules,
        "arms": arms,
        "cells": cells,
        "expected_cells": len(cells),
        "stream": {
            "chunk_events": max(
                int(profile["minimum_chunk_events"]), rung // int(profile["chunks_per_stream"])
            ),
            "n_domains": 4,
            "n_classes": 4,
            "gradual_width_events": max(1, rung // int(profile["gradual_width_divisor"])),
            "deletion_event": (
                rung * int(profile["deletion_numerator"]) // int(profile["deletion_denominator"])
            ),
        },
        "profile": {
            "checkpoint_every": int(profile["checkpoint_every_events"]),
            "replay_capacity": int(profile["replay_capacity"]),
            "future_window_events": int(profile["future_window_events"]),
            "threshold_window_events": int(profile["threshold_window_events"]),
            "future_accuracy_threshold": float(profile["future_accuracy_threshold"]),
            "matched_updates_per_event": int(profile["matched_updates_per_event"]),
        },
    }


def _p6_rung_authority_problems(payload: dict[str, Any], evidence_root: Path) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P6_RUNG_SCHEMA:
            problems.append("P6 rung schema drift")
        core = dict(payload)
        declared_payload_sha = core.pop("payload_sha256", None)
        if not _is_sha256(declared_payload_sha) or _canonical_sha256(core) != declared_payload_sha:
            problems.append("P6 rung payload digest drift")
        if payload.get("claim_scope") != P6_CLAIM_SCOPE:
            problems.append("P6 rung claim scope drift")
        if payload.get("all_mechanics_ok") is not True:
            problems.append("P6 rung mechanics did not complete")
        if payload.get("independent_metric_verifier_complete") is not False:
            problems.append("P6 source rung cannot self-assert independent verification")
        if payload.get("scientific_promotion") is not False:
            problems.append("P6 source rung cannot self-promote")

        config = yaml.safe_load(P6_RUN_CONFIG.read_text())
        if not isinstance(config, dict):
            raise ValueError("live P6 rung config is invalid")
        evidence = _p6_resource_evidence()
        live_preflight = json.loads(P6_PREFLIGHT.read_text())
        if not isinstance(live_preflight, dict):
            raise ValueError("live P6 preflight is invalid")
        live_source_authority = _p6_source_live_binding_authority(live_preflight)
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("P6 rung identity is missing")
        expected_live_identity = {
            "config_sha256": evidence["config_sha256"],
            "runner_sha256": _sha256_file(REPO_ROOT / "scripts/continual_million_event_rung.py"),
            "source_preflight_file_sha256": evidence["preflight_file_sha256"],
            "source_preflight_payload_sha256": evidence["preflight_payload_sha256"],
            "source_live_bindings_sha256": evidence["preflight_live_bindings_sha256"],
        }
        for field, expected in expected_live_identity.items():
            if identity.get(field) != expected:
                problems.append(f"P6 rung live identity drift: {field}")
        if identity.get("claim_scope") != P6_CLAIM_SCOPE:
            problems.append("P6 rung identity claim scope drift")
        if payload.get("identity_sha256") != _canonical_sha256(identity):
            problems.append("P6 rung identity digest drift")
        if payload.get("source_live_authority") != live_source_authority:
            problems.append("P6 rung embedded source authority drift")

        rung = payload.get("rung")
        mode = payload.get("mode")
        if not isinstance(rung, int) or isinstance(rung, bool) or rung not in (10_000, 100_000, 1_000_000):
            raise ValueError("P6 rung value is invalid")
        if not isinstance(mode, str):
            raise ValueError("P6 rung mode is invalid")
        expected_plan = _p6_expected_plan(config, rung=rung, mode=mode)
        plan = payload.get("plan")
        if plan != expected_plan or identity.get("plan") != expected_plan:
            problems.append("P6 rung exact plan or identity plan drift")
        expected_replication = mode == "replication"
        if payload.get("replication_execution_complete") is not expected_replication:
            problems.append("P6 rung replication completion drift")

        expected_keys = {
            f"seed_{cell['seed']}/{cell['schedule']}/{cell['arm']}" for cell in expected_plan["cells"]
        }
        cells = payload.get("cells")
        if not isinstance(cells, dict) or set(cells) != expected_keys:
            problems.append("P6 rung completed cell matrix drift")
            cells = cells if isinstance(cells, dict) else {}
        for key in sorted(expected_keys & set(cells)):
            row = cells[key]
            if not isinstance(row, dict):
                problems.append(f"P6 cell {key} is not an object")
                continue
            if f"seed_{row.get('seed')}/{row.get('schedule')}/{row.get('arm')}" != key:
                problems.append(f"P6 cell {key} coordinate drift")
            if not all(
                _is_sha256(row.get(field))
                for field in (
                    "stream_identity_sha256",
                    "stream_sha256",
                    "checkpoint_sha256",
                    "state_sha256",
                )
            ):
                problems.append(f"P6 cell {key} authority digest invalid")
            if row.get("all_mechanics_ok") is not True or not isinstance(
                row.get("resumed_from_atomic_checkpoint"), bool
            ):
                problems.append(f"P6 cell {key} mechanics or resume observation invalid")
            controls = row.get("controls")
            arm = row.get("arm")
            if not isinstance(controls, dict) or controls != {
                "replay_enabled": arm == "replay",
                "fresh_init_on_transition": arm == "fresh-init",
                "matched_updates_per_event": 2,
                "actual_updates_per_event": 2.0,
                "fixed_topology": True,
                "reset_count": 3 if arm == "fresh-init" else 0,
            }:
                problems.append(f"P6 cell {key} control contract drift")
            metrics = row.get("metrics")
            resources = metrics.get("resources") if isinstance(metrics, dict) else None
            if not isinstance(metrics, dict) or set(metrics) != {
                "retention",
                "acquisition",
                "future_learnability",
                "stale_memory",
                "deletion",
                "resources",
            }:
                problems.append(f"P6 cell {key} metric family drift")
            elif not isinstance(resources, dict) or (
                resources.get("events_processed") != rung
                or resources.get("updates") != rung * 2
                or resources.get("updates_per_event") != 2.0
                or resources.get("model_weights_loaded") is not False
                or resources.get("accelerator_required") is not False
                or not isinstance(resources.get("checkpoint_state_bytes"), int)
                or int(resources["checkpoint_state_bytes"]) <= 0
                or not isinstance(resources.get("stream_disk_bytes"), int)
                or int(resources["stream_disk_bytes"]) <= 0
            ):
                problems.append(f"P6 cell {key} resource metric drift")

        progress_binding = payload.get("progress")
        if not isinstance(progress_binding, dict):
            raise ValueError("P6 rung progress binding is missing")
        progress_path = _safe_evidence_path(progress_binding.get("path"), evidence_root)
        progress = json.loads(progress_path.read_text())
        if not isinstance(progress, dict):
            raise ValueError("P6 rung progress authority is invalid")
        if (
            _sha256_file(progress_path) != progress_binding.get("sha256")
            or progress.get("schema") != "mop-continual-progressive-rung-progress/v1"
            or progress.get("identity") != identity
            or progress.get("identity_sha256") != _canonical_sha256(identity)
            or progress.get("cells") != cells
            or progress.get("complete") is not True
            or progress_binding.get("completed_cells") != len(expected_keys)
            or progress_binding.get("expected_cells") != len(expected_keys)
        ):
            problems.append("P6 rung live progress authority drift")

        work_root = progress_path.parent
        for key, row in sorted(cells.items()):
            if not isinstance(row, dict):
                continue
            checkpoint_path = (
                work_root
                / "checkpoints"
                / f"seed_{row.get('seed')}"
                / str(row.get("schedule"))
                / f"{row.get('arm')}.json"
            )
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
                if not isinstance(checkpoint, dict):
                    raise ValueError("checkpoint is not an object")
                checkpoint_identity = checkpoint.get("identity")
                state = checkpoint.get("state")
                result = checkpoint.get("result")
                if (
                    _sha256_file(checkpoint_path) != row.get("checkpoint_sha256")
                    or checkpoint.get("schema") != "mop-continual-smoke-checkpoint/v1"
                    or checkpoint.get("complete") is not True
                    or not isinstance(checkpoint_identity, dict)
                    or checkpoint.get("identity_sha256") != _canonical_sha256(checkpoint_identity)
                    or not isinstance(state, dict)
                    or checkpoint.get("state_sha256") != _canonical_sha256(state)
                    or checkpoint.get("state_sha256") != row.get("state_sha256")
                    or not isinstance(result, dict)
                    or result.get("all_mechanics_ok") is not True
                    or result.get("metrics") != row.get("metrics")
                    or result.get("controls") != row.get("controls")
                ):
                    problems.append(f"P6 cell {key} live checkpoint authority drift")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                problems.append(f"P6 cell {key} live checkpoint authority invalid: {exc}")

        semantic_audit = audit_rung_semantics(payload, repo_root=evidence_root)
        if semantic_audit.get("all_ok") is not True:
            errors = semantic_audit.get("errors")
            problems.append(
                "P6 rung raw-stream semantic audit failed: "
                + "; ".join(str(value) for value in errors if isinstance(errors, list))
                if isinstance(errors, list)
                else "P6 rung raw-stream semantic audit failed without an error list"
            )

        measurement = payload.get("resource_measurement")
        rss_bytes = measurement.get("max_rss_bytes") if isinstance(measurement, dict) else None
        if (
            not isinstance(measurement, dict)
            or not isinstance(rss_bytes, int)
            or isinstance(rss_bytes, bool)
            or rss_bytes < P6_MINIMUM_SANE_RSS_BYTES
            or measurement.get("measured_after_complete") is not True
            or measurement.get("events_per_stream") != rung
        ):
            problems.append(
                f"P6 rung resource measurement must be complete and at least "
                f"{P6_MINIMUM_SANE_RSS_BYTES} bytes RSS"
            )
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError) as exc:
        problems.append(f"P6 rung authority invalid: {exc}")
    return problems


def _p6_verifier_required_fields(source_rung: int, next_rung: int) -> dict[str, Any]:
    return {
        "verification_complete": True,
        "errors": [],
        "source_rung.mode": "replication",
        "source_rung.rung": source_rung,
        "checks.source_payload_self_hash": True,
        "checks.live_dependencies_current": True,
        "checks.progress_and_checkpoints_current": True,
        "checks.full_replication_structure_valid": True,
        "checks.all_metrics_independently_recomputed": True,
        "checks.all_controls_present_and_valid": True,
        "checks.tie_is_null": True,
        "checks.all_mutations_rejected": True,
        "checks.scientific_promotion_blocked": True,
        "independent_recompute.cell_count": 30,
        "independent_recompute.checkpoint_state_recomputed": True,
        "independent_recompute.controls_recomputed": True,
        "independent_recompute.paired_metrics_recomputed": True,
        "independent_recompute.decision.tie_rule": P6_TIE_RULE,
        "independent_recompute.decision.aggregate_tie_count": 0,
        "independent_recompute.decision.strict_joint_gain_all_schedules_and_controls": True,
        "independent_recompute.decision.verdict": "favorable-rung-pattern",
        "independent_recompute.decision.null_supported": False,
        "independent_recompute.decision.scientific_promotion": False,
        "mutation_suite.count": 12,
        "mutation_suite.rejected": 12,
        "mutation_suite.all_rejected": True,
        "prerequisite.source_rung": source_rung,
        "prerequisite.verification_complete": True,
        "prerequisite.valid_controls": True,
        "prerequisite.tie_is_null": True,
        "prerequisite.mutation_suite_all_rejected": True,
        "prerequisite.next_rung": next_rung,
        "prerequisite.next_rung_allowed": True,
        "prerequisite.next_rung_reason": (
            "strict favorable programmatic pattern requires the next scale confirmation"
        ),
        "scientific_promotion": False,
    }


def _p6_verifier_authority_problems(payload: dict[str, Any], evidence_root: Path) -> list[str]:

    problems: list[str] = []
    try:
        if payload.get("schema") != P6_VERIFIER_SCHEMA:
            problems.append("P6 verifier schema drift")
        core = dict(payload)
        declared_payload_sha = core.pop("payload_sha256", None)
        if not _is_sha256(declared_payload_sha) or _canonical_sha256(core) != declared_payload_sha:
            problems.append("P6 verifier payload digest drift")
        if payload.get("claim_scope") != P6_CLAIM_SCOPE:
            problems.append("P6 verifier claim scope drift")
        if payload.get("verification_complete") is not True or payload.get("errors") != []:
            problems.append("P6 verifier is incomplete or reports errors")
        if payload.get("scientific_promotion") is not False:
            problems.append("P6 verifier cannot self-promote")

        implementation = payload.get("implementation")
        expected_implementation = [
            {"path": path, "sha256": _sha256_file(REPO_ROOT / path)}
            for path in P6_VERIFIER_IMPLEMENTATION_PATHS
        ]
        if implementation != expected_implementation:
            problems.append("P6 verifier live implementation binding drift")

        source_info = payload.get("source_rung")
        if not isinstance(source_info, dict):
            raise ValueError("P6 verifier source rung binding is missing")
        source_rung = source_info.get("rung")
        expected_paths = {
            10_000: "proof/P6_CONTINUAL_10K.json",
            100_000: "proof/P6_CONTINUAL_100K.json",
            1_000_000: "proof/P6_CONTINUAL_1M.json",
        }
        if source_rung not in expected_paths or source_info.get("path") != expected_paths[source_rung]:
            raise ValueError("P6 verifier source rung path or value drift")
        source_path = _safe_evidence_path(source_info["path"], evidence_root)
        source = json.loads(source_path.read_text())
        if not isinstance(source, dict):
            raise ValueError("P6 verifier source rung is invalid")
        source_problems = _p6_rung_authority_problems(source, evidence_root)
        problems.extend(f"P6 verifier source: {problem}" for problem in source_problems)
        if (
            source_info.get("mode") != "replication"
            or source_info.get("file_sha256") != _sha256_file(source_path)
            or source_info.get("payload_sha256") != source.get("payload_sha256")
            or source_info.get("identity_sha256") != source.get("identity_sha256")
        ):
            problems.append("P6 verifier source file, payload, or identity join drift")

        identity = source.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("P6 verifier source identity is missing")
        expected_live_dependencies = {
            field: identity.get(field)
            for field in (
                "config_sha256",
                "runner_sha256",
                "source_preflight_file_sha256",
                "source_preflight_payload_sha256",
                "source_live_bindings_sha256",
            )
        }
        if payload.get("live_dependencies") != expected_live_dependencies:
            problems.append("P6 verifier live dependency receipt drift")

        progress = source.get("progress")
        expected_progress = {
            "path": progress.get("path") if isinstance(progress, dict) else None,
            "file_sha256": progress.get("sha256") if isinstance(progress, dict) else None,
            "identity_sha256": source.get("identity_sha256"),
            "complete": True,
            "cell_count": 30,
        }
        if payload.get("progress_authority") != expected_progress:
            problems.append("P6 verifier progress authority join drift")

        checks = payload.get("checks")
        expected_check_names = {
            "source_payload_self_hash",
            "live_dependencies_current",
            "progress_and_checkpoints_current",
            "full_replication_structure_valid",
            "all_metrics_independently_recomputed",
            "all_controls_present_and_valid",
            "tie_is_null",
            "all_mutations_rejected",
            "scientific_promotion_blocked",
        }
        if (
            not isinstance(checks, dict)
            or set(checks) != expected_check_names
            or not all(value is True for value in checks.values())
        ):
            problems.append("P6 verifier canonical check set is incomplete or false")

        recompute = payload.get("independent_recompute")
        if not isinstance(recompute, dict) or (
            recompute.get("cell_count") != 30
            or recompute.get("metric_families")
            != [
                "retention",
                "acquisition",
                "future_learnability",
                "stale_memory",
                "deletion",
                "resources",
            ]
            or recompute.get("checkpoint_state_recomputed") is not True
            or recompute.get("controls_recomputed") is not True
            or recompute.get("paired_metrics_recomputed") is not True
        ):
            problems.append("P6 verifier independent recompute envelope drift")
        decision = recompute.get("decision") if isinstance(recompute, dict) else None
        if not isinstance(decision, dict) or (
            decision.get("primary_endpoints")
            != [
                "retention.domain_zero_final_accuracy",
                "future_learnability.first_window_accuracy",
            ]
            or decision.get("independent_unit") != "seed within transition schedule"
            or decision.get("controls") != ["no-replay", "fresh-init"]
            or decision.get("tie_rule") != P6_TIE_RULE
            or decision.get("aggregate_tie_count") != 0
            or decision.get("strict_joint_gain_all_schedules_and_controls") is not True
            or decision.get("verdict") != "favorable-rung-pattern"
            or decision.get("null_supported") is not False
            or decision.get("scientific_promotion") is not False
        ):
            problems.append("P6 verifier favorable non-tie decision drift")
        contrasts = decision.get("contrasts") if isinstance(decision, dict) else None
        expected_coordinates = {
            (schedule, control)
            for schedule in ("abrupt", "gradual")
            for control in ("no-replay", "fresh-init")
        }
        observed_coordinates: set[tuple[object, object]] = set()
        source_plan = source.get("plan")
        seeds = source_plan.get("seeds") if isinstance(source_plan, dict) else None
        if not isinstance(contrasts, list) or len(contrasts) != 4 or not isinstance(seeds, list):
            problems.append("P6 verifier paired contrast matrix drift")
        else:
            for contrast in contrasts:
                if not isinstance(contrast, dict):
                    problems.append("P6 verifier paired contrast row invalid")
                    continue
                observed_coordinates.add((contrast.get("schedule"), contrast.get("control")))
                retention = contrast.get("retention_mean_delta")
                future = contrast.get("future_first_window_mean_delta")
                pairs = contrast.get("paired_seed_deltas")
                if (
                    contrast.get("independent_units") != len(seeds)
                    or not isinstance(retention, (int, float))
                    or isinstance(retention, bool)
                    or not math.isfinite(retention)
                    or retention <= 0
                    or not isinstance(future, (int, float))
                    or isinstance(future, bool)
                    or not math.isfinite(future)
                    or future <= 0
                    or contrast.get("aggregate_tie_is_null") is not False
                    or contrast.get("any_seed_nonpositive_is_null") is not False
                    or contrast.get("null_contrast") is not False
                    or contrast.get("strict_joint_gain") is not True
                    or not isinstance(pairs, list)
                    or len(pairs) != len(seeds)
                ):
                    problems.append("P6 verifier favorable contrast row drift")
                    continue
                for expected_seed, pair in zip(seeds, pairs, strict=True):
                    if not isinstance(pair, dict):
                        problems.append("P6 verifier paired seed row invalid")
                        continue
                    retention_delta = pair.get("retention_delta")
                    future_delta = pair.get("future_first_window_delta")
                    if (
                        pair.get("seed") != expected_seed
                        or pair.get("tie_is_null") is not False
                        or pair.get("nonpositive_is_null") is not False
                        or not isinstance(retention_delta, (int, float))
                        or isinstance(retention_delta, bool)
                        or not math.isfinite(retention_delta)
                        or retention_delta <= 0
                        or not isinstance(future_delta, (int, float))
                        or isinstance(future_delta, bool)
                        or not math.isfinite(future_delta)
                        or future_delta <= 0
                    ):
                        problems.append("P6 verifier paired seed favorable non-tie drift")
            if observed_coordinates != expected_coordinates:
                problems.append("P6 verifier paired contrast coordinates drift")

        mutation_suite = payload.get("mutation_suite")
        mutation_rows = mutation_suite.get("mutations") if isinstance(mutation_suite, dict) else None
        if (
            not isinstance(mutation_suite, dict)
            or mutation_suite.get("count") != 12
            or mutation_suite.get("rejected") != 12
            or mutation_suite.get("all_rejected") is not True
            or not isinstance(mutation_rows, list)
            or len(mutation_rows) != 12
            or len(
                {
                    row.get("mutation")
                    for row in mutation_rows
                    if isinstance(row, dict) and isinstance(row.get("mutation"), str)
                }
            )
            != 12
            or any(
                not isinstance(row, dict)
                or row.get("rejected") is not True
                or not isinstance(row.get("problems"), list)
                or not row["problems"]
                for row in mutation_rows or []
            )
        ):
            problems.append("P6 verifier mutation suite authority drift")

        next_rung = {10_000: 100_000, 100_000: 1_000_000}.get(source_rung)
        prerequisite = payload.get("prerequisite")
        next_allowed = next_rung is not None
        expected_prerequisite = {
            "source_rung": source_rung,
            "source_rung_file_sha256": _sha256_file(source_path),
            "source_identity_sha256": source.get("identity_sha256"),
            "verification_complete": True,
            "valid_controls": True,
            "tie_is_null": True,
            "mutation_suite_all_rejected": True,
            "next_rung": next_rung,
            "next_rung_allowed": next_allowed,
            "next_rung_reason": (
                "strict favorable programmatic pattern requires the next scale confirmation"
                if next_allowed
                else "verified tie, null, invalid evidence, or final rung does not admit scaling"
            ),
        }
        if prerequisite != expected_prerequisite:
            problems.append("P6 verifier next-rung prerequisite join drift")

        canonical = build_p6_verification_receipt(source_path, repo_root=evidence_root)
        if payload != canonical:
            problems.append("P6 verifier semantic canonical rebuild drift")
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        problems.append(f"P6 verifier authority invalid: {exc}")
    return problems


def load_policy(path: Path | str = DEFAULT_POLICY) -> ThrottlePolicy:
    _assert_task_policy_helper_pin()
    _assert_external_coexistence_helper_pin()
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
    external_thresholds = thresholds.get("external_coexistence")
    if external_thresholds is not None:
        expected_external_fields = {
            "minimum_memory_available_percent",
            "minimum_memory_available_gb",
            "minimum_memory_pressure_free_percent",
            "maximum_swap_used_gb",
        }
        if not isinstance(external_thresholds, dict) or set(external_thresholds) != (
            expected_external_fields
        ):
            problems.append(
                "thresholds.external_coexistence must contain the exact reviewed memory and swap fields"
            )
        elif any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in external_thresholds.values()
        ):
            problems.append("thresholds.external_coexistence values must be finite nonnegative numbers")
    if "p5_context_fresh_challenge.py" not in monitor.get("known_heavy_markers", []):
        problems.append("monitor.known_heavy_markers must include the P5 fresh challenge")
    for field in ("foreground_markers", "known_heavy_markers"):
        markers = monitor.get(field)
        if (
            not isinstance(markers, list)
            or any(not isinstance(value, str) or not value.strip() for value in markers)
            or len(markers) != len(set(markers))
        ):
            problems.append(f"monitor.{field} must be a unique nonempty-string list")
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
        coexistence_value = value.get("external_coexistence")
        invocation_value = value.get("max_invocations_per_run")
        if coexistence_value is not None:
            if coexistence_value != EXTERNAL_COEXISTENCE_PROFILE:
                problems.append(f"task {task.task_id}: external coexistence profile is unsupported")
            problems.extend(
                f"task {task.task_id}: {problem}" for problem in _external_coexistence_task_problems(task)
            )
            if task.task_id in SEED_BOUNDARY_TASKS and invocation_value != 1:
                problems.append(f"task {task.task_id}: every run must yield after exactly one invocation")
            if task.task_id not in SEED_BOUNDARY_TASKS and invocation_value is not None:
                problems.append(f"task {task.task_id}: full verifier must not yield between invocations")
        elif invocation_value is not None:
            problems.append(f"task {task.task_id}: invocation yield is not reviewed")
    order_raw = raw.get("execution_order") or {}
    if not isinstance(order_raw, dict):
        problems.append("execution_order must be a mapping")
        order_raw = {}
    execution_order = {
        str(key): tuple(str(value) for value in values)
        for key, values in order_raw.items()
        if isinstance(values, list)
    }
    expected_p5_order = (
        "p5smoke_cpu",
        "p5_traingrid_memory_probe_cpu",
        "p5pilot_cpu",
        "p5fresh_challenge_cpu",
        "p5verify_cpu",
    )
    if execution_order.get("p5_cpu") != expected_p5_order:
        problems.append(f"execution_order.p5_cpu must be {list(expected_p5_order)}")
    expected_p5_pilot_null_order = (
        "p5smoke_cpu",
        "p5_traingrid_memory_probe_cpu",
        "p5pilot_cpu",
        "p5verify_pilot_null_cpu",
    )
    if execution_order.get("p5_pilot_null_cpu") != expected_p5_pilot_null_order:
        problems.append(f"execution_order.p5_pilot_null_cpu must be {list(expected_p5_pilot_null_order)}")
    expected_p5_terminal_order = (
        "p5smoke_cpu",
        "p5verify_smoke_null_cpu",
    )
    if execution_order.get("p5_terminal_null_cpu") != expected_p5_terminal_order:
        problems.append(f"execution_order.p5_terminal_null_cpu must be {list(expected_p5_terminal_order)}")
    expected_p5_dependencies = {
        expected_p5_order[0]: (),
        expected_p5_order[1]: (expected_p5_order[0],),
        expected_p5_order[2]: (expected_p5_order[1],),
        expected_p5_order[3]: (expected_p5_order[2],),
        expected_p5_order[4]: (expected_p5_order[3],),
        expected_p5_pilot_null_order[3]: (expected_p5_pilot_null_order[2],),
        expected_p5_terminal_order[1]: (expected_p5_terminal_order[0],),
    }
    expected_p6_order = (
        "p6_10k_resource_probe_cpu",
        "p6_10k_replication_cpu",
        "p6_10k_verify_cpu",
        "p6_100k_replication_cpu",
        "p6_100k_verify_cpu",
        "p6_1m_replication_cpu",
        "p6_1m_verify_cpu",
    )
    if execution_order.get("p6_cpu") != expected_p6_order:
        problems.append(f"execution_order.p6_cpu must be {list(expected_p6_order)}")
    expected_p6_dependencies = {
        expected_p6_order[0]: (),
        expected_p6_order[1]: (expected_p6_order[0],),
        expected_p6_order[2]: (expected_p6_order[1],),
        expected_p6_order[3]: (expected_p6_order[2],),
        expected_p6_order[4]: (expected_p6_order[3],),
        expected_p6_order[5]: (expected_p6_order[4],),
        expected_p6_order[6]: (expected_p6_order[5],),
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
        if _requires_completion_provenance(task):
            try:
                output_path = _task_output_path(task)
            except ThrottleRefused as exc:
                problems.append(str(exc))
                output_path = None
            if output_path is None or output_path not in task.checkpoint_globs:
                problems.append(
                    f"task {task_id}: every provenance-governed output must be in checkpoint_globs"
                )
    for task_id, expected_dependencies in expected_p6_dependencies.items():
        p6_task = tasks.get(task_id)
        if p6_task is not None and p6_task.depends_on != expected_dependencies:
            problems.append(f"task {task_id}: P6 dependency chain drifted")
    for task_id, expected_dependencies in expected_p5_dependencies.items():
        p5_task = tasks.get(task_id)
        if p5_task is not None and p5_task.depends_on != expected_dependencies:
            problems.append(f"task {task_id}: P5 dependency chain drifted")
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
        "p5fresh_challenge_cpu": (
            ".venv/bin/python",
            "scripts/p5_context_fresh_challenge.py",
            "--primary",
            "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
            "--primary-run-dir",
            "runs/p5_context/p5pilot",
            "--run-dir",
            "runs/p5_context/fresh_challenge",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
            "--device",
            "cpu",
        ),
        "p5verify_cpu": (
            ".venv/bin/python",
            "scripts/verify_p5_context_capability.py",
            "--primary",
            "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
            "--primary-run-dir",
            "runs/p5_context/p5pilot",
            "--fresh-challenge",
            "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
        ),
        "p5verify_pilot_null_cpu": (
            ".venv/bin/python",
            "scripts/verify_p5_context_capability.py",
            "--primary",
            "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
            "--primary-run-dir",
            "runs/p5_context/p5pilot",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
        ),
        "p5verify_smoke_null_cpu": (
            ".venv/bin/python",
            "scripts/verify_p5_context_capability.py",
            "--primary",
            "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
            "--primary-run-dir",
            "runs/p5_context/p5smoke",
            "--out",
            "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",
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
        "p6_10k_verify_cpu": (
            ".venv/bin/python",
            "scripts/verify_continual_million_event_rung.py",
            "--source",
            "proof/P6_CONTINUAL_10K.json",
            "--out",
            "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
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
        "p6_100k_verify_cpu": (
            ".venv/bin/python",
            "scripts/verify_continual_million_event_rung.py",
            "--source",
            "proof/P6_CONTINUAL_100K.json",
            "--out",
            "proof/P6_CONTINUAL_100K_INDEPENDENT_VERIFICATION.json",
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
        "p6_1m_verify_cpu": (
            ".venv/bin/python",
            "scripts/verify_continual_million_event_rung.py",
            "--source",
            "proof/P6_CONTINUAL_1M.json",
            "--out",
            "proof/P6_CONTINUAL_1M_INDEPENDENT_VERIFICATION.json",
        ),
    }
    for task_id, expected in expected_commands.items():
        canonical_task = tasks.get(task_id)
        if canonical_task is None:
            problems.append(f"required canonical task {task_id} is missing")
        elif canonical_task.command != expected:
            problems.append(f"task {task_id}: canonical command drifted")
    for task_id in (
        "p4_resume_cpu",
        "p4_resume_mps",
        "p5smoke_cpu",
        "p5pilot_cpu",
        "p5fresh_challenge_cpu",
    ):
        canonical_task = tasks.get(task_id)
        if canonical_task is not None and canonical_task.restart_exit_codes != (2,):
            problems.append(f"task {task_id}: exit code 2 must remain the resume boundary")
    p5_smoke_fields = {
        "profile": "p5smoke",
        "complete": True,
        "all_ok": True,
        "resumable": False,
        "execution_status": "complete",
        "terminal_scientific_stop": False,
        "terminal_stop_reason": None,
        "trainability_gate_failed": False,
        "fresh_challenge_required": False,
        "promotion.confirmatory_promotable": False,
    }
    p5_terminal_smoke_fields = {
        "profile": "p5smoke",
        "complete": True,
        "all_ok": True,
        "resumable": False,
        "execution_status": "terminal-scientific-null",
        "terminal_scientific_stop": True,
        "terminal_stop_reason": "f64-trainability-gate-null",
        "trainability_gate_failed": True,
        "fresh_challenge_required": False,
        "promotion.confirmatory_promotable": False,
    }
    p5_grid_fields = {
        "all_ok": True,
        "claim_boundary.mechanics_only": True,
        "config.repeats": 3,
        "config.seed": 0,
        "atomic_progress.completed_rows": 72,
    }
    p5_pilot_fields = {
        "profile": "p5pilot",
        "complete": True,
        "all_ok": True,
        "resumable": False,
        "promotion.confirmatory_promotable": False,
    }
    p5_challenge_pilot_fields = {
        **p5_pilot_fields,
        "fresh_challenge_required": True,
    }
    p5_null_pilot_fields = {
        **p5_pilot_fields,
        "fresh_challenge_required": False,
    }
    p5_challenge_fields = {
        "complete": True,
        "all_ok": True,
        "verification_ready": True,
        "resumable": False,
        "problems": [],
        "fresh_seeds_disjoint_from_primary": True,
        "promotion.confirmatory_promotable": False,
        "scientific_promotion": False,
    }
    expected_p5_requirements = {
        "p5_traingrid_memory_probe_cpu": (
            ("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", "mop-p5-context-screen/v1", p5_smoke_fields),
        ),
        "p5pilot_cpu": (
            ("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", "mop-p5-context-screen/v1", p5_smoke_fields),
            (
                "proof/P5_TRAINGRID_MEMORY_TRACE.json",
                "mop-p5-traingrid-memory-trace/v1",
                p5_grid_fields,
            ),
        ),
        "p5fresh_challenge_cpu": (
            ("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", P5_SCREEN_SCHEMA, p5_smoke_fields),
            ("proof/P5_TRAINGRID_MEMORY_TRACE.json", P5_GRID_SCHEMA, p5_grid_fields),
            (
                "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
                P5_SCREEN_SCHEMA,
                p5_challenge_pilot_fields,
            ),
        ),
        "p5verify_cpu": (
            ("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", P5_SCREEN_SCHEMA, p5_smoke_fields),
            ("proof/P5_TRAINGRID_MEMORY_TRACE.json", P5_GRID_SCHEMA, p5_grid_fields),
            (
                "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
                P5_SCREEN_SCHEMA,
                p5_challenge_pilot_fields,
            ),
            (
                "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
                P5_CHALLENGE_SCHEMA,
                p5_challenge_fields,
            ),
        ),
        "p5verify_pilot_null_cpu": (
            ("proof/P5_CONTEXT_CAPABILITY_SMOKE.json", P5_SCREEN_SCHEMA, p5_smoke_fields),
            ("proof/P5_TRAINGRID_MEMORY_TRACE.json", P5_GRID_SCHEMA, p5_grid_fields),
            (
                "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
                P5_SCREEN_SCHEMA,
                p5_null_pilot_fields,
            ),
        ),
        "p5verify_smoke_null_cpu": (
            (
                "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
                P5_SCREEN_SCHEMA,
                p5_terminal_smoke_fields,
            ),
        ),
    }
    for task_id, expected_requirements in expected_p5_requirements.items():
        p5_task = tasks.get(task_id)
        if p5_task is None:
            continue
        observed_requirements = tuple(
            (requirement.path, requirement.schema, dict(requirement.fields))
            for requirement in p5_task.prerequisites
        )
        if observed_requirements != expected_requirements:
            problems.append(f"task {task_id}: exact P5 completion prerequisites drifted")
    expected_p5_checkpoint_globs = {
        "p5smoke_cpu": (
            "runs/p5_context/p5smoke/frames/f*/seed_*/*/checkpoint.pt",
            "runs/p5_context/p5smoke/frames/f*/seed_*/*/arm_receipt.json",
            "runs/p5_context/p5smoke/frames/f*/seed_*/seed_result.json",
            "runs/p5_context/p5smoke/frames/f*/cell_receipt.json",
            "runs/p5_context/p5smoke/p5_context_receipt.json",
            "runs/p5_context/p5smoke/resolved_config.json",
            "proof/P5_CONTEXT_CAPABILITY_SMOKE.json",
        ),
        "p5pilot_cpu": (
            "runs/p5_context/p5pilot/frames/f*/seed_*/*/checkpoint.pt",
            "runs/p5_context/p5pilot/frames/f*/seed_*/*/arm_receipt.json",
            "runs/p5_context/p5pilot/frames/f*/seed_*/seed_result.json",
            "runs/p5_context/p5pilot/frames/f*/cell_receipt.json",
            "runs/p5_context/p5pilot/p5_context_receipt.json",
            "runs/p5_context/p5pilot/resolved_config.json",
            "proof/P5_CONTEXT_CAPABILITY_PILOT.json",
        ),
        "p5fresh_challenge_cpu": (
            "runs/p5_context/fresh_challenge/seed_*/frames/f*/seed_*/*/checkpoint.pt",
            "runs/p5_context/fresh_challenge/seed_*/frames/f*/seed_*/*/arm_receipt.json",
            "runs/p5_context/fresh_challenge/seed_*/frames/f*/seed_*/seed_result.json",
            "runs/p5_context/fresh_challenge/seed_*/frames/f*/cell_receipt.json",
            "runs/p5_context/fresh_challenge/seed_*/p5_context_receipt.json",
            "runs/p5_context/fresh_challenge/seed_*/resolved_config.json",
            "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
        ),
    }
    for task_id, expected_globs in expected_p5_checkpoint_globs.items():
        p5_task = tasks.get(task_id)
        if p5_task is not None and p5_task.checkpoint_globs != expected_globs:
            problems.append(f"task {task_id}: P5 checkpoint globs drifted")
    p5_challenge = tasks.get("p5fresh_challenge_cpu")
    if p5_challenge is not None and (
        p5_challenge.lane != "heavy"
        or p5_challenge.cpu_cores != 10
        or p5_challenge.estimated_unified_memory_gb != 10.0
        or p5_challenge.forecast_write_gb != 8.0
        or not p5_challenge.pause_safe
    ):
        problems.append("task p5fresh_challenge_cpu: resource and resume contract drifted")
    for task_id in (
        "p5verify_cpu",
        "p5verify_pilot_null_cpu",
        "p5verify_smoke_null_cpu",
    ):
        p5_verifier = tasks.get(task_id)
        if p5_verifier is not None and (
            p5_verifier.lane != "light"
            or p5_verifier.accelerator != "none"
            or p5_verifier.cpu_cores != 2
            or p5_verifier.estimated_unified_memory_gb != 2.0
            or p5_verifier.pause_safe
            or not p5_verifier.atomic_checkpoints
            or p5_verifier.checkpoint_globs != ("proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json",)
            or p5_verifier.restart_exit_codes
        ):
            problems.append(f"task {task_id}: P5 verifier resource contract drifted")
    p6_probe = tasks.get("p6_10k_resource_probe_cpu")
    p5_verification_fields = {
        "verification_complete": True,
        "all_ok": True,
        "prerequisite_ready": True,
        "problems": [],
        "all_controls_passed": True,
        "all_mutations_rejected": True,
        "controls.seed_arm_checkpoint_artifacts_exactly_joined": True,
        "controls.nonterminal_outcome_has_off_ceiling_multiunit_support": True,
        "controls.threshold_tie_is_null": True,
        "independence.checkpoint_files_opened_with_weights_only": True,
        "independence.checkpoint_model_and_target_state_hashes_recomputed": True,
        "independence.heldout_metrics_reexecuted_from_checkpoint": False,
        "outcome_contract.tie_is_null": True,
        "promotion.confirmatory_promotable": False,
        "scientific_promotion": False,
    }
    if p6_probe is not None and not any(
        requirement.path == "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
        and requirement.schema == "mop-p5-context-independent-verifier/v1"
        and dict(requirement.fields) == p5_verification_fields
        for requirement in p6_probe.prerequisites
    ):
        problems.append("task p6_10k_resource_probe_cpu: P5 verification prerequisite drifted")
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
    p6_verifier_outputs = {
        "p6_10k_verify_cpu": "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
        "p6_100k_verify_cpu": "proof/P6_CONTINUAL_100K_INDEPENDENT_VERIFICATION.json",
        "p6_1m_verify_cpu": "proof/P6_CONTINUAL_1M_INDEPENDENT_VERIFICATION.json",
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
                    "identity.source_preflight_payload_sha256": p6_evidence["preflight_payload_sha256"],
                    "identity.source_live_bindings_sha256": p6_evidence["preflight_live_bindings_sha256"],
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
        expected_verifier_atomic = _p6_verifier_atomic_write_projection(
            p6_evidence,
            seeds=5,
            schedules=2,
            arms=3,
        )
        for task_id, output_path in p6_verifier_outputs.items():
            verifier_task = tasks.get(task_id)
            if verifier_task is None:
                continue
            if (
                verifier_task.lane != "cpu"
                or verifier_task.accelerator != "none"
                or verifier_task.cpu_cores != 1
                or verifier_task.pause_safe
                or not verifier_task.atomic_checkpoints
            ):
                problems.append(f"task {task_id}: P6 verifier resource contract drifted")
            if verifier_task.checkpoint_globs != (output_path,):
                problems.append(f"task {task_id}: verifier output checkpoint glob drifted")
            if not math.isclose(
                verifier_task.atomic_write_gb,
                expected_verifier_atomic,
                abs_tol=1e-12,
            ):
                problems.append(f"task {task_id}: verifier atomic write projection drifted")
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
    if p6_evidence is not None:
        live_identity_fields = {
            "identity.config_sha256": p6_evidence["config_sha256"],
            "identity.runner_sha256": _sha256_file(REPO_ROOT / "scripts/continual_million_event_rung.py"),
            "identity.source_preflight_file_sha256": p6_evidence["preflight_file_sha256"],
            "identity.source_preflight_payload_sha256": p6_evidence["preflight_payload_sha256"],
            "identity.source_live_bindings_sha256": p6_evidence["preflight_live_bindings_sha256"],
        }

        def rung_requirement_fields(mode: str, rung: int) -> dict[str, Any]:
            cell_count = 1 if mode == "resource-probe" else 30
            fields: dict[str, Any] = {
                "mode": mode,
                "rung": rung,
                "all_mechanics_ok": True,
                "replication_execution_complete": mode == "replication",
                "independent_metric_verifier_complete": False,
                "scientific_promotion": False,
                "progress.completed_cells": cell_count,
                "progress.expected_cells": cell_count,
                **live_identity_fields,
            }
            if mode == "resource-probe":
                fields["resource_measurement.measured_after_complete"] = True
            return fields

        p6_source_requirements = {
            "p6_10k_replication_cpu": (
                "proof/P6_CONTINUAL_10K_RESOURCE_PILOT.json",
                "resource-probe",
                10_000,
            ),
            "p6_10k_verify_cpu": ("proof/P6_CONTINUAL_10K.json", "replication", 10_000),
            "p6_100k_replication_cpu": (
                "proof/P6_CONTINUAL_10K.json",
                "replication",
                10_000,
            ),
            "p6_100k_verify_cpu": (
                "proof/P6_CONTINUAL_100K.json",
                "replication",
                100_000,
            ),
            "p6_1m_replication_cpu": (
                "proof/P6_CONTINUAL_100K.json",
                "replication",
                100_000,
            ),
            "p6_1m_verify_cpu": (
                "proof/P6_CONTINUAL_1M.json",
                "replication",
                1_000_000,
            ),
        }
        for task_id, (path, mode, rung) in p6_source_requirements.items():
            p6_task = tasks.get(task_id)
            if p6_task is None:
                continue
            matches = [
                requirement
                for requirement in p6_task.prerequisites
                if requirement.path == path and requirement.schema == P6_RUNG_SCHEMA
            ]
            expected_fields = rung_requirement_fields(mode, rung)
            if len(matches) != 1 or dict(matches[0].fields) != expected_fields:
                problems.append(f"task {task_id}: exact live-bound P6 rung prerequisite drifted")

        p6_next_rung_verifiers = {
            "p6_100k_replication_cpu": (
                "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
                10_000,
                100_000,
            ),
            "p6_1m_replication_cpu": (
                "proof/P6_CONTINUAL_100K_INDEPENDENT_VERIFICATION.json",
                100_000,
                1_000_000,
            ),
        }
        for task_id, (path, source_rung, next_rung) in p6_next_rung_verifiers.items():
            dependent_task = tasks.get(task_id)
            if dependent_task is None:
                continue
            matches = [
                requirement
                for requirement in dependent_task.prerequisites
                if requirement.path == path and requirement.schema == P6_VERIFIER_SCHEMA
            ]
            if len(matches) != 1 or dict(matches[0].fields) != _p6_verifier_required_fields(
                source_rung, next_rung
            ):
                problems.append(f"task {task_id}: exact favorable independent verifier prerequisite drifted")
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


def _processes(
    policy: ThrottlePolicy,
    excluded_pids: set[int],
    excluded_process_groups: set[int] | None = None,
) -> dict[str, Any]:

    foreground_markers = [str(value).lower() for value in policy.monitor.get("foreground_markers", [])]
    heavy_markers = [str(value).lower() for value in policy.monitor.get("known_heavy_markers", [])]
    owned_groups = set(excluded_process_groups or ())
    foreground: list[dict[str, Any]] = []
    unmanaged_heavy: list[dict[str, Any]] = []
    inaccessible = 0
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "username"]):
        try:
            info = process.info
            pid = int(info["pid"])
            if pid in excluded_pids or pid in {os.getpid(), os.getppid()}:
                continue
            if owned_groups:
                try:
                    if os.getpgid(pid) in owned_groups:
                        continue
                except ProcessLookupError:
                    continue
                except PermissionError:
                    pass
            argv = [str(value) for value in (info.get("cmdline") or [])]
            command = " ".join(argv)
            searchable = f"{info.get('name') or ''} {command}".lower()
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
                identity_error: str | None = None
                try:
                    environment = process.environ()
                    row = {
                        **row,
                        "uid": int(process.uids().real),
                        "create_time": float(process.create_time()),
                        "exe": process.exe(),
                        "cwd": process.cwd(),
                        "cmdline": argv,
                        "rss_bytes": int(process.memory_info().rss),
                        "cpu_percent": float(process.cpu_percent(interval=0.01)),
                        "environment": {
                            "DOCTOR_DEVICE": environment.get("DOCTOR_DEVICE"),
                            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
                            "PYTORCH_ENABLE_MPS_FALLBACK": environment.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                        },
                    }
                except (
                    AttributeError,
                    psutil.AccessDenied,
                    psutil.NoSuchProcess,
                    psutil.ZombieProcess,
                    OSError,
                ) as exc:
                    identity_error = f"{type(exc).__name__}: {exc}"
                    row = {
                        **row,
                        "uid": None,
                        "create_time": None,
                        "exe": None,
                        "cwd": None,
                        "cmdline": argv,
                        "rss_bytes": None,
                        "cpu_percent": None,
                        "environment": None,
                    }
                row["identity_error"] = identity_error
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
    excluded_process_groups: set[int] | None = None,
) -> dict[str, Any]:

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
        processes = _processes(
            policy,
            excluded_pids or set(),
            excluded_process_groups=excluded_process_groups,
        )
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


def _producer_tasks_for_receipt(
    receipt_path: str,
    payload: dict[str, Any],
    policy: ThrottlePolicy,
) -> list[TaskDeclaration]:
    candidates = [
        task
        for task in policy.tasks.values()
        if _requires_completion_provenance(task) and _task_output_path(task) == receipt_path
    ]
    if receipt_path == "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json":
        primary_path = _dotted_value(payload, "primary_receipt.path")
        if primary_path == "proof/P5_CONTEXT_CAPABILITY_SMOKE.json":
            expected_task = "p5verify_smoke_null_cpu"
        elif primary_path == "proof/P5_CONTEXT_CAPABILITY_PILOT.json":
            expected_task = (
                "p5verify_cpu"
                if payload.get("fresh_challenge_required") is True
                else "p5verify_pilot_null_cpu"
            )
        else:
            expected_task = None
        candidates = [task for task in candidates if task.task_id == expected_task]
    substrate_producers = {
        (EDCM_RECEIPT_PATH, EDCM_RECEIPT_SCHEMA): "edcm1_official_cpu",
        (EDCM_VERIFICATION_PATH, EDCM_VERIFICATION_SCHEMA): "edcm1_verify_cpu",
        (X0_RECEIPT_PATH, X0_RECEIPT_SCHEMA): "escs_x0_official_cpu",
        (X0_VERIFICATION_PATH, X0_VERIFICATION_SCHEMA): "escs_x0_verify_cpu",
    }
    payload_schema = payload.get("schema")
    expected_substrate_task = (
        substrate_producers.get((receipt_path, payload_schema)) if isinstance(payload_schema, str) else None
    )
    if receipt_path in {
        EDCM_RECEIPT_PATH,
        EDCM_VERIFICATION_PATH,
        X0_RECEIPT_PATH,
        X0_VERIFICATION_PATH,
    }:
        candidates = [task for task in candidates if task.task_id == expected_substrate_task]
    return candidates


def _completion_receipt_problems(
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    task: TaskDeclaration,
    output_path: str,
    policy: ThrottlePolicy,
    evidence_root: Path,
) -> list[str]:
    problems: list[str] = []
    core = dict(receipt)
    declared_digest = core.pop("payload_sha256", None)
    if not _is_sha256(declared_digest) or _canonical_sha256(core) != declared_digest:
        problems.append("governor run receipt self seal drift")
    expected_task = _json_value(asdict(task))
    expected_command = list(task.command)
    expected_command_sha = _command_sha256(task.command)
    current_implementation = {
        "path": str(IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(IMPLEMENTATION_PATH),
    }
    declared_policy = receipt.get("policy")
    declared_implementation = receipt.get("implementation")
    declared_task = receipt.get("task")
    embedded_task_policy_authority = receipt.get("task_policy_authority")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("mode") != "execute":
        problems.append("governor run receipt schema or mode drift")
    if (
        receipt.get("status") != "complete"
        or receipt.get("command_executed") is not True
        or receipt.get("final_returncode") != 0
    ):
        problems.append("governor run did not complete with return code zero")
    if declared_task != expected_task:
        problems.append("governor run exact task declaration drift")
    if not isinstance(declared_implementation, dict) or (
        declared_implementation.get("path") != current_implementation["path"]
    ):
        problems.append("governor run implementation path drift")
    if embedded_task_policy_authority is not None and (
        not isinstance(declared_implementation, dict)
        or declared_implementation.get("sha256") != current_implementation["sha256"]
    ):
        problems.append("embedded task-policy authority uses a noncurrent governor implementation")
    try:
        legacy_manifests = _load_legacy_policy_baselines() if embedded_task_policy_authority is None else ()
        current_context = _task_policy_context(policy, task)
        problems.extend(
            task_policy.receipt_task_policy_authority_problems(
                declared_policy=declared_policy,
                declared_implementation=declared_implementation,
                declared_task_id=(
                    str(declared_task.get("task_id", "")) if isinstance(declared_task, dict) else ""
                ),
                declared_task_payload=declared_task if isinstance(declared_task, dict) else {},
                embedded_authority=embedded_task_policy_authority,
                legacy_manifests=legacy_manifests,
                current_policy_schema=current_context["policy_schema"],
                current_policy_path=current_context["policy_path"],
                current_full_policy_sha256=current_context["full_policy_sha256"],
                current_profile_name=current_context["profile_name"],
                current_safety_contract=current_context["safety_contract"],
                current_foreground_markers=current_context["foreground_markers"],
                current_known_heavy_markers=current_context["known_heavy_markers"],
                current_task_payload=current_context["task_payload"],
            )
        )
    except (OSError, TypeError, ValueError, ThrottleRefused) as exc:
        problems.append(f"governor task-policy authority invalid: {type(exc).__name__}: {exc}")
    invocations = receipt.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        problems.append("governor run has no completed invocation")
    else:
        for invocation in invocations:
            if (
                not isinstance(invocation, dict)
                or invocation.get("command") != expected_command
                or invocation.get("command_sha256") != expected_command_sha
            ):
                problems.append("governor run exact invocation command drift")
                break
        final_invocation = invocations[-1] if isinstance(invocations[-1], dict) else {}
        if final_invocation.get("returncode") != 0:
            problems.append("governor final invocation did not return zero")

    output_file = (evidence_root / output_path).resolve()
    output_sha = _sha256_file(output_file) if output_file.is_file() else None
    if not output_file.is_relative_to(evidence_root.resolve()) or output_sha is None:
        problems.append("governor bound output is missing or escapes the evidence root")
    current_snapshot = checkpoint_snapshot(task, evidence_root)
    final_checkpoint = receipt.get("final_checkpoint")
    if final_checkpoint != current_snapshot:
        problems.append("governor final checkpoint snapshot is stale")
    output_rows = [
        row for row in current_snapshot["files"] if isinstance(row, dict) and row.get("path") == output_path
    ]
    if len(output_rows) != 1 or output_rows[0].get("sha256") != output_sha:
        problems.append("governor output is absent from the final checkpoint authority")

    completion = receipt.get("completion_authority")
    child_resource = completion.get("child_resource") if isinstance(completion, dict) else None
    if not isinstance(completion, dict) or (
        completion.get("schema") != COMPLETION_AUTHORITY_SCHEMA
        or completion.get("task_id") != task.task_id
        or completion.get("task") != expected_task
        or completion.get("command") != expected_command
        or completion.get("command_sha256") != expected_command_sha
        or completion.get("policy") != declared_policy
        or completion.get("implementation") != declared_implementation
        or completion.get("task_policy_authority") != embedded_task_policy_authority
        or completion.get("returncode") != 0
        or completion.get("output") != {"path": output_path, "sha256": output_sha}
        or completion.get("final_checkpoint_aggregate_sha256") != current_snapshot["aggregate_sha256"]
        or completion.get("owned_child_active") is not False
    ):
        problems.append("governor completion authority drift")
    if not isinstance(child_resource, dict):
        problems.append("governor child resource authority is missing")
    else:
        psutil_peak = child_resource.get("psutil_peak_rss_bytes")
        rusage_peak = child_resource.get("direct_child_rusage_peak_rss_bytes")
        peak = child_resource.get("peak_rss_bytes")
        if (
            not isinstance(psutil_peak, int)
            or isinstance(psutil_peak, bool)
            or psutil_peak < 0
            or not isinstance(rusage_peak, int)
            or isinstance(rusage_peak, bool)
            or rusage_peak < 0
            or not isinstance(peak, int)
            or isinstance(peak, bool)
            or peak <= 0
            or peak != max(psutil_peak, rusage_peak)
        ):
            problems.append("governor child peak RSS authority is invalid")

    state_root = receipt_path.parents[1]
    try:
        if any(row.get("run_id") == receipt.get("run_id") for row in active_lanes(state_root)):
            problems.append("governor run still has an active owned child")
    except ThrottleRefused as exc:
        problems.append(f"governor active registry is invalid: {exc}")
    return problems


def _governor_provenance_report(
    receipt_path: str,
    payload: dict[str, Any],
    policy: ThrottlePolicy,
    evidence_root: Path,
) -> dict[str, Any]:
    producers = _producer_tasks_for_receipt(receipt_path, payload, policy)
    problems: list[str] = []
    if len(producers) != 1:
        problems.append(f"receipt maps to {len(producers)} governed producer tasks, expected exactly one")
        return {
            "all_ok": False,
            "producer_task_id": None,
            "run_receipt": None,
            "governor_peak_rss_bytes": None,
            "problems": problems,
        }
    task = producers[0]
    state_root = evidence_root / "runs/local_throttle"
    reports: list[dict[str, Any]] = []
    for path in sorted(state_root.glob("*/run_receipt.json")):
        try:
            loaded = json.loads(path.read_text())
            receipt = loaded if isinstance(loaded, dict) else {}
            receipt_problems = _completion_receipt_problems(
                receipt,
                receipt_path=path,
                task=task,
                output_path=receipt_path,
                policy=policy,
                evidence_root=evidence_root,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError, ThrottleRefused) as exc:
            receipt = {}
            receipt_problems = [f"governor run receipt invalid: {type(exc).__name__}: {exc}"]
        reports.append(
            {
                "path": str(path.relative_to(evidence_root)),
                "run_id": receipt.get("run_id"),
                "all_ok": not receipt_problems,
                "problems": receipt_problems,
                "governor_peak_rss_bytes": _dotted_value(
                    receipt, "completion_authority.child_resource.peak_rss_bytes"
                ),
            }
        )
    valid = [report for report in reports if report["all_ok"]]
    if not valid:
        problems.append("no matching successful sealed governor run receipt")
        problems.extend(
            f"{report['path']}: {problem}" for report in reports for problem in report["problems"]
        )
    selected = valid[-1] if valid else None
    return {
        "all_ok": selected is not None,
        "producer_task_id": task.task_id,
        "run_receipt": selected["path"] if selected else None,
        "governor_peak_rss_bytes": selected["governor_peak_rss_bytes"] if selected else None,
        "problems": problems,
    }


def _scoped_file_receipt(path: Path, evidence_root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        label = str(path.resolve().relative_to(evidence_root.resolve()))
    except ValueError:
        label = str(path.resolve())
    return {"path": label, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _joined_producer_payload(
    relative_path: str,
    expected_schema: str,
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    path = (evidence_root / relative_path).resolve()
    problems: list[str] = []
    if not path.is_relative_to(evidence_root.resolve()) or not path.is_file():
        return {}, None, [f"joined producer {relative_path} is missing or escapes the evidence root"]
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, None, [f"joined producer is unreadable: {type(exc).__name__}"]
    payload = loaded if isinstance(loaded, dict) else {}
    if payload.get("schema") != expected_schema:
        problems.append("joined producer schema drift")
    problems.extend(_native_schema_authority_problems(expected_schema, payload, evidence_root))
    try:
        file_receipt = _scoped_file_receipt(path, evidence_root)
    except OSError as exc:
        problems.append(f"joined producer changed during validation: {type(exc).__name__}")
        file_receipt = None
    return payload, file_receipt, problems


def _native_schema_authority_problems(
    schema: str,
    payload: dict[str, Any],
    evidence_root: Path,
) -> list[str]:

    problems: list[str] = []
    seal_field = NATIVE_SEAL_FIELDS.get(schema)
    if seal_field is None:
        return problems
    if payload.get("schema") != schema:
        problems.append("native authority schema drift")
        return problems
    core = dict(payload)
    declared_seal = core.pop(seal_field, None)
    if not _is_sha256(declared_seal) or declared_seal != _canonical_sha256(core):
        problems.append(f"native {schema} {seal_field} drift")
    if schema == ESCS_PREFLIGHT_SCHEMA:
        if payload.get("scientific_promotion_allowed") is not False:
            problems.append("substrate preflight scientific promotion escaped")
        return problems
    if payload.get("scientific_promotion") is not False:
        problems.append("native substrate scientific promotion escaped")
    if schema == EDCM_RECEIPT_SCHEMA:
        deterministic_core = dict(core)
        declared_deterministic = deterministic_core.pop("deterministic_core_sha256", None)
        if not _is_sha256(declared_deterministic) or declared_deterministic != _canonical_sha256(
            deterministic_core
        ):
            problems.append("EDCM deterministic core seal drift")
        return problems
    if schema == EDCM_VERIFICATION_SCHEMA:
        producer, producer_file, producer_problems = _joined_producer_payload(
            EDCM_RECEIPT_PATH,
            EDCM_RECEIPT_SCHEMA,
            evidence_root,
        )
        problems.extend(producer_problems)
        verification = payload.get("verification")
        if not isinstance(verification, dict):
            problems.append("EDCM verification result is missing")
            return problems
        sources = verification.get("verified_sources")
        if not isinstance(sources, dict) or producer_file is None:
            problems.append("EDCM verified producer sources are missing")
        else:
            if sources.get("receipt") != producer_file:
                problems.append("EDCM verifier producer file join drift")
            declared_path = sources.get("receipt_path")
            if not isinstance(declared_path, str):
                problems.append("EDCM verifier producer path is missing")
            else:
                source_path = Path(declared_path)
                resolved = (
                    source_path.resolve()
                    if source_path.is_absolute()
                    else (evidence_root / source_path).resolve()
                )
                if resolved != (evidence_root / EDCM_RECEIPT_PATH).resolve():
                    problems.append("EDCM verifier producer path join drift")
        if verification.get("authority_sha256") != producer.get("authority_sha256"):
            problems.append("EDCM verifier config authority join drift")
        if verification.get("implementation_authority_sha256") != producer.get(
            "implementation_authority_sha256"
        ):
            problems.append("EDCM verifier implementation authority join drift")
        if verification.get("execution_status") != producer.get("execution_status"):
            problems.append("EDCM verifier execution-status join drift")
        if verification.get("verdict") != _dotted_value(producer, "aggregate.verdict"):
            problems.append("EDCM verifier verdict join drift")
        if verification.get("scientific_promotion") is not False:
            problems.append("EDCM verification scientific promotion escaped")
        return problems
    if schema == X0_VERIFICATION_SCHEMA:
        producer, producer_file, producer_problems = _joined_producer_payload(
            X0_RECEIPT_PATH,
            X0_RECEIPT_SCHEMA,
            evidence_root,
        )
        problems.extend(producer_problems)
        if producer_file is None or payload.get("producer_receipt") != producer_file:
            problems.append("X0 verifier producer file join drift")
        if payload.get("producer_receipt_sha256") != producer.get("receipt_sha256"):
            problems.append("X0 verifier producer receipt seal join drift")
        if payload.get("implementation_authority_sha256") != _dotted_value(
            producer, "implementation_authority.manifest_sha256"
        ):
            problems.append("X0 verifier implementation authority join drift")
        if payload.get("primary_aggregate") != producer.get("aggregate"):
            problems.append("X0 verifier primary aggregate join drift")
    return problems


def _receipt_requirement_report(
    requirement: ReceiptRequirement,
    evidence_root: Path,
    policy: ThrottlePolicy,
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
    if payload and requirement.schema in PAYLOAD_DIGEST_REQUIRED_SCHEMAS:
        receipt_payload = dict(payload)
        declared_digest = receipt_payload.pop("payload_sha256", None)
        if _canonical_sha256(receipt_payload) != declared_digest:
            problems.append("prerequisite receipt payload digest drift")
    if payload and requirement.schema in {
        P5_SCREEN_SCHEMA,
        P5_GRID_SCHEMA,
        P5_CHALLENGE_SCHEMA,
        P5_VERIFIER_SCHEMA,
    }:
        problems.extend(_p5_schema_authority_problems(requirement.schema, payload, evidence_root))
    elif payload and requirement.schema == P6_RUNG_SCHEMA:
        problems.extend(_p6_rung_authority_problems(payload, evidence_root))
    elif payload and requirement.schema == P6_VERIFIER_SCHEMA:
        problems.extend(_p6_verifier_authority_problems(payload, evidence_root))
    elif payload and requirement.schema in NATIVE_SEAL_FIELDS:
        problems.extend(_native_schema_authority_problems(requirement.schema, payload, evidence_root))
    provenance: dict[str, Any] | None = None
    if payload and requirement.schema in GOVERNED_PROVENANCE_SCHEMAS:
        provenance = _governor_provenance_report(
            requirement.path,
            payload,
            policy,
            evidence_root,
        )
        problems.extend(provenance["problems"])
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
        "governor_provenance": provenance,
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
        problems.extend(_p6_rung_authority_problems(payload, evidence_root))
    if payload and payload.get("rung") != task.resource_receipt_rung:
        problems.append("resource receipt rung drift")
    if payload and payload.get("all_mechanics_ok") is not True:
        problems.append("resource receipt mechanics did not complete")
    measurement = payload.get("resource_measurement") if isinstance(payload, dict) else None
    runner_rss_bytes = measurement.get("max_rss_bytes") if isinstance(measurement, dict) else None
    if (
        not isinstance(runner_rss_bytes, int)
        or isinstance(runner_rss_bytes, bool)
        or runner_rss_bytes < P6_MINIMUM_SANE_RSS_BYTES
    ):
        problems.append(
            f"resource receipt max_rss_bytes is below the {P6_MINIMUM_SANE_RSS_BYTES}-byte sanity floor"
        )
        runner_rss_bytes = 0
    if isinstance(measurement, dict) and measurement.get("measured_after_complete") is not True:
        problems.append("resource receipt was not measured after a complete rung/probe")
    provenance: dict[str, Any] = (
        _governor_provenance_report(str(task.resource_receipt_path), payload, policy, evidence_root)
        if payload
        else {
            "all_ok": False,
            "producer_task_id": None,
            "run_receipt": None,
            "governor_peak_rss_bytes": None,
            "problems": ["resource receipt has no governor provenance candidate"],
        }
    )
    problems.extend(provenance["problems"])
    governor_rss_bytes = provenance.get("governor_peak_rss_bytes")
    if (
        not isinstance(governor_rss_bytes, int)
        or isinstance(governor_rss_bytes, bool)
        or governor_rss_bytes <= 0
    ):
        problems.append("resource receipt governor peak RSS is missing")
        governor_rss_bytes = 0
    rss_bytes = max(runner_rss_bytes, governor_rss_bytes)
    uncertainty = float(policy.limits["forecast_uncertainty_fraction"])
    effective = rss_bytes / 1e9 * (1.0 + uncertainty)
    return effective, {
        "mode": "measured-prior-rung",
        "path": str(task.resource_receipt_path),
        "schema": payload.get("schema"),
        "rung": payload.get("rung"),
        "runner_max_rss_bytes": runner_rss_bytes or None,
        "governor_max_rss_bytes": governor_rss_bytes or None,
        "max_rss_bytes": rss_bytes or None,
        "governor_provenance": provenance,
        "uncertainty_fraction": uncertainty,
        "effective_unified_memory_gb": effective if not problems else None,
        "problems": problems,
        "all_ok": not problems,
    }


def _external_coexistence_report(
    task: TaskDeclaration,
    unmanaged: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not unmanaged or not _is_external_coexistence_task(task):
        return None
    v5_report: dict[str, Any] | None = None
    try:
        v5_profile = _hawking_v5_coexistence_profile()
        v5_report = coexistence.validate_hawking_v5_ultra_snapshot(unmanaged, v5_profile)
        if v5_report.get("allowed") is True:
            return v5_report
    except (OSError, TypeError, ValueError, ThrottleRefused) as exc:
        v5_report = {
            "schema": coexistence.REPORT_SCHEMA,
            "profile": coexistence.V5_PROFILE_NAME,
            "allowed": False,
            "all_ok": False,
            "problems": [f"v5 profile validation failed closed: {type(exc).__name__}: {exc}"],
            "ownership": "observation-only; external processes must never receive signals",
            "scientific_promotion": False,
        }
    try:
        profile = _hawking_coexistence_profile()
        legacy_report = coexistence.validate_hawking_serial_cpu_snapshot(unmanaged, profile)
        if legacy_report.get("allowed") is True:
            return legacy_report
        return {
            **(v5_report or legacy_report),
            "problems": [
                *(list((v5_report or {}).get("problems") or [])),
                *(f"legacy profile: {problem}" for problem in legacy_report.get("problems") or []),
            ],
        }
    except (OSError, TypeError, ValueError, ThrottleRefused) as exc:
        return {
            "schema": coexistence.REPORT_SCHEMA,
            "profile": (v5_report or {}).get("profile", EXTERNAL_COEXISTENCE_PROFILE),
            "allowed": False,
            "all_ok": False,
            "problems": [
                *(list((v5_report or {}).get("problems") or [])),
                f"legacy profile validation failed closed: {type(exc).__name__}: {exc}",
            ],
            "ownership": "observation-only; external processes must never receive signals",
            "scientific_promotion": False,
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

    active = list(active or [])
    evidence_path = Path(evidence_root).resolve()
    gates: list[dict[str, Any]] = []
    missing = list(telemetry.get("missing_required_telemetry") or [])
    _gate(gates, "required_telemetry", not missing, missing, [], "all required probes must report")
    prerequisite_reports = [
        _receipt_requirement_report(requirement, evidence_path, policy) for requirement in task.prerequisites
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
    external_report = _external_coexistence_report(task, unmanaged)
    validated_external_coexistence = bool(
        unmanaged and external_report is not None and external_report.get("allowed") is True
    )
    validated_external_v5 = bool(
        validated_external_coexistence
        and external_report is not None
        and external_report.get("profile") == coexistence.V5_PROFILE_NAME
    )
    _gate(
        gates,
        "unmanaged_heavy_process",
        not unmanaged or validated_external_coexistence,
        (
            external_report
            if external_report is not None
            else [{"pid": row.get("pid"), "name": row.get("name")} for row in unmanaged]
        ),
        [],
        (
            "exact Hawking CPU profile permits one bounded background-QoS CPU lane"
            if validated_external_coexistence
            else (
                "unknown or profile-invalid heavy work has no resource declaration, so admission fails closed"
            )
        ),
        critical=bool(
            task_already_active
            and unmanaged
            and not validated_external_coexistence
        ),
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
    decision_tier_name = "external_coexistence" if validated_external_coexistence else tier_name
    thresholds = policy.thresholds[tier_name]
    cpu = telemetry.get("cpu") or {}
    load = cpu.get("load_1m_per_logical_cpu")
    load_max = (
        1.0
        if validated_external_v5
        else (0.85 if validated_external_coexistence else thresholds["maximum_load_per_logical_cpu"])
    )
    cpu_is_admission_only = task_already_active and not validated_external_coexistence
    v5_background_qos = validated_external_v5
    cpu_limit_label: float | str = "kernel-enforced background QoS" if v5_background_qos else load_max
    _gate(
        gates,
        "cpu_load",
        cpu_is_admission_only
        or v5_background_qos
        or (isinstance(load, int | float) and float(load) <= load_max),
        load,
        "admission-only" if cpu_is_admission_only else cpu_limit_label,
        (
            "owned task CPU load is expected at runtime; memory, swap, thermal, and disk remain gated"
            if cpu_is_admission_only
            else (
                "one-thread v5 work yields through kernel-enforced background QoS"
                if v5_background_qos
                else f"{decision_tier_name} normalized one-minute load ceiling"
            )
        ),
    )
    utilization = cpu.get("utilization_fraction")
    utilization_max = (
        1.0 if validated_external_coexistence else thresholds["maximum_cpu_utilization_fraction"]
    )
    _gate(
        gates,
        "cpu_utilization",
        cpu_is_admission_only
        or v5_background_qos
        or (isinstance(utilization, int | float) and float(utilization) <= utilization_max),
        utilization,
        (
            "admission-only"
            if cpu_is_admission_only
            else ("kernel-enforced background QoS" if v5_background_qos else utilization_max)
        ),
        (
            "owned task CPU utilization is expected at runtime; pressure gates remain active"
            if cpu_is_admission_only
            else (
                "one-thread v5 work yields through kernel-enforced background QoS"
                if v5_background_qos
                else f"{decision_tier_name} instantaneous CPU ceiling"
            )
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
    if validated_external_v5 and external_report is not None:
        observed_external = external_report.get("observed") or {}
        process_rows = observed_external.get("processes") or []
        quant_threads = sum(
            int(row.get("quant_threads") or 0) for row in process_rows if isinstance(row, dict)
        )
        aggregate_cpu_percent = float(observed_external.get("aggregate_cpu_percent") or 0.0)
        observed_external_cores = max(quant_threads, math.ceil(aggregate_cpu_percent / 100.0))
        coexistence_core_ceiling = math.floor(logical * 0.95)
        effective_candidate_cores = _effective_external_task_cores(task)
        observed_cpu_budget = {
            "candidate_cores": effective_candidate_cores,
            "observed_external_cores": observed_external_cores,
            "quant_threads": quant_threads,
        }
        if effective_candidate_cores != task.cpu_cores:
            observed_cpu_budget["declared_idle_cores"] = task.cpu_cores
        _gate(
            gates,
            "external_coexistence_cpu_budget",
            logical > 0 and effective_candidate_cores + observed_external_cores <= coexistence_core_ceiling,
            observed_cpu_budget,
            coexistence_core_ceiling,
            "parallel MOP plus observed Hawking demand remains within ninety-five percent of CPUs",
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
    if validated_external_coexistence:
        coexistence_thresholds = policy.thresholds.get("external_coexistence") or {}
        coexistence_memory_percent = float(
            coexistence_thresholds.get("minimum_memory_available_percent", 40.0)
        )
        coexistence_memory_gb = float(coexistence_thresholds.get("minimum_memory_available_gb", 40.0))
        coexistence_pressure_percent = float(
            coexistence_thresholds.get("minimum_memory_pressure_free_percent", 75.0)
        )
        _gate(
            gates,
            "external_coexistence_memory_percent",
            effective_percent is not None and effective_percent >= coexistence_memory_percent,
            effective_percent,
            coexistence_memory_percent,
            "validated Hawking coexistence retains the policy-declared available-memory fraction",
            critical=effective_percent is not None and effective_percent < 8.0,
        )
        _gate(
            gates,
            "external_coexistence_memory_gb",
            available_gb >= coexistence_memory_gb,
            available_gb,
            coexistence_memory_gb,
            "validated Hawking coexistence retains the policy-declared absolute memory floor",
            critical=available_gb < 6.0,
        )
        _gate(
            gates,
            "external_coexistence_pressure_free",
            isinstance(pressure_percent, int | float)
            and float(pressure_percent) >= coexistence_pressure_percent,
            pressure_percent,
            coexistence_pressure_percent,
            "memory_pressure must retain the policy-declared free fraction during coexistence",
            critical=(isinstance(pressure_percent, int | float) and float(pressure_percent) < 30.0),
        )
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
    if validated_external_coexistence:
        if validated_external_v5:
            coexistence_swap_max = float(
                (policy.thresholds.get("external_coexistence") or {}).get("maximum_swap_used_gb", 0.25)
            )
            _gate(
                gates,
                "external_coexistence_low_swap",
                isinstance(swap_used, int | float) and float(swap_used) <= coexistence_swap_max,
                swap_used,
                coexistence_swap_max,
                "v5 coexistence retains the policy-declared historical swap ceiling",
                critical=isinstance(swap_used, int | float) and float(swap_used) > 6.0,
            )
        else:
            _gate(
                gates,
                "external_coexistence_zero_swap",
                isinstance(swap_used, int | float) and float(swap_used) == 0.0,
                swap_used,
                0.0,
                "legacy coexistence requires exactly zero swap use",
                critical=isinstance(swap_used, int | float) and float(swap_used) > 0.0,
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
        "threshold_tier": decision_tier_name,
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
    denied_reasons: list[str] = []
    for decision in tail:
        if bool(decision.get("allowed")):
            continue
        values = decision.get("denied_reasons")
        if not isinstance(values, list):
            continue
        for value in values:
            reason = str(value)
            if reason and reason not in denied_reasons:
                denied_reasons.append(reason)
    if not allowed and len(tail) < required_good:
        denied_reasons.append(
            f"admission observed {len(tail)} of {required_good} required consecutive samples"
        )
    return {
        "allowed": allowed,
        "required_consecutive_good_samples": required_good,
        "samples_observed": len(decisions),
        "consecutive_good_samples": _trailing_count(decisions, True),
        "consecutive_bad_samples": _trailing_count(decisions, False),
        "denied_reasons": denied_reasons,
        "reason": (
            "admission hysteresis satisfied"
            if allowed
            else "; ".join(denied_reasons) or "admission requires the configured consecutive healthy samples"
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
    policy_binding = {"path": str(policy.path), "sha256": policy.sha256}
    implementation_binding = {
        "path": str(IMPLEMENTATION_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(IMPLEMENTATION_PATH),
    }
    task_payload = _json_value(asdict(task))
    admission_task_policy_authority = _build_task_policy_authority(policy, task)
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
        "policy": policy_binding,
        "implementation": implementation_binding,
        "task_policy_authority": admission_task_policy_authority,
        "profile": get_profile(policy.profile_name).as_dict(),
        "task": task_payload,
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
        "external_coexistence": (
            EXTERNAL_COEXISTENCE_PROFILE if _is_external_coexistence_task(task) else None
        ),
        "max_invocations_per_run": 1 if _is_seed_boundary_task(task) else None,
        "command": list(task.command),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _signal_owned_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:

    if process.poll() is None:
        os.killpg(process.pid, sig)


def _stop_at_wall(
    process: subprocess.Popen[Any],
    task: TaskDeclaration,
    receipt: dict[str, Any],
    receipt_path: Path,
    grace_seconds: float,
    evidence_root: Path,
) -> int | None:
    _signal_owned_group(process, signal.SIGSTOP)
    snapshot = checkpoint_snapshot(task, evidence_root)
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


def _stop_non_pause_safe_child(
    process: subprocess.Popen[Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    grace_seconds: float,
) -> int | None:

    _signal_owned_group(process, signal.SIGTERM)
    _event(
        receipt,
        "dynamic-safety-stop",
        child_pid=process.pid,
        note="non-pause-safe owned child received SIGTERM and cannot receive completion authority",
    )
    _atomic_json(receipt_path, receipt)
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        _signal_owned_group(process, signal.SIGSTOP)
        _event(
            receipt,
            "managed-child-stopped",
            child_pid=process.pid,
            reason="non-pause-safe child did not exit after SIGTERM and remains stopped",
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

    if not run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ThrottleRefused("run_id must contain only letters, digits, dot, underscore, or hyphen")
    root = Path(state_root)
    evidence_root = Path(disk_root).resolve()
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
        admission_receipt["run_id"] = run_id
        admission_receipt["status"] = "admission-refused"
        _atomic_json(receipt_path, admission_receipt)
        return admission_receipt
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
    dynamic_safety_stop = False
    psutil_peak_rss_bytes = 0
    rusage_peak_rss_bytes = _rusage_children_peak_rss_bytes()
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
                psutil_peak_rss_bytes = max(
                    psutil_peak_rss_bytes,
                    _process_tree_rss_bytes(process.pid),
                )
                _event(receipt, "invocation-start", index=invocation, child_pid=process.pid)
                _update_registry(root, run_id, _registry_row(task, run_id, process.pid, "running"))
                _atomic_json(receipt_path, receipt)
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(max(0.1, float(policy.monitor["sample_interval_seconds"])))
                    psutil_peak_rss_bytes = max(
                        psutil_peak_rss_bytes,
                        _process_tree_rss_bytes(process.pid),
                    )
                    other_active = [row for row in active_lanes(root) if row.get("run_id") != run_id]
                    telemetry = collect_host_telemetry(
                        policy,
                        disk_root=disk_root,
                        excluded_pids={process.pid},
                        excluded_process_groups={process.pid},
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
                        checkpoint = checkpoint_snapshot(task, evidence_root)
                        _event(
                            receipt,
                            "dynamic-pause",
                            decision=decision,
                            checkpoint=checkpoint,
                            note="SIGSTOP targets this scheduler-owned process group only",
                        )
                        _update_registry(root, run_id, _registry_row(task, run_id, process.pid, "paused"))
                    elif transition["action"] == "pause":
                        dynamic_safety_stop = True
                        final_returncode = _stop_non_pause_safe_child(
                            process,
                            receipt,
                            receipt_path,
                            float(policy.monitor["graceful_stop_seconds"]),
                        )
                        receipt["dynamic_safety_decision"] = decision
                        break
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
                        evidence_root,
                    )
                    rusage_peak_rss_bytes = max(
                        rusage_peak_rss_bytes,
                        _rusage_children_peak_rss_bytes(),
                    )
                    invocation_row["returncode"] = final_returncode
                    invocation_row["finished_at"] = datetime.now(UTC).isoformat()
                    receipt["status"] = "resumable-wall-boundary"
                    break
                final_returncode = process.poll()
                psutil_peak_rss_bytes = max(
                    psutil_peak_rss_bytes,
                    _process_tree_rss_bytes(process.pid),
                )
                rusage_peak_rss_bytes = max(
                    rusage_peak_rss_bytes,
                    _rusage_children_peak_rss_bytes(),
                )
                invocation_row["returncode"] = final_returncode
                invocation_row["finished_at"] = datetime.now(UTC).isoformat()
                invocation_row["checkpoint_after"] = checkpoint_snapshot(task, evidence_root)
                _event(
                    receipt,
                    "invocation-exit",
                    index=invocation,
                    returncode=final_returncode,
                )
                _atomic_json(receipt_path, receipt)
            if dynamic_safety_stop:
                receipt["status"] = "failed-dynamic-safety-stop"
                break
            if final_returncode == 0:
                receipt["status"] = "complete"
                break
            if final_returncode not in task.restart_exit_codes:
                receipt["status"] = "failed"
                break
            if _is_seed_boundary_task(task) and invocation >= 1:
                receipt["status"] = "resumable-invocation-boundary"
                _event(
                    receipt,
                    "invocation-boundary-yield",
                    invocation=invocation,
                    returncode=final_returncode,
                    checkpoint=checkpoint_snapshot(task, evidence_root),
                    note="campaign must re-run admission before the next deterministic seed boundary",
                )
                _atomic_json(receipt_path, receipt)
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
        receipt["final_checkpoint"] = checkpoint_snapshot(task, evidence_root)
        child_peak_rss_bytes = max(psutil_peak_rss_bytes, rusage_peak_rss_bytes)
        child_resource = {
            "psutil_peak_rss_bytes": psutil_peak_rss_bytes,
            "direct_child_rusage_peak_rss_bytes": rusage_peak_rss_bytes,
            "peak_rss_bytes": child_peak_rss_bytes,
            "methods": ["psutil-process-tree", "getrusage-RUSAGE_CHILDREN"],
        }
        receipt["child_resource"] = child_resource
        if receipt["status"] in {
            "resumable-wall-boundary",
            "resumable-invocation-boundary",
        }:
            progress_problems: list[str] = []
            if final_returncode not in task.restart_exit_codes and receipt["status"] == (
                "resumable-invocation-boundary"
            ):
                progress_problems.append("invocation-boundary return code is not restart-authorized")
            if child_peak_rss_bytes <= 0:
                progress_problems.append("resumable task has no observed child peak RSS")
            if process is not None and process.poll() is None:
                progress_problems.append("resumable task still has an active owned child")
            admission_task = receipt.get("task")
            admission_policy = receipt.get("policy")
            admission_implementation = receipt.get("implementation")
            admission_task_policy_authority = receipt.get("task_policy_authority")
            if admission_task != _json_value(asdict(task)):
                progress_problems.append("resumable task admission declaration drifted")
            if not isinstance(admission_policy, dict):
                progress_problems.append("resumable task has no admission-time policy binding")
            if not isinstance(admission_implementation, dict):
                progress_problems.append("resumable task has no admission-time implementation binding")
            if not isinstance(admission_task_policy_authority, dict):
                progress_problems.append("resumable task has no task-policy authority")
            if progress_problems:
                receipt["status"] = "failed-progress-authority"
                receipt["progress_problems"] = progress_problems
            else:
                receipt["progress_authority"] = {
                    "schema": PROGRESS_AUTHORITY_SCHEMA,
                    "task_id": task.task_id,
                    "task": admission_task,
                    "command": list(task.command),
                    "command_sha256": _command_sha256(task.command),
                    "policy": admission_policy,
                    "implementation": admission_implementation,
                    "task_policy_authority": admission_task_policy_authority,
                    "returncode": final_returncode,
                    "final_checkpoint_aggregate_sha256": receipt["final_checkpoint"]["aggregate_sha256"],
                    "owned_child_active": False,
                    "child_resource": child_resource,
                }
                core = dict(receipt)
                core.pop("payload_sha256", None)
                receipt["payload_sha256"] = _canonical_sha256(core)
        if receipt["status"] == "complete" and _requires_completion_provenance(task):
            output_path = _task_output_path(task)
            output_file = (evidence_root / str(output_path)).resolve()
            output_rows = [
                row
                for row in receipt["final_checkpoint"]["files"]
                if isinstance(row, dict) and row.get("path") == output_path
            ]
            completion_problems: list[str] = []
            if (
                output_path is None
                or not output_file.is_relative_to(evidence_root)
                or not output_file.is_file()
            ):
                completion_problems.append("successful task did not publish its declared output")
            output_sha = _sha256_file(output_file) if output_file.is_file() else None
            if len(output_rows) != 1 or output_rows[0].get("sha256") != output_sha:
                completion_problems.append(
                    "successful task output is absent from its final checkpoint snapshot"
                )
            if child_peak_rss_bytes <= 0:
                completion_problems.append("successful task has no observed child peak RSS")
            if process is not None and process.poll() is None:
                completion_problems.append("successful task still has an active owned child")
            admission_policy = receipt.get("policy")
            admission_implementation = receipt.get("implementation")
            admission_task = receipt.get("task")
            admission_task_policy_authority = receipt.get("task_policy_authority")
            if not isinstance(admission_policy, dict):
                completion_problems.append("successful task has no admission-time policy binding")
            if not isinstance(admission_implementation, dict):
                completion_problems.append("successful task has no admission-time implementation binding")
            if admission_task != _json_value(asdict(task)):
                completion_problems.append("successful task admission declaration drifted")
            if not isinstance(admission_task_policy_authority, dict):
                completion_problems.append("successful task has no admission-time task-policy authority")
            if completion_problems:
                receipt["status"] = "failed-completion-authority"
                receipt["completion_problems"] = completion_problems
            else:
                receipt["completion_authority"] = {
                    "schema": COMPLETION_AUTHORITY_SCHEMA,
                    "task_id": task.task_id,
                    "task": admission_task,
                    "command": list(task.command),
                    "command_sha256": _command_sha256(task.command),
                    "policy": admission_policy,
                    "implementation": admission_implementation,
                    "task_policy_authority": admission_task_policy_authority,
                    "returncode": 0,
                    "output": {"path": output_path, "sha256": output_sha},
                    "final_checkpoint_aggregate_sha256": receipt["final_checkpoint"]["aggregate_sha256"],
                    "owned_child_active": False,
                    "child_resource": child_resource,
                }
                core = dict(receipt)
                core.pop("payload_sha256", None)
                receipt["payload_sha256"] = _canonical_sha256(core)
        _atomic_json(receipt_path, receipt)
    return receipt


def write_receipt(path: Path | str, payload: dict[str, Any]) -> None:
    _atomic_json(Path(path), payload)
