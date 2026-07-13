"""Detached null-safe routing for the first ESCS/P6 campaign sequence.

The existing campaign supervisor intentionally has one exact terminal artifact
contract per step.  This router keeps that supervisor unchanged by dividing the
P6 scale ladder into three sealed campaigns.  It advances to the next campaign
only after independently checking the completed verifier receipt and its sealed
campaign join.  A valid null is a successful terminal route, never an admission
retry for a higher rung.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from ..config import REPO_ROOT
from ..studies.continual_million_event_verify import IMPLEMENTATION_PATHS, TIE_RULE
from .campaign_supervisor import (
    PLAN_SCHEMA,
    STATUS_FILE,
    TERMINAL_STATES,
    load_campaign_plan,
    read_campaign_status,
    request_drain,
    sha256_file,
    start_detached,
)
from .local_throttle import TaskDeclaration, load_policy
from .policy_overlay import load_task_overlay

ROUTER_PLAN_SCHEMA = "mop-p6-null-safe-router-plan/v1"
ROUTER_STATE_SCHEMA = "mop-p6-null-safe-router-state/v1"
ROUTER_STATUS_SCHEMA = "mop-p6-null-safe-router-status/v1"
ROUTE_RECEIPT_SCHEMA = "mop-p6-null-safe-route/v1"
CONTROL_SCHEMA = "mop-p6-null-safe-router-control/v1"
P6_VERIFIER_SCHEMA = "mop-continual-progressive-rung-independent-verifier/v1"
P6_RUNG_SCHEMA = "mop-continual-progressive-rung/v1"

DEFAULT_ROUTER_PLAN = REPO_ROOT / "configs/campaign/mac_studio_substrate_null_safe_router.json"
DEFAULT_OVERLAY = REPO_ROOT / "configs/campaign/substrate_coexistence_task_overlay.yaml"
DEFAULT_POLICY = REPO_ROOT / "configs/local_execution_throttle.yaml"
ROUTER_STATE_FILE = "router_state.json"
ROUTER_STATUS_FILE = "current_status.json"
ROUTER_LOCK_FILE = "router.lock"
ROUTER_START_LOCK_FILE = "start.lock"
ROUTER_CONTROL_FILE = "control/stop-request.json"
ROUTES_DIR = "routes"
MAX_JSON_BYTES = 256 * 1024 * 1024

EXPECTED_CHECK_NAMES = frozenset(
    {
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
)
EXPECTED_METRIC_FAMILIES = (
    "retention",
    "acquisition",
    "future_learnability",
    "stale_memory",
    "deletion",
    "resources",
)
EXPECTED_CONTROLS = ("no-replay", "fresh-init")
EXPECTED_SCHEDULES = ("abrupt", "gradual")
NULL_REASON = "verified tie, null, invalid evidence, or final rung does not admit scaling"
FAVORABLE_REASON = "strict favorable programmatic pattern requires the next scale confirmation"


class RouterRefused(RuntimeError):
    """Fail-closed router validation or execution refusal."""


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouterRefused(f"{label} must be an object")
    if set(value) != expected:
        raise RouterRefused(
            f"{label} fields drifted; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _safe_repo_path(value: object, label: str, *, root: Path = REPO_ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise RouterRefused(f"{label} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RouterRefused(f"{label} must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RouterRefused(f"{label} escapes the repository")
    return path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: canonical_sha256(core)}


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    core = dict(payload)
    declared = core.pop(field, None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise RouterRefused(f"{label} self-seal mismatch")


def _read_json_snapshot(path: Path, label: str, *, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise RouterRefused(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise RouterRefused(f"{label} must be a regular non-symlink file")
    if stat.st_size <= 0 or stat.st_size > maximum_bytes:
        raise RouterRefused(f"{label} byte envelope is invalid")
    raw = path.read_bytes()
    if len(raw) != stat.st_size:
        raise RouterRefused(f"{label} changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouterRefused(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RouterRefused(f"{label} must be an object")
    return payload


@dataclass(frozen=True, slots=True)
class Stage:
    stage_id: str
    plan_path: Path
    verifier_path: Path
    source_path: Path
    rung: int
    next_rung: int | None


@dataclass(frozen=True, slots=True)
class RouterPlan:
    path: Path
    sha256: str
    router_id: str
    router_root: Path
    poll_seconds: float
    campaign_entrypoint: Path
    stages: tuple[Stage, ...]

    @property
    def out_dir(self) -> Path:
        return self.router_root / self.router_id


def load_router_plan(path: Path | str = DEFAULT_ROUTER_PLAN) -> RouterPlan:
    source = Path(path).resolve()
    payload = _read_json_snapshot(source, "router plan", maximum_bytes=1024 * 1024)
    _exact_keys(
        payload,
        {
            "schema",
            "router_id",
            "router_root",
            "poll_seconds",
            "campaign_entrypoint",
            "stages",
        },
        "router plan",
    )
    if payload["schema"] != ROUTER_PLAN_SCHEMA:
        raise RouterRefused(f"router plan schema must be {ROUTER_PLAN_SCHEMA}")
    router_id = payload["router_id"]
    if (
        not isinstance(router_id, str)
        or not router_id
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in router_id
        )
    ):
        raise RouterRefused("router id is invalid")
    poll_seconds = payload["poll_seconds"]
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, int | float) or poll_seconds <= 0:
        raise RouterRefused("router poll_seconds must be positive")
    router_root = _safe_repo_path(payload["router_root"], "router_root")
    campaign_entrypoint = _safe_repo_path(payload["campaign_entrypoint"], "campaign_entrypoint")
    if not campaign_entrypoint.is_file():
        raise RouterRefused("campaign entrypoint is missing")
    rows = payload["stages"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise RouterRefused("router requires exactly the 10k, 100k, and 1m stages")
    stages: list[Stage] = []
    for index, raw in enumerate(rows):
        stage = _exact_keys(raw, {"id", "plan", "verifier", "source", "rung", "next_rung"}, f"stage {index}")
        stage_id = stage["id"]
        if not isinstance(stage_id, str) or not stage_id:
            raise RouterRefused(f"stage {index} id is invalid")
        rung = stage["rung"]
        next_rung = stage["next_rung"]
        if isinstance(rung, bool) or not isinstance(rung, int):
            raise RouterRefused(f"stage {index} rung is invalid")
        if next_rung is not None and (isinstance(next_rung, bool) or not isinstance(next_rung, int)):
            raise RouterRefused(f"stage {index} next rung is invalid")
        plan_path = _safe_repo_path(stage["plan"], f"stage {index} plan")
        if not plan_path.is_file():
            raise RouterRefused(f"stage {index} campaign plan is missing")
        stages.append(
            Stage(
                stage_id=stage_id,
                plan_path=plan_path,
                verifier_path=_safe_repo_path(stage["verifier"], f"stage {index} verifier"),
                source_path=_safe_repo_path(stage["source"], f"stage {index} source"),
                rung=rung,
                next_rung=next_rung,
            )
        )
    expected = ((10_000, 100_000), (100_000, 1_000_000), (1_000_000, None))
    if tuple((stage.rung, stage.next_rung) for stage in stages) != expected:
        raise RouterRefused("router rung progression drifted")
    if len({stage.stage_id for stage in stages}) != len(stages):
        raise RouterRefused("router stage ids are duplicated")
    return RouterPlan(
        path=source,
        sha256=sha256_file(source),
        router_id=router_id,
        router_root=router_root,
        poll_seconds=float(poll_seconds),
        campaign_entrypoint=campaign_entrypoint,
        stages=tuple(stages),
    )


def _declared_output(command: Sequence[str]) -> str | None:
    flags = {"--out", "--output", "--verification-out"}
    matches = [command[index + 1] for index, value in enumerate(command[:-1]) if value in flags]
    return matches[0] if len(matches) == 1 else None


def _prepared_tasks() -> tuple[dict[str, TaskDeclaration], set[str]]:
    policy = load_policy(DEFAULT_POLICY)
    tasks = dict(policy.tasks)
    overlay = load_task_overlay(DEFAULT_OVERLAY, repository_root=REPO_ROOT)
    overlay_ids: set[str] = set()
    for task_id, raw in overlay.tasks.items():
        tasks[task_id] = TaskDeclaration.from_mapping(task_id, dict(raw))
        overlay_ids.add(task_id)
    return tasks, overlay_ids


def _campaign_document(path: Path) -> dict[str, Any]:
    payload = _read_json_snapshot(path, "rung-bounded campaign plan", maximum_bytes=4 * 1024 * 1024)
    _exact_keys(
        payload,
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
        "rung-bounded campaign plan",
    )
    if payload["schema"] != PLAN_SCHEMA:
        raise RouterRefused(f"campaign plan schema must be {PLAN_SCHEMA}")
    if payload["policy_path"] != "configs/local_execution_throttle.yaml":
        raise RouterRefused("rung-bounded campaign policy path drifted")
    return payload


def validate_prepared_router_plan(plan: RouterPlan) -> dict[str, Any]:
    """Validate new plans against the live baseline plus the prepared additive overlay.

    This does not authorize execution.  It exists so the files can be reviewed before
    the final policy migration installs the overlay tasks into the live policy.
    """

    tasks, overlay_ids = _prepared_tasks()
    live_task_ids = set(load_policy(DEFAULT_POLICY).tasks)
    expected_sequences = {
        10_000: [
            "edcm1_official_cpu",
            "edcm1_verify_cpu",
            "escs_x0_official_cpu",
            "escs_x0_verify_cpu",
            "p6_10k_resource_probe_cpu",
            "p6_10k_replication_cpu",
            "p6_10k_verify_cpu",
        ],
        100_000: ["p6_100k_replication_cpu", "p6_100k_verify_cpu"],
        1_000_000: ["p6_1m_replication_cpu", "p6_1m_verify_cpu"],
    }
    problems: list[str] = []
    documents: list[dict[str, Any]] = []
    all_task_ids: list[str] = []
    for stage in plan.stages:
        try:
            document = _campaign_document(stage.plan_path)
        except RouterRefused as exc:
            problems.append(f"{stage.stage_id}: {exc}")
            continue
        documents.append(document)
        steps = document.get("steps")
        if not isinstance(steps, list) or not steps:
            problems.append(f"{stage.stage_id}: campaign steps are missing")
            continue
        task_ids = [str(step.get("task")) for step in steps if isinstance(step, dict)]
        all_task_ids.extend(task_ids)
        if task_ids != expected_sequences[stage.rung]:
            problems.append(f"{stage.stage_id}: task sequence drifted")
        seen: set[str] = set()
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                problems.append(f"{stage.stage_id}: step {index} is not an object")
                continue
            try:
                _exact_keys(
                    step,
                    {
                        "id",
                        "kind",
                        "depends_on",
                        "task",
                        "artifact",
                        "max_failures",
                        "max_no_progress_legs",
                    },
                    f"{stage.stage_id} step {index}",
                )
            except RouterRefused as exc:
                problems.append(str(exc))
                continue
            step_id = str(step["id"])
            dependencies = step["depends_on"]
            if not isinstance(dependencies, list) or any(value not in seen for value in dependencies):
                problems.append(f"{stage.stage_id}: step {step_id} has a nonprior dependency")
            seen.add(step_id)
            task_id = str(step["task"])
            task = tasks.get(task_id)
            artifact = step.get("artifact")
            if task is None:
                problems.append(f"{stage.stage_id}: task {task_id} is not prepared")
            if not isinstance(artifact, dict):
                problems.append(f"{stage.stage_id}: step {step_id} artifact is missing")
            elif task is not None and artifact.get("path") != _declared_output(task.command):
                problems.append(f"{stage.stage_id}: step {step_id} output/task command drift")
        final_artifact = steps[-1].get("artifact") if isinstance(steps[-1], dict) else None
        if not isinstance(final_artifact, dict) or final_artifact.get("path") != str(
            stage.verifier_path.relative_to(REPO_ROOT)
        ):
            problems.append(f"{stage.stage_id}: final verifier path drifted")
        elif final_artifact.get("schema") != P6_VERIFIER_SCHEMA:
            problems.append(f"{stage.stage_id}: final verifier schema drifted")
        else:
            fields = final_artifact.get("fields")
            if not isinstance(fields, dict):
                problems.append(f"{stage.stage_id}: terminal verifier fields are missing")
            else:
                for forbidden in (
                    "independent_recompute.decision.verdict",
                    "independent_recompute.decision.null_supported",
                    "independent_recompute.decision.strict_joint_gain_all_schedules_and_controls",
                ):
                    if forbidden in fields:
                        problems.append(
                            f"{stage.stage_id}: terminal campaign must not preselect "
                            f"outcome field {forbidden}"
                        )
                if fields.get("verification_complete") is not True or fields.get("errors") != []:
                    problems.append(f"{stage.stage_id}: terminal verifier validity fields drifted")
    duplicate_campaign_ids = [document.get("campaign_id") for document in documents]
    if len(duplicate_campaign_ids) != len(set(duplicate_campaign_ids)):
        problems.append("rung-bounded campaign ids are duplicated")
    live_missing = sorted(set(all_task_ids) - live_task_ids)
    expected_overlay_only = sorted(set(all_task_ids) & overlay_ids)
    return {
        "schema": "mop-p6-null-safe-prepared-validation/v1",
        "valid": not problems,
        "router_plan": {"path": str(plan.path), "sha256": plan.sha256},
        "stage_count": len(plan.stages),
        "campaign_ids": duplicate_campaign_ids,
        "prepared_task_count": len(set(all_task_ids)),
        "live_missing_tasks": live_missing,
        "expected_overlay_tasks": expected_overlay_only,
        "execution_authorized": False,
        "problems": problems,
    }


def validate_live_router_plan(plan: RouterPlan) -> dict[str, Any]:
    campaigns = []
    for stage in plan.stages:
        campaign = load_campaign_plan(stage.plan_path)
        campaigns.append(
            {
                "stage_id": stage.stage_id,
                "campaign_id": campaign.campaign_id,
                "plan_sha256": campaign.sha256,
                "policy_sha256": campaign.policy.sha256,
                "steps": [step.step_id for step in campaign.steps],
            }
        )
    return {
        "schema": "mop-p6-null-safe-live-validation/v1",
        "valid": True,
        "router_plan": {"path": str(plan.path), "sha256": plan.sha256},
        "campaigns": campaigns,
        "execution_authorized": False,
    }


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise RouterRefused(f"{label} is not finite numeric evidence")
    return float(value)


def _validate_source(stage: Stage, source: Mapping[str, Any]) -> None:
    if source.get("schema") != P6_RUNG_SCHEMA:
        raise RouterRefused("source rung schema mismatch")
    if source.get("mode") != "replication" or source.get("rung") != stage.rung:
        raise RouterRefused("source rung identity mismatch")
    if (
        source.get("all_mechanics_ok") is not True
        or source.get("replication_execution_complete") is not True
        or source.get("scientific_promotion") is not False
    ):
        raise RouterRefused("source rung is not a complete nonpromoting replication")
    core = dict(source)
    declared = core.pop("payload_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise RouterRefused("source rung payload self-hash mismatch")


def _validate_decision(
    decision: object,
    *,
    source: Mapping[str, Any],
) -> str:
    if not isinstance(decision, dict):
        raise RouterRefused("P6 verifier decision is missing")
    if decision.get("controls") != list(EXPECTED_CONTROLS):
        raise RouterRefused("P6 verifier controls drifted")
    if (
        decision.get("primary_endpoints")
        != [
            "retention.domain_zero_final_accuracy",
            "future_learnability.first_window_accuracy",
        ]
        or decision.get("independent_unit") != "seed within transition schedule"
        or decision.get("tie_rule") != TIE_RULE
    ):
        raise RouterRefused("P6 verifier decision contract drifted")
    if decision.get("scientific_promotion") is not False:
        raise RouterRefused("P6 verifier decision escaped promotion boundary")
    plan = source.get("plan")
    seeds = plan.get("seeds") if isinstance(plan, dict) else None
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise RouterRefused("source rung seed authority is invalid")
    contrasts = decision.get("contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != 4:
        raise RouterRefused("P6 verifier contrast matrix is incomplete")
    expected_coordinates = {
        (schedule, control) for schedule in EXPECTED_SCHEDULES for control in EXPECTED_CONTROLS
    }
    observed_coordinates: set[tuple[object, object]] = set()
    recomputed_strict: list[bool] = []
    recomputed_nulls: list[bool] = []
    for row in contrasts:
        if not isinstance(row, dict):
            raise RouterRefused("P6 verifier contrast row is invalid")
        observed_coordinates.add((row.get("schedule"), row.get("control")))
        pairs = row.get("paired_seed_deltas")
        if not isinstance(pairs, list) or len(pairs) != len(seeds):
            raise RouterRefused("P6 verifier paired seed matrix is incomplete")
        nonpositive = False
        for expected_seed, pair in zip(seeds, pairs, strict=True):
            if not isinstance(pair, dict) or pair.get("seed") != expected_seed:
                raise RouterRefused("P6 verifier paired seed order drifted")
            retention = _finite_number(pair.get("retention_delta"), "retention delta")
            future = _finite_number(pair.get("future_first_window_delta"), "future delta")
            tie = retention == 0.0 or future == 0.0
            pair_nonpositive = retention <= 0.0 or future <= 0.0
            if pair.get("tie_is_null") is not tie or pair.get("nonpositive_is_null") is not pair_nonpositive:
                raise RouterRefused("P6 verifier paired tie/null semantics drifted")
            nonpositive = nonpositive or pair_nonpositive
        retention_mean = _finite_number(row.get("retention_mean_delta"), "retention mean")
        future_mean = _finite_number(row.get("future_first_window_mean_delta"), "future mean")
        recomputed_retention_mean = sum(
            float(pair["retention_delta"]) for pair in pairs if isinstance(pair, dict)
        ) / len(pairs)
        recomputed_future_mean = sum(
            float(pair["future_first_window_delta"]) for pair in pairs if isinstance(pair, dict)
        ) / len(pairs)
        if not math.isclose(
            retention_mean,
            recomputed_retention_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            future_mean,
            recomputed_future_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RouterRefused("P6 verifier contrast means drifted from paired seeds")
        aggregate_tie = retention_mean == 0.0 or future_mean == 0.0
        null_contrast = aggregate_tie or nonpositive
        strict = retention_mean > 0.0 and future_mean > 0.0 and not null_contrast
        if (
            row.get("independent_units") != len(seeds)
            or row.get("aggregate_tie_is_null") is not aggregate_tie
            or row.get("any_seed_nonpositive_is_null") is not nonpositive
            or row.get("null_contrast") is not null_contrast
            or row.get("strict_joint_gain") is not strict
        ):
            raise RouterRefused("P6 verifier contrast decision semantics drifted")
        recomputed_strict.append(strict)
        recomputed_nulls.append(null_contrast)
    if observed_coordinates != expected_coordinates:
        raise RouterRefused("P6 verifier contrast coordinates drifted")
    favorable = all(recomputed_strict)
    if decision.get("aggregate_tie_count") != sum(int(value) for value in recomputed_nulls):
        raise RouterRefused("P6 verifier null count drifted")
    if decision.get("strict_joint_gain_all_schedules_and_controls") is not favorable:
        raise RouterRefused("P6 verifier aggregate decision drifted")
    expected_verdict = "favorable-rung-pattern" if favorable else "null"
    if decision.get("verdict") != expected_verdict or decision.get("null_supported") is not (not favorable):
        raise RouterRefused("P6 verifier verdict/null semantics drifted")
    return "favorable" if favorable else "null"


def validate_terminal_verifier(stage: Stage, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    verifier_path = (root / stage.verifier_path.relative_to(REPO_ROOT)).resolve()
    source_path = (root / stage.source_path.relative_to(REPO_ROOT)).resolve()
    verifier = _read_json_snapshot(verifier_path, f"{stage.stage_id} verifier")
    source = _read_json_snapshot(source_path, f"{stage.stage_id} source rung")
    _validate_source(stage, source)
    if verifier.get("schema") != P6_VERIFIER_SCHEMA:
        raise RouterRefused("P6 verifier schema mismatch")
    core = dict(verifier)
    declared = core.pop("payload_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise RouterRefused("P6 verifier payload self-hash mismatch")
    if verifier.get("verification_complete") is not True or verifier.get("errors") != []:
        raise RouterRefused("P6 verifier is not valid and terminal")
    if verifier.get("scientific_promotion") is not False:
        raise RouterRefused("P6 verifier escaped promotion boundary")
    checks = verifier.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != EXPECTED_CHECK_NAMES
        or not all(value is True for value in checks.values())
    ):
        raise RouterRefused("P6 verifier canonical checks are incomplete")
    source_binding = verifier.get("source_rung")
    expected_source_label = str(stage.source_path.relative_to(REPO_ROOT))
    if not isinstance(source_binding, dict) or (
        source_binding.get("path") != expected_source_label
        or source_binding.get("file_sha256") != sha256_file(source_path)
        or source_binding.get("payload_sha256") != source.get("payload_sha256")
        or source_binding.get("identity_sha256") != source.get("identity_sha256")
        or source_binding.get("rung") != stage.rung
        or source_binding.get("mode") != "replication"
    ):
        raise RouterRefused("P6 verifier/source rung binding drifted")
    recompute = verifier.get("independent_recompute")
    if not isinstance(recompute, dict) or (
        recompute.get("cell_count") != 30
        or recompute.get("metric_families") != list(EXPECTED_METRIC_FAMILIES)
        or recompute.get("checkpoint_state_recomputed") is not True
        or recompute.get("controls_recomputed") is not True
        or recompute.get("paired_metrics_recomputed") is not True
    ):
        raise RouterRefused("P6 verifier recompute envelope drifted")
    outcome = _validate_decision(recompute.get("decision"), source=source)
    mutation = verifier.get("mutation_suite")
    rows = mutation.get("mutations") if isinstance(mutation, dict) else None
    if not isinstance(mutation, dict) or (
        mutation.get("count") != 12
        or mutation.get("rejected") != 12
        or mutation.get("all_rejected") is not True
        or not isinstance(rows, list)
        or len(rows) != 12
        or any(
            not isinstance(row, dict)
            or row.get("rejected") is not True
            or not isinstance(row.get("problems"), list)
            or not row["problems"]
            for row in rows
        )
        or len(
            {
                row.get("mutation")
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("mutation"), str)
            }
        )
        != 12
    ):
        raise RouterRefused("P6 verifier mutation suite drifted")
    favorable = outcome == "favorable"
    next_allowed = favorable and stage.next_rung is not None
    prerequisite = verifier.get("prerequisite")
    expected_prerequisite = {
        "source_rung": stage.rung,
        "source_rung_file_sha256": sha256_file(source_path),
        "source_identity_sha256": source.get("identity_sha256"),
        "verification_complete": True,
        "valid_controls": True,
        "tie_is_null": True,
        "mutation_suite_all_rejected": True,
        "next_rung": stage.next_rung,
        "next_rung_allowed": next_allowed,
        "next_rung_reason": FAVORABLE_REASON if next_allowed else NULL_REASON,
    }
    if prerequisite != expected_prerequisite:
        raise RouterRefused("P6 verifier next-rung authority drifted")
    implementation = verifier.get("implementation")
    if not isinstance(implementation, list) or not implementation:
        raise RouterRefused("P6 verifier implementation receipts are missing")
    if [row.get("path") for row in implementation if isinstance(row, dict)] != list(IMPLEMENTATION_PATHS):
        raise RouterRefused("P6 verifier implementation path set drifted")
    seen_paths: set[str] = set()
    for row in implementation:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RouterRefused("P6 verifier implementation receipt is invalid")
        implementation_path = _safe_repo_path(row["path"], "verifier implementation path", root=root)
        if row["path"] in seen_paths or not implementation_path.is_file():
            raise RouterRefused("P6 verifier implementation path is missing or duplicated")
        seen_paths.add(str(row["path"]))
        if row["sha256"] != sha256_file(implementation_path):
            raise RouterRefused("P6 verifier implementation receipt drifted")
    return {
        "outcome": outcome,
        "rung": stage.rung,
        "next_rung": stage.next_rung,
        "next_rung_allowed": next_allowed,
        "verifier_file_sha256": sha256_file(verifier_path),
        "verifier_payload_sha256": verifier["payload_sha256"],
        "source_file_sha256": sha256_file(source_path),
        "source_payload_sha256": source["payload_sha256"],
        "scientific_promotion": False,
    }


def _campaign_out_dir(stage: Stage) -> Path:
    document = _campaign_document(stage.plan_path)
    campaign_root = _safe_repo_path(document["campaign_root"], "campaign root")
    return campaign_root / str(document["campaign_id"])


def _stage_campaign_join(stage: Stage) -> dict[str, Any]:
    out_dir = _campaign_out_dir(stage)
    status = read_campaign_status(out_dir)
    if status.get("state") != "complete":
        raise RouterRefused(f"{stage.stage_id} campaign is not complete")
    plan_sha = sha256_file(stage.plan_path)
    if status.get("plan") != {"path": str(stage.plan_path), "sha256": plan_sha}:
        raise RouterRefused(f"{stage.stage_id} campaign plan binding drifted")
    steps = status.get("steps")
    if not isinstance(steps, dict) or not steps:
        raise RouterRefused(f"{stage.stage_id} campaign step state is missing")
    campaign_document = _campaign_document(stage.plan_path)
    declared_steps = campaign_document.get("steps")
    if not isinstance(declared_steps, list) or not declared_steps:
        raise RouterRefused(f"{stage.stage_id} campaign plan has no terminal step")
    final_step_id = declared_steps[-1].get("id") if isinstance(declared_steps[-1], dict) else None
    final = steps.get(final_step_id) if isinstance(final_step_id, str) else None
    verifier_sha = sha256_file(stage.verifier_path)
    if not isinstance(final, dict) or (
        final.get("status") != "complete"
        or final.get("artifact_sha256") != verifier_sha
        or not isinstance(final.get("governor_receipt_path"), str)
    ):
        raise RouterRefused(f"{stage.stage_id} final campaign/verifier join drifted")
    governor_receipt = Path(str(final["governor_receipt_path"])).resolve()
    if not governor_receipt.is_file() or not governor_receipt.is_relative_to(REPO_ROOT):
        raise RouterRefused(f"{stage.stage_id} governor completion receipt is missing")
    return {
        "path": str(out_dir / STATUS_FILE),
        "sha256": sha256_file(out_dir / STATUS_FILE),
        "status_sha256": status["status_sha256"],
        "governor_receipt": str(governor_receipt.relative_to(REPO_ROOT)),
        "governor_receipt_sha256": sha256_file(governor_receipt),
    }


def build_route_receipt(stage: Stage) -> dict[str, Any]:
    result = validate_terminal_verifier(stage)
    campaign = _stage_campaign_join(stage)
    core = {
        "schema": ROUTE_RECEIPT_SCHEMA,
        "created_at": _now(),
        "stage_id": stage.stage_id,
        "campaign_plan": {
            "path": str(stage.plan_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(stage.plan_path),
        },
        "campaign_status": campaign,
        "verifier": {
            "path": str(stage.verifier_path.relative_to(REPO_ROOT)),
            "file_sha256": result["verifier_file_sha256"],
            "payload_sha256": result["verifier_payload_sha256"],
        },
        "source": {
            "path": str(stage.source_path.relative_to(REPO_ROOT)),
            "file_sha256": result["source_file_sha256"],
            "payload_sha256": result["source_payload_sha256"],
        },
        "rung": stage.rung,
        "outcome": result["outcome"],
        "next_rung": stage.next_rung,
        "next_rung_allowed": result["next_rung_allowed"],
        "scientific_promotion": False,
    }
    return {**core, "route_sha256": canonical_sha256(core)}


def _route_path(plan: RouterPlan, stage: Stage) -> Path:
    return plan.out_dir / ROUTES_DIR / f"{stage.stage_id}.json"


def _publish_route(plan: RouterPlan, stage: Stage) -> dict[str, Any]:
    path = _route_path(plan, stage)
    expected = build_route_receipt(stage)
    if path.exists():
        existing = _read_json_snapshot(path, f"{stage.stage_id} route receipt", maximum_bytes=1024 * 1024)
        _validate_seal(existing, "route_sha256", "route receipt")
        for field in (
            "stage_id",
            "campaign_plan",
            "campaign_status",
            "verifier",
            "source",
            "rung",
            "outcome",
            "next_rung",
            "next_rung_allowed",
            "scientific_promotion",
        ):
            if existing.get(field) != expected.get(field):
                raise RouterRefused(f"existing {stage.stage_id} route receipt drifted at {field}")
        return existing
    _atomic_json(path, expected)
    return expected


class RouterLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> RouterLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise RouterRefused("null-safe router is already running") from exc
        return self

    def __exit__(self, *_: object) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _router_identity() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    return {"pid": process.pid, "create_time": process.create_time()}


def _initial_state(plan: RouterPlan, *, execute: bool) -> dict[str, Any]:
    return {
        "schema": ROUTER_STATE_SCHEMA,
        "router_id": plan.router_id,
        "router_plan": {"path": str(plan.path), "sha256": plan.sha256},
        "supervisor": _router_identity(),
        "execution_enabled": execute,
        "status": "starting",
        "current_stage": None,
        "stage_index": 0,
        "stage_results": [],
        "started_at": _now(),
        "updated_at": _now(),
        "finished_at": None,
        "problems": [],
    }


def _state_path(plan: RouterPlan) -> Path:
    return plan.out_dir / ROUTER_STATE_FILE


def _status_path(plan: RouterPlan) -> Path:
    return plan.out_dir / ROUTER_STATUS_FILE


def _load_state(plan: RouterPlan, *, execute: bool) -> dict[str, Any]:
    path = _state_path(plan)
    if not path.is_file():
        return _initial_state(plan, execute=execute)
    state = _read_json_snapshot(path, "router state", maximum_bytes=4 * 1024 * 1024)
    _validate_seal(state, "state_sha256", "router state")
    if state.get("schema") != ROUTER_STATE_SCHEMA or state.get("router_id") != plan.router_id:
        raise RouterRefused("router state identity drifted")
    if state.get("router_plan") != {"path": str(plan.path), "sha256": plan.sha256}:
        raise RouterRefused("router plan drift requires a new router id")
    state["supervisor"] = _router_identity()
    state["execution_enabled"] = execute
    return state


def _publish_state(plan: RouterPlan, state: dict[str, Any]) -> dict[str, Any]:
    state["updated_at"] = _now()
    sealed_state = _sealed(state, "state_sha256")
    _atomic_json(_state_path(plan), sealed_state)
    status_core = {
        "schema": ROUTER_STATUS_SCHEMA,
        "router_id": plan.router_id,
        "created_at": _now(),
        "router_plan": {"path": str(plan.path), "sha256": plan.sha256},
        "supervisor": sealed_state["supervisor"],
        "execution_enabled": sealed_state["execution_enabled"],
        "state": sealed_state["status"],
        "current_stage": sealed_state["current_stage"],
        "stage_index": sealed_state["stage_index"],
        "stage_results": sealed_state["stage_results"],
        "started_at": sealed_state["started_at"],
        "updated_at": sealed_state["updated_at"],
        "finished_at": sealed_state["finished_at"],
        "problems": sealed_state["problems"],
    }
    status = {**status_core, "status_sha256": canonical_sha256(status_core)}
    _atomic_json(_status_path(plan), status)
    return status


def read_router_status(plan: RouterPlan) -> dict[str, Any]:
    status = _read_json_snapshot(_status_path(plan), "router status", maximum_bytes=4 * 1024 * 1024)
    if status.get("schema") != ROUTER_STATUS_SCHEMA or status.get("router_id") != plan.router_id:
        raise RouterRefused("router status identity drifted")
    _validate_seal(status, "status_sha256", "router status")
    return status


def _process_alive(identity: object) -> bool:
    if not isinstance(identity, Mapping):
        return False
    pid = identity.get("pid")
    created = identity.get("create_time")
    if isinstance(pid, bool) or not isinstance(pid, int) or not isinstance(created, int | float):
        return False
    try:
        process = psutil.Process(pid)
        return math.isclose(process.create_time(), float(created), rel_tol=0.0, abs_tol=0.01)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return False


def _campaign_process_alive(status: Mapping[str, Any]) -> bool:
    return _process_alive(status.get("supervisor"))


def _ensure_campaign_started(stage: Stage) -> None:
    plan = load_campaign_plan(stage.plan_path)
    out_dir = plan.out_dir
    if (out_dir / STATUS_FILE).is_file():
        status = read_campaign_status(out_dir)
        if status.get("state") in TERMINAL_STATES or _campaign_process_alive(status):
            return
    start_detached(plan, out_dir=out_dir, execute=True, use_caffeinate=False)


def _stop_requested(plan: RouterPlan) -> bool:
    return (plan.out_dir / ROUTER_CONTROL_FILE).is_file()


def _wait_for_campaign(plan: RouterPlan, stage: Stage, state: dict[str, Any]) -> str:
    out_dir = _campaign_out_dir(stage)
    while True:
        if _stop_requested(plan):
            request_drain(out_dir, "null-safe router stop requested")
        try:
            status = read_campaign_status(out_dir)
        except (OSError, ValueError, json.JSONDecodeError):
            time.sleep(min(plan.poll_seconds, 30.0))
            continue
        campaign_state = str(status.get("state"))
        state["status"] = "draining" if _stop_requested(plan) else "waiting_campaign"
        _publish_state(plan, state)
        if campaign_state in TERMINAL_STATES:
            return campaign_state
        time.sleep(plan.poll_seconds)


def run_router(plan: RouterPlan, *, execute: bool) -> dict[str, Any]:
    if not execute:
        raise RouterRefused("router execution requires explicit --execute")
    validate_live_router_plan(plan)
    with RouterLock(plan.out_dir / ROUTER_LOCK_FILE):
        state = _load_state(plan, execute=True)
        if state.get("status") in {"complete", "complete_null_stop", "drained"}:
            return _publish_state(plan, state)
        try:
            for index, stage in enumerate(plan.stages):
                prior = [
                    row
                    for row in state.get("stage_results", [])
                    if isinstance(row, dict) and row.get("stage_id") == stage.stage_id
                ]
                if prior:
                    if prior[-1].get("outcome") == "null":
                        state["status"] = "complete_null_stop"
                        state["finished_at"] = state.get("finished_at") or _now()
                        return _publish_state(plan, state)
                    continue
                if _stop_requested(plan):
                    state["status"] = "drained"
                    state["finished_at"] = _now()
                    return _publish_state(plan, state)
                state["stage_index"] = index
                state["current_stage"] = stage.stage_id
                state["status"] = "starting_campaign"
                _publish_state(plan, state)
                _ensure_campaign_started(stage)
                campaign_state = _wait_for_campaign(plan, stage, state)
                if campaign_state == "drained" or _stop_requested(plan):
                    state["status"] = "drained"
                    state["finished_at"] = _now()
                    return _publish_state(plan, state)
                if campaign_state != "complete":
                    raise RouterRefused(
                        f"stage {stage.stage_id} entered terminal campaign state {campaign_state}"
                    )
                route = _publish_route(plan, stage)
                state.setdefault("stage_results", []).append(
                    {
                        "stage_id": stage.stage_id,
                        "route_path": str(_route_path(plan, stage).relative_to(REPO_ROOT)),
                        "route_sha256": route["route_sha256"],
                        "rung": stage.rung,
                        "outcome": route["outcome"],
                        "next_rung": route["next_rung"],
                        "next_rung_allowed": route["next_rung_allowed"],
                    }
                )
                if route["outcome"] == "null":
                    state["status"] = "complete_null_stop"
                    state["current_stage"] = None
                    state["finished_at"] = _now()
                    return _publish_state(plan, state)
                if stage.next_rung is not None and route["next_rung_allowed"] is not True:
                    raise RouterRefused("favorable verifier did not authorize its exact next rung")
            state["status"] = "complete"
            state["current_stage"] = None
            state["stage_index"] = len(plan.stages)
            state["finished_at"] = _now()
            return _publish_state(plan, state)
        except Exception as exc:
            state["status"] = "failure_hold"
            problem = f"{type(exc).__name__}: {exc}"
            if problem not in state.setdefault("problems", []):
                state["problems"].append(problem)
            state["finished_at"] = _now()
            _publish_state(plan, state)
            raise


def request_router_stop(plan: RouterPlan, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise RouterRefused("stop reason must not be empty")
    core = {
        "schema": CONTROL_SCHEMA,
        "router_id": plan.router_id,
        "created_at": _now(),
        "action": "drain",
        "reason": reason,
    }
    payload = {**core, "control_sha256": canonical_sha256(core)}
    _atomic_json(plan.out_dir / ROUTER_CONTROL_FILE, payload)
    return payload


def start_router_detached(
    plan: RouterPlan,
    *,
    execute: bool,
    use_caffeinate: bool,
    acknowledgement_seconds: float = 15.0,
) -> dict[str, Any]:
    if not execute:
        raise RouterRefused("detached start requires explicit --execute")
    validate_live_router_plan(plan)
    plan.out_dir.mkdir(parents=True, exist_ok=True)
    with RouterLock(plan.out_dir / ROUTER_START_LOCK_FILE):
        if _status_path(plan).is_file():
            status = read_router_status(plan)
            if status.get("state") in {"complete", "complete_null_stop", "drained"}:
                return {"already_terminal": True, "status": status}
            if _process_alive(status.get("supervisor")):
                return {"already_running": True, "status": status}
        wrapper = REPO_ROOT / "scripts/mop_null_safe_campaign.py"
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(wrapper),
            "run",
            "--config",
            str(plan.path),
            "--execute",
        ]
        caffeinate = shutil.which("caffeinate") if use_caffeinate else None
        launched_command = [caffeinate, "-ims", *command] if caffeinate else command
        stdout_path = plan.out_dir / "router.stdout.log"
        stderr_path = plan.out_dir / "router.stderr.log"
        before_mtime = _status_path(plan).stat().st_mtime_ns if _status_path(plan).exists() else 0
        environment = dict(os.environ)
        python_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["PYTHONUNBUFFERED"] = "1"
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
            if _status_path(plan).is_file() and _status_path(plan).stat().st_mtime_ns > before_mtime:
                try:
                    acknowledged = read_router_status(plan)
                    break
                except (OSError, RouterRefused, json.JSONDecodeError):
                    pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
        if acknowledged is None:
            raise RouterRefused(f"detached router did not acknowledge; inspect {stderr_path}")
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
    validate = subparsers.add_parser("validate", help="validate without launching anything")
    validate.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    validate.add_argument(
        "--live",
        action="store_true",
        help="require the final migrated live policy instead of the prepared overlay",
    )
    run = subparsers.add_parser("run", help="run the foreground router worker")
    run.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    run.add_argument("--execute", action="store_true")
    start = subparsers.add_parser("start", help="start the detached router")
    start.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    start.add_argument("--execute", action="store_true")
    start.add_argument("--no-caffeinate", action="store_true")
    status = subparsers.add_parser("status", help="read the current sealed router status")
    status.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    wait = subparsers.add_parser("wait", help="wait for a terminal router state")
    wait.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    wait.add_argument("--poll-seconds", type=float, default=30.0)
    wait.add_argument("--timeout-seconds", type=float, default=0.0)
    stop = subparsers.add_parser("stop", help="request a non-signaling drain")
    stop.add_argument("--config", type=Path, default=DEFAULT_ROUTER_PLAN)
    stop.add_argument("--reason", default="operator requested null-safe campaign drain")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    plan = load_router_plan(arguments.config)
    if arguments.command == "validate":
        result = validate_live_router_plan(plan) if arguments.live else validate_prepared_router_plan(plan)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("valid") is True else 2
    if arguments.command == "run":
        try:
            status = run_router(plan, execute=arguments.execute)
        except (RouterRefused, OSError, ValueError) as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
            return 2
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status.get("state") in {"complete", "complete_null_stop", "drained"} else 2
    if arguments.command == "start":
        try:
            result = start_router_detached(
                plan,
                execute=arguments.execute,
                use_caffeinate=not arguments.no_caffeinate,
            )
        except (RouterRefused, OSError, ValueError) as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "status":
        print(json.dumps(read_router_status(plan), indent=2, sort_keys=True))
        return 0
    if arguments.command == "stop":
        print(json.dumps(request_router_stop(plan, arguments.reason), indent=2, sort_keys=True))
        return 0
    if arguments.command == "wait":
        started = time.monotonic()
        while True:
            status = read_router_status(plan)
            if status.get("state") in {"complete", "complete_null_stop", "drained", "failure_hold"}:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 0 if status.get("state") != "failure_hold" else 2
            if arguments.timeout_seconds > 0 and time.monotonic() - started >= arguments.timeout_seconds:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 2
            time.sleep(max(0.1, arguments.poll_seconds))
    raise AssertionError(arguments.command)


__all__ = [
    "DEFAULT_ROUTER_PLAN",
    "ROUTE_RECEIPT_SCHEMA",
    "ROUTER_PLAN_SCHEMA",
    "RouterPlan",
    "RouterRefused",
    "Stage",
    "build_parser",
    "build_route_receipt",
    "canonical_sha256",
    "load_router_plan",
    "main",
    "read_router_status",
    "request_router_stop",
    "run_router",
    "start_router_detached",
    "validate_live_router_plan",
    "validate_prepared_router_plan",
    "validate_terminal_verifier",
]
