"""Fail-closed preparation of additive local-throttle task migrations.

This module renders a candidate policy in memory.  It never writes the live policy;
the controlled campaign transition remains the only authority for adopting the
rendered bytes after active legacy work and its verifier are terminal.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .profiles import get_profile
from .task_policy_authority import (
    build_policy_safety_contract,
    canonical_sha256,
    policy_baseline_manifest_problems,
)

TASK_OVERLAY_SCHEMA = "mop-local-throttle-task-overlay/v1"
POLICY_MIGRATION_PREVIEW_SCHEMA = "mop-local-throttle-policy-migration-preview/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _relative_path(value: object, label: str) -> Path:
    path = Path(str(value))
    if not str(value) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return path


@dataclass(frozen=True, slots=True)
class TaskOverlay:
    path: Path
    sha256: str
    baseline_policy_sha256: str
    baseline_manifest_path: Path
    baseline_manifest_sha256: str
    add_known_heavy_markers: tuple[str, ...]
    execution_order: Mapping[str, tuple[str, ...]]
    tasks: Mapping[str, Mapping[str, Any]]


def load_task_overlay(path: str | Path, *, repository_root: str | Path) -> TaskOverlay:
    source = Path(path).resolve()
    raw_bytes = source.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    _exact_keys(
        payload,
        {
            "schema",
            "baseline_policy_sha256",
            "baseline_manifest",
            "add_known_heavy_markers",
            "execution_order",
            "tasks",
        },
        "task overlay",
    )
    if payload["schema"] != TASK_OVERLAY_SCHEMA:
        raise ValueError(f"task overlay schema must be {TASK_OVERLAY_SCHEMA}")
    baseline_sha = str(payload["baseline_policy_sha256"])
    if _SHA256_RE.fullmatch(baseline_sha) is None:
        raise ValueError("overlay baseline policy SHA-256 is invalid")
    root = Path(repository_root).resolve()
    manifest_relative = _relative_path(payload["baseline_manifest"], "baseline manifest")
    manifest_path = (root / manifest_relative).resolve()
    if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
        raise ValueError("baseline manifest is missing or escapes the repository")
    manifest = json.loads(manifest_path.read_text())
    problems = policy_baseline_manifest_problems(manifest)
    if problems:
        raise ValueError(f"baseline manifest invalid: {problems}")
    if manifest["policy"]["sha256"] != baseline_sha:
        raise ValueError("overlay and baseline manifest policy digests differ")
    markers_raw = payload["add_known_heavy_markers"]
    if not isinstance(markers_raw, list) or any(
        not isinstance(value, str) or not value.strip() for value in markers_raw
    ):
        raise ValueError("overlay markers must be a nonempty-string list")
    markers = tuple(markers_raw)
    if len(markers) != len(set(markers)):
        raise ValueError("overlay markers must be unique")
    orders_raw = payload["execution_order"]
    tasks_raw = payload["tasks"]
    if not isinstance(orders_raw, dict) or not orders_raw:
        raise ValueError("overlay execution_order must be a nonempty mapping")
    if not isinstance(tasks_raw, dict) or not tasks_raw:
        raise ValueError("overlay tasks must be a nonempty mapping")
    orders: dict[str, tuple[str, ...]] = {}
    for order_id, task_ids in orders_raw.items():
        if not isinstance(order_id, str) or not order_id.strip() or not isinstance(task_ids, list):
            raise ValueError("overlay execution order is invalid")
        rows = tuple(str(value) for value in task_ids)
        if not rows or len(rows) != len(set(rows)):
            raise ValueError("overlay execution order must contain unique task ids")
        orders[order_id] = rows
    ordered_tasks = {task_id for rows in orders.values() for task_id in rows}
    if not set(tasks_raw) <= ordered_tasks:
        raise ValueError("every overlay task must occur in an overlay execution order")
    for task_id, task in tasks_raw.items():
        if not isinstance(task_id, str) or not task_id.strip() or not isinstance(task, dict):
            raise ValueError("overlay task declaration is invalid")
        if task.get("command") is None or task.get("checkpoint_globs") is None:
            raise ValueError(f"overlay task {task_id} lacks command/checkpoint authority")
    return TaskOverlay(
        path=source,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        baseline_policy_sha256=baseline_sha,
        baseline_manifest_path=manifest_path,
        baseline_manifest_sha256=str(manifest["manifest_sha256"]),
        add_known_heavy_markers=markers,
        execution_order=orders,
        tasks={task_id: copy.deepcopy(task) for task_id, task in tasks_raw.items()},
    )


def render_policy_overlay(
    policy_path: str | Path,
    overlay: TaskOverlay,
) -> tuple[str, dict[str, Any]]:
    source = Path(policy_path).resolve()
    old_bytes = source.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    if old_sha != overlay.baseline_policy_sha256:
        raise ValueError("live policy is not the overlay's reviewed baseline")
    raw = yaml.safe_load(old_bytes)
    if not isinstance(raw, dict):
        raise ValueError("live policy must be a mapping")
    before = copy.deepcopy(raw)
    monitor = raw.get("monitor")
    tasks = raw.get("tasks")
    execution_order = raw.get("execution_order")
    if not isinstance(monitor, dict) or not isinstance(tasks, dict) or not isinstance(execution_order, dict):
        raise ValueError("live policy monitor/tasks/execution_order are invalid")
    collisions = set(tasks) & set(overlay.tasks)
    order_collisions = set(execution_order) & set(overlay.execution_order)
    if collisions or order_collisions:
        raise ValueError(
            f"overlay is not additive; task collisions={sorted(collisions)}, "
            f"order collisions={sorted(order_collisions)}"
        )
    missing_order_tasks = {
        task_id
        for rows in overlay.execution_order.values()
        for task_id in rows
        if task_id not in tasks and task_id not in overlay.tasks
    }
    if missing_order_tasks:
        raise ValueError(f"overlay execution order names missing tasks {sorted(missing_order_tasks)}")
    known = monitor.get("known_heavy_markers")
    if not isinstance(known, list):
        raise ValueError("live known-heavy markers are invalid")
    for marker in overlay.add_known_heavy_markers:
        if marker not in known:
            known.append(marker)
    for task_id, task in overlay.tasks.items():
        tasks[task_id] = copy.deepcopy(task)
    for order_id, rows in overlay.execution_order.items():
        execution_order[order_id] = list(rows)
    rendered = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    def safety(document: Mapping[str, Any]) -> dict[str, Any]:
        return build_policy_safety_contract(
            profile=get_profile(str(document["profile"])).as_dict(),
            limits=document["limits"],
            monitor=document["monitor"],
            thresholds=document["thresholds"],
        )

    old_safety = safety(before)
    new_safety = safety(raw)
    if old_safety != new_safety:
        raise ValueError("overlay changed the non-marker safety contract")
    core = {
        "schema": POLICY_MIGRATION_PREVIEW_SCHEMA,
        "old_policy": {"path": str(source), "sha256": old_sha},
        "new_policy_sha256": rendered_sha,
        "overlay": {"path": str(overlay.path), "sha256": overlay.sha256},
        "baseline_manifest": {
            "path": str(overlay.baseline_manifest_path),
            "manifest_sha256": overlay.baseline_manifest_sha256,
        },
        "added_known_heavy_markers": list(overlay.add_known_heavy_markers),
        "added_task_ids": sorted(overlay.tasks),
        "added_execution_order_ids": sorted(overlay.execution_order),
        "safety_contract_sha256": canonical_sha256(old_safety),
        "scientific_promotion": False,
    }
    return rendered, {**core, "preview_sha256": canonical_sha256(core)}


__all__ = [
    "POLICY_MIGRATION_PREVIEW_SCHEMA",
    "TASK_OVERLAY_SCHEMA",
    "TaskOverlay",
    "load_task_overlay",
    "render_policy_overlay",
]
