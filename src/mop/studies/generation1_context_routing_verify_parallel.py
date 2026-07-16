"""Adaptive parallel execution for the independent Generation 1 C2 verifier.

The scientific checks remain implemented by the separately authored serial
verifier.  This module only distributes receipt reads, dataset reproduction,
and metric rebuilding, then restores canonical coordinate order before the
aggregate, canary, and mutation checks run in the parent process.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies import generation1_context_routing_verify as independent
from mop.studies.generation1_context_routing import (
    CLAIM_SCOPE,
    _queue_worker_target,
    atomic_write_json,
    canonical_sha256,
    load_config,
)

Coordinate = tuple[int, int]
CellTask = tuple[int, int, str, str | None]
CellResult = tuple[int, int, str, list[str], dict[str, Any] | None]

_WORKER_CONFIG: Mapping[str, Any] | None = None
_WORKER_CONFIG_SHA256: str | None = None
_WORKER_REPO_ROOT: Path | None = None


def _initialize_worker(
    config: Mapping[str, Any],
    config_sha256: str,
    repo_root: str,
) -> None:
    global _WORKER_CONFIG, _WORKER_CONFIG_SHA256, _WORKER_REPO_ROOT
    _WORKER_CONFIG = config
    _WORKER_CONFIG_SHA256 = config_sha256
    _WORKER_REPO_ROOT = Path(repo_root).resolve()
    torch.set_num_threads(1)
    set_process_label("mop-c2-verify-worker")


def _verify_cell(task: CellTask) -> CellResult:
    seed, difficulty, relative_path, expected_digest = task
    prefix = f"seed {seed} difficulty {difficulty}"
    if _WORKER_CONFIG is None or _WORKER_CONFIG_SHA256 is None or _WORKER_REPO_ROOT is None:
        return seed, difficulty, relative_path, [f"{prefix}: worker authority is uninitialized"], None
    try:
        path = (_WORKER_REPO_ROOT / relative_path).resolve()
        if not path.is_relative_to(_WORKER_REPO_ROOT):
            return seed, difficulty, relative_path, [f"{prefix}: receipt path escapes repository"], None
        receipt = independent._read(path, "C2 cell")
        problems: list[str] = []
        if expected_digest != receipt.get("cell_sha256"):
            problems.append(f"{prefix}: inventory digest drifted")
        cell_problems, metrics = independent._cell_problems(
            receipt,
            _WORKER_CONFIG,
            _WORKER_CONFIG_SHA256,
            seed,
            difficulty,
            reproduce_dataset=True,
        )
        problems.extend(cell_problems)
        return seed, difficulty, relative_path, problems, metrics
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            seed,
            difficulty,
            relative_path,
            [f"{prefix}: {type(exc).__name__}: {exc}"],
            None,
        )
    except Exception as exc:  # pragma: no cover - defensive worker containment
        return (
            seed,
            difficulty,
            relative_path,
            [f"{prefix}: worker verification failed: {type(exc).__name__}: {exc}"],
            None,
        )


def _parallel_reproduce(
    tasks: Sequence[CellTask],
    config: Mapping[str, Any],
    config_sha256: str,
    repo_root: Path,
) -> tuple[dict[Coordinate, CellResult], list[dict[str, Any]]]:
    remaining = deque(tasks)
    results: dict[Coordinate, CellResult] = {}
    failures: dict[Coordinate, int] = {}
    mode_history: list[dict[str, Any]] = []
    while remaining:
        workers, mode, state_problem = _queue_worker_target(config)
        entry: dict[str, Any] = {
            "mode": mode,
            "workers": workers,
            "starting_remaining_cells": len(remaining),
        }
        if state_problem is not None:
            entry["state_problem"] = state_problem
        completed_in_mode = 0
        mode_changed = False
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(config, config_sha256, str(repo_root.resolve())),
        )
        active: dict[Future[CellResult], CellTask] = {}
        next_mode_check = time.monotonic() + 2.0
        try:
            while remaining or active:
                now = time.monotonic()
                if not mode_changed and now >= next_mode_check:
                    observed_workers, observed_mode, _ = _queue_worker_target(config)
                    mode_changed = observed_workers != workers or observed_mode != mode
                    next_mode_check = now + 2.0
                while not mode_changed and remaining and len(active) < workers:
                    task = remaining.popleft()
                    coordinate = (task[0], task[1])
                    try:
                        future = executor.submit(_verify_cell, task)
                    except Exception as exc:  # pragma: no cover - broken executor containment
                        count = failures.get(coordinate, 0) + 1
                        failures[coordinate] = count
                        if count < 3:
                            remaining.append(task)
                        else:
                            prefix = f"seed {task[0]} difficulty {task[1]}"
                            results[coordinate] = (
                                task[0],
                                task[1],
                                task[2],
                                [
                                    f"{prefix}: parallel submission failed after three attempts: "
                                    f"{type(exc).__name__}: {exc}"
                                ],
                                None,
                            )
                        mode_changed = True
                    else:
                        active[future] = task
                if not active:
                    break
                timeout = max(0.01, min(2.0, next_mode_check - time.monotonic()))
                done, _ = wait(tuple(active), timeout=timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    coordinate = (task[0], task[1])
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - executor/process failure
                        count = failures.get(coordinate, 0) + 1
                        failures[coordinate] = count
                        if count < 3:
                            remaining.append(task)
                        else:
                            prefix = f"seed {task[0]} difficulty {task[1]}"
                            results[coordinate] = (
                                task[0],
                                task[1],
                                task[2],
                                [
                                    f"{prefix}: parallel worker failed after three attempts: "
                                    f"{type(exc).__name__}: {exc}"
                                ],
                                None,
                            )
                    else:
                        results[coordinate] = result
                    completed_in_mode += 1
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        entry["ending_remaining_cells"] = len(remaining)
        entry["completed_cells"] = completed_in_mode
        entry["mode_changed"] = mode_changed
        mode_history.append(entry)
    return results, mode_history


def _inventory_tasks(
    inventory: object,
    expected_coordinates: Sequence[Coordinate],
) -> tuple[list[CellTask], list[str]]:
    problems: list[str] = []
    tasks: list[CellTask] = []
    if not isinstance(inventory, list) or len(inventory) != len(expected_coordinates):
        return [], ["result cell inventory length drifted"]
    observed_coordinates = [
        (row.get("seed"), row.get("difficulty_index")) if isinstance(row, dict) else (None, None)
        for row in inventory
    ]
    if observed_coordinates != list(expected_coordinates):
        problems.append("result cell inventory order or coordinates drifted")
    for row, (seed, difficulty) in zip(inventory, expected_coordinates, strict=True):
        if not isinstance(row, dict):
            problems.append(f"seed {seed} difficulty {difficulty}: inventory row is invalid")
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            problems.append(f"seed {seed} difficulty {difficulty}: inventory path is invalid")
            continue
        digest = row.get("cell_sha256")
        tasks.append((seed, difficulty, path, digest if isinstance(digest, str) else None))
    return tasks, problems


def verify_result_parallel(
    config_path: Path | str,
    result_path: Path | str,
    out_path: Path | str,
    *,
    idle_workers: int,
    hawking_workers: int,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    config, config_sha256, prerequisite = load_config(config_path, repo_root=repo_root)
    resources = config.get("adaptive_resources") or {}
    if idle_workers != resources.get("idle_workers") or hawking_workers != resources.get("hawking_workers"):
        raise ValueError("verifier worker declaration differs from the sealed C2 config")
    result = independent._read(result_path, "C2 result")
    expected_coordinates = [
        (seed, difficulty)
        for seed in range(int(config["seed_start"]), int(config["seed_start"]) + int(config["seed_count"]))
        for difficulty in range(len(config["difficulty_separations"]))
    ]
    tasks, problems = _inventory_tasks(result.get("cell_receipts"), expected_coordinates)
    results, mode_history = _parallel_reproduce(tasks, config, config_sha256, repo_root)
    scheduled_coordinates = {(task[0], task[1]) for task in tasks}

    rebuilt_rows: list[tuple[int, int, Mapping[str, Any]]] = []
    clean_paths: list[str] = []
    for coordinate in expected_coordinates:
        cell = results.get(coordinate)
        if cell is None:
            if coordinate in scheduled_coordinates:
                problems.append(f"seed {coordinate[0]} difficulty {coordinate[1]}: no worker result")
            continue
        _, _, relative_path, cell_problems, metrics = cell
        problems.extend(cell_problems)
        if metrics is not None:
            rebuilt_rows.append((coordinate[0], coordinate[1], metrics))
        if not cell_problems:
            clean_paths.append(relative_path)

    rebuilt = (
        independent._rebuild_aggregate(config, rebuilt_rows)
        if len(rebuilt_rows) == len(expected_coordinates)
        else None
    )
    problems.extend(independent._semantic_problems(result, config, config_sha256, rebuilt))
    canary_receipt: dict[str, Any] | None = None
    if clean_paths:
        try:
            canary_path = (repo_root / clean_paths[0]).resolve()
            if not canary_path.is_relative_to(repo_root.resolve()):
                raise ValueError("canary receipt path escapes repository")
            canary_receipt = independent._read(canary_path, "C2 canary cell")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"fresh actor canary receipt failed: {type(exc).__name__}: {exc}")
    canary = (
        independent._run_canary(config, canary_receipt)
        if canary_receipt is not None
        else {"passed": False, "problem": "no clean cell was available for canary replay"}
    )
    if canary.get("passed") is not True:
        problems.append("fresh actor canary replay failed")
    mutation_suite = (
        independent._mutation_suite(result, config, config_sha256, rebuilt) if rebuilt is not None else None
    )
    if mutation_suite is None or mutation_suite.get("all_rejected") is not True:
        problems.append("mutation suite did not reject every corruption")
    unique_problems = list(dict.fromkeys(problems))
    core = {
        "schema": independent.VERIFICATION_SCHEMA,
        "campaign_id": config["campaign_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_file_sha256": config_sha256,
        "result_path": str(Path(result_path).resolve().relative_to(repo_root.resolve())),
        "result_sha256": result.get("result_sha256"),
        "prerequisite": prerequisite,
        "independent_recompute": rebuilt,
        "dataset_reproduction": {
            "expected_cells": len(expected_coordinates),
            "reproduced_cells": len(rebuilt_rows),
            "all_dataset_and_metric_reproductions_passed": len(rebuilt_rows) == len(expected_coordinates),
        },
        "parallel_execution": {
            "idle_workers": idle_workers,
            "hawking_workers": hawking_workers,
            "mode_history": mode_history,
            "worker_result_count": len(results),
        },
        "fresh_actor_canary": canary,
        "mutation_suite": mutation_suite,
        "verification_complete": not unique_problems,
        "problems": unique_problems,
        "interpretation_limit": (
            "Independent verification remains bounded to generated latent data with supplied "
            "diagnostic context labels and grants no learned-dispatch, activation, or substrate claim."
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    verification = {**core, "verification_sha256": canonical_sha256(core)}
    atomic_write_json(out_path, verification)
    return verification


__all__ = ["verify_result_parallel"]
