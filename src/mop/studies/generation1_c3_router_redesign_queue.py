
from __future__ import annotations

import datetime as dt
import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies import generation1_c3_router_redesign as redesign
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256, sha256_file
from mop.studies.generation1_c3_dispatch_queue import hawking_mode

PROGRAM_ID = "generation1-c3-d1-router-redesign-screen-v1"
STATUS_SCHEMA = "mop-generation1-c3-router-redesign-queue-status/v1"
RESULT_SCHEMA = "mop-generation1-c3-router-redesign-screen/v1"
DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_C3_D1_ROUTER_REDESIGN_SCREEN.json"
DEFAULT_RUNG_COUNT = 48
SEEDS_PER_RUNG = 192


def rung_config(index: int) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("rung index must be a nonnegative integer")
    return redesign.redesign_config(
        train_seed_start=20_300_001 + index * SEEDS_PER_RUNG,
        train_seed_count=SEEDS_PER_RUNG,
        heldout_seed_start=20_310_001 + index * SEEDS_PER_RUNG,
        heldout_seed_count=SEEDS_PER_RUNG,
        difficulty_indices=(0, 1, 2, 3, 4),
        n_train=720,
        n_test=240,
        n_classes=10,
        dim=64,
        actor_epochs=6,
        router_training_seed=31_040_001 + index * 100_003,
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _rung_path(root: Path, index: int) -> Path:
    return root / "replicates" / f"rung_{index:02d}.json"


def _load_valid_rung(root: Path, index: int) -> dict[str, Any] | None:
    path = _rung_path(root, index)
    if not path.is_file():
        return None
    try:
        result = _read_object(path)
        redesign.validate_result(result, rung_config(index))
        return result
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _run_rung(index: int, root_string: str) -> tuple[int, dict[str, Any], float]:
    set_process_label(f"mop-c3-redesign-r{index:02d}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    config = rung_config(index)
    result = redesign.run_redesign(config)
    redesign.validate_result(result, config)
    atomic_write_json(_rung_path(Path(root_string), index), result)
    return index, result, time.perf_counter() - started


def _mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def aggregate(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rungs:
        raise ValueError("router redesign screen requires at least one rung")
    cells = [cell for rung in rungs for cell in rung["cells"]]
    variant_ids = [str(row["variant_id"]) for row in rungs[0]["config"]["variants"]]
    summary = redesign.summarize_cells(cells, variant_ids)
    best = str(summary["ranking"][0])
    best_row = summary["variants"][best]
    candidates = [
        variant_id
        for variant_id in summary["ranking"]
        if all(
            float(summary["variants"][variant_id]["differences"][control]["mean"]) >= 0.01
            and float(summary["variants"][variant_id]["differences"][control]["lo"]) > 0.0
            for control in ("global_static", "difficulty_static")
        )
    ]
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": redesign.CLAIM_SCOPE,
        "rungs": [
            {
                "index": index,
                "result_sha256": rung["result_sha256"],
                "config_sha256": rung["config_sha256"],
                "completed_cell_count": rung["grid"]["completed_cell_count"],
            }
            for index, rung in enumerate(rungs)
        ],
        "grid": {
            "rung_count": len(rungs),
            "variant_count": len(variant_ids),
            "train_seed_count": sum(int(rung["grid"]["train_seed_count"]) for rung in rungs),
            "heldout_seed_count": sum(int(rung["grid"]["heldout_seed_count"]) for rung in rungs),
            "completed_cell_count": len(cells),
            "shared_actor_evaluation": True,
        },
        "best_exploratory_variant": best,
        "overall": {
            "learned_dispatch": best_row["learned_dispatch"],
            **summary["controls"],
        },
        "learned_dispatch_differences": best_row["differences"],
        "variant_summary": summary,
        "decision": {
            "redesign_screen_complete": True,
            "variants_clearing_frozen_static_margin_gate": candidates,
            "best_variant_clears_gate": best in candidates,
            "next_action": (
                "freeze_best_variant_for_untouched_confirmation_design"
                if candidates
                else "iterate_router_representation_before_confirmation"
            ),
            "ready_for_confirmatory_claim": False,
            "independent_verification_required": True,
        },
        "interpretation_limit": (
            "This repeated paired redesign screen is tuning evidence. Its selected winner must be frozen "
            "before any untouched confirmation and cannot authorize activation or scientific promotion."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_aggregate(value: Mapping[str, Any], rung_count: int) -> None:
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("router redesign aggregate seal drifted")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("grid", {}).get("rung_count") != rung_count
        or value.get("decision", {}).get("ready_for_confirmatory_claim") is not False
    ):
        raise ValueError("router redesign aggregate identity or safety drifted")


def _status(
    *,
    root: Path,
    rung_count: int,
    receipts: Mapping[int, Mapping[str, Any]],
    attempts: Mapping[int, int],
    durations: Mapping[int, float],
    state: str,
    mode: str,
    workers: int,
    problem: str | None,
) -> dict[str, Any]:
    capsules: dict[str, Any] = {}
    for index in range(rung_count):
        row: dict[str, Any] = {"attempts": attempts.get(index, 0), "artifacts": []}
        if index in receipts:
            path = _rung_path(root, index)
            row.update(
                {
                    "returncode": 0,
                    "finished_at": dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat(
                        timespec="seconds"
                    ),
                    "artifacts": [
                        {
                            "path": str(path.relative_to(REPO_ROOT)),
                            "sha256": sha256_file(path),
                            "all_ok": True,
                            "schema": receipts[index]["schema"],
                            "problems": [],
                        }
                    ],
                }
            )
        capsules[f"g1_c3_router_redesign_rung_{index:02d}"] = row
    average = statistics.fmean(durations.values()) if durations else None
    remaining = rung_count - len(receipts)
    eta = None if average is None else average * remaining / max(1, workers)
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "state": state,
        "capsules": capsules,
        "adaptive_execution": {
            "mode": mode,
            "workers": workers,
            "hawking_state_problem": problem,
            "average_rung_seconds": average,
            "eta_seconds": eta,
        },
        "counts": {"complete": len(receipts), "total": rung_count, "remaining": remaining},
        "problems": [] if state != "failure_hold" else [problem or "rung retry limit exhausted"],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": canonical_sha256(core)}


def run_queue(
    *,
    root: Path = DEFAULT_ROOT,
    result_path: Path = DEFAULT_RESULT,
    rung_count: int = DEFAULT_RUNG_COUNT,
    retry_limit: int = 3,
) -> dict[str, Any]:
    set_process_label("mop-c3-router-redesign-queue")
    root = Path(root).resolve()
    result_path = Path(result_path).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()) or not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("router redesign paths must remain inside the repository")
    if isinstance(rung_count, bool) or not isinstance(rung_count, int) or rung_count <= 0:
        raise ValueError("router redesign rung count must be positive")
    root.mkdir(parents=True, exist_ok=True)
    receipts: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    durations: dict[int, float] = {}
    for index in range(rung_count):
        result = _load_valid_rung(root, index)
        if result is not None:
            receipts[index] = result
    mode, workers, state_problem = hawking_mode()

    def publish(state: str, problem: str | None = state_problem) -> dict[str, Any]:
        status = _status(
            root=root,
            rung_count=rung_count,
            receipts=receipts,
            attempts=attempts,
            durations=durations,
            state=state,
            mode=mode,
            workers=workers,
            problem=problem,
        )
        atomic_write_json(root / "current_status.json", status)
        return status

    publish("running")
    while len(receipts) < rung_count:
        mode, workers, state_problem = hawking_mode()
        pending = [index for index in range(rung_count) if index not in receipts]
        wave = pending[:workers]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = {}
            for index in wave:
                attempts[index] = attempts.get(index, 0) + 1
                futures[pool.submit(_run_rung, index, str(root))] = index
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed_index, result, duration = future.result()
                    receipts[completed_index] = result
                    durations[completed_index] = duration
                    publish("running")
                except Exception as exc:  # noqa: BLE001 - sealed retry boundary
                    if attempts[index] >= retry_limit:
                        problem = (
                            f"rung {index} failed after {attempts[index]} attempts: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        return publish("failure_hold", problem)
        publish("running")
    ordered = [receipts[index] for index in range(rung_count)]
    result = aggregate(ordered)
    validate_aggregate(result, rung_count)
    atomic_write_json(result_path, result)
    return publish("complete")
