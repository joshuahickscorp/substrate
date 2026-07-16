"""Adaptive, restart-safe queue for the expanded G1-C3/D1 dispatch canary."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies.generation1_c3_dispatch import (
    atomic_write_json,
    canonical_sha256,
    pilot_config,
    run_pilot,
    sha256_file,
    validate_result,
)

PROGRAM_ID = "generation1-c3-d1-expanded-canary-v1"
STATUS_SCHEMA = "mop-generation1-c3-dispatch-queue-status/v1"
RESULT_SCHEMA = "mop-generation1-c3-dispatch-expanded-canary/v1"
DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_C3_D1_EXPANDED_CANARY.json"
HAWKING_QUEUE = Path("/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra/queue_state.json")
HAWKING_SCHEMA = "hawking.doctor_v5_ultra_queue_state.v1"


def rung_config(index: int) -> dict[str, Any]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("rung index must be a nonnegative integer")
    return pilot_config(
        train_seed_start=20_278_101 + index * 2,
        train_seed_count=2,
        heldout_seed_start=20_279_101 + index * 2,
        heldout_seed_count=2,
        difficulty_indices=(0, 1, 2, 3, 4),
        n_train=240,
        n_test=90,
        n_classes=10,
        dim=32,
        actor_epochs=3,
        router_epochs=20,
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
        validate_result(result, rung_config(index))
        return result
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _run_rung(index: int, root_string: str) -> tuple[int, dict[str, Any], float]:
    set_process_label(f"mop-c3-d1-r{index:02d}")
    root = Path(root_string)
    started = time.perf_counter()
    config = rung_config(index)
    result = run_pilot(config)
    validate_result(result, config)
    atomic_write_json(_rung_path(root, index), result)
    return index, result, time.perf_counter() - started


def hawking_mode(path: Path = HAWKING_QUEUE) -> tuple[str, int, str | None]:
    """Return a fail-closed worker target from the sealed Hawking queue."""

    try:
        value = _read_object(path)
        core = {key: item for key, item in value.items() if key != "state_sha256"}
        if value.get("schema") != HAWKING_SCHEMA or value.get("state_sha256") != canonical_sha256(core):
            raise ValueError("Hawking queue identity or seal drifted")
        active = value.get("active_cells")
        if not isinstance(active, list):
            raise ValueError("Hawking active cell inventory is invalid")
        return ("hawking_active", 1, None) if active else ("hawking_idle", 8, None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "hawking_unknown_fail_closed", 1, f"{type(exc).__name__}: {exc}"


def _mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def aggregate(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rungs:
        raise ValueError("expanded canary requires at least one rung")
    cells = [cell for rung in rungs for cell in rung["cells"]]
    arms = tuple(cells[0]["accuracy"])
    overall = {arm: _mean_ci([float(cell["accuracy"][arm]) for cell in cells]) for arm in arms}
    controls = (
        "global_static",
        "difficulty_static",
        "random_actor",
        "context_route_nonpromotable",
    )
    differences = {
        control: _mean_ci(
            [float(cell["accuracy"]["learned_dispatch"]) - float(cell["accuracy"][control]) for cell in cells]
        )
        for control in controls
    }
    beats_static = (
        float(differences["global_static"]["mean"]) > 0.0
        and float(differences["difficulty_static"]["mean"]) > 0.0
    )
    clears_static_gate = all(
        float(differences[control]["mean"]) >= 0.01 and float(differences[control]["lo"]) > 0.0
        for control in ("global_static", "difficulty_static")
    )
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": "expanded generated learned-dispatch canary only",
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
            "train_seed_count": sum(int(rung["grid"]["train_seed_count"]) for rung in rungs),
            "heldout_seed_count": sum(int(rung["grid"]["heldout_seed_count"]) for rung in rungs),
            "completed_cell_count": len(cells),
        },
        "overall": overall,
        "learned_dispatch_differences": differences,
        "decision": {
            "expanded_canary_complete": True,
            "advisory_learned_dispatch_beats_both_static_controls": beats_static,
            "clears_frozen_static_margin_gate": clears_static_gate,
            "next_action": (
                "design_confirmatory_preregistration"
                if clears_static_gate
                else "redesign_visible_router_before_confirmation"
            ),
            "ready_for_confirmatory_claim": False,
            "independent_verification_required": True,
        },
        "interpretation_limit": (
            "Multiple downsized fresh-seed pilots estimate router feasibility only. They do not "
            "confirm learned dispatch, authorize activation, or promote substrate science."
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
        raise ValueError("expanded canary aggregate seal drifted")
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
        raise ValueError("expanded canary aggregate identity or safety drifted")


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
                    "finished_at": dt_from_mtime(path),
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
        capsules[f"g1_c3_d1_expanded_rung_{index:02d}"] = row
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


def dt_from_mtime(path: Path) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat(timespec="seconds")


def run_queue(
    *,
    root: Path = DEFAULT_ROOT,
    result_path: Path = DEFAULT_RESULT,
    rung_count: int = 8,
    retry_limit: int = 3,
) -> dict[str, Any]:
    set_process_label("mop-c3-d1-queue")
    root = Path(root).resolve()
    result_path = Path(result_path).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()) or not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("expanded canary paths must remain inside the repository")
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
