"""Restart-safe frozen D1 producer and challenge-replication queue.

The queue freezes the only router variant that cleared the exploratory static
margin gate, then evaluates it on two large, disjoint fresh-seed partitions.
The second partition is a challenge replication, not an independent verifier:
both partitions deliberately remain non-promotable until separately authored
verification code reproduces the result.
"""

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

PROGRAM_ID = "generation1-c3-d1-frozen-producer-challenge-v1"
STATUS_SCHEMA = "mop-generation1-c3-d1-frozen-queue-status/v1"
RESULT_SCHEMA = "mop-generation1-c3-d1-frozen-producer-challenge/v1"
CLAIM_SCOPE = "frozen generated D1 producer and challenge replication; independent verification pending"

SCREEN_RELATIVE = Path("proof/GENERATION1_C3_D1_ROUTER_REDESIGN_SCREEN.json")
SCREEN_FILE_SHA256 = "edd845aae19d679cbd9eabb4c7fc4d6571c5d9be2a3af7535d653eeb8f2262d9"
SCREEN_RESULT_SHA256 = "457547218ccc134276720acb804ad57c39a53a061f64519c0b8e654a3cb08c92"
FROZEN_VARIANT_ID = "centroid-h64-e60-lr03"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_C3_D1_FROZEN_PRODUCER_CHALLENGE.json"
SEEDS_PER_RUNG = 576
RUNGS_PER_PHASE = 288
DEFAULT_RUNG_COUNT = RUNGS_PER_PHASE * 2

PHASES = ("producer", "challenge")
PHASE_BASES = {
    "producer": {"train": 20_330_001, "heldout": 20_600_001, "router": 51_000_001},
    "challenge": {"train": 20_900_001, "heldout": 21_200_001, "router": 91_000_001},
}

CRITERIA = {
    "minimum_mean_advantage_over_each_static_control": 0.01,
    "comparison_interval_lower_bound_must_exceed": 0.0,
    "minimum_favorable_seed_fraction": 0.75,
    "maximum_mean_gap_below_fixed_c2_context_route": 0.02,
    "minimum_work_saving_vs_all_five_actors": 0.70,
    "all_conditions_required": True,
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def screen_binding(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = repo_root / SCREEN_RELATIVE
    if sha256_file(path) != SCREEN_FILE_SHA256:
        raise ValueError("D1 redesign screen file drifted")
    value = _read_object(path)
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("D1 redesign screen self-seal is invalid")
    if (
        value.get("result_sha256") != SCREEN_RESULT_SHA256
        or value.get("best_exploratory_variant") != FROZEN_VARIANT_ID
        or value.get("decision", {}).get("best_variant_clears_gate") is not True
        or value.get("complete") is not True
        or value.get("problems") != []
    ):
        raise ValueError("D1 redesign screen does not authorize the frozen replication")
    return {
        "path": str(SCREEN_RELATIVE),
        "file_sha256": SCREEN_FILE_SHA256,
        "result_sha256": SCREEN_RESULT_SHA256,
        "frozen_variant_id": FROZEN_VARIANT_ID,
    }


def frozen_variant() -> dict[str, Any]:
    return next(dict(value) for value in redesign.variant_grid() if value["variant_id"] == FROZEN_VARIANT_ID)


def phase_of(index: int) -> tuple[str, int]:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("rung index must be a nonnegative integer")
    if index < RUNGS_PER_PHASE:
        return "producer", index
    return "challenge", index - RUNGS_PER_PHASE


def rung_config(index: int) -> dict[str, Any]:
    phase, local_index = phase_of(index)
    bases = PHASE_BASES[phase]
    return redesign.redesign_config(
        train_seed_start=bases["train"] + local_index * SEEDS_PER_RUNG,
        train_seed_count=SEEDS_PER_RUNG,
        heldout_seed_start=bases["heldout"] + local_index * SEEDS_PER_RUNG,
        heldout_seed_count=SEEDS_PER_RUNG,
        difficulty_indices=(0, 1, 2, 3, 4),
        n_train=720,
        n_test=240,
        n_classes=10,
        dim=64,
        actor_epochs=6,
        router_training_seed=bases["router"] + local_index * 100_003,
        variants=(frozen_variant(),),
    )


def _rung_path(root: Path, index: int) -> Path:
    phase, local_index = phase_of(index)
    return root / phase / f"rung_{local_index:03d}.json"


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
    phase, local_index = phase_of(index)
    set_process_label(f"mop-d1-{phase[:4]}-r{local_index:03d}")
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


def _favorable_seed_fraction(cells: Sequence[Mapping[str, Any]], control: str) -> float:
    by_seed: dict[int, list[float]] = {}
    for cell in cells:
        seed = int(cell["seed"])
        margin = float(cell["variant_accuracy"][FROZEN_VARIANT_ID]) - float(cell["control_accuracy"][control])
        by_seed.setdefault(seed, []).append(margin)
    return sum(statistics.fmean(values) > 0.0 for values in by_seed.values()) / len(by_seed)


def summarize_phase(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rungs:
        raise ValueError("a frozen D1 phase requires at least one rung")
    cells = [cell for rung in rungs for cell in rung["cells"]]
    summary = redesign.summarize_cells(cells, [FROZEN_VARIANT_ID])
    variant = summary["variants"][FROZEN_VARIANT_ID]
    differences = variant["differences"]
    favorable = {
        control: _favorable_seed_fraction(cells, control)
        for control in ("global_static", "difficulty_static")
    }
    static_margin = all(
        float(differences[control]["mean"])
        >= float(CRITERIA["minimum_mean_advantage_over_each_static_control"])
        and float(differences[control]["lo"]) > float(CRITERIA["comparison_interval_lower_bound_must_exceed"])
        for control in ("global_static", "difficulty_static")
    )
    favorable_gate = all(
        value >= float(CRITERIA["minimum_favorable_seed_fraction"]) for value in favorable.values()
    )
    context_gap = -float(differences["context_route_nonpromotable"]["mean"])
    context_gate = context_gap <= float(CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"])
    work_saving = 1.0 - (1.0 / 5.0)
    work_gate = work_saving >= float(CRITERIA["minimum_work_saving_vs_all_five_actors"])
    conditions = {
        "static_margin_gate": static_margin,
        "favorable_seed_fraction_gate": favorable_gate,
        "context_route_gap_gate": context_gate,
        "work_saving_gate": work_gate,
    }
    return {
        "grid": {
            "rung_count": len(rungs),
            "train_seed_count": sum(int(rung["grid"]["train_seed_count"]) for rung in rungs),
            "heldout_seed_count": sum(int(rung["grid"]["heldout_seed_count"]) for rung in rungs),
            "completed_cell_count": len(cells),
        },
        "overall": {
            "learned_dispatch": variant["learned_dispatch"],
            **summary["controls"],
        },
        "learned_dispatch_differences": differences,
        "favorable_seed_fraction": favorable,
        "mean_gap_below_context_route": context_gap,
        "work_saving_vs_all_five_actors": work_saving,
        "conditions": conditions,
        "all_frozen_criteria_passed": all(conditions.values()),
    }


def aggregate(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rungs) != DEFAULT_RUNG_COUNT:
        raise ValueError("frozen D1 aggregate requires the complete two-phase inventory")
    binding = screen_binding()
    producer = summarize_phase(rungs[:RUNGS_PER_PHASE])
    challenge = summarize_phase(rungs[RUNGS_PER_PHASE:])
    repeated = bool(producer["all_frozen_criteria_passed"] and challenge["all_frozen_criteria_passed"])
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "screen_binding": binding,
        "frozen_variant": frozen_variant(),
        "criteria": dict(CRITERIA),
        "rungs": [
            {
                "index": index,
                "phase": phase_of(index)[0],
                "result_sha256": rung["result_sha256"],
                "config_sha256": rung["config_sha256"],
                "completed_cell_count": rung["grid"]["completed_cell_count"],
            }
            for index, rung in enumerate(rungs)
        ],
        "phases": {"producer": producer, "challenge": challenge},
        "grid": {
            "phase_count": 2,
            "rung_count": len(rungs),
            "train_seed_count": sum(int(rung["grid"]["train_seed_count"]) for rung in rungs),
            "heldout_seed_count": sum(int(rung["grid"]["heldout_seed_count"]) for rung in rungs),
            "completed_cell_count": sum(int(rung["grid"]["completed_cell_count"]) for rung in rungs),
        },
        "overall": challenge["overall"],
        "learned_dispatch_differences": challenge["learned_dispatch_differences"],
        "decision": {
            "producer_all_frozen_criteria_passed": producer["all_frozen_criteria_passed"],
            "challenge_all_frozen_criteria_passed": challenge["all_frozen_criteria_passed"],
            "frozen_pattern_repeated": repeated,
            "independent_verification_complete": False,
            "ready_for_confirmatory_claim": False,
            "next_action": "run_v1_m1_g1_sibling_batch",
        },
        "interpretation_limit": (
            "The producer and challenge use disjoint fresh seeds but share one implementation. "
            "They can establish repeatability of the frozen generated pattern, not independent "
            "verification, activation, or scientific promotion."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_aggregate(value: Mapping[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "result_sha256"}
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("frozen D1 aggregate seal drifted")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("screen_binding") != screen_binding()
        or value.get("frozen_variant") != frozen_variant()
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("grid", {}).get("rung_count") != DEFAULT_RUNG_COUNT
        or value.get("decision", {}).get("independent_verification_complete") is not False
        or value.get("decision", {}).get("ready_for_confirmatory_claim") is not False
    ):
        raise ValueError("frozen D1 aggregate identity or safety drifted")


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
        phase, local_index = phase_of(index)
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
        capsules[f"g1_d1_{phase}_rung_{local_index:03d}"] = row
    average = statistics.fmean(durations.values()) if durations else None
    remaining = rung_count - len(receipts)
    eta = None if average is None else average * remaining / max(1, workers)
    complete_producer = sum(index < RUNGS_PER_PHASE for index in receipts)
    complete_challenge = len(receipts) - complete_producer
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "state": state,
        "capsules": capsules,
        "phase_progress": {
            "producer": {"complete": complete_producer, "total": RUNGS_PER_PHASE},
            "challenge": {"complete": complete_challenge, "total": RUNGS_PER_PHASE},
        },
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
    set_process_label("mop-d1-frozen-queue")
    screen_binding()
    root = Path(root).resolve()
    result_path = Path(result_path).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()) or not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("frozen D1 paths must remain inside the repository")
    if rung_count != DEFAULT_RUNG_COUNT:
        raise ValueError("frozen D1 phase size is sealed and cannot be overridden")
    root.mkdir(parents=True, exist_ok=True)
    receipts: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    durations: dict[int, float] = {}
    for index in range(rung_count):
        result = _load_valid_rung(root, index)
        if result is not None:
            receipts[index] = result
    hawking_state, _, state_problem = hawking_mode()
    mode, workers = f"{hawking_state}_frozen_single_worker", 1

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
        hawking_state, _, state_problem = hawking_mode()
        mode, workers = f"{hawking_state}_frozen_single_worker", 1
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
    validate_aggregate(result)
    atomic_write_json(result_path, result)
    return publish("complete")
