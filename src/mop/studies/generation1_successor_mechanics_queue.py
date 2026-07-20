
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.ladder.stage3_registry import build_pair
from mop.process_labels import set_process_label
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256, sha256_file
from mop.studies.generation1_c3_dispatch_queue import hawking_mode

PROGRAM_ID = "generation1-successor-mechanics-extended-v1"
STATUS_SCHEMA = "mop-generation1-successor-mechanics-queue-status/v1"
RUNG_SCHEMA = "mop-generation1-successor-mechanics-rung/v1"
RESULT_SCHEMA = "mop-generation1-successor-mechanics-extended/v1"
CLAIM_SCOPE = "deterministic generated successor mechanics stress only; no confirmation or activation"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_MECHANICS_EXTENDED.json"


@dataclass(frozen=True, slots=True)
class LaneSpec:
    lane_id: str
    mechanism: str
    dependencies: tuple[str, ...]
    current_c3_lane: bool
    canary_start: int
    producer_start: int
    challenge_start: int
    rungs_per_phase: int
    seeds_per_rung: int


@dataclass(frozen=True, slots=True)
class WorkItem:
    index: int
    lane_id: str
    mechanism: str
    phase: str
    rung_index: int
    seed_start: int
    seed_count: int

    @property
    def capsule_id(self) -> str:
        return f"g1_{self.lane_id[3:].lower()}_{self.phase}_rung_{self.rung_index:03d}"


LANES = (
    LaneSpec("G1-V1", "messaging_repair", ("G1-C1",), True, 39_000_001, 40_000_001, 45_000_001, 64, 65_536),
    LaneSpec("G1-M1", "messaging_repair", ("G1-C2",), True, 50_000_001, 51_000_001, 56_000_001, 64, 65_536),
    LaneSpec(
        "G1-G1",
        "construction_search",
        ("G1-C1", "G1-C2"),
        True,
        61_000_001,
        62_000_001,
        67_000_001,
        480,
        2_048,
    ),
    LaneSpec("G1-E1", "event_formation", ("G1-C0",), False, 72_000_001, 73_000_001, 78_000_001, 64, 65_536),
    LaneSpec(
        "G1-R1", "memory_organization", ("G1-C0",), False, 83_000_001, 84_000_001, 89_000_001, 64, 65_536
    ),
    LaneSpec(
        "G1-P1", "stability_plasticity", ("G1-C0",), False, 94_000_001, 95_000_001, 100_000_001, 64, 65_536
    ),
    LaneSpec("G1-C0", "trace_stability", (), False, 105_000_001, 106_000_001, 111_000_001, 64, 65_536),
    LaneSpec(
        "G1-D1",
        "niche_dispatch",
        ("G1-C1", "G1-C2"),
        False,
        116_000_001,
        117_000_001,
        122_000_001,
        64,
        65_536,
    ),
    LaneSpec(
        "G1-K1", "messaging_repair", ("G1-C1",), False, 127_000_001, 128_000_001, 133_000_001, 64, 65_536
    ),
    LaneSpec(
        "G1-A1",
        "intervention_simulation",
        ("G1-C0",),
        False,
        138_000_001,
        139_000_001,
        144_000_001,
        64,
        65_536,
    ),
    LaneSpec(
        "G1-S1",
        "intervention_simulation",
        ("G1-C0",),
        False,
        149_000_001,
        150_000_001,
        155_000_001,
        64,
        65_536,
    ),
    LaneSpec(
        "G1-I1",
        "integrated_escs",
        ("G1-E1", "G1-D1", "G1-M1", "G1-V1", "G1-R1", "G1-P1"),
        False,
        160_000_001,
        161_000_001,
        166_000_001,
        64,
        65_536,
    ),
)

CANARY_SEEDS = 256
STATUS_BLOCK_RUNGS = 5

PLANNED_SECONDS_PER_SEED = {
    "messaging_repair": 0.000_136,
    "construction_search": 83.530_492_166 / 2_048,
    "event_formation": 0.000_192,
    "memory_organization": 0.000_271,
    "stability_plasticity": 0.000_211,
    "trace_stability": 0.000_958,
    "niche_dispatch": 0.000_170,
    "intervention_simulation": 0.000_112,
    "integrated_escs": 0.000_230,
}


def build_work_items() -> tuple[WorkItem, ...]:
    items = []
    for lane in LANES:
        items.append(
            WorkItem(
                index=len(items),
                lane_id=lane.lane_id,
                mechanism=lane.mechanism,
                phase="canary",
                rung_index=0,
                seed_start=lane.canary_start,
                seed_count=CANARY_SEEDS,
            )
        )
    for lane in LANES:
        for phase, base in (("producer", lane.producer_start), ("challenge", lane.challenge_start)):
            for rung_index in range(lane.rungs_per_phase):
                items.append(
                    WorkItem(
                        index=len(items),
                        lane_id=lane.lane_id,
                        mechanism=lane.mechanism,
                        phase=phase,
                        rung_index=rung_index,
                        seed_start=base + rung_index * lane.seeds_per_rung,
                        seed_count=lane.seeds_per_rung,
                    )
                )
    return tuple(items)


WORK_ITEMS = build_work_items()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _item_path(root: Path, item: WorkItem) -> Path:
    return root / item.lane_id.lower() / item.phase / f"rung_{item.rung_index:03d}.json"


def _run_item_payload(item: WorkItem) -> dict[str, Any]:
    bed, runner = build_pair(item.mechanism)
    verdict_counts: dict[str, int] = {}
    control_clear_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    confirmations = 0
    for seed in range(item.seed_start, item.seed_start + item.seed_count):
        results = runner.run(bed, seed)
        receipt = runner.mint(results)
        if receipt.mechanism_id != item.mechanism:
            raise ValueError("mechanics runner receipt identity drifted")
        confirmations += int(receipt.is_confirmation)
        verdict_counts[receipt.verdict] = verdict_counts.get(receipt.verdict, 0) + 1
        for control in receipt.controls_cleared:
            control_clear_counts[control] = control_clear_counts.get(control, 0) + 1
        digest.update(receipt.digest().encode("ascii"))
    core = {
        "schema": RUNG_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "item": {
            "index": item.index,
            "lane_id": item.lane_id,
            "mechanism": item.mechanism,
            "phase": item.phase,
            "rung_index": item.rung_index,
            "seed_start": item.seed_start,
            "seed_count": item.seed_count,
        },
        "receipt_count": item.seed_count,
        "verdict_counts": verdict_counts,
        "control_clear_counts": control_clear_counts,
        "confirmation_count": confirmations,
        "receipt_digest_fold": digest.hexdigest(),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_rung(value: Mapping[str, Any], item: WorkItem) -> None:
    core = {key: row for key, row in value.items() if key != "result_sha256"}
    expected_item = {
        "index": item.index,
        "lane_id": item.lane_id,
        "mechanism": item.mechanism,
        "phase": item.phase,
        "rung_index": item.rung_index,
        "seed_start": item.seed_start,
        "seed_count": item.seed_count,
    }
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("successor mechanics rung seal drifted")
    if (
        value.get("schema") != RUNG_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("item") != expected_item
        or value.get("receipt_count") != item.seed_count
        or value.get("confirmation_count") != 0
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("successor mechanics rung identity or safety drifted")


def _load_valid_item(root: Path, item: WorkItem) -> dict[str, Any] | None:
    path = _item_path(root, item)
    if not path.is_file():
        return None
    try:
        result = _read_object(path)
        validate_rung(result, item)
        return result
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _run_item(item: WorkItem, root_string: str) -> tuple[int, dict[str, Any], float]:
    set_process_label(f"mop-{item.lane_id.lower()}-{item.phase[:4]}-r{item.rung_index:03d}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    result = _run_item_payload(item)
    validate_rung(result, item)
    atomic_write_json(_item_path(Path(root_string), item), result)
    return item.index, result, time.perf_counter() - started


def _mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    half = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": len(values)}


def _phase_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    receipt_count = sum(int(row["receipt_count"]) for row in rows)
    verdict_counts: dict[str, int] = {}
    control_counts: dict[str, int] = {}
    for row in rows:
        for verdict, count in row["verdict_counts"].items():
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + int(count)
        for control, count in row["control_clear_counts"].items():
            control_counts[control] = control_counts.get(control, 0) + int(count)
    return {
        "rung_count": len(rows),
        "receipt_count": receipt_count,
        "verdict_counts": verdict_counts,
        "verdict_fractions": {
            verdict: count / receipt_count for verdict, count in sorted(verdict_counts.items())
        },
        "control_clear_fractions": {
            control: count / receipt_count for control, count in sorted(control_counts.items())
        },
        "confirmation_count": 0,
    }


def aggregate(receipts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    expected = {item.index for item in _eligible_items(receipts)}
    if set(receipts) != expected:
        raise ValueError("successor mechanics aggregate requires every canary and eligible work item")
    lanes: dict[str, Any] = {}
    for lane in LANES:
        phases = {}
        for phase in ("canary", "producer", "challenge"):
            rows = [
                receipts[item.index]
                for item in WORK_ITEMS
                if item.lane_id == lane.lane_id and item.phase == phase and item.index in receipts
            ]
            phases[phase] = _phase_summary(rows)
        producer_fractions = phases["producer"]["verdict_fractions"]
        challenge_fractions = phases["challenge"]["verdict_fractions"]
        verdicts = set(producer_fractions) | set(challenge_fractions)
        repeatability_delta = {
            verdict: challenge_fractions.get(verdict, 0.0) - producer_fractions.get(verdict, 0.0)
            for verdict in sorted(verdicts)
        }
        lanes[lane.lane_id] = {
            "mechanism": lane.mechanism,
            "dependencies": list(lane.dependencies),
            "current_c3_lane": lane.current_c3_lane,
            "phases": phases,
            "canary_gate_passed": phases["canary"]["verdict_counts"] == {"mechanics-ok": CANARY_SEEDS},
            "long_work_executed": bool(phases["producer"]["rung_count"]),
            "execution_decision": (
                "continue_long_stress"
                if phases["canary"]["verdict_counts"] == {"mechanics-ok": CANARY_SEEDS}
                else "prune_after_canary"
            ),
            "verdict_fraction_challenge_minus_producer": repeatability_delta,
            "same_code_challenge_only": True,
            "independent_verification_complete": False,
        }
    total_receipts = sum(int(row["receipt_count"]) for row in receipts.values())
    pruned_lanes = [lane_id for lane_id, row in lanes.items() if not row["canary_gate_passed"]]
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "lane_order": [lane.lane_id for lane in LANES],
        "lanes": lanes,
        "grid": {
            "lane_count": len(LANES),
            "current_c3_lane_count": sum(lane.current_c3_lane for lane in LANES),
            "added_lane_count": sum(not lane.current_c3_lane for lane in LANES),
            "phase_count": len(LANES) * 3,
            "planned_rung_count": len(WORK_ITEMS),
            "executed_rung_count": len(receipts),
            "pruned_rung_count": len(WORK_ITEMS) - len(receipts),
            "receipt_count": total_receipts,
        },
        "decision": {
            "all_eligible_scaffold_stress_lanes_complete": True,
            "current_lanes_covered": ["G1-V1", "G1-M1", "G1-G1"],
            "additional_lanes_covered": [
                "G1-E1",
                "G1-R1",
                "G1-P1",
                "G1-C0",
                "G1-D1",
                "G1-K1",
                "G1-A1",
                "G1-S1",
                "G1-I1",
            ],
            "all_canary_gates_passed": all(row["canary_gate_passed"] for row in lanes.values()),
            "pruned_lanes": pruned_lanes,
            "independent_verification_complete": False,
            "scientific_confirmation": False,
            "next_action": "implement_lane_specific_scientific_producers_and_verifiers",
        },
        "interpretation_limit": (
            "Large disjoint deterministic partitions test reproducibility, control wiring, and receipt "
            "stability in toy beds only. They do not provide natural-task efficacy or confirmation."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_aggregate(value: Mapping[str, Any]) -> None:
    core = {key: row for key, row in value.items() if key != "result_sha256"}
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("successor mechanics aggregate seal drifted")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("grid", {}).get("planned_rung_count") != len(WORK_ITEMS)
        or value.get("grid", {}).get("executed_rung_count") + value.get("grid", {}).get("pruned_rung_count")
        != len(WORK_ITEMS)
        or value.get("decision", {}).get("independent_verification_complete") is not False
        or value.get("decision", {}).get("scientific_confirmation") is not False
    ):
        raise ValueError("successor mechanics aggregate identity or safety drifted")


def execution_mode() -> tuple[str, int, str | None]:
    mode, _, problem = hawking_mode()
    return f"{mode}_stable_single_worker", 1, problem


def _canary_gate(receipts: Mapping[int, Mapping[str, Any]], lane_id: str) -> bool | None:
    canary = next(item for item in WORK_ITEMS if item.lane_id == lane_id and item.phase == "canary")
    row = receipts.get(canary.index)
    if row is None:
        return None
    return row.get("verdict_counts") == {"mechanics-ok": CANARY_SEEDS} and row.get("confirmation_count") == 0


def _eligible_items(receipts: Mapping[int, Mapping[str, Any]]) -> tuple[WorkItem, ...]:

    return tuple(
        item for item in WORK_ITEMS if item.phase == "canary" or _canary_gate(receipts, item.lane_id) is True
    )


def _eta_items(receipts: Mapping[int, Mapping[str, Any]]) -> tuple[WorkItem, ...]:

    return tuple(
        item
        for item in WORK_ITEMS
        if item.index not in receipts
        and (item.phase == "canary" or _canary_gate(receipts, item.lane_id) is not False)
    )


def _status_items(receipts: Mapping[int, Mapping[str, Any]]) -> tuple[WorkItem, ...]:
    if any(_canary_gate(receipts, lane.lane_id) is None for lane in LANES):
        return tuple(item for item in WORK_ITEMS if item.phase == "canary")
    return _eligible_items(receipts)


def _eta_estimates(
    receipts: Mapping[int, Mapping[str, Any]], durations: Mapping[int, float]
) -> tuple[float, float, float]:
    observed_seconds: dict[str, float] = {}
    observed_seeds: dict[str, int] = {}
    by_index = {item.index: item for item in WORK_ITEMS}
    for index, duration in durations.items():
        item = by_index[index]
        observed_seconds[item.mechanism] = observed_seconds.get(item.mechanism, 0.0) + duration
        observed_seeds[item.mechanism] = observed_seeds.get(item.mechanism, 0) + item.seed_count
    rates = dict(PLANNED_SECONDS_PER_SEED)
    for mechanism, seconds in observed_seconds.items():
        rates[mechanism] = seconds / observed_seeds[mechanism]
    pending = list(_eta_items(receipts))
    if not pending:
        return 0.0, 0.0, 0.0

    def estimate(item: WorkItem) -> float:
        return item.seed_count * rates[item.mechanism]

    next_rung = estimate(pending[0])
    next_block = sum(estimate(item) for item in pending[:STATUS_BLOCK_RUNGS])
    session = sum(estimate(item) for item in pending)
    return next_rung, next_block, session


def _status(
    *,
    root: Path,
    receipts: Mapping[int, Mapping[str, Any]],
    attempts: Mapping[int, int],
    durations: Mapping[int, float],
    state: str,
    mode: str,
    workers: int,
    problem: str | None,
) -> dict[str, Any]:
    capsules: dict[str, Any] = {}
    status_items = _status_items(receipts)
    for item in status_items:
        row: dict[str, Any] = {"attempts": attempts.get(item.index, 0), "artifacts": []}
        if item.index in receipts:
            path = _item_path(root, item)
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
                            "schema": receipts[item.index]["schema"],
                            "problems": [],
                        }
                    ],
                }
            )
        capsules[item.capsule_id] = row
    average = statistics.fmean(durations.values()) if durations else None
    remaining = len(status_items) - len(receipts)
    next_rung_eta, next_block_eta, eta = _eta_estimates(receipts, durations)
    lane_progress = {}
    for lane in LANES:
        indexes = {item.index for item in status_items if item.lane_id == lane.lane_id}
        lane_progress[lane.lane_id] = {
            "complete": len(indexes & set(receipts)),
            "total": len(indexes),
        }
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "state": state,
        "capsules": capsules,
        "lane_progress": lane_progress,
        "adaptive_execution": {
            "mode": mode,
            "workers": workers,
            "hawking_state_problem": problem,
            "average_rung_seconds": average,
            "next_rung_seconds": next_rung_eta,
            "next_block_seconds": next_block_eta,
            "eta_seconds": eta,
        },
        "counts": {"complete": len(receipts), "total": len(status_items), "remaining": remaining},
        "problems": [] if state != "failure_hold" else [problem or "rung retry limit exhausted"],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": canonical_sha256(core)}


def run_queue(
    *,
    root: Path = DEFAULT_ROOT,
    result_path: Path = DEFAULT_RESULT,
    retry_limit: int = 3,
) -> dict[str, Any]:
    set_process_label("mop-successor-mechanics-queue")
    root = Path(root).resolve()
    result_path = Path(result_path).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()) or not result_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("successor mechanics paths must remain inside the repository")
    root.mkdir(parents=True, exist_ok=True)
    receipts: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    durations: dict[int, float] = {}
    for item in WORK_ITEMS:
        result = _load_valid_item(root, item)
        if result is not None:
            receipts[item.index] = result
    mode, workers, state_problem = execution_mode()

    def publish(state: str, problem: str | None = state_problem) -> dict[str, Any]:
        status = _status(
            root=root,
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
    while True:
        mode, workers, state_problem = execution_mode()
        pending = [item for item in _eligible_items(receipts) if item.index not in receipts]
        if not pending:
            unresolved = [lane.lane_id for lane in LANES if _canary_gate(receipts, lane.lane_id) is None]
            if unresolved:
                return publish(
                    "failure_hold",
                    "no eligible work remains before canaries resolve: " + ", ".join(unresolved),
                )
            break
        wave = pending[:workers]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            futures = {}
            for item in wave:
                attempts[item.index] = attempts.get(item.index, 0) + 1
                futures[pool.submit(_run_item, item, str(root))] = item
            for future in as_completed(futures):
                item = futures[future]
                try:
                    completed_index, result, duration = future.result()
                    receipts[completed_index] = result
                    durations[completed_index] = duration
                    publish("running")
                except Exception as exc:  # noqa: BLE001 - sealed retry boundary
                    if attempts[item.index] >= retry_limit:
                        problem = (
                            f"{item.capsule_id} failed after {attempts[item.index]} attempts: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        return publish("failure_hold", problem)
        publish("running")
    result = aggregate(receipts)
    validate_aggregate(result)
    atomic_write_json(result_path, result)
    return publish("complete")
