"""Bounded post-consolidation robustness horizon for Generation 1.

The horizon deliberately starts after the existing consolidated-v1 authority.
It adds five disjoint, result-gated robustness epochs.  Each epoch is split into
sub-five-hour supervisor capsules, checkpoints every underlying work item, and
publishes a sealed classification before the next epoch is admitted.

This is same-code robustness evidence.  It never becomes independent scientific
confirmation merely because it uses more seeds or compute.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.process_labels import set_process_label
from mop.studies import generation1_c3_d1_frozen_queue as d1
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256, sha256_file

PROGRAM_ID = "generation1-successor-horizon-v1"
ADMISSION_SCHEMA = "mop-generation1-successor-horizon-admission/v1"
SHARD_SCHEMA = "mop-generation1-successor-horizon-shard/v1"
CLASSIFICATION_SCHEMA = "mop-generation1-successor-horizon-classification/v1"
RESULT_SCHEMA = "mop-generation1-successor-horizon-result/v1"
REPORT_RECEIPT_SCHEMA = "mop-generation1-successor-horizon-report-receipt/v1"
CLAIM_SCOPE = (
    "result-gated same-code fresh-seed robustness and independently verified aggregation; "
    "no independent scientific confirmation or activation"
)

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_ADMISSION = DEFAULT_ROOT / "admission.json"
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_HORIZON.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_SUCCESSOR_HORIZON.verification.json"
DEFAULT_REPORT = REPO_ROOT / "runs/generation1/GENERATION1_SUCCESSOR_EVIDENCE_REPORT.md"
DEFAULT_REPORT_RECEIPT = DEFAULT_ROOT / "report_receipt.json"
PROGRAM_MANIFEST = REPO_ROOT / "configs/campaign/generation1_successor_horizon_v1.json"

EPOCH_IDS = ("H01", "H02", "H03", "H04", "H05")
FIRST_FRESH_CYCLE = consolidated.FRESH_D1_CYCLES
EPOCH_CYCLES = tuple(FIRST_FRESH_CYCLE + index for index in range(len(EPOCH_IDS)))
D1_SHARD_COUNT = 5
MECHANICS_SHARD_COUNT = 8
RETRY_LIMIT = 3
# Declared idle-host pool ceiling: the measured Hawking-idle aggregate throughput peak on this host
# (dynamic_worker_controller.WORKER_CEILING = 20; 24 workers regress). The ACTUAL pool width floats
# dynamically from 1 to this ceiling with live host load (see _dynamic_pool_width); this constant is
# only the declared ceiling recorded in each shard execution block and the planning divisor, never the
# fluctuating instantaneous count, so every sealed planning field stays deterministic.
IDLE_WORKERS = 20
HAWKING_WORKERS = 1


class _Moments:
    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def summary(self) -> dict[str, float | int]:
        half = 0.0 if self.count < 2 else 1.96 * math.sqrt(self.m2 / (self.count - 1)) / math.sqrt(self.count)
        return {
            "mean": self.mean,
            "lo": self.mean - half,
            "hi": self.mean + half,
            "n": self.count,
        }


class _D1PhaseAccumulator:
    __slots__ = (
        "completed_cell_count",
        "favorable",
        "heldout_seed_count",
        "learned",
        "margins",
        "rung_count",
        "seed_count",
        "train_seed_count",
    )

    def __init__(self) -> None:
        self.rung_count = 0
        self.train_seed_count = 0
        self.heldout_seed_count = 0
        self.completed_cell_count = 0
        self.seed_count = 0
        self.learned = _Moments()
        self.margins = {
            name: _Moments()
            for name in (
                "global_static",
                "difficulty_static",
                "context_route_nonpromotable",
            )
        }
        self.favorable = {"global_static": 0, "difficulty_static": 0}

    def add(self, value: Mapping[str, Any]) -> None:
        grid = value.get("grid") or {}
        cells = value.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("fresh D1 receipt has no heldout cells")
        self.rung_count += 1
        self.train_seed_count += int(grid["train_seed_count"])
        self.heldout_seed_count += int(grid["heldout_seed_count"])
        self.completed_cell_count += len(cells)
        by_seed: dict[int, dict[str, list[float]]] = {}
        for cell in cells:
            learned = float(cell["variant_accuracy"][d1.FROZEN_VARIANT_ID])
            self.learned.add(learned)
            seed = int(cell["seed"])
            seed_row = by_seed.setdefault(
                seed,
                {"global_static": [], "difficulty_static": []},
            )
            for control, moments in self.margins.items():
                margin = learned - float(cell["control_accuracy"][control])
                moments.add(margin)
                if control in seed_row:
                    seed_row[control].append(margin)
        self.seed_count += len(by_seed)
        for seed_row in by_seed.values():
            for control in self.favorable:
                self.favorable[control] += int(statistics.fmean(seed_row[control]) > 0.0)

    def result(self) -> dict[str, Any]:
        if not self.rung_count or not self.seed_count:
            raise ValueError("fresh D1 phase is empty")
        differences = {name: moments.summary() for name, moments in self.margins.items()}
        favorable = {name: count / self.seed_count for name, count in self.favorable.items()}
        static_margin = all(
            float(differences[control]["mean"])
            >= float(d1.CRITERIA["minimum_mean_advantage_over_each_static_control"])
            and float(differences[control]["lo"])
            > float(d1.CRITERIA["comparison_interval_lower_bound_must_exceed"])
            for control in ("global_static", "difficulty_static")
        )
        favorable_gate = all(
            value >= float(d1.CRITERIA["minimum_favorable_seed_fraction"]) for value in favorable.values()
        )
        context_gap = -float(differences["context_route_nonpromotable"]["mean"])
        work_saving = 1.0 - (1.0 / 5.0)
        conditions = {
            "static_margin_gate": static_margin,
            "favorable_seed_fraction_gate": favorable_gate,
            "context_route_gap_gate": context_gap
            <= float(d1.CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"]),
            "work_saving_gate": work_saving >= float(d1.CRITERIA["minimum_work_saving_vs_all_five_actors"]),
        }
        return {
            "grid": {
                "rung_count": self.rung_count,
                "train_seed_count": self.train_seed_count,
                "heldout_seed_count": self.heldout_seed_count,
                "completed_cell_count": self.completed_cell_count,
            },
            "learned_dispatch": self.learned.summary(),
            "learned_dispatch_differences": differences,
            "favorable_seed_fraction": favorable,
            "mean_gap_below_context_route": context_gap,
            "work_saving_vs_all_five_actors": work_saving,
            "conditions": conditions,
            "all_frozen_criteria_passed": all(conditions.values()),
        }


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(core):
        raise ValueError(f"{label} seal drifted")


def planned_epoch_compute_seconds() -> float:
    d1_seconds = d1.DEFAULT_RUNG_COUNT * consolidated.D1_PLANNED_RUNG_SECONDS
    mechanics_seconds = sum(
        item.seed_count * mechanics.PLANNED_SECONDS_PER_SEED[item.mechanism] for item in mechanics.WORK_ITEMS
    )
    return d1_seconds + mechanics_seconds


def planned_horizon_compute_seconds() -> float:
    return len(EPOCH_IDS) * planned_epoch_compute_seconds()


def d1_partitions() -> tuple[tuple[int, ...], ...]:
    base, extra = divmod(d1.DEFAULT_RUNG_COUNT, D1_SHARD_COUNT)
    rows: list[tuple[int, ...]] = []
    start = 0
    for shard in range(D1_SHARD_COUNT):
        width = base + int(shard < extra)
        rows.append(tuple(range(start, start + width)))
        start += width
    return tuple(rows)


def mechanics_partitions() -> tuple[tuple[int, ...], ...]:
    """Return stable contiguous shards balanced by the sealed planning rates."""

    costs = [
        item.seed_count * mechanics.PLANNED_SECONDS_PER_SEED[item.mechanism] for item in mechanics.WORK_ITEMS
    ]
    target = sum(costs) / MECHANICS_SHARD_COUNT
    rows: list[tuple[int, ...]] = []
    current: list[int] = []
    cumulative = 0.0
    for index, cost in enumerate(costs):
        current.append(index)
        cumulative += cost
        boundary = target * (len(rows) + 1)
        if cumulative >= boundary and len(rows) < MECHANICS_SHARD_COUNT - 1:
            rows.append(tuple(current))
            current = []
    if current:
        rows.append(tuple(current))
    if len(rows) != MECHANICS_SHARD_COUNT:
        raise ValueError("mechanics planning did not produce the sealed shard count")
    return tuple(rows)


D1_PARTITIONS = d1_partitions()
MECHANICS_PARTITIONS = mechanics_partitions()


def _final_d1_robustness_passed(value: Mapping[str, Any]) -> bool:
    if value.get("authorities", {}).get("d1", {}).get("frozen_pattern_repeated") is not True:
        return False
    cycles = value.get("fresh_d1_cycles")
    if not isinstance(cycles, dict) or not cycles:
        return False
    return all(
        isinstance(cycle, dict)
        and cycle.get("producer", {}).get("all_frozen_criteria_passed") is True
        and cycle.get("challenge", {}).get("all_frozen_criteria_passed") is True
        for cycle in cycles.values()
    )


def build_admission(final_path: Path = consolidated.DEFAULT_RESULT) -> dict[str, Any]:
    final_path = Path(final_path).resolve()
    final_value = _read_object(final_path)
    consolidated.validate_result(final_value)
    pruned = final_value.get("authorities", {}).get("successor_mechanics", {}).get("pruned_lanes")
    if not isinstance(pruned, list) or any(not isinstance(item, str) for item in pruned):
        raise ValueError("consolidated mechanics pruning authority is invalid")
    known_lanes = [lane.lane_id for lane in mechanics.LANES]
    if not set(pruned) <= set(known_lanes):
        raise ValueError("consolidated result prunes an unknown mechanics lane")
    core = {
        "schema": ADMISSION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "consolidated_authority": {
            "path": _relative(final_path),
            "file_sha256": sha256_file(final_path),
            "result_sha256": final_value["result_sha256"],
        },
        "epoch_ids": list(EPOCH_IDS),
        "fresh_cycle_indices": list(EPOCH_CYCLES),
        "d1_initially_eligible": _final_d1_robustness_passed(final_value),
        "mechanics_initially_eligible_lanes": [lane for lane in known_lanes if lane not in pruned],
        "mechanics_source_pruned_lanes": sorted(pruned),
        "boundary_rules": {
            "stable_d1_null_prunes_later_d1": True,
            "mixed_or_candidate_d1_continues": True,
            "mechanics_warning_prunes_lane": True,
            "past_or_active_epoch_mutation_allowed": False,
            "new_work_requires_append_only_supervisor_injection": True,
        },
        "planned_compute": {
            "epoch_serial_seconds": planned_epoch_compute_seconds(),
            "horizon_serial_seconds": planned_horizon_compute_seconds(),
            "horizon_idle_eight_worker_hours": planned_horizon_compute_seconds() / IDLE_WORKERS / 3_600,
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "admission_sha256": canonical_sha256(core)}


def validate_admission(value: Mapping[str, Any]) -> None:
    _validate_seal(value, "admission_sha256", "successor horizon admission")
    authority = value.get("consolidated_authority") or {}
    path_value = authority.get("path")
    if not isinstance(path_value, str):
        raise ValueError("successor horizon admission path is invalid")
    path = (REPO_ROOT / path_value).resolve()
    if (
        value.get("schema") != ADMISSION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_ids") != list(EPOCH_IDS)
        or value.get("fresh_cycle_indices") != list(EPOCH_CYCLES)
        or not path.is_file()
        or sha256_file(path) != authority.get("file_sha256")
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("successor horizon admission identity or authority drifted")
    final_value = _read_object(path)
    consolidated.validate_result(final_value)
    if final_value.get("result_sha256") != authority.get("result_sha256"):
        raise ValueError("successor horizon consolidated result binding drifted")
    pruned = final_value.get("authorities", {}).get("successor_mechanics", {}).get("pruned_lanes")
    known_lanes = [lane.lane_id for lane in mechanics.LANES]
    if (
        not isinstance(pruned, list)
        or value.get("d1_initially_eligible") != _final_d1_robustness_passed(final_value)
        or value.get("mechanics_source_pruned_lanes") != sorted(pruned)
        or value.get("mechanics_initially_eligible_lanes")
        != [lane for lane in known_lanes if lane not in pruned]
        or value.get("problems") != []
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("successor horizon admission routing authority drifted")


def admit(
    *,
    output: Path = DEFAULT_ADMISSION,
    final_path: Path = consolidated.DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.is_file():
        value = _read_object(output)
        validate_admission(value)
        return value
    value = build_admission(final_path)
    validate_admission(value)
    atomic_write_json(output, value)
    return value


def _classification_path(root: Path, epoch_index: int) -> Path:
    return root / "classifications" / f"{EPOCH_IDS[epoch_index].lower()}.json"


def _shard_path(root: Path, epoch_index: int, lane: str, shard_index: int) -> Path:
    return root / "shards" / EPOCH_IDS[epoch_index].lower() / f"{lane}_{shard_index:02d}.json"


def _previous_classification(root: Path, epoch_index: int) -> dict[str, Any] | None:
    if epoch_index == 0:
        return None
    value = _read_object(_classification_path(root, epoch_index - 1))
    validate_classification(value, epoch_index - 1, root=root)
    return value


def _d1_enabled(admission: Mapping[str, Any], previous: Mapping[str, Any] | None) -> bool:
    if admission.get("d1_initially_eligible") is not True:
        return False
    return previous is None or previous.get("routing", {}).get("continue_d1") is True


def _eligible_mechanics_lanes(admission: Mapping[str, Any], previous: Mapping[str, Any] | None) -> set[str]:
    initial = admission.get("mechanics_initially_eligible_lanes")
    if not isinstance(initial, list) or any(not isinstance(item, str) for item in initial):
        raise ValueError("horizon mechanics admission lanes are invalid")
    if previous is None:
        return set(initial)
    rows = previous.get("routing", {}).get("mechanics_lanes_for_next_epoch")
    if not isinstance(rows, list) or any(not isinstance(item, str) for item in rows):
        raise ValueError("previous horizon mechanics routing is invalid")
    return set(rows)


def _d1_work(epoch_index: int, source_index: int) -> consolidated.WorkItem:
    phase, local = d1.phase_of(source_index)
    return consolidated.WorkItem(
        key=f"{EPOCH_IDS[epoch_index].lower()}_d1_{phase}_{local:03d}",
        kind="d1_fresh",
        source_index=source_index,
        cycle=EPOCH_CYCLES[epoch_index],
        preflight=False,
    )


def _mechanics_work(epoch_index: int, source_index: int) -> consolidated.WorkItem:
    source = mechanics.WORK_ITEMS[source_index]
    return consolidated.WorkItem(
        key=f"{EPOCH_IDS[epoch_index].lower()}_{source.capsule_id}",
        kind="mechanics_fresh",
        source_index=source_index,
        cycle=EPOCH_CYCLES[epoch_index],
        preflight=False,
    )


def shard_work_items(*, epoch_index: int, lane: str, shard_index: int) -> tuple[consolidated.WorkItem, ...]:
    if not 0 <= epoch_index < len(EPOCH_IDS):
        raise ValueError("horizon epoch index is outside the sealed plan")
    if lane == "d1":
        if not 0 <= shard_index < D1_SHARD_COUNT:
            raise ValueError("D1 shard index is outside the sealed plan")
        return tuple(_d1_work(epoch_index, index) for index in D1_PARTITIONS[shard_index])
    if lane == "mechanics":
        if not 0 <= shard_index < MECHANICS_SHARD_COUNT:
            raise ValueError("mechanics shard index is outside the sealed plan")
        return tuple(_mechanics_work(epoch_index, index) for index in MECHANICS_PARTITIONS[shard_index])
    raise ValueError(f"unsupported horizon lane: {lane}")


def _effective_work_items(
    *,
    admission: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    works: Sequence[consolidated.WorkItem],
    lane: str,
) -> tuple[consolidated.WorkItem, ...]:
    if lane == "d1":
        return tuple(works) if _d1_enabled(admission, previous) else ()
    eligible_lanes = _eligible_mechanics_lanes(admission, previous)
    return tuple(work for work in works if mechanics.WORK_ITEMS[work.source_index].lane_id in eligible_lanes)


def _load_work_receipts(root: Path, works: Sequence[consolidated.WorkItem]) -> dict[str, dict[str, Any]]:
    return consolidated._load_receipts(root, works)


def _artifact_row(root: Path, work: consolidated.WorkItem, value: Mapping[str, Any]) -> dict[str, Any]:
    path = consolidated._artifact_path(root, work)
    return {
        "key": work.key,
        "source_index": work.source_index,
        "path": _relative(path),
        "file_sha256": sha256_file(path),
        "result_sha256": value.get("result_sha256"),
    }


def validate_shard(
    value: Mapping[str, Any],
    *,
    epoch_index: int,
    lane: str,
    shard_index: int,
    root: Path = DEFAULT_ROOT,
) -> None:
    _validate_seal(value, "shard_sha256", "successor horizon shard")
    if (
        value.get("schema") != SHARD_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_id") != EPOCH_IDS[epoch_index]
        or value.get("cycle_index") != EPOCH_CYCLES[epoch_index]
        or value.get("lane") != lane
        or value.get("shard_index") != shard_index
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("successor horizon shard identity or safety drifted")
    rows = value.get("artifact_index")
    if not isinstance(rows, list):
        raise ValueError("successor horizon shard artifact index is invalid")
    planned = shard_work_items(epoch_index=epoch_index, lane=lane, shard_index=shard_index)
    planned_by_source = {work.source_index: work for work in planned}
    seen_sources: set[int] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or isinstance(row.get("source_index"), bool)
            or not isinstance(row.get("source_index"), int)
        ):
            raise ValueError("successor horizon shard artifact row is invalid")
        source_index = int(row["source_index"])
        work = planned_by_source.get(source_index)
        if work is None or source_index in seen_sources:
            raise ValueError("successor horizon shard artifact inventory drifted")
        path = (REPO_ROOT / row["path"]).resolve()
        expected_path = consolidated._artifact_path(Path(root).resolve(), work).resolve()
        if path != expected_path or not path.is_file() or sha256_file(path) != row.get("file_sha256"):
            raise ValueError("successor horizon shard artifact binding drifted")
        artifact = _read_object(path)
        consolidated._validate_artifact(artifact, work)
        if artifact.get("result_sha256") != row.get("result_sha256"):
            raise ValueError("successor horizon shard raw result binding drifted")
        seen_sources.add(source_index)
    executed = value.get("executed_item_count")
    skipped = value.get("skipped_item_count")
    if (
        value.get("planned_item_count") != len(planned)
        or executed != len(rows)
        or isinstance(skipped, bool)
        or not isinstance(skipped, int)
        or executed + skipped != len(planned)
    ):
        raise ValueError("successor horizon shard counts drifted")


def _dynamic_pool_width(pending_count: int) -> tuple[int, int | None]:
    """Size the process pool from live host load, floating in ``[1, min(IDLE_WORKERS, pending)]``.

    The pool width is a WALL-TIME lever only: it is passed to ``ProcessPoolExecutor(max_workers=...)``
    and is never written to any receipt, so every seeded-sha256 capsule result is byte-identical at
    any width. ``IDLE_WORKERS`` (20) is the declared ceiling; the dynamic worker controller floats the
    instantaneous width beneath it with the live host state, backing off fast under the external
    Hawking workload and ramping back toward the ceiling when the host is idle. The controller is
    imported lazily so the wave pays no import cost at module load. If it cannot sample a live host
    (psutil absent or a refusal) the width falls back to the static ``min(IDLE_WORKERS, pending)``
    exactly as before, so the wave never fails to run. Returns ``(width, nice_level)`` where
    ``nice_level`` is the controller's advisory worker priority, or ``None`` when no live sample was
    available (the fallback path applies no initializer).
    """

    static_cap = min(IDLE_WORKERS, pending_count)
    if static_cap <= 1:
        return max(1, static_cap), None
    try:
        from mop.studio import dynamic_worker_controller as controller

        sample = controller.sample_host_state(current_workers=static_cap)
        width = controller.recommended_workers(sample.state)
        nice_level = controller.recommended_priority(sample.state).nice_level
    except Exception:
        return static_cap, None
    return max(1, min(int(width), static_cap)), int(nice_level)


def _pool_worker_priority_init(nice_level: int) -> None:
    """Best-effort worker priority: nudge each pool worker toward the advised nice level.

    Never fatal and never touches a receipt: OS scheduling priority does not enter the seeded sha256
    capsule result, so this only changes how the surviving pool yields cores to the external Hawking
    workload by priority. Any failure is swallowed.
    """

    with suppress(Exception):
        os.nice(int(nice_level))


def run_shard(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
    epoch_index: int,
    lane: str,
    shard_index: int,
    idle_workers: int = IDLE_WORKERS,
    hawking_workers: int = HAWKING_WORKERS,
    retry_limit: int = RETRY_LIMIT,
) -> dict[str, Any]:
    set_process_label(f"mop-g1-horizon-{EPOCH_IDS[epoch_index].lower()}-{lane}-{shard_index:02d}")
    root = Path(root).resolve()
    admission_path = Path(admission_path).resolve()
    if not root.is_relative_to(REPO_ROOT.resolve()) or not admission_path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("successor horizon paths must remain inside the repository")
    if min(idle_workers, hawking_workers, retry_limit) < 1:
        raise ValueError("successor horizon worker and retry counts must be positive")
    admission_value = _read_object(admission_path)
    validate_admission(admission_value)
    shard_path = _shard_path(root, epoch_index, lane, shard_index)
    if shard_path.is_file():
        existing = _read_object(shard_path)
        validate_shard(
            existing,
            epoch_index=epoch_index,
            lane=lane,
            shard_index=shard_index,
            root=root,
        )
        return existing
    previous = _previous_classification(root, epoch_index)
    planned = shard_work_items(epoch_index=epoch_index, lane=lane, shard_index=shard_index)
    effective = _effective_work_items(
        admission=admission_value,
        previous=previous,
        works=planned,
        lane=lane,
    )
    receipts = _load_work_receipts(root, effective)
    attempts: dict[str, int] = {}
    durations: dict[str, float] = {}
    first_wave = not receipts
    while len(receipts) < len(effective):
        pending = [work for work in effective if work.key not in receipts]
        # Size the pool dynamically from live host load. The width and priority are wall-time levers
        # only and never enter any shard receipt, so every seeded-sha256 capsule result is
        # byte-identical at any width; the sealed idle_workers/hawking_workers fields stay fixed.
        width, nice_level = _dynamic_pool_width(len(pending))
        wave = pending[: 1 if first_wave else width]
        first_wave = False
        initializer = _pool_worker_priority_init if nice_level is not None else None
        initargs = (int(nice_level),) if nice_level is not None else ()
        with ProcessPoolExecutor(max_workers=len(wave), initializer=initializer, initargs=initargs) as pool:
            futures = {}
            for work in wave:
                attempts[work.key] = attempts.get(work.key, 0) + 1
                futures[pool.submit(consolidated._execute_work, work, str(root))] = work
            for future in as_completed(futures):
                work = futures[future]
                try:
                    key, duration = future.result()
                    path = consolidated._artifact_path(root, work)
                    value = _read_object(path)
                    consolidated._validate_artifact(value, work)
                    receipts[key] = value
                    durations[key] = duration
                except Exception:  # noqa: BLE001 - exact retry boundary is sealed by the supervisor
                    if attempts[work.key] >= retry_limit:
                        raise
    artifact_index = [_artifact_row(root, work, receipts[work.key]) for work in effective]
    core = {
        "schema": SHARD_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "epoch_id": EPOCH_IDS[epoch_index],
        "cycle_index": EPOCH_CYCLES[epoch_index],
        "lane": lane,
        "shard_index": shard_index,
        "planned_item_count": len(planned),
        "executed_item_count": len(effective),
        "skipped_item_count": len(planned) - len(effective),
        "skip_reason": (
            None if effective else "branch_pruned_by_sealed_prior_classification_or_consolidated_authority"
        ),
        "artifact_index": artifact_index,
        "execution": {
            "idle_workers": idle_workers,
            "hawking_workers": hawking_workers,
            "observed_item_seconds": sum(durations.values()),
            "average_observed_item_seconds": statistics.fmean(durations.values()) if durations else None,
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    result = {**core, "shard_sha256": canonical_sha256(core)}
    validate_shard(
        result,
        epoch_index=epoch_index,
        lane=lane,
        shard_index=shard_index,
        root=root,
    )
    atomic_write_json(shard_path, result)
    return result


def _load_epoch_shards(root: Path, epoch_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane, count in (("d1", D1_SHARD_COUNT), ("mechanics", MECHANICS_SHARD_COUNT)):
        for shard_index in range(count):
            value = _read_object(_shard_path(root, epoch_index, lane, shard_index))
            validate_shard(
                value,
                epoch_index=epoch_index,
                lane=lane,
                shard_index=shard_index,
                root=root,
            )
            rows.append(value)
    return rows


def _d1_epoch_summary(root: Path, epoch_index: int, shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    d1_rows = [row for row in shards if row.get("lane") == "d1"]
    executed = sum(int(row["executed_item_count"]) for row in d1_rows)
    if executed == 0:
        return {
            "classification": "not_run_pruned",
            "terminal_route": "blocked",
            "producer": None,
            "challenge": None,
            "continue_d1": False,
        }
    if executed != d1.DEFAULT_RUNG_COUNT:
        raise ValueError("fresh D1 epoch is partially executed")
    accumulators = {
        "producer": _D1PhaseAccumulator(),
        "challenge": _D1PhaseAccumulator(),
    }
    for source_index in range(d1.DEFAULT_RUNG_COUNT):
        work = _d1_work(epoch_index, source_index)
        value = _read_object(consolidated._artifact_path(root, work))
        consolidated._validate_artifact(value, work)
        phase, _ = d1.phase_of(source_index)
        accumulators[phase].add(value)
    producer = accumulators["producer"].result()
    challenge = accumulators["challenge"].result()
    producer_pass = producer["all_frozen_criteria_passed"] is True
    challenge_pass = challenge["all_frozen_criteria_passed"] is True
    if producer_pass and challenge_pass:
        classification, route, keep = "stable_candidate_trace", "positive", True
    elif producer_pass != challenge_pass:
        classification, route, keep = "mixed_or_seed_sensitive", "mixed_or_seed_sensitive", True
    else:
        classification, route, keep = "stable_null", "null", False
    return {
        "classification": classification,
        "terminal_route": route,
        "producer": producer,
        "challenge": challenge,
        "continue_d1": keep,
    }


def _mechanics_epoch_summary(
    root: Path,
    epoch_index: int,
    admission: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    eligible = _eligible_mechanics_lanes(admission, previous)
    lanes: dict[str, Any] = {}
    retained: list[str] = []
    for lane in mechanics.LANES:
        if lane.lane_id not in eligible:
            lanes[lane.lane_id] = {
                "classification": "not_run_pruned",
                "terminal_route": "blocked",
                "receipt_count": 0,
                "verdict_counts": {},
                "continue_lane": False,
            }
            continue
        verdict_counts: dict[str, int] = {}
        receipt_count = 0
        lane_works = [
            _mechanics_work(epoch_index, item.index)
            for item in mechanics.WORK_ITEMS
            if item.lane_id == lane.lane_id
        ]
        for work in lane_works:
            value = _read_object(consolidated._artifact_path(root, work))
            consolidated._validate_artifact(value, work)
            receipt_count += int(value["receipt_count"])
            for verdict, count in value["verdict_counts"].items():
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + int(count)
        clean = verdict_counts == {"mechanics-ok": receipt_count}
        if clean:
            retained.append(lane.lane_id)
        lanes[lane.lane_id] = {
            "classification": "mechanics_noninferential" if clean else "mechanics_warning",
            "terminal_route": "descriptive_only" if clean else "null",
            "receipt_count": receipt_count,
            "verdict_counts": verdict_counts,
            "continue_lane": clean,
        }
    return lanes, retained


def build_classification(*, root: Path, admission_path: Path, epoch_index: int) -> dict[str, Any]:
    root = Path(root).resolve()
    admission = _read_object(Path(admission_path).resolve())
    validate_admission(admission)
    previous = _previous_classification(root, epoch_index)
    shards = _load_epoch_shards(root, epoch_index)
    d1_summary = _d1_epoch_summary(root, epoch_index, shards)
    mechanics_summary, retained_lanes = _mechanics_epoch_summary(root, epoch_index, admission, previous)
    core = {
        "schema": CLASSIFICATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "epoch_id": EPOCH_IDS[epoch_index],
        "epoch_index": epoch_index,
        "cycle_index": EPOCH_CYCLES[epoch_index],
        "admission_sha256": admission["admission_sha256"],
        "parent_classification_sha256": (
            previous.get("classification_sha256") if previous is not None else None
        ),
        "shard_bindings": [
            {
                "lane": row["lane"],
                "shard_index": row["shard_index"],
                "path": _relative(_shard_path(root, epoch_index, row["lane"], row["shard_index"])),
                "file_sha256": sha256_file(_shard_path(root, epoch_index, row["lane"], row["shard_index"])),
                "shard_sha256": row["shard_sha256"],
            }
            for row in shards
        ],
        "d1": d1_summary,
        "mechanics": mechanics_summary,
        "routing": {
            "continue_d1": d1_summary["continue_d1"],
            "mechanics_lanes_for_next_epoch": retained_lanes,
            "past_or_active_work_mutable": False,
            "future_change_requires_new_sealed_child": True,
        },
        "interpretation_limit": (
            "The classification routes later robustness work only. It cannot convert same-code "
            "fresh seeds or mechanics receipts into independent confirmation."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "classification_sha256": canonical_sha256(core)}


def validate_classification(value: Mapping[str, Any], epoch_index: int, *, root: Path = DEFAULT_ROOT) -> None:
    _validate_seal(value, "classification_sha256", "successor horizon classification")
    if (
        value.get("schema") != CLASSIFICATION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_id") != EPOCH_IDS[epoch_index]
        or value.get("epoch_index") != epoch_index
        or value.get("cycle_index") != EPOCH_CYCLES[epoch_index]
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("successor horizon classification identity or safety drifted")
    d1_row = value.get("d1") or {}
    if d1_row.get("classification") not in {
        "stable_candidate_trace",
        "stable_null",
        "mixed_or_seed_sensitive",
        "not_run_pruned",
    }:
        raise ValueError("successor horizon D1 classification is invalid")
    expected_route = {
        "stable_candidate_trace": ("positive", True),
        "stable_null": ("null", False),
        "mixed_or_seed_sensitive": ("mixed_or_seed_sensitive", True),
        "not_run_pruned": ("blocked", False),
    }[d1_row["classification"]]
    routing = value.get("routing") or {}
    if (
        d1_row.get("terminal_route") != expected_route[0]
        or d1_row.get("continue_d1") is not expected_route[1]
        or routing.get("continue_d1") is not expected_route[1]
        or routing.get("past_or_active_work_mutable") is not False
        or routing.get("future_change_requires_new_sealed_child") is not True
    ):
        raise ValueError("successor horizon D1 terminal routing drifted")
    mechanics_rows = value.get("mechanics")
    known_lanes = [lane.lane_id for lane in mechanics.LANES]
    retained = routing.get("mechanics_lanes_for_next_epoch")
    if (
        not isinstance(mechanics_rows, dict)
        or set(mechanics_rows) != set(known_lanes)
        or not isinstance(retained, list)
        or retained != [lane for lane in known_lanes if mechanics_rows[lane].get("continue_lane")]
    ):
        raise ValueError("successor horizon mechanics routing drifted")
    bindings = value.get("shard_bindings")
    expected_identities = {
        (lane, shard)
        for lane, count in (("d1", D1_SHARD_COUNT), ("mechanics", MECHANICS_SHARD_COUNT))
        for shard in range(count)
    }
    if not isinstance(bindings, list) or len(bindings) != len(expected_identities):
        raise ValueError("successor horizon classification shard inventory drifted")
    seen: set[tuple[str, int]] = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("successor horizon classification shard binding is invalid")
        identity = (binding.get("lane"), binding.get("shard_index"))
        if identity not in expected_identities or identity in seen:
            raise ValueError("successor horizon classification shard identity drifted")
        lane, shard_index = identity
        path = _shard_path(Path(root).resolve(), epoch_index, str(lane), int(shard_index))
        if (
            binding.get("path") != _relative(path)
            or not path.is_file()
            or binding.get("file_sha256") != sha256_file(path)
        ):
            raise ValueError("successor horizon classification shard file binding drifted")
        shard = _read_object(path)
        validate_shard(
            shard,
            epoch_index=epoch_index,
            lane=str(lane),
            shard_index=int(shard_index),
            root=root,
        )
        if binding.get("shard_sha256") != shard.get("shard_sha256"):
            raise ValueError("successor horizon classification shard payload drifted")
        seen.add((str(lane), int(shard_index)))
    previous_hash = None
    if epoch_index:
        previous = _read_object(_classification_path(Path(root).resolve(), epoch_index - 1))
        validate_classification(previous, epoch_index - 1, root=root)
        previous_hash = previous["classification_sha256"]
    if value.get("parent_classification_sha256") != previous_hash:
        raise ValueError("successor horizon classification ancestry drifted")


def classify_epoch(
    *, root: Path = DEFAULT_ROOT, admission_path: Path = DEFAULT_ADMISSION, epoch_index: int
) -> dict[str, Any]:
    root = Path(root).resolve()
    target = _classification_path(root, epoch_index)
    if target.is_file():
        existing = _read_object(target)
        validate_classification(existing, epoch_index, root=root)
        return existing
    result = build_classification(root=root, admission_path=admission_path, epoch_index=epoch_index)
    validate_classification(result, epoch_index, root=root)
    atomic_write_json(target, result)
    return result


def build_result(*, root: Path = DEFAULT_ROOT, admission_path: Path = DEFAULT_ADMISSION) -> dict[str, Any]:
    root = Path(root).resolve()
    admission_path = Path(admission_path).resolve()
    admission = _read_object(admission_path)
    validate_admission(admission)
    classifications = []
    shard_bindings = []
    for epoch_index in range(len(EPOCH_IDS)):
        classification = _read_object(_classification_path(root, epoch_index))
        validate_classification(classification, epoch_index, root=root)
        classifications.append(classification)
        for lane, count in (("d1", D1_SHARD_COUNT), ("mechanics", MECHANICS_SHARD_COUNT)):
            for shard_index in range(count):
                path = _shard_path(root, epoch_index, lane, shard_index)
                shard = _read_object(path)
                validate_shard(
                    shard,
                    epoch_index=epoch_index,
                    lane=lane,
                    shard_index=shard_index,
                    root=root,
                )
                shard_bindings.append(
                    {
                        "epoch_id": EPOCH_IDS[epoch_index],
                        "lane": lane,
                        "shard_index": shard_index,
                        "path": _relative(path),
                        "file_sha256": sha256_file(path),
                        "shard_sha256": shard["shard_sha256"],
                    }
                )
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "admission": {
            "path": _relative(admission_path),
            "file_sha256": sha256_file(admission_path),
            "admission_sha256": admission["admission_sha256"],
        },
        "grid": {
            "epoch_count": len(EPOCH_IDS),
            "d1_shard_count": len(EPOCH_IDS) * D1_SHARD_COUNT,
            "mechanics_shard_count": len(EPOCH_IDS) * MECHANICS_SHARD_COUNT,
            "executed_d1_rung_count": sum(
                row["executed_item_count"]
                for row in map(
                    _read_object,
                    [
                        _shard_path(root, epoch, "d1", shard)
                        for epoch in range(len(EPOCH_IDS))
                        for shard in range(D1_SHARD_COUNT)
                    ],
                )
            ),
            "executed_mechanics_rung_count": sum(
                row["executed_item_count"]
                for row in map(
                    _read_object,
                    [
                        _shard_path(root, epoch, "mechanics", shard)
                        for epoch in range(len(EPOCH_IDS))
                        for shard in range(MECHANICS_SHARD_COUNT)
                    ],
                )
            ),
        },
        "classifications": [
            {
                "epoch_id": row["epoch_id"],
                "cycle_index": row["cycle_index"],
                "d1_classification": row["d1"]["classification"],
                "d1_terminal_route": row["d1"]["terminal_route"],
                "mechanics_lanes_for_next_epoch": row["routing"]["mechanics_lanes_for_next_epoch"],
                "path": _relative(_classification_path(root, index)),
                "file_sha256": sha256_file(_classification_path(root, index)),
                "classification_sha256": row["classification_sha256"],
            }
            for index, row in enumerate(classifications)
        ],
        "shard_index": shard_bindings,
        "compute_plan": {
            "planned_serial_seconds": planned_horizon_compute_seconds(),
            "planned_idle_eight_worker_hours": planned_horizon_compute_seconds() / IDLE_WORKERS / 3_600,
        },
        "decision": {
            "bounded_horizon_complete": True,
            "independent_artifact_verification_pending": True,
            "independent_scientific_confirmation": False,
            "next_action": "run_successor_horizon_independent_artifact_verifier",
        },
        "interpretation_limit": (
            "Five disjoint same-code epochs extend robustness and seed-sensitivity evidence. "
            "They do not supply a separately implemented generator or scientific confirmation."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_result(value: Mapping[str, Any], *, root: Path = DEFAULT_ROOT) -> None:
    _validate_seal(value, "result_sha256", "successor horizon result")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("grid", {}).get("epoch_count") != len(EPOCH_IDS)
        or value.get("decision", {}).get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("successor horizon result identity or safety drifted")
    classifications = value.get("classifications")
    if not isinstance(classifications, list) or len(classifications) != len(EPOCH_IDS):
        raise ValueError("successor horizon result classification inventory drifted")
    for epoch_index, binding in enumerate(classifications):
        if not isinstance(binding, dict):
            raise ValueError("successor horizon result classification binding is invalid")
        path = _classification_path(Path(root).resolve(), epoch_index)
        classification = _read_object(path)
        validate_classification(classification, epoch_index, root=root)
        if (
            binding.get("epoch_id") != EPOCH_IDS[epoch_index]
            or binding.get("cycle_index") != EPOCH_CYCLES[epoch_index]
            or binding.get("path") != _relative(path)
            or binding.get("file_sha256") != sha256_file(path)
            or binding.get("classification_sha256") != classification.get("classification_sha256")
        ):
            raise ValueError("successor horizon result classification binding drifted")
    shard_rows = value.get("shard_index")
    expected_shards = {
        (EPOCH_IDS[epoch], lane, shard)
        for epoch in range(len(EPOCH_IDS))
        for lane, count in (("d1", D1_SHARD_COUNT), ("mechanics", MECHANICS_SHARD_COUNT))
        for shard in range(count)
    }
    if not isinstance(shard_rows, list) or len(shard_rows) != len(expected_shards):
        raise ValueError("successor horizon result shard inventory drifted")
    seen: set[tuple[str, str, int]] = set()
    executed = {"d1": 0, "mechanics": 0}
    for binding in shard_rows:
        if not isinstance(binding, dict):
            raise ValueError("successor horizon result shard binding is invalid")
        identity = (
            binding.get("epoch_id"),
            binding.get("lane"),
            binding.get("shard_index"),
        )
        if identity not in expected_shards or identity in seen:
            raise ValueError("successor horizon result shard identity drifted")
        epoch_id, lane, shard_index = identity
        epoch_index = EPOCH_IDS.index(str(epoch_id))
        path = _shard_path(Path(root).resolve(), epoch_index, str(lane), int(shard_index))
        shard = _read_object(path)
        validate_shard(
            shard,
            epoch_index=epoch_index,
            lane=str(lane),
            shard_index=int(shard_index),
            root=root,
        )
        if (
            binding.get("path") != _relative(path)
            or binding.get("file_sha256") != sha256_file(path)
            or binding.get("shard_sha256") != shard.get("shard_sha256")
        ):
            raise ValueError("successor horizon result shard binding drifted")
        executed[str(lane)] += int(shard["executed_item_count"])
        seen.add((str(epoch_id), str(lane), int(shard_index)))
    grid = value.get("grid") or {}
    if (
        grid.get("d1_shard_count") != len(EPOCH_IDS) * D1_SHARD_COUNT
        or grid.get("mechanics_shard_count") != len(EPOCH_IDS) * MECHANICS_SHARD_COUNT
        or grid.get("executed_d1_rung_count") != executed["d1"]
        or grid.get("executed_mechanics_rung_count") != executed["mechanics"]
    ):
        raise ValueError("successor horizon result grid drifted")


def aggregate(
    *,
    root: Path = DEFAULT_ROOT,
    admission_path: Path = DEFAULT_ADMISSION,
    output: Path = DEFAULT_RESULT,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.is_file():
        existing = _read_object(output)
        validate_result(existing, root=root)
        return existing
    result = build_result(root=root, admission_path=admission_path)
    validate_result(result, root=root)
    atomic_write_json(output, result)
    return result


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_verification_for_report(value: Mapping[str, Any], *, result_path: Path) -> None:
    core = {key: item for key, item in value.items() if key != "verification_sha256"}
    source = value.get("source") or {}
    if (
        value.get("verification_sha256") != canonical_sha256(core)
        or value.get("schema") != "mop-generation1-successor-horizon-verification/v1"
        or value.get("program_id") != PROGRAM_ID
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
    ):
        raise ValueError("successor horizon verification is not a clean bound authority")


def validate_report_receipt(value: Mapping[str, Any]) -> None:
    _validate_seal(value, "receipt_sha256", "successor horizon report receipt")
    if (
        value.get("schema") != REPORT_RECEIPT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("successor horizon report receipt identity or safety drifted")
    for label in ("result", "verification", "report"):
        binding = value.get(label) or {}
        path_value = binding.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"successor horizon {label} report binding is invalid")
        path = (REPO_ROOT / path_value).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"successor horizon {label} report path drifted")
        if binding.get("file_sha256") != sha256_file(path):
            raise ValueError(f"successor horizon {label} report hash drifted")


def render_report(
    *,
    result_path: Path = DEFAULT_RESULT,
    verification_path: Path = DEFAULT_VERIFICATION,
    report_path: Path = DEFAULT_REPORT,
    receipt_path: Path = DEFAULT_REPORT_RECEIPT,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    verification_path = Path(verification_path).resolve()
    report_path = Path(report_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    result = _read_object(result_path)
    validate_result(result)
    verification = _read_object(verification_path)
    _validate_verification_for_report(verification, result_path=result_path)
    if receipt_path.is_file():
        existing = _read_object(receipt_path)
        validate_report_receipt(existing)
        return existing
    lines = [
        "# Generation 1 successor evidence report",
        "",
        (
            "This is a generated operational and evidence-classification view. "
            "Sealed JSON remains authoritative."
        ),
        "",
        "| Epoch | Fresh cycle | D1 classification | Terminal route | Mechanics lanes retained |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in result["classifications"]:
        lines.append(
            f"| {row['epoch_id']} | {row['cycle_index']} | {row['d1_classification']} | "
            f"{row['d1_terminal_route']} | {len(row['mechanics_lanes_for_next_epoch'])} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        (
            "The five epochs are same-code robustness evidence. The separate verifier "
            "reconstructs receipt and aggregation integrity. Neither artifact is independent "
            "scientific confirmation, activation authority, or Stage 3 promotion."
        ),
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
        f"Verification SHA-256: `{verification.get('verification_sha256')}`",
        "",
    ]
    encoded = ("\n".join(lines)).encode("utf-8")
    _atomic_write_text(report_path, encoded.decode("utf-8"))
    report_sha = hashlib.sha256(encoded).hexdigest()
    core = {
        "schema": REPORT_RECEIPT_SCHEMA,
        "program_id": PROGRAM_ID,
        "result": {"path": _relative(result_path), "file_sha256": sha256_file(result_path)},
        "verification": {
            "path": _relative(verification_path),
            "file_sha256": sha256_file(verification_path),
        },
        "report": {"path": _relative(report_path), "file_sha256": report_sha},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    validate_report_receipt(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "ADMISSION_SCHEMA",
    "CLASSIFICATION_SCHEMA",
    "DEFAULT_ADMISSION",
    "DEFAULT_REPORT",
    "DEFAULT_REPORT_RECEIPT",
    "DEFAULT_RESULT",
    "DEFAULT_ROOT",
    "DEFAULT_VERIFICATION",
    "D1_PARTITIONS",
    "D1_SHARD_COUNT",
    "EPOCH_CYCLES",
    "EPOCH_IDS",
    "MECHANICS_PARTITIONS",
    "MECHANICS_SHARD_COUNT",
    "PROGRAM_ID",
    "REPORT_RECEIPT_SCHEMA",
    "RESULT_SCHEMA",
    "SHARD_SCHEMA",
    "admit",
    "aggregate",
    "build_admission",
    "build_classification",
    "build_result",
    "classify_epoch",
    "planned_epoch_compute_seconds",
    "planned_horizon_compute_seconds",
    "render_report",
    "run_shard",
    "shard_work_items",
    "validate_admission",
    "validate_classification",
    "validate_result",
    "validate_report_receipt",
    "validate_shard",
]
