
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.ladder.stage3_registry import build_pair
from mop.mechanisms.construction_search_vec_runner import ConstructionSearchVecRunner
from mop.process_labels import set_process_label
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_d1_redesign_v2 as redesign_v2
from mop.studies import generation1_new_mechanisms_queue as newq
from mop.studies import generation1_successor_categorized_batch_wave as predecessor
from mop.studies import generation1_successor_categorized_batch_wave_verify as predecessor_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio import generation1_supervisor as supervisor
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256, sha256_file

PROGRAM_ID = "generation1-full-generations-wave-v1"
CLAIM_SCOPE = (
    "full-generations deterministic successor mechanics stress across carried and redesigned lanes "
    "organized by category; no independent scientific confirmation, activation, or promotion"
)
VERIFICATION_CLAIM_SCOPE = (
    "independent artifact verification of carried categorized-wave authority, new mechanism-lane "
    "admission, categorized fresh-cycle mechanics receipts, non-resurrecting routing, single "
    "substituted-dependency I1 integration, advisory release audit, and mutations"
)

GATE_SCHEMA = "mop-generation1-full-generations-gate/v1"
CATEGORY_SCHEMA = "mop-generation1-full-generations-category-result/v1"
CLASSIFICATION_SCHEMA = "mop-generation1-full-generations-classification/v1"
INTEGRATION_SCHEMA = "mop-generation1-full-generations-integration-result/v1"
INTEGRATION_CLASSIFICATION_SCHEMA = "mop-generation1-full-generations-integration-classification/v1"
RESULT_SCHEMA = "mop-generation1-full-generations-result/v1"
VERIFICATION_SCHEMA = "mop-generation1-full-generations-verification/v1"
REPORT_RECEIPT_SCHEMA = "mop-generation1-full-generations-report-receipt/v1"
RELEASE_AUDIT_SCHEMA = "mop-generation1-full-generations-release-audit/v1"

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_FULL_GENERATIONS_WAVE.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_FULL_GENERATIONS_WAVE.verification.json"
DEFAULT_REPORT = REPO_ROOT / "runs/generation1/GENERATION1_FULL_GENERATIONS_WAVE.md"
DEFAULT_REPORT_RECEIPT = DEFAULT_ROOT / "report_receipt.json"
DEFAULT_RELEASE_AUDIT = DEFAULT_ROOT / "release_audit_advisory.json"
PROGRAM_MANIFEST = REPO_ROOT / "configs/campaign/generation1_full_generations_wave_v1.json"
DEFAULT_PARENT_RESULT = predecessor.DEFAULT_RESULT
DEFAULT_PARENT_VERIFICATION = predecessor.DEFAULT_VERIFICATION
DEFAULT_PARENT_REPORT_RECEIPT = predecessor.DEFAULT_REPORT_RECEIPT

GATE_IDS = (
    "admit_wave_v1",
    "carry_d1_freeze",
    "rescreen_d1_redesign",
    "admit_new_lanes",
    "freeze_routing",
)
EPOCH_IDS = (
    "W08",
    "W09",
    "W10",
    "W11",
    "W12",
    "W13",
    "W14",
    "W15",
    "W16",
    "W17",
    "W18",
    "W19",
    "W20",
    "W21",
)
EPOCH_CYCLES = (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
INTERNAL_SHARD_COUNT = 8
IDLE_WORKERS = 20
CAPSULE_COUNT = 123
RETRY_LIMIT = 3
I1_LANE_ID = "G1-I1"

I1_DEPENDENCIES = ("G1-E1", "G1-D1", "G1-M1", "G1-V1", "G1-R1", "G1-P1R")

_OLD_ORIGIN = "consolidated"
_NEW_ORIGIN = "new_mechanisms"
_NEW_ARTIFACT_KIND = "new_mechanics_fresh"

_REPO_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_MECHANISM_DIRNAME = "src/mop/mechanisms"

_CONSTRUCTION_MECHANISM = "construction_search"

CONSTRUCTION_VEC_SPEEDUP = 6.7


@dataclass(frozen=True, slots=True)
class WaveWorkItem:

    key: str
    origin: str
    source_index: int
    cycle: int


@dataclass(frozen=True)
class Category:
    category_id: str
    name: str
    order: int
    lane_ids: tuple[str, ...]


@dataclass(frozen=True)
class _WaveValidationContext:
    predecessor_binding: dict[str, Any]
    routing: dict[str, dict[str, Any]]
    parent_classification_sha256: str | None


CATEGORIES = (
    Category("formation_trace", "Formation and trace", 0, ("G1-C0", "G1-E1")),
    Category(
        "communication_repair",
        "Communication and repair",
        1,
        ("G1-V1", "G1-M1", "G1-K1"),
    ),
    Category("memory_plasticity", "Memory and plasticity", 2, ("G1-R1", "G1-P1R")),
    Category("action_simulation", "Action and simulation", 3, ("G1-A1", "G1-S1")),
    Category("construction", "Construction", 4, ("G1-G1",)),
    Category(
        "dispatch_redesign",
        "Dispatch mechanics and gated D1-v2 efficacy",
        5,
        ("G1-D1",),
    ),
    Category("uncertainty_curiosity", "Uncertainty and curiosity", 6, ("G1-U1", "G1-N1")),
)
CATEGORY_IDS = tuple(category.category_id for category in CATEGORIES)
CATEGORY_NAMES = {category.category_id: category.name for category in CATEGORIES}
CATEGORY_LANES = {category.category_id: category.lane_ids for category in CATEGORIES}
_CATEGORY_BY_ID = {category.category_id: category for category in CATEGORIES}
_OLD_LANE_BY_ID = {lane.lane_id: lane for lane in mechanics.LANES}
_NEW_LANE_BY_ID = {lane.lane_id: lane for lane in newq.NEW_LANES}
_I1_SPEC = _OLD_LANE_BY_ID[I1_LANE_ID]
NEW_LANE_IDS = tuple(lane.lane_id for lane in newq.NEW_LANES)
_CATEGORY_LANE_IDS = frozenset(lane for category in CATEGORIES for lane in category.lane_ids)

_CANARY_POLICY = {
    "canary_epoch_id": EPOCH_IDS[0],
    "canary_epoch_index": 0,
    "canary_cycle_index": EPOCH_CYCLES[0],
    "rule": (
        "each new lane W08 canary partition must be all mechanics-ok or the lane is pruned from W09 onward"
    ),
    "enforced_by": "serial_classifier_null_safe_pruning",
}

_OLD_CATEGORY_ITEM_COUNT = len([item for item in mechanics.WORK_ITEMS if item.lane_id in _CATEGORY_LANE_IDS])
_NEW_CATEGORY_ITEM_COUNT = len([item for item in newq.NEW_WORK_ITEMS if item.lane_id in _CATEGORY_LANE_IDS])
_ITEMS_PER_EPOCH = _OLD_CATEGORY_ITEM_COUNT + _NEW_CATEGORY_ITEM_COUNT
_I1_ITEM_COUNT = len([item for item in mechanics.WORK_ITEMS if item.lane_id == I1_LANE_ID])
MAXIMUM_RAW_RECEIPT_COUNT = _ITEMS_PER_EPOCH * len(EPOCH_IDS) + _I1_ITEM_COUNT


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON authority must be a regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError("artifact path is outside the repository") from exc


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is not repository-relative")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"{label} path escapes the repository")
    return path


def _validate_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(core):
        raise ValueError(f"{label} seal drifted")


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{label} fields drifted")


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} timestamp is invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{label} timestamp is not UTC")


def _binding(path: Path, payload: Mapping[str, Any], seal_field: str) -> dict[str, Any]:
    if not isinstance(payload.get(seal_field), str):
        raise ValueError(f"bound artifact does not contain {seal_field}: {path}")
    return {
        "path": _relative(path),
        "file_sha256": sha256_file(path),
        seal_field: payload[seal_field],
    }


def _load_binding(binding: Any, seal_field: str, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256", seal_field}:
        raise ValueError(f"{label} binding fields drifted")
    path = _repo_path(binding.get("path"), label)
    payload = _read_object(path)
    _validate_seal(payload, seal_field, label)
    if binding.get("file_sha256") != sha256_file(path) or binding.get(seal_field) != payload.get(seal_field):
        raise ValueError(f"{label} binding drifted")
    return path, payload


def _gate_path(root: Path, gate_id: str) -> Path:
    return Path(root).resolve() / "gates" / f"{gate_id}.json"


def _category_path(root: Path, epoch_index: int, category_id: str) -> Path:
    return Path(root).resolve() / "waves" / EPOCH_IDS[epoch_index].lower() / f"{category_id}.json"


def _category_raw_root(root: Path, epoch_index: int, category_id: str) -> Path:
    return Path(root).resolve() / "raw" / EPOCH_IDS[epoch_index].lower() / category_id


def _classification_path(root: Path, epoch_index: int) -> Path:
    return Path(root).resolve() / "classifications" / f"{EPOCH_IDS[epoch_index].lower()}.json"


def _integration_path(root: Path) -> Path:
    return Path(root).resolve() / "integration" / "i1.json"


def _integration_raw_root(root: Path) -> Path:
    return Path(root).resolve() / "raw" / "i1"


def _integration_classification_path(root: Path) -> Path:
    return Path(root).resolve() / "integration" / "i1_classification.json"


def _source_item(work: WaveWorkItem) -> mechanics.WorkItem:
    table = newq.NEW_WORK_ITEMS if work.origin == _NEW_ORIGIN else mechanics.WORK_ITEMS
    return table[work.source_index]


def _work_lane_id(work: WaveWorkItem) -> str:
    return str(_source_item(work).lane_id)


def _work_mechanism(work: WaveWorkItem) -> str:
    return str(_source_item(work).mechanism)


def _consolidated_work(work: WaveWorkItem) -> consolidated.WorkItem:
    return consolidated.WorkItem(
        key=work.key,
        kind="mechanics_fresh",
        source_index=work.source_index,
        cycle=work.cycle,
        preflight=False,
    )


def _fresh_new_item(work: WaveWorkItem) -> mechanics.WorkItem:

    source = newq.NEW_WORK_ITEMS[work.source_index]
    cycle_offset = consolidated.MECHANICS_FRESH_BASE + work.cycle * consolidated.MECHANICS_CYCLE_STRIDE
    return mechanics.WorkItem(
        index=source.index,
        lane_id=source.lane_id,
        mechanism=source.mechanism,
        phase=f"fresh_c{work.cycle}_{source.phase}",
        rung_index=source.rung_index,
        seed_start=source.seed_start + cycle_offset,
        seed_count=source.seed_count,
    )


def _fresh_item(work: WaveWorkItem) -> mechanics.WorkItem:
    if work.origin == _NEW_ORIGIN:
        return _fresh_new_item(work)
    return consolidated.fresh_mechanics_item(_consolidated_work(work))


def _work_seconds(work: WaveWorkItem) -> float:
    item = _fresh_item(work)
    if work.origin == _NEW_ORIGIN:
        rates = newq.NEW_PLANNED_SECONDS_PER_SEED
    else:
        rates = mechanics.PLANNED_SECONDS_PER_SEED
    seconds = float(item.seed_count * rates[item.mechanism])
    if item.mechanism == _CONSTRUCTION_MECHANISM:
        seconds /= CONSTRUCTION_VEC_SPEEDUP
    return seconds


def _new_artifact_path(raw_root: Path, work: WaveWorkItem) -> Path:
    return Path(raw_root).resolve() / _NEW_ARTIFACT_KIND / f"{work.key}.json"


def _artifact_path(raw_root: Path, work: WaveWorkItem) -> Path:
    if work.origin == _NEW_ORIGIN:
        return _new_artifact_path(raw_root, work)
    return consolidated._artifact_path(Path(raw_root).resolve(), _consolidated_work(work))


def _validate_artifact(value: Mapping[str, Any], work: WaveWorkItem) -> None:
    if work.origin == _NEW_ORIGIN:
        newq.validate_rung(value, _fresh_new_item(work))
    else:
        consolidated._validate_artifact(value, _consolidated_work(work))


def _load_receipts(
    raw_root: Path,
    works: Sequence[WaveWorkItem],
) -> dict[str, dict[str, Any]]:
    raw_root = Path(raw_root).resolve()
    receipts: dict[str, dict[str, Any]] = {}
    for work in works:
        path = _artifact_path(raw_root, work)
        if not path.is_file():
            continue
        try:
            value = _read_object(path)
            _validate_artifact(value, work)
            receipts[work.key] = value
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return receipts


def _execute_work(work: WaveWorkItem, raw_root: str) -> tuple[str, float]:
    key, duration = consolidated._execute_work(_consolidated_work(work), raw_root)
    return str(key), float(duration)


def _construction_rung_payload(item: mechanics.WorkItem) -> dict[str, Any]:

    if item.mechanism != _CONSTRUCTION_MECHANISM:
        raise ValueError("construction rung payload requires the construction_search mechanism")
    bed, _scalar_runner = build_pair(item.mechanism)
    runner = ConstructionSearchVecRunner()
    verdict_counts: dict[str, int] = {}
    control_clear_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    confirmations = 0
    for seed in range(item.seed_start, item.seed_start + item.seed_count):
        results = runner.run(bed, seed)
        receipt = runner.mint(results)
        if receipt.mechanism_id != item.mechanism:
            raise ValueError("construction vec runner receipt identity drifted")
        confirmations += int(receipt.is_confirmation)
        verdict_counts[receipt.verdict] = verdict_counts.get(receipt.verdict, 0) + 1
        for control in receipt.controls_cleared:
            control_clear_counts[control] = control_clear_counts.get(control, 0) + 1
        digest.update(receipt.digest().encode("ascii"))
    core = {
        "schema": mechanics.RUNG_SCHEMA,
        "program_id": mechanics.PROGRAM_ID,
        "claim_scope": mechanics.CLAIM_SCOPE,
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


def _execute_construction_work(work: WaveWorkItem, raw_root: str) -> tuple[str, float]:

    set_process_label(f"mop-fullgen-vec-{work.key[:20]}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    item = _fresh_item(work)
    payload = _construction_rung_payload(item)
    mechanics.validate_rung(payload, item)
    atomic_write_json(_artifact_path(Path(raw_root), work), payload)
    return work.key, time.perf_counter() - started


def _execute_new_work(work: WaveWorkItem, raw_root: str) -> tuple[str, float]:
    set_process_label(f"mop-fullgen-{work.key[:24]}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    item = _fresh_new_item(work)
    payload = newq._run_item_payload(item)
    newq.validate_rung(payload, item)
    atomic_write_json(_new_artifact_path(Path(raw_root), work), payload)
    return work.key, time.perf_counter() - started


def _runner_for(work: WaveWorkItem):

    if work.origin == _NEW_ORIGIN:
        return _execute_new_work
    if _work_mechanism(work) == _CONSTRUCTION_MECHANISM:
        return _execute_construction_work
    return _execute_work


def _require_complete_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
    works: Sequence[WaveWorkItem],
    label: str,
) -> None:
    expected = {work.key for work in works}
    if set(receipts) != expected:
        raise ValueError(f"{label} eligible raw receipt inventory is incomplete")


def _dynamic_pool_width(pending_count: int) -> tuple[int, int | None]:

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

    with suppress(Exception):
        os.nice(int(nice_level))


def _execute_pending(
    raw_root: Path,
    works: Sequence[WaveWorkItem],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], int]:
    raw_root = Path(raw_root).resolve()
    receipts = _load_receipts(raw_root, works)
    pending = [work for work in works if work.key not in receipts]
    durations: dict[str, float] = {}
    if pending:
        width, nice_level = _dynamic_pool_width(len(pending))
        initializer = _pool_worker_priority_init if nice_level is not None else None
        initargs = (int(nice_level),) if nice_level is not None else ()
        with ProcessPoolExecutor(max_workers=width, initializer=initializer, initargs=initargs) as pool:
            futures = {}
            for work in pending:
                runner = _runner_for(work)
                futures[pool.submit(runner, work, str(raw_root))] = work
            for future in as_completed(futures):
                work = futures[future]
                key, duration = future.result()
                value = _read_object(_artifact_path(raw_root, work))
                _validate_artifact(value, work)
                receipts[key] = value
                durations[key] = duration
    _require_complete_receipts(receipts, works, "full-generations-wave execution")
    return receipts, durations, len(pending)


def _artifact_index(
    raw_root: Path,
    works: Sequence[WaveWorkItem],
    receipts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_root = Path(raw_root).resolve()
    rows = []
    for work in works:
        if work.key not in receipts:
            continue
        path = _artifact_path(raw_root, work)
        rows.append(
            {
                "key": work.key,
                "origin": work.origin,
                "source_index": work.source_index,
                "lane_id": _work_lane_id(work),
                "path": _relative(path),
                "file_sha256": sha256_file(path),
                "result_sha256": receipts[work.key]["result_sha256"],
            }
        )
    return rows


def _lane_verdicts(
    works: Sequence[WaveWorkItem],
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    lane_ids = tuple(dict.fromkeys(_work_lane_id(work) for work in works))
    rows = {}
    for lane_id in lane_ids:
        lane_works = [work for work in works if _work_lane_id(work) == lane_id]
        counts: dict[str, int] = {}
        for work in lane_works:
            receipt = receipts.get(work.key)
            if receipt is None:
                continue
            for verdict, count in receipt["verdict_counts"].items():
                counts[verdict] = counts.get(verdict, 0) + int(count)
        complete = len([work for work in lane_works if work.key in receipts]) == len(lane_works)
        rows[lane_id] = {
            "planned_item_count": len(lane_works),
            "executed_item_count": sum(work.key in receipts for work in lane_works),
            "verdict_counts": counts,
            "continue_lane": complete and set(counts) == {"mechanics-ok"},
        }
    return rows


def _balanced_partitions(
    works: Sequence[WaveWorkItem],
) -> tuple[tuple[WaveWorkItem, ...], ...]:
    partitions: list[list[WaveWorkItem]] = [[] for _ in range(INTERNAL_SHARD_COUNT)]
    loads = [0.0] * INTERNAL_SHARD_COUNT
    for work in sorted(works, key=lambda row: (-_work_seconds(row), row.origin, row.source_index)):
        target = min(range(INTERNAL_SHARD_COUNT), key=lambda index: (loads[index], index))
        partitions[target].append(work)
        loads[target] += _work_seconds(work)
    return tuple(tuple(sorted(rows, key=lambda row: (row.origin, row.source_index))) for rows in partitions)


def _balanced_planning_shards(
    works: Sequence[WaveWorkItem],
    eligible_works: Sequence[WaveWorkItem],
    receipts: Mapping[str, Mapping[str, Any]],
    durations: Mapping[str, float],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    eligible_keys = {work.key for work in eligible_works}
    rows = []
    for index, partition in enumerate(_balanced_partitions(works)):
        eligible_partition = tuple(work for work in partition if work.key in eligible_keys)
        pruned_partition = tuple(work for work in partition if work.key not in eligible_keys)
        if not partition:
            status = "empty"
        elif not eligible_partition:
            status = "pruned"
        elif pruned_partition:
            status = "partial"
        else:
            status = "complete"
        rows.append(
            {
                "planning_shard_id": f"{prefix}-{index:02d}",
                "planning_shard_index": index,
                "partition_pinned": False,
                "work_item_count": len(partition),
                "eligible_item_count": len(eligible_partition),
                "planned_serial_seconds": sum(_work_seconds(work) for work in partition),
                "eligible_planned_serial_seconds": sum(_work_seconds(work) for work in eligible_partition),
                "pruned_planned_serial_seconds": sum(_work_seconds(work) for work in pruned_partition),
                "executed_item_count": sum(work.key in receipts for work in partition),
                "skipped_item_count": len(pruned_partition),
                "observed_seconds": sum(durations.get(work.key, 0.0) for work in partition),
                "status": status,
            }
        )
    return rows


def _category_execution_status(
    eligible_works: Sequence[WaveWorkItem],
    pruned_works: Sequence[WaveWorkItem],
) -> str:
    if not eligible_works:
        return "pruned"
    if pruned_works:
        return "partial"
    return "complete"


def category_work_items(
    epoch_index: int,
    category_id: str,
) -> tuple[WaveWorkItem, ...]:
    if not 0 <= epoch_index < len(EPOCH_IDS):
        raise ValueError("full-generations-wave epoch index is invalid")
    category = _CATEGORY_BY_ID.get(category_id)
    if category is None:
        raise ValueError(f"unknown full-generations-wave category: {category_id}")
    cycle = EPOCH_CYCLES[epoch_index]
    prefix = EPOCH_IDS[epoch_index].lower()
    items: list[WaveWorkItem] = []
    for source in mechanics.WORK_ITEMS:
        if source.lane_id in category.lane_ids:
            items.append(
                WaveWorkItem(
                    key=f"{prefix}_{source.capsule_id}",
                    origin=_OLD_ORIGIN,
                    source_index=source.index,
                    cycle=cycle,
                )
            )
    for source in newq.NEW_WORK_ITEMS:
        if source.lane_id in category.lane_ids:
            items.append(
                WaveWorkItem(
                    key=f"{prefix}_{source.capsule_id}",
                    origin=_NEW_ORIGIN,
                    source_index=source.index,
                    cycle=cycle,
                )
            )
    return tuple(items)


def integration_work_items() -> tuple[WaveWorkItem, ...]:
    cycle = EPOCH_CYCLES[-1]
    return tuple(
        WaveWorkItem(
            key=f"i1_{source.capsule_id}",
            origin=_OLD_ORIGIN,
            source_index=source.index,
            cycle=cycle,
        )
        for source in mechanics.WORK_ITEMS
        if source.lane_id == I1_LANE_ID
    )


def category_planned_serial_seconds(epoch_index: int, category_id: str) -> float:
    return sum(_work_seconds(work) for work in category_work_items(epoch_index, category_id))


def planned_epoch_compute_seconds(epoch_index: int) -> float:
    return sum(category_planned_serial_seconds(epoch_index, category_id) for category_id in CATEGORY_IDS)


def planned_program_compute_seconds() -> float:
    return sum(planned_epoch_compute_seconds(index) for index in range(len(EPOCH_IDS))) + sum(
        _work_seconds(work) for work in integration_work_items()
    )


def planned_serial_hours() -> float:
    return planned_program_compute_seconds() / 3_600


def planned_ideal_worker_hours() -> float:
    return planned_program_compute_seconds() / IDLE_WORKERS / 3_600


def _artifact_report(
    program: supervisor.Program,
    expectation: supervisor.ArtifactExpectation,
) -> dict[str, Any]:
    path = (program.repo_root / expectation.path).resolve()
    payload = _read_object(path)
    problems: list[str] = []
    if payload.get("schema") != expectation.schema:
        problems.append("schema drifted")
    for dotted, expected in expectation.fields:
        observed: object = payload
        for component in dotted.split("."):
            if not isinstance(observed, Mapping):
                observed = None
                break
            observed = observed.get(component)
        if observed != expected:
            problems.append(f"{dotted} drifted")
    if expectation.seal_field is not None:
        try:
            _validate_seal(payload, expectation.seal_field, "predecessor artifact")
        except ValueError as exc:
            problems.append(str(exc))
    return {
        "path": expectation.path,
        "sha256": sha256_file(path),
        "schema": payload.get("schema"),
        "problems": problems,
        "all_ok": not problems,
    }


def _validate_parent_supervisor() -> dict[str, Any]:
    manifest_path = Path(predecessor.PROGRAM_MANIFEST).resolve()
    try:
        program = supervisor.load_program(manifest_path, repo_root=REPO_ROOT)
        status = supervisor.read_status(program)
    except (OSError, ValueError, supervisor.Generation1Refused) as exc:
        raise ValueError("full-generations-wave predecessor supervisor authority is invalid") from exc
    rows = status.get("capsules")
    by_id = {capsule.capsule_id: capsule for capsule in program.capsules}
    if (
        program.program_id != predecessor.PROGRAM_ID
        or status.get("state") != "complete"
        or status.get("execution_enabled") is not True
        or status.get("accepted_injection_count") != 0
        or status.get("next_injection_sequence") != 1
        or status.get("current_capsule") is not None
        or status.get("lane_reservation") is not None
        or status.get("problems") != []
        or not isinstance(rows, Mapping)
        or set(rows) != set(by_id)
    ):
        raise ValueError("full-generations-wave predecessor supervisor is not a clean completion")
    for capsule_id, capsule in by_id.items():
        row = rows[capsule_id]
        attempts = row.get("attempts") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("capsule_sha256") != capsule.capsule_sha256
            or row.get("status") != "complete"
            or isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or row.get("returncode") != 0
            or row.get("last_problem") is not None
            or row.get("artifacts") != [_artifact_report(program, artifact) for artifact in capsule.artifacts]
        ):
            raise ValueError(f"full-generations-wave predecessor capsule drifted: {capsule_id}")
    manifest = _read_object(manifest_path)
    return {
        "program_manifest": _binding(manifest_path, manifest, "program_sha256"),
        "supervisor_status": _binding(program.status_path, status, "status_sha256"),
    }


def _validated_parent(
    *,
    result_path: Path,
    verification_path: Path,
    report_receipt_path: Path,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    verification_path = Path(verification_path).resolve()
    report_receipt_path = Path(report_receipt_path).resolve()
    result = _read_object(result_path)
    predecessor.validate_result(result)
    verification = _read_object(verification_path)
    predecessor_verify.validate_verification(verification)
    receipt = _read_object(report_receipt_path)
    predecessor.validate_report_receipt(receipt)
    source = verification.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
        or source.get("result_sha256") != result.get("result_sha256")
        or receipt.get("result") != {"path": _relative(result_path), "file_sha256": sha256_file(result_path)}
        or receipt.get("verification")
        != {"path": _relative(verification_path), "file_sha256": sha256_file(verification_path)}
    ):
        raise ValueError("full-generations-wave parent categorized-wave result chain is not byte-bound")
    waves = result.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("full-generations-wave parent categorized-wave inventory is empty")
    final = waves[-1]
    if not isinstance(final, Mapping):
        raise ValueError("full-generations-wave parent final wave is invalid")
    routing = final.get("routing")
    if not isinstance(routing, Mapping):
        raise ValueError("full-generations-wave parent final route is invalid")
    surviving: set[str] = set()
    for category_route in routing.values():
        if not isinstance(category_route, Mapping):
            raise ValueError("full-generations-wave parent final route row is invalid")
        eligible = category_route.get("eligible_lane_ids")
        if not isinstance(eligible, list) or any(not isinstance(lane, str) for lane in eligible):
            raise ValueError("full-generations-wave parent final route lanes are invalid")
        surviving.update(eligible)
    return {
        "bindings": {
            **_validate_parent_supervisor(),
            "result": _binding(result_path, result, "result_sha256"),
            "verification": _binding(verification_path, verification, "verification_sha256"),
            "report_receipt": _binding(report_receipt_path, receipt, "receipt_sha256"),
        },
        "mechanics_lanes": sorted(surviving),
    }


def _prior_gate(root: Path, gate_index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if gate_index == 0:
        return None, None
    path = _gate_path(root, GATE_IDS[gate_index - 1])
    value = _read_object(path)
    validate_gate(value, gate_index - 1, root=root)
    return _binding(path, value, "gate_sha256"), value


def _initial_route_seed(
    allowed_lane_ids: Sequence[str] | set[str],
) -> dict[str, dict[str, Any]]:
    allowed = set(allowed_lane_ids)
    return {
        category.category_id: {
            "continue": bool(set(category.lane_ids) & allowed),
            "eligible_lane_ids": [lane for lane in category.lane_ids if lane in allowed],
            "pruned_lane_ids": [lane for lane in category.lane_ids if lane not in allowed],
            "reason": (
                "carried_v1_survivors_and_admitted_new_lanes"
                if set(category.lane_ids) & allowed
                else "route_pruned_every_category_lane"
            ),
        }
        for category in CATEGORIES
    }


def _new_lane_admission_rows() -> list[dict[str, Any]]:
    rows = []
    for lane in newq.NEW_LANES:
        rows.append(
            {
                "lane_id": lane.lane_id,
                "mechanism": lane.mechanism,
                "canary_start": lane.canary_start,
                "producer_start": lane.producer_start,
                "challenge_start": lane.challenge_start,
                "rungs_per_phase": lane.rungs_per_phase,
                "seeds_per_rung": lane.seeds_per_rung,
                "canary_seeds": newq.CANARY_SEEDS,
                "planned_seconds_per_seed": newq.NEW_PLANNED_SECONDS_PER_SEED[lane.mechanism],
            }
        )
    return rows


def _mechanism_authorities(mechanism: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for role in ("bed", "runner"):
        relative = f"{_MECHANISM_DIRNAME}/{mechanism}_{role}.py"
        path = (_REPO_SOURCE_ROOT / relative).resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"full-generations-wave new mechanism authority is missing: {relative}")
        rows[role] = {"path": relative, "file_sha256": sha256_file(path)}
    return rows


def _new_mechanism_authorities() -> dict[str, dict[str, dict[str, str]]]:
    return {lane.mechanism: _mechanism_authorities(lane.mechanism) for lane in newq.NEW_LANES}


def _i1_substitution_row(parent_result_path: str) -> dict[str, Any]:
    justification = (
        "G1-P1 (stability_plasticity) was retired under the doc-20 redesign route; its lane fate is "
        f"recorded in the carried parent categorized-wave result at {parent_result_path}; the "
        "redesigned successor lane G1-P1R (stability_plasticity_r2) replaces it and its W08 canary "
        "partition gates admission before any downstream epoch."
    )
    return {
        "old_dependency": "G1-P1",
        "new_dependency": "G1-P1R",
        "substituted_dependency_lane_ids": list(I1_DEPENDENCIES),
        "justification": justification,
    }


def build_gate(
    *,
    root: Path = DEFAULT_ROOT,
    gate_index: int,
    parent_result_path: Path = DEFAULT_PARENT_RESULT,
    parent_verification_path: Path = DEFAULT_PARENT_VERIFICATION,
    parent_report_receipt_path: Path = DEFAULT_PARENT_REPORT_RECEIPT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if not 0 <= gate_index < len(GATE_IDS):
        raise ValueError("full-generations-wave gate index is invalid")
    prior_binding, prior = _prior_gate(root, gate_index)
    gate_id = GATE_IDS[gate_index]
    if gate_id == "admit_wave_v1":
        parent = _validated_parent(
            result_path=parent_result_path,
            verification_path=parent_verification_path,
            report_receipt_path=parent_report_receipt_path,
        )
        payload = {
            "operation": "bind_clean_categorized_batch_wave_v1",
            "parent_authority": parent["bindings"],
            "mechanics_lanes": parent["mechanics_lanes"],
            "admitted": True,
        }
    elif gate_id == "carry_d1_freeze":
        assert prior is not None
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        result_binding = admission["payload"]["parent_authority"]["result"]
        catalog = redesign_v2.candidate_catalog()
        payload = {
            "operation": "carry_forward_v1_d1_retirement_and_no_candidate_freeze",
            "v1_wave_result": dict(result_binding),
            "redesign_program_id": redesign_v2.PROGRAM_ID,
            "freeze_schema": redesign_v2.FREEZE_SCHEMA,
            "candidate_catalog_sha256": canonical_sha256(catalog),
            "carried_forward": True,
            "old_d1_retired": True,
            "old_d1_resurrection_allowed": False,
            "d1_redesign_efficacy_execution_authorized": False,
        }
    elif gate_id == "rescreen_d1_redesign":
        assert prior is not None
        if prior["payload"]["old_d1_retired"] is not True:
            raise ValueError("D1-v2 catalog cannot rescreen until old D1 retirement is carried forward")
        catalog = redesign_v2.candidate_catalog()
        candidate_ids = [row["candidate_id"] for row in catalog]
        evidence_count = sum(1 for row in catalog if row.get("execution_authorized") is True)
        payload = {
            "operation": "rescreen_d1_redesign_v2_candidate_catalog_without_execution",
            "redesign_program_id": redesign_v2.PROGRAM_ID,
            "screen_schema": redesign_v2.SCREEN_SCHEMA,
            "candidate_ids": candidate_ids,
            "candidate_catalog_sha256": canonical_sha256(catalog),
            "candidate_evidence_count": evidence_count,
            "eligible_for_future_authority": evidence_count > 0,
            "execution_authorized": False,
            "screen_state": "no_candidate_execution_evidence",
        }
    elif gate_id == "admit_new_lanes":
        assert prior is not None
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        parent_result_path_value = admission["payload"]["parent_authority"]["result"]["path"]
        payload = {
            "operation": "admit_three_new_mechanism_lanes",
            "new_lanes": _new_lane_admission_rows(),
            "mechanism_authorities": _new_mechanism_authorities(),
            "canary_policy": dict(_CANARY_POLICY),
            "i1_dependency_substitution": _i1_substitution_row(parent_result_path_value),
            "execution_authorized": False,
        }
    else:
        assert prior is not None
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        mechanics_lanes = admission["payload"]["mechanics_lanes"]
        allowed = set(mechanics_lanes) | set(NEW_LANE_IDS)
        payload = {
            "operation": "freeze_full_generations_routing",
            "route_seed": _initial_route_seed(allowed),
            "admitted_new_lane_ids": list(NEW_LANE_IDS),
            "i1_dependency_lane_ids": list(I1_DEPENDENCIES),
            "i1_initially_eligible": set(I1_DEPENDENCIES) <= allowed,
            "redesign_efficacy_execution_authorized": False,
            "future_change_requires_new_sealed_child": True,
        }
    core = {
        "schema": GATE_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "gate_id": gate_id,
        "gate_index": gate_index,
        "prior_gate": prior_binding,
        "payload": payload,
        "compute_started": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "gate_sha256": canonical_sha256(core)}


def validate_gate(
    value: Mapping[str, Any],
    gate_index: int,
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "gate_id",
            "gate_index",
            "prior_gate",
            "payload",
            "compute_started",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "independent_scientific_confirmation",
            "gate_sha256",
        },
        "full-generations-wave transition gate",
    )
    _validate_seal(value, "gate_sha256", "full-generations-wave transition gate")
    _validate_timestamp(value.get("created_at"), "full-generations-wave transition gate")
    if (
        value.get("schema") != GATE_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("gate_id") != GATE_IDS[gate_index]
        or value.get("gate_index") != gate_index
        or value.get("compute_started") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("full-generations-wave transition gate identity or safety drifted")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("full-generations-wave transition gate payload is invalid")
    if gate_index == 0:
        _require_exact_fields(
            payload,
            {"operation", "parent_authority", "mechanics_lanes", "admitted"},
            "full-generations-wave admission payload",
        )
        parent_authority = payload.get("parent_authority")
        mechanics_lanes = payload.get("mechanics_lanes")
        if (
            value.get("prior_gate") is not None
            or payload.get("operation") != "bind_clean_categorized_batch_wave_v1"
            or payload.get("admitted") is not True
            or not isinstance(parent_authority, Mapping)
            or not isinstance(mechanics_lanes, list)
        ):
            raise ValueError("full-generations-wave admission gate drifted")
        _require_exact_fields(
            parent_authority,
            {
                "program_manifest",
                "supervisor_status",
                "result",
                "verification",
                "report_receipt",
            },
            "full-generations-wave parent authority",
        )
        result_binding = parent_authority.get("result")
        verification_binding = parent_authority.get("verification")
        report_binding = parent_authority.get("report_receipt")
        if (
            not isinstance(result_binding, Mapping)
            or not isinstance(verification_binding, Mapping)
            or not isinstance(report_binding, Mapping)
        ):
            raise ValueError("full-generations-wave parent result chain is invalid")
        expected_parent = _validated_parent(
            result_path=_repo_path(result_binding.get("path"), "full-generations-wave parent result"),
            verification_path=_repo_path(
                verification_binding.get("path"),
                "full-generations-wave parent verification",
            ),
            report_receipt_path=_repo_path(
                report_binding.get("path"),
                "full-generations-wave parent report receipt",
            ),
        )
        if (
            parent_authority != expected_parent["bindings"]
            or mechanics_lanes != expected_parent["mechanics_lanes"]
            or len(mechanics_lanes) != len(set(mechanics_lanes))
        ):
            raise ValueError("full-generations-wave admission authority replay drifted")
        return
    prior_path, prior = _load_binding(
        value.get("prior_gate"),
        "gate_sha256",
        "full-generations-wave prior transition gate",
    )
    if prior_path != _gate_path(root, GATE_IDS[gate_index - 1]):
        raise ValueError("full-generations-wave prior transition gate path drifted")
    validate_gate(prior, gate_index - 1, root=root)
    gate_id = GATE_IDS[gate_index]
    if gate_id == "carry_d1_freeze":
        _require_exact_fields(
            payload,
            {
                "operation",
                "v1_wave_result",
                "redesign_program_id",
                "freeze_schema",
                "candidate_catalog_sha256",
                "carried_forward",
                "old_d1_retired",
                "old_d1_resurrection_allowed",
                "d1_redesign_efficacy_execution_authorized",
            },
            "full-generations-wave D1 carry-forward payload",
        )
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        result_binding = admission["payload"]["parent_authority"]["result"]
        catalog = redesign_v2.candidate_catalog()
        if (
            payload.get("operation") != "carry_forward_v1_d1_retirement_and_no_candidate_freeze"
            or payload.get("v1_wave_result") != result_binding
            or payload.get("redesign_program_id") != redesign_v2.PROGRAM_ID
            or payload.get("freeze_schema") != redesign_v2.FREEZE_SCHEMA
            or payload.get("candidate_catalog_sha256") != canonical_sha256(catalog)
            or payload.get("carried_forward") is not True
            or payload.get("old_d1_retired") is not True
            or payload.get("old_d1_resurrection_allowed") is not False
            or payload.get("d1_redesign_efficacy_execution_authorized") is not False
        ):
            raise ValueError("full-generations-wave D1 carry-forward gate drifted")
    elif gate_id == "rescreen_d1_redesign":
        _require_exact_fields(
            payload,
            {
                "operation",
                "redesign_program_id",
                "screen_schema",
                "candidate_ids",
                "candidate_catalog_sha256",
                "candidate_evidence_count",
                "eligible_for_future_authority",
                "execution_authorized",
                "screen_state",
            },
            "full-generations-wave D1 rescreen payload",
        )
        if prior["payload"]["old_d1_retired"] is not True:
            raise ValueError("full-generations-wave D1 rescreen prerequisite drifted")
        catalog = redesign_v2.candidate_catalog()
        candidate_ids = [row["candidate_id"] for row in catalog]
        evidence_count = sum(1 for row in catalog if row.get("execution_authorized") is True)
        if (
            payload.get("operation") != "rescreen_d1_redesign_v2_candidate_catalog_without_execution"
            or payload.get("redesign_program_id") != redesign_v2.PROGRAM_ID
            or payload.get("screen_schema") != redesign_v2.SCREEN_SCHEMA
            or payload.get("candidate_ids") != candidate_ids
            or payload.get("candidate_catalog_sha256") != canonical_sha256(catalog)
            or payload.get("candidate_evidence_count") != evidence_count
            or payload.get("eligible_for_future_authority") is not (evidence_count > 0)
            or payload.get("execution_authorized") is not False
            or payload.get("screen_state") != "no_candidate_execution_evidence"
        ):
            raise ValueError("full-generations-wave D1 rescreen gate drifted")
    elif gate_id == "admit_new_lanes":
        _require_exact_fields(
            payload,
            {
                "operation",
                "new_lanes",
                "mechanism_authorities",
                "canary_policy",
                "i1_dependency_substitution",
                "execution_authorized",
            },
            "full-generations-wave new-lane admission payload",
        )
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        parent_result_path_value = admission["payload"]["parent_authority"]["result"]["path"]
        if (
            payload.get("operation") != "admit_three_new_mechanism_lanes"
            or payload.get("new_lanes") != _new_lane_admission_rows()
            or payload.get("mechanism_authorities") != _new_mechanism_authorities()
            or payload.get("canary_policy") != dict(_CANARY_POLICY)
            or payload.get("i1_dependency_substitution") != _i1_substitution_row(parent_result_path_value)
            or payload.get("execution_authorized") is not False
        ):
            raise ValueError("full-generations-wave new-lane admission gate drifted")
    else:
        _require_exact_fields(
            payload,
            {
                "operation",
                "route_seed",
                "admitted_new_lane_ids",
                "i1_dependency_lane_ids",
                "i1_initially_eligible",
                "redesign_efficacy_execution_authorized",
                "future_change_requires_new_sealed_child",
            },
            "full-generations-wave routing freeze payload",
        )
        admission = _read_object(_gate_path(root, "admit_wave_v1"))
        validate_gate(admission, 0, root=root)
        mechanics_lanes = admission["payload"]["mechanics_lanes"]
        allowed = set(mechanics_lanes) | set(NEW_LANE_IDS)
        if (
            payload.get("operation") != "freeze_full_generations_routing"
            or payload.get("route_seed") != _initial_route_seed(allowed)
            or payload.get("admitted_new_lane_ids") != list(NEW_LANE_IDS)
            or payload.get("i1_dependency_lane_ids") != list(I1_DEPENDENCIES)
            or payload.get("i1_initially_eligible") is not (set(I1_DEPENDENCIES) <= allowed)
            or payload.get("redesign_efficacy_execution_authorized") is not False
            or payload.get("future_change_requires_new_sealed_child") is not True
        ):
            raise ValueError("full-generations-wave routing freeze gate drifted")


def materialize_gate(
    *,
    root: Path = DEFAULT_ROOT,
    gate_index: int,
    parent_result_path: Path = DEFAULT_PARENT_RESULT,
    parent_verification_path: Path = DEFAULT_PARENT_VERIFICATION,
    parent_report_receipt_path: Path = DEFAULT_PARENT_REPORT_RECEIPT,
) -> dict[str, Any]:
    path = _gate_path(root, GATE_IDS[gate_index])
    if path.is_file():
        existing = _read_object(path)
        validate_gate(existing, gate_index, root=root)
        return existing
    value = build_gate(
        root=root,
        gate_index=gate_index,
        parent_result_path=parent_result_path,
        parent_verification_path=parent_verification_path,
        parent_report_receipt_path=parent_report_receipt_path,
    )
    validate_gate(value, gate_index, root=root)
    atomic_write_json(path, value)
    return value


def _routing_for_epoch(
    root: Path,
    epoch_index: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    context = _validated_context_for_epoch(Path(root).resolve(), epoch_index)
    return context.predecessor_binding, context.routing


def _validated_context_for_epoch(
    root: Path,
    epoch_index: int,
) -> _WaveValidationContext:
    if not 0 <= epoch_index < len(EPOCH_IDS):
        raise ValueError("full-generations-wave epoch index is invalid")
    root = Path(root).resolve()
    gate_path = _gate_path(root, "freeze_routing")
    gate = _read_object(gate_path)
    validate_gate(gate, len(GATE_IDS) - 1, root=root)
    context = _WaveValidationContext(
        predecessor_binding={
            "kind": "freeze_routing",
            "authority": _binding(gate_path, gate, "gate_sha256"),
        },
        routing={key: dict(row) for key, row in gate["payload"]["route_seed"].items()},
        parent_classification_sha256=None,
    )
    for prior_epoch_index in range(epoch_index):
        classification_path = _classification_path(root, prior_epoch_index)
        classification = _read_object(classification_path)
        routing = _validate_classification_with_context(
            classification,
            prior_epoch_index,
            root=root,
            context=context,
        )
        context = _WaveValidationContext(
            predecessor_binding={
                "kind": "full_generations_classification",
                "authority": _binding(
                    classification_path,
                    classification,
                    "classification_sha256",
                ),
            },
            routing=routing,
            parent_classification_sha256=str(classification["classification_sha256"]),
        )
    return context


def _route_lane_partition(
    category_id: str,
    route: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    category = _CATEGORY_BY_ID[category_id]
    eligible_value = route.get("eligible_lane_ids")
    pruned_value = route.get("pruned_lane_ids")
    if (
        not isinstance(eligible_value, list)
        or any(not isinstance(item, str) for item in eligible_value)
        or not isinstance(pruned_value, list)
        or any(not isinstance(item, str) for item in pruned_value)
    ):
        raise ValueError("full-generations-wave route lane inventory is invalid")
    eligible_set = set(eligible_value)
    pruned_set = set(pruned_value)
    expected_eligible = tuple(lane_id for lane_id in category.lane_ids if lane_id in eligible_set)
    expected_pruned = tuple(lane_id for lane_id in category.lane_ids if lane_id in pruned_set)
    if (
        tuple(eligible_value) != expected_eligible
        or tuple(pruned_value) != expected_pruned
        or eligible_set & pruned_set
        or eligible_set | pruned_set != set(category.lane_ids)
        or route.get("continue") is not bool(expected_eligible)
        or not isinstance(route.get("reason"), str)
        or not route["reason"]
    ):
        raise ValueError("full-generations-wave route lane partition drifted")
    return expected_eligible, expected_pruned


def _partition_category_work(
    works: Sequence[WaveWorkItem],
    eligible_lane_ids: Sequence[str],
) -> tuple[tuple[WaveWorkItem, ...], tuple[WaveWorkItem, ...]]:
    eligible = set(eligible_lane_ids)
    eligible_works = tuple(work for work in works if _work_lane_id(work) in eligible)
    pruned_works = tuple(work for work in works if _work_lane_id(work) not in eligible)
    return eligible_works, pruned_works


def _redesign_efficacy_status(root: Path) -> dict[str, Any]:
    rescreen_gate = _read_object(_gate_path(root, "rescreen_d1_redesign"))
    validate_gate(rescreen_gate, GATE_IDS.index("rescreen_d1_redesign"), root=root)
    carry_gate = _read_object(_gate_path(root, "carry_d1_freeze"))
    validate_gate(carry_gate, GATE_IDS.index("carry_d1_freeze"), root=root)
    return {
        "screen_state": rescreen_gate["payload"]["screen_state"],
        "carried_forward": carry_gate["payload"]["carried_forward"],
        "candidate_evidence_count": rescreen_gate["payload"]["candidate_evidence_count"],
        "selected_candidate_id": None,
        "execution_authorized": False,
        "planned_serial_seconds": 0.0,
        "executed_item_count": 0,
    }


def _build_category_result(
    *,
    root: Path,
    epoch_index: int,
    category_id: str,
    predecessor_binding: Mapping[str, Any],
    route: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
    durations: Mapping[str, float],
    newly_executed: int,
) -> dict[str, Any]:
    category = _CATEGORY_BY_ID[category_id]
    works = category_work_items(epoch_index, category_id)
    eligible_lane_ids, pruned_lane_ids = _route_lane_partition(category_id, route)
    eligible_works, pruned_works = _partition_category_work(works, eligible_lane_ids)
    _require_complete_receipts(
        receipts,
        eligible_works,
        "full-generations-wave category build",
    )
    eligible = bool(eligible_works)
    raw_root = _category_raw_root(root, epoch_index, category_id)
    lane_verdicts = _lane_verdicts(works, receipts)
    core = {
        "schema": CATEGORY_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "epoch_id": EPOCH_IDS[epoch_index],
        "epoch_index": epoch_index,
        "cycle_index": EPOCH_CYCLES[epoch_index],
        "category": {
            "id": category.category_id,
            "name": category.name,
            "order": category.order,
            "lane_ids": list(category.lane_ids),
        },
        "predecessor": dict(predecessor_binding),
        "routing_input": {
            **dict(route),
            "null_safe_pruning": True,
            "resurrection_allowed": False,
        },
        "balanced_planning_shard_count": INTERNAL_SHARD_COUNT,
        "balanced_planning_shards": _balanced_planning_shards(
            works,
            eligible_works,
            receipts,
            durations,
            prefix=f"{EPOCH_IDS[epoch_index].lower()}-{category_id}",
        ),
        "artifact_index": _artifact_index(raw_root, works, receipts),
        "lane_results": lane_verdicts,
        "redesign_v2_efficacy": (
            _redesign_efficacy_status(root) if category_id == "dispatch_redesign" else None
        ),
        "execution": {
            "eligible": eligible,
            "status": _category_execution_status(eligible_works, pruned_works),
            "worker_pool": "dynamic_process_pool",
            "worker_pool_max_workers": IDLE_WORKERS,
            "eligible_lane_ids": list(eligible_lane_ids),
            "pruned_lane_ids": list(pruned_lane_ids),
            "planned_item_count": len(works),
            "eligible_item_count": len(eligible_works),
            "pruned_item_count": len(pruned_works),
            "executed_item_count": len(receipts),
            "newly_executed_item_count": newly_executed,
            "reused_item_count": len(receipts) - newly_executed,
            "skipped_item_count": len(pruned_works),
            "planned_serial_seconds": sum(_work_seconds(work) for work in works),
            "eligible_planned_serial_seconds": sum(_work_seconds(work) for work in eligible_works),
            "pruned_planned_serial_seconds": sum(_work_seconds(work) for work in pruned_works),
            "observed_seconds": sum(durations.values()),
            "compute_started": eligible and bool(receipts),
        },
        "interpretation_limit": (
            "Fresh-cycle deterministic mechanics stress validates repeatability and control wiring. "
            "It is not independent scientific confirmation."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "category_sha256": canonical_sha256(core)}


def _validate_category_with_context(
    value: Mapping[str, Any],
    *,
    epoch_index: int,
    category_id: str,
    root: Path,
    predecessor_binding: Mapping[str, Any],
    expected_route: Mapping[str, Any],
) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "epoch_id",
            "epoch_index",
            "cycle_index",
            "category",
            "predecessor",
            "routing_input",
            "balanced_planning_shard_count",
            "balanced_planning_shards",
            "artifact_index",
            "lane_results",
            "redesign_v2_efficacy",
            "execution",
            "interpretation_limit",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "independent_scientific_confirmation",
            "category_sha256",
        },
        "full-generations-wave category",
    )
    _validate_seal(value, "category_sha256", "full-generations-wave category")
    _validate_timestamp(value.get("created_at"), "full-generations-wave category")
    category = _CATEGORY_BY_ID[category_id]
    route = value.get("routing_input")
    eligible_lane_ids, pruned_lane_ids = _route_lane_partition(category_id, expected_route)
    works = category_work_items(epoch_index, category_id)
    eligible_works, pruned_works = _partition_category_work(works, eligible_lane_ids)
    eligible = bool(eligible_works)
    raw_root = _category_raw_root(root, epoch_index, category_id)
    receipts = _load_receipts(raw_root, eligible_works)
    forbidden_receipts = _load_receipts(raw_root, pruned_works)
    if forbidden_receipts:
        raise ValueError("full-generations-wave pruned lane emitted a raw receipt")
    _require_complete_receipts(
        receipts,
        eligible_works,
        "full-generations-wave category",
    )
    lane_results = _lane_verdicts(works, receipts)
    execution = value.get("execution")
    if isinstance(execution, Mapping):
        _require_exact_fields(
            execution,
            {
                "eligible",
                "status",
                "worker_pool",
                "worker_pool_max_workers",
                "eligible_lane_ids",
                "pruned_lane_ids",
                "planned_item_count",
                "eligible_item_count",
                "pruned_item_count",
                "executed_item_count",
                "newly_executed_item_count",
                "reused_item_count",
                "skipped_item_count",
                "planned_serial_seconds",
                "eligible_planned_serial_seconds",
                "pruned_planned_serial_seconds",
                "observed_seconds",
                "compute_started",
            },
            "full-generations-wave category execution",
        )
    if (
        value.get("schema") != CATEGORY_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_id") != EPOCH_IDS[epoch_index]
        or value.get("epoch_index") != epoch_index
        or value.get("cycle_index") != EPOCH_CYCLES[epoch_index]
        or value.get("category")
        != {
            "id": category.category_id,
            "name": category.name,
            "order": category.order,
            "lane_ids": list(category.lane_ids),
        }
        or value.get("predecessor") != predecessor_binding
        or route
        != {
            **expected_route,
            "null_safe_pruning": True,
            "resurrection_allowed": False,
        }
        or value.get("balanced_planning_shard_count") != INTERNAL_SHARD_COUNT
        or value.get("lane_results") != lane_results
        or not isinstance(execution, Mapping)
        or execution.get("eligible") is not eligible
        or execution.get("status") != _category_execution_status(eligible_works, pruned_works)
        or execution.get("worker_pool") != "dynamic_process_pool"
        or execution.get("worker_pool_max_workers") != IDLE_WORKERS
        or execution.get("eligible_lane_ids") != list(eligible_lane_ids)
        or execution.get("pruned_lane_ids") != list(pruned_lane_ids)
        or execution.get("planned_item_count") != len(works)
        or execution.get("eligible_item_count") != len(eligible_works)
        or execution.get("pruned_item_count") != len(pruned_works)
        or execution.get("executed_item_count") != len(receipts)
        or execution.get("skipped_item_count") != len(pruned_works)
        or execution.get("planned_serial_seconds") != sum(_work_seconds(work) for work in works)
        or execution.get("eligible_planned_serial_seconds")
        != sum(_work_seconds(work) for work in eligible_works)
        or execution.get("pruned_planned_serial_seconds") != sum(_work_seconds(work) for work in pruned_works)
        or execution.get("compute_started") is not (eligible and bool(receipts))
        or value.get("interpretation_limit")
        != (
            "Fresh-cycle deterministic mechanics stress validates repeatability and control wiring. "
            "It is not independent scientific confirmation."
        )
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("full-generations-wave category identity, execution, or safety drifted")
    artifact_rows = value.get("artifact_index")
    if not isinstance(artifact_rows, list) or len(artifact_rows) != len(receipts):
        raise ValueError("full-generations-wave category artifact inventory drifted")
    expected_index = _artifact_index(raw_root, works, receipts)
    if artifact_rows != expected_index:
        raise ValueError("full-generations-wave category raw receipt bindings drifted")
    newly_executed = execution.get("newly_executed_item_count")
    reused = execution.get("reused_item_count")
    observed_seconds = execution.get("observed_seconds")
    if (
        not isinstance(newly_executed, int)
        or isinstance(newly_executed, bool)
        or not 0 <= newly_executed <= len(receipts)
        or reused != len(receipts) - newly_executed
        or not isinstance(observed_seconds, int | float)
        or observed_seconds < 0
    ):
        raise ValueError("full-generations-wave category observed execution accounting drifted")
    shard_rows = value.get("balanced_planning_shards")
    if not isinstance(shard_rows, list) or len(shard_rows) != INTERNAL_SHARD_COUNT:
        raise ValueError("full-generations-wave category planning shard inventory drifted")
    partitions = _balanced_partitions(works)
    if sum(int(row.get("work_item_count", -1)) for row in shard_rows) != len(works):
        raise ValueError("full-generations-wave category planning shard work count drifted")
    eligible_keys = {work.key for work in eligible_works}
    for index, (row, partition) in enumerate(zip(shard_rows, partitions, strict=True)):
        eligible_partition = tuple(work for work in partition if work.key in eligible_keys)
        pruned_partition = tuple(work for work in partition if work.key not in eligible_keys)
        if not partition:
            expected_status = "empty"
        elif not eligible_partition:
            expected_status = "pruned"
        elif pruned_partition:
            expected_status = "partial"
        else:
            expected_status = "complete"
        if isinstance(row, Mapping):
            _require_exact_fields(
                row,
                {
                    "planning_shard_id",
                    "planning_shard_index",
                    "partition_pinned",
                    "work_item_count",
                    "eligible_item_count",
                    "planned_serial_seconds",
                    "eligible_planned_serial_seconds",
                    "pruned_planned_serial_seconds",
                    "executed_item_count",
                    "skipped_item_count",
                    "observed_seconds",
                    "status",
                },
                "full-generations-wave category planning shard",
            )
        if (
            not isinstance(row, Mapping)
            or row.get("planning_shard_id") != f"{EPOCH_IDS[epoch_index].lower()}-{category_id}-{index:02d}"
            or row.get("planning_shard_index") != index
            or row.get("partition_pinned") is not False
            or row.get("work_item_count") != len(partition)
            or row.get("eligible_item_count") != len(eligible_partition)
            or row.get("planned_serial_seconds") != sum(_work_seconds(work) for work in partition)
            or row.get("eligible_planned_serial_seconds")
            != sum(_work_seconds(work) for work in eligible_partition)
            or row.get("pruned_planned_serial_seconds")
            != sum(_work_seconds(work) for work in pruned_partition)
            or row.get("executed_item_count") != sum(work.key in receipts for work in partition)
            or row.get("skipped_item_count") != len(pruned_partition)
            or not isinstance(row.get("observed_seconds"), int | float)
            or row["observed_seconds"] < 0
            or row.get("status") != expected_status
        ):
            raise ValueError("full-generations-wave category planning shard drifted")
    if sum(float(row["observed_seconds"]) for row in shard_rows) != float(observed_seconds):
        raise ValueError("full-generations-wave category planning shard timing drifted")
    if category_id == "dispatch_redesign":
        if value.get("redesign_v2_efficacy") != _redesign_efficacy_status(Path(root)):
            raise ValueError("full-generations-wave D1-v2 efficacy gate drifted")
    elif value.get("redesign_v2_efficacy") is not None:
        raise ValueError("full-generations-wave non-dispatch category contains D1-v2 efficacy")


def validate_category(
    value: Mapping[str, Any],
    *,
    epoch_index: int,
    category_id: str,
    root: Path = DEFAULT_ROOT,
) -> None:
    context = _validated_context_for_epoch(Path(root).resolve(), epoch_index)
    _validate_category_with_context(
        value,
        epoch_index=epoch_index,
        category_id=category_id,
        root=Path(root).resolve(),
        predecessor_binding=context.predecessor_binding,
        expected_route=context.routing[category_id],
    )


def run_category(
    *,
    root: Path = DEFAULT_ROOT,
    epoch_index: int,
    category_id: str,
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = _category_path(root, epoch_index, category_id)
    if path.is_file():
        existing = _read_object(path)
        validate_category(existing, epoch_index=epoch_index, category_id=category_id, root=root)
        return existing
    predecessor_binding, routing = _routing_for_epoch(root, epoch_index)
    route = routing[category_id]
    works = category_work_items(epoch_index, category_id)
    eligible_lane_ids, _ = _route_lane_partition(category_id, route)
    eligible_works, pruned_works = _partition_category_work(works, eligible_lane_ids)
    if _load_receipts(_category_raw_root(root, epoch_index, category_id), pruned_works):
        raise ValueError("full-generations-wave pruned lane already has a raw receipt")
    if eligible_works:
        receipts, durations, newly_executed = _execute_pending(
            _category_raw_root(root, epoch_index, category_id),
            eligible_works,
        )
    else:
        receipts, durations, newly_executed = {}, {}, 0
    value = _build_category_result(
        root=root,
        epoch_index=epoch_index,
        category_id=category_id,
        predecessor_binding=predecessor_binding,
        route=route,
        receipts=receipts,
        durations=durations,
        newly_executed=newly_executed,
    )
    validate_category(value, epoch_index=epoch_index, category_id=category_id, root=root)
    atomic_write_json(path, value)
    return value


def _classification_components(
    *,
    root: Path,
    epoch_index: int,
    context: _WaveValidationContext,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int, int, float]:
    bindings = []
    routing: dict[str, dict[str, Any]] = {}
    executed = 0
    skipped = 0
    observed = 0.0
    for category_id in CATEGORY_IDS:
        path = _category_path(root, epoch_index, category_id)
        category = _read_object(path)
        _validate_category_with_context(
            category,
            epoch_index=epoch_index,
            category_id=category_id,
            root=root,
            predecessor_binding=context.predecessor_binding,
            expected_route=context.routing[category_id],
        )
        bindings.append({"category_id": category_id, **_binding(path, category, "category_sha256")})
        eligible_lanes = [
            lane_id
            for lane_id in CATEGORY_LANES[category_id]
            if isinstance(category["lane_results"].get(lane_id), Mapping)
            and category["lane_results"][lane_id]["continue_lane"] is True
        ]
        all_lanes = list(CATEGORY_LANES[category_id])
        if not set(eligible_lanes) <= set(context.routing[category_id]["eligible_lane_ids"]):
            raise ValueError("full-generations-wave classifier attempted lane resurrection")
        routing[category_id] = {
            "continue": bool(eligible_lanes),
            "eligible_lane_ids": eligible_lanes,
            "pruned_lane_ids": [lane for lane in all_lanes if lane not in eligible_lanes],
            "reason": (
                "all_retained_lanes_completed_mechanics_ok"
                if eligible_lanes
                else "null_safe_pruning_after_route_or_mechanics_warning"
            ),
        }
        executed += int(category["execution"]["executed_item_count"])
        skipped += int(category["execution"]["skipped_item_count"])
        observed += float(category["execution"]["observed_seconds"])
    return bindings, routing, executed, skipped, observed


def build_classification(*, root: Path = DEFAULT_ROOT, epoch_index: int) -> dict[str, Any]:
    root = Path(root).resolve()
    context = _validated_context_for_epoch(root, epoch_index)
    bindings, routing, executed, skipped, observed = _classification_components(
        root=root,
        epoch_index=epoch_index,
        context=context,
    )
    core = {
        "schema": CLASSIFICATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "epoch_id": EPOCH_IDS[epoch_index],
        "epoch_index": epoch_index,
        "cycle_index": EPOCH_CYCLES[epoch_index],
        "parent_classification_sha256": context.parent_classification_sha256,
        "category_bindings": bindings,
        "routing": routing,
        "summary": {
            "executed_item_count": executed,
            "skipped_item_count": skipped,
            "observed_seconds": observed,
            "compute_started": executed > 0,
        },
        "interpretation_limit": (
            "The serial barrier routes deterministic mechanics lanes only; it cannot create a "
            "scientific efficacy or confirmation claim."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "classification_sha256": canonical_sha256(core)}


def _validate_classification_with_context(
    value: Mapping[str, Any],
    epoch_index: int,
    *,
    root: Path,
    context: _WaveValidationContext,
) -> dict[str, dict[str, Any]]:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "epoch_id",
            "epoch_index",
            "cycle_index",
            "parent_classification_sha256",
            "category_bindings",
            "routing",
            "summary",
            "interpretation_limit",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "independent_scientific_confirmation",
            "classification_sha256",
        },
        "full-generations-wave classification",
    )
    _validate_seal(value, "classification_sha256", "full-generations-wave classification")
    _validate_timestamp(value.get("created_at"), "full-generations-wave classification")
    bindings, routing, executed, skipped, observed = _classification_components(
        root=root,
        epoch_index=epoch_index,
        context=context,
    )
    if (
        value.get("schema") != CLASSIFICATION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("epoch_id") != EPOCH_IDS[epoch_index]
        or value.get("epoch_index") != epoch_index
        or value.get("cycle_index") != EPOCH_CYCLES[epoch_index]
        or value.get("parent_classification_sha256") != context.parent_classification_sha256
        or value.get("category_bindings") != bindings
        or value.get("routing") != routing
        or value.get("summary")
        != {
            "executed_item_count": executed,
            "skipped_item_count": skipped,
            "observed_seconds": observed,
            "compute_started": executed > 0,
        }
        or value.get("interpretation_limit")
        != (
            "The serial barrier routes deterministic mechanics lanes only; it cannot create a "
            "scientific efficacy or confirmation claim."
        )
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("full-generations-wave classification identity, routing, or safety drifted")
    return routing


def validate_classification(
    value: Mapping[str, Any],
    epoch_index: int,
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    root = Path(root).resolve()
    context = _validated_context_for_epoch(root, epoch_index)
    _validate_classification_with_context(
        value,
        epoch_index,
        root=root,
        context=context,
    )


def classify_epoch(*, root: Path = DEFAULT_ROOT, epoch_index: int) -> dict[str, Any]:
    path = _classification_path(root, epoch_index)
    if path.is_file():
        existing = _read_object(path)
        validate_classification(existing, epoch_index, root=root)
        return existing
    value = build_classification(root=root, epoch_index=epoch_index)
    validate_classification(value, epoch_index, root=root)
    atomic_write_json(path, value)
    return value


def _i1_eligible(root: Path) -> tuple[dict[str, Any], bool, bool]:
    path = _classification_path(root, len(EPOCH_IDS) - 1)
    classification = _read_object(path)
    validate_classification(classification, len(EPOCH_IDS) - 1, root=root)
    freeze_gate = _read_object(_gate_path(root, "freeze_routing"))
    validate_gate(freeze_gate, len(GATE_IDS) - 1, root=root)
    initially_eligible = freeze_gate["payload"]["i1_initially_eligible"] is True
    retained = {lane for row in classification["routing"].values() for lane in row["eligible_lane_ids"]}
    eligible = initially_eligible and set(I1_DEPENDENCIES) <= retained
    return (
        _binding(path, classification, "classification_sha256"),
        initially_eligible,
        eligible,
    )


def _build_integration_result(
    *,
    root: Path,
    predecessor_binding: Mapping[str, Any],
    initially_eligible: bool,
    eligible: bool,
    receipts: Mapping[str, Mapping[str, Any]],
    durations: Mapping[str, float],
    newly_executed: int,
) -> dict[str, Any]:
    works = integration_work_items()
    eligible_works = works if eligible else ()
    pruned_works = () if eligible else works
    _require_complete_receipts(
        receipts,
        eligible_works,
        "full-generations-wave I1 build",
    )
    raw_root = _integration_raw_root(root)
    lane_results = _lane_verdicts(works, receipts)
    core = {
        "schema": INTEGRATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "integration_id": "I1",
        "cycle_index": EPOCH_CYCLES[-1],
        "lane_id": I1_LANE_ID,
        "dependency_lane_ids": list(I1_DEPENDENCIES),
        "i1_initially_eligible": initially_eligible,
        "predecessor_classification": dict(predecessor_binding),
        "balanced_planning_shard_count": INTERNAL_SHARD_COUNT,
        "balanced_planning_shards": _balanced_planning_shards(
            works,
            eligible_works,
            receipts,
            durations,
            prefix="i1",
        ),
        "artifact_index": _artifact_index(raw_root, works, receipts),
        "lane_result": lane_results[I1_LANE_ID],
        "execution": {
            "eligible": eligible,
            "status": "complete" if eligible else "pruned",
            "worker_pool": "dynamic_process_pool",
            "worker_pool_max_workers": IDLE_WORKERS,
            "planned_item_count": len(works),
            "eligible_item_count": len(eligible_works),
            "pruned_item_count": len(pruned_works),
            "executed_item_count": len(receipts),
            "newly_executed_item_count": newly_executed,
            "reused_item_count": len(receipts) - newly_executed,
            "skipped_item_count": len(pruned_works),
            "planned_serial_seconds": sum(_work_seconds(work) for work in works),
            "eligible_planned_serial_seconds": sum(_work_seconds(work) for work in eligible_works),
            "pruned_planned_serial_seconds": sum(_work_seconds(work) for work in pruned_works),
            "observed_seconds": sum(durations.values()),
            "compute_started": eligible and bool(receipts),
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "integration_sha256": canonical_sha256(core)}


def validate_integration(value: Mapping[str, Any], *, root: Path = DEFAULT_ROOT) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "integration_id",
            "cycle_index",
            "lane_id",
            "dependency_lane_ids",
            "i1_initially_eligible",
            "predecessor_classification",
            "balanced_planning_shard_count",
            "balanced_planning_shards",
            "artifact_index",
            "lane_result",
            "execution",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "independent_scientific_confirmation",
            "integration_sha256",
        },
        "full-generations-wave I1 integration",
    )
    _validate_seal(value, "integration_sha256", "full-generations-wave I1 integration")
    _validate_timestamp(value.get("created_at"), "full-generations-wave I1 integration")
    predecessor_binding, initially_eligible, eligible = _i1_eligible(Path(root).resolve())
    works = integration_work_items()
    eligible_works = works if eligible else ()
    pruned_works = () if eligible else works
    raw_root = _integration_raw_root(root)
    receipts = _load_receipts(raw_root, eligible_works)
    forbidden_receipts = _load_receipts(raw_root, pruned_works)
    if forbidden_receipts:
        raise ValueError("full-generations-wave pruned I1 emitted a raw receipt")
    _require_complete_receipts(
        receipts,
        eligible_works,
        "full-generations-wave I1",
    )
    execution = value.get("execution")
    if isinstance(execution, Mapping):
        _require_exact_fields(
            execution,
            {
                "eligible",
                "status",
                "worker_pool",
                "worker_pool_max_workers",
                "planned_item_count",
                "eligible_item_count",
                "pruned_item_count",
                "executed_item_count",
                "newly_executed_item_count",
                "reused_item_count",
                "skipped_item_count",
                "planned_serial_seconds",
                "eligible_planned_serial_seconds",
                "pruned_planned_serial_seconds",
                "observed_seconds",
                "compute_started",
            },
            "full-generations-wave I1 execution",
        )
    if (
        value.get("schema") != INTEGRATION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("integration_id") != "I1"
        or value.get("cycle_index") != EPOCH_CYCLES[-1]
        or value.get("lane_id") != I1_LANE_ID
        or value.get("dependency_lane_ids") != list(I1_DEPENDENCIES)
        or value.get("i1_initially_eligible") is not initially_eligible
        or value.get("predecessor_classification") != predecessor_binding
        or value.get("balanced_planning_shard_count") != INTERNAL_SHARD_COUNT
        or value.get("lane_result") != _lane_verdicts(works, receipts)[I1_LANE_ID]
        or not isinstance(execution, Mapping)
        or execution.get("eligible") is not eligible
        or execution.get("status") != ("complete" if eligible else "pruned")
        or execution.get("worker_pool") != "dynamic_process_pool"
        or execution.get("worker_pool_max_workers") != IDLE_WORKERS
        or execution.get("planned_item_count") != len(works)
        or execution.get("eligible_item_count") != len(eligible_works)
        or execution.get("pruned_item_count") != len(pruned_works)
        or execution.get("executed_item_count") != len(receipts)
        or execution.get("skipped_item_count") != len(pruned_works)
        or execution.get("planned_serial_seconds") != sum(_work_seconds(work) for work in works)
        or execution.get("eligible_planned_serial_seconds")
        != sum(_work_seconds(work) for work in eligible_works)
        or execution.get("pruned_planned_serial_seconds") != sum(_work_seconds(work) for work in pruned_works)
        or execution.get("compute_started") is not (eligible and bool(receipts))
        or value.get("artifact_index") != _artifact_index(raw_root, works, receipts)
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("full-generations-wave I1 integration identity, execution, or safety drifted")
    newly_executed = execution.get("newly_executed_item_count")
    reused = execution.get("reused_item_count")
    observed_seconds = execution.get("observed_seconds")
    if (
        not isinstance(newly_executed, int)
        or isinstance(newly_executed, bool)
        or not 0 <= newly_executed <= len(receipts)
        or reused != len(receipts) - newly_executed
        or not isinstance(observed_seconds, int | float)
        or observed_seconds < 0
    ):
        raise ValueError("full-generations-wave I1 observed execution accounting drifted")
    shard_rows = value.get("balanced_planning_shards")
    if not isinstance(shard_rows, list) or len(shard_rows) != INTERNAL_SHARD_COUNT:
        raise ValueError("full-generations-wave I1 planning shard inventory drifted")
    partitions = _balanced_partitions(works)
    eligible_keys = {work.key for work in eligible_works}
    for index, (row, partition) in enumerate(zip(shard_rows, partitions, strict=True)):
        eligible_partition = tuple(work for work in partition if work.key in eligible_keys)
        pruned_partition = tuple(work for work in partition if work.key not in eligible_keys)
        if not partition:
            expected_status = "empty"
        elif not eligible_partition:
            expected_status = "pruned"
        else:
            expected_status = "complete"
        if isinstance(row, Mapping):
            _require_exact_fields(
                row,
                {
                    "planning_shard_id",
                    "planning_shard_index",
                    "partition_pinned",
                    "work_item_count",
                    "eligible_item_count",
                    "planned_serial_seconds",
                    "eligible_planned_serial_seconds",
                    "pruned_planned_serial_seconds",
                    "executed_item_count",
                    "skipped_item_count",
                    "observed_seconds",
                    "status",
                },
                "full-generations-wave I1 planning shard",
            )
        if (
            not isinstance(row, Mapping)
            or row.get("planning_shard_id") != f"i1-{index:02d}"
            or row.get("planning_shard_index") != index
            or row.get("partition_pinned") is not False
            or row.get("work_item_count") != len(partition)
            or row.get("eligible_item_count") != len(eligible_partition)
            or row.get("planned_serial_seconds") != sum(_work_seconds(work) for work in partition)
            or row.get("eligible_planned_serial_seconds")
            != sum(_work_seconds(work) for work in eligible_partition)
            or row.get("pruned_planned_serial_seconds")
            != sum(_work_seconds(work) for work in pruned_partition)
            or row.get("executed_item_count") != sum(work.key in receipts for work in partition)
            or row.get("skipped_item_count") != len(pruned_partition)
            or not isinstance(row.get("observed_seconds"), int | float)
            or row["observed_seconds"] < 0
            or row.get("status") != expected_status
        ):
            raise ValueError("full-generations-wave I1 planning shard drifted")
    if sum(float(row["observed_seconds"]) for row in shard_rows) != float(observed_seconds):
        raise ValueError("full-generations-wave I1 planning shard timing drifted")


def run_integration(*, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    path = _integration_path(root)
    if path.is_file():
        existing = _read_object(path)
        validate_integration(existing, root=root)
        return existing
    predecessor_binding, initially_eligible, eligible = _i1_eligible(root)
    works = integration_work_items()
    if eligible:
        receipts, durations, newly_executed = _execute_pending(_integration_raw_root(root), works)
    else:
        if _load_receipts(_integration_raw_root(root), works):
            raise ValueError("full-generations-wave pruned I1 already has a raw receipt")
        receipts, durations, newly_executed = {}, {}, 0
    value = _build_integration_result(
        root=root,
        predecessor_binding=predecessor_binding,
        initially_eligible=initially_eligible,
        eligible=eligible,
        receipts=receipts,
        durations=durations,
        newly_executed=newly_executed,
    )
    validate_integration(value, root=root)
    atomic_write_json(path, value)
    return value


def build_integration_classification(*, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = _integration_path(root)
    integration = _read_object(path)
    validate_integration(integration, root=root)
    continued = integration["lane_result"]["continue_lane"] is True
    core = {
        "schema": INTEGRATION_CLASSIFICATION_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "integration": _binding(path, integration, "integration_sha256"),
        "classification": "mechanics_ok" if continued else "null_safe_prune",
        "continue_i1": continued,
        "executed_item_count": integration["execution"]["executed_item_count"],
        "observed_seconds": integration["execution"]["observed_seconds"],
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "classification_sha256": canonical_sha256(core)}


def validate_integration_classification(
    value: Mapping[str, Any],
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "integration",
            "classification",
            "continue_i1",
            "executed_item_count",
            "observed_seconds",
            "independent_scientific_confirmation",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "classification_sha256",
        },
        "full-generations-wave I1 classification",
    )
    _validate_seal(
        value,
        "classification_sha256",
        "full-generations-wave I1 classification",
    )
    _validate_timestamp(value.get("created_at"), "full-generations-wave I1 classification")
    path, integration = _load_binding(
        value.get("integration"),
        "integration_sha256",
        "full-generations-wave I1 classification integration",
    )
    if path != _integration_path(root):
        raise ValueError("full-generations-wave I1 classification integration path drifted")
    validate_integration(integration, root=root)
    continued = integration["lane_result"]["continue_lane"] is True
    if (
        value.get("schema") != INTEGRATION_CLASSIFICATION_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("classification") != ("mechanics_ok" if continued else "null_safe_prune")
        or value.get("continue_i1") is not continued
        or value.get("executed_item_count") != integration["execution"]["executed_item_count"]
        or value.get("observed_seconds") != integration["execution"]["observed_seconds"]
        or value.get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("full-generations-wave I1 classification identity or safety drifted")


def classify_integration(*, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    path = _integration_classification_path(root)
    if path.is_file():
        existing = _read_object(path)
        validate_integration_classification(existing, root=root)
        return existing
    value = build_integration_classification(root=root)
    validate_integration_classification(value, root=root)
    atomic_write_json(path, value)
    return value


def _validated_wave_graph(
    root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    root = Path(root).resolve()
    context = _validated_context_for_epoch(root, 0)
    rows: list[tuple[Path, dict[str, Any]]] = []
    for epoch_index in range(len(EPOCH_IDS)):
        path = _classification_path(root, epoch_index)
        classification = _read_object(path)
        routing = _validate_classification_with_context(
            classification,
            epoch_index,
            root=root,
            context=context,
        )
        rows.append((path, classification))
        context = _WaveValidationContext(
            predecessor_binding={
                "kind": "full_generations_classification",
                "authority": _binding(path, classification, "classification_sha256"),
            },
            routing=routing,
            parent_classification_sha256=str(classification["classification_sha256"]),
        )
    return rows


def build_result(*, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    gates = []
    for gate_index, gate_id in enumerate(GATE_IDS):
        path = _gate_path(root, gate_id)
        gate = _read_object(path)
        validate_gate(gate, gate_index, root=root)
        gates.append({"gate_id": gate_id, **_binding(path, gate, "gate_sha256")})
    waves = []
    executed = 0
    skipped = 0
    observed = 0.0
    for epoch_index, (path, classification) in enumerate(_validated_wave_graph(root)):
        waves.append(
            {
                "epoch_id": EPOCH_IDS[epoch_index],
                "cycle_index": EPOCH_CYCLES[epoch_index],
                "categories": classification["category_bindings"],
                "classification": _binding(path, classification, "classification_sha256"),
                "routing": classification["routing"],
                "executed_item_count": classification["summary"]["executed_item_count"],
                "skipped_item_count": classification["summary"]["skipped_item_count"],
                "observed_seconds": classification["summary"]["observed_seconds"],
            }
        )
        executed += int(classification["summary"]["executed_item_count"])
        skipped += int(classification["summary"]["skipped_item_count"])
        observed += float(classification["summary"]["observed_seconds"])
    integration_path = _integration_path(root)
    integration = _read_object(integration_path)
    validate_integration(integration, root=root)
    integration_classification_path = _integration_classification_path(root)
    integration_classification = _read_object(integration_classification_path)
    validate_integration_classification(integration_classification, root=root)
    executed += int(integration["execution"]["executed_item_count"])
    skipped += int(integration["execution"]["skipped_item_count"])
    observed += float(integration["execution"]["observed_seconds"])
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "grid": {
            "gate_ids": list(GATE_IDS),
            "epoch_ids": list(EPOCH_IDS),
            "fresh_cycle_indices": list(EPOCH_CYCLES),
            "category_ids": list(CATEGORY_IDS),
            "category_lanes": {key: list(value) for key, value in CATEGORY_LANES.items()},
            "new_lane_ids": list(NEW_LANE_IDS),
            "i1_dependency_lane_ids": list(I1_DEPENDENCIES),
            "balanced_planning_shards_per_compute_capsule": INTERNAL_SHARD_COUNT,
            "manifest_capsule_count": CAPSULE_COUNT,
            "maximum_raw_receipt_count": MAXIMUM_RAW_RECEIPT_COUNT,
        },
        "gates": gates,
        "waves": waves,
        "integration": _binding(integration_path, integration, "integration_sha256"),
        "integration_classification": _binding(
            integration_classification_path,
            integration_classification,
            "classification_sha256",
        ),
        "execution": {
            "maximum_planned_serial_seconds": planned_program_compute_seconds(),
            "maximum_planned_serial_hours": planned_serial_hours(),
            "maximum_ideal_worker_hours": planned_ideal_worker_hours(),
            "executed_item_count": executed,
            "skipped_item_count": skipped,
            "observed_seconds": observed,
            "compute_started": executed > 0,
        },
        "decision": {
            "full_generations_mechanics_complete": True,
            "new_mechanism_lanes_admitted": True,
            "independent_artifact_verification_pending": True,
            "independent_scientific_confirmation": False,
            "d1_redesign_efficacy_executed": False,
        },
        "interpretation_limit": (
            "Fresh deterministic mechanics receipts are robustness and wiring evidence only."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_result(
    value: Mapping[str, Any],
    *,
    root: Path = DEFAULT_ROOT,
) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "grid",
            "gates",
            "waves",
            "integration",
            "integration_classification",
            "execution",
            "decision",
            "interpretation_limit",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "result_sha256",
        },
        "full-generations-wave result",
    )
    _validate_seal(value, "result_sha256", "full-generations-wave result")
    grid = value.get("grid")
    expected_grid = {
        "gate_ids": list(GATE_IDS),
        "epoch_ids": list(EPOCH_IDS),
        "fresh_cycle_indices": list(EPOCH_CYCLES),
        "category_ids": list(CATEGORY_IDS),
        "category_lanes": {key: list(row) for key, row in CATEGORY_LANES.items()},
        "new_lane_ids": list(NEW_LANE_IDS),
        "i1_dependency_lane_ids": list(I1_DEPENDENCIES),
        "balanced_planning_shards_per_compute_capsule": INTERNAL_SHARD_COUNT,
        "manifest_capsule_count": CAPSULE_COUNT,
        "maximum_raw_receipt_count": MAXIMUM_RAW_RECEIPT_COUNT,
    }
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or grid != expected_grid
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("full-generations-wave result identity or safety drifted")
    gates = value.get("gates")
    if not isinstance(gates, list) or len(gates) != len(GATE_IDS):
        raise ValueError("full-generations-wave result gate inventory drifted")
    for gate_index, (gate_id, binding) in enumerate(zip(GATE_IDS, gates, strict=True)):
        if not isinstance(binding, Mapping):
            raise ValueError("full-generations-wave result gate ordering drifted")
        _require_exact_fields(
            binding,
            {"gate_id", "path", "file_sha256", "gate_sha256"},
            "full-generations-wave result gate binding",
        )
        if binding.get("gate_id") != gate_id:
            raise ValueError("full-generations-wave result gate ordering drifted")
        path, gate = _load_binding(
            {key: item for key, item in binding.items() if key != "gate_id"},
            "gate_sha256",
            "full-generations-wave result gate",
        )
        if path != _gate_path(root, gate_id):
            raise ValueError("full-generations-wave result gate path drifted")
        validate_gate(gate, gate_index, root=root)
    waves = value.get("waves")
    if not isinstance(waves, list) or len(waves) != len(EPOCH_IDS):
        raise ValueError("full-generations-wave result wave inventory drifted")
    expected_graph = _validated_wave_graph(Path(root).resolve())
    executed = 0
    skipped = 0
    observed = 0.0
    for epoch_index, (row, (classification_path, classification)) in enumerate(
        zip(waves, expected_graph, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ValueError("full-generations-wave result wave is invalid")
        _require_exact_fields(
            row,
            {
                "epoch_id",
                "cycle_index",
                "categories",
                "classification",
                "routing",
                "executed_item_count",
                "skipped_item_count",
                "observed_seconds",
            },
            "full-generations-wave result wave",
        )
        if row.get("categories") != classification["category_bindings"]:
            raise ValueError("full-generations-wave result category inventory drifted")
        path, classification = _load_binding(
            row.get("classification"),
            "classification_sha256",
            "full-generations-wave result classification",
        )
        if path != classification_path:
            raise ValueError("full-generations-wave result classification path drifted")
        if (
            row.get("epoch_id") != EPOCH_IDS[epoch_index]
            or row.get("cycle_index") != EPOCH_CYCLES[epoch_index]
            or row.get("routing") != classification["routing"]
            or row.get("executed_item_count") != classification["summary"]["executed_item_count"]
            or row.get("skipped_item_count") != classification["summary"]["skipped_item_count"]
            or row.get("observed_seconds") != classification["summary"]["observed_seconds"]
        ):
            raise ValueError("full-generations-wave result wave summary drifted")
        executed += int(classification["summary"]["executed_item_count"])
        skipped += int(classification["summary"]["skipped_item_count"])
        observed += float(classification["summary"]["observed_seconds"])
    integration_path, integration = _load_binding(
        value.get("integration"),
        "integration_sha256",
        "full-generations-wave result I1",
    )
    if integration_path != _integration_path(root):
        raise ValueError("full-generations-wave result I1 path drifted")
    validate_integration(integration, root=root)
    classification_path, integration_classification = _load_binding(
        value.get("integration_classification"),
        "classification_sha256",
        "full-generations-wave result I1 classification",
    )
    if classification_path != _integration_classification_path(root):
        raise ValueError("full-generations-wave result I1 classification path drifted")
    validate_integration_classification(integration_classification, root=root)
    executed += int(integration["execution"]["executed_item_count"])
    skipped += int(integration["execution"]["skipped_item_count"])
    observed += float(integration["execution"]["observed_seconds"])
    if value.get("execution") != {
        "maximum_planned_serial_seconds": planned_program_compute_seconds(),
        "maximum_planned_serial_hours": planned_serial_hours(),
        "maximum_ideal_worker_hours": planned_ideal_worker_hours(),
        "executed_item_count": executed,
        "skipped_item_count": skipped,
        "observed_seconds": observed,
        "compute_started": executed > 0,
    }:
        raise ValueError("full-generations-wave result execution summary drifted")
    decision = value.get("decision")
    if decision != {
        "full_generations_mechanics_complete": True,
        "new_mechanism_lanes_admitted": True,
        "independent_artifact_verification_pending": True,
        "independent_scientific_confirmation": False,
        "d1_redesign_efficacy_executed": False,
    }:
        raise ValueError("full-generations-wave result decision drifted")
    if value.get("interpretation_limit") != (
        "Fresh deterministic mechanics receipts are robustness and wiring evidence only."
    ):
        raise ValueError("full-generations-wave result interpretation boundary drifted")


def aggregate(*, root: Path = DEFAULT_ROOT, output: Path = DEFAULT_RESULT) -> dict[str, Any]:
    output = Path(output).resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("full-generations-wave result is outside the repository")
    if output.is_file():
        existing = _read_object(output)
        validate_result(existing, root=root)
        return existing
    value = build_result(root=root)
    validate_result(value, root=root)
    atomic_write_json(output, value)
    return value


def validate_report_receipt(value: Mapping[str, Any]) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "result",
            "verification",
            "report",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "receipt_sha256",
        },
        "full-generations-wave report receipt",
    )
    _validate_seal(value, "receipt_sha256", "full-generations-wave report receipt")
    if (
        value.get("schema") != REPORT_RECEIPT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("full-generations-wave report receipt identity or safety drifted")
    for label in ("result", "verification", "report"):
        binding = value.get(label)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256"}:
            raise ValueError(f"full-generations-wave report {label} binding drifted")
        path = _repo_path(binding.get("path"), f"full-generations-wave report {label}")
        if not path.is_file() or path.is_symlink() or binding.get("file_sha256") != sha256_file(path):
            raise ValueError(f"full-generations-wave report {label} bytes drifted")


def render_report(
    *,
    result_path: Path = DEFAULT_RESULT,
    verification_path: Path = DEFAULT_VERIFICATION,
    report_path: Path = DEFAULT_REPORT,
    receipt_path: Path = DEFAULT_REPORT_RECEIPT,
) -> dict[str, Any]:
    from mop.studies import generation1_full_generations_wave_verify as verifier

    result_path, verification_path, report_path, receipt_path = (
        Path(path).resolve() for path in (result_path, verification_path, report_path, receipt_path)
    )
    result = _read_object(result_path)
    validate_result(result)
    verification = _read_object(verification_path)
    verifier.validate_verification(verification)
    source = verification.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("path") != _relative(result_path)
        or source.get("file_sha256") != sha256_file(result_path)
        or source.get("result_sha256") != result.get("result_sha256")
    ):
        raise ValueError("full-generations-wave verification does not bind the requested result")
    if receipt_path.is_file():
        existing = _read_object(receipt_path)
        validate_report_receipt(existing)
        return existing
    lines = [
        "# Generation 1 full-generations mechanics wave",
        "",
        (
            "Five admission gates carry the categorized batch-wave v1 authority, its retired frozen "
            "D1 and no-candidate redesign freeze, rescreen the D1-v2 catalog, admit three redesigned "
            "mechanism lanes (G1-U1, G1-N1, G1-P1R), and freeze routing. Fourteen fresh mechanics "
            "waves then run seven categories through a dynamic process pool capped at sixteen workers "
            "and described by eight balanced planning shards, then seal one serial classifier. I1 "
            "runs once after W21 with its dependency set substituted (G1-P1 replaced by G1-P1R)."
        ),
        "",
        "| Wave | Cycle | Executed items | Skipped items | Observed seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["waves"]:
        lines.append(
            f"| {row['epoch_id']} | {row['cycle_index']} | {row['executed_item_count']} | "
            f"{row['skipped_item_count']} | {row['observed_seconds']:.3f} |"
        )
    lines += [
        "",
        f"Maximum envelope: {planned_serial_hours():.3f} serial hours; "
        f"{planned_ideal_worker_hours():.3f} ideal worker hours.",
        f"Maximum raw receipts: {MAXIMUM_RAW_RECEIPT_COUNT}.",
        "",
        "Old D1 was not rerun and the pruned G1-P1 lane never reappears. G1-D1 mechanics is retained "
        "in dispatch_redesign, while D1-v2 efficacy remains separately gated and unauthorized without "
        "real candidate evidence.",
        "",
        f"Result SHA-256: `{result['result_sha256']}`",
        f"Verification SHA-256: `{verification['verification_sha256']}`",
        "",
    ]
    encoded = ("\n".join(lines)).encode("utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(report_path)
    core = {
        "schema": REPORT_RECEIPT_SCHEMA,
        "program_id": PROGRAM_ID,
        "result": {"path": _relative(result_path), "file_sha256": sha256_file(result_path)},
        "verification": {
            "path": _relative(verification_path),
            "file_sha256": sha256_file(verification_path),
        },
        "report": {
            "path": _relative(report_path),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    validate_report_receipt(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def build_release_audit(*, paths: Any = None) -> dict[str, Any]:

    from mop.studies import generation1_release_audit as release_audit

    release_paths = paths if paths is not None else release_audit.ReleasePaths.defaults(REPO_ROOT)
    report = release_audit.audit_generation1_release(release_paths)
    if not isinstance(report, Mapping):
        raise ValueError("full-generations-wave release audit did not return a report object")
    audit_exit_code = 0 if report.get("release_complete") is True else 1
    core = {
        "schema": RELEASE_AUDIT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "created_at": _now(),
        "audit_program_schema": release_audit.SCHEMA,
        "audit_exit_code": audit_exit_code,
        "audit_report": dict(report),
        "advisory": True,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    return {**core, "release_audit_sha256": canonical_sha256(core)}


def validate_release_audit(value: Mapping[str, Any]) -> None:
    _require_exact_fields(
        value,
        {
            "schema",
            "program_id",
            "claim_scope",
            "created_at",
            "audit_program_schema",
            "audit_exit_code",
            "audit_report",
            "advisory",
            "complete",
            "problems",
            "activation_allowed",
            "scientific_promotion",
            "independent_scientific_confirmation",
            "release_audit_sha256",
        },
        "full-generations-wave release audit",
    )
    _validate_seal(value, "release_audit_sha256", "full-generations-wave release audit")
    _validate_timestamp(value.get("created_at"), "full-generations-wave release audit")
    audit_exit_code = value.get("audit_exit_code")
    if (
        value.get("schema") != RELEASE_AUDIT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or not isinstance(audit_exit_code, int)
        or isinstance(audit_exit_code, bool)
        or audit_exit_code not in (0, 1)
        or not isinstance(value.get("audit_report"), Mapping)
        or not isinstance(value.get("audit_program_schema"), str)
        or value.get("advisory") is not True
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("full-generations-wave release audit identity or safety drifted")


def write_release_audit(
    *,
    output: Path = DEFAULT_RELEASE_AUDIT,
    paths: Any = None,
) -> dict[str, Any]:
    output = Path(output).resolve()
    if not output.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("full-generations-wave release audit is outside the repository")
    if output.is_file():
        existing = _read_object(output)
        validate_release_audit(existing)
        return existing
    value = build_release_audit(paths=paths)
    validate_release_audit(value)
    atomic_write_json(output, value)
    return value


__all__ = [
    "CAPSULE_COUNT",
    "CATEGORIES",
    "CATEGORY_IDS",
    "CATEGORY_LANES",
    "CATEGORY_NAMES",
    "CATEGORY_SCHEMA",
    "CLASSIFICATION_SCHEMA",
    "CLAIM_SCOPE",
    "CONSTRUCTION_VEC_SPEEDUP",
    "DEFAULT_PARENT_REPORT_RECEIPT",
    "DEFAULT_PARENT_RESULT",
    "DEFAULT_PARENT_VERIFICATION",
    "DEFAULT_RELEASE_AUDIT",
    "DEFAULT_REPORT",
    "DEFAULT_REPORT_RECEIPT",
    "DEFAULT_RESULT",
    "DEFAULT_ROOT",
    "DEFAULT_VERIFICATION",
    "EPOCH_CYCLES",
    "EPOCH_IDS",
    "GATE_IDS",
    "GATE_SCHEMA",
    "I1_DEPENDENCIES",
    "IDLE_WORKERS",
    "INTEGRATION_CLASSIFICATION_SCHEMA",
    "INTEGRATION_SCHEMA",
    "INTERNAL_SHARD_COUNT",
    "MAXIMUM_RAW_RECEIPT_COUNT",
    "NEW_LANE_IDS",
    "PROGRAM_ID",
    "PROGRAM_MANIFEST",
    "RELEASE_AUDIT_SCHEMA",
    "REPORT_RECEIPT_SCHEMA",
    "RESULT_SCHEMA",
    "VERIFICATION_CLAIM_SCOPE",
    "VERIFICATION_SCHEMA",
    "WaveWorkItem",
    "aggregate",
    "build_classification",
    "build_gate",
    "build_integration_classification",
    "build_release_audit",
    "build_result",
    "category_planned_serial_seconds",
    "category_work_items",
    "classify_epoch",
    "classify_integration",
    "integration_work_items",
    "materialize_gate",
    "planned_epoch_compute_seconds",
    "planned_ideal_worker_hours",
    "planned_program_compute_seconds",
    "planned_serial_hours",
    "render_report",
    "run_category",
    "run_integration",
    "validate_category",
    "validate_classification",
    "validate_gate",
    "validate_integration",
    "validate_integration_classification",
    "validate_release_audit",
    "validate_report_receipt",
    "validate_result",
    "write_release_audit",
]
