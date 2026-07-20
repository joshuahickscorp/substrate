
from __future__ import annotations

import datetime as dt
import json
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
from mop.process_labels import set_process_label
from mop.studies import generation1_c3_d1_frozen_queue as d1
from mop.studies import generation1_c3_router_redesign as redesign
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studies.generation1_c3_dispatch import atomic_write_json, canonical_sha256, sha256_file
from mop.studies.generation1_c3_dispatch_queue import hawking_mode

PROGRAM_ID = "generation1-consolidated-final-campaign-v1"
STATUS_SCHEMA = "mop-generation1-consolidated-final-status/v1"
MANIFEST_SCHEMA = "mop-generation1-consolidated-final-manifest/v1"
AUDIT_SCHEMA = "mop-generation1-consolidated-regeneration-audit/v1"
RESULT_SCHEMA = "mop-generation1-consolidated-final-result/v1"
CLAIM_SCOPE = (
    "deterministic source regeneration and result-gated fresh stress; "
    "no independent scientific confirmation or activation"
)

DEFAULT_ROOT = REPO_ROOT / "runs/generation1" / PROGRAM_ID
DEFAULT_STATUS = DEFAULT_ROOT / "current_status.json"
DEFAULT_MANIFEST = REPO_ROOT / "proof/GENERATION1_CONSOLIDATED_FINAL_MANIFEST.json"
DEFAULT_RESULT = REPO_ROOT / "proof/GENERATION1_CONSOLIDATED_FINAL_RESULT.json"

FRESH_D1_CYCLES = 2
FRESH_MECHANICS_CYCLES = 2
D1_CYCLE_STRIDE = 10_000_000
MECHANICS_CYCLE_STRIDE = 300_000_000
MECHANICS_FRESH_BASE = 1_000_000_000
WAIT_SECONDS = 120
RETRY_LIMIT = 3
D1_PLANNED_RUNG_SECONDS = 115.0


@dataclass(frozen=True, slots=True)
class WorkItem:
    key: str
    kind: str
    source_index: int
    cycle: int | None
    preflight: bool


def planned_max_compute_seconds() -> float:
    d1_seconds = d1.DEFAULT_RUNG_COUNT * (1 + FRESH_D1_CYCLES) * D1_PLANNED_RUNG_SECONDS
    mechanics_cycle = sum(
        item.seed_count * mechanics.PLANNED_SECONDS_PER_SEED[item.mechanism] for item in mechanics.WORK_ITEMS
    )
    return d1_seconds + mechanics_cycle * (1 + FRESH_MECHANICS_CYCLES)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def build_manifest() -> dict[str, Any]:
    source_paths = (
        Path(__file__).resolve(),
        Path(d1.__file__).resolve(),
        Path(mechanics.__file__).resolve(),
        Path(redesign.__file__).resolve(),
    )
    core = {
        "schema": MANIFEST_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "authorities": {
            "d1": {
                "program_id": d1.PROGRAM_ID,
                "result_path": _relative(d1.DEFAULT_RESULT),
                "rung_count": d1.DEFAULT_RUNG_COUNT,
                "screen_binding": d1.screen_binding(),
            },
            "successor_mechanics": {
                "program_id": mechanics.PROGRAM_ID,
                "result_path": _relative(mechanics.DEFAULT_RESULT),
                "planned_rung_count": len(mechanics.WORK_ITEMS),
                "lane_order": [lane.lane_id for lane in mechanics.LANES],
                "canary_seeds": mechanics.CANARY_SEEDS,
            },
        },
        "conditional_graph": [
            "consume_each_clean authority as soon as it is available",
            "regenerate a cheap preflight before exhaustive source regeneration",
            "run D1 fresh cycles only if producer and challenge repeat the frozen criteria",
            "run mechanics fresh cycles only for lanes whose canary gate passed",
            "aggregate without changing any confirmation or activation gate",
        ],
        "fresh_expansion": {
            "d1_cycles": FRESH_D1_CYCLES,
            "mechanics_cycles": FRESH_MECHANICS_CYCLES,
            "d1_cycle_stride": D1_CYCLE_STRIDE,
            "mechanics_fresh_base": MECHANICS_FRESH_BASE,
            "mechanics_cycle_stride": MECHANICS_CYCLE_STRIDE,
        },
        "execution": {
            "waiting_time_limit": None,
            "retry_limit": RETRY_LIMIT,
            "hawking_active_workers": 1,
            "hawking_idle_workers": 8,
            "checkpoint_granularity": "one atomic artifact per work item",
            "planned_max_compute_seconds": planned_max_compute_seconds(),
            "planned_max_serial_hours": planned_max_compute_seconds() / 3_600,
            "planned_max_hawking_idle_hours_at_eight_workers": (planned_max_compute_seconds() / 8 / 3_600),
        },
        "source_files": [{"path": _relative(path), "sha256": sha256_file(path)} for path in source_paths],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def validate_manifest(value: Mapping[str, Any]) -> None:
    core = {key: row for key, row in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != canonical_sha256(core):
        raise ValueError("final campaign manifest seal drifted")
    if (
        value.get("schema") != MANIFEST_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("claim_scope") != CLAIM_SCOPE
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("final campaign manifest identity or safety drifted")


def _load_d1_authority() -> dict[str, Any] | None:
    if not d1.DEFAULT_RESULT.is_file():
        return None
    value = _read_object(d1.DEFAULT_RESULT)
    d1.validate_aggregate(value)
    return value


def _load_mechanics_authority() -> dict[str, Any] | None:
    if not mechanics.DEFAULT_RESULT.is_file():
        return None
    value = _read_object(mechanics.DEFAULT_RESULT)
    mechanics.validate_aggregate(value)
    return value


def _d1_pattern_repeated(authority: Mapping[str, Any]) -> bool:
    return authority.get("decision", {}).get("frozen_pattern_repeated") is True


def _mechanics_pruned(authority: Mapping[str, Any]) -> set[str]:
    rows = authority.get("decision", {}).get("pruned_lanes", [])
    if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
        raise ValueError("mechanics pruned-lane decision is invalid")
    return set(rows)


def _mechanics_source_items(authority: Mapping[str, Any]) -> tuple[mechanics.WorkItem, ...]:
    pruned = _mechanics_pruned(authority)
    return tuple(
        item for item in mechanics.WORK_ITEMS if item.phase == "canary" or item.lane_id not in pruned
    )


def build_work_items(
    d1_authority: Mapping[str, Any] | None,
    mechanics_authority: Mapping[str, Any] | None,
) -> tuple[WorkItem, ...]:
    rows: list[WorkItem] = []
    if d1_authority is not None:
        for index in range(d1.DEFAULT_RUNG_COUNT):
            phase, local = d1.phase_of(index)
            rows.append(
                WorkItem(
                    key=f"d1_source_{phase}_{local:03d}",
                    kind="d1_source",
                    source_index=index,
                    cycle=None,
                    preflight=index == 0,
                )
            )
        if _d1_pattern_repeated(d1_authority):
            for cycle in range(FRESH_D1_CYCLES):
                for index in range(d1.DEFAULT_RUNG_COUNT):
                    phase, local = d1.phase_of(index)
                    rows.append(
                        WorkItem(
                            key=f"d1_fresh_c{cycle}_{phase}_{local:03d}",
                            kind="d1_fresh",
                            source_index=index,
                            cycle=cycle,
                            preflight=False,
                        )
                    )
    if mechanics_authority is not None:
        source_items = _mechanics_source_items(mechanics_authority)
        first_by_mechanism: dict[str, int] = {}
        for item in source_items:
            first_by_mechanism.setdefault(item.mechanism, item.index)
            rows.append(
                WorkItem(
                    key=f"mechanics_source_{item.capsule_id}",
                    kind="mechanics_source",
                    source_index=item.index,
                    cycle=None,
                    preflight=first_by_mechanism[item.mechanism] == item.index,
                )
            )
        pruned = _mechanics_pruned(mechanics_authority)
        fresh_sources = tuple(item for item in source_items if item.lane_id not in pruned)
        for cycle in range(FRESH_MECHANICS_CYCLES):
            for item in fresh_sources:
                rows.append(
                    WorkItem(
                        key=f"mechanics_fresh_c{cycle}_{item.capsule_id}",
                        kind="mechanics_fresh",
                        source_index=item.index,
                        cycle=cycle,
                        preflight=False,
                    )
                )
    return tuple(rows)


def fresh_d1_config(work: WorkItem) -> dict[str, Any]:
    if work.kind != "d1_fresh" or work.cycle is None:
        raise ValueError("fresh D1 config requires a fresh D1 work item")
    phase, local = d1.phase_of(work.source_index)
    cycle_base = 500_000_001 + work.cycle * D1_CYCLE_STRIDE
    phase_offset = 0 if phase == "producer" else 2_000_000
    router_base = 700_000_001 + work.cycle * D1_CYCLE_STRIDE + phase_offset
    return redesign.redesign_config(
        train_seed_start=cycle_base + phase_offset + local * d1.SEEDS_PER_RUNG,
        train_seed_count=d1.SEEDS_PER_RUNG,
        heldout_seed_start=cycle_base + phase_offset + 1_000_000 + local * d1.SEEDS_PER_RUNG,
        heldout_seed_count=d1.SEEDS_PER_RUNG,
        difficulty_indices=(0, 1, 2, 3, 4),
        n_train=720,
        n_test=240,
        n_classes=10,
        dim=64,
        actor_epochs=6,
        router_training_seed=router_base + local * 100_003,
        variants=(d1.frozen_variant(),),
    )


def fresh_mechanics_item(work: WorkItem) -> mechanics.WorkItem:
    if work.kind != "mechanics_fresh" or work.cycle is None:
        raise ValueError("fresh mechanics item requires fresh mechanics work")
    source = mechanics.WORK_ITEMS[work.source_index]
    return mechanics.WorkItem(
        index=source.index,
        lane_id=source.lane_id,
        mechanism=source.mechanism,
        phase=f"fresh_c{work.cycle}_{source.phase}",
        rung_index=source.rung_index,
        seed_start=(source.seed_start + MECHANICS_FRESH_BASE + work.cycle * MECHANICS_CYCLE_STRIDE),
        seed_count=source.seed_count,
    )


def _artifact_path(root: Path, work: WorkItem) -> Path:
    return root / work.kind / f"{work.key}.json"


def _audit_receipt(work: WorkItem, source_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema": AUDIT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "work": {
            "key": work.key,
            "kind": work.kind,
            "source_index": work.source_index,
            "cycle": work.cycle,
            "preflight": work.preflight,
        },
        "source": {
            "path": _relative(source_path),
            "file_sha256": sha256_file(source_path),
            "result_sha256": source["result_sha256"],
        },
        "regenerated_result_sha256": source["result_sha256"],
        "canonical_regeneration_match": True,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


def _execute_work(work: WorkItem, root_string: str) -> tuple[str, float]:
    set_process_label(f"mop-final-{work.kind[:8]}-{work.source_index:04d}")
    with suppress(OSError):
        os.nice(10)
    started = time.perf_counter()
    target = _artifact_path(Path(root_string), work)
    if work.kind == "d1_source":
        source_path = d1._rung_path(d1.DEFAULT_ROOT, work.source_index)
        source = _read_object(source_path)
        config = d1.rung_config(work.source_index)
        redesign.validate_result(source, config)
        regenerated = redesign.run_redesign(config)
        redesign.validate_result(regenerated, config)
        if canonical_sha256(regenerated) != canonical_sha256(source):
            raise ValueError("D1 canonical regeneration mismatch")
        payload = _audit_receipt(work, source_path, source)
    elif work.kind == "d1_fresh":
        config = fresh_d1_config(work)
        payload = redesign.run_redesign(config)
        redesign.validate_result(payload, config)
    elif work.kind == "mechanics_source":
        item = mechanics.WORK_ITEMS[work.source_index]
        source_path = mechanics._item_path(mechanics.DEFAULT_ROOT, item)
        source = _read_object(source_path)
        mechanics.validate_rung(source, item)
        regenerated = mechanics._run_item_payload(item)
        mechanics.validate_rung(regenerated, item)
        if canonical_sha256(regenerated) != canonical_sha256(source):
            raise ValueError("mechanics canonical regeneration mismatch")
        payload = _audit_receipt(work, source_path, source)
    elif work.kind == "mechanics_fresh":
        item = fresh_mechanics_item(work)
        payload = mechanics._run_item_payload(item)
        mechanics.validate_rung(payload, item)
    else:
        raise ValueError(f"unsupported final work kind: {work.kind}")
    atomic_write_json(target, payload)
    return work.key, time.perf_counter() - started


def _validate_artifact(value: Mapping[str, Any], work: WorkItem) -> None:
    if work.kind.endswith("_source"):
        core = {key: row for key, row in value.items() if key != "audit_sha256"}
        expected_work = {
            "key": work.key,
            "kind": work.kind,
            "source_index": work.source_index,
            "cycle": work.cycle,
            "preflight": work.preflight,
        }
        if value.get("audit_sha256") != canonical_sha256(core):
            raise ValueError("final source audit seal drifted")
        if (
            value.get("schema") != AUDIT_SCHEMA
            or value.get("work") != expected_work
            or value.get("canonical_regeneration_match") is not True
            or value.get("complete") is not True
            or value.get("problems") != []
            or value.get("activation_allowed") is not False
            or value.get("scientific_promotion") is not False
        ):
            raise ValueError("final source audit identity or safety drifted")
    elif work.kind == "d1_fresh":
        redesign.validate_result(value, fresh_d1_config(work))
    elif work.kind == "mechanics_fresh":
        mechanics.validate_rung(value, fresh_mechanics_item(work))
    else:
        raise ValueError(f"unsupported final artifact kind: {work.kind}")


def _load_receipts(root: Path, works: Sequence[WorkItem]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for work in works:
        path = _artifact_path(root, work)
        if not path.is_file():
            continue
        try:
            value = _read_object(path)
            _validate_artifact(value, work)
            receipts[work.key] = value
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return receipts


def _estimate_seconds(work: WorkItem) -> float:
    if work.kind.startswith("d1_"):
        return D1_PLANNED_RUNG_SECONDS
    item = (
        mechanics.WORK_ITEMS[work.source_index]
        if work.kind == "mechanics_source"
        else fresh_mechanics_item(work)
    )
    return item.seed_count * mechanics.PLANNED_SECONDS_PER_SEED[item.mechanism]


def _authority_max_seconds(label: str) -> float:
    if label == "d1":
        return d1.DEFAULT_RUNG_COUNT * (1 + FRESH_D1_CYCLES) * D1_PLANNED_RUNG_SECONDS
    if label == "mechanics":
        cycle = sum(
            item.seed_count * mechanics.PLANNED_SECONDS_PER_SEED[item.mechanism]
            for item in mechanics.WORK_ITEMS
        )
        return cycle * (1 + FRESH_MECHANICS_CYCLES)
    raise ValueError(f"unsupported authority label: {label}")


def _prerequisite_eta(path: Path) -> float:
    try:
        value = _read_object(path)
        seconds = value.get("adaptive_execution", {}).get("eta_seconds")
        return float(seconds) if isinstance(seconds, int | float) and not isinstance(seconds, bool) else 0.0
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0


def _prerequisite_row(path: Path, value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"attempts": 0, "artifacts": []}
    return {
        "attempts": 0,
        "returncode": 0,
        "finished_at": dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat(
            timespec="seconds"
        ),
        "artifacts": [
            {
                "path": _relative(path),
                "sha256": sha256_file(path),
                "all_ok": True,
                "schema": value["schema"],
                "problems": [],
            }
        ],
    }


def _status(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    d1_authority: Mapping[str, Any] | None,
    mechanics_authority: Mapping[str, Any] | None,
    works: Sequence[WorkItem],
    receipts: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[str, int],
    durations: Mapping[str, float],
    state: str,
    mode: str,
    workers: int,
    problems: Sequence[str],
) -> dict[str, Any]:
    capsules = {
        "prerequisite_d1": _prerequisite_row(d1.DEFAULT_RESULT, d1_authority),
        "prerequisite_successor_mechanics": _prerequisite_row(mechanics.DEFAULT_RESULT, mechanics_authority),
    }
    for work in works:
        row: dict[str, Any] = {"attempts": attempts.get(work.key, 0), "artifacts": []}
        if work.key in receipts:
            path = _artifact_path(root, work)
            row.update(
                {
                    "returncode": 0,
                    "finished_at": dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC).isoformat(
                        timespec="seconds"
                    ),
                    "artifacts": [
                        {
                            "path": _relative(path),
                            "sha256": sha256_file(path),
                            "all_ok": True,
                            "schema": receipts[work.key]["schema"],
                            "problems": [],
                        }
                    ],
                }
            )
        capsules[work.key] = row
    pending = [work for work in works if work.key not in receipts]
    unknown_compute = 0.0
    if d1_authority is None:
        unknown_compute += _authority_max_seconds("d1")
    if mechanics_authority is None:
        unknown_compute += _authority_max_seconds("mechanics")
    prerequisite_wait = max(
        _prerequisite_eta(d1.DEFAULT_ROOT / "current_status.json") if d1_authority is None else 0.0,
        _prerequisite_eta(mechanics.DEFAULT_ROOT / "current_status.json")
        if mechanics_authority is None
        else 0.0,
    )
    session = prerequisite_wait + (sum(_estimate_seconds(work) for work in pending) + unknown_compute) / max(
        1, workers
    )
    block = sum(_estimate_seconds(work) for work in pending[:5]) / max(1, workers) if pending else None
    complete = sum(value is not None for value in (d1_authority, mechanics_authority)) + len(receipts)
    average = statistics.fmean(durations.values()) if durations else None
    core = {
        "schema": STATUS_SCHEMA,
        "program_id": PROGRAM_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "state": state,
        "capsules": capsules,
        "prerequisites": {
            "d1": "complete" if d1_authority is not None else "waiting",
            "successor_mechanics": "complete" if mechanics_authority is not None else "waiting",
        },
        "conditional_decisions": {
            "d1_fresh_cycles_enabled": (
                _d1_pattern_repeated(d1_authority) if d1_authority is not None else None
            ),
            "mechanics_pruned_lanes": (
                sorted(_mechanics_pruned(mechanics_authority)) if mechanics_authority is not None else None
            ),
        },
        "adaptive_execution": {
            "mode": mode,
            "workers": workers,
            "average_rung_seconds": average,
            "next_block_seconds": block,
            "eta_seconds": session,
            "waiting_time_limit": None,
            "planned_max_compute_seconds": planned_max_compute_seconds(),
        },
        "counts": {"complete": complete, "total": len(capsules), "remaining": len(capsules) - complete},
        "problems": list(problems),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": canonical_sha256(core)}


def _aggregate_fresh_d1(root: Path, works: Sequence[WorkItem]) -> dict[str, Any]:
    cycles: dict[str, Any] = {}
    for cycle in range(FRESH_D1_CYCLES):
        cycle_works = [work for work in works if work.kind == "d1_fresh" and work.cycle == cycle]
        if not cycle_works:
            continue
        rows = [_read_object(_artifact_path(root, work)) for work in cycle_works]
        cycles[str(cycle)] = {
            "producer": d1.summarize_phase(rows[: d1.RUNGS_PER_PHASE]),
            "challenge": d1.summarize_phase(rows[d1.RUNGS_PER_PHASE :]),
        }
    return cycles


def aggregate(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    d1_authority: Mapping[str, Any],
    mechanics_authority: Mapping[str, Any],
    works: Sequence[WorkItem],
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if {work.key for work in works} != set(receipts):
        raise ValueError("final aggregate requires every conditionally eligible work item")
    source_audits = [work for work in works if work.kind.endswith("_source")]
    fresh_d1 = _aggregate_fresh_d1(root, works)
    fresh_mechanics_counts: dict[str, dict[str, int]] = {}
    for work in works:
        if work.kind != "mechanics_fresh" or work.cycle is None:
            continue
        value = receipts[work.key]
        cycle = fresh_mechanics_counts.setdefault(str(work.cycle), {})
        for verdict, count in value["verdict_counts"].items():
            cycle[verdict] = cycle.get(verdict, 0) + int(count)
    artifact_index = [
        {
            "key": work.key,
            "kind": work.kind,
            "sha256": sha256_file(_artifact_path(root, work)),
        }
        for work in works
    ]
    core = {
        "schema": RESULT_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "manifest": {
            "path": _relative(DEFAULT_MANIFEST),
            "file_sha256": sha256_file(DEFAULT_MANIFEST),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "authorities": {
            "d1": {
                "path": _relative(d1.DEFAULT_RESULT),
                "file_sha256": sha256_file(d1.DEFAULT_RESULT),
                "result_sha256": d1_authority["result_sha256"],
                "frozen_pattern_repeated": _d1_pattern_repeated(d1_authority),
            },
            "successor_mechanics": {
                "path": _relative(mechanics.DEFAULT_RESULT),
                "file_sha256": sha256_file(mechanics.DEFAULT_RESULT),
                "result_sha256": mechanics_authority["result_sha256"],
                "pruned_lanes": sorted(_mechanics_pruned(mechanics_authority)),
            },
        },
        "grid": {
            "work_item_count": len(works),
            "source_regeneration_count": len(source_audits),
            "fresh_d1_rung_count": sum(work.kind == "d1_fresh" for work in works),
            "fresh_mechanics_rung_count": sum(work.kind == "mechanics_fresh" for work in works),
        },
        "source_regeneration": {
            "all_canonical_matches": True,
            "count": len(source_audits),
        },
        "fresh_d1_cycles": fresh_d1,
        "fresh_mechanics_verdict_counts": fresh_mechanics_counts,
        "artifact_index": artifact_index,
        "decision": {
            "conditional_final_campaign_complete": True,
            "d1_fresh_expansion_executed": bool(fresh_d1),
            "mechanics_fresh_expansion_executed": True,
            "independent_scientific_verification_complete": False,
            "scientific_confirmation": False,
            "next_action": "interpret_consolidated_result_and_author_independent_verifiers",
        },
        "interpretation_limit": (
            "Full source regeneration tests deterministic integrity. Fresh cycles test same-code "
            "robustness on disjoint seeds. Neither is a separately authored scientific verifier."
        ),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": canonical_sha256(core)}


def validate_result(value: Mapping[str, Any]) -> None:
    core = {key: row for key, row in value.items() if key != "result_sha256"}
    if value.get("result_sha256") != canonical_sha256(core):
        raise ValueError("final campaign result seal drifted")
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("program_id") != PROGRAM_ID
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("decision", {}).get("independent_scientific_verification_complete") is not False
        or value.get("decision", {}).get("scientific_confirmation") is not False
    ):
        raise ValueError("final campaign result identity or safety drifted")


def _prerequisite_terminal_problem(path: Path, label: str) -> str | None:
    if not path.is_file():
        return None
    value = _read_object(path)
    state = value.get("state")
    if state in {"failure_hold", "failed", "stopped"}:
        return f"{label} prerequisite is terminal: {state}: {value.get('problems', [])}"
    return None


def run_campaign(
    *,
    root: Path = DEFAULT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST,
    result_path: Path = DEFAULT_RESULT,
    wait_seconds: int = WAIT_SECONDS,
) -> dict[str, Any]:
    set_process_label("mop-final-campaign")
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    result_path = Path(result_path).resolve()
    if any(not path.is_relative_to(REPO_ROOT.resolve()) for path in (root, manifest_path, result_path)):
        raise ValueError("final campaign paths must remain inside the repository")
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    validate_manifest(manifest)
    atomic_write_json(manifest_path, manifest)
    attempts: dict[str, int] = {}
    durations: dict[str, float] = {}

    while True:
        mode, workers, hawking_problem = hawking_mode()
        d1_authority = _load_d1_authority()
        mechanics_authority = _load_mechanics_authority()
        works = build_work_items(d1_authority, mechanics_authority)
        receipts = _load_receipts(root, works)

        problems = [
            problem
            for problem in (
                _prerequisite_terminal_problem(d1.DEFAULT_ROOT / "current_status.json", "D1"),
                _prerequisite_terminal_problem(
                    mechanics.DEFAULT_ROOT / "current_status.json", "successor mechanics"
                ),
            )
            if problem is not None
        ]
        if hawking_problem is not None:
            mode, workers = "hawking_unknown_fail_closed", 1
        if problems:
            status = _status(
                root=root,
                manifest=manifest,
                d1_authority=d1_authority,
                mechanics_authority=mechanics_authority,
                works=works,
                receipts=receipts,
                attempts=attempts,
                durations=durations,
                state="failure_hold",
                mode=mode,
                workers=workers,
                problems=problems,
            )
            atomic_write_json(root / "current_status.json", status)
            return status

        pending = [work for work in works if work.key not in receipts]
        preflight = [work for work in pending if work.preflight]
        eligible = preflight if preflight else pending
        if eligible:
            wave_workers = 1 if preflight else workers
            wave = eligible[:wave_workers]
            with ProcessPoolExecutor(max_workers=len(wave)) as pool:
                futures = {}
                for work in wave:
                    attempts[work.key] = attempts.get(work.key, 0) + 1
                    futures[pool.submit(_execute_work, work, str(root))] = work
                for future in as_completed(futures):
                    work = futures[future]
                    try:
                        key, duration = future.result()
                        value = _read_object(_artifact_path(root, work))
                        _validate_artifact(value, work)
                        receipts[key] = value
                        durations[key] = duration
                    except Exception as exc:  # noqa: BLE001 - sealed retry boundary
                        if attempts[work.key] >= RETRY_LIMIT:
                            status = _status(
                                root=root,
                                manifest=manifest,
                                d1_authority=d1_authority,
                                mechanics_authority=mechanics_authority,
                                works=works,
                                receipts=receipts,
                                attempts=attempts,
                                durations=durations,
                                state="failure_hold",
                                mode=mode,
                                workers=workers,
                                problems=[
                                    f"{work.key} failed after {attempts[work.key]} attempts: "
                                    f"{type(exc).__name__}: {exc}"
                                ],
                            )
                            atomic_write_json(root / "current_status.json", status)
                            return status
            state = "running"
        elif d1_authority is None or mechanics_authority is None:
            state = "waiting"
        else:
            result = aggregate(
                root=root,
                manifest=manifest,
                d1_authority=d1_authority,
                mechanics_authority=mechanics_authority,
                works=works,
                receipts=receipts,
            )
            validate_result(result)
            atomic_write_json(result_path, result)
            status = _status(
                root=root,
                manifest=manifest,
                d1_authority=d1_authority,
                mechanics_authority=mechanics_authority,
                works=works,
                receipts=receipts,
                attempts=attempts,
                durations=durations,
                state="complete",
                mode=mode,
                workers=workers,
                problems=[],
            )
            atomic_write_json(root / "current_status.json", status)
            return status

        status = _status(
            root=root,
            manifest=manifest,
            d1_authority=d1_authority,
            mechanics_authority=mechanics_authority,
            works=works,
            receipts=receipts,
            attempts=attempts,
            durations=durations,
            state=state,
            mode=mode,
            workers=workers,
            problems=[],
        )
        atomic_write_json(root / "current_status.json", status)
        if state == "waiting":
            time.sleep(max(1, wait_seconds))
