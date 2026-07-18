from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_d1_frozen_verify as frozen
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio import dynamic_worker_controller as dwc
from mop.studio.generation1_supervisor import atomic_write_json
from tests.unit.test_generation1_d1_frozen_verify import (
    _fake_rung_replay,
    aggregate_fixture,
)


class _InlineExecutor:
    """In-process ProcessPoolExecutor double for exercising the real _execute_pending path.

    Runs each submitted runner synchronously in the test process so a unit test can drive the real
    pool-sizing and result-collection logic without spawning subprocesses. Records every
    ``max_workers`` it is constructed with. The initializer is deliberately NOT run: its only effect
    is a best-effort ``os.nice`` on a worker, which is irrelevant to receipts and must never renice
    the test process.
    """

    created_with: list[int] = []

    def __init__(self, max_workers: int, initializer: Any = None, initargs: Any = ()) -> None:
        _InlineExecutor.created_with.append(int(max_workers))

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - surfaced through future.result()
            future.set_exception(exc)
        return future


def _host_sample(state: dwc.HostState) -> dwc.HostSample:
    """Wrap a HostState in a HostSample using the real pure derivations (no live host reads)."""

    return dwc.HostSample(
        state=state,
        priority=dwc.recommended_priority(state),
        recommended_workers=dwc.recommended_workers(state),
        bounds=dwc.worker_bounds(state),
        hawking_processes=[],
        telemetry={},
        created_at=0.0,
    )


def _idle_state(current_workers: int) -> dwc.HostState:
    return dwc.HostState(
        free_p_cores=28,
        hawking_active=False,
        mem_available_gb=90.0,
        thermal_ok=True,
        on_ac=True,
        current_load=1.0,
        current_workers=current_workers,
    )


def _hawking_state(current_workers: int) -> dwc.HostState:
    return dwc.HostState(
        free_p_cores=28,
        hawking_active=True,
        mem_available_gb=90.0,
        thermal_ok=True,
        on_ac=True,
        current_load=8.0,
        current_workers=current_workers,
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _materialize_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mechanics_lanes: list[str],
) -> Path:
    monkeypatch.setattr(wave, "REPO_ROOT", tmp_path)
    parent = {
        "bindings": {
            "program_manifest": {
                "path": "configs/parent.json",
                "file_sha256": "1" * 64,
                "program_sha256": "2" * 64,
            },
            "supervisor_status": {
                "path": "runs/parent/status.json",
                "file_sha256": "3" * 64,
                "status_sha256": "4" * 64,
            },
            "result": {
                "path": "proof/parent.json",
                "file_sha256": "5" * 64,
                "result_sha256": "6" * 64,
            },
            "verification": {
                "path": "proof/parent.verify.json",
                "file_sha256": "7" * 64,
                "verification_sha256": "8" * 64,
            },
            "report_receipt": {
                "path": "runs/parent/report.json",
                "file_sha256": "9" * 64,
                "receipt_sha256": "a" * 64,
            },
        },
        "mechanics_lanes": mechanics_lanes,
    }
    monkeypatch.setattr(wave, "_validated_parent", lambda **_kwargs: parent)
    monkeypatch.setattr(frozen, "_replay_raw_rungs", _fake_rung_replay)
    d1_path = tmp_path / "proof/d1.json"
    _write(d1_path, aggregate_fixture())
    root = tmp_path / "runs/categorized"
    for gate_index in range(len(wave.GATE_IDS)):
        wave.materialize_gate(
            root=root,
            gate_index=gate_index,
            d1_result_path=d1_path,
        )
    return root


def test_taxonomy_cycles_and_real_mechanics_envelope_are_exact() -> None:
    assert wave.PROGRAM_ID == "generation1-successor-categorized-batch-wave-v1"
    assert wave.GATE_IDS == (
        "admit_v2",
        "verify_old_d1",
        "classify_retire_old_d1",
        "screen_d1_redesign_v2",
        "freeze_d1_redesign_v2",
    )
    assert wave.EPOCH_IDS == ("W01", "W02", "W03", "W04", "W05", "W06", "W07")
    assert wave.EPOCH_CYCLES == (12, 13, 14, 15, 16, 17, 18)
    assert wave.CATEGORY_LANES == {
        "formation_trace": ("G1-C0", "G1-E1"),
        "communication_repair": ("G1-V1", "G1-M1", "G1-K1"),
        "memory_plasticity": ("G1-R1", "G1-P1"),
        "action_simulation": ("G1-A1", "G1-S1"),
        "construction": ("G1-G1",),
        "dispatch_redesign": ("G1-D1",),
    }
    assert wave.CAPSULE_COUNT == 59
    assert wave.INTERNAL_SHARD_COUNT == 8
    assert wave.IDLE_WORKERS == 20
    assert wave.MAXIMUM_RAW_RECEIPT_COUNT == 15_886
    assert wave.planned_serial_hours() == pytest.approx(196.1824722467)
    assert wave.planned_ideal_eight_worker_hours() == pytest.approx(9.8091236123)
    assert wave.planned_ideal_eight_worker_hours() > 9


def test_work_items_use_fresh_cycles_exact_taxonomy_and_disjoint_seed_space() -> None:
    all_non_i1 = []
    for epoch_index, cycle in enumerate(wave.EPOCH_CYCLES):
        cycle_sources = []
        for category_id in wave.CATEGORY_IDS:
            works = wave.category_work_items(epoch_index, category_id)
            assert {mechanics.WORK_ITEMS[work.source_index].lane_id for work in works} == set(
                wave.CATEGORY_LANES[category_id]
            )
            assert all(work.kind == "mechanics_fresh" and work.cycle == cycle for work in works)
            cycle_sources.extend(work.source_index for work in works)
        assert len(cycle_sources) == len(set(cycle_sources)) == 2_251
        all_non_i1.extend(cycle_sources)

    i1 = wave.integration_work_items()
    assert len(i1) == 129
    assert {mechanics.WORK_ITEMS[work.source_index].lane_id for work in i1} == {"G1-I1"}
    assert all(work.cycle == 18 for work in i1)

    first = consolidated.fresh_mechanics_item(wave.category_work_items(0, "formation_trace")[0])
    last = consolidated.fresh_mechanics_item(wave.category_work_items(6, "formation_trace")[0])
    assert first.seed_start < last.seed_start
    assert last.seed_start - first.seed_start == 6 * consolidated.MECHANICS_CYCLE_STRIDE


def test_frozen_d1_gates_use_real_verifier_and_never_synthesize_redesign_efficacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lanes = [lane.lane_id for lane in mechanics.LANES if lane.lane_id != "G1-I1"]
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=lanes)

    verified = wave._read_object(root / "gates/verify_old_d1.json")
    frozen.validate_byte_bound_verification(verified["payload"]["byte_bound_verification"])
    assert (
        verified["payload"]["byte_bound_verification"]["classification"]["evidence_classification"]
        == frozen.NULL_SAFE_PRUNE
    )
    retired = wave._read_object(root / "gates/classify_retire_old_d1.json")
    assert retired["payload"]["classification"]["evidence_classification"] == frozen.NULL_SAFE_PRUNE
    assert retired["payload"]["old_d1_retired"] is True
    screened = wave._read_object(root / "gates/screen_d1_redesign_v2.json")
    assert screened["payload"]["candidate_evidence_count"] == 0
    assert screened["payload"]["screen_state"] == "no_candidate_preregistered"
    frozen_redesign = wave._read_object(root / "gates/freeze_d1_redesign_v2.json")
    assert frozen_redesign["payload"]["freeze_state"] == "no_candidate_preregistered"
    assert all(
        row["status"] == "no_candidate" and row["execution_authorized"] is False
        for row in frozen_redesign["payload"]["categorized_program_admission"]
    )
    assert frozen_redesign["payload"]["redesign_efficacy_execution_authorized"] is False


def test_run_category_uses_existing_executor_and_balanced_planning_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=["G1-C0"],
    )
    source = next(item for item in mechanics.WORK_ITEMS if item.lane_id == "G1-C0")
    work = consolidated.WorkItem(
        key="w01_test_c0",
        kind="mechanics_fresh",
        source_index=source.index,
        cycle=12,
        preflight=False,
    )
    monkeypatch.setattr(
        wave,
        "category_work_items",
        lambda _epoch, category: (work,) if category == "formation_trace" else (),
    )

    def execute(raw_root: Path, works: tuple[consolidated.WorkItem, ...]):
        assert works == (work,)
        fresh = consolidated.fresh_mechanics_item(work)
        reduced = mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=2,
        )
        payload = mechanics._run_item_payload(reduced)
        original = consolidated.fresh_mechanics_item
        monkeypatch.setattr(consolidated, "fresh_mechanics_item", lambda _work: reduced)
        atomic_write_json(consolidated._artifact_path(raw_root, work), payload)
        receipts = {work.key: payload}
        assert wave._load_receipts(raw_root, works) == receipts
        monkeypatch.setattr(consolidated, "fresh_mechanics_item", original)
        return receipts, {work.key: 0.25}, 1

    monkeypatch.setattr(wave, "_execute_pending", execute)
    original_fresh = consolidated.fresh_mechanics_item
    fresh = original_fresh(work)
    reduced = mechanics.WorkItem(
        index=fresh.index,
        lane_id=fresh.lane_id,
        mechanism=fresh.mechanism,
        phase=fresh.phase,
        rung_index=fresh.rung_index,
        seed_start=fresh.seed_start,
        seed_count=2,
    )
    monkeypatch.setattr(consolidated, "fresh_mechanics_item", lambda _work: reduced)
    value = wave.run_category(root=root, epoch_index=0, category_id="formation_trace")

    assert value["execution"]["executed_item_count"] == 1
    assert value["execution"]["compute_started"] is True
    assert value["execution"]["observed_seconds"] == 0.25
    assert value["execution"]["worker_pool"] == "dynamic_process_pool"
    assert value["execution"]["worker_pool_max_workers"] == 20
    assert len(value["balanced_planning_shards"]) == 8
    assert sum(row["work_item_count"] for row in value["balanced_planning_shards"]) == 1
    assert all(row["partition_pinned"] is False for row in value["balanced_planning_shards"])
    assert value["lane_results"]["G1-C0"]["continue_lane"] is True


def test_pruned_category_executes_nothing_and_cannot_resurrect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    called = False

    def forbidden(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("pruned category executed")

    monkeypatch.setattr(wave, "_execute_pending", forbidden)
    value = wave.run_category(root=root, epoch_index=0, category_id="dispatch_redesign")
    assert called is False
    assert value["execution"]["executed_item_count"] == 0
    assert value["execution"]["skipped_item_count"] == 129
    assert value["execution"]["compute_started"] is False
    assert value["redesign_v2_efficacy"]["execution_authorized"] is False


def test_partial_sibling_pruning_is_filtered_and_never_resurrected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=["G1-C0", "G1-E1"],
    )
    original_category_work_items = wave.category_work_items
    original_fresh = consolidated.fresh_mechanics_item
    source_by_lane = {
        lane_id: next(item for item in mechanics.WORK_ITEMS if item.lane_id == lane_id)
        for lane_id in ("G1-C0", "G1-E1")
    }

    def tiny_category_work_items(
        epoch_index: int,
        category_id: str,
    ) -> tuple[consolidated.WorkItem, ...]:
        if category_id != "formation_trace":
            return original_category_work_items(epoch_index, category_id)
        return tuple(
            consolidated.WorkItem(
                key=f"partial_w{epoch_index + 1:02d}_{lane_id.lower()}",
                kind="mechanics_fresh",
                source_index=source_by_lane[lane_id].index,
                cycle=wave.EPOCH_CYCLES[epoch_index],
                preflight=False,
            )
            for lane_id in ("G1-C0", "G1-E1")
        )

    def reduced_fresh(work: consolidated.WorkItem) -> mechanics.WorkItem:
        fresh = original_fresh(work)
        if not work.key.startswith("partial_"):
            return fresh
        return mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=1,
        )

    executed_lane_batches: list[tuple[str, ...]] = []

    def execute(
        raw_root: Path,
        works: tuple[consolidated.WorkItem, ...],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float], int]:
        executed_lane_batches.append(tuple(wave._work_lane_id(work) for work in works))
        receipts: dict[str, dict[str, Any]] = {}
        durations: dict[str, float] = {}
        for work in works:
            payload = mechanics._run_item_payload(reduced_fresh(work))
            atomic_write_json(consolidated._artifact_path(raw_root, work), payload)
            receipts[work.key] = payload
            durations[work.key] = 0.01
        return receipts, durations, len(works)

    original_lane_verdicts = wave._lane_verdicts

    def lane_verdicts(
        works: tuple[consolidated.WorkItem, ...],
        receipts: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        rows = original_lane_verdicts(works, receipts)
        if any(work.key.startswith("partial_w01_") for work in works):
            rows["G1-C0"] = {
                **rows["G1-C0"],
                "verdict_counts": {"mechanics-warning": 1},
                "continue_lane": False,
            }
        return rows

    monkeypatch.setattr(wave, "category_work_items", tiny_category_work_items)
    monkeypatch.setattr(consolidated, "fresh_mechanics_item", reduced_fresh)
    monkeypatch.setattr(wave, "_execute_pending", execute)
    monkeypatch.setattr(wave, "_lane_verdicts", lane_verdicts)

    wave.run_category(root=root, epoch_index=0, category_id="formation_trace")
    for category_id in wave.CATEGORY_IDS[1:]:
        wave.run_category(root=root, epoch_index=0, category_id=category_id)
    first = wave.classify_epoch(root=root, epoch_index=0)
    assert first["routing"]["formation_trace"]["eligible_lane_ids"] == ["G1-E1"]
    assert first["routing"]["formation_trace"]["pruned_lane_ids"] == ["G1-C0"]

    second_category = wave.run_category(
        root=root,
        epoch_index=1,
        category_id="formation_trace",
    )
    for category_id in wave.CATEGORY_IDS[1:]:
        wave.run_category(root=root, epoch_index=1, category_id=category_id)
    second = wave.classify_epoch(root=root, epoch_index=1)

    assert executed_lane_batches == [("G1-C0", "G1-E1"), ("G1-E1",)]
    assert second_category["execution"]["status"] == "partial"
    assert second_category["execution"]["eligible_item_count"] == 1
    assert second_category["execution"]["pruned_item_count"] == 1
    assert second_category["execution"]["skipped_item_count"] == 1
    assert second_category["lane_results"]["G1-C0"]["executed_item_count"] == 0
    assert second_category["lane_results"]["G1-C0"]["continue_lane"] is False
    assert second_category["lane_results"]["G1-E1"]["continue_lane"] is True
    assert {row["lane_id"] for row in second_category["artifact_index"]} == {"G1-E1"}
    second_c0 = tiny_category_work_items(1, "formation_trace")[0]
    assert not consolidated._artifact_path(
        wave._category_raw_root(root, 1, "formation_trace"),
        second_c0,
    ).exists()
    assert second["routing"]["formation_trace"]["eligible_lane_ids"] == ["G1-E1"]
    assert second["routing"]["formation_trace"]["pruned_lane_ids"] == ["G1-C0"]
    expected_skipped = sum(
        wave._read_object(wave._category_path(root, 1, category_id))["execution"]["skipped_item_count"]
        for category_id in wave.CATEGORY_IDS
    )
    assert second["summary"]["skipped_item_count"] == expected_skipped


def test_i1_cannot_resurrect_when_clean_v2_pruned_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency_lanes = list(wave._I1_SPEC.dependencies)
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=dependency_lanes,
    )
    classification_path = wave._classification_path(root, len(wave.EPOCH_IDS) - 1)
    classification = {
        "routing": {
            category_id: {
                "eligible_lane_ids": [
                    lane_id for lane_id in wave.CATEGORY_LANES[category_id] if lane_id in dependency_lanes
                ]
            }
            for category_id in wave.CATEGORY_IDS
        },
        "classification_sha256": "f" * 64,
    }
    _write(classification_path, classification)
    monkeypatch.setattr(wave, "validate_classification", lambda *_args, **_kwargs: None)

    called = False

    def forbidden(*_args: Any, **_kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("pruned I1 executed")

    monkeypatch.setattr(wave, "_execute_pending", forbidden)
    value = wave.run_integration(root=root)

    assert called is False
    assert value["i1_initially_eligible"] is False
    assert value["execution"]["eligible"] is False
    assert value["execution"]["executed_item_count"] == 0
    assert value["execution"]["skipped_item_count"] == len(wave.integration_work_items())


def test_worker_count_is_receipt_invariant_across_pool_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seeded-sha256 raw receipt is byte-identical whether the pool floats to 1 or 20 workers.

    Drives the real _execute_pending through an in-process executor double under two forced widths
    (1 vs 20) and asserts every raw capsule receipt is byte-for-byte identical, proving the dynamic
    worker count never leaks into any hashed or sealed field. worker_pool_max_workers stays the fixed
    IDLE_WORKERS ceiling regardless of the instantaneous width.
    """

    sources = {
        lane_id: next(item for item in mechanics.WORK_ITEMS if item.lane_id == lane_id)
        for lane_id in ("G1-C0", "G1-E1")
    }
    works = tuple(
        consolidated.WorkItem(
            key=f"w01_inv_{lane_id.lower()}",
            kind="mechanics_fresh",
            source_index=sources[lane_id].index,
            cycle=12,
            preflight=False,
        )
        for lane_id in ("G1-C0", "G1-E1")
    )
    original_fresh = consolidated.fresh_mechanics_item

    def reduced_fresh(work: consolidated.WorkItem) -> mechanics.WorkItem:
        fresh = original_fresh(work)
        return mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=2,
        )

    # Run the real seeded runners in-process (no subprocess) with the process-label and nice side
    # effects neutralized, so the test never renames itself into the live mop-final-* worker family.
    monkeypatch.setattr(consolidated, "fresh_mechanics_item", reduced_fresh)
    monkeypatch.setattr(consolidated, "set_process_label", lambda *_a, **_k: None)
    monkeypatch.setattr(consolidated.os, "nice", lambda *_a: 0)
    monkeypatch.setattr(wave, "ProcessPoolExecutor", _InlineExecutor)
    _InlineExecutor.created_with = []

    def run_at_width(width: int, raw_root: Path) -> dict[str, dict[str, Any]]:
        monkeypatch.setattr(wave, "_dynamic_pool_width", lambda _pending: (width, None))
        receipts, _durations, newly = wave._execute_pending(raw_root, works)
        assert newly == len(works)
        return receipts

    receipts_one = run_at_width(1, tmp_path / "width_one")
    receipts_twenty = run_at_width(20, tmp_path / "width_twenty")

    assert set(receipts_one) == set(receipts_twenty) == {work.key for work in works}
    # The pool genuinely floated to two different widths (1 then 20) across the two builds.
    assert _InlineExecutor.created_with == [1, 20]
    for work in works:
        assert receipts_one[work.key]["result_sha256"] == receipts_twenty[work.key]["result_sha256"]
        bytes_one = consolidated._artifact_path(tmp_path / "width_one", work).read_bytes()
        bytes_twenty = consolidated._artifact_path(tmp_path / "width_twenty", work).read_bytes()
        assert bytes_one == bytes_twenty


def test_dynamic_pool_width_backs_off_under_hawking_and_ramps_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool floats with live host load: it holds the twenty-worker ceiling when the host is idle,
    sheds to the tiny reserve under a faked Hawking workload, and falls back to the static
    min(IDLE_WORKERS, pending) width when no live host sample is available."""

    assert wave.IDLE_WORKERS == dwc.WORKER_CEILING == 20

    # Idle host: hold the declared twenty-worker ceiling at the mild utility nice level.
    monkeypatch.setattr(dwc, "sample_host_state", lambda *_a, **_k: _host_sample(_idle_state(20)))
    width, nice_level = wave._dynamic_pool_width(30)
    assert width == 20
    assert nice_level == dwc.IDLE_NICE

    # Hawking active: back the worker count off to the tiny reserve and yield cores by maximum nice.
    monkeypatch.setattr(dwc, "sample_host_state", lambda *_a, **_k: _host_sample(_hawking_state(20)))
    width, nice_level = wave._dynamic_pool_width(30)
    assert width == dwc.HAWKING_RESERVE_WORKERS == 2
    assert width < 20
    assert nice_level == dwc.HAWKING_NICE

    # Controller cannot sample (psutil absent or a refusal): fall back to the static width exactly as
    # before, with no priority initializer (nice_level is None so _execute_pending passes no
    # initializer to the pool).
    def _refuse(*_a: Any, **_k: Any) -> dwc.HostSample:
        raise dwc.WorkerControllerRefused("psutil is required to read a live host state")

    monkeypatch.setattr(dwc, "sample_host_state", _refuse)
    assert wave._dynamic_pool_width(5) == (5, None)
    assert wave._dynamic_pool_width(30) == (20, None)

    # A single pending item never even samples the host: it takes exactly one worker.
    assert wave._dynamic_pool_width(1) == (1, None)
