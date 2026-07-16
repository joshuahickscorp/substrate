from __future__ import annotations

from mop.studies import generation1_successor_mechanics_queue as queue


def test_inventory_covers_current_and_added_lanes() -> None:
    current = {lane.lane_id for lane in queue.LANES if lane.current_c3_lane}
    added = {lane.lane_id for lane in queue.LANES if not lane.current_c3_lane}
    assert current == {"G1-V1", "G1-M1", "G1-G1"}
    assert added == {
        "G1-E1",
        "G1-R1",
        "G1-P1",
        "G1-C0",
        "G1-D1",
        "G1-K1",
        "G1-A1",
        "G1-S1",
        "G1-I1",
    }
    assert len(queue.WORK_ITEMS) == 2_380


def test_every_mechanics_seed_partition_is_disjoint() -> None:
    intervals = sorted(
        (item.seed_start, item.seed_start + item.seed_count, item.capsule_id) for item in queue.WORK_ITEMS
    )
    for previous, current in zip(intervals, intervals[1:], strict=False):
        assert previous[1] <= current[0], (previous, current)
    assert intervals[0][0] > 21_365_888


def test_small_rung_is_deterministic_sealed_and_nonconfirmatory() -> None:
    item = queue.WorkItem(
        index=0,
        lane_id="G1-E1",
        mechanism="event_formation",
        phase="producer",
        rung_index=0,
        seed_start=101,
        seed_count=3,
    )
    first = queue._run_item_payload(item)
    second = queue._run_item_payload(item)
    queue.validate_rung(first, item)
    assert first == second
    assert first["receipt_count"] == 3
    assert first["confirmation_count"] == 0
    assert first["activation_allowed"] is False
    assert first["scientific_promotion"] is False


def test_execution_mode_is_wide_only_when_hawking_is_idle(monkeypatch) -> None:
    monkeypatch.setattr(queue, "hawking_mode", lambda: ("hawking_active", 1, None))
    assert queue.execution_mode() == ("hawking_active_stable_single_worker", 1, None)
    monkeypatch.setattr(queue, "hawking_mode", lambda: ("hawking_idle", 8, None))
    assert queue.execution_mode() == ("hawking_idle_stable_single_worker", 1, None)


def test_long_work_is_gated_by_clean_canary() -> None:
    lane = queue.LANES[0]
    canary = next(
        item for item in queue.WORK_ITEMS if item.lane_id == lane.lane_id and item.phase == "canary"
    )
    assert queue._canary_gate({}, lane.lane_id) is None
    assert (
        queue._canary_gate(
            {
                canary.index: {
                    "verdict_counts": {"mechanics-ok": queue.CANARY_SEEDS},
                    "confirmation_count": 0,
                }
            },
            lane.lane_id,
        )
        is True
    )
    failed = {
        canary.index: {
            "verdict_counts": {"mechanics-ok": queue.CANARY_SEEDS - 1, "null": 1},
            "confirmation_count": 0,
        }
    }
    assert queue._canary_gate(failed, lane.lane_id) is False
    assert all(
        item.phase == "canary" for item in queue._eligible_items(failed) if item.lane_id == lane.lane_id
    )


def test_eta_uses_planned_mechanism_costs_before_observations() -> None:
    next_rung, next_block, session = queue._eta_estimates({}, {})
    assert next_rung > 0
    assert next_block >= next_rung
    assert session >= 24 * 60 * 60


def test_aggregate_accepts_canary_pruning_without_promotion() -> None:
    receipts = {
        item.index: {
            "receipt_count": item.seed_count,
            "verdict_counts": {"null": item.seed_count},
            "control_clear_counts": {},
            "confirmation_count": 0,
        }
        for item in queue.WORK_ITEMS
        if item.phase == "canary"
    }
    result = queue.aggregate(receipts)
    queue.validate_aggregate(result)
    assert result["grid"]["executed_rung_count"] == len(queue.LANES)
    assert result["grid"]["pruned_rung_count"] == len(queue.WORK_ITEMS) - len(queue.LANES)
    assert set(result["decision"]["pruned_lanes"]) == {lane.lane_id for lane in queue.LANES}
    assert result["scientific_promotion"] is False
