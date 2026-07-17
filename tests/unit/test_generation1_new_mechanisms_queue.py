"""Tests for the new-mechanisms queue (G1-U1, G1-N1, G1-P1R).

Covers the item table shape, seed-band disjointness against the sealed successor mechanics queue,
payload seal round trips on tiny synthetic items, canary-scale sanity for every lane, and the
fail-closed refusal paths. Everything stays a mechanics demonstration; nothing here promotes.
"""

from __future__ import annotations

import pytest

from mop.studies import generation1_new_mechanisms_queue as newq
from mop.studies import generation1_successor_mechanics_queue as mechanics


def test_item_table_shape_and_local_indexes() -> None:
    assert {lane.lane_id for lane in newq.NEW_LANES} == {"G1-U1", "G1-N1", "G1-P1R"}
    assert all(lane.current_c3_lane is False for lane in newq.NEW_LANES)
    expected = len(newq.NEW_LANES) + sum(2 * lane.rungs_per_phase for lane in newq.NEW_LANES)
    assert len(newq.NEW_WORK_ITEMS) == expected == 387
    assert [item.index for item in newq.NEW_WORK_ITEMS] == list(range(len(newq.NEW_WORK_ITEMS)))
    canaries = newq.NEW_WORK_ITEMS[: len(newq.NEW_LANES)]
    assert all(item.phase == "canary" and item.seed_count == newq.CANARY_SEEDS for item in canaries)
    assert newq.CANARY_SEEDS == 256


def test_planned_rates_cover_every_lane_mechanism() -> None:
    assert set(newq.NEW_PLANNED_SECONDS_PER_SEED) == {lane.mechanism for lane in newq.NEW_LANES}
    assert all(rate > 0 for rate in newq.NEW_PLANNED_SECONDS_PER_SEED.values())


def test_new_seed_bands_are_disjoint_from_mechanics_lanes() -> None:
    intervals = sorted(
        [(item.seed_start, item.seed_start + item.seed_count) for item in mechanics.WORK_ITEMS]
        + [(item.seed_start, item.seed_start + item.seed_count) for item in newq.NEW_WORK_ITEMS]
    )
    for previous, current in zip(intervals, intervals[1:], strict=False):
        assert previous[1] <= current[0], (previous, current)
    mechanics_max_end = max(item.seed_start + item.seed_count for item in mechanics.WORK_ITEMS)
    new_min_start = min(item.seed_start for item in newq.NEW_WORK_ITEMS)
    assert new_min_start >= mechanics_max_end


def test_small_rung_is_deterministic_sealed_and_nonconfirmatory() -> None:
    item = newq.WorkItem(
        index=0,
        lane_id="G1-U1",
        mechanism="calibrated_uncertainty",
        phase="producer",
        rung_index=0,
        seed_start=172_000_001,
        seed_count=3,
    )
    first = newq._run_item_payload(item)
    second = newq._run_item_payload(item)
    newq.validate_rung(first, item)
    assert first == second
    assert first["receipt_count"] == 3
    assert first["confirmation_count"] == 0
    assert first["activation_allowed"] is False
    assert first["scientific_promotion"] is False


@pytest.mark.parametrize("lane", newq.NEW_LANES, ids=lambda lane: lane.lane_id)
def test_canary_scale_sanity_item_is_all_mechanics_ok(lane: newq.LaneSpec) -> None:
    item = newq.WorkItem(
        index=0,
        lane_id=lane.lane_id,
        mechanism=lane.mechanism,
        phase="canary",
        rung_index=0,
        seed_start=lane.canary_start,
        seed_count=8,
    )
    payload = newq._run_item_payload(item)
    newq.validate_rung(payload, item)
    assert payload["verdict_counts"] == {"mechanics-ok": 8}
    assert payload["confirmation_count"] == 0


def test_validate_rung_refuses_tampered_seal_item_or_confirmation() -> None:
    item = newq.WorkItem(
        index=1,
        lane_id="G1-N1",
        mechanism="reducible_novelty",
        phase="challenge",
        rung_index=0,
        seed_start=188_000_001,
        seed_count=2,
    )
    payload = newq._run_item_payload(item)
    newq.validate_rung(payload, item)

    tampered_seal = dict(payload)
    tampered_seal["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seal"):
        newq.validate_rung(tampered_seal, item)

    other_item = newq.WorkItem(
        index=2,
        lane_id="G1-N1",
        mechanism="reducible_novelty",
        phase="challenge",
        rung_index=1,
        seed_start=188_000_003,
        seed_count=2,
    )
    with pytest.raises(ValueError, match="identity or safety"):
        newq.validate_rung(payload, other_item)

    from mop.studies.generation1_c3_dispatch import canonical_sha256

    confirmed = {key: value for key, value in payload.items() if key != "result_sha256"}
    confirmed["confirmation_count"] = 1
    confirmed = {**confirmed, "result_sha256": canonical_sha256(confirmed)}
    with pytest.raises(ValueError, match="identity or safety"):
        newq.validate_rung(confirmed, item)


def test_run_item_round_trip_writes_a_valid_receipt(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny = newq.WorkItem(
        index=0,
        lane_id="G1-P1R",
        mechanism="stability_plasticity_r2",
        phase="canary",
        rung_index=0,
        seed_start=193_000_001,
        seed_count=4,
    )
    monkeypatch.setattr(newq, "NEW_WORK_ITEMS", (tiny,))
    monkeypatch.setattr(newq, "REPO_ROOT", tmp_path)
    root = tmp_path / "runs"
    result = newq.run_item(0, root)
    assert result["wall_seconds"] > 0
    path = newq._item_path(root.resolve(), tiny)
    assert path.is_file()
    summary = newq.status(root)
    assert summary["counts"] == {"complete": 1, "total": 1, "remaining": 0}
    assert summary["lane_progress"]["G1-P1R"] == {"complete": 1, "total": 1}
    assert summary["activation_allowed"] is False
    assert summary["scientific_promotion"] is False


def test_unknown_index_and_escaping_root_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown"):
        newq.run_item(len(newq.NEW_WORK_ITEMS))
    with pytest.raises(ValueError, match="unknown"):
        newq.run_item(-1)
    with pytest.raises(ValueError, match="repository"):
        newq.run_item(0, newq.Path("/"))
