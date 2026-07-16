from __future__ import annotations

from mop.studies import generation1_consolidated_final_campaign as final
from mop.studies import generation1_successor_mechanics_queue as mechanics


def _d1_authority(repeated: bool) -> dict:
    return {"decision": {"frozen_pattern_repeated": repeated}}


def _mechanics_authority(*pruned: str) -> dict:
    return {"decision": {"pruned_lanes": list(pruned)}}


def test_manifest_is_sealed_nonpromotable_and_large() -> None:
    manifest = final.build_manifest()
    final.validate_manifest(manifest)
    assert manifest["activation_allowed"] is False
    assert manifest["scientific_promotion"] is False
    assert manifest["execution"]["waiting_time_limit"] is None
    assert final.planned_max_compute_seconds() >= 120 * 60 * 60


def test_work_graph_is_conditionally_expanded() -> None:
    null_d1 = final.build_work_items(_d1_authority(False), None)
    positive_d1 = final.build_work_items(_d1_authority(True), None)
    assert len(null_d1) == final.d1.DEFAULT_RUNG_COUNT
    assert len(positive_d1) == final.d1.DEFAULT_RUNG_COUNT * (1 + final.FRESH_D1_CYCLES)

    mechanics_work = final.build_work_items(None, _mechanics_authority("G1-P1"))
    source = [work for work in mechanics_work if work.kind == "mechanics_source"]
    fresh = [work for work in mechanics_work if work.kind == "mechanics_fresh"]
    assert len(source) == len(mechanics.WORK_ITEMS) - 128
    assert len(fresh) == (len(mechanics.WORK_ITEMS) - 129) * final.FRESH_MECHANICS_CYCLES
    assert sum(work.preflight for work in source) == len({lane.mechanism for lane in mechanics.LANES})


def test_fresh_seed_spaces_are_disjoint_from_sources_and_each_other() -> None:
    works = final.build_work_items(_d1_authority(True), _mechanics_authority("G1-P1"))
    d1_intervals = []
    for work in works:
        if work.kind != "d1_fresh":
            continue
        config = final.fresh_d1_config(work)
        d1_intervals.extend(
            (
                config["train_seed_start"],
                config["train_seed_start"] + config["train_seed_count"],
            )
            for _ in (0,)
        )
        d1_intervals.extend(
            (
                config["heldout_seed_start"],
                config["heldout_seed_start"] + config["heldout_seed_count"],
            )
            for _ in (0,)
        )
    unique_d1 = sorted(set(d1_intervals))
    for previous, current in zip(unique_d1, unique_d1[1:], strict=False):
        assert previous[1] <= current[0]

    fresh_mechanics = [final.fresh_mechanics_item(work) for work in works if work.kind == "mechanics_fresh"]
    intervals = sorted((item.seed_start, item.seed_start + item.seed_count) for item in fresh_mechanics)
    for previous, current in zip(intervals, intervals[1:], strict=False):
        assert previous[1] <= current[0]
    assert unique_d1[-1][1] < intervals[0][0]


def test_audit_receipt_seal_and_validation() -> None:
    source = final.REPO_ROOT / "pyproject.toml"
    work = final.WorkItem("test", "d1_source", 0, None, True)
    value = final._audit_receipt(work, source, {"result_sha256": "a" * 64})
    final._validate_artifact(value, work)
    assert value["canonical_regeneration_match"] is True
    assert value["scientific_promotion"] is False


def test_fresh_mechanics_payload_is_deterministic_and_nonconfirmatory() -> None:
    source = next(item for item in mechanics.WORK_ITEMS if item.lane_id == "G1-E1" and item.phase == "canary")
    work = final.WorkItem("fresh", "mechanics_fresh", source.index, 0, False)
    item = final.fresh_mechanics_item(work)
    first = mechanics._run_item_payload(item)
    second = mechanics._run_item_payload(item)
    assert first == second
    assert first["confirmation_count"] == 0
    assert first["scientific_promotion"] is False
