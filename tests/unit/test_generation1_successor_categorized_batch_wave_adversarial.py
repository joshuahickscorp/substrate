from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studies import generation1_successor_categorized_batch_wave_verify as verifier
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256
from tests.unit.test_generation1_successor_categorized_batch_wave import (
    _materialize_gates,
)
from tests.unit.test_generation1_successor_categorized_batch_wave_verify import (
    _build_all_pruned_graph,
)


def _reseal(value: dict[str, Any], field: str) -> None:
    value.pop(field, None)
    value[field] = canonical_sha256(value)


def _tiny_work(lane_id: str, *, key: str, cycle: int) -> consolidated.WorkItem:
    source = next(item for item in mechanics.WORK_ITEMS if item.lane_id == lane_id)
    return consolidated.WorkItem(
        key=key,
        kind="mechanics_fresh",
        source_index=source.index,
        cycle=cycle,
        preflight=False,
    )


def _install_single_seed_fresh_items(monkeypatch: pytest.MonkeyPatch) -> None:
    original = consolidated.fresh_mechanics_item

    def reduced(work: consolidated.WorkItem) -> mechanics.WorkItem:
        fresh = original(work)
        return mechanics.WorkItem(
            index=fresh.index,
            lane_id=fresh.lane_id,
            mechanism=fresh.mechanism,
            phase=fresh.phase,
            rung_index=fresh.rung_index,
            seed_start=fresh.seed_start,
            seed_count=1,
        )

    monkeypatch.setattr(consolidated, "fresh_mechanics_item", reduced)


def _write_receipt(
    raw_root: Path,
    work: consolidated.WorkItem,
) -> dict[str, Any]:
    payload = cast(
        dict[str, Any],
        mechanics._run_item_payload(consolidated.fresh_mechanics_item(work)),
    )
    atomic_write_json(consolidated._artifact_path(raw_root, work), payload)
    return payload


def test_eligible_category_builder_and_validator_reject_incomplete_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=["G1-C0"],
    )
    work = _tiny_work("G1-C0", key="tiny_w01_c0", cycle=wave.EPOCH_CYCLES[0])
    monkeypatch.setattr(
        wave,
        "category_work_items",
        lambda _epoch, category: (work,) if category == "formation_trace" else (),
    )
    _install_single_seed_fresh_items(monkeypatch)
    predecessor, routing = wave._routing_for_epoch(root, 0)

    with pytest.raises(ValueError, match="eligible raw receipt inventory is incomplete"):
        wave._build_category_result(
            root=root,
            epoch_index=0,
            category_id="formation_trace",
            predecessor_binding=predecessor,
            route=routing["formation_trace"],
            receipts={},
            durations={},
            newly_executed=0,
        )

    raw_root = wave._category_raw_root(root, 0, "formation_trace")
    receipt = _write_receipt(raw_root, work)
    value = wave._build_category_result(
        root=root,
        epoch_index=0,
        category_id="formation_trace",
        predecessor_binding=predecessor,
        route=routing["formation_trace"],
        receipts={work.key: receipt},
        durations={work.key: 0.01},
        newly_executed=1,
    )
    wave.validate_category(
        value,
        epoch_index=0,
        category_id="formation_trace",
        root=root,
    )

    consolidated._artifact_path(raw_root, work).unlink()
    with pytest.raises(ValueError, match="eligible raw receipt inventory is incomplete"):
        wave.validate_category(
            value,
            epoch_index=0,
            category_id="formation_trace",
            root=root,
        )


def test_eligible_i1_builder_and_validator_reject_incomplete_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wave, "REPO_ROOT", tmp_path)
    root = tmp_path / "runs/categorized"
    work = _tiny_work("G1-I1", key="tiny_i1", cycle=wave.EPOCH_CYCLES[-1])
    monkeypatch.setattr(wave, "integration_work_items", lambda: (work,))
    _install_single_seed_fresh_items(monkeypatch)
    predecessor = {
        "path": "runs/categorized/classifications/w07.json",
        "file_sha256": "1" * 64,
        "classification_sha256": "2" * 64,
    }

    with pytest.raises(ValueError, match="eligible raw receipt inventory is incomplete"):
        wave._build_integration_result(
            root=root,
            predecessor_binding=predecessor,
            initially_eligible=True,
            eligible=True,
            receipts={},
            durations={},
            newly_executed=0,
        )

    raw_root = wave._integration_raw_root(root)
    receipt = _write_receipt(raw_root, work)
    value = wave._build_integration_result(
        root=root,
        predecessor_binding=predecessor,
        initially_eligible=True,
        eligible=True,
        receipts={work.key: receipt},
        durations={work.key: 0.01},
        newly_executed=1,
    )
    monkeypatch.setattr(
        wave,
        "_i1_eligible",
        lambda _root: (predecessor, True, True),
    )
    wave.validate_integration(value, root=root)

    consolidated._artifact_path(raw_root, work).unlink()
    with pytest.raises(ValueError, match="eligible raw receipt inventory is incomplete"):
        wave.validate_integration(value, root=root)


def test_gate_zero_rejects_empty_forged_and_duplicate_parent_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=["G1-C0"],
    )
    original = wave._read_object(wave._gate_path(root, "admit_v2"))

    empty = copy.deepcopy(original)
    empty["payload"]["parent_authority"] = {}
    _reseal(empty, "gate_sha256")

    forged = copy.deepcopy(original)
    forged["payload"]["parent_authority"]["result"]["file_sha256"] = "f" * 64
    _reseal(forged, "gate_sha256")

    duplicate = copy.deepcopy(original)
    duplicate["payload"]["mechanics_lanes"].append("G1-C0")
    _reseal(duplicate, "gate_sha256")

    for candidate in (empty, forged, duplicate):
        with pytest.raises(ValueError):
            wave.validate_gate(candidate, 0, root=root)


def test_d1_v2_screen_gate_rejects_extra_synthetic_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    candidate = wave._read_object(wave._gate_path(root, "screen_d1_redesign_v2"))
    candidate["payload"]["selected_candidate_id"] = "synthetic-winner"
    candidate["payload"]["ready_for_confirmatory_claim"] = True
    _reseal(candidate, "gate_sha256")

    with pytest.raises(ValueError, match="fields drifted"):
        wave.validate_gate(candidate, 3, root=root)


def test_category_and_i1_planning_shard_ids_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    category_work = _tiny_work(
        "G1-C0",
        key="tiny_w01_c0",
        cycle=wave.EPOCH_CYCLES[0],
    )
    monkeypatch.setattr(
        wave,
        "category_work_items",
        lambda _epoch, category: (category_work,) if category == "formation_trace" else (),
    )
    category = wave.run_category(
        root=root,
        epoch_index=0,
        category_id="formation_trace",
    )
    category["balanced_planning_shards"][0]["planning_shard_id"] = "forged-category-shard"
    _reseal(category, "category_sha256")
    with pytest.raises(ValueError, match="planning shard drifted"):
        wave.validate_category(
            category,
            epoch_index=0,
            category_id="formation_trace",
            root=root,
        )

    i1_work = _tiny_work("G1-I1", key="tiny_i1", cycle=wave.EPOCH_CYCLES[-1])
    monkeypatch.setattr(wave, "integration_work_items", lambda: (i1_work,))
    predecessor = {
        "path": "runs/categorized/classifications/w07.json",
        "file_sha256": "1" * 64,
        "classification_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        wave,
        "_i1_eligible",
        lambda _root: (predecessor, False, False),
    )
    integration = wave._build_integration_result(
        root=root,
        predecessor_binding=predecessor,
        initially_eligible=False,
        eligible=False,
        receipts={},
        durations={},
        newly_executed=0,
    )
    wave.validate_integration(integration, root=root)
    integration["balanced_planning_shards"][0]["planning_shard_id"] = "forged-i1-shard"
    _reseal(integration, "integration_sha256")
    with pytest.raises(ValueError, match="planning shard drifted"):
        wave.validate_integration(integration, root=root)


def test_communication_route_order_survives_sorted_json_into_w02(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_lanes = ["G1-V1", "G1-M1", "G1-K1"]
    root = _materialize_gates(
        tmp_path,
        monkeypatch,
        mechanics_lanes=expected_lanes,
    )
    _install_single_seed_fresh_items(monkeypatch)

    def tiny_category_work_items(
        epoch_index: int,
        category_id: str,
    ) -> tuple[consolidated.WorkItem, ...]:
        return tuple(
            _tiny_work(
                lane_id,
                key=f"tiny_{wave.EPOCH_IDS[epoch_index].lower()}_{lane_id.lower()}",
                cycle=wave.EPOCH_CYCLES[epoch_index],
            )
            for lane_id in wave.CATEGORY_LANES[category_id]
        )

    def execute(
        raw_root: Path,
        works: tuple[consolidated.WorkItem, ...],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, float], int]:
        receipts = {work.key: _write_receipt(raw_root, work) for work in works}
        return receipts, {work.key: 0.01 for work in works}, len(works)

    original_lane_verdicts = wave._lane_verdicts

    def force_clean_lane_verdicts(
        works: tuple[consolidated.WorkItem, ...],
        receipts: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        rows = cast(
            dict[str, dict[str, Any]],
            original_lane_verdicts(works, receipts),
        )
        for row in rows.values():
            if row["planned_item_count"] and row["executed_item_count"] == row["planned_item_count"]:
                row["verdict_counts"] = {"mechanics-ok": row["executed_item_count"]}
                row["continue_lane"] = True
        return rows

    monkeypatch.setattr(wave, "category_work_items", tiny_category_work_items)
    monkeypatch.setattr(wave, "_execute_pending", execute)
    monkeypatch.setattr(wave, "_lane_verdicts", force_clean_lane_verdicts)

    for category_id in wave.CATEGORY_IDS:
        wave.run_category(root=root, epoch_index=0, category_id=category_id)
    wave.classify_epoch(root=root, epoch_index=0)

    persisted_category = wave._read_object(wave._category_path(root, 0, "communication_repair"))
    assert list(persisted_category["lane_results"]) == [
        "G1-K1",
        "G1-M1",
        "G1-V1",
    ]
    persisted_classification = wave._read_object(wave._classification_path(root, 0))
    assert persisted_classification["routing"]["communication_repair"]["eligible_lane_ids"] == expected_lanes

    _predecessor, w02_routing = wave._routing_for_epoch(root, 1)
    assert w02_routing["communication_repair"]["eligible_lane_ids"] == expected_lanes
    assert w02_routing["communication_repair"]["pruned_lane_ids"] == []


def test_verifier_rejects_exact_recomputation_and_mutation_case_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, result_path, _result = _build_all_pruned_graph(tmp_path, monkeypatch)
    verification = verifier.build_verification(result_path)

    recomputation_mutations = {
        "pruned_category_route_count": 41,
        "raw_receipt_count": 1,
        "observed_seconds": 0.5,
    }
    for field, forged_value in recomputation_mutations.items():
        candidate = copy.deepcopy(verification)
        candidate["recomputation"][field] = forged_value
        _reseal(candidate, "verification_sha256")
        with pytest.raises(ValueError, match="recomputation drifted"):
            verifier.validate_verification(candidate)

    renamed_case = copy.deepcopy(verification)
    renamed_case["mutation_suite"]["cases"][0]["mutation"] = "renamed-program-identity"
    _reseal(renamed_case, "verification_sha256")
    with pytest.raises(ValueError, match="recomputation drifted"):
        verifier.validate_verification(renamed_case)


def test_w07_validation_visits_each_wave_and_category_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _result_path, _result = _build_all_pruned_graph(tmp_path, monkeypatch)
    w07 = wave._read_object(wave._classification_path(root, len(wave.EPOCH_IDS) - 1))
    original_classification = wave._validate_classification_with_context
    original_category = wave._validate_category_with_context
    classification_calls: list[int] = []
    category_calls: list[tuple[int, str]] = []

    def counted_classification(
        value: dict[str, Any],
        epoch_index: int,
        *,
        root: Path,
        context: wave._WaveValidationContext,
    ) -> dict[str, dict[str, Any]]:
        classification_calls.append(epoch_index)
        return cast(
            dict[str, dict[str, Any]],
            original_classification(
                value,
                epoch_index,
                root=root,
                context=context,
            ),
        )

    def counted_category(
        value: dict[str, Any],
        *,
        epoch_index: int,
        category_id: str,
        root: Path,
        predecessor_binding: dict[str, Any],
        expected_route: dict[str, Any],
    ) -> None:
        category_calls.append((epoch_index, category_id))
        original_category(
            value,
            epoch_index=epoch_index,
            category_id=category_id,
            root=root,
            predecessor_binding=predecessor_binding,
            expected_route=expected_route,
        )

    monkeypatch.setattr(
        wave,
        "_validate_classification_with_context",
        counted_classification,
    )
    monkeypatch.setattr(wave, "_validate_category_with_context", counted_category)

    wave.validate_classification(
        w07,
        len(wave.EPOCH_IDS) - 1,
        root=root,
    )

    assert classification_calls == list(range(len(wave.EPOCH_IDS)))
    assert len(category_calls) == len(wave.EPOCH_IDS) * len(wave.CATEGORY_IDS)
    assert Counter(epoch for epoch, _category in category_calls) == {
        epoch_index: len(wave.CATEGORY_IDS) for epoch_index in range(len(wave.EPOCH_IDS))
    }
