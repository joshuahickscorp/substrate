from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studies import generation1_successor_categorized_batch_wave_verify as verifier
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256
from tests.unit.test_generation1_successor_categorized_batch_wave import (
    _materialize_gates,
)


def _build_all_pruned_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    source_by_lane = {
        lane_id: next(item for item in mechanics.WORK_ITEMS if item.lane_id == lane_id)
        for lane_id in {
            *(lanes[0] for lanes in wave.CATEGORY_LANES.values()),
            wave.I1_LANE_ID,
        }
    }

    def tiny_category_work_items(
        epoch_index: int,
        category_id: str,
    ) -> tuple[consolidated.WorkItem, ...]:
        lane_id = wave.CATEGORY_LANES[category_id][0]
        source = source_by_lane[lane_id]
        return (
            consolidated.WorkItem(
                key=f"tiny_{wave.EPOCH_IDS[epoch_index].lower()}_{category_id}",
                kind="mechanics_fresh",
                source_index=source.index,
                cycle=wave.EPOCH_CYCLES[epoch_index],
                preflight=False,
            ),
        )

    i1_source = source_by_lane[wave.I1_LANE_ID]
    monkeypatch.setattr(wave, "category_work_items", tiny_category_work_items)
    monkeypatch.setattr(
        wave,
        "integration_work_items",
        lambda: (
            consolidated.WorkItem(
                key="tiny_i1",
                kind="mechanics_fresh",
                source_index=i1_source.index,
                cycle=wave.EPOCH_CYCLES[-1],
                preflight=False,
            ),
        ),
    )
    original_validate_gate = wave.validate_gate
    validated_gates: set[tuple[str, int, str, str]] = set()

    def cached_validate_gate(
        value: dict[str, object],
        gate_index: int,
        *,
        root: Path = wave.DEFAULT_ROOT,
        replay_d1_raw: bool = False,
        d1_rung_root: Path | None = None,
    ) -> None:
        key = (
            str(value.get("gate_sha256")),
            gate_index,
            str(Path(root).resolve()),
            str(replay_d1_raw),
        )
        if key not in validated_gates:
            original_validate_gate(
                value,
                gate_index,
                root=root,
                replay_d1_raw=replay_d1_raw,
                d1_rung_root=d1_rung_root,
            )
            validated_gates.add(key)

    monkeypatch.setattr(wave, "validate_gate", cached_validate_gate)
    for epoch_index in range(len(wave.EPOCH_IDS)):
        for category_id in wave.CATEGORY_IDS:
            category = wave.run_category(
                root=root,
                epoch_index=epoch_index,
                category_id=category_id,
            )
            assert category["execution"]["eligible"] is False
            assert category["execution"]["executed_item_count"] == 0
        wave.classify_epoch(root=root, epoch_index=epoch_index)
    integration = wave.run_integration(root=root)
    assert integration["i1_initially_eligible"] is False
    assert integration["execution"]["eligible"] is False
    wave.classify_integration(root=root)
    result_path = tmp_path / "proof/categorized.json"
    result = wave.aggregate(root=root, output=result_path)
    return root, result_path, result


def test_full_all_pruned_graph_aggregates_and_independently_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, result_path, result = _build_all_pruned_graph(tmp_path, monkeypatch)

    assert result["grid"]["manifest_capsule_count"] == 59
    assert result["grid"]["balanced_planning_shards_per_compute_capsule"] == wave.INTERNAL_SHARD_COUNT
    assert result["execution"]["executed_item_count"] == 0
    assert result["execution"]["skipped_item_count"] == (len(wave.EPOCH_IDS) * len(wave.CATEGORY_IDS) + 1)
    assert result["execution"]["compute_started"] is False
    assert result["decision"]["d1_redesign_efficacy_executed"] is False

    verification = verifier.build_verification(result_path)
    verifier.validate_verification(verification)
    assert verification["checks"]["result_seal_valid"] is True
    assert verification["checks"]["fresh_mechanics_receipts_valid"] is True
    assert verification["checks"]["null_safe_pruning_valid"] is True
    assert verification["checks"]["dynamic_eight_worker_pool_valid"] is True
    assert verification["checks"]["balanced_planning_shards_valid"] is True
    assert verification["checks"]["d1_redesign_efficacy_not_executed"] is True
    assert verification["recomputation"]["gate_count"] == 5
    assert verification["recomputation"]["wave_count"] == 7
    assert verification["recomputation"]["category_artifact_count"] == 42
    assert verification["recomputation"]["planning_shard_descriptor_count"] == 344
    assert verification["recomputation"]["pruned_category_route_count"] == 42
    assert verification["recomputation"]["raw_receipt_count"] == 0
    assert verification["recomputation"]["maximum_raw_receipt_count"] == 15_886
    assert verification["mutation_suite"]["count"] == verifier.MUTATION_COUNT == 8
    assert verification["mutation_suite"]["rejected"] == 8
    assert verification["mutation_suite"]["all_rejected"] is True

    output = tmp_path / "proof/categorized.verification.json"
    materialized = verifier.verify(result_path=result_path, output=output)
    assert materialized == verifier.verify(result_path=result_path, output=output)
    verifier.validate_verification(materialized)


def test_verifier_rejects_source_byte_drift_and_resealed_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, result_path, result = _build_all_pruned_graph(tmp_path, monkeypatch)
    verification = verifier.build_verification(result_path)

    original_bytes = result_path.read_bytes()
    result_path.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="source bytes drifted"):
        verifier.validate_verification(verification)
    result_path.write_bytes(original_bytes)

    mutant = copy.deepcopy(result)
    mutant["waves"][0]["routing"][wave.CATEGORY_IDS[0]]["continue"] = True
    mutant.pop("result_sha256")
    mutant["result_sha256"] = canonical_sha256(mutant)
    with pytest.raises(ValueError):
        wave.validate_result(mutant, root=root)

    category_path = wave._category_path(root, 0, wave.CATEGORY_IDS[0])
    category = wave._read_object(category_path)
    category["execution"]["compute_started"] = True
    category.pop("category_sha256")
    category["category_sha256"] = canonical_sha256(category)
    atomic_write_json(category_path, category)
    with pytest.raises(ValueError):
        verifier.build_verification(result_path)
