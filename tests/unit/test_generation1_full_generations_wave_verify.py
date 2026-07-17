from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mop.studies import generation1_full_generations_wave as wave
from mop.studies import generation1_full_generations_wave_verify as verifier
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256
from tests.unit.test_generation1_full_generations_wave import _materialize_gates


def _build_all_pruned_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    """Drive a fourteen-epoch, seven-category miniature where every route prunes to zero compute.

    The three new lanes are always admitted by the freeze gate, so the categorized grid cannot be
    made empty by the parent mechanics survivors alone.  Returning empty work-item tables keeps the
    graph self-consistent (every category eligible flag resolves False) while executing nothing, and
    the single tiny I1 work keeps the substituted-dependency lane verdict present without compute.
    """

    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    root = _materialize_gates(tmp_path, monkeypatch, mechanics_lanes=[])
    i1_source = next(item for item in mechanics.WORK_ITEMS if item.lane_id == wave.I1_LANE_ID)
    monkeypatch.setattr(wave, "category_work_items", lambda _epoch, _category: ())
    monkeypatch.setattr(
        wave,
        "integration_work_items",
        lambda: (
            wave.WaveWorkItem(
                key="tiny_i1",
                origin=wave._OLD_ORIGIN,
                source_index=i1_source.index,
                cycle=wave.EPOCH_CYCLES[-1],
            ),
        ),
    )
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
    result_path = tmp_path / "proof/full_generations.json"
    result = wave.aggregate(root=root, output=result_path)
    return root, result_path, result


def test_full_all_pruned_graph_aggregates_and_independently_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, result_path, result = _build_all_pruned_graph(tmp_path, monkeypatch)

    assert result["grid"]["manifest_capsule_count"] == 123
    assert result["grid"]["balanced_planning_shards_per_compute_capsule"] == wave.INTERNAL_SHARD_COUNT
    assert result["grid"]["maximum_raw_receipt_count"] == 35_255
    assert result["execution"]["executed_item_count"] == 0
    assert result["execution"]["skipped_item_count"] == 1
    assert result["execution"]["compute_started"] is False
    assert result["decision"]["new_mechanism_lanes_admitted"] is True
    assert result["decision"]["d1_redesign_efficacy_executed"] is False

    verification = verifier.build_verification(result_path)
    verifier.validate_verification(verification)
    assert verification["checks"]["result_seal_valid"] is True
    assert verification["checks"]["fresh_mechanics_receipts_valid"] is True
    assert verification["checks"]["null_safe_pruning_valid"] is True
    assert verification["checks"]["dynamic_worker_pool_valid"] is True
    assert verification["checks"]["balanced_planning_shards_valid"] is True
    assert verification["checks"]["new_mechanism_lanes_admitted"] is True
    assert verification["checks"]["substituted_i1_dependency_valid"] is True
    assert verification["checks"]["carried_d1_not_recomputed"] is True
    assert verification["checks"]["d1_redesign_efficacy_not_executed"] is True
    assert verification["checks"]["independent_generator_family_present"] is False
    assert verification["recomputation"]["gate_count"] == 5
    assert verification["recomputation"]["wave_count"] == 14
    assert verification["recomputation"]["category_artifact_count"] == 98
    assert verification["recomputation"]["planning_shard_descriptor_count"] == 98 * 8 + 8
    assert verification["recomputation"]["pruned_category_route_count"] == 98
    assert verification["recomputation"]["raw_receipt_count"] == 0
    assert verification["recomputation"]["new_mechanism_lane_count"] == 3
    assert verification["recomputation"]["i1_dependency_count"] == 6
    assert verification["recomputation"]["maximum_raw_receipt_count"] == 35_255
    assert verification["recomputation"]["manifest_capsule_count"] == 123
    assert verification["mutation_suite"]["count"] == verifier.MUTATION_COUNT == 8
    assert verification["mutation_suite"]["rejected"] == 8
    assert verification["mutation_suite"]["all_rejected"] is True
    assert verification["independent_scientific_confirmation"] is False
    assert verification["activation_allowed"] is False
    assert verification["scientific_promotion"] is False

    output = tmp_path / "proof/full_generations.verification.json"
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


def test_v2_boundary_mutations_and_check_tamper_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, result_path, result = _build_all_pruned_graph(tmp_path, monkeypatch)

    suite = verifier._mutation_suite(result, root)
    assert [row["mutation"] for row in suite["cases"]] == list(verifier.MUTATION_NAMES)
    assert suite["count"] == suite["rejected"] == verifier.MUTATION_COUNT == 8
    assert suite["all_rejected"] is True

    # The two numeric v2 boundary values (19 -> 18 fresh cycle, 123 -> 122 capsules) each reject.
    assert verifier._mutation_rejected(
        result,
        root,
        lambda candidate: candidate["grid"]["fresh_cycle_indices"].__setitem__(0, 18),
    )
    assert verifier._mutation_rejected(
        result,
        root,
        lambda candidate: candidate["grid"].__setitem__("manifest_capsule_count", 122),
    )

    # A tampered verifier check ledger fails the sealed round trip even after resealing.
    verification = verifier.build_verification(result_path)
    tampered = copy.deepcopy(verification)
    tampered["checks"]["independent_generator_family_present"] = True
    tampered.pop("verification_sha256")
    tampered["verification_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ValueError):
        verifier.validate_verification(tampered)
