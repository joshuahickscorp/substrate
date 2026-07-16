from __future__ import annotations

import copy

import pytest

from mop.config import REPO_ROOT
from mop.studies import generation1_successor_batch as successor


def _reseal(batch: dict) -> None:
    batch["batch_sha256"] = successor.canonical_sha256(
        {key: value for key, value in batch.items() if key != "batch_sha256"}
    )


def test_verified_c2_builds_four_fail_closed_successor_drafts() -> None:
    batch = successor.build_batch()
    successor.validate_batch(batch)
    readiness = successor.build_readiness(batch)
    successor.validate_readiness(readiness, batch)

    assert [study["canonical_epoch"] for study in batch["studies"]] == [
        "G1-D1",
        "G1-V1",
        "G1-M1",
        "G1-G1",
    ]
    assert batch["prerequisite"]["c2_complete"] is True
    assert batch["prerequisite"]["independent_verification_complete"] is True
    assert batch["prerequisite"]["c2_training_authorized"] is False
    assert readiness["parallel_preparation_ready"] is True
    assert readiness["execution_ready"] is False
    assert readiness["runnable_study_ids"] == []
    assert batch["execution_strategy"]["lane_order"] == ["G1-D1", "G1-V1", "G1-M1", "G1-G1"]
    assert batch["execution_strategy"]["study_scheduling"] == "sequential_cpu_saturation"


def test_batch_self_seal_rejects_unsealed_mutation() -> None:
    batch = successor.build_batch()
    batch["studies"][0]["question"] += " mutated"
    with pytest.raises(ValueError, match="self-seal"):
        successor.validate_batch(batch)


def test_c3_visible_input_leakage_is_rejected() -> None:
    batch = successor.build_batch()
    c3 = batch["studies"][0]
    c3["visible_inputs"].append("context_id")
    _reseal(batch)
    with pytest.raises(ValueError, match="forbidden input"):
        successor.validate_batch(batch)


def test_seed_overlap_is_rejected_even_after_resealing() -> None:
    batch = successor.build_batch()
    batch["studies"][1]["seed_partitions"][0]["start"] = 20270001
    _reseal(batch)
    with pytest.raises(ValueError, match="seed partitions overlap"):
        successor.validate_batch(batch)


def test_dependency_drift_is_rejected_even_after_resealing() -> None:
    batch = successor.build_batch()
    batch["studies"][3]["dependencies"] = ["G1-C0"]
    _reseal(batch)
    with pytest.raises(ValueError, match="dependency or epoch drifted"):
        successor.validate_batch(batch)


def test_readiness_cannot_claim_a_runnable_study() -> None:
    batch = successor.build_batch()
    readiness = successor.build_readiness(batch)
    readiness["runnable_study_ids"] = ["G1-C3-D1-LEARNED-DISPATCH"]
    readiness["readiness_sha256"] = successor.canonical_sha256(
        {key: value for key, value in readiness.items() if key != "readiness_sha256"}
    )
    with pytest.raises(ValueError, match="falsely authorizes"):
        successor.validate_readiness(readiness, batch)


def test_c2_proof_binding_is_content_addressed() -> None:
    batch = successor.build_batch()
    mutated = copy.deepcopy(batch)
    mutated["prerequisite"]["result_file_sha256"] = "0" * 64
    _reseal(mutated)
    with pytest.raises(ValueError, match="proof binding drifted"):
        successor.validate_batch(mutated, repo_root=REPO_ROOT)
