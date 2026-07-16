from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from mop.studies import generation1_d1_frozen_verify as verifier


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ci(mean: float, *, n: int, half: float = 0.001) -> dict[str, float | int]:
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n}


def _phase() -> dict[str, Any]:
    n = verifier.PHASE_CELL_COUNT
    overall = {
        "context_route_nonpromotable": _ci(0.174, n=n),
        "difficulty_static": _ci(0.141, n=n),
        "global_static": _ci(0.130, n=n),
        "learned_dispatch": _ci(0.150, n=n),
        "oracle_nonpromotable": _ci(0.400, n=n),
        "random_actor": _ci(0.142, n=n),
    }
    differences = {
        "context_route_nonpromotable": _ci(-0.024, n=n),
        "difficulty_static": _ci(0.009, n=n),
        "global_static": _ci(0.020, n=n),
        "random_actor": _ci(0.008, n=n),
    }
    return {
        "grid": {
            "rung_count": verifier.RUNGS_PER_PHASE,
            "train_seed_count": verifier.PHASE_SEED_COUNT,
            "heldout_seed_count": verifier.PHASE_SEED_COUNT,
            "completed_cell_count": n,
        },
        "overall": overall,
        "learned_dispatch_differences": differences,
        "favorable_seed_fraction": {
            "difficulty_static": 0.80,
            "global_static": 0.90,
        },
        "mean_gap_below_context_route": 0.024,
        "work_saving_vs_all_five_actors": 0.80,
        "conditions": {
            "static_margin_gate": False,
            "favorable_seed_fraction_gate": True,
            "context_route_gap_gate": False,
            "work_saving_gate": True,
        },
        "all_frozen_criteria_passed": False,
    }


def aggregate_fixture() -> dict[str, Any]:
    producer = _phase()
    challenge = _phase()
    core = {
        "schema": verifier.AGGREGATE_SCHEMA,
        "program_id": verifier.PROGRAM_ID,
        "claim_scope": verifier.AGGREGATE_CLAIM_SCOPE,
        "screen_binding": dict(verifier.SCREEN_BINDING),
        "frozen_variant": dict(verifier.FROZEN_VARIANT),
        "criteria": dict(verifier.CRITERIA),
        "rungs": [
            {
                "index": index,
                "phase": "producer" if index < verifier.RUNGS_PER_PHASE else "challenge",
                "result_sha256": _sha(f"result-{index}"),
                "config_sha256": _sha(f"config-{index}"),
                "completed_cell_count": verifier.CELLS_PER_RUNG,
            }
            for index in range(verifier.RUNG_COUNT)
        ],
        "phases": {"producer": producer, "challenge": challenge},
        "grid": {
            "phase_count": 2,
            "rung_count": verifier.RUNG_COUNT,
            "train_seed_count": verifier.TOTAL_SEED_COUNT,
            "heldout_seed_count": verifier.TOTAL_SEED_COUNT,
            "completed_cell_count": verifier.TOTAL_CELL_COUNT,
        },
        "overall": copy.deepcopy(challenge["overall"]),
        "learned_dispatch_differences": copy.deepcopy(challenge["learned_dispatch_differences"]),
        "decision": {
            "producer_all_frozen_criteria_passed": False,
            "challenge_all_frozen_criteria_passed": False,
            "frozen_pattern_repeated": False,
            "independent_verification_complete": False,
            "ready_for_confirmatory_claim": False,
            "next_action": "run_v1_m1_g1_sibling_batch",
        },
        "interpretation_limit": verifier.INTERPRETATION_LIMIT,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": verifier.canonical_sha256(core)}


def _reseal(value: dict[str, Any], field: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    value[field] = verifier.canonical_sha256(core)


def test_structural_summary_stays_unclosed_until_raw_rung_replay() -> None:
    aggregate = aggregate_fixture()
    classification = verifier.classify_frozen_aggregate(aggregate)
    receipt = verifier.verify_frozen_aggregate(aggregate)
    verifier.validate_verification(receipt)

    assert classification == {
        "evidence_classification": "structural_summary_only_unclosed",
        "scientific_interpretation": "summary_consistent_pending_raw_rung_replay",
        "producer_candidate_gate_passed": False,
        "challenge_candidate_gate_passed": False,
        "independent_structural_artifact_verification_complete": False,
        "independent_scientific_generation": False,
        "exact_centroid_design_eligible_for_future_freeze": False,
        "disposition": "replay_all_raw_rungs_before_evidence_closure",
    }
    assert receipt["classification"] == classification
    assert receipt["source"]["result_sha256"] == aggregate["result_sha256"]
    assert receipt["independent_scientific_generation"] is False
    assert receipt["ready_for_confirmatory_claim"] is False


def test_validator_recomputes_gates_and_summary_arithmetic_after_reseal() -> None:
    gate_mutant = aggregate_fixture()
    gate_mutant["phases"]["producer"]["conditions"]["static_margin_gate"] = True
    _reseal(gate_mutant, "result_sha256")
    with pytest.raises(ValueError, match="gate recomputation"):
        verifier.validate_frozen_aggregate(gate_mutant)

    arithmetic_mutant = aggregate_fixture()
    arithmetic_mutant["phases"]["challenge"]["learned_dispatch_differences"]["global_static"]["mean"] = 0.021
    arithmetic_mutant["learned_dispatch_differences"] = copy.deepcopy(
        arithmetic_mutant["phases"]["challenge"]["learned_dispatch_differences"]
    )
    _reseal(arithmetic_mutant, "result_sha256")
    with pytest.raises(ValueError, match="mean difference"):
        verifier.validate_frozen_aggregate(arithmetic_mutant)


def test_validator_binds_exact_rung_and_field_inventories() -> None:
    rung_mutant = aggregate_fixture()
    rung_mutant["rungs"][288]["phase"] = "producer"
    _reseal(rung_mutant, "result_sha256")
    with pytest.raises(ValueError, match="rung 288 identity"):
        verifier.validate_frozen_aggregate(rung_mutant)

    field_mutant = aggregate_fixture()
    field_mutant["scientific_claim"] = "escaped"
    _reseal(field_mutant, "result_sha256")
    with pytest.raises(ValueError, match="field inventory"):
        verifier.validate_frozen_aggregate(field_mutant)

    duplicate_mutant = aggregate_fixture()
    duplicate_mutant["rungs"][1]["result_sha256"] = duplicate_mutant["rungs"][0]["result_sha256"]
    _reseal(duplicate_mutant, "result_sha256")
    with pytest.raises(ValueError, match="duplicate rung result or config"):
        verifier.validate_frozen_aggregate(duplicate_mutant)


def test_verification_rejects_resealed_classification_drift() -> None:
    receipt = verifier.verify_frozen_aggregate(aggregate_fixture())
    receipt["classification"]["evidence_classification"] = "candidate_pattern_nonconfirmatory"
    _reseal(receipt, "verification_sha256")
    with pytest.raises(ValueError, match="classification drifted"):
        verifier.validate_verification(receipt)


def test_explicit_loader_has_no_import_time_result_dependency(tmp_path) -> None:
    aggregate = aggregate_fixture()
    path = tmp_path / "d1.json"
    path.write_text(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt = verifier.load_and_verify_frozen_aggregate(path)
    assert receipt["verification_complete"] is True

    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="not a regular file"):
        verifier.load_and_verify_frozen_aggregate(link)


def _fake_rung_replay(
    aggregate: dict[str, Any],
    rung_root: Any,
) -> dict[str, Any]:
    inventory = [
        {
            **row,
            "path": f"{rung_root}/{row['phase']}/rung_{row['index']:03d}.json",
            "file_sha256": _sha(f"file-{row['index']}"),
        }
        for row in aggregate["rungs"]
    ]
    return {
        "root_path": str(rung_root),
        "rung_count": verifier.RUNG_COUNT,
        "raw_file_bytes": verifier.RUNG_COUNT,
        "rung_inventory": inventory,
        "rung_inventory_sha256": verifier.canonical_sha256(inventory),
        "result_hashes_unique": True,
        "config_hashes_unique": True,
        "seed_partitions_disjoint": True,
        "router_training_seeds_unique": True,
        "producer_summary_sha256": verifier.canonical_sha256(aggregate["phases"]["producer"]),
        "challenge_summary_sha256": verifier.canonical_sha256(aggregate["phases"]["challenge"]),
        "recomputed_result_sha256": aggregate["result_sha256"],
        "aggregate_exactly_recomputed": True,
    }


def test_byte_bound_verification_materializes_and_replays_exact_source(
    tmp_path,
    monkeypatch,
) -> None:
    aggregate = aggregate_fixture()
    source = tmp_path / "d1.json"
    source.write_bytes(verifier.canonical_bytes(aggregate) + b"\n")
    output = tmp_path / "d1.verification.json"
    rung_root = tmp_path / "rungs"
    monkeypatch.setattr(verifier, "_replay_raw_rungs", _fake_rung_replay)

    artifact = verifier.build_byte_bound_verification(source, rung_root=rung_root)
    verifier.validate_byte_bound_verification(
        artifact,
        source_path=source,
        rung_root=rung_root,
    )
    materialized = verifier.materialize_frozen_verification(
        source,
        output,
        rung_root=rung_root,
    )

    assert materialized == artifact
    assert artifact["source"]["file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert output.read_bytes() == verifier.canonical_bytes(artifact) + b"\n"
    assert output.stat().st_mode & 0o777 == 0o644
    assert artifact["independent_scientific_generation"] is False
    assert artifact["ready_for_confirmatory_claim"] is False
    assert artifact["classification"]["evidence_classification"] == "null_safe_prune"
    assert (
        artifact["verification"]["classification"]["evidence_classification"]
        == "structural_summary_only_unclosed"
    )
    assert artifact["classification"]["independent_structural_artifact_verification_complete"] is True

    inventory_mutant = copy.deepcopy(artifact)
    inventory_mutant["rung_replay"]["rung_inventory"][0]["file_sha256"] = _sha("forged-rung-bytes")
    inventory_mutant["rung_replay"]["rung_inventory_sha256"] = verifier.canonical_sha256(
        inventory_mutant["rung_replay"]["rung_inventory"]
    )
    _reseal(inventory_mutant, "artifact_sha256")
    with pytest.raises(ValueError, match="source replay drifted"):
        verifier.validate_byte_bound_verification(
            inventory_mutant,
            source_path=source,
            rung_root=rung_root,
        )

    changed = aggregate_fixture()
    changed["rungs"][0]["result_sha256"] = _sha("changed-result")
    _reseal(changed, "result_sha256")
    source.write_bytes(verifier.canonical_bytes(changed) + b"\n")
    with pytest.raises(ValueError, match="source replay drifted"):
        verifier.validate_byte_bound_verification(
            artifact,
            source_path=source,
            rung_root=rung_root,
        )


def test_raw_recompute_rejects_consistent_summary_mutation_with_unchanged_rungs() -> None:
    recomputed = aggregate_fixture()
    mutant = copy.deepcopy(recomputed)
    producer = mutant["phases"]["producer"]
    for field in ("mean", "lo", "hi"):
        producer["overall"]["learned_dispatch"][field] += 0.002
        for control in verifier.DIFFERENCE_CONTROLS:
            producer["learned_dispatch_differences"][control][field] += 0.002
    producer["mean_gap_below_context_route"] = -producer["learned_dispatch_differences"][
        "context_route_nonpromotable"
    ]["mean"]
    producer["conditions"]["static_margin_gate"] = True
    _reseal(mutant, "result_sha256")

    verifier.validate_frozen_aggregate(mutant)
    with pytest.raises(ValueError, match="does not exactly replay"):
        verifier._assert_recomputed_aggregate_match(mutant, recomputed)
