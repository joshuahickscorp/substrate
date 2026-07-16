from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import mop.studies.escs_x2_event_graph as x2


def _tiny_config() -> dict:
    config = x2.load_config()
    config["mechanics"].update(
        {
            "branch_count_per_referent": 1,
            "history_distractors": 5,
            "queries_per_seed": 12,
            "referents_per_world": 4,
            "session_count": 3,
        }
    )
    config["controls"].update(
        {
            "bounded_history_records": 5,
            "fixed_recurrent_slots": 2,
            "kv_capacity": 8,
            "periodic_summary_interval": 3,
        }
    )
    return config


def _exploratory_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    config = _tiny_config()
    envelope = {
        "schema": x2.ENVELOPE_SCHEMA,
        "authority": {
            "schema": x2.AUTHORITY_SCHEMA,
            "mode": "exploratory",
            "contract_id": "tiny-x2-test",
            "payload_sha256": x2.canonical_sha256(config),
        },
        "payload": config,
    }
    config_path = tmp_path / "tiny-x2.json"
    x2._atomic_json(config_path, envelope)
    manifest = x2.build_implementation_authority(
        config_authority_sha256=x2.canonical_sha256(config),
        mode="exploratory",
        review_status="tiny-test-only",
    )
    manifest_path = tmp_path / "tiny-x2.implementation-authority.json"
    x2._atomic_json(manifest_path, manifest)
    return config_path, manifest_path, manifest["manifest_sha256"]


def test_official_config_is_frozen_disabled_and_action_value_gated() -> None:
    config = x2.load_config()
    assert x2.canonical_sha256(config) == x2.OFFICIAL_CONFIG_AUTHORITY_SHA256
    assert tuple(config["arms"]) == x2.ARM_NAMES
    assert len(config["seeds"]) >= 5
    assert len(config["fresh_verifier_seeds"]) >= 5
    assert set(config["seeds"]).isdisjoint(config["fresh_verifier_seeds"])
    assert config["activation"] == {
        "enabled": False,
        "scientific_promotion_allowed": False,
    }
    assert config["criteria"]["min_intervention_ranking_gain"] == 0.05
    assert config["criteria"]["min_saving_fraction"] == 0.30
    assert config["verdict"]["prediction_only_claim_forbidden"] is True
    assert config["verdict"]["scientific_promotion"] == "blocked"
    prerequisite_paths = {row["path"] for row in config["prerequisites"]}
    assert "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json" in prerequisite_paths
    assert "proof/P6_CONTINUAL_STREAM_10K_VERIFICATION.json" in prerequisite_paths
    assert "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json" in prerequisite_paths


def test_fixture_has_revision_deletion_poison_branch_schema_and_disjoint_sessions() -> None:
    config = _tiny_config()
    fixture = x2.generate_fixture(config, seed=config["seeds"][0], split="producer")
    assert fixture.records and fixture.queries
    assert any(row.supersedes_record_id is not None for row in fixture.records)
    assert any(row.deletion_target_id is not None for row in fixture.records)
    assert any(row.poisoned for row in fixture.records)
    assert any(not row.factual for row in fixture.records)
    assert {row.session_id for row in fixture.records}.isdisjoint({row.session_id for row in fixture.queries})
    assert all(row.schema_id != "canonical" for row in fixture.queries)
    assert all(len(row.action_values_milli) == len(fixture.actions) for row in fixture.queries)
    assert any(row.ambiguous for row in fixture.queries)
    payload = fixture.payload()
    expected = payload.pop("payload_sha256")
    assert expected == x2.canonical_sha256(payload)


def test_repository_native_event_archive_and_accounting_mechanics_pass() -> None:
    result = x2.build_escs_mechanics_fixture(42)
    assert result["all_checks_passed"]
    assert all(result["checks"].values())
    assert result["factual_event_count"] == 5
    assert result["counterfactual_event_count"] == 3
    assert result["archive_replay_authority"] == "disabled-after-erasure"
    assert result["lifecycle_work"]["idle_floor"] > 0
    assert result["lifecycle_work"]["indexing_and_graph_maintenance"] > 0
    assert result["lifecycle_work"]["counterfactual_credit"] > 0


def test_all_arms_report_full_boundary_and_graph_integrity_without_prediction_claim() -> None:
    config = _tiny_config()
    row = x2.run_seed(config, seed=config["seeds"][0], split="producer")
    assert tuple(row["arms"]) == x2.ARM_NAMES
    assert row["difficulty_gate"]["passed"]
    for arm, result in row["arms"].items():
        assert tuple(result["work_components"]) == x2.WORK_COMPONENTS
        assert result["abstract_operation_work"] > 0
        assert result["retained_state_bytes"] > 0
        assert result["retained_byte_time"] > 0
        assert result["serialized_bytes"] > 0
        assert 0.0 <= result["heldout_intervention_ranking_accuracy"] <= 1.0
        assert 0.0 <= result["heldout_realized_action_value"] <= 1.0
        assert result["prediction_only_claim"] is False
        assert result["activation_enabled"] is False
        assert result["scientific_promotion"] is False
        if arm != "escs_event_graph":
            assert result["evidence_standing"] in {"control_only", "oracle_nonpromotable"}
    graph = row["arms"]["escs_event_graph"]
    assert graph["all_integrity_gates_passed"]
    assert all(graph["integrity_gates"].values())
    assert row["arms"]["oracle_state_nonpromotable"]["evidence_standing"] == "oracle_nonpromotable"


def test_paired_gate_routes_valid_result_without_promoting_science() -> None:
    config = _tiny_config()
    rows = [x2.run_seed(config, seed=seed, split="producer") for seed in config["seeds"]]
    aggregate = x2.aggregate_rows(rows, config)
    assert aggregate["valid_experimental_bed"]
    assert aggregate["minimum_paired_seed_gate_passed"]
    assert aggregate["all_integrity_gates_passed"]
    assert aggregate["terminal_route"] in {"positive_candidate", "controlled_null"}
    assert aggregate["prediction_only_claim"] is False
    assert aggregate["scientific_promotion"] is False

    too_few = x2.aggregate_rows(rows[:1], config)
    assert not too_few["minimum_paired_seed_gate_passed"]
    assert too_few["terminal_route"] == "controlled_null"

    invalid_rows = copy.deepcopy(rows)
    invalid_rows[0]["difficulty_gate"]["passed"] = False
    assert x2.aggregate_rows(invalid_rows, config)["terminal_route"] == "invalid_bed"

    failed_rows = copy.deepcopy(rows)
    failed_rows[0]["escs_mechanics"]["all_checks_passed"] = False
    assert x2.aggregate_rows(failed_rows, config)["terminal_route"] == "failed"


def test_producer_and_fresh_verifier_resume_with_disjoint_sealed_receipts(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _exploratory_bundle(tmp_path)
    producer = tmp_path / "producer.json"
    producer_checkpoint = tmp_path / "producer.checkpoint.json"
    partial = x2.run_from_config(
        config_path,
        producer,
        producer_checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    assert partial["execution_status"] == "partial"
    assert partial["resumable"]
    complete = x2.run_from_config(
        config_path,
        producer,
        producer_checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert complete["execution_status"] == "complete"
    assert complete["fresh_verifier_status"] == "required"
    producer_bytes = producer.read_bytes()
    repeated = x2.run_from_config(
        config_path,
        producer,
        producer_checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert repeated["receipt_sha256"] == complete["receipt_sha256"]
    assert producer.read_bytes() == producer_bytes

    verification = tmp_path / "verification.json"
    verification_checkpoint = tmp_path / "verification.checkpoint.json"
    partial_verification = x2.verify_receipt(
        producer,
        config_path,
        manifest_path,
        verification,
        verification_checkpoint,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    assert partial_verification["verification_status"] == "partial"
    final = x2.verify_receipt(
        producer,
        config_path,
        manifest_path,
        verification,
        verification_checkpoint,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert final["verification_status"] == "complete"
    assert final["producer_regeneration_match"]
    assert final["seed_sets_disjoint"]
    assert final["fresh_session_prefix_disjoint"]
    assert final["activation_enabled"] is False
    assert final["prediction_only_claim"] is False
    assert final["scientific_promotion"] is False


def test_receipt_checkpoint_and_authority_tampering_fail_closed(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _exploratory_bundle(tmp_path)
    producer = tmp_path / "producer.json"
    checkpoint = tmp_path / "checkpoint.json"
    x2.run_from_config(
        config_path,
        producer,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    forged_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
    forged_checkpoint["config_authority_sha256"] = "0" * 64
    x2._atomic_json(checkpoint, forged_checkpoint)
    with pytest.raises(ValueError, match="checkpoint self-hash"):
        x2.run_from_config(
            config_path,
            producer,
            checkpoint,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["activation_enabled"] = True
    x2._atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="authority self-hash"):
        x2.preflight(config_path, manifest_path, exploratory=True)


def test_official_execution_stays_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="official X2 activation is disabled|X2 implementation files differ from authority",
    ):
        x2.run_from_config(
            x2.DEFAULT_CONFIG_PATH,
            Path("/tmp/x2-should-not-exist.json"),
            Path("/tmp/x2-should-not-exist.checkpoint.json"),
            x2.DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
            exploratory=False,
        )
