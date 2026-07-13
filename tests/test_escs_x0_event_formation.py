from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

import mop.studies.escs_x0_event_formation as x0


def _tiny_config() -> dict:
    config = x0.load_config()
    for split in config["splits"].values():
        split["episodes"] = 1
    config["world"]["horizon_ticks"] = 48
    config["world"]["uncertain_identity_rate"] = 1.0
    config["difficulty_gate"].update(
        {
            "min_always_on_utility": -1.0,
            "max_useful_update_rate": 1.0,
            "min_idle_fraction": 0.0,
            "min_irrelevant_fraction": 0.0,
        }
    )
    return config


def _write_exploratory_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    config = _tiny_config()
    envelope = {
        "schema": x0.ENVELOPE_SCHEMA,
        "authority": {
            "schema": x0.AUTHORITY_SCHEMA,
            "mode": "exploratory",
            "contract_id": "tiny-x0-test",
            "payload_sha256": x0.canonical_sha256(config),
        },
        "payload": config,
    }
    config_path = tmp_path / "tiny-x0.json"
    x0._atomic_json(config_path, envelope)
    manifest = x0.build_implementation_authority(
        config_authority_sha256=x0.canonical_sha256(config),
        mode="exploratory",
        review_status="tiny-test-only",
    )
    manifest_path = tmp_path / "tiny-x0.implementation-authority.json"
    x0._atomic_json(manifest_path, manifest)
    return config_path, manifest_path, manifest["manifest_sha256"]


def test_official_preregistration_is_frozen_and_nonpromotable() -> None:
    config = x0.load_config()
    assert x0.OFFICIAL_CONFIG_AUTHORITY_SHA256 != "__X0_CONFIG_AUTHORITY_SHA256__"
    assert x0.canonical_sha256(config) == x0.OFFICIAL_CONFIG_AUTHORITY_SHA256
    assert tuple(config["arms"]) == x0.ARM_NAMES
    assert len(config["seeds"]) >= 5
    assert len(config["fresh_verifier_seeds"]) >= 5
    assert set(config["seeds"]).isdisjoint(config["fresh_verifier_seeds"])
    assert config["verdict"]["scientific_promotion"] == "blocked"
    assert config["criteria"]["max_utility_loss_vs_always_on"] == 0.01
    assert config["criteria"]["min_work_saving_vs_always_on"] == 0.25


def test_generated_trace_exercises_every_registered_raw_stream_property() -> None:
    config = _tiny_config()
    trace = x0.generate_episode(config, seed=config["seeds"][0], split="heldout", episode=0)
    cases = trace.packets
    assert trace.idle_wall_ticks > 0
    assert any(case.evaluator.useful for case in cases)
    assert any(case.evaluator.irrelevant_change for case in cases)
    assert any(case.evaluator.irreducible_noise for case in cases)
    assert any(case.evaluator.storm for case in cases)
    assert any(case.visible.identity_token == "source:unknown" for case in cases)
    assert any(case.visible.arrival_tick > case.visible.capture_tick for case in cases)
    assert len({case.visible.arrival_tick for case in cases}) < len(cases)
    assert all(
        case.evaluator.consequence_tick > case.visible.capture_tick for case in cases if case.evaluator.useful
    )
    tune_events = set(config["splits"]["tune"]["event_types"])
    tune_clocks = set(config["splits"]["tune"]["clock_families"])
    assert tune_events.isdisjoint(trace.event_types)
    assert trace.clock_family not in tune_clocks


def test_policy_boundary_exposes_only_visible_packets_and_delayed_public_value() -> None:
    config = _tiny_config()
    observations: list[x0.TrainingObservation] = []
    packets: list[x0.VisiblePacket] = []
    bootstrap_keys: list[set[str]] = []

    class SpyPolicy:
        descriptor = x0.PolicyDescriptor("test:spy", "candidate_unverified", False)
        retained_state_bytes = 8
        threshold = 0.5

        def fit(self, visible, feedback):
            packets.extend(visible)
            observations.extend(feedback)
            assert all(type(packet) is x0.VisiblePacket for packet in visible)
            assert all(type(row) is x0.TrainingObservation for row in feedback)
            return {
                "schema": "mop-escs-x0-training-receipt/v1",
                "policy_id": "test:spy",
                "visible_packet_count": len(visible),
                "public_feedback_count": len(feedback),
                "positive_public_consequence_count": sum(
                    row.consequence.realized_action_value > 0 for row in feedback
                ),
                "feedback_delivery_min_tick": min(row.consequence.delivered_tick for row in feedback),
                "feedback_delivery_max_tick": max(row.consequence.delivered_tick for row in feedback),
                "training_operations": len(feedback),
                "retained_state_bytes": 8,
                "threshold": 0.5,
                "state_sha256": x0.canonical_sha256(["spy", len(feedback)]),
                "evaluator_labels_visible": False,
            }

        def score(self, packet):
            assert type(packet) is x0.VisiblePacket
            return 0.75 if packet.header_delta_milli > 5000 else 0.25

    def factory(policy_config, _seed):
        bootstrap_keys.append(set(policy_config))
        assert set(policy_config) == {"learned_policy"}
        assert set(policy_config["learned_policy"]) == x0._POLICY_BOOTSTRAP_FIELDS
        return SpyPolicy()

    row = x0.run_seed(
        config,
        seed=config["seeds"][0],
        split="heldout",
        policy_factory=factory,
    )
    assert packets and observations and bootstrap_keys
    visible_fields = {field.name for field in dataclasses.fields(x0.VisiblePacket)}
    evaluator_fields = {field.name for field in dataclasses.fields(x0.EvaluatorTruth)}
    consequence_fields = {field.name for field in dataclasses.fields(x0.PublicConsequence)}
    assert visible_fields.isdisjoint(evaluator_fields)
    assert consequence_fields == x0.PUBLIC_CONSEQUENCE_FIELDS
    assert row["leakage_gate"]["passed"]


def test_tiny_seed_has_all_arms_full_lifecycle_metrics_and_exact_rate_shuffle() -> None:
    config = _tiny_config()
    row = x0.run_seed(config, seed=config["seeds"][0], split="heldout")
    assert tuple(row["arms"]) == x0.ARM_NAMES
    learned = row["arms"]["learned_event_former"]
    shuffled = row["arms"]["shuffled_rate_matched"]
    assert learned["admitted_count"] == shuffled["admitted_count"]
    assert set(learned["work_components"]) == set(x0.WORK_COMPONENTS)
    assert learned["total_lifecycle_work"] == x0._work_total(learned["work_components"], config)
    assert learned["header_encoded_bytes"] > 0
    assert learned["header_operations"] > 0
    assert learned["candidate_count"] == learned["packet_count"]
    assert learned["discarded_candidate_count"] + learned["admitted_count"] == learned["packet_count"]
    assert learned["idle_adapter_and_event_former_work"] > 0
    assert learned["retained_state_byte_ticks"] > 0
    assert learned["queue_retained_state_byte_ticks"] >= 0
    assert learned["retained_state_bytes"] == (
        learned["policy_retained_state_bytes"] + learned["peak_queue_retained_state_bytes"]
    )
    assert learned["work_components"]["serialization_and_receipts"] >= learned["header_encoded_bytes"]
    assert learned["calibration"]["scored_packet_count"] == learned["packet_count"]
    assert learned["scientific_promotion"] is False
    oracle = row["arms"]["oracle_semantic_nonpromotable"]
    assert oracle["oracle_access"] is True
    assert oracle["evidence_standing"] == "oracle_nonpromotable"
    assert oracle["scientific_promotion"] is False


def test_difficulty_gate_failure_is_invalid_bed_not_null() -> None:
    config = _tiny_config()
    rows = [x0.run_seed(config, seed=seed, split="gate") for seed in config["seeds"]]
    valid = x0.difficulty_gate(rows, config)
    assert valid["passed"]
    broken = copy.deepcopy(config)
    broken["difficulty_gate"]["min_always_on_utility"] = 2.0
    invalid = x0.difficulty_gate(rows, broken)
    assert not invalid["passed"]
    assert invalid["failure_interpretation"] == "invalid_bed_not_mechanism_null"


def test_atomic_phase_seed_resume_and_disjoint_fresh_verification(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _write_exploratory_bundle(tmp_path)
    config = _tiny_config()
    output = tmp_path / "receipt.json"
    checkpoint = tmp_path / "checkpoint.json"
    partial = x0.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    assert partial["resumable"]
    assert len(partial["gate_rows"]) == 1
    complete = x0.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert complete["execution_status"] == "complete"
    first_bytes = output.read_bytes()
    resumed = x0.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert not resumed["resumable"]
    assert output.read_bytes() == first_bytes
    verification = x0.verify_receipt(
        output,
        config_path,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert verification["producer_regeneration_match"] is True
    assert verification["fresh_difficulty_gate"]["passed"] is True
    assert verification["scientific_promotion"] is False
    assert set(verification["fresh_seed_ids"]).isdisjoint(config["seeds"])
    assert verification["fresh_seed_ids"] == config["fresh_verifier_seeds"]


def test_receipt_and_checkpoint_tampering_fail_closed(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _write_exploratory_bundle(tmp_path)
    output = tmp_path / "receipt.json"
    checkpoint = tmp_path / "checkpoint.json"
    receipt = x0.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    forged = copy.deepcopy(receipt)
    forged["fresh_verifier_status"] = "forged"
    output.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt self-hash"):
        x0.verify_receipt(
            output,
            config_path,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
        )
    checkpoint_document = json.loads(checkpoint.read_text(encoding="utf-8"))
    checkpoint_document["config_authority_sha256"] = "0" * 64
    x0._atomic_json(checkpoint, checkpoint_document)
    with pytest.raises(ValueError, match="checkpoint self-hash"):
        x0.run_from_config(
            config_path,
            tmp_path / "other-receipt.json",
            checkpoint,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
        )


def test_canonical_implementation_authority_binds_every_scoped_file() -> None:
    config = x0.load_config()
    manifest = json.loads(x0.DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.read_text(encoding="utf-8"))
    loaded = x0.load_implementation_authority(
        x0.DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
        config,
        expected_sha256=manifest["manifest_sha256"],
        exploratory=False,
    )
    assert loaded["review_status"] == x0.OFFICIAL_IMPLEMENTATION_REVIEW_STATUS
    assert loaded["files"] == [x0._file_receipt(x0.REPO_ROOT / path) for path in x0.IMPLEMENTATION_PATHS]


def test_header_only_cheaper_noninferior_control_blocks_favorable_pattern() -> None:
    config = _tiny_config()
    source = x0.run_seed(config, seed=config["seeds"][0], split="heldout")
    rows = []
    for seed in config["seeds"]:
        row = copy.deepcopy(source)
        row["seed"] = seed
        learned = row["arms"]["learned_event_former"]
        header = row["arms"]["header_only"]
        header["mean_utility"] = learned["mean_utility"]
        header["total_lifecycle_work"] = max(1, learned["total_lifecycle_work"] - 1)
        rows.append(row)

    aggregate = x0.aggregate_rows(rows, config)

    assert aggregate["checks"]["header_does_not_explain_full_result"] is False


def test_official_run_refuses_an_unpinned_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent manifest digest"):
        x0.run_from_config(
            output_path=tmp_path / "receipt.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )
