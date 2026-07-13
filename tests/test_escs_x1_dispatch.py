from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

import mop.studies.escs_x1_dispatch as x1


def _tiny_config() -> dict:
    config = x1.load_config()
    for split in config["splits"].values():
        split["episodes"] = 1
        split["horizon"] = 20
    config["difficulty_and_complementarity_gate"].update(
        {
            "min_always_on_utility": 0.0,
            "min_oracle_headroom_over_best_single": 0.0,
            "max_best_single_utility": 1.0,
            "min_unique_niche_cases_per_core_actor": 1,
            "min_synergy_cases": 1,
            "min_synergy_pair_interaction": 0.1,
        }
    )
    return config


def _entry_gate(passed: bool = True) -> dict:
    return {
        "schema": "mop-escs-x1-edcm-entry-gate/v1",
        "valid_terminal_evidence": True,
        "passed": passed,
        "route": "continue_x1" if passed else "invalid_bed_return_to_edcm1",
        "fixture": "tests-only-not-official-evidence",
        "scientific_promotion": False,
    }


def _write_exploratory_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    config = _tiny_config()
    envelope = {
        "schema": x1.ENVELOPE_SCHEMA,
        "authority": {
            "schema": x1.AUTHORITY_SCHEMA,
            "mode": "exploratory",
            "contract_id": "tiny-x1-test",
            "payload_sha256": x1.canonical_sha256(config),
        },
        "payload": config,
    }
    config_path = tmp_path / "tiny-x1.json"
    x1._atomic_json(config_path, envelope)
    manifest = x1.build_implementation_authority(
        config_authority_sha256=x1.canonical_sha256(config),
        mode="exploratory",
        review_status="tiny-test-only",
    )
    manifest_path = tmp_path / "tiny-x1.implementation-authority.json"
    x1._atomic_json(manifest_path, manifest)
    return config_path, manifest_path, manifest["manifest_sha256"]


def _write_edcm_pair(tmp_path: Path, *, passed: bool) -> tuple[Path, Path]:
    producer_core = {
        "schema": x1.EDCM_RECEIPT_SCHEMA,
        "execution_status": "complete" if passed else "terminal_scientific_stop",
        "all_ok": True,
        "problems": [],
        "resumable": False,
        "gate": {"status": "complete", "passed": passed},
        "authority_sha256": "a" * 64,
        "implementation_authority_sha256": "b" * 64,
        "scientific_promotion": False,
    }
    producer_with_core = {
        **producer_core,
        "deterministic_core_sha256": x1.canonical_sha256(producer_core),
    }
    producer = {
        **producer_with_core,
        "receipt_sha256": x1.canonical_sha256(producer_with_core),
    }
    producer_path = tmp_path / "edcm.json"
    x1._atomic_json(producer_path, producer)
    source = x1._file_receipt(producer_path)
    verification_result = {
        "valid": True,
        "execution_status": producer["execution_status"],
        "verifier_mode": "full-deterministic-regeneration/v1",
        "scientific_promotion": False,
        "verified_sources": {"receipt": source},
    }
    verifier_core = {
        "schema": x1.EDCM_VERIFICATION_SCHEMA,
        "study_id": "edcm1-event-triggered-heterogeneous-coalition-crossover-v3",
        "claim_scope": "event-triggered-coalition-mechanics-only",
        "verification": verification_result,
        "scientific_promotion": False,
    }
    verifier = {
        **verifier_core,
        "verification_artifact_sha256": x1.canonical_sha256(verifier_core),
    }
    verifier_path = tmp_path / "edcm.verification.json"
    x1._atomic_json(verifier_path, verifier)
    return producer_path, verifier_path


def test_official_config_is_frozen_disabled_and_nonpromotable() -> None:
    config = x1.load_config()
    assert x1.canonical_sha256(config) == x1.OFFICIAL_CONFIG_AUTHORITY_SHA256
    assert tuple(config["arms"]) == x1.ARM_NAMES
    assert len(config["seeds"]) == len(config["fresh_verifier_seeds"]) == 5
    assert set(config["seeds"]).isdisjoint(config["fresh_verifier_seeds"])
    assert config["candidate_activation_enabled"] is False
    assert config["scientific_promotion"] is False
    assert config["criteria"]["max_utility_loss_vs_always_on"] == 0.01
    assert config["criteria"]["min_work_saving_vs_always_on"] == 0.25


def test_policy_boundary_has_no_evaluator_or_future_fields() -> None:
    visible = {field.name for field in dataclasses.fields(x1.VisibleHeader)}
    evaluator = {field.name for field in dataclasses.fields(x1.EvaluatorTruth)}
    credit = {field.name for field in dataclasses.fields(x1.ExactCreditRecord)}
    assert visible == x1.VISIBLE_HEADER_FIELDS
    assert evaluator == x1.EVALUATOR_ONLY_FIELDS
    assert credit == x1.EXACT_CREDIT_FIELDS
    assert visible.isdisjoint(evaluator)
    assert "future_consequence_milli" not in visible
    assert x1.leakage_gate()["passed"]


def test_compact_decision_and_work_bridge_to_shared_escs_contracts() -> None:
    config = _tiny_config()
    case = x1.generate_cases(config, seed=config["seeds"][0], split="heldout")[0]
    decision = x1.CoalitionDecision.create(
        "test:bridge",
        case.header,
        ("reactive_spatial", "contradiction_verifier"),
        candidates_considered=4,
        coalitions_considered=6,
    )
    projected = decision.runtime_selection()
    assert projected.selected_actor_ids == decision.actor_ids
    charges = x1.WorkCharges(
        structured_intake=1,
        idle_header_floor=2,
        candidate_retrieval=3,
        readiness_bids=4,
        dispatch_search=5,
        exploration=6,
        actor_execution=7,
        message_operations=8,
        exact_counterfactuals=9,
        critic_training=10,
        stale_reactivation=11,
        receipt_serialization=12,
    )
    shared = charges.as_escs_work_vector()
    assert shared.total_work == charges.total
    assert shared.dispatch_and_exploration == 3 + 4 + 5 + 6
    assert shared.counterfactual_credit == 9


def test_exact_credit_fixtures_distinguish_synergy_and_redundancy() -> None:
    config = _tiny_config()
    cases = x1.generate_cases(config, seed=config["seeds"][0], split="heldout")
    synergy = next(case for case in cases if case.evaluator.niche_label == "binding")
    synergy_credit = x1.exact_credit_record(synergy, tuple(config["actors"]))
    assert synergy_credit.individual_marginal_milli["binder_left"] == 0
    assert synergy_credit.individual_marginal_milli["binder_right"] == 0
    assert synergy_credit.pair_interaction_milli["binder_left|binder_right"] > 0
    differences = x1.exact_difference_credit(synergy, x1.SYNERGY_PAIR)
    assert differences["binder_left"] > 0 and differences["binder_right"] > 0

    redundant = next(case for case in cases if case.evaluator.niche_label == "redundancy")
    redundant_credit = x1.exact_credit_record(redundant, tuple(config["actors"]))
    assert redundant_credit.pair_interaction_milli["episodic_retrieval|redundant_retrieval"] < 0
    assert synergy_credit.fork_count == 1 + len(config["actors"]) + 28


def test_generated_worlds_are_deterministic_and_split_disjoint() -> None:
    config = _tiny_config()
    first = x1.generate_cases(config, seed=config["seeds"][0], split="heldout")
    second = x1.generate_cases(config, seed=config["seeds"][0], split="heldout")
    fresh = x1.generate_cases(config, seed=config["fresh_verifier_seeds"][0], split="fresh_verifier")
    assert first == second
    assert {case.header.world_token for case in first}.isdisjoint(case.header.world_token for case in fresh)
    assert any(case.header.idle for case in first)
    assert any(case.evaluator.irreducible_noise for case in first)
    assert all("noise" not in case.header.factor_scope for case in first if case.evaluator.irreducible_noise)
    assert any(case.header.storm for case in first)
    storm_ticks = [case.header.created_tick for case in first if case.header.storm]
    assert len(storm_ticks) > len(set(storm_ticks))


def test_one_seed_charges_full_boundary_and_exact_rate_controls() -> None:
    config = _tiny_config()
    row = x1.run_seed(config, seed=config["seeds"][0], split="heldout")
    assert tuple(row["arms"]) == x1.ARM_NAMES
    assert all(row["invariants"].values())
    primary = row["arms"][x1.PRIMARY_ARM]
    assert set(primary["work_components"]) == set(x1.WORK_COMPONENTS)
    assert sum(primary["work_components"].values()) == primary["total_lifecycle_work"]
    assert primary["idle_boundary_work"] > 0
    assert primary["total_encoded_bytes"] > 0
    assert primary["retained_state_byte_ticks"] > 0
    assert primary["deployment_oracle_forks"] == 0
    assert primary["temporary_coalitions_only"] is True
    assert primary["queue_stable"] is True
    assert primary["max_queue_depth"] >= 3
    scaling = primary["dormant_population_scaling"]
    assert scaling["global_scan"] is False
    assert scaling["total_charged_index_and_query_work"] == primary["training"]["dormant_scaling_assay_work"]
    assert scaling["total_retained_index_byte_ticks"] <= primary["retained_state_byte_ticks"]
    for control in ("periodic_exact_rate", "random_exact_rate", "shuffled_exact_rate"):
        assert row["arms"][control]["actor_activation_counts"] == primary["actor_activation_counts"]
    oracle = row["arms"]["oracle_dispatch_nonpromotable"]
    assert oracle["oracle_access"] is True
    assert oracle["evidence_standing"] == "oracle_nonpromotable"
    assert oracle["scientific_promotion"] is False


def test_terminal_routing_separates_invalid_bed_null_and_failed() -> None:
    config = _tiny_config()
    row = x1.run_seed(config, seed=config["seeds"][0], split="heldout")
    valid = x1.aggregate_rows([row], config)
    assert valid["terminal_route"] in {"positive", "null"}

    invalid_row = copy.deepcopy(row)
    invalid_row["bed_gate"]["passed"] = False
    invalid = x1.aggregate_rows([invalid_row], config)
    assert invalid["terminal_route"] == "invalid_bed"

    failed_row = copy.deepcopy(row)
    failed_row["invariants"]["no_global_actor_scan"] = False
    failed = x1.aggregate_rows([failed_row], config)
    assert failed["terminal_route"] == "failed"


def test_edcm_entry_gate_requires_bound_independent_verification(tmp_path: Path) -> None:
    producer, verifier = _write_edcm_pair(tmp_path, passed=True)
    entry = x1.load_edcm_entry_gate(producer, verifier)
    assert entry["valid_terminal_evidence"] is True
    assert entry["passed"] is True
    assert entry["route"] == "continue_x1"

    invalid_producer, invalid_verifier = _write_edcm_pair(tmp_path / "invalid", passed=False)
    invalid = x1.load_edcm_entry_gate(invalid_producer, invalid_verifier)
    assert invalid["passed"] is False
    assert invalid["route"] == "invalid_bed_return_to_edcm1"

    artifact = json.loads(verifier.read_text(encoding="utf-8"))
    artifact["verification"]["verified_sources"]["receipt"]["sha256"] = "0" * 64
    core = dict(artifact)
    core.pop("verification_artifact_sha256")
    artifact["verification_artifact_sha256"] = x1.canonical_sha256(core)
    x1._atomic_json(verifier, artifact)
    with pytest.raises(ValueError, match="not bound"):
        x1.load_edcm_entry_gate(producer, verifier)


def test_seed_boundary_resume_and_disjoint_fresh_verifier(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _write_exploratory_bundle(tmp_path)
    output = tmp_path / "receipt.json"
    checkpoint = tmp_path / "checkpoint.json"
    partial = x1.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
        entry_gate_override=_entry_gate(),
    )
    assert partial["resumable"] is True
    assert len(partial["gate_rows"]) == 1
    complete = x1.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
        entry_gate_override=_entry_gate(),
    )
    assert complete["execution_status"] == "complete"
    assert complete["candidate_activation_enabled"] is False
    before = output.read_bytes()
    repeated = x1.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
        entry_gate_override=_entry_gate(),
    )
    assert repeated["resumable"] is False
    assert output.read_bytes() == before
    verification = x1.verify_receipt(
        output,
        config_path,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
        entry_gate_override=_entry_gate(),
    )
    assert verification["producer_regeneration_match"] is True
    assert verification["scientific_promotion"] is False
    assert set(verification["fresh_seed_ids"]).isdisjoint(_tiny_config()["seeds"])
    assert verification["fresh_seed_ids"] == _tiny_config()["fresh_verifier_seeds"]
    artifact = x1.build_verification_artifact(verification)
    assert artifact["candidate_activation_enabled"] is False
    assert artifact["scientific_promotion"] is False


def test_checkpoint_and_receipt_tampering_fail_closed(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _write_exploratory_bundle(tmp_path)
    output = tmp_path / "receipt.json"
    checkpoint = tmp_path / "checkpoint.json"
    x1.run_from_config(
        config_path,
        output,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
        entry_gate_override=_entry_gate(),
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    original_receipt = copy.deepcopy(receipt)
    receipt["candidate_activation_enabled"] = True
    x1._atomic_json(output, receipt)
    with pytest.raises(ValueError, match="receipt self-hash"):
        x1.verify_receipt(
            output,
            config_path,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
            entry_gate_override=_entry_gate(),
        )
    spliced = copy.deepcopy(original_receipt)
    spliced["unknown_claim"] = "forged"
    spliced_core = dict(spliced)
    spliced_core.pop("receipt_sha256")
    spliced["receipt_sha256"] = x1.canonical_sha256(spliced_core)
    x1._atomic_json(output, spliced)
    with pytest.raises(ValueError, match="keys differ"):
        x1._load_receipt(output)
    checkpoint_document = json.loads(checkpoint.read_text(encoding="utf-8"))
    checkpoint_document["entry_gate_sha256"] = "0" * 64
    x1._atomic_json(checkpoint, checkpoint_document)
    with pytest.raises(ValueError, match="checkpoint self-hash"):
        x1.run_from_config(
            config_path,
            tmp_path / "second-receipt.json",
            checkpoint,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
            entry_gate_override=_entry_gate(),
        )


def test_official_run_refuses_unpinned_manifest_before_edcm_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="independent X1 implementation-manifest digest"):
        x1.run_from_config(
            output_path=tmp_path / "receipt.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )
