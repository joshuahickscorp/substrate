from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mop.studies import escs_x3_topology_adaptation as x3


def _reseal_row(row: dict[str, object]) -> None:
    body = copy.deepcopy(row)
    body.pop("row_sha256", None)
    row["row_sha256"] = x3.canonical_sha256(body)


def _exploratory_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    envelope = json.loads(x3.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    envelope["authority"]["mode"] = "exploratory"
    envelope["payload"]["stream"]["events_per_seed"] = 96
    envelope["authority"]["payload_sha256"] = x3.canonical_sha256(envelope["payload"])
    config_path = tmp_path / "x3-config.json"
    x3._atomic_json(config_path, envelope)
    manifest = x3.build_implementation_authority(
        config_authority_sha256=envelope["authority"]["payload_sha256"],
        mode="exploratory",
        review_status="test-scaffold-only",
        scoped_paths=(
            config_path,
            Path(x3.__file__).resolve(),
            x3.REPO_ROOT / "scripts/run_escs_x3_topology_adaptation.py",
            x3.REPO_ROOT / "docs/audits/escs_x3_topology_adaptation.md",
        ),
    )
    manifest_path = tmp_path / "x3-implementation-authority.json"
    x3._atomic_json(manifest_path, manifest)
    return config_path, manifest_path, manifest["manifest_sha256"]


def test_official_config_is_disabled_and_reports_every_current_gap() -> None:
    config = x3.load_config()
    assert x3.canonical_sha256(config) == x3.OFFICIAL_CONFIG_AUTHORITY_SHA256
    assert config["activation"] == {
        "enabled": False,
        "scientific_promotion_allowed": False,
    }
    assert tuple(config["arms"]) == x3.ARM_NAMES
    assert tuple(config["permutations"]) == x3.PERMUTATIONS
    assert config["prerequisites"]["p6_1m"]["path"] == (
        "proof/P6_CONTINUAL_1M_INDEPENDENT_VERIFICATION.json"
    )
    assert set(config["seeds"]).isdisjoint(config["fresh_verifier_seeds"])
    assert config["verdict"]["terminal_routes"] == [
        "positive",
        "null",
        "invalid_bed",
        "failed",
    ]
    readiness = x3.official_readiness()
    assert readiness["ready"] is False
    assert {
        "x3-official-activation-disabled",
        "g0-implementation-complete-false",
        "g0-not-frozen",
        "g0-activation-disabled",
        "g0-freeze-authority-absent",
        "p6-1m-positive-verification-absent",
        "x1-or-x2-positive-fresh-verification-absent",
    } <= set(readiness["problems"])
    assert "f63-candidate-or-f64-control-evidence-invalid" not in readiness["problems"]
    assert readiness["scientific_promotion"] is False


def test_stream_covers_drift_lesion_recovery_recurrence_and_future_learning() -> None:
    config = x3.load_config()
    stream = x3.generate_stream(config, seed=config["seeds"][0], split="producer")
    assert {event.phase for event in stream} == set(x3.STREAM_PHASES)
    assert any(event.phase == "abrupt_drift" and event.drift_strength == 1.0 for event in stream)
    gradual = [event.drift_strength for event in stream if event.phase == "gradual_drift"]
    assert gradual == sorted(gradual)
    assert gradual[0] < gradual[-1]
    assert any(event.lesion_active for event in stream)
    assert any(event.phase == "post_lesion_recovery" and event.heldout for event in stream)
    assert any(event.phase == "old_regime_return" and event.regime == 0 for event in stream)
    assert any(event.phase == "future_learning" and event.heldout for event in stream)


def test_g0_transaction_fixture_refuses_effect_and_replays_every_guard() -> None:
    fixture = x3.build_transactional_mechanics_fixture(20260712)
    assert fixture["all_checks_passed"]
    assert all(fixture["checks"].values())
    assert fixture["grammar_status"] == "scaffold"
    assert fixture["g0_implementation_complete"] is False
    assert fixture["assessment"]["structurally_valid"] is True
    assert fixture["assessment"]["shadow_authorized"] is False
    assert fixture["assessment"]["factual_commitment_authorized"] is False
    assert {
        "operator-disabled:add_factor_scope",
        "grammar-not-frozen",
        "grammar-activation-disabled",
    } <= set(fixture["assessment"]["blockers"])
    assert fixture["shadow_trace_sha256"]
    assert fixture["canary_trace_sha256"]
    assert fixture["commitment_event_id"].startswith("event:")
    assert fixture["consequence_event_id"].startswith("event:")
    assert fixture["rollback_snapshot_sha256"] == fixture["effective_topology_sha256"]
    assert fixture["proposed_topology_sha256"] != fixture["effective_topology_sha256"]
    assert fixture["factual_topology_effects"] is False
    assert fixture["scientific_promotion"] is False


def test_seed_has_all_controls_permutations_and_full_structural_accounting() -> None:
    config = x3.load_config()
    row = x3.run_seed(config, seed=config["seeds"][0], split="producer")
    assert row["difficulty_gate"]["passed"]
    assert all(row["invariants"].values())
    assert tuple(row["conditions"]) == x3.PERMUTATIONS
    assert row["neutral_initialization"]["role_labels_available"] is False
    for condition in row["conditions"].values():
        assert tuple(condition["arms"]) == x3.ARM_NAMES
        assert all(condition["controls"].values())
        primary = condition["arms"][x3.PRIMARY_ARM]
        same_genotype = condition["arms"]["same_final_genotype_from_start"]
        assert primary["final_genotype_sha256"] == same_genotype["final_genotype_sha256"]
        for arm in condition["arms"].values():
            assert set(arm["structural_accounting"]) == set(x3.STRUCTURAL_ACCOUNTING_COMPONENTS)
            assert set(arm["lifecycle_work"]) == set(x3.WorkVector.zero().payload())
            assert arm["structural_accounting"]["factual_mutations_committed"] == 0
            assert arm["factual_topology_effects"] is False
            assert arm["activation_enabled"] is False
            assert arm["scientific_promotion"] is False
    oracle = row["conditions"]["canonical"]["arms"][x3.ORACLE_ARM]
    assert oracle["evidence_standing"] == "oracle_nonpromotable"


def test_terminal_routing_distinguishes_positive_null_invalid_and_failed() -> None:
    config = x3.load_config()
    rows = [x3.run_seed(config, seed=seed, split="producer") for seed in config["seeds"]]
    null_result = x3.aggregate_rows(rows, config)
    assert null_result["terminal_route"] == "null"
    assert null_result["scientific_promotion"] is False

    invalid = copy.deepcopy(rows)
    invalid[0]["difficulty_gate"]["passed"] = False
    _reseal_row(invalid[0])
    assert x3.aggregate_rows(invalid, config)["terminal_route"] == "invalid_bed"

    failed = copy.deepcopy(rows)
    failed[0]["invariants"]["all_controls_complete"] = False
    _reseal_row(failed[0])
    assert x3.aggregate_rows(failed, config)["terminal_route"] == "failed"

    unsealed = copy.deepcopy(rows)
    unsealed[0]["conditions"]["canonical"]["arms"][x3.PRIMARY_ARM]["online_utility_area"] = 1.0
    assert x3.aggregate_rows(unsealed, config)["terminal_route"] == "failed"

    favorable = copy.deepcopy(rows)
    for row in favorable:
        for condition in row["conditions"].values():
            primary = condition["arms"][x3.PRIMARY_ARM]
            primary["online_utility_area"] = 1.0
            primary["post_lesion_recovery"] = 1.0
            primary["abstract_operation_work"] = 1
            primary["peak_state_bytes"] = 1
            primary["old_regime_regression"] = 0.0
            primary["future_learnability"] = 1.0
            for control_name in x3.REQUIRED_COMPARATORS:
                control = condition["arms"][control_name]
                control["online_utility_area"] = min(float(control["online_utility_area"]), 0.8)
                control["post_lesion_recovery"] = min(float(control["post_lesion_recovery"]), 0.8)
                control["abstract_operation_work"] = max(int(control["abstract_operation_work"]), 2)
                control["peak_state_bytes"] = max(int(control["peak_state_bytes"]), 2)
        _reseal_row(row)
    positive = x3.aggregate_rows(favorable, config)
    assert positive["terminal_route"] == "positive"
    assert positive["scientific_promotion"] is False


def test_exploratory_producer_and_fresh_verifier_resume_with_sealed_receipts(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, manifest_sha = _exploratory_bundle(tmp_path)
    producer = tmp_path / "producer.json"
    producer_checkpoint = tmp_path / "producer.checkpoint.json"
    partial = x3.run_from_config(
        config_path,
        producer,
        producer_checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=2,
        exploratory=True,
    )
    assert partial["execution_status"] == "partial"
    assert partial["resumable"]
    complete = x3.run_from_config(
        config_path,
        producer,
        producer_checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        exploratory=True,
    )
    assert complete["execution_status"] == "complete"
    assert complete["fresh_verifier_status"] == "required"
    assert complete["activation_enabled"] is False
    assert complete["factual_topology_effects"] is False

    verification = tmp_path / "verification.json"
    verification_checkpoint = tmp_path / "verification.checkpoint.json"
    partial_verification = x3.verify_receipt(
        producer,
        config_path,
        manifest_path,
        verification,
        verification_checkpoint,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=2,
        exploratory=True,
    )
    assert partial_verification["verification_status"] == "partial"
    final = x3.verify_receipt(
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
    assert final["terminal_route"] in {"positive", "null", "invalid_bed", "failed"}
    assert final["activation_enabled"] is False
    assert final["factual_topology_effects"] is False
    assert final["scientific_promotion"] is False


def test_tampering_and_official_execution_fail_closed(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _exploratory_bundle(tmp_path)
    producer = tmp_path / "producer.json"
    checkpoint = tmp_path / "producer.checkpoint.json"
    x3.run_from_config(
        config_path,
        producer,
        checkpoint,
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    forged = json.loads(checkpoint.read_text(encoding="utf-8"))
    forged["completed_seeds"] = []
    x3._atomic_json(checkpoint, forged)
    with pytest.raises(ValueError, match="checkpoint self-hash"):
        x3.run_from_config(
            config_path,
            producer,
            checkpoint,
            manifest_path,
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
        )

    with pytest.raises(x3.OfficialExecutionRefused) as refusal:
        x3.run_from_config(
            x3.DEFAULT_CONFIG_PATH,
            tmp_path / "official.json",
            tmp_path / "official.checkpoint.json",
            x3.DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
        )
    message = str(refusal.value)
    assert "g0-implementation-complete-false" in message
    assert "p6-1m-positive-verification-absent" in message
    assert "x1-or-x2-positive-fresh-verification-absent" in message
    assert "x3-official-activation-disabled" in message


def test_partial_producer_cannot_be_verified(tmp_path: Path) -> None:
    config_path, manifest_path, manifest_sha = _exploratory_bundle(tmp_path)
    producer = tmp_path / "partial.json"
    x3.run_from_config(
        config_path,
        producer,
        tmp_path / "producer.checkpoint.json",
        manifest_path,
        implementation_authority_sha256=manifest_sha,
        max_new_seeds=1,
        exploratory=True,
    )
    with pytest.raises(ValueError, match="partial X3 producer"):
        x3.verify_receipt(
            producer,
            config_path,
            manifest_path,
            tmp_path / "verification.json",
            tmp_path / "verification.checkpoint.json",
            implementation_authority_sha256=manifest_sha,
            exploratory=True,
        )
