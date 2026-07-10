from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import mop.studies.frontier_localization as localization
from mop.config import REPO_ROOT
from mop.studies.frontier_localization import (
    AUDIT_SCHEMA,
    HISTORICAL_FRONTIER_IDS,
    PREFLIGHT_SCHEMA,
    PREFLIGHTS,
    ScientificLaunchBlocked,
    assert_scientific_launch_ready,
    build_frontier_audit,
    run_local_preflights,
    scientific_readiness,
    write_frontier_receipts,
)


def test_all_owned_frontier_mechanics_execute_and_science_fails_closed():
    receipt = run_local_preflights(seeds=(101,))
    assert receipt["schema"] == PREFLIGHT_SCHEMA
    assert set(receipt["results"]) == set(PREFLIGHTS)
    assert receipt["all_mechanics_verified"] is True
    assert receipt["all_scientific_paths_fail_closed"] is True
    for row in receipt["results"].values():
        assert row["mechanics_verified"] is True
        assert row["scientific_readiness"]["ready"] is False
        assert row["scientific_readiness"]["missing"]


def test_scientific_gate_does_not_accept_descriptions_or_implicit_nearby_files():
    result = scientific_readiness("mop_at2_mode_substrate_dep")
    assert result["ready"] is False
    assert "matched_random_cache" in result["missing"]
    with pytest.raises(ScientificLaunchBlocked, match="matched_random_cache"):
        assert_scientific_launch_ready("mop_at2_mode_substrate_dep")


def test_scientific_gate_requires_repository_local_receipts(tmp_path: Path):
    outside = tmp_path / "claim.json"
    outside.write_text(json.dumps({"claim": True}))
    supplied = {
        "compatible_primitives": str(outside),
        "distinct_strategy_gate": str(outside),
        "independent_verifier": str(outside),
    }
    result = scientific_readiness("mop_mt4_reasoning_router", supplied)
    assert result["ready"] is False
    assert all(row["receipts"][0].get("rejected") for row in result["checks"])


def test_scientific_gate_rejects_unbound_repository_json():
    supplied = {
        "compatible_primitives": "proof/REAL_ENCODER_LOCAL_ATTEMPT.json",
        "distinct_strategy_gate": "proof/REAL_ENCODER_LOCAL_ATTEMPT.json",
        "independent_verifier": "proof/REAL_ENCODER_LOCAL_ATTEMPT.json",
    }
    result = scientific_readiness("mop_mt4_reasoning_router", supplied)
    assert result["ready"] is False
    assert all(row["receipts"][0]["semantic_contract_ok"] is False for row in result["checks"])


def test_scientific_gate_can_clear_only_exact_cross_bound_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(localization, "REPO_ROOT", tmp_path)
    evidence = tmp_path / "source.json"
    evidence.write_text('{"measured": true}\n')
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    supplied = {}
    for requirement in localization.SCIENTIFIC_REQUIREMENTS["mop_mt4_reasoning_router"]:
        path = tmp_path / f"{requirement.key}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": localization.PREREQUISITE_SCHEMA,
                    "experiment_id": "mop_mt4_reasoning_router",
                    "requirement": requirement.key,
                    "verified": True,
                    "claim_scope": "unit contract fixture",
                    "verifier": {"name": "independent-test-verifier", "independent": True},
                    "evidence": [{"path": "source.json", "sha256": digest}],
                }
            )
        )
        supplied[requirement.key] = path.name
    result = localization.scientific_readiness("mop_mt4_reasoning_router", supplied)
    assert result["ready"] is True
    assert result["missing"] == []


def test_frontier_audit_accounts_for_every_historical_tag_without_hardware_inference():
    preflight = run_local_preflights(seeds=(2026,))
    audit = build_frontier_audit(preflight)
    assert audit["schema"] == AUDIT_SCHEMA
    assert audit["coverage"]["accounted_exactly_once"] is True
    assert audit["coverage"]["tagged_non_f_count"] >= len(HISTORICAL_FRONTIER_IDS)
    assert audit["coverage"]["measured_hardware_blocked_count"] == 0
    assert audit["coverage"]["current_stale_planning_tag_count"] == 0
    assert audit["coverage"]["all_historical_rows_reclassified"] is True
    by_id = {row["id"]: row for row in audit["entries"]}
    assert set(PREFLIGHTS) <= set(by_id)
    assert set(HISTORICAL_FRONTIER_IDS) <= set(by_id)
    assert all(by_id[experiment_id]["preflight_mechanics_verified"] for experiment_id in PREFLIGHTS)
    assert not any(row["hardware_boundary_proven"] for row in audit["entries"])
    assert all(row["reclassified_away_from_hardware_planning_tags"] for row in audit["entries"])
    assert audit["measured_real_encoder"]["receipt"]["path"] == ("proof/REAL_ENCODER_LOCAL_ATTEMPT.json")
    assert by_id["mop_cm7_min_objective_probe"]["localization"] == (
        "local-custom-training-five-seed-null-bound"
    )
    assert "four additional" not in by_id["mop_cm7_min_objective_probe"]["first_remaining_scientific_blocker"]
    assert by_id["mop_cm8_custom_jepa_pilot"]["localization"] == (
        "local-custom-preflight-proven-upstream-blocked"
    )
    assert (
        "four complete seeds" not in by_id["mop_cm8_custom_jepa_pilot"]["first_remaining_scientific_blocker"]
    )
    assert audit["measured_cm7_bound_null"]["valid"] is True
    assert audit["measured_cm7_bound_null"]["seed_count"] == 5
    assert audit["measured_cm7_bound_null"]["verdict"] == "not-promoted"
    runtime = audit["measured_vjepa21_vitb_runtime"]
    assert runtime["runtime_availability_blocker_retired"] is True
    assert runtime["e6_scientific_compatibility_proven"] is False
    assert runtime["forward_frames"] == [8, 64]
    assert runtime["task_cache_control_integration_verified"] is True
    assert runtime["dense_task_preflight"]["heavy_model_work_executed"] is False
    assert runtime["e6_remaining_classification"] == "rights-data-blocked"
    assert "natural" in runtime["e6_remaining_scientific_blocker"]
    assert by_id["mop_dr14_corruption"]["preflight_mechanics_verified"] is True
    assert by_id["mop_dr14_corruption"]["localization"] == (
        "local-dense-task-integration-proven-data-blocked"
    )
    assert by_id["mop_al2_shared_latent_alignment"]["localization"] == "external-input-blocked"
    assert by_id["mop_dr5_cross_substrate_consistency"]["localization"] == "external-input-blocked"
    assert "scale_atlas_receipt" not in by_id["mop_al2_shared_latent_alignment"]


def test_frontier_audit_rejects_partial_or_promoted_fixture_receipts():
    with pytest.raises(ValueError, match="complete, verified, fail-closed"):
        build_frontier_audit({"schema": PREFLIGHT_SCHEMA, "results": {}})


def test_e6_dense_preflight_rejects_promoted_or_heavy_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = json.loads(localization.E6_DENSE_PREFLIGHT_PATH.read_text())
    source["scientific_promotion"] = True
    source["forward_executed"] = True
    mutated = tmp_path / "mutated-e6.json"
    mutated.write_text(json.dumps(source))
    monkeypatch.setattr(
        localization,
        "_file_receipt",
        lambda path: {"path": str(path), "exists": path.is_file()},
    )
    result = localization._validate_e6_dense_preflight(mutated)
    assert result["valid"] is False
    assert result["task_cache_control_integration_verified"] is False
    assert result["heavy_model_work_executed"] is True


def test_shipped_frontier_receipts_are_durable_and_parseable():
    for rel, schema in (
        ("proof/LOCAL_FRONTIER_PREFLIGHTS.json", PREFLIGHT_SCHEMA),
        ("proof/FRONTIER_LOCALIZATION.json", AUDIT_SCHEMA),
    ):
        path = REPO_ROOT / rel
        assert path.is_file()
        payload = json.loads(path.read_text())
        assert payload["schema"] == schema


def test_frontier_audit_can_reuse_preflight_without_invalidating_upstream_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    preflight = run_local_preflights(seeds=(101,))
    run_path = tmp_path / "local-preflights.json"
    proof_path = tmp_path / "proof-preflights.json"
    audit_path = tmp_path / "audit.json"
    run_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before = hashlib.sha256(run_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        localization,
        "run_local_preflights",
        lambda **_: pytest.fail("reuse mode must not rerun timing-bearing preflights"),
    )
    real_file_receipt = localization._file_receipt

    def file_receipt(path: Path):
        if path.is_relative_to(REPO_ROOT):
            return real_file_receipt(path)
        return {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(localization, "_file_receipt", file_receipt)

    reused, audit = write_frontier_receipts(
        run_receipt=run_path,
        preflight_proof=proof_path,
        audit_proof=audit_path,
        seeds=(101,),
        reuse_existing_preflight=True,
    )

    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == before
    assert reused == preflight
    assert json.loads(proof_path.read_text(encoding="utf-8")) == preflight
    assert audit["schema"] == AUDIT_SCHEMA
