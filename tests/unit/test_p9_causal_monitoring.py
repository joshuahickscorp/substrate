from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

import mop.studies.p9_causal_monitoring as p9_causal_monitoring
from mop.studies.p9_causal_monitoring import (
    ARM_ORDER,
    DEFAULT_CONFIG,
    HISTOGRAM_ARMS,
    _evaluate_unit,
    _load_config,
    build_dataset,
    build_dataset_resumable,
    build_preflight,
    mutation_suite,
    verify_dataset,
)
from mop.studies.runtime_integrity import ForbiddenRuntimeImport
from mop.substrate.events import canonical_sha256


def test_dataset_has_disjoint_splits_and_complete_same_parent_interventions() -> None:
    config = _load_config(DEFAULT_CONFIG)
    dataset = build_dataset(config)
    assert dataset["budget_contract"] == {
        "independent_units": 5,
        "chunks": 15,
        "lineages_per_unit": 52,
        "branches_per_lineage": 5,
        "total_lineages": 260,
        "total_branches": 1300,
        "split_lineages_per_unit": {"train": 24, "calibration": 12, "heldout": 16},
    }
    assert verify_dataset(dataset)["verified"] is True
    unit = dataset["units"][0]
    split_lineages = {
        split: {group["lineage_id"] for group in unit["splits"][split]["groups"]}
        for split in ("train", "calibration", "heldout")
    }
    assert split_lineages["train"].isdisjoint(split_lineages["calibration"])
    assert split_lineages["train"].isdisjoint(split_lineages["heldout"])
    for group in unit["splits"]["heldout"]["groups"]:
        assert len({row["parent_state_ref"] for row in group["branches"]}) == 1
        assert {row["intervention"]["name"] for row in group["branches"]} == {
            "observational",
            "queue_pressure",
            "memory_pressure",
            "retry_pressure",
            "resource_relief",
        }
        assert {row["telemetry"]["proxy_regime"] for row in group["branches"]} == {"reversed"}


def test_dataset_verifier_rejects_every_registered_mutation() -> None:
    dataset = build_dataset(_load_config(DEFAULT_CONFIG))
    mutations = mutation_suite(dataset)
    assert mutations["count"] == 8
    assert mutations["rejected"] == 8
    assert mutations["all_rejected"] is True


def test_interrupted_resume_is_exact_and_corruption_fails_closed(tmp_path: Path) -> None:
    config = _load_config(DEFAULT_CONFIG)
    clean = build_dataset(config)
    checkpoint = tmp_path / "resume.json"
    assert build_dataset_resumable(config, checkpoint, stop_after_chunks=4) is None
    resumed = build_dataset_resumable(config, checkpoint)
    assert resumed is not None
    assert resumed["payload_sha256"] == clean["payload_sha256"]

    corrupt = tmp_path / "corrupt.json"
    payload = json.loads(checkpoint.read_text())
    payload["completed_chunks"][0]["payload"]["groups"][0]["branches"][0]["telemetry"]["queue_depth"] = 99
    corrupt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="chunk digest mismatch"):
        build_dataset_resumable(config, corrupt)


def test_unit_executes_causal_correlational_negative_and_structural_controls() -> None:
    config = _load_config(DEFAULT_CONFIG)
    dataset = build_dataset(config)
    result = _evaluate_unit(dataset["units"][0], config)
    assert tuple(result["arms"]) == ARM_ORDER
    assert result["matched_histogram_capacity"]["arms"] == list(HISTOGRAM_ARMS)
    assert result["matched_histogram_capacity"]["matched"] is True
    assert result["checks"]["causal_monitor_responds_to_some_interventions"] is True
    assert result["checks"]["correlational_proxy_is_branch_invariant"] is True
    assert result["checks"]["oracle_detects_fixture_signal"] is True
    assert result["all_mechanics_ok"] is True
    assert result["scientific_promotion_allowed"] is False


def test_preflight_core_is_exact_and_resource_claims_remain_blocked(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", object())
    first = build_preflight(DEFAULT_CONFIG)
    second = build_preflight(DEFAULT_CONFIG)
    assert first["status"] == "mechanics-pass"
    assert first["deterministic_core_sha256"] == second["deterministic_core_sha256"]
    core = {
        key: value
        for key, value in first.items()
        if key not in {"resource_observation", "deterministic_core_sha256"}
    }
    assert canonical_sha256(core) == first["deterministic_core_sha256"]
    assert first["claim_boundary"]["scientific_promotion_allowed"] is False
    assert first["claim_boundary"]["energy_measured"] is False
    assert first["resource_observation"]["model_weights_loaded"] is False
    assert first["resource_observation"]["external_data_loaded"] is False
    assert first["checks"]["no_model_or_download_modules"] is True
    assert first["runtime_integrity"]["all_ok"] is True
    assert first["runtime_integrity"]["runtime_import_attempts"] == []
    assert first["resource_observation"]["rss_measurement"]["all_ok"] is True
    assert first["resource_observation"]["phase_local_peak_rss_increment_bytes"] <= 1024**3


def test_preflight_rejects_cached_importlib_model_import(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", object())
    real_build_dataset = p9_causal_monitoring.build_dataset

    def attempted_import(config):
        importlib.import_module("huggingface_hub")
        return real_build_dataset(config)

    monkeypatch.setattr(p9_causal_monitoring, "build_dataset", attempted_import)
    with pytest.raises(ForbiddenRuntimeImport, match="huggingface_hub"):
        build_preflight(DEFAULT_CONFIG)


def test_config_fails_closed_on_arm_or_claim_scope_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(DEFAULT_CONFIG.read_text())
    changed = copy.deepcopy(payload)
    changed["evaluation"]["histogram_arms"].pop()
    path = tmp_path / "bad-arms.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="arm set drift"):
        _load_config(path)

    changed = copy.deepcopy(payload)
    changed["claim_scope"] = "scientific claim"
    path = tmp_path / "bad-claim.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="claim scope drift"):
        _load_config(path)
