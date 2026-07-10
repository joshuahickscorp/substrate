"""Focused tests for the executed f36 and f37 toy beds and independent verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from mop.studies.integration_broadcast_runs import (
    DEFAULT_CONFIG,
    EXPERIMENT_IDS,
    assert_receipt,
    build_receipt,
    load_config,
)
from mop.studies.integration_broadcast_verify import F36, F37, build_verification
from mop.substrate.events import canonical_sha256


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def test_broadcast_run_is_deterministic_calibrated_and_compute_matched():
    first = build_receipt()
    second = build_receipt()
    assert first["payload_sha256"] == second["payload_sha256"]
    assert_receipt(first)
    assert tuple(first["aggregate"]) == EXPERIMENT_IDS
    assert set(first["null_contract"]["per_experiment"]) == set(EXPERIMENT_IDS)
    assert first["null_contract"]["per_experiment_sha256"] == canonical_sha256(
        first["null_contract"]["per_experiment"]
    )
    for unit in first["independent_units"]:
        result = unit["result"]
        assert result["exact_replay"] is True
        assert result["compute_match"]["all_arms_exact_budget"] is True
        assert result["necessity"]["difficulty_calibration"]["off_floor_and_ceiling"] is True
        assert result["sufficiency"]["difficulty_calibration"]["off_floor_and_ceiling"] is True


@pytest.mark.parametrize("null_value", [None, "   "])
def test_broadcast_config_requires_nonblank_top_level_null(tmp_path: Path, null_value) -> None:
    payload = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    if null_value is None:
        changed.pop("null_hypothesis")
    else:
        changed["null_hypothesis"] = null_value
    path = tmp_path / "broadcast.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="null hypothesis"):
        load_config(path)


def test_f36_is_favorable_but_f37_exact_tie_is_null():
    receipt = build_receipt()
    assert receipt["aggregate"][F36]["programmatic_favorable"] is True
    assert receipt["aggregate"][F37]["programmatic_favorable"] is False
    assert receipt["aggregate"][F37]["all_units_tie_is_null"] is True
    for unit in receipt["independent_units"]:
        assert unit["result"]["sufficiency"]["delta_vs_best_comparator"] == 0.0
        assert unit["result"]["sufficiency"]["tie_is_null"] is True


def test_broadcast_favorable_gets_fresh_adversarial_verification(tmp_path):
    run_path = tmp_path / "run.json"
    _write_json(run_path, build_receipt())
    verification = build_verification(run_path=run_path)
    assert verification["all_ok"] is True
    assert verification["primary_recompute_exact"] is True
    assert verification["per_experiment"][F36]["programmatic_pattern_verified"] is True
    assert verification["per_experiment"][F36]["scientific_promotion_allowed"] is False
    assert verification["per_experiment"][F37]["verdict"] == "verified-null"
    assert all(verification["mutation_checks"].values())


def test_broadcast_verifier_rejects_erased_tie_null(tmp_path):
    mutated = build_receipt()
    mutated["independent_units"][0]["result"]["sufficiency"]["tie_is_null"] = False
    mutated["payload_sha256"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "payload_sha256"}
    )
    run_path = tmp_path / "mutated.json"
    _write_json(run_path, mutated)
    verification = build_verification(run_path=run_path)
    assert verification["all_ok"] is False
    assert any("tie_is_null" in problem for problem in verification["problems"])


def test_broadcast_verifier_rejects_digest_valid_null_contract_mutation(tmp_path):
    mutated = build_receipt()
    mutated["null_contract"]["aggregate"] += "."
    mutated["payload_sha256"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "payload_sha256"}
    )
    run_path = tmp_path / "mutated-null.json"
    _write_json(run_path, mutated)
    with pytest.raises(ValueError, match="null contract"):
        build_verification(run_path=run_path)
