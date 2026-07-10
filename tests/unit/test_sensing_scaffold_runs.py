"""Focused tests for the executed f21, f26, and f27 toy beds and verifier."""

from __future__ import annotations

import json

from mop.studies.sensing_scaffold_runs import (
    EXPERIMENT_IDS,
    assert_receipt,
    build_receipt,
)
from mop.studies.sensing_scaffold_verify import build_verification
from mop.substrate.events import canonical_sha256


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def test_sensing_run_is_deterministic_complete_and_calibrated():
    first = build_receipt()
    second = build_receipt()
    assert first["payload_sha256"] == second["payload_sha256"]
    assert_receipt(first)
    assert set(first["aggregate"]) == set(EXPERIMENT_IDS)
    assert len(first["independent_units"]) == 5
    for unit in first["independent_units"]:
        for experiment_id in EXPERIMENT_IDS:
            row = unit["experiments"][experiment_id]
            assert row["exact_replay"] is True
            assert row["difficulty_calibration"]["off_floor_and_ceiling"] is True


def test_sensing_declared_controls_all_execute():
    receipt = build_receipt()
    assert receipt["registry_bindings"][EXPERIMENT_IDS[0]]["controls"] == [
        "fixed-window-baseline",
        "shuffled-timing",
        "wrong-time",
        "exact-replay",
    ]
    assert receipt["registry_bindings"][EXPERIMENT_IDS[1]]["controls"] == [
        "majority-vote",
        "mean-fusion",
        "raw-residual",
        "exact-replay",
    ]
    assert receipt["registry_bindings"][EXPERIMENT_IDS[2]]["controls"] == [
        "temporal-correlation",
        "wrong-event",
        "synchronous-unrelated",
        "exact-replay",
    ]
    for unit in receipt["independent_units"]:
        f21 = unit["experiments"][EXPERIMENT_IDS[0]]
        f26 = unit["experiments"][EXPERIMENT_IDS[1]]
        f27 = unit["experiments"][EXPERIMENT_IDS[2]]
        assert f21["shuffled_time_rejection_rate"] >= 0
        assert f21["wrong_time_rejection_rate"] >= 0
        assert set(f26["baseline_auroc"]) == {"majority-vote", "mean-fusion", "raw-residual"}
        assert f27["wrong_event_rejection_rate"] >= 0
        assert f27["synchronous_unrelated_rejection_rate"] >= 0


def test_sensing_independent_verifier_reexecutes_primary_and_fresh_seeds(tmp_path):
    run_path = tmp_path / "run.json"
    _write_json(run_path, build_receipt())
    verification = build_verification(run_path=run_path)
    assert verification["all_ok"] is True
    assert verification["primary_recompute_exact"] is True
    assert len(verification["fresh_units"]) == 5
    assert all(verification["mutation_checks"].values())
    for experiment_id in EXPERIMENT_IDS:
        primary = verification["per_experiment"][experiment_id]["primary_programmatic_favorable"]
        verified = verification["per_experiment"][experiment_id]["programmatic_pattern_verified"]
        assert verified is False or primary is True
        assert verification["per_experiment"][experiment_id]["scientific_promotion_allowed"] is False


def test_sensing_verifier_rejects_digest_valid_metric_mutation(tmp_path):
    mutated = build_receipt()
    mutated["independent_units"][0]["experiments"][EXPERIMENT_IDS[0]]["delta_vs_fixed_window"] += 0.01
    mutated["payload_sha256"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "payload_sha256"}
    )
    run_path = tmp_path / "mutated.json"
    _write_json(run_path, mutated)
    verification = build_verification(run_path=run_path)
    assert verification["all_ok"] is False
    assert any("delta_vs_fixed_window" in problem for problem in verification["problems"])
