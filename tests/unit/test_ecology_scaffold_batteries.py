"""Focused tests for the executed ecology scaffold toy-world batteries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from scripts.run_ecology_scaffold_batteries import PRIMARY_SEEDS, build_receipt
from scripts.verify_ecology_scaffold_batteries import (
    FRESH_SEEDS,
    verify_payload_sha256,
    verify_receipt,
)

from mop.config import REPO_ROOT
from mop.devel.north_star import assert_no_sentience_claims
from mop.environments.ecology_battery import EXPERIMENT_IDS, run_ecology_battery_seed
from mop.substrate.events import canonical_sha256


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt()


def test_seed_unit_is_deterministic_and_complete() -> None:
    first = run_ecology_battery_seed(11)
    repeated = run_ecology_battery_seed(11)
    other = run_ecology_battery_seed(12)
    assert first == repeated
    assert first["unit_sha256"] != other["unit_sha256"]
    assert tuple(first["experiments"]) == EXPERIMENT_IDS
    assert first["fixture_sha256"] == canonical_sha256(first["fixture"])


def test_registry_controls_are_exercised_exactly(receipt: dict) -> None:
    registry = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in registry["experiments"] if row.get("id") in EXPERIMENT_IDS}
    for experiment_id, row in receipt["experiments"].items():
        assert row["registry_binding"]["controls"] == rows[experiment_id]["controls"]
        assert set(row["control_evidence"]) == set(rows[experiment_id]["controls"])
        assert all(value != "missing" for value in row["control_evidence"].values())
        assert row["all_declared_controls_exercised"] is True
        assert set(row["metric_evidence"]) == set(rows[experiment_id]["metrics"])
        assert all(value != "missing" for value in row["metric_evidence"].values())
        assert row["all_declared_metrics_reported"] is True


def test_every_bed_is_calibrated_and_promotion_is_refused(receipt: dict) -> None:
    for row in receipt["experiments"].values():
        calibration = row["difficulty_calibration"]
        assert calibration["calibrated"] is True
        assert calibration["ceilinged_tie"] is False
        assert row["tie_is_null"] is True
        assert row["promotion"] is False


def test_active_perception_covers_costed_actions_and_matched_budget(receipt: dict) -> None:
    unit = receipt["units"][0]["experiments"]["f22_active_form_acquisition"]
    assert unit["controls"] == [
        "random-acquisition",
        "uncertainty-acquisition",
        "saliency-acquisition",
        "full-observation",
    ]
    assert unit["mechanism_checks"]["matched_charge_budget"] is True
    assert unit["mechanism_checks"]["all_sensing_actions_costed"] is True
    assert unit["mechanism_checks"]["downstream_training_updates"] == 0


def test_f28_hindsight_control_forces_an_honest_null(receipt: dict) -> None:
    row = receipt["experiments"]["f28_sensor_value_forecast"]
    aggregate = row["aggregate_metrics"]
    assert aggregate["post-hoc-value"]["forecast_rank_correlation"] == 1.0
    assert aggregate["post-hoc-value"]["payoff_calibration_error"] == 0.0
    assert row["candidate_result"] == "null"
    assert receipt["verified_outcomes"][row["experiment_id"]] == "null"


def test_goldilocks_safe_play_and_quality_diverse_controls(receipt: dict) -> None:
    unit = receipt["units"][0]["experiments"]
    f50 = unit["f50_curriculum_goldilocks_test"]
    assert f50["arms"]["goldilocks-learning-progress"]["noisy_tv_captures"] == 0
    assert f50["arms"]["goldilocks-learning-progress"]["metrics"]["band_occupancy_rate"] == 1.0
    f51 = unit["f51_safe_play_goal_babbling"]
    assert f51["arms"]["guarded-goal-babbling"]["metrics"]["guard_violation_rate"] == 0.0
    assert f51["arms"]["stop-rule-audit"]["metrics"]["audit_accuracy"] == 1.0
    assert f51["mechanism_checks"]["all_stop_rules_exercised"] is True
    assert set(f51["reported_metrics"]["refusal_rate_by_rule"]) == {
        "noisy-tv",
        "reward-hacking",
        "unsafe-goal",
        "archive-bloat",
    }
    f52 = unit["f52_quality_diverse_mode_ecology"]
    assert f52["mechanism_checks"]["matched_compute"] is True
    assert f52["mechanism_checks"]["stop_reason"] is None


def test_all_partner_fixtures_and_controls_execute(receipt: dict) -> None:
    unit = receipt["units"][0]["experiments"]
    for experiment_id in (
        "f53_joint_referent_establishment",
        "f54_communicative_repair",
        "f55_selective_imitation",
        "f56_teaching_value",
        "f58_cultural_accumulation",
    ):
        row = unit[experiment_id]
        assert row["primary_arm"] in row["arms"]
        assert set(row["controls"]).issubset(set(row["arms"]))
    assert unit["f53_joint_referent_establishment"]["mechanism_checks"]["private_observation_leak"] is False
    assert unit["f54_communicative_repair"]["mechanism_checks"]["matched_extra_bits"] is True
    assert unit["f55_selective_imitation"]["mechanism_checks"]["decorative_steps_present"] is True
    assert unit["f58_cultural_accumulation"]["mechanism_checks"]["external_tasks_used"] is True


def test_teaching_tie_and_direct_state_win_stay_null(receipt: dict) -> None:
    teaching = receipt["experiments"]["f56_teaching_value"]
    gain = teaching["aggregate_metrics"]
    assert gain["learner-progress-teacher"] == gain["uncertainty-selection"]
    assert teaching["candidate_result"] == "null"
    grounding = receipt["experiments"]["f57_emergent_symbol_grounding"]
    metrics = grounding["aggregate_metrics"]
    for metric in ("compositional_transfer", "causal_grounding_score", "code_stability"):
        assert metrics["direct-state"][metric] > metrics["consequence-bound-code"][metric]
    assert grounding["candidate_result"] == "null"


def test_favorable_candidates_have_fresh_disjoint_verification(receipt: dict) -> None:
    verifier = receipt["independent_verifier"]
    assert set(PRIMARY_SEEDS).isdisjoint(FRESH_SEEDS)
    assert verifier["fresh_seeds_disjoint_from_primary"] is True
    assert verifier["favorable_candidates_all_fresh_verified"] is True
    assert verifier["verified"] is True
    assert verifier["errors"] == []
    assert all(verifier["fresh_outcomes_stable_by_experiment"].values())


def test_independent_mutation_battery_rejects_every_attack(receipt: dict) -> None:
    verifier = receipt["independent_verifier"]
    assert verifier["all_mutations_rejected"] is True
    assert [row["id"] for row in verifier["mutation_tests"]] == [
        "raw-metric",
        "control-drop",
        "null-flip",
        "calibration-flip",
        "seed-overlap",
        "source-drift",
    ]
    assert all(row["rejected"] for row in verifier["mutation_tests"])


def test_verifier_fails_closed_on_receipt_mutation(receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["experiments"]["f56_teaching_value"]["candidate_result"] = "favorable-candidate"
    report = verify_receipt(mutated, check_live_files=False, run_mutations=False)
    assert report["verified"] is False
    assert any(
        "candidate result drift" in error or "payload digest drift" in error for error in report["errors"]
    )


def test_receipt_payload_is_bound_and_sentience_rail_clean(receipt: dict, tmp_path: Path) -> None:
    assert verify_payload_sha256(receipt)
    text = json.dumps(receipt, sort_keys=True)
    assert_no_sentience_claims(text, where="ecology scaffold run receipt")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert path.is_file()
