from __future__ import annotations

import json

import yaml

from mop.studies.local_action_environment import SCHEMA, build_preflight, write_preflight


def _profile(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "mop-local-action-environment-profile/v1",
                "scope": "test fixture",
                "grid_size": 6,
                "horizon": 8,
                "episodes": 12,
                "policy": "mixed",
                "seeds": [501, 502],
                "train_episode_fraction": 0.75,
                "ridge_l2": 0.001,
                "required_action_count": 4,
                "required_observation_dim": 16,
                "required_counterfactual_branches": 4,
                "scientific_promotion_allowed": False,
            }
        )
    )
    return path


def test_all_four_lanes_execute_but_science_stays_fail_closed(tmp_path) -> None:
    result = build_preflight(config_path=_profile(tmp_path), run_dir=tmp_path / "runs")
    assert result["schema"] == SCHEMA
    assert result["all_mechanics_verified"] is True
    assert result["scientific_promotion_allowed"] is False
    assert set(result["demoted_blockers"]) == {
        "f6_sensorimotor_form_closure",
        "f15_embodied_affordance_form",
        "e5_curiosity",
        "mop_cm10_action_forward_model",
    }
    for seed in result["seeds"]:
        assert seed["exact_replay_verified"] is True
        assert set(seed["lanes"]) == {"f6", "f15", "cm10", "e5"}
        assert all(row["mechanics_verified"] for row in seed["lanes"].values())
        cm10 = seed["lanes"]["cm10"]
        assert cm10["scientific_ready"] is False
        assert cm10["matched_dynamics_parameters"] is True
        assert cm10["matched_planner_inference_calls"] is True
        assert "frozen-vjepa2-ac-on-the-exact-rendered-referents" in cm10["controls_not_yet_citable"]


def test_receipts_and_per_seed_trajectories_are_durable_json(tmp_path) -> None:
    profile = _profile(tmp_path)
    run_dir = tmp_path / "runs"
    run_receipt = run_dir / "preflight.json"
    proof = tmp_path / "proof.json"
    result = write_preflight(
        config_path=profile,
        run_dir=run_dir,
        run_receipt=run_receipt,
        proof_path=proof,
    )
    assert json.loads(run_receipt.read_text())["all_mechanics_verified"] is True
    assert json.loads(proof.read_text())["scientific_promotion_allowed"] is False
    for row in result["seeds"]:
        artifact = row["artifact"]
        assert artifact["exists"] is True
        assert len(artifact["sha256"]) == 64
