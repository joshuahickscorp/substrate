from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

import mop.studies.action_world_model as action_world_model
from mop.studies.action_world_model import (
    ARM_ORDER,
    DEFAULT_CONFIG,
    EQUAL_CORE_COMPUTE_ARMS,
    GridState,
    _construction,
    _load_config,
    build_preflight,
    build_unit_dataset,
    mutation_suite,
    render_observation,
    verify_unit_dataset,
)
from mop.studies.runtime_integrity import ForbiddenRuntimeImport


def test_render_is_seed_state_exact_and_hidden_state_sensitive() -> None:
    config = _load_config(DEFAULT_CONFIG)
    construction = _construction(config, config["environment"]["independent_units"][0])
    dataset = build_unit_dataset(construction)
    world = dataset["independent_unit"]["world_payload"]
    from mop.environments.persistent_grid import WorldSpec

    spec = WorldSpec.from_payload(world)
    state = GridState(episode_index=0, step_index=1, row=0, col=0, goal_row=5, goal_col=4)
    changed = GridState(episode_index=0, step_index=1, row=0, col=1, goal_row=5, goal_col=4)
    first = render_observation(spec, state, cell_pixels=2)
    second = render_observation(spec, state, cell_pixels=2)
    different = render_observation(spec, changed, cell_pixels=2)
    assert first.dtype == torch.uint8 and tuple(first.shape) == (12, 12, 3)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_unit_dataset_has_exact_heldout_branches_and_rejects_mutations() -> None:
    config = _load_config(DEFAULT_CONFIG)
    dataset = build_unit_dataset(_construction(config, config["environment"]["independent_units"][0]))
    audit = verify_unit_dataset(dataset)
    assert audit["verified"] is True
    assert dataset["budget_contract"] == {
        "train_event_roots": 32,
        "heldout_event_roots": 16,
        "train_branch_actions": 128,
        "heldout_branch_actions": 64,
        "heldout_unchosen_interventions": 48,
        "rollout_roots": 8,
        "planning_episode_actions_maximum": 32,
        "render_pairs_materialized": 192,
        "render_receipts": 384,
    }
    for group in dataset["splits"]["heldout_interventions"]["groups"]:
        branches = group["branches"]
        assert {row["action"] for row in branches} == set(range(4))
        assert len({row["wave_e0_branch"]["parent_state"]["sha256"] for row in branches}) == 1
        assert sum(row["action_was_chosen"] for row in branches) == 1
    mutations = mutation_suite(dataset)
    assert mutations["all_rejected"] is True
    assert set(mutations["mutations"]) == {
        "render_digest",
        "hidden_state",
        "action_intervention",
        "shuffled_consequence",
        "environment_seed",
        "declared_budget",
    }


def test_full_preflight_executes_all_arms_with_exact_core_matching(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", object())
    receipt = build_preflight(DEFAULT_CONFIG)
    assert receipt["status"] == "mechanics-pass"
    assert receipt["all_mechanics_ok"] is True
    assert len(receipt["units"]) == 3
    assert receipt["claim_boundary"]["scientific_promotion_allowed"] is False
    assert receipt["resource_observation"]["model_weights_loaded"] is False
    assert receipt["resource_observation"]["model_downloads_performed"] is False
    assert receipt["checks"]["no_inherited_model_modules_loaded"] is True
    assert receipt["runtime_integrity"]["all_ok"] is True
    assert receipt["runtime_integrity"]["runtime_import_attempts"] == []
    assert receipt["resource_observation"]["rss_measurement"]["all_ok"] is True
    assert receipt["resource_observation"]["phase_local_peak_rss_increment_bytes"] <= 1024**3
    for unit in receipt["units"]:
        assert tuple(unit["arms"]) == ARM_ORDER
        assert unit["equal_core_compute"]["arms"] == list(EQUAL_CORE_COMPUTE_ARMS)
        assert unit["equal_core_compute"]["matched"] is True
        assert unit["arms"]["oracle_state"]["prediction"]["one_step"]["agent_cell_accuracy"] == 1.0
        assert unit["arms"]["oracle_state"]["prediction"]["one_step"]["calibration_brier"] == 0.0
        assert unit["arms"]["reactive_rendered"]["prediction"]["applicable"] is False
        assert unit["arms"]["matched_depth_reactive"]["prediction"]["applicable"] is False
        assert unit["wave_e0_f29_binding"]["shuffled_consequence_mutation_rejected"] is True
        assert unit["fixture_outcome"]["fixture_null_supported"] is True
        assert unit["fixture_outcome"]["scientific_verdict"] == "not-eligible"


def test_preflight_rejects_cached_forbidden_runtime_import(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", object())
    real_run_unit = action_world_model._run_unit

    def attempted_import(config, unit):
        __import__("transformers")
        return real_run_unit(config, unit)

    monkeypatch.setattr(action_world_model, "_run_unit", attempted_import)
    with pytest.raises(ForbiddenRuntimeImport, match="transformers"):
        build_preflight(DEFAULT_CONFIG)


def test_config_fails_closed_when_equal_compute_arm_is_removed(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    changed = copy.deepcopy(config)
    changed["evaluation"]["equal_core_compute_group"].pop()
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="equal-core-compute group drift"):
        _load_config(path)
