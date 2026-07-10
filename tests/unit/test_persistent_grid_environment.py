from __future__ import annotations

import copy

import pytest
import torch

from mop.environments import (
    PersistentGridEnvironment,
    bounded_trajectory_contract,
    collect_trajectory_bundle,
    make_world_spec,
    trajectory_tensors,
    verify_trajectory_bundle,
)


def test_trajectory_is_seed_exact_and_counterfactuals_share_clone() -> None:
    spec = make_world_spec(seed=71, horizon=8)
    first = collect_trajectory_bundle(spec, episodes=6)
    second = collect_trajectory_bundle(spec, episodes=6)
    assert first == second
    assert first["content_sha256"] == second["content_sha256"]
    assert verify_trajectory_bundle(first)["verified"] is True

    groups: dict[str, list[dict]] = {}
    for row in first["counterfactual_transitions"]:
        groups.setdefault(row["counterfactual_group_ref"], []).append(row)
    assert len(groups) == first["transition_count"]
    for rows in groups.values():
        assert {row["action"] for row in rows} == {0, 1, 2, 3}
        assert len({row["cloned_from_state_ref"] for row in rows}) == 1
        assert sum(row["action_was_chosen"] for row in rows) == 1


def test_replay_verifier_fails_closed_on_causal_mutation() -> None:
    bundle = collect_trajectory_bundle(make_world_spec(seed=72, horizon=7), episodes=4)
    broken = copy.deepcopy(bundle)
    broken["actual_transitions"][0]["consequence"]["action_cost"] += 0.25
    result = verify_trajectory_bundle(broken)
    assert result["verified"] is False
    assert result["checks"]["content_hash"] is False or result["checks"]["exact_seeded_replay"] is False


def test_step_records_boundaries_costs_entities_and_stable_refs() -> None:
    env = PersistentGridEnvironment(make_world_spec(seed=73, horizon=5))
    state = env.reset(0)
    actual, alternatives = env.step_with_counterfactuals(0)
    assert actual["episode_start"] is True
    assert actual["state_before"] == state.payload()
    assert actual["action_name"] == "north"
    assert actual["consequence"]["action_cost"] >= 1.0
    assert actual["entity_refs"]
    assert len(actual["paired_counterfactual_refs"]) == 4
    assert len(alternatives) == 4
    assert actual["actual_branch_ref"] == alternatives[0]["counterfactual_branch_ref"]


def test_numeric_adapter_preserves_referents_and_noisy_tv_control() -> None:
    bundle = collect_trajectory_bundle(make_world_spec(seed=74, horizon=10), episodes=8)
    batch = trajectory_tensors(bundle)
    assert batch["observation"].shape[1] == 16
    assert batch["observation"].shape == batch["next_observation"].shape
    assert batch["action_one_hot"].shape[1] == 4
    assert torch.equal(batch["action_one_hot"].argmax(1), batch["action"])
    assert batch["is_noisy_tv"].any()
    assert (~batch["is_noisy_tv"]).any()
    assert len(set(batch["event_refs"])) == len(batch["event_refs"])


def test_invalid_world_and_contract_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least five"):
        make_world_spec(seed=0, grid_size=4)
    contract = bounded_trajectory_contract(seed=75)
    assert contract["verified"] is True
    assert contract["programmatic_only"] is True
    assert contract["natural_embodiment_claim"] is False
