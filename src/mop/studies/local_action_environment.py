"""Bounded local action-environment and trajectory-lane preflights.

This study exercises one shared deterministic adapter against four consumers:

* F6: observation/action/next-state closure and action-blind/shuffled seams;
* F15: affordances and paired alternative consequences from cloned states;
* CM10: a small action-forward-model and true-dynamics planning mechanics pilot;
* E5: learnable versus noisy-TV trajectory regions.

Every numerical result is a programmatic pilot.  The proof refuses scientific promotion and names
the still-external evidence required for natural embodiment, V-JEPA action planning, and sustained
open-endedness.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import REPO_ROOT
from ..environments import (
    PersistentGridEnvironment,
    collect_trajectory_bundle,
    make_world_spec,
    trajectory_tensors,
    verify_trajectory_bundle,
    write_trajectory_bundle,
)

SCHEMA = "mop-local-action-environment-preflight/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "environment" / "local_persistent_grid.yaml"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "local_action_environment"
DEFAULT_RUN_RECEIPT = DEFAULT_RUN_DIR / "preflight.json"
DEFAULT_PROOF = REPO_ROOT / "proof" / "LOCAL_ACTION_ENVIRONMENT.json"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_evidence(path: Path) -> dict[str, Any]:
    shown = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    return {
        "path": shown,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, l2: float) -> torch.Tensor:
    design = torch.cat([x.float(), torch.ones(len(x), 1)], dim=1)
    train = design[mask]
    target = y.float()[mask]
    eye = torch.eye(train.shape[1]) * l2
    return torch.linalg.solve(train.T @ train + eye, train.T @ target)


def _ridge_predict(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.float(), torch.ones(len(x), 1)], dim=1) @ weights


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    residual = float(((pred - target) ** 2).sum())
    centered = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum())
    return 1.0 - residual / max(centered, 1.0e-12)


def _episode_split(episode_ids: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    unique = sorted(int(value) for value in torch.unique(episode_ids).tolist())
    cut = max(1, min(len(unique) - 1, int(len(unique) * fraction)))
    train_ids = set(unique[:cut])
    train = torch.tensor([int(value) in train_ids for value in episode_ids.tolist()])
    return train, ~train, cut


def _f6_preflight(bundle: dict[str, Any]) -> dict[str, Any]:
    actual = trajectory_tensors(bundle)
    counts = torch.bincount(actual["action"], minlength=4)
    moved = [bool(row["consequence"]["moved"]) for row in bundle["actual_transitions"]]
    blocked = [bool(row["consequence"]["blocked"]) for row in bundle["actual_transitions"]]
    verified = bool(
        (counts > 0).all()
        and any(moved)
        and any(blocked)
        and actual["observation"].shape == actual["next_observation"].shape
        and len(set(actual["event_refs"])) == len(actual["event_refs"])
    )
    return {
        "experiment_id": "f6_sensorimotor_form_closure",
        "mechanics_verified": verified,
        "actual_action_counts": counts.tolist(),
        "actual_observation_action_consequence_rows": len(actual["action"]),
        "action_blind_control_constructible": True,
        "action_shuffle_control_constructible": True,
        "programmatic_only": True,
    }


def _f15_preflight(bundle: dict[str, Any]) -> dict[str, Any]:
    rows = bundle["counterfactual_transitions"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["counterfactual_group_ref"]), []).append(row)
    complete = all(
        len(group) == 4 and {row["action"] for row in group} == set(range(4)) for group in groups.values()
    )
    divergent = sum(
        len({str(row["state_after_ref"]) for row in group}) > 1
        or len({float(row["consequence"]["action_cost"]) for row in group}) > 1
        for group in groups.values()
    )
    affordance_patterns = {
        tuple(bool(value) for value in group[0]["affordances_before"]) for group in groups.values()
    }
    goal_labels = {
        min(index for index, value in enumerate(group[0]["goal_affordances_before"]) if value)
        for group in groups.values()
    }
    verified = bool(complete and divergent > 0 and len(affordance_patterns) > 1 and len(goal_labels) == 4)
    return {
        "experiment_id": "f15_embodied_affordance_form",
        "mechanics_verified": verified,
        "paired_clone_groups": len(groups),
        "complete_four_action_groups": complete,
        "groups_with_distinct_consequences": divergent,
        "affordance_pattern_count": len(affordance_patterns),
        "goal_affordance_classes": sorted(goal_labels),
        "passive_zero_consequence_control_constructible": True,
        "action_shuffle_control_constructible": True,
        "programmatic_only": True,
    }


def _planning_success(
    spec_seed: int,
    horizon: int,
    weights: torch.Tensor,
    *,
    mode: str,
    episode_ids: list[int],
) -> tuple[float, int]:
    spec = make_world_spec(seed=spec_seed, horizon=horizon)
    env = PersistentGridEnvironment(spec)
    successes = 0
    inference_calls = 0
    for episode_id in episode_ids:
        state = env.reset(episode_id)
        last: dict[str, Any] | None = None
        steps = 0
        while not state.terminal:
            obs = torch.tensor(env.observe(state), dtype=torch.float32)[:12].repeat(4, 1)
            actions = torch.eye(4) if mode != "blind" else torch.zeros(4, 4)
            pred = _ridge_predict(torch.cat([obs, actions], dim=1), weights)
            inference_calls += 4
            steps += 1
            goal = torch.tensor(
                [
                    2.0 * state.goal_row / max(1, spec.grid_size - 1) - 1.0,
                    2.0 * state.goal_col / max(1, spec.grid_size - 1) - 1.0,
                ]
            )
            action = int(((pred - goal) ** 2).sum(dim=1).argmin())
            last, _branches = env.step_with_counterfactuals(action)
            assert env.state is not None
            state = env.state
        # Planning arms can reach a terminal state at different times.  Execute shape-identical
        # inference-only padding so the comparison prices the predeclared horizon rather than
        # rewarding early termination with a smaller compute bill.
        while steps < horizon:
            obs = torch.tensor(env.observe(state), dtype=torch.float32)[:12].repeat(4, 1)
            actions = torch.eye(4) if mode != "blind" else torch.zeros(4, 4)
            _ridge_predict(torch.cat([obs, actions], dim=1), weights)
            inference_calls += 4
            steps += 1
        successes += bool(last and last["consequence"]["goal_reached"])
    return successes / max(1, len(episode_ids)), inference_calls


def _reactive_success(spec_seed: int, horizon: int, episode_ids: list[int]) -> float:
    spec = make_world_spec(seed=spec_seed, horizon=horizon)
    env = PersistentGridEnvironment(spec)
    successes = 0
    for episode_id in episode_ids:
        state = env.reset(episode_id)
        last: dict[str, Any] | None = None
        while not state.terminal:
            goal = env.goal_affordances(state)
            available = env.affordances(state)
            candidates = [i for i in range(4) if goal[i] and available[i]]
            action = candidates[0] if candidates else next((i for i, ok in enumerate(available) if ok), 0)
            last, _branches = env.step_with_counterfactuals(action)
            assert env.state is not None
            state = env.state
        successes += bool(last and last["consequence"]["goal_reached"])
    return successes / max(1, len(episode_ids))


def _cm10_preflight(bundle: dict[str, Any], *, train_fraction: float, l2: float) -> dict[str, Any]:
    data = trajectory_tensors(bundle, counterfactuals=True)
    obs = data["observation"][:, :12]
    action = data["action_one_hot"]
    target = data["next_observation"][:, :2]
    train, test, cut = _episode_split(data["episode_index"], train_fraction)
    generator = torch.Generator().manual_seed(int(bundle["seed"]) + 41)
    shuffled = action.clone()
    for mask in (train, test):
        indices = torch.where(mask)[0]
        shuffled[indices] = action[indices[torch.randperm(len(indices), generator=generator)]]
    arm_actions = {
        "true_action": action,
        "action_blind": torch.zeros_like(action),
        "action_shuffled": shuffled,
    }
    weights: dict[str, torch.Tensor] = {}
    r2: dict[str, float] = {}
    for arm, supplied in arm_actions.items():
        design = torch.cat([obs, supplied], dim=1)
        weights[arm] = _ridge_fit(design, target, train, l2)
        r2[arm] = _r2(_ridge_predict(design[test], weights[arm]), target[test])

    test_ids = sorted(int(value) for value in torch.unique(data["episode_index"][test]).tolist())
    horizon = int(bundle["world"]["horizon"])
    planning: dict[str, float] = {}
    calls: dict[str, int] = {}
    for arm, mode in (("true_action", "true"), ("action_blind", "blind"), ("action_shuffled", "shuffled")):
        planning[arm], calls[arm] = _planning_success(
            int(bundle["seed"]),
            horizon,
            weights[arm],
            mode=mode,
            episode_ids=test_ids,
        )
    planning["reactive_true_state"] = _reactive_success(int(bundle["seed"]), horizon, test_ids)
    params = {arm: int(weight.numel()) for arm, weight in weights.items()}
    mechanics_verified = bool(
        train.any()
        and test.any()
        and not bool((train & test).any())
        and len(set(params.values())) == 1
        and all(torch.isfinite(weight).all() for weight in weights.values())
        and len(set(calls.values())) == 1
    )
    pilot_deltas = {
        "r2_over_strongest_action_control": r2["true_action"]
        - max(r2["action_blind"], r2["action_shuffled"]),
        "planning_over_strongest_action_control": planning["true_action"]
        - max(planning["action_blind"], planning["action_shuffled"]),
        "planning_over_reactive": planning["true_action"] - planning["reactive_true_state"],
    }
    return {
        "experiment_id": "mop_cm10_action_forward_model",
        "mechanics_verified": mechanics_verified,
        "train_episode_count": cut,
        "test_episode_ids": test_ids,
        "train_test_episode_disjoint": not bool((train & test).any()),
        "one_step_r2_by_arm": r2,
        "planning_success_by_arm": planning,
        "pilot_deltas": pilot_deltas,
        "parameter_count_by_dynamics_arm": params,
        "matched_dynamics_parameters": len(set(params.values())) == 1,
        "matched_planner_inference_calls": len(set(calls.values())) == 1,
        "planner_inference_calls_by_arm": calls,
        "controls_executed": [
            "action-blind-same-shape",
            "action-shuffled-same-shape",
            "reactive-true-state-upper-control",
            "true-dynamics-execution",
        ],
        "controls_not_yet_citable": [
            "frozen-vjepa2-ac-on-the-exact-rendered-referents",
            "matched-compute-unrolled-reactive-depth-on-the-citable-substrate",
        ],
        "scientific_ready": False,
        "programmatic_only": True,
    }


def _e5_preflight(bundle: dict[str, Any]) -> dict[str, Any]:
    data = trajectory_tensors(bundle)
    sensor = data["next_observation"][:, 12:]
    noisy = data["is_noisy_tv"]
    learnable = ~noisy
    noisy_variance = float(sensor[noisy].var()) if int(noisy.sum()) > 1 else 0.0
    learnable_variance = float(sensor[learnable].var()) if int(learnable.sum()) > 1 else 0.0
    verified = bool(noisy.any() and learnable.any() and noisy_variance > learnable_variance)
    return {
        "experiment_id": "e5_curiosity",
        "mechanics_verified": verified,
        "learnable_transition_rows": int(learnable.sum()),
        "noisy_tv_transition_rows": int(noisy.sum()),
        "learnable_sensor_variance": learnable_variance,
        "noisy_tv_sensor_variance": noisy_variance,
        "prediction_error_and_learning_progress_selection_ready": True,
        "programmatic_only": True,
    }


def build_preflight(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    profile = yaml.safe_load(config_path.read_text())
    if profile.get("schema") != "mop-local-action-environment-profile/v1":
        raise ValueError("unsupported action-environment profile")
    started = time.perf_counter()
    seed_rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in profile["seeds"]]:
        spec = make_world_spec(
            seed=seed,
            grid_size=int(profile["grid_size"]),
            horizon=int(profile["horizon"]),
        )
        bundle = collect_trajectory_bundle(
            spec,
            episodes=int(profile["episodes"]),
            policy=str(profile["policy"]),
        )
        verification = verify_trajectory_bundle(bundle)
        bundle_path = run_dir / f"trajectory_seed_{seed}.json"
        write_trajectory_bundle(bundle_path, bundle)
        lanes = {
            "f6": _f6_preflight(bundle),
            "f15": _f15_preflight(bundle),
            "cm10": _cm10_preflight(
                bundle,
                train_fraction=float(profile["train_episode_fraction"]),
                l2=float(profile["ridge_l2"]),
            ),
            "e5": _e5_preflight(bundle),
        }
        seed_rows.append(
            {
                "seed": seed,
                "world_sha256": bundle["world_sha256"],
                "trajectory_sha256": bundle["content_sha256"],
                "episode_count": bundle["episode_count"],
                "transition_count": bundle["transition_count"],
                "counterfactual_count": bundle["counterfactual_count"],
                "exact_replay_verified": verification["verified"],
                "verification": verification,
                "lanes": lanes,
                "artifact": _file_evidence(bundle_path),
            }
        )

    all_verified = all(
        row["exact_replay_verified"] and all(lane["mechanics_verified"] for lane in row["lanes"].values())
        for row in seed_rows
    )
    source_paths = (
        config_path,
        REPO_ROOT / "src" / "mop" / "environments" / "persistent_grid.py",
        REPO_ROOT / "src" / "mop" / "studies" / "local_action_environment.py",
        REPO_ROOT / "scripts" / "local_action_environment.py",
    )
    return {
        "schema": SCHEMA,
        "scope": "bounded deterministic programmatic mechanics; no natural embodiment claim",
        "profile": profile,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "max_rss_bytes": _max_rss_bytes(),
        "seed_count": len(seed_rows),
        "seeds": seed_rows,
        "all_mechanics_verified": all_verified,
        "scientific_promotion_allowed": False,
        "demoted_blockers": {
            "f6_sensorimotor_form_closure": (
                "environment mechanics and exact action controls are locally executable"
            ),
            "f15_embodied_affordance_form": (
                "paired cloned-state intervention mechanics are locally executable"
            ),
            "e5_curiosity": "bounded learnable-versus-noisy trajectory rollouts are locally executable",
            "mop_cm10_action_forward_model": (
                "the environment adapter and action-forward-model mechanics are locally executable"
            ),
        },
        "remaining_scientific_blockers": {
            "f6_sensorimotor_form_closure": (
                "natural sensorimotor or physical-embodiment claims require external evidence"
            ),
            "f15_embodied_affordance_form": (
                "natural-object affordance claims require rights-cleared embodied interventions"
            ),
            "e5_curiosity": (
                "generalization beyond this programmatic ecology requires independent environments "
                "or trajectories"
            ),
            "mop_cm10_action_forward_model": (
                "the P7 successor now closes programmatic rendering, same-parent interventions, "
                "and matched controls; the registered scientific claim still needs independently "
                "sourced trajectories, an exact-referent action control, and replication"
            ),
            "e10_openended": (
                "persistent action mechanics now exist, but sustained open-endedness still needs "
                "population-level search, environment generation, transfer, and a predeclared "
                "non-plateau horizon"
            ),
        },
        "source_evidence": [_file_evidence(path) for path in source_paths],
    }


def write_preflight(
    *,
    config_path: Path = DEFAULT_CONFIG,
    run_dir: Path = DEFAULT_RUN_DIR,
    run_receipt: Path = DEFAULT_RUN_RECEIPT,
    proof_path: Path = DEFAULT_PROOF,
) -> dict[str, Any]:
    result = build_preflight(config_path=config_path, run_dir=run_dir)
    if not result["all_mechanics_verified"]:
        raise ValueError("local action-environment mechanics did not verify")
    _atomic_json(run_receipt, result)
    _atomic_json(proof_path, result)
    return result
