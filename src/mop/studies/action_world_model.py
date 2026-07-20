
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ..config import REPO_ROOT
from ..environments.persistent_grid import (
    ACTION_DELTAS,
    GridState,
    PersistentGridEnvironment,
    WorldSpec,
    collect_trajectory_bundle,
    verify_trajectory_bundle,
)
from ..shell.predictor import Predictor
from ..substrate.events import (
    BranchMeta,
    BranchRef,
    EventRef,
    FrozenJSON,
    canonical_bytes,
    canonical_sha256,
)
from .process_resources import PeakRSSMonitor
from .runtime_integrity import (
    FORBIDDEN_MODEL_MODULES,
    deny_forbidden_runtime_imports,
    forbidden_source_imports,
)

CONFIG_SCHEMA = "mop-p7-action-world-model-config/v1"
DATASET_SCHEMA = "mop-p7-rendered-action-unit/v1"
PREFLIGHT_SCHEMA = "mop-p7-action-world-model-preflight/v1"
CLAIM_SCOPE = (
    "deterministic rendered fixture mechanics only; no natural-data, embodiment, or capability claim"
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "p7_action_world_model_preflight.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "P7_ACTION_WORLD_MODEL_PREFLIGHT.json"

ARM_ORDER = (
    "reactive_rendered",
    "model_free_recurrent",
    "compact_latent_transition",
    "object_centered_transition",
    "oracle_state",
    "action_blind",
    "action_shuffled",
    "matched_depth_reactive",
)
TRANSITION_ARMS = (
    "compact_latent_transition",
    "object_centered_transition",
    "action_blind",
    "action_shuffled",
)
EQUAL_CORE_COMPUTE_ARMS = (
    "compact_latent_transition",
    "object_centered_transition",
    "action_blind",
    "action_shuffled",
    "matched_depth_reactive",
)

PALETTE = {
    "background": (10, 12, 18),
    "wall": (90, 90, 100),
    "noisy_tv": (38, 26, 126),
    "goal": (20, 220, 60),
    "agent": (232, 44, 38),
    "agent_on_goal": (244, 220, 20),
}

IMPLEMENTATION_PATHS = (
    "configs/experiment/p7_action_world_model_preflight.yaml",
    "src/mop/studies/action_world_model.py",
    "scripts/p7_action_world_model_preflight.py",
    "tests/unit/test_action_world_model.py",
    "src/mop/studies/process_resources.py",
    "src/mop/studies/runtime_integrity.py",
)

UPSTREAM_PATHS = (
    "src/mop/environments/persistent_grid.py",
    "src/mop/substrate/events.py",
    "src/mop/experiments/expansion_harness.py",
    "src/mop/shell/predictor.py",
    "src/mop/studies/local_action_environment.py",
    "configs/experiment/e5_curiosity.yaml",
    "configs/custom_substrate/requirements.yaml",
    "proof/LOCAL_ACTION_ENVIRONMENT.json",
    "proof/EXPANSION_WAVE0.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    os.replace(tmp, path)


def _state_from_payload(payload: dict[str, Any]) -> GridState:
    return GridState(
        episode_index=int(payload["episode_index"]),
        step_index=int(payload["step_index"]),
        row=int(payload["agent"][0]),
        col=int(payload["agent"][1]),
        goal_row=int(payload["goal"][0]),
        goal_col=int(payload["goal"][1]),
        terminal=bool(payload["terminal"]),
    )


def _spec_from_construction(construction: dict[str, Any]) -> WorldSpec:
    unit = construction["unit"]
    env = construction["environment"]
    return WorldSpec(
        seed=int(unit["seed"]),
        grid_size=int(env["grid_size"]),
        horizon=int(env["horizon"]),
        walls=tuple((int(row), int(col)) for row, col in unit["walls"]),
        noisy_tv_cell=(int(unit["noisy_tv_cell"][0]), int(unit["noisy_tv_cell"][1])),
    )


def render_observation(spec: WorldSpec, state: GridState, *, cell_pixels: int) -> torch.Tensor:

    if cell_pixels < 1:
        raise ValueError("cell_pixels must be positive")
    side = spec.grid_size * cell_pixels
    frame = torch.empty((side, side, 3), dtype=torch.uint8)
    frame[:] = torch.tensor(PALETTE["background"], dtype=torch.uint8)

    def paint(cell: tuple[int, int], colour: tuple[int, int, int]) -> None:
        row, col = cell
        frame[
            row * cell_pixels : (row + 1) * cell_pixels,
            col * cell_pixels : (col + 1) * cell_pixels,
        ] = torch.tensor(colour, dtype=torch.uint8)

    for wall in spec.walls:
        paint(wall, PALETTE["wall"])

    noisy_row, noisy_col = spec.noisy_tv_cell
    for py in range(cell_pixels):
        for px in range(cell_pixels):
            digest = hashlib.sha256(
                canonical_bytes(
                    {
                        "world": spec.sha256,
                        "state": state.payload(),
                        "sensor": "p7-noisy-tv-raster",
                        "pixel": [py, px],
                    }
                )
            ).digest()
            base = PALETTE["noisy_tv"]
            colour = tuple(min(255, max(0, base[i] + int(digest[i]) % 41 - 20)) for i in range(3))
            frame[noisy_row * cell_pixels + py, noisy_col * cell_pixels + px] = torch.tensor(
                colour, dtype=torch.uint8
            )

    paint((state.goal_row, state.goal_col), PALETTE["goal"])
    agent_colour = (
        PALETTE["agent_on_goal"]
        if (state.row, state.col) == (state.goal_row, state.goal_col)
        else PALETTE["agent"]
    )
    paint((state.row, state.col), agent_colour)
    return frame


def _render_receipt(frame: torch.Tensor) -> dict[str, Any]:
    raw = bytes(int(value) for value in frame.flatten().tolist())
    digest = hashlib.sha256(
        canonical_bytes({"shape": list(frame.shape), "dtype": str(frame.dtype)}) + raw
    ).hexdigest()
    return {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "bytes": len(raw),
        "sha256": digest,
    }


def _wave_branch(unit_id: str, world_sha256: str, row: dict[str, Any]) -> BranchMeta:
    fork_key = canonical_sha256(
        {
            "unit_id": unit_id,
            "world_sha256": world_sha256,
            "persistent_event_ref": row["event_ref"],
            "parent_state_ref": row["state_before_ref"],
        }
    )[:24]
    action = int(row["action"])
    parent = FrozenJSON.from_value(
        {
            "world_sha256": world_sha256,
            "persistent_state_ref": row["state_before_ref"],
            "state": row["state_before"],
        }
    )
    intervention = FrozenJSON.from_value(
        {
            "action": action,
            "action_name": row["action_name"],
            "persistent_branch_ref": row["counterfactual_branch_ref"],
        }
    )
    consequence = FrozenJSON.from_value(
        {
            "persistent_state_ref": row["state_after_ref"],
            "state": row["state_after"],
            "consequence": row["consequence"],
        }
    )
    return BranchMeta.create(
        ref=BranchRef(f"branch:p7/{unit_id}/{fork_key}/a{action}"),
        fork_event_ref=EventRef(f"event:p7/{unit_id}/{fork_key}/fork"),
        parent_state=parent,
        intervention=intervention,
        consequence_event_ref=EventRef(f"event:p7/{unit_id}/{fork_key}/consequence/a{action}"),
        consequence_state=consequence,
        chosen=bool(row["action_was_chosen"]),
    )


def _branch_summary(
    spec: WorldSpec,
    unit_id: str,
    row: dict[str, Any],
    *,
    cell_pixels: int,
) -> dict[str, Any]:
    before = _state_from_payload(row["state_before"])
    after = _state_from_payload(row["state_after"])
    return {
        "persistent_event_ref": row["event_ref"],
        "persistent_group_ref": row["counterfactual_group_ref"],
        "persistent_branch_ref": row["counterfactual_branch_ref"],
        "state_before_ref": row["state_before_ref"],
        "state_after_ref": row["state_after_ref"],
        "state_before": row["state_before"],
        "state_after": row["state_after"],
        "action": int(row["action"]),
        "action_name": row["action_name"],
        "action_was_chosen": bool(row["action_was_chosen"]),
        "consequence": row["consequence"],
        "vector_observation_before_sha256": canonical_sha256(row["observation_before"]),
        "vector_observation_after_sha256": canonical_sha256(row["observation_after"]),
        "render_before": _render_receipt(render_observation(spec, before, cell_pixels=cell_pixels)),
        "render_after": _render_receipt(render_observation(spec, after, cell_pixels=cell_pixels)),
        "wave_e0_branch": _wave_branch(unit_id, spec.sha256, row).payload(),
    }


def _select_actual(rows: list[dict[str, Any]], episode_ids: set[int], budget: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["state_before"]["episode_index"]) in episode_ids][:budget]
    if len(selected) != budget:
        raise ValueError(f"event budget {budget} cannot be met from episodes {sorted(episode_ids)}")
    return selected


def _rollout_sequences(
    rows: list[dict[str, Any]], episode_ids: set[int], *, budget: int, horizon: int
) -> list[dict[str, Any]]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        episode = int(row["state_before"]["episode_index"])
        if episode in episode_ids:
            by_episode.setdefault(episode, []).append(row)
    output: list[dict[str, Any]] = []
    for episode in sorted(by_episode):
        episode_rows = by_episode[episode]
        for start in range(max(0, len(episode_rows) - horizon + 1)):
            sequence = episode_rows[start : start + horizon]
            output.append(
                {
                    "episode_index": episode,
                    "root_event_ref": sequence[0]["event_ref"],
                    "state_before": sequence[0]["state_before"],
                    "actions": [int(row["action"]) for row in sequence],
                    "states_after": [row["state_after"] for row in sequence],
                    "state_after_refs": [row["state_after_ref"] for row in sequence],
                }
            )
            if len(output) == budget:
                return output
    raise ValueError("rollout-root budget cannot be met without crossing episode boundaries")


def _group_payloads(
    spec: WorldSpec,
    unit_id: str,
    actual_rows: list[dict[str, Any]],
    branches_by_group: dict[str, list[dict[str, Any]]],
    *,
    cell_pixels: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for actual in actual_rows:
        group_ref = str(actual["counterfactual_group_ref"])
        branches = sorted(branches_by_group[group_ref], key=lambda row: int(row["action"]))
        if len(branches) != 4 or {int(row["action"]) for row in branches} != set(range(4)):
            raise ValueError("persistent-grid branch group is not a complete four-action intervention")
        groups.append(
            {
                "persistent_event_ref": actual["event_ref"],
                "persistent_group_ref": group_ref,
                "episode_index": int(actual["state_before"]["episode_index"]),
                "step_index": int(actual["state_before"]["step_index"]),
                "actual_action": int(actual["action"]),
                "branches": [
                    _branch_summary(spec, unit_id, row, cell_pixels=cell_pixels) for row in branches
                ],
            }
        )
    return groups


def build_unit_dataset(construction: dict[str, Any]) -> dict[str, Any]:

    spec = _spec_from_construction(construction)
    env_cfg = construction["environment"]
    budgets = construction["budgets"]
    unit_id = str(construction["unit"]["unit_id"])
    total_episodes = int(env_cfg["total_episode_count"])
    bundle = collect_trajectory_bundle(
        spec,
        episodes=total_episodes,
        policy=str(env_cfg["trajectory_policy"]),
        policy_seed=int(env_cfg["policy_seed"]),
    )
    audit = verify_trajectory_bundle(bundle)
    if audit["verified"] is not True:
        raise ValueError(f"persistent-grid bundle failed replay: {audit['errors']}")

    train_count = int(env_cfg["train_episode_count"])
    heldout_count = int(env_cfg["heldout_episode_count"])
    planning_count = int(env_cfg["planning_episode_count"])
    train_ids = set(range(train_count))
    heldout_ids = set(range(train_count, train_count + heldout_count))
    planning_ids = list(range(train_count + heldout_count, train_count + heldout_count + planning_count))
    actual_rows = list(bundle["actual_transitions"])
    train_actual = _select_actual(actual_rows, train_ids, int(budgets["train_event_budget"]))
    heldout_actual = _select_actual(actual_rows, heldout_ids, int(budgets["heldout_event_budget"]))
    branches_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in bundle["counterfactual_transitions"]:
        branches_by_group.setdefault(str(row["counterfactual_group_ref"]), []).append(row)
    cell_pixels = int(env_cfg["cell_pixels"])
    max_horizon = max(int(value) for value in construction["rollout_horizons"])
    rollouts = _rollout_sequences(
        actual_rows,
        heldout_ids,
        budget=int(budgets["rollout_root_budget"]),
        horizon=max_horizon,
    )
    train_groups = _group_payloads(spec, unit_id, train_actual, branches_by_group, cell_pixels=cell_pixels)
    heldout_groups = _group_payloads(
        spec, unit_id, heldout_actual, branches_by_group, cell_pixels=cell_pixels
    )
    action_count = int(budgets["counterfactual_actions_per_event"])
    core: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "construction": construction,
        "independent_unit": {
            "unit_ref": f"unit:p7-{unit_id}",
            "unit_id": unit_id,
            "seed": spec.seed,
            "layout_sha256": canonical_sha256(
                {
                    "grid_size": spec.grid_size,
                    "walls": [list(cell) for cell in spec.walls],
                    "noisy_tv_cell": list(spec.noisy_tv_cell),
                }
            ),
            "world_sha256": spec.sha256,
            "world_payload": spec.payload(),
            "bundle_sha256": bundle["content_sha256"],
        },
        "split_contract": {
            "train_episode_ids": sorted(train_ids),
            "heldout_intervention_episode_ids": sorted(heldout_ids),
            "planning_episode_ids": planning_ids,
            "episode_sets_disjoint": not bool(
                train_ids & heldout_ids or train_ids & set(planning_ids) or heldout_ids & set(planning_ids)
            ),
        },
        "budget_contract": {
            "train_event_roots": len(train_groups),
            "heldout_event_roots": len(heldout_groups),
            "train_branch_actions": len(train_groups) * action_count,
            "heldout_branch_actions": len(heldout_groups) * action_count,
            "heldout_unchosen_interventions": len(heldout_groups) * (action_count - 1),
            "rollout_roots": len(rollouts),
            "planning_episode_actions_maximum": planning_count * spec.horizon,
            "render_pairs_materialized": (len(train_groups) + len(heldout_groups)) * action_count,
            "render_receipts": 2 * (len(train_groups) + len(heldout_groups)) * action_count,
        },
        "splits": {
            "train": {"groups": train_groups},
            "heldout_interventions": {"groups": heldout_groups},
            "heldout_rollouts": {"sequences": rollouts},
        },
        "persistent_bundle_verification": audit,
    }
    core["payload_sha256"] = canonical_sha256(core)
    return core


def verify_unit_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["schema"] = dataset.get("schema") == DATASET_SCHEMA
    without_hash = copy.deepcopy(dataset)
    expected_hash = without_hash.pop("payload_sha256", None)
    checks["payload_hash"] = expected_hash == canonical_sha256(without_hash)
    try:
        rebuilt = build_unit_dataset(copy.deepcopy(dataset["construction"]))
        checks["exact_seed_layout_replay"] = canonical_bytes(rebuilt) == canonical_bytes(dataset)
    except (KeyError, TypeError, ValueError) as exc:
        checks["exact_seed_layout_replay"] = False
        errors.append(f"rebuild: {exc}")

    train = dataset.get("splits", {}).get("train", {}).get("groups", [])
    heldout = dataset.get("splits", {}).get("heldout_interventions", {}).get("groups", [])
    all_groups = [*train, *heldout]
    checks["four_same_parent_branches"] = bool(
        all_groups
        and all(
            len(group.get("branches", [])) == 4
            and {row.get("action") for row in group["branches"]} == set(range(4))
            and len({row["wave_e0_branch"]["parent_state"]["sha256"] for row in group["branches"]}) == 1
            and sum(bool(row.get("action_was_chosen")) for row in group["branches"]) == 1
            for group in all_groups
        )
    )
    train_refs = {group.get("persistent_event_ref") for group in train}
    heldout_refs = {group.get("persistent_event_ref") for group in heldout}
    checks["heldout_event_identity"] = bool(train_refs and heldout_refs and not train_refs & heldout_refs)
    budgets = dataset.get("budget_contract", {})
    checks["exact_budgets"] = bool(
        budgets.get("train_event_roots") == len(train)
        and budgets.get("heldout_event_roots") == len(heldout)
        and budgets.get("train_branch_actions") == 4 * len(train)
        and budgets.get("heldout_branch_actions") == 4 * len(heldout)
        and budgets.get("heldout_unchosen_interventions") == 3 * len(heldout)
        and budgets.get("render_receipts") == 8 * (len(train) + len(heldout))
    )
    checks["render_receipts_bound"] = bool(
        all(
            row.get(key, {}).get("dtype") == "torch.uint8"
            and isinstance(row.get(key, {}).get("sha256"), str)
            and len(row[key]["sha256"]) == 64
            for group in all_groups
            for row in group["branches"]
            for key in ("render_before", "render_after")
        )
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"verified": all(checks.values()), "checks": checks, "errors": errors}


def _rehash_mutation(dataset: dict[str, Any]) -> None:
    dataset.pop("payload_sha256", None)
    dataset["payload_sha256"] = canonical_sha256(dataset)


def mutation_suite(dataset: dict[str, Any]) -> dict[str, Any]:

    mutations: dict[str, Any] = {}

    def record(name: str, mutate: Any) -> None:
        changed = copy.deepcopy(dataset)
        mutate(changed)
        _rehash_mutation(changed)
        audit = verify_unit_dataset(changed)
        mutations[name] = {"rejected": audit["verified"] is False, "errors": audit["errors"]}

    record(
        "render_digest",
        lambda row: row["splits"]["heldout_interventions"]["groups"][0]["branches"][0][
            "render_after"
        ].__setitem__("sha256", "0" * 64),
    )
    record(
        "hidden_state",
        lambda row: row["splits"]["heldout_interventions"]["groups"][0]["branches"][0]["state_after"][
            "agent"
        ].__setitem__(0, 5),
    )
    record(
        "action_intervention",
        lambda row: row["splits"]["heldout_interventions"]["groups"][0]["branches"][0].__setitem__(
            "action", 3
        ),
    )

    def swap_consequence(row: dict[str, Any]) -> None:
        branches = row["splits"]["heldout_interventions"]["groups"][0]["branches"]
        (
            branches[0]["wave_e0_branch"]["consequence_state"],
            branches[1]["wave_e0_branch"]["consequence_state"],
        ) = (
            branches[1]["wave_e0_branch"]["consequence_state"],
            branches[0]["wave_e0_branch"]["consequence_state"],
        )

    record("shuffled_consequence", swap_consequence)
    record(
        "environment_seed",
        lambda row: row["construction"]["unit"].__setitem__(
            "seed", int(row["construction"]["unit"]["seed"]) + 1
        ),
    )
    record(
        "declared_budget",
        lambda row: row["budget_contract"].__setitem__(
            "heldout_branch_actions", int(row["budget_contract"]["heldout_branch_actions"]) - 1
        ),
    )
    return {
        "mutations": mutations,
        "all_rejected": all(row["rejected"] is True for row in mutations.values()),
    }


def compact_render_latent(frame: torch.Tensor) -> torch.Tensor:

    image = frame.float() / 255.0
    height, width, _ = image.shape
    rows = torch.linspace(-1.0, 1.0, height).view(height, 1, 1)
    cols = torch.linspace(-1.0, 1.0, width).view(1, width, 1)
    return torch.cat(
        [
            image.mean(dim=(0, 1)),
            (image * rows).mean(dim=(0, 1)),
            (image * cols).mean(dim=(0, 1)),
            (image.square()).mean(dim=(0, 1)),
        ]
    )


def _cell_grid(frame: torch.Tensor, spec: WorldSpec, cell_pixels: int) -> torch.Tensor:
    expected = spec.grid_size * cell_pixels
    if tuple(frame.shape) != (expected, expected, 3):
        raise ValueError("render shape does not match grid and cell-pixel contract")
    return frame.reshape(spec.grid_size, cell_pixels, spec.grid_size, cell_pixels, 3).float().mean(dim=(1, 3))


def object_render_slots(frame: torch.Tensor, spec: WorldSpec, *, cell_pixels: int) -> torch.Tensor:

    cells = _cell_grid(frame, spec, cell_pixels)
    agent_colour = torch.tensor(PALETTE["agent"], dtype=torch.float32)
    goal_colour = torch.tensor(PALETTE["goal"], dtype=torch.float32)
    overlap_colour = torch.tensor(PALETTE["agent_on_goal"], dtype=torch.float32)
    wall_colour = torch.tensor(PALETTE["wall"], dtype=torch.float32)
    agent_mask = ((cells - agent_colour).abs().sum(dim=-1) < 1.0) | (
        (cells - overlap_colour).abs().sum(dim=-1) < 1.0
    )
    goal_mask = ((cells - goal_colour).abs().sum(dim=-1) < 1.0) | (
        (cells - overlap_colour).abs().sum(dim=-1) < 1.0
    )
    wall_mask = (cells - wall_colour).abs().sum(dim=-1) < 1.0
    agent_cells = torch.nonzero(agent_mask)
    goal_cells = torch.nonzero(goal_mask)
    if len(agent_cells) != 1 or len(goal_cells) != 1:
        raise ValueError("rendered object slots cannot identify exactly one agent and goal")
    agent_row, agent_col = (int(value) for value in agent_cells[0].tolist())
    goal_row, goal_col = (int(value) for value in goal_cells[0].tolist())
    scale = max(1, spec.grid_size - 1)

    available: list[float] = []
    for delta_row, delta_col in ACTION_DELTAS:
        row, col = agent_row + delta_row, agent_col + delta_col
        valid = 0 <= row < spec.grid_size and 0 <= col < spec.grid_size
        available.append(float(valid and not bool(wall_mask[row, col])) if valid else 0.0)
    distance = (abs(agent_row - goal_row) + abs(agent_col - goal_col)) / (2.0 * scale)
    return torch.tensor(
        [
            2.0 * agent_row / scale - 1.0,
            2.0 * agent_col / scale - 1.0,
            2.0 * goal_row / scale - 1.0,
            2.0 * goal_col / scale - 1.0,
            (goal_row - agent_row) / scale,
            (goal_col - agent_col) / scale,
            *available,
            distance,
            float((agent_row, agent_col) == spec.noisy_tv_cell),
        ],
        dtype=torch.float32,
    )


def _coordinate_target(state: GridState, spec: WorldSpec) -> torch.Tensor:
    scale = max(1, spec.grid_size - 1)
    return torch.tensor(
        [
            2.0 * state.row / scale - 1.0,
            2.0 * state.col / scale - 1.0,
            2.0 * state.goal_row / scale - 1.0,
            2.0 * state.goal_col / scale - 1.0,
        ],
        dtype=torch.float32,
    )


def _examples(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    spec = _spec_from_construction(dataset["construction"])
    cell_pixels = int(dataset["construction"]["environment"]["cell_pixels"])
    output: list[dict[str, Any]] = []
    for group in dataset["splits"][split]["groups"]:
        for branch in group["branches"]:
            before = _state_from_payload(branch["state_before"])
            after = _state_from_payload(branch["state_after"])
            before_frame = render_observation(spec, before, cell_pixels=cell_pixels)
            after_frame = render_observation(spec, after, cell_pixels=cell_pixels)
            output.append(
                {
                    "event_ref": group["persistent_event_ref"],
                    "episode_index": group["episode_index"],
                    "step_index": group["step_index"],
                    "action": int(branch["action"]),
                    "chosen": bool(branch["action_was_chosen"]),
                    "before_state": before,
                    "after_state": after,
                    "before_frame": before_frame,
                    "after_frame": after_frame,
                    "compact_before": compact_render_latent(before_frame),
                    "compact_after": compact_render_latent(after_frame),
                    "object_before": object_render_slots(before_frame, spec, cell_pixels=cell_pixels),
                    "object_after": object_render_slots(after_frame, spec, cell_pixels=cell_pixels),
                    "coordinate_after": _coordinate_target(after, spec),
                    "reward": float(branch["consequence"]["reward"]),
                    "action_cost": float(branch["consequence"]["action_cost"]),
                }
            )
    return output


def _stack(examples: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.stack([row[key] for row in examples]).float()


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, l2: float = 1.0e-4) -> torch.Tensor:
    design = torch.cat([x.float(), torch.ones(len(x), 1)], dim=1)
    eye = torch.eye(design.shape[1]) * l2
    return torch.linalg.solve(design.T @ design + eye, design.T @ y.float())


def _ridge_predict(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return torch.cat([x.float(), torch.ones(len(x), 1)], dim=1) @ weights


def _predictor_linear_macs(model: nn.Module) -> int:
    return sum(
        int(layer.in_features * layer.out_features)
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    )


def _fit_predictor(
    x: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    model_cfg: dict[str, Any],
) -> tuple[Predictor, dict[str, Any]]:
    torch.manual_seed(seed)
    model = Predictor(
        dim=int(model_cfg["representation_dim"]),
        hidden=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        action_dim=int(model_cfg["action_dim"]),
        layernorm=False,
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["learning_rate"]))
    updates = int(model_cfg["train_updates"])
    with torch.no_grad():
        initial = float(F.mse_loss(model(x, action), target))
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, action), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = float(F.mse_loss(model(x, action), target))
    return model, {
        "initial_loss": initial,
        "final_loss": final,
        "updates": updates,
        "rows": len(x),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "linear_macs_per_example": _predictor_linear_macs(model),
    }


@dataclass
class ArmRuntime:
    model: Predictor
    decoder: torch.Tensor
    representation: str
    action_mode: str
    matched_depth: bool
    training: dict[str, Any]

    @property
    def trainable_parameters(self) -> int:
        return int(self.training["trainable_parameters"] + self.decoder.numel())

    @property
    def core_macs_per_call(self) -> int:
        return int(self.training["linear_macs_per_example"])


def _fit_world_models(
    dataset: dict[str, Any], model_cfg: dict[str, Any]
) -> tuple[dict[str, ArmRuntime], dict[str, Any]]:
    examples = _examples(dataset, "train")
    compact = _stack(examples, "compact_before")
    compact_next = _stack(examples, "compact_after")
    objects = _stack(examples, "object_before")
    objects_next = _stack(examples, "object_after")
    coordinates = _stack(examples, "coordinate_after")
    actions = F.one_hot(torch.tensor([row["action"] for row in examples]), num_classes=4).float()
    zeros = torch.zeros_like(actions)
    generator = torch.Generator().manual_seed(int(dataset["independent_unit"]["seed"]) + 31_337)
    permutation = torch.randperm(len(actions), generator=generator)
    shuffled = actions[permutation]
    if torch.equal(shuffled, actions):
        shuffled = actions.roll(1, dims=0)

    potential = torch.zeros_like(compact_next)
    potential[:, 0] = objects_next[:, 10]
    potential[:, 1] = torch.tensor([row["action_cost"] for row in examples]) / 2.0
    potential[:, 2] = -torch.tensor([row["reward"] for row in examples]) / 6.5
    seed = int(dataset["independent_unit"]["seed"]) + 70_001
    specifications = {
        "compact_latent_transition": (compact, actions, compact_next, "compact", "true", False),
        "object_centered_transition": (objects, actions, objects_next, "object", "true", False),
        "action_blind": (compact, zeros, compact_next, "compact", "blind", False),
        "action_shuffled": (compact, shuffled, compact_next, "compact", "true", False),
        "matched_depth_reactive": (compact, actions, potential, "compact", "true", True),
    }
    compact_decoder = _ridge_fit(compact_next, coordinates)
    object_decoder = _ridge_fit(objects_next, coordinates)
    runtimes: dict[str, ArmRuntime] = {}
    for arm, (x, action, target, representation, mode, matched) in specifications.items():
        model, training = _fit_predictor(
            x,
            action,
            target,
            seed=seed,
            model_cfg=model_cfg,
        )
        decoder = object_decoder if representation == "object" else compact_decoder
        runtimes[arm] = ArmRuntime(
            model=model,
            decoder=decoder,
            representation=representation,
            action_mode=mode,
            matched_depth=matched,
            training=training,
        )
    return runtimes, {
        "action_shuffle_is_nonidentity": not torch.equal(shuffled, actions),
        "training_branch_rows": len(examples),
        "all_action_classes": sorted({row["action"] for row in examples}) == list(range(4)),
        "decoder_shape": list(compact_decoder.shape),
    }


def _representation(
    runtime: ArmRuntime, frame: torch.Tensor, spec: WorldSpec, cell_pixels: int
) -> torch.Tensor:
    if runtime.representation == "compact":
        return compact_render_latent(frame)
    if runtime.representation == "object":
        return object_render_slots(frame, spec, cell_pixels=cell_pixels)
    raise ValueError(f"unknown representation {runtime.representation!r}")


def _model_action(runtime: ArmRuntime, action: int) -> torch.Tensor:
    if runtime.action_mode == "blind":
        return torch.zeros(1, 4)
    return F.one_hot(torch.tensor([action]), num_classes=4).float()


def _decode(runtime: ArmRuntime, representation: torch.Tensor) -> torch.Tensor:
    return _ridge_predict(representation, runtime.decoder).squeeze(0)


def _coords_to_state(coordinates: torch.Tensor, target: GridState, spec: WorldSpec) -> GridState:
    scale = max(1, spec.grid_size - 1)

    def cell(value: float) -> int:
        return min(spec.grid_size - 1, max(0, round((value + 1.0) * scale / 2.0)))

    row, col, goal_row, goal_col = (cell(float(value)) for value in coordinates[:4])
    terminal = (row, col) == (goal_row, goal_col) or target.step_index >= spec.horizon
    return GridState(
        episode_index=target.episode_index,
        step_index=target.step_index,
        row=row,
        col=col,
        goal_row=goal_row,
        goal_col=goal_col,
        terminal=terminal,
    )


def _cell_probabilities(coordinates: torch.Tensor, spec: WorldSpec, *, temperature: float) -> torch.Tensor:
    scale = max(1, spec.grid_size - 1)
    cells = torch.tensor(
        [
            [2.0 * row / scale - 1.0, 2.0 * col / scale - 1.0]
            for row in range(spec.grid_size)
            for col in range(spec.grid_size)
        ]
    )
    logits = -((cells - coordinates[:2]) ** 2).sum(dim=1) / max(temperature, 1.0e-6)
    return torch.softmax(logits, dim=0)


def _calibration(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int,
) -> dict[str, float]:
    one_hot = F.one_hot(targets, num_classes=probabilities.shape[1]).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean())
    confidence, predicted = probabilities.max(dim=1)
    correct = (predicted == targets).float()
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence >= lower) & (confidence <= upper if index == bins - 1 else confidence < upper)
        if bool(mask.any()):
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].mean()).abs())
    return {"brier": brier, "ece": ece}


def _r2(predicted: torch.Tensor, target: torch.Tensor) -> float:
    residual = float(((predicted - target) ** 2).sum())
    centered = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum())
    return 1.0 - residual / max(centered, 1.0e-12)


def _prediction_metrics(
    arm: str,
    runtime: ArmRuntime | None,
    dataset: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    if arm not in (*TRANSITION_ARMS, "oracle_state"):
        return {
            "applicable": False,
            "reason": "this control selects actions directly and has no next-observation transition output",
        }

    spec = _spec_from_construction(dataset["construction"])
    cell_pixels = int(dataset["construction"]["environment"]["cell_pixels"])
    examples = _examples(dataset, "heldout_interventions")
    predicted_coordinates: list[torch.Tensor] = []
    target_coordinates: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    target_cells: list[int] = []
    render_errors: list[float] = []
    counterfactual_correct: list[float] = []
    with torch.no_grad():
        for row in examples:
            target = row["after_state"]
            if arm == "oracle_state":
                coordinates = row["coordinate_after"]
            else:
                assert runtime is not None
                state = row[f"{runtime.representation}_before"].unsqueeze(0)
                action = _model_action(runtime, int(row["action"]))
                coordinates = _decode(runtime, runtime.model(state, action).squeeze(0))
            predicted_coordinates.append(coordinates)
            target_coordinates.append(row["coordinate_after"])
            target_cell = target.row * spec.grid_size + target.col
            target_cells.append(target_cell)
            if arm == "oracle_state":
                probability = F.one_hot(torch.tensor(target_cell), num_classes=spec.grid_size**2).float()
            else:
                probability = _cell_probabilities(
                    coordinates,
                    spec,
                    temperature=float(model_cfg["calibration_temperature"]),
                )
            probabilities.append(probability)
            predicted_state = _coords_to_state(coordinates, target, spec)
            predicted_frame = render_observation(spec, predicted_state, cell_pixels=cell_pixels).float()
            target_frame = row["after_frame"].float()
            render_errors.append(float(((predicted_frame - target_frame) / 255.0).square().mean()))
            if not row["chosen"]:
                predicted_cell = int(probability.argmax())
                counterfactual_correct.append(float(predicted_cell == target_cell))

    predicted = torch.stack(predicted_coordinates)
    target = torch.stack(target_coordinates)
    probs = torch.stack(probabilities)
    target_index = torch.tensor(target_cells)
    calibration = _calibration(probs, target_index, bins=int(evaluation_cfg["calibration_bins"]))
    one_step = {
        "coordinate_r2": _r2(predicted, target),
        "agent_position_rmse": float(torch.sqrt(((predicted[:, :2] - target[:, :2]) ** 2).mean())),
        "agent_cell_accuracy": float((probs.argmax(dim=1) == target_index).float().mean()),
        "heldout_counterfactual_agent_cell_accuracy": sum(counterfactual_correct)
        / max(1, len(counterfactual_correct)),
        "render_mse": sum(render_errors) / max(1, len(render_errors)),
        "calibration_brier": calibration["brier"],
        "calibration_ece": calibration["ece"],
        "branch_rows": len(examples),
        "unchosen_intervention_rows": len(counterfactual_correct),
    }

    horizon_rows: dict[str, dict[str, float]] = {}
    sequences = dataset["splits"]["heldout_rollouts"]["sequences"]
    for horizon in [int(value) for value in evaluation_cfg["rollout_horizons"]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        pixel_errors: list[float] = []
        with torch.no_grad():
            for sequence in sequences:
                before = _state_from_payload(sequence["state_before"])
                frame = render_observation(spec, before, cell_pixels=cell_pixels)
                if arm == "oracle_state":
                    representation = None
                else:
                    assert runtime is not None
                    representation = _representation(runtime, frame, spec, cell_pixels)
                coordinates = _coordinate_target(before, spec)
                for index, action in enumerate(sequence["actions"][:horizon]):
                    target_state = _state_from_payload(sequence["states_after"][index])
                    if arm == "oracle_state":
                        coordinates = _coordinate_target(target_state, spec)
                    else:
                        assert runtime is not None and representation is not None
                        representation = runtime.model(
                            representation.unsqueeze(0), _model_action(runtime, int(action))
                        ).squeeze(0)
                        coordinates = _decode(runtime, representation)
                final_target = _state_from_payload(sequence["states_after"][horizon - 1])
                target_coordinates_row = _coordinate_target(final_target, spec)
                predictions.append(coordinates)
                targets.append(target_coordinates_row)
                predicted_state = _coords_to_state(coordinates, final_target, spec)
                predicted_frame = render_observation(spec, predicted_state, cell_pixels=cell_pixels).float()
                target_frame = render_observation(spec, final_target, cell_pixels=cell_pixels).float()
                pixel_errors.append(float(((predicted_frame - target_frame) / 255.0).square().mean()))
        pred = torch.stack(predictions)
        truth = torch.stack(targets)
        predicted_cells_list: list[int] = []
        for index, row in enumerate(pred):
            final_state = _state_from_payload(sequences[index]["states_after"][horizon - 1])
            predicted_state = _coords_to_state(row, final_state, spec)
            predicted_cells_list.append(predicted_state.row * spec.grid_size + predicted_state.col)
        predicted_cells = torch.tensor(predicted_cells_list)
        true_cells = torch.tensor(
            [
                int(sequence["states_after"][horizon - 1]["agent"][0]) * spec.grid_size
                + int(sequence["states_after"][horizon - 1]["agent"][1])
                for sequence in sequences
            ]
        )
        horizon_rows[str(horizon)] = {
            "agent_position_rmse": float(torch.sqrt(((pred[:, :2] - truth[:, :2]) ** 2).mean())),
            "agent_cell_accuracy": float((predicted_cells == true_cells).float().mean()),
            "render_mse": sum(pixel_errors) / max(1, len(pixel_errors)),
            "rollout_roots": len(sequences),
        }
    return {"applicable": True, "one_step": one_step, "horizons": horizon_rows}


def _reactive_action(frame: torch.Tensor, spec: WorldSpec, cell_pixels: int) -> int:
    slots = object_render_slots(frame, spec, cell_pixels=cell_pixels)
    scale = max(1, spec.grid_size - 1)
    row = round((float(slots[0]) + 1.0) * scale / 2.0)
    col = round((float(slots[1]) + 1.0) * scale / 2.0)
    goal_row = round((float(slots[2]) + 1.0) * scale / 2.0)
    goal_col = round((float(slots[3]) + 1.0) * scale / 2.0)
    scored: list[tuple[int, int]] = []
    for action, (delta_row, delta_col) in enumerate(ACTION_DELTAS):
        if not bool(slots[6 + action]):
            next_row, next_col = row, col
        else:
            next_row, next_col = row + delta_row, col + delta_col
        scored.append((abs(next_row - goal_row) + abs(next_col - goal_col), action))
    return min(scored)[1]


def _recurrent_hidden(
    hidden: torch.Tensor,
    observation: torch.Tensor,
    previous_action: int | None,
    *,
    decay: float,
) -> torch.Tensor:
    action_trace = torch.zeros_like(hidden)
    if previous_action is not None:
        action_trace[previous_action] = 1.0
    return torch.tanh(decay * hidden + (1.0 - decay) * observation + 0.10 * action_trace)


def _fit_model_free_recurrent(
    dataset: dict[str, Any], model_cfg: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    spec = _spec_from_construction(dataset["construction"])
    cell_pixels = int(dataset["construction"]["environment"]["cell_pixels"])
    decay = float(model_cfg["recurrent_decay"])
    features: list[torch.Tensor] = []
    targets: list[float] = []
    hidden = torch.zeros(int(model_cfg["representation_dim"]))
    previous_episode: int | None = None
    previous_action: int | None = None
    for group in dataset["splits"]["train"]["groups"]:
        episode = int(group["episode_index"])
        if previous_episode != episode:
            hidden.zero_()
            previous_action = None
            previous_episode = episode
        state = _state_from_payload(group["branches"][0]["state_before"])
        frame = render_observation(spec, state, cell_pixels=cell_pixels)
        observation = compact_render_latent(frame)
        hidden = _recurrent_hidden(hidden, observation, previous_action, decay=decay)
        before_distance = abs(state.row - state.goal_row) + abs(state.col - state.goal_col)
        for branch in group["branches"]:
            after = _state_from_payload(branch["state_after"])
            after_distance = abs(after.row - after.goal_row) + abs(after.col - after.goal_col)
            action = int(branch["action"])
            action_form = F.one_hot(torch.tensor(action), num_classes=4).float()
            features.append(torch.cat([hidden, observation, action_form]))
            progress = float(before_distance - after_distance)
            targets.append(float(branch["consequence"]["reward"]) + 0.5 * progress)
        previous_action = int(group["actual_action"])
    x = torch.stack(features)
    y = torch.tensor(targets).unsqueeze(1)
    weights = _ridge_fit(x, y)
    return weights, {
        "trainable_parameters": weights.numel(),
        "training_rows": len(x),
        "transition_model": False,
        "objective": "direct action value from recurrent observation history",
    }


def _model_free_action(
    weights: torch.Tensor,
    hidden: torch.Tensor,
    observation: torch.Tensor,
) -> int:
    rows = []
    for action in range(4):
        rows.append(torch.cat([hidden, observation, F.one_hot(torch.tensor(action), num_classes=4).float()]))
    values = _ridge_predict(torch.stack(rows), weights).squeeze(1)
    return int(values.argmax())


def _plan_learned(
    runtime: ArmRuntime,
    frame: torch.Tensor,
    spec: WorldSpec,
    *,
    cell_pixels: int,
    depth: int,
) -> tuple[int, int, int]:
    start = _representation(runtime, frame, spec, cell_pixels)
    best: tuple[float, tuple[int, ...]] | None = None
    calls = 0
    decoder_calls = 0
    with torch.no_grad():
        for sequence in itertools.product(range(4), repeat=depth):
            state = start
            potential = 0.0
            final_output = state
            for action in sequence:
                output = runtime.model(
                    (start if runtime.matched_depth else state).unsqueeze(0),
                    _model_action(runtime, int(action)),
                ).squeeze(0)
                calls += 1
                final_output = output
                if runtime.matched_depth:
                    potential += float(output[0])
                else:
                    state = output
            coordinates = _decode(runtime, final_output)
            decoder_calls += 1
            if runtime.matched_depth:
                score = potential
            else:
                score = float(abs(coordinates[0] - coordinates[2]) + abs(coordinates[1] - coordinates[3]))
            candidate = (score, tuple(int(value) for value in sequence))
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1][0], calls, decoder_calls


def _oracle_action(env: PersistentGridEnvironment, state: GridState, *, depth: int) -> tuple[int, int]:
    best: tuple[float, tuple[int, ...]] | None = None
    calls = 0
    for sequence in itertools.product(range(4), repeat=depth):
        simulated = state
        reward = 0.0
        for action in sequence:
            if simulated.terminal:
                break
            simulated, consequence = env.simulate(simulated, int(action))
            reward += float(consequence["reward"])
            calls += 1
        distance = abs(simulated.row - simulated.goal_row) + abs(simulated.col - simulated.goal_col)
        reached_goal = (simulated.row, simulated.col) == (
            simulated.goal_row,
            simulated.goal_col,
        )
        score = distance - 100.0 * float(reached_goal)
        score -= 0.01 * reward
        candidate = (score, tuple(int(value) for value in sequence))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1][0], calls


def _planning_metrics(
    arm: str,
    runtime: ArmRuntime | None,
    recurrent_weights: torch.Tensor,
    dataset: dict[str, Any],
    model_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
) -> dict[str, Any]:
    spec = _spec_from_construction(dataset["construction"])
    cell_pixels = int(dataset["construction"]["environment"]["cell_pixels"])
    episode_ids = dataset["split_contract"]["planning_episode_ids"]
    depth = int(evaluation_cfg["planner_depth"])
    successes = 0
    returns: list[float] = []
    costs: list[float] = []
    executed_actions = 0
    model_calls = 0
    decoder_calls = 0
    padded_decisions = 0
    for episode in episode_ids:
        env = PersistentGridEnvironment(spec)
        state = env.reset(int(episode))
        total_reward = 0.0
        total_cost = 0.0
        hidden = torch.zeros(int(model_cfg["representation_dim"]))
        previous_action: int | None = None
        last_goal = False
        for _ in range(spec.horizon):
            frame = render_observation(spec, state, cell_pixels=cell_pixels)
            if arm in EQUAL_CORE_COMPUTE_ARMS:
                assert runtime is not None
                action, calls, decodes = _plan_learned(
                    runtime,
                    frame,
                    spec,
                    cell_pixels=cell_pixels,
                    depth=depth,
                )
                model_calls += calls
                decoder_calls += decodes
            elif arm == "oracle_state":
                if state.terminal:
                    action = 0
                    calls = 0
                else:
                    action, calls = _oracle_action(env, state, depth=depth)
                model_calls += calls
            elif arm == "reactive_rendered":
                action = _reactive_action(frame, spec, cell_pixels)
            elif arm == "model_free_recurrent":
                observation = compact_render_latent(frame)
                hidden = _recurrent_hidden(
                    hidden,
                    observation,
                    previous_action,
                    decay=float(model_cfg["recurrent_decay"]),
                )
                action = _model_free_action(recurrent_weights, hidden, observation)
            else:
                raise ValueError(f"unsupported planning arm {arm!r}")

            if state.terminal:
                padded_decisions += 1
                continue
            actual, _branches = env.step_with_counterfactuals(action)
            executed_actions += 1
            total_reward += float(actual["consequence"]["reward"])
            total_cost += float(actual["consequence"]["action_cost"])
            last_goal = bool(actual["consequence"]["goal_reached"])
            previous_action = action
            assert env.state is not None
            state = env.state
        successes += int(last_goal)
        returns.append(total_reward)
        costs.append(total_cost)
    count = max(1, len(episode_ids))
    return {
        "success_rate": successes / count,
        "mean_return": sum(returns) / count,
        "mean_action_cost": sum(costs) / count,
        "planning_episodes": len(episode_ids),
        "action_opportunities": len(episode_ids) * spec.horizon,
        "executed_true_dynamics_actions": executed_actions,
        "padded_compute_decisions": padded_decisions,
        "planner_model_calls": model_calls,
        "planner_decoder_calls": decoder_calls,
        "planner_depth": depth,
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_tree(item) for item in value)
    return True


def _construction(config: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    environment = config["environment"]
    total = (
        int(environment["train_episode_count"])
        + int(environment["heldout_episode_count"])
        + int(environment["planning_episode_count"])
    )
    return {
        "unit": copy.deepcopy(unit),
        "environment": {
            "grid_size": int(environment["grid_size"]),
            "horizon": int(environment["horizon"]),
            "cell_pixels": int(environment["cell_pixels"]),
            "trajectory_policy": str(environment["trajectory_policy"]),
            "policy_seed": int(unit["seed"]) + 7_919,
            "train_episode_count": int(environment["train_episode_count"]),
            "heldout_episode_count": int(environment["heldout_episode_count"]),
            "planning_episode_count": int(environment["planning_episode_count"]),
            "total_episode_count": total,
        },
        "budgets": {
            "train_event_budget": int(environment["train_event_budget"]),
            "heldout_event_budget": int(environment["heldout_event_budget"]),
            "rollout_root_budget": int(environment["rollout_root_budget"]),
            "counterfactual_actions_per_event": int(environment["counterfactual_actions_per_event"]),
        },
        "rollout_horizons": [int(value) for value in config["evaluation"]["rollout_horizons"]],
    }


def _run_unit(config: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    dataset = build_unit_dataset(_construction(config, unit))
    verification = verify_unit_dataset(dataset)
    if verification["verified"] is not True:
        raise ValueError(f"P7 rendered dataset failed replay: {verification['errors']}")
    mutations = mutation_suite(dataset)
    if mutations["all_rejected"] is not True:
        raise ValueError("P7 replay mutation suite did not fail closed")

    runtimes, training_audit = _fit_world_models(dataset, config["model"])
    recurrent_weights, recurrent_training = _fit_model_free_recurrent(dataset, config["model"])
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        runtime = runtimes.get(arm)
        prediction = _prediction_metrics(
            arm,
            runtime,
            dataset,
            config["evaluation"],
            config["model"],
        )
        planning = _planning_metrics(
            arm,
            runtime,
            recurrent_weights,
            dataset,
            config["model"],
            config["evaluation"],
        )
        if runtime is not None:
            cost = {
                "trainable_parameters": runtime.trainable_parameters,
                "training_updates": int(runtime.training["updates"]),
                "training_rows": int(runtime.training["rows"]),
                "core_macs_per_model_call": runtime.core_macs_per_call,
                "planner_core_macs": runtime.core_macs_per_call * int(planning["planner_model_calls"]),
                "decoder_parameters": runtime.decoder.numel(),
                "training_loss": {
                    "initial": runtime.training["initial_loss"],
                    "final": runtime.training["final_loss"],
                },
            }
        elif arm == "model_free_recurrent":
            cost = {
                **recurrent_training,
                "training_updates": 1,
                "planner_core_macs": int(planning["action_opportunities"])
                * int(recurrent_weights.numel())
                * 4,
            }
        else:
            cost = {
                "trainable_parameters": 0,
                "training_updates": 0,
                "planner_core_macs": 0,
            }
        arms[arm] = {
            "role": (
                "primary"
                if arm in {"compact_latent_transition", "object_centered_transition"}
                else (
                    "negative-control"
                    if arm in {"action_blind", "action_shuffled", "matched_depth_reactive"}
                    else "baseline"
                )
            ),
            "prediction": prediction,
            "planning": planning,
            "cost": cost,
            "mechanics_only": True,
        }

    reactive_success = float(arms["reactive_rendered"]["planning"]["success_rate"])
    reactive_return = float(arms["reactive_rendered"]["planning"]["mean_return"])
    for arm in arms:
        arms[arm]["planning_benefit"] = {
            "success_delta_vs_reactive": float(arms[arm]["planning"]["success_rate"]) - reactive_success,
            "return_delta_vs_reactive": float(arms[arm]["planning"]["mean_return"]) - reactive_return,
        }

    strongest_action_control_prediction = max(
        float(arms["action_blind"]["prediction"]["one_step"]["agent_cell_accuracy"]),
        float(arms["action_shuffled"]["prediction"]["one_step"]["agent_cell_accuracy"]),
    )
    strongest_non_oracle_planning_control = max(
        float(arms[arm]["planning"]["success_rate"])
        for arm in (
            "reactive_rendered",
            "model_free_recurrent",
            "action_blind",
            "action_shuffled",
            "matched_depth_reactive",
        )
    )
    primary_beats_controls = {
        arm: bool(
            float(arms[arm]["prediction"]["one_step"]["agent_cell_accuracy"])
            > strongest_action_control_prediction
            and float(arms[arm]["planning"]["success_rate"]) > strongest_non_oracle_planning_control
        )
        for arm in ("compact_latent_transition", "object_centered_transition")
    }

    equal_rows = [arms[arm] for arm in EQUAL_CORE_COMPUTE_ARMS]
    equal_params = {int(row["cost"]["trainable_parameters"]) for row in equal_rows}
    equal_calls = {int(row["planning"]["planner_model_calls"]) for row in equal_rows}
    equal_decodes = {int(row["planning"]["planner_decoder_calls"]) for row in equal_rows}
    equal_macs = {int(row["cost"]["planner_core_macs"]) for row in equal_rows}
    spec = _spec_from_construction(dataset["construction"])
    sample = _state_from_payload(dataset["splits"]["train"]["groups"][0]["branches"][0]["state_before"])
    first_render = render_observation(
        spec,
        sample,
        cell_pixels=int(dataset["construction"]["environment"]["cell_pixels"]),
    )
    second_render = render_observation(
        spec,
        sample,
        cell_pixels=int(dataset["construction"]["environment"]["cell_pixels"]),
    )
    checks = {
        "dataset_exact_replay": verification["verified"],
        "all_replay_mutations_rejected": mutations["all_rejected"],
        "all_eight_arms_executed": tuple(arms) == ARM_ORDER,
        "all_action_classes_in_training": training_audit["all_action_classes"],
        "action_shuffle_is_nonidentity": training_audit["action_shuffle_is_nonidentity"],
        "equal_core_parameter_count": len(equal_params) == 1,
        "equal_planner_model_calls": len(equal_calls) == 1,
        "equal_planner_decoder_calls": len(equal_decodes) == 1,
        "equal_planner_core_macs": len(equal_macs) == 1,
        "render_repeat_exact": torch.equal(first_render, second_render),
        "all_metrics_finite": _finite_tree(arms),
        "all_models_cpu": all(
            all(parameter.device.type == "cpu" for parameter in runtime.model.parameters())
            for runtime in runtimes.values()
        ),
    }
    return {
        "independent_unit": dataset["independent_unit"],
        "split_contract": dataset["split_contract"],
        "budget_contract": dataset["budget_contract"],
        "dataset_payload_sha256": dataset["payload_sha256"],
        "dataset_verification": verification,
        "mutation_suite": mutations,
        "wave_e0_f29_binding": {
            "same_parent_pre_action_branches": verification["checks"]["four_same_parent_branches"],
            "unique_action_interventions_per_fork": True,
            "exact_branch_replay_digests": True,
            "shuffled_consequence_mutation_rejected": mutations["mutations"]["shuffled_consequence"][
                "rejected"
            ],
        },
        "training_audit": training_audit,
        "arms": arms,
        "fixture_outcome": {
            "strongest_action_control_agent_cell_accuracy": strongest_action_control_prediction,
            "strongest_non_oracle_planning_control_success": strongest_non_oracle_planning_control,
            "primary_beats_both_prediction_and_planning_controls": primary_beats_controls,
            "fixture_null_supported": not any(primary_beats_controls.values()),
            "scientific_verdict": "not-eligible",
        },
        "equal_core_compute": {
            "arms": list(EQUAL_CORE_COMPUTE_ARMS),
            "trainable_parameter_count": next(iter(equal_params)),
            "planner_model_calls": next(iter(equal_calls)),
            "planner_decoder_calls": next(iter(equal_decodes)),
            "planner_core_macs": next(iter(equal_macs)),
            "matched": all(
                [
                    len(equal_params) == 1,
                    len(equal_calls) == 1,
                    len(equal_decodes) == 1,
                    len(equal_macs) == 1,
                ]
            ),
            "boundary": (
                "linear-layer active MACs and decoder calls are exact; optimizer objective and "
                "nonlinear/scoring constants are reported but not claimed identical"
            ),
        },
        "checks": checks,
        "all_mechanics_ok": all(checks.values()),
        "scientific_promotion_allowed": False,
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("P7 action-world-model config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("P7 claim scope drift")
    if not str(config.get("null_hypothesis", "")).strip():
        raise ValueError("P7 null hypothesis is required")
    envelope = config.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or envelope.get("accelerator_required") is not False
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("model_downloads_allowed") is not False
    ):
        raise ValueError("P7 preflight must remain one-thread CPU with no weights or downloads")
    evaluation = config.get("evaluation", {})
    declared = tuple(
        [
            *evaluation.get("baselines", ()),
            *evaluation.get("primary_arms", ()),
            *evaluation.get("negative_controls", ()),
        ]
    )
    if set(declared) != set(ARM_ORDER):
        raise ValueError("P7 arm set drift")
    if tuple(evaluation.get("equal_core_compute_group", ())) != EQUAL_CORE_COMPUTE_ARMS:
        raise ValueError("P7 equal-core-compute group drift")
    units = config.get("environment", {}).get("independent_units", ())
    if len(units) < int(config.get("stop_contract", {}).get("minimum_independent_units", 0)):
        raise ValueError("P7 independent-unit minimum is unmet")
    if len({unit["unit_id"] for unit in units}) != len(units):
        raise ValueError("P7 unit ids must be unique")
    if len({int(unit["seed"]) for unit in units}) != len(units):
        raise ValueError("P7 unit seeds must be unique")
    return config


def build_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    source_import_problems = forbidden_source_imports(Path(__file__))
    with PeakRSSMonitor() as rss_monitor, deny_forbidden_runtime_imports() as runtime_import_attempts:
        torch.set_num_threads(int(config["resource_envelope"]["cpu_threads"]))
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        units: list[dict[str, Any]] = []
        for unit in config["environment"]["independent_units"]:
            units.append(_run_unit(config, unit))
            if time.perf_counter() - started > float(config["resource_envelope"]["maximum_wall_seconds"]):
                raise ValueError("P7 no-heavy wall envelope exceeded")
    max_rss = rss_monitor.peak_rss_bytes
    if not rss_monitor.all_ok:
        raise ValueError("P7 phase-local RSS sampling failed")
    if rss_monitor.peak_increment_bytes > int(config["resource_envelope"]["maximum_rss_bytes"]):
        raise ValueError("P7 no-heavy RSS envelope exceeded")

    checks = {
        "minimum_independent_units": len(units) >= int(config["stop_contract"]["minimum_independent_units"]),
        "unique_seed_layout_units": len(
            {
                (
                    int(row["independent_unit"]["seed"]),
                    row["independent_unit"]["layout_sha256"],
                )
                for row in units
            }
        )
        == len(units),
        "all_unit_mechanics_ok": all(row["all_mechanics_ok"] is True for row in units),
        "all_scientific_promotion_blocked": all(
            row["scientific_promotion_allowed"] is False for row in units
        ),
        "no_forbidden_model_imports_in_source": not source_import_problems,
        "no_runtime_model_import_attempts": not runtime_import_attempts,
        "no_inherited_model_modules_loaded": not source_import_problems and not runtime_import_attempts,
        "resource_envelope_observed": rss_monitor.peak_increment_bytes
        <= int(config["resource_envelope"]["maximum_rss_bytes"]),
        "resource_rss_sampling_complete": rss_monitor.all_ok,
    }
    deterministic_core: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "status": "mechanics-pass" if all(checks.values()) else "mechanics-fail",
        "null_hypothesis": config["null_hypothesis"],
        "audit": {
            "already_existed": [
                "persistent-grid exact hidden states and four-way pre-action branches",
                "Wave E0 immutable event, state, intervention, and branch replay identities",
                "F6 action-blind and action-shuffled transition controls",
                "CM10 ridge forward-model and true-dynamics planning pilot",
                "E5 learnable-versus-noisy trajectory regions",
                "the shell action-conditioned Predictor",
            ],
            "true_gap_closed": [
                "deterministic RGB rendering bound to exact hidden state",
                "one shared eight-arm held-out intervention comparison surface",
                "compact rendered and object-centered transition arms",
                "model-free recurrent, oracle-state, and matched-depth reactive baselines",
                "prediction, calibration, horizon, planning-benefit, and cost ledgers",
                "exact core-MAC matching and replay mutation checks",
            ],
            "not_duplicated": (
                "the persistent transition function, counterfactual collector, Wave branch digest, "
                "and shell Predictor are imported rather than reimplemented"
            ),
        },
        "config": {
            "path": str(config_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(config_path),
            "payload_sha256": canonical_sha256(config),
            "payload": config,
        },
        "runtime_integrity": {
            "forbidden_module_prefixes": list(FORBIDDEN_MODEL_MODULES),
            "source_import_problems": source_import_problems,
            "runtime_import_attempts": runtime_import_attempts,
            "all_ok": not source_import_problems and not runtime_import_attempts,
        },
        "units": units,
        "checks": checks,
        "claim_boundary": {
            "mechanics_only": True,
            "natural_data": False,
            "embodiment": False,
            "sentience_or_cognition_claim": False,
            "scientific_promotion_allowed": False,
            "current_inherited_action_control": {
                "executed": False,
                "reason": (
                    "the no-weight fixture cannot make an inherited action model a fair exact-referent "
                    "control; inventing or substituting one would be scientifically invalid"
                ),
            },
            "remaining_external_validity_gate": (
                "independently sourced action-conditioned rendered or natural trajectories with an "
                "exact-referent control and predeclared replication"
            ),
        },
        "implementation": [
            _file_receipt(REPO_ROOT / path) for path in (*IMPLEMENTATION_PATHS, *UPSTREAM_PATHS)
        ],
        "all_mechanics_ok": all(checks.values()),
    }
    deterministic_core["deterministic_core_sha256"] = canonical_sha256(deterministic_core)
    receipt = {
        **deterministic_core,
        "resource_observation": {
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": max_rss,
            "phase_local_peak_rss_increment_bytes": rss_monitor.peak_increment_bytes,
            "rss_limit_scope": "phase-local sampled peak increment above phase-start RSS",
            "rss_measurement": rss_monitor.receipt(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "accelerator_required": False,
            "model_weights_loaded": False,
            "model_downloads_performed": False,
            "command_executed_heavy_work": False,
        },
    }
    return receipt


def write_preflight(
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    receipt = build_preflight(config_path)
    if receipt["all_mechanics_ok"] is not True:
        raise ValueError("P7 action-world-model mechanics did not pass")
    _atomic_json(output_path, receipt)
    return receipt
