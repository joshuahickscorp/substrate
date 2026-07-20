from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mop.substrate.events import canonical_bytes

WORLD_SCHEMA = "mop-persistent-grid-world/v1"
TRAJECTORY_SCHEMA = "mop-action-trajectory-bundle/v1"
ACTION_NAMES = ("north", "south", "west", "east")
ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OBSERVATION_DIM = 16


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stable_index(parts: tuple[Any, ...], count: int) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], "big") % count


@dataclass(frozen=True)
class WorldSpec:
    seed: int
    grid_size: int
    horizon: int
    walls: tuple[tuple[int, int], ...]
    noisy_tv_cell: tuple[int, int]

    def payload(self) -> dict[str, Any]:
        return {
            "schema": WORLD_SCHEMA,
            "seed": self.seed,
            "grid_size": self.grid_size,
            "horizon": self.horizon,
            "walls": [list(cell) for cell in self.walls],
            "noisy_tv_cell": list(self.noisy_tv_cell),
            "action_names": list(ACTION_NAMES),
            "observation_dim": OBSERVATION_DIM,
        }

    @property
    def sha256(self) -> str:
        return _sha256_value(self.payload())

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WorldSpec:
        if payload.get("schema") != WORLD_SCHEMA:
            raise ValueError("unsupported world schema")
        if tuple(payload.get("action_names", ())) != ACTION_NAMES:
            raise ValueError("action vocabulary drift")
        if int(payload.get("observation_dim", -1)) != OBSERVATION_DIM:
            raise ValueError("observation dimension drift")
        spec = cls(
            seed=int(payload["seed"]),
            grid_size=int(payload["grid_size"]),
            horizon=int(payload["horizon"]),
            walls=tuple((int(cell[0]), int(cell[1])) for cell in payload["walls"]),
            noisy_tv_cell=(
                int(payload["noisy_tv_cell"][0]),
                int(payload["noisy_tv_cell"][1]),
            ),
        )
        _validate_spec(spec)
        return spec


def make_world_spec(*, seed: int, grid_size: int = 6, horizon: int = 12) -> WorldSpec:
    if grid_size < 5:
        raise ValueError("grid_size must be at least five")
    center = grid_size // 2
    walls = tuple(
        sorted(
            {
                (center - 1, center),
                (center, center),
                (center, center - 1),
            }
        )
    )
    spec = WorldSpec(
        seed=int(seed),
        grid_size=int(grid_size),
        horizon=int(horizon),
        walls=walls,
        noisy_tv_cell=(grid_size - 1, grid_size - 1),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: WorldSpec) -> None:
    if spec.grid_size < 5 or spec.horizon < 2:
        raise ValueError("invalid bounded-world dimensions")
    cells = set(spec.walls)
    if len(cells) != len(spec.walls):
        raise ValueError("duplicate wall cells")
    for row, col in (*spec.walls, spec.noisy_tv_cell):
        if not (0 <= row < spec.grid_size and 0 <= col < spec.grid_size):
            raise ValueError("world cell outside grid")
    if spec.noisy_tv_cell in cells:
        raise ValueError("noisy-TV cell cannot be a wall")


@dataclass(frozen=True)
class GridState:
    episode_index: int
    step_index: int
    row: int
    col: int
    goal_row: int
    goal_col: int
    terminal: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "step_index": self.step_index,
            "agent": [self.row, self.col],
            "goal": [self.goal_row, self.goal_col],
            "terminal": self.terminal,
        }


class PersistentGridEnvironment:
    def __init__(self, spec: WorldSpec):
        _validate_spec(spec)
        self.spec = spec
        self.state: GridState | None = None

    def _free_cells(self, *, include_noisy: bool = False) -> list[tuple[int, int]]:
        blocked = set(self.spec.walls)
        cells = [
            (row, col)
            for row in range(self.spec.grid_size)
            for col in range(self.spec.grid_size)
            if (row, col) not in blocked
        ]
        if not include_noisy:
            cells.remove(self.spec.noisy_tv_cell)
        return cells

    def reset(self, episode_index: int) -> GridState:
        free = self._free_cells(include_noisy=False)
        start_i = _stable_index((self.spec.seed, "start", episode_index), len(free))
        start = free[start_i]
        goal_pool = [cell for cell in free if cell != start]
        goal_i = _stable_index((self.spec.seed, "goal", episode_index), len(goal_pool))
        goal = goal_pool[goal_i]
        self.state = GridState(
            episode_index=int(episode_index),
            step_index=0,
            row=start[0],
            col=start[1],
            goal_row=goal[0],
            goal_col=goal[1],
        )
        return self.state

    @property
    def world_ref(self) -> str:
        return f"world:{self.spec.sha256[:20]}"

    def episode_ref(self, state: GridState) -> str:
        return f"{self.world_ref}/episode:{state.episode_index:05d}"

    def state_ref(self, state: GridState) -> str:
        return f"state:{_sha256_value({'world': self.spec.sha256, 'state': state.payload()})[:24]}"

    def entity_refs(self, state: GridState) -> list[str]:
        prefix = self.episode_ref(state)
        refs = [
            f"{prefix}/entity:agent",
            f"{prefix}/entity:goal",
            f"{self.world_ref}/entity:noisy-tv",
        ]
        refs.extend(f"{self.world_ref}/entity:wall:{row}:{col}" for row, col in self.spec.walls)
        return refs

    def _candidate_position(self, state: GridState, action: int) -> tuple[int, int]:
        if action not in range(len(ACTION_NAMES)):
            raise ValueError(f"invalid action {action}")
        dr, dc = ACTION_DELTAS[action]
        row, col = state.row + dr, state.col + dc
        if not (0 <= row < self.spec.grid_size and 0 <= col < self.spec.grid_size):
            return state.row, state.col
        if (row, col) in set(self.spec.walls):
            return state.row, state.col
        return row, col

    def affordances(self, state: GridState) -> tuple[bool, ...]:
        if state.terminal:
            return (False,) * len(ACTION_NAMES)
        return tuple(self._candidate_position(state, action) != (state.row, state.col) for action in range(4))

    def goal_affordances(self, state: GridState) -> tuple[bool, ...]:
        distances = []
        for action in range(4):
            row, col = self._candidate_position(state, action)
            distances.append(abs(row - state.goal_row) + abs(col - state.goal_col))
        best = min(distances)
        return tuple(value == best for value in distances)

    def _sensor_channels(self, state: GridState) -> tuple[float, float, float, float]:
        if (state.row, state.col) != self.spec.noisy_tv_cell:
            return (0.0, 0.0, 0.0, 0.0)
        digest = hashlib.sha256(
            canonical_bytes(
                [self.spec.seed, "noisy-tv", state.episode_index, state.step_index, self.state_ref(state)]
            )
        ).digest()
        values = []
        for offset in range(0, 8, 2):
            integer = int.from_bytes(digest[offset : offset + 2], "big")
            values.append(2.0 * integer / 65535.0 - 1.0)
        return tuple(values)  # type: ignore[return-value]

    def observe(self, state: GridState) -> tuple[float, ...]:
        scale = max(1, self.spec.grid_size - 1)
        row = 2.0 * state.row / scale - 1.0
        col = 2.0 * state.col / scale - 1.0
        goal_row = 2.0 * state.goal_row / scale - 1.0
        goal_col = 2.0 * state.goal_col / scale - 1.0
        dx = (state.goal_row - state.row) / scale
        dy = (state.goal_col - state.col) / scale
        available = tuple(float(value) for value in self.affordances(state))
        progress = state.step_index / max(1, self.spec.horizon)
        noisy = float((state.row, state.col) == self.spec.noisy_tv_cell)
        obs = (
            row,
            col,
            goal_row,
            goal_col,
            dx,
            dy,
            *available,
            progress,
            noisy,
            *self._sensor_channels(state),
        )
        if len(obs) != OBSERVATION_DIM:
            raise AssertionError("observation contract drift")
        return obs

    def simulate(self, state: GridState, action: int) -> tuple[GridState, dict[str, Any]]:
        if state.terminal:
            raise ValueError("cannot step a terminal state")
        row, col = self._candidate_position(state, action)
        moved = (row, col) != (state.row, state.col)
        reached_goal = (row, col) == (state.goal_row, state.goal_col)
        entered_noisy = (row, col) == self.spec.noisy_tv_cell and (
            state.row,
            state.col,
        ) != self.spec.noisy_tv_cell
        left_noisy = (state.row, state.col) == self.spec.noisy_tv_cell and (
            row,
            col,
        ) != self.spec.noisy_tv_cell
        next_step = state.step_index + 1
        horizon_end = next_step >= self.spec.horizon
        terminal = reached_goal or horizon_end
        action_cost = 1.0 + (0.5 if not moved else 0.0) + (0.25 if entered_noisy else 0.0)
        reward = (5.0 if reached_goal else 0.0) - action_cost
        events = ["move" if moved else "blocked"]
        if entered_noisy:
            events.append("entered-noisy-tv")
        if left_noisy:
            events.append("left-noisy-tv")
        if reached_goal:
            events.append("goal-reached")
        if horizon_end and not reached_goal:
            events.append("horizon-ended")
        next_state = GridState(
            episode_index=state.episode_index,
            step_index=next_step,
            row=row,
            col=col,
            goal_row=state.goal_row,
            goal_col=state.goal_col,
            terminal=terminal,
        )
        consequence = {
            "moved": moved,
            "blocked": not moved,
            "goal_reached": reached_goal,
            "entered_noisy_tv": entered_noisy,
            "left_noisy_tv": left_noisy,
            "action_cost": action_cost,
            "reward": reward,
            "terminal_reason": "goal" if reached_goal else ("horizon" if horizon_end else None),
            "events": events,
        }
        return next_state, consequence

    def _branch_record(self, state: GridState, action: int, chosen_action: int) -> dict[str, Any]:
        before_ref = self.state_ref(state)
        next_state, consequence = self.simulate(state, action)
        episode_ref = self.episode_ref(state)
        event_ref = f"{episode_ref}/event:{state.step_index:04d}"
        group_ref = f"counterfactual:{_sha256_value({'world': self.spec.sha256, 'state': before_ref})[:24]}"
        branch_ref = f"{group_ref}/action:{action}"
        return {
            "schema": "mop-action-transition/v1",
            "record_kind": "counterfactual-branch",
            "world_ref": self.world_ref,
            "episode_ref": episode_ref,
            "event_ref": event_ref,
            "entity_refs": self.entity_refs(state),
            "cloned_from_state_ref": before_ref,
            "state_before_ref": before_ref,
            "state_after_ref": self.state_ref(next_state),
            "state_before": state.payload(),
            "state_after": next_state.payload(),
            "observation_before": list(self.observe(state)),
            "observation_after": list(self.observe(next_state)),
            "action": action,
            "action_name": ACTION_NAMES[action],
            "action_was_chosen": action == chosen_action,
            "affordances_before": list(self.affordances(state)),
            "goal_affordances_before": list(self.goal_affordances(state)),
            "consequence": consequence,
            "counterfactual_group_ref": group_ref,
            "counterfactual_branch_ref": branch_ref,
            "episode_start": state.step_index == 0,
            "episode_end": next_state.terminal,
        }

    def step_with_counterfactuals(self, action: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.state is None:
            raise ValueError("reset must be called before step")
        branches = [self._branch_record(self.state, candidate, action) for candidate in range(4)]
        chosen = branches[action]
        actual = dict(chosen)
        actual["record_kind"] = "actual-transition"
        actual["paired_counterfactual_refs"] = [row["counterfactual_branch_ref"] for row in branches]
        actual["actual_branch_ref"] = chosen["counterfactual_branch_ref"]
        self.state = GridState(
            episode_index=int(chosen["state_after"]["episode_index"]),
            step_index=int(chosen["state_after"]["step_index"]),
            row=int(chosen["state_after"]["agent"][0]),
            col=int(chosen["state_after"]["agent"][1]),
            goal_row=int(chosen["state_after"]["goal"][0]),
            goal_col=int(chosen["state_after"]["goal"][1]),
            terminal=bool(chosen["state_after"]["terminal"]),
        )
        return actual, branches


def _select_action(env: PersistentGridEnvironment, state: GridState, policy: str, policy_seed: int) -> int:
    target = (state.goal_row, state.goal_col)
    if policy == "mixed":
        target = (state.goal_row, state.goal_col) if state.episode_index % 2 == 0 else env.spec.noisy_tv_cell
    elif policy == "goal-directed":
        pass
    elif policy == "noisy-tv":
        target = env.spec.noisy_tv_cell
    elif policy == "seeded-random":
        return _stable_index((policy_seed, state.episode_index, state.step_index, env.state_ref(state)), 4)
    else:
        raise ValueError(f"unknown policy {policy!r}")

    scored: list[tuple[int, float, int]] = []
    for action in range(4):
        next_state, consequence = env.simulate(state, action)
        distance = abs(next_state.row - target[0]) + abs(next_state.col - target[1])
        tie = _stable_index((policy_seed, env.state_ref(state), action), 1_000_003)
        scored.append((distance, float(consequence["action_cost"]), tie))
    best = min(scored)
    return scored.index(best)


def collect_trajectory_bundle(
    spec: WorldSpec,
    *,
    episodes: int = 12,
    policy: str = "mixed",
    policy_seed: int | None = None,
) -> dict[str, Any]:
    if episodes < 2:
        raise ValueError("at least two episodes are required")
    policy_seed = spec.seed + 7_919 if policy_seed is None else int(policy_seed)
    env = PersistentGridEnvironment(spec)
    episode_rows: list[dict[str, Any]] = []
    actual: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        start = env.reset(episode_index)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "episode_ref": env.episode_ref(start),
                "initial_state": start.payload(),
                "initial_state_ref": env.state_ref(start),
                "policy": policy,
            }
        )
        while env.state is not None and not env.state.terminal:
            action = _select_action(env, env.state, policy, policy_seed)
            row, branches = env.step_with_counterfactuals(action)
            actual.append(row)
            counterfactuals.extend(branches)

    content: dict[str, Any] = {
        "schema": TRAJECTORY_SCHEMA,
        "scope": "programmatic local mechanics; not natural embodiment or open-ended science",
        "world": spec.payload(),
        "world_sha256": spec.sha256,
        "seed": spec.seed,
        "policy": policy,
        "policy_seed": policy_seed,
        "episode_count": episodes,
        "episodes": episode_rows,
        "transition_count": len(actual),
        "counterfactual_count": len(counterfactuals),
        "actual_transitions": actual,
        "counterfactual_transitions": counterfactuals,
    }
    content["content_sha256"] = _sha256_value(content)
    return content


def verify_trajectory_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    checks["schema"] = bundle.get("schema") == TRAJECTORY_SCHEMA
    try:
        spec = WorldSpec.from_payload(dict(bundle["world"]))
    except (KeyError, TypeError, ValueError) as exc:
        return {"verified": False, "checks": checks, "errors": [f"world: {exc}"]}
    checks["world_hash"] = bundle.get("world_sha256") == spec.sha256
    without_hash = dict(bundle)
    expected_content_hash = without_hash.pop("content_sha256", None)
    checks["content_hash"] = expected_content_hash == _sha256_value(without_hash)
    actual_rows = bundle.get("actual_transitions")
    cf_rows = bundle.get("counterfactual_transitions")
    episode_rows = bundle.get("episodes")
    checks["collections"] = all(isinstance(value, list) for value in (actual_rows, cf_rows, episode_rows))
    if not checks["collections"]:
        return {"verified": False, "checks": checks, "errors": ["trajectory collections malformed"]}
    assert isinstance(actual_rows, list) and isinstance(cf_rows, list) and isinstance(episode_rows, list)
    checks["counts"] = bool(
        bundle.get("episode_count") == len(episode_rows)
        and bundle.get("transition_count") == len(actual_rows)
        and bundle.get("counterfactual_count") == len(cf_rows) == 4 * len(actual_rows)
    )
    cf_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in cf_rows:
        if isinstance(row, dict) and isinstance(row.get("counterfactual_group_ref"), str):
            cf_by_group.setdefault(row["counterfactual_group_ref"], []).append(row)
    checks["four_branches_per_clone"] = bool(
        cf_by_group
        and all(
            len(rows) == 4
            and {int(row["action"]) for row in rows} == set(range(4))
            and sum(bool(row["action_was_chosen"]) for row in rows) == 1
            and len({str(row["cloned_from_state_ref"]) for row in rows}) == 1
            for rows in cf_by_group.values()
        )
    )
    event_refs = [row.get("event_ref") for row in actual_rows if isinstance(row, dict)]
    state_refs = [row.get("state_before_ref") for row in actual_rows if isinstance(row, dict)]
    checks["stable_unique_event_refs"] = len(event_refs) == len(set(event_refs)) == len(actual_rows)
    checks["stable_unique_sequence_states"] = len(state_refs) == len(set(state_refs)) == len(actual_rows)
    checks["required_causal_fields"] = all(
        isinstance(row, dict)
        and len(row.get("observation_before", [])) == OBSERVATION_DIM
        and len(row.get("observation_after", [])) == OBSERVATION_DIM
        and isinstance(row.get("action"), int)
        and isinstance(row.get("consequence"), dict)
        and isinstance(row["consequence"].get("action_cost"), (int, float))
        and len(row.get("affordances_before", [])) == 4
        and isinstance(row.get("entity_refs"), list)
        and bool(row["entity_refs"])
        for row in actual_rows
    )

    replay_ok = True
    replay_cursor = 0
    env = PersistentGridEnvironment(spec)
    for descriptor in episode_rows:
        if not isinstance(descriptor, dict):
            replay_ok = False
            break
        state = env.reset(int(descriptor["episode_index"]))
        if descriptor.get("initial_state") != state.payload() or descriptor.get(
            "initial_state_ref"
        ) != env.state_ref(state):
            replay_ok = False
            break
        episode_actual: list[dict[str, Any]] = []
        while env.state is not None and not env.state.terminal:
            if replay_cursor >= len(actual_rows) or not isinstance(actual_rows[replay_cursor], dict):
                replay_ok = False
                break
            expected = actual_rows[replay_cursor]
            if expected.get("episode_ref") != env.episode_ref(env.state):
                replay_ok = False
                break
            replayed, branches = env.step_with_counterfactuals(int(expected["action"]))
            group = replayed["counterfactual_group_ref"]
            recorded_branches = sorted(cf_by_group.get(group, []), key=lambda row: int(row["action"]))
            if canonical_bytes(replayed) != canonical_bytes(expected) or canonical_bytes(
                branches
            ) != canonical_bytes(recorded_branches):
                replay_ok = False
                break
            episode_actual.append(expected)
            replay_cursor += 1
        if not replay_ok:
            break
        if (
            not episode_actual
            or not episode_actual[0]["episode_start"]
            or not episode_actual[-1]["episode_end"]
        ):
            replay_ok = False
            break
        if any(row["episode_start"] for row in episode_actual[1:]) or any(
            row["episode_end"] for row in episode_actual[:-1]
        ):
            replay_ok = False
            break
    checks["exact_seeded_replay"] = replay_ok and replay_cursor == len(actual_rows)
    checks["episode_boundaries"] = bool(
        actual_rows
        and sum(bool(row.get("episode_start")) for row in actual_rows if isinstance(row, dict))
        == len(episode_rows)
        and sum(bool(row.get("episode_end")) for row in actual_rows if isinstance(row, dict))
        == len(episode_rows)
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {"verified": all(checks.values()), "checks": checks, "errors": errors}


def trajectory_tensors(bundle: dict[str, Any], *, counterfactuals: bool = False) -> dict[str, Any]:
    rows = bundle["counterfactual_transitions" if counterfactuals else "actual_transitions"]
    observations = torch.tensor([row["observation_before"] for row in rows], dtype=torch.float32)
    next_observations = torch.tensor([row["observation_after"] for row in rows], dtype=torch.float32)
    actions = torch.tensor([row["action"] for row in rows], dtype=torch.long)
    return {
        "observation": observations,
        "next_observation": next_observations,
        "action": actions,
        "action_one_hot": F.one_hot(actions, num_classes=4).float(),
        "action_cost": torch.tensor([row["consequence"]["action_cost"] for row in rows]),
        "affordances": torch.tensor([row["affordances_before"] for row in rows], dtype=torch.bool),
        "goal_affordances": torch.tensor([row["goal_affordances_before"] for row in rows], dtype=torch.bool),
        "is_noisy_tv": next_observations[:, 11].bool(),
        "episode_index": torch.tensor([row["state_before"]["episode_index"] for row in rows]),
        "event_refs": [row["event_ref"] for row in rows],
        "entity_refs": [row["entity_refs"] for row in rows],
    }


def bounded_trajectory_contract(*, seed: int, episodes: int = 4, horizon: int = 8) -> dict[str, Any]:
    bundle = collect_trajectory_bundle(make_world_spec(seed=seed, horizon=horizon), episodes=episodes)
    verification = verify_trajectory_bundle(bundle)
    tensors = trajectory_tensors(bundle)
    return {
        "schema": "mop-bounded-action-contract/v1",
        "programmatic_only": True,
        "natural_embodiment_claim": False,
        "world_sha256": bundle["world_sha256"],
        "trajectory_sha256": bundle["content_sha256"],
        "episodes": bundle["episode_count"],
        "transitions": bundle["transition_count"],
        "counterfactuals": bundle["counterfactual_count"],
        "observation_dim": int(tensors["observation"].shape[1]),
        "action_count": 4,
        "has_learnable_rows": bool((~tensors["is_noisy_tv"]).any()),
        "has_noisy_tv_rows": bool(tensors["is_noisy_tv"].any()),
        "verified": verification["verified"],
        "checks": verification["checks"],
    }


def write_trajectory_bundle(path: Path, bundle: dict[str, Any]) -> None:
    verification = verify_trajectory_bundle(bundle)
    if not verification["verified"]:
        raise ValueError(f"refusing to write invalid trajectory bundle: {verification['errors']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
