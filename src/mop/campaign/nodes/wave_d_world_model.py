"""Wave D science node: does an action-conditioned world model earn its keep for prediction AND planning?

Wave D asks a prediction and agency question on independent deterministic gridworlds. The candidate is a
compact, action-conditioned tabular transition map fit on observed (state, action, next_state) triples. The
named control is the same-shape transition table trained on action-shuffled data: the action column is
permuted so the label carries no consequence, which makes the control action-blind at matched parameter
count. A second, equal-parameter reactive control (a memoryless Manhattan-greedy policy that uses no learned
model) is the planning baseline.

Two things are measured per unit: held-out one-step transition error, and whether greedy planning with the
learned model reaches the goal in fewer steps than the reactive policy. The paired per-unit delta for the
exact sign-flip is control_error minus candidate_error, so a positive delta favors the action-conditioned
model. Crucially a prediction win is not promoted unless planning also improves: a model can match marginal
next-state statistics without the action carrying transferable consequence, so the verdict is gated on
measured goal-reaching, never on prediction alone. A tie, a wrong-direction delta, or a prediction win with
no planning improvement is a legitimate null.

Determinism: every random choice is seeded from ctx.seed via the framework rng. No wall clock enters the
sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from mop.campaign.nodes.framework import (
    LifecycleCost,
    assert_matched_budget,
    exact_sign_flip_one_sided,
    honest_envelope,
    rng,
    verdict_from,
)
from mop.campaign.runners import NodeContext, RunResult, register_runner

# Four-connected moves: up, down, left, right. The action label is what the control shuffles away.
_ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _idx(cell: tuple[int, int], cols: int) -> int:
    return cell[0] * cols + cell[1]


def _build_gridworld(gen: np.random.Generator, rows: int, cols: int) -> dict[str, Any]:
    """A planning-required world: a vertical wall with one gap separates a left start from a right goal.

    The wall class is a fixed structural choice, not a per-unit tune toward a positive: the gap row and the
    start and goal rows vary per unit, so the geometry a reactive policy must cope with differs each unit.
    """

    wall_col = cols // 2
    gap_row = int(gen.integers(0, rows))
    obstacles = frozenset((r, wall_col) for r in range(rows) if r != gap_row)
    start = (int(gen.integers(0, rows)), 0)
    goal = (int(gen.integers(0, rows)), cols - 1)
    return {"rows": rows, "cols": cols, "obstacles": obstacles, "start": start, "goal": goal}


def _step(cell: tuple[int, int], action: int, world: dict[str, Any]) -> tuple[int, int]:
    """Deterministic true dynamics: move unless the target is out of bounds or an obstacle, else stay."""

    dr, dc = _ACTIONS[action]
    nr, nc = cell[0] + dr, cell[1] + dc
    if 0 <= nr < world["rows"] and 0 <= nc < world["cols"] and (nr, nc) not in world["obstacles"]:
        return (nr, nc)
    return cell


def _free_cells(world: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (r, c) for r in range(world["rows"]) for c in range(world["cols"]) if (r, c) not in world["obstacles"]
    ]


def _collect(world: dict[str, Any], walk_len: int, gen: np.random.Generator) -> list[tuple[int, int, int]]:
    """Exploration data: one short random walk restarted from every free cell for even coverage."""

    cols = world["cols"]
    data: list[tuple[int, int, int]] = []
    for start in _free_cells(world):
        cell = start
        for _ in range(walk_len):
            a = int(gen.integers(0, len(_ACTIONS)))
            nxt = _step(cell, a, world)
            data.append((_idx(cell, cols), a, _idx(nxt, cols)))
            cell = nxt
    return data


def _fit_table(data: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    """A compact majority-vote transition map keyed by (state, action). Deterministic tie-break by index."""

    counts: dict[tuple[int, int], dict[int, int]] = {}
    for s, a, s2 in data:
        bucket = counts.setdefault((s, a), {})
        bucket[s2] = bucket.get(s2, 0) + 1
    table: dict[tuple[int, int], int] = {}
    for key, dist in counts.items():
        table[key] = min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return table


def _shuffle_actions(
    data: list[tuple[int, int, int]], gen: np.random.Generator
) -> list[tuple[int, int, int]]:
    """The named control: permute the action column so the action label is decoupled from the outcome."""

    actions = np.array([a for _, a, _ in data], dtype=np.int64)
    shuffled = actions[gen.permutation(len(actions))]
    return [(s, int(shuffled[i]), s2) for i, (s, _, s2) in enumerate(data)]


def _predict(table: dict[tuple[int, int], int], s: int, a: int) -> int:
    return table.get((s, a), s)  # fallback: predict "stay" for an unseen (state, action) pair


def _one_step_error(table: dict[tuple[int, int], int], test: list[tuple[int, int, int]]) -> float:
    if not test:
        return 0.0
    wrong = sum(1 for s, a, s2 in test if _predict(table, s, a) != s2)
    return wrong / len(test)


def _plan_actions(table: dict[tuple[int, int], int], start_s: int, goal_s: int) -> list[int] | None:
    """Breadth-first planning over the LEARNED transition graph. Returns an action list or None."""

    prev: dict[int, tuple[int | None, int | None]] = {start_s: (None, None)}
    queue: deque[int] = deque([start_s])
    while queue:
        s = queue.popleft()
        if s == goal_s:
            break
        for a in range(len(_ACTIONS)):
            ns = _predict(table, s, a)
            if ns not in prev:
                prev[ns] = (s, a)
                queue.append(ns)
    if goal_s not in prev:
        return None
    actions: list[int] = []
    cur = goal_s
    while prev[cur][0] is not None:
        ps, pa = prev[cur]
        if pa is None or ps is None:
            break
        actions.append(int(pa))
        cur = int(ps)
    actions.reverse()
    return actions


def _execute_plan(actions: list[int] | None, world: dict[str, Any], max_steps: int) -> tuple[int, bool]:
    """Execute a planned action list in the TRUE environment. Model error can derail an executed plan."""

    if actions is None:
        return max_steps, False
    cell = world["start"]
    for steps, a in enumerate(actions, start=1):
        cell = _step(cell, a, world)
        if cell == world["goal"]:
            return steps, True
        if steps >= max_steps:
            return steps, False
    return min(len(actions), max_steps), cell == world["goal"]


def _reactive_rollout(world: dict[str, Any], max_steps: int) -> tuple[int, bool]:
    """The equal-parameter reactive control: memoryless Manhattan-greedy, deterministic index tie-break."""

    goal = world["goal"]
    cell = world["start"]
    for steps in range(1, max_steps + 1):
        best_a, best_d = 0, None
        for a in range(len(_ACTIONS)):
            nxt = _step(cell, a, world)
            dist = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            if best_d is None or dist < best_d:
                best_d, best_a = dist, a
        cell = _step(cell, best_a, world)
        if cell == goal:
            return steps, True
    return max_steps, False


def _run_unit(unit: int, base_seed: int, rows: int, cols: int, walk_len: int, n_test: int) -> dict[str, Any]:
    """One independent gridworld: fit both transition arms, score prediction, and score planning."""

    world = _build_gridworld(rng(base_seed, "env", unit), rows, cols)
    train = _collect(world, walk_len, rng(base_seed, "train", unit))
    test = _collect(world, max(1, n_test // len(_free_cells(world)) + 1), rng(base_seed, "test", unit))
    test = test[:n_test]

    candidate_table = _fit_table(train)
    control_table = _fit_table(_shuffle_actions(train, rng(base_seed, "shuffle", unit)))

    candidate_error = _one_step_error(candidate_table, test)
    control_error = _one_step_error(control_table, test)

    max_steps = 2 * rows * cols
    start_s = _idx(world["start"], cols)
    goal_s = _idx(world["goal"], cols)
    plan = _plan_actions(candidate_table, start_s, goal_s)
    cand_steps, cand_ok = _execute_plan(plan, world, max_steps)
    react_steps, react_ok = _reactive_rollout(world, max_steps)

    return {
        "unit_id": f"gridworld_{unit:02d}",
        "gap_row": int([r for r in range(rows) if (r, cols // 2) not in world["obstacles"]][0]),
        "candidate_error": round(float(candidate_error), 12),
        "control_error": round(float(control_error), 12),
        "prediction_delta": round(float(control_error - candidate_error), 12),
        "candidate_steps": int(cand_steps),
        "reactive_steps": int(react_steps),
        "planning_improvement": int(react_steps - cand_steps),
        "candidate_reaches_goal": bool(cand_ok),
        "reactive_reaches_goal": bool(react_ok),
        "covered_state_action_pairs": int(len(candidate_table)),
    }


@register_runner("wave_d.action_conditioned_transition")
def wave_d_world_model_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """Wave D: an action-conditioned world model must beat an action-shuffled table on one-step prediction
    AND let greedy planning outrun a reactive policy before it can be called anything more than correlational.
    """

    n_units = int(params.get("n_units", 12))
    rows = int(params.get("rows", 7))
    cols = int(params.get("cols", 7))
    walk_len = int(params.get("walk_len", 40))
    n_test = int(params.get("n_test", 400))
    n_units = max(7, min(n_units, 22))  # keep the exact sign-flip enumerable and meaningful

    units = [_run_unit(u, ctx.seed, rows, cols, walk_len, n_test) for u in range(n_units)]

    deltas = [u["prediction_delta"] for u in units]
    sign_flip = exact_sign_flip_one_sided(deltas)
    sesoi = round(1.0 / (rows * cols), 12)  # one cell of the state space: a small structural effect floor

    improvements = [u["planning_improvement"] for u in units]
    mean_improvement = round(float(np.mean(improvements)), 12) if improvements else 0.0
    n_planning_favorable = int(sum(1 for x in improvements if x > 0))
    candidate_success_rate = round(float(np.mean([u["candidate_reaches_goal"] for u in units])), 12)
    reactive_success_rate = round(float(np.mean([u["reactive_reaches_goal"] for u in units])), 12)
    planning_gate = mean_improvement > 0.0 and n_planning_favorable >= math.ceil(n_units / 2)

    base_verdict = verdict_from(sign_flip["mean_delta"], sign_flip["one_sided_p"], sesoi)
    verdict = "survives" if (base_verdict == "survives" and planning_gate) else "null"
    is_null = verdict != "survives"

    n_free = rows * cols - (rows - 1)  # one gap in a full column
    train_n = n_free * walk_len
    params_per_arm = rows * cols * len(_ACTIONS)
    matched_budget = assert_matched_budget(
        {
            "action_conditioned": LifecycleCost(train_flops=train_n, inference_flops=n_test),
            "action_shuffled": LifecycleCost(train_flops=train_n, inference_flops=n_test),
        },
        ceiling=train_n + n_test,
    )

    content = {
        **honest_envelope(
            ctx.node_id,
            "mop-campaign-wave_d_world_model/v1",
            {
                "form_family": "action",
                "phenomenon": "prediction_planning",
                "mechanism_family": "action_conditioned_transition_model",
                "unit_class": "deterministic_gridworld",
                "evidence_level": "M1",
            },
        ),
        "design": {
            "n_units": n_units,
            "grid": {"rows": rows, "cols": cols},
            "walk_len": walk_len,
            "n_test": n_test,
            "parameters_per_transition_arm": params_per_arm,
            "matched_budget": matched_budget,
        },
        "control_description": (
            "Named control is the same-shape (state, action) transition table fit on action-shuffled data: "
            "the action column is permuted so the action label is decoupled from the observed next state, "
            "yielding an action-blind model at matched parameter count. A second equal-parameter reactive "
            "control is a memoryless Manhattan-greedy policy that plans with no learned model at all."
        ),
        "per_unit": units,
        "prediction": {
            "delta_definition": "control_error minus candidate_error; positive favors action-conditioned",
            "deltas": deltas,
            "sign_flip": sign_flip,
            "sesoi": sesoi,
            "sesoi_rationale": "one cell of the state space, 1/(rows*cols); below this a prediction gain "
            "is scientifically negligible",
        },
        "planning": {
            "improvement_definition": "reactive_steps minus candidate_steps; positive favors model planning",
            "improvements": improvements,
            "mean_improvement": mean_improvement,
            "n_planning_favorable": n_planning_favorable,
            "candidate_success_rate": candidate_success_rate,
            "reactive_success_rate": reactive_success_rate,
            "planning_gate_passed": bool(planning_gate),
            "gate_rule": "promotion requires mean_improvement > 0 and a majority of units improving; a "
            "prediction win alone is never promoted",
        },
        "base_verdict_prediction_only": base_verdict,
        "verdict": verdict,
        "is_null": is_null,
        "alternative_explanation": (
            "Correlational next-state prediction without usable action consequence: an action-shuffled or "
            "action-blind table can match marginal next-state statistics on covered states, so a lower "
            "one-step error need not mean the action label carries transferable consequence. That is why a "
            "prediction win is gated on measured goal-reaching improvement before any promotion."
        ),
        "failure_domain": (
            "Stochastic or partially observed dynamics a compact deterministic tabular map cannot capture, "
            "or obstacle-free worlds where a memoryless reactive policy already reaches the goal so planning "
            "has no headroom to demonstrate the model's value."
        ),
    }

    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=is_null,
        detail={
            "mean_prediction_delta": sign_flip["mean_delta"],
            "one_sided_p": sign_flip["one_sided_p"],
            "mean_planning_improvement": mean_improvement,
            "planning_gate_passed": bool(planning_gate),
        },
    )
