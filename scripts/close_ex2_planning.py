#!/usr/bin/env python
"""Closure script for the ex2_latent_planning live contention.

CONTENTION: Does ex2 latent planning survive being scored on the TRUE dynamics instead of
its own belief?

The shipped EX2.run scores the MPC planner IN-BELIEF: _mpc_plan returns the terminal distance
of the best candidate as rolled out through the LEARNED dynamics model (_rollout uses `model`),
and _mpc_plan_shuffled does the same. But the flat-reactive baseline is rolled out through the
TRUE synthetic environment (_run_flat_head_trial uses _step_true). So planner and shuffle are
graded against the model's own optimistic self-assessment while the flat head is graded honestly
in the real environment. "Planner beats flat head" partly reflects that mismatch, not planning.

The decisive fix, applied here and ONLY to scoring: let the planner and the shuffle control
SELECT their action sequence exactly as they do today (search through the learned model is
legitimate, that is what planning is), then EXECUTE that selected sequence through _step_true
and measure the REAL terminal distance to goal, identically to how the flat baseline is already
scored. All three are then on the same yardstick (true-env terminal distance). We recompute
goal_success, goal_distance_reduction, planner-vs-flat and planner-vs-shuffle on this honest
scoring, across multiple seeds (the original was single-seed).

This script reuses the shipped experiment's own helpers (imported from
mop.experiments.ex2_latent_planning): _true_dynamics_params, _step_true, _sample_actions,
_DynamicsModel, _train_dynamics, _r2, _rollout, _mpc_plan, _mpc_plan_shuffled, _FlatReactiveHead,
_train_flat_head, _run_flat_head_trial. It does NOT edit the shipped module. It builds the same
ground-truth system, learned model, flat head and planner, and changes ONLY how the planner's and
shuffle's terminal distances are measured.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: .venv/bin/python scripts/close_ex2_planning.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

from mop.experiments.ex2_latent_planning import (
    _DynamicsModel,
    _FlatReactiveHead,
    _mpc_plan,
    _r2,
    _rollout,
    _run_flat_head_trial,
    _sample_actions,
    _step_true,
    _train_dynamics,
    _train_flat_head,
    _true_dynamics_params,
)
from mop.seeding import seed_everything

# ------------------------------------------------------------------------------------------------
# scale (matches configs/experiment/ex2_latent_planning.yaml, trimmed n_trials for a bounded
# multi-seed budget: small MLPs, seconds to a couple of minutes on CPU, nothing encoder-heavy)
# ------------------------------------------------------------------------------------------------
DIM = 32
ACTION_DIM = 8
HIDDEN = 256
EPOCHS = 300
LR = 1e-3
N_TRAIN_TRANSITIONS = 4000
NONLINEARITY = 0.5
NOISE = 0.02
ROLLOUT_K = 8

N_TRIALS = 120  # per seed (config uses 200 single-seed; 120 x 3 keeps it bounded and honest)
HORIZON = 8
N_SHOOTING_SAMPLES = 400
CEM_ITERS = 3
CEM_ELITE_FRAC = 0.1

SUCCESS_FRAC_OF_START = 0.5
PLANNING_MARGIN = 0.05
R2_FLOOR = 0.5

SEEDS = [0, 1, 2]


def _plan_true_terminal_dist(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    true_params,
    g: torch.Generator,
) -> tuple[float, float]:
    """Plan with the learned model (legitimate: search is in-belief), then EXECUTE the selected
    action sequence through the TRUE dynamics and measure the real terminal distance to goal.

    Returns (in_belief_dist, true_env_dist):
      in_belief_dist: what the shipped EX2 reports (planner's own optimistic self-assessment).
      true_env_dist:  the honest, same-yardstick number (executed through _step_true, noise=0.0,
                      exactly as _run_flat_head_trial scores the flat baseline).
    """
    best_seq, in_belief_dist = _mpc_plan(
        model, z0, goal, HORIZON, N_SHOOTING_SAMPLES, ACTION_DIM, CEM_ITERS, CEM_ELITE_FRAC, g
    )
    # execute the SAME selected sequence through the true environment (noise=0.0, matching the flat
    # baseline's _run_flat_head_trial, so planner and flat are graded identically)
    z = z0.clone()
    with torch.no_grad():
        for t in range(HORIZON):
            z = _step_true(
                z.unsqueeze(0), best_seq[t].unsqueeze(0), true_params, NONLINEARITY, 0.0, g
            ).squeeze(0)
    true_env_dist = float((z - goal).norm())
    return in_belief_dist, true_env_dist


def _shuffle_true_terminal_dist(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    true_params,
    g: torch.Generator,
) -> tuple[float, float]:
    """Action-shuffle control, scored on the TRUE dynamics.

    We mirror _mpc_plan_shuffled's selection procedure exactly (sample candidates, permute the
    action assignment across candidates, roll the SHUFFLED sequences through the learned model,
    take the argmin terminal distance in-belief), then execute the SELECTED shuffled sequence
    through _step_true and measure the real terminal distance. This keeps the control's
    information-destroying step (the permutation) intact while grading it on the same true-env
    yardstick as the planner and the flat head.
    """
    cand = torch.randn(N_SHOOTING_SAMPLES, HORIZON, ACTION_DIM, generator=g).clamp(-2.0, 2.0)
    perm = torch.randperm(N_SHOOTING_SAMPLES, generator=g)
    shuffled = cand[perm]
    terminal = _rollout(model, z0, shuffled)  # in-belief, matches _mpc_plan_shuffled
    dist = (terminal - goal.unsqueeze(0)).norm(dim=-1)
    in_belief_dist = float(dist.min())
    best_idx = int(torch.argmin(dist))
    best_seq = shuffled[best_idx]
    # execute the selected shuffled sequence through the TRUE dynamics
    z = z0.clone()
    with torch.no_grad():
        for t in range(HORIZON):
            z = _step_true(
                z.unsqueeze(0), best_seq[t].unsqueeze(0), true_params, NONLINEARITY, 0.0, g
            ).squeeze(0)
    true_env_dist = float((z - goal).norm())
    return in_belief_dist, true_env_dist


def _run_seed(seed: int) -> dict:
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)

    # 1) ground-truth synthetic action-conditioned dynamical system (reused helper)
    true_params = _true_dynamics_params(DIM, ACTION_DIM, NONLINEARITY, g)

    # 2) fit the learned dynamics model g(z,a) -> z' on random transitions (reused helpers)
    z_train = torch.randn(N_TRAIN_TRANSITIONS, DIM, generator=g)
    a_train = _sample_actions(N_TRAIN_TRANSITIONS, ACTION_DIM, g)
    znext_train = _step_true(z_train, a_train, true_params, NONLINEARITY, NOISE, g)
    model = _DynamicsModel(DIM, ACTION_DIM, HIDDEN)
    _train_dynamics(model, z_train, a_train, znext_train, EPOCHS, LR)

    # held-out one-step R2 (the gate's core number, reused helper)
    z_val = torch.randn(max(64, N_TRAIN_TRANSITIONS // 4), DIM, generator=g)
    a_val = _sample_actions(z_val.shape[0], ACTION_DIM, g)
    znext_val = _step_true(z_val, a_val, true_params, NONLINEARITY, NOISE, g)
    with torch.no_grad():
        pred_val = model(z_val, a_val)
    one_step_r2 = _r2(pred_val, znext_val)

    # k-step rollout R2 (learned vs true under the same action sequence, reused pattern)
    k_starts = torch.randn(max(32, N_TRIALS // 4), DIM, generator=g)
    k_actions = torch.randn(k_starts.shape[0], ROLLOUT_K, ACTION_DIM, generator=g).clamp(-2.0, 2.0)
    z_true, z_pred = k_starts.clone(), k_starts.clone()
    with torch.no_grad():
        for t in range(ROLLOUT_K):
            z_true = _step_true(z_true, k_actions[:, t, :], true_params, NONLINEARITY, 0.0, g)
            z_pred = model(z_pred, k_actions[:, t, :])
    kstep_r2 = _r2(z_pred, z_true)

    # 3) flat-reactive-head baseline (reused helper; already scored in the true env)
    flat_head = _FlatReactiveHead(DIM, ACTION_DIM, HIDDEN)
    z0_train = torch.randn(N_TRAIN_TRANSITIONS // 2, DIM, generator=g)
    goal_train = torch.randn(N_TRAIN_TRANSITIONS // 2, DIM, generator=g)
    _train_flat_head(flat_head, model, z0_train, goal_train, ACTION_DIM, EPOCHS, LR, 32, g)

    # 4) goal-reaching trials, scoring planner + shuffle on the TRUE dynamics (the fix)
    planner_true, flat_true, shuffle_true = [], [], []
    planner_belief, shuffle_belief = [], []  # kept only to quantify the in-belief optimism
    planner_success_true, flat_success, planner_success_belief = 0, 0, 0
    for _ in range(N_TRIALS):
        z0 = torch.randn(DIM, generator=g)
        goal = torch.randn(DIM, generator=g)
        start_dist = float((z0 - goal).norm())
        success_threshold = SUCCESS_FRAC_OF_START * start_dist

        p_belief, p_true = _plan_true_terminal_dist(model, z0, goal, true_params, g)
        planner_belief.append(p_belief)
        planner_true.append(p_true)
        if p_true < success_threshold:
            planner_success_true += 1
        if p_belief < success_threshold:
            planner_success_belief += 1

        f_true = _run_flat_head_trial(flat_head, model, z0, goal, HORIZON, true_params, NONLINEARITY, g)
        flat_true.append(f_true)
        if f_true < success_threshold:
            flat_success += 1

        s_belief, s_true = _shuffle_true_terminal_dist(model, z0, goal, true_params, g)
        shuffle_belief.append(s_belief)
        shuffle_true.append(s_true)

    def _mean(v):
        return sum(v) / max(1, len(v))

    mean_planner_true = _mean(planner_true)
    mean_flat_true = _mean(flat_true)
    mean_shuffle_true = _mean(shuffle_true)
    mean_planner_belief = _mean(planner_belief)
    mean_shuffle_belief = _mean(shuffle_belief)

    # honest, same-yardstick comparisons (all three in the TRUE env)
    goal_distance_reduction_true = mean_flat_true - mean_planner_true  # positive = planner closer
    shuffle_gap_true = mean_shuffle_true - mean_planner_true  # positive = real actions beat shuffled
    goal_success_true = planner_success_true / max(1, N_TRIALS)
    flat_success_rate = flat_success / max(1, N_TRIALS)

    rollout_predictable = one_step_r2 > R2_FLOOR and kstep_r2 > R2_FLOOR
    planner_beats_flat_true = (
        goal_distance_reduction_true > PLANNING_MARGIN and (goal_success_true - flat_success_rate) > 0.0
    )
    planner_beats_shuffle_true = shuffle_gap_true > PLANNING_MARGIN

    return {
        "seed": seed,
        "one_step_rollout_r2": round(one_step_r2, 4),
        "kstep_rollout_r2": round(kstep_r2, 4),
        "rollout_predictable": bool(rollout_predictable),
        # honest TRUE-env numbers (the fix)
        "mean_planner_terminal_dist_true": round(mean_planner_true, 4),
        "mean_flat_terminal_dist_true": round(mean_flat_true, 4),
        "mean_shuffle_terminal_dist_true": round(mean_shuffle_true, 4),
        "goal_distance_reduction_true": round(goal_distance_reduction_true, 4),
        "shuffle_gap_true": round(shuffle_gap_true, 4),
        "goal_success_true": round(goal_success_true, 4),
        "flat_head_goal_success_true": round(flat_success_rate, 4),
        "planner_beats_flat_head_true": bool(planner_beats_flat_true),
        "planner_beats_action_shuffle_true": bool(planner_beats_shuffle_true),
        # in-belief numbers (what the shipped EX2 reports), kept to quantify the optimism gap
        "mean_planner_terminal_dist_belief": round(mean_planner_belief, 4),
        "mean_shuffle_terminal_dist_belief": round(mean_shuffle_belief, 4),
        "goal_success_belief": round(planner_success_belief / max(1, N_TRIALS), 4),
        "planner_belief_minus_true": round(mean_planner_true - mean_planner_belief, 4),
    }


def main() -> int:
    t0 = time.time()
    per_seed = []
    for seed in SEEDS:
        print(f"[close_ex2_planning] running seed={seed} ...", file=sys.stderr)
        r = _run_seed(seed)
        per_seed.append(r)
        print(
            f"[close_ex2_planning] seed={seed} TRUE-env: planner={r['mean_planner_terminal_dist_true']} "
            f"flat={r['mean_flat_terminal_dist_true']} shuffle={r['mean_shuffle_terminal_dist_true']} "
            f"| beats_flat={r['planner_beats_flat_head_true']} "
            f"beats_shuffle={r['planner_beats_action_shuffle_true']} "
            f"| belief={r['mean_planner_terminal_dist_belief']} (optimism={r['planner_belief_minus_true']})",
            file=sys.stderr,
        )

    def _agg(key):
        vals = [r[key] for r in per_seed]
        return {
            "mean": round(statistics.mean(vals), 4),
            "stdev": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "per_seed": vals,
        }

    n_beats_flat = sum(1 for r in per_seed if r["planner_beats_flat_head_true"])
    n_beats_shuffle = sum(1 for r in per_seed if r["planner_beats_action_shuffle_true"])
    n_predictable = sum(1 for r in per_seed if r["rollout_predictable"])
    n_seeds = len(per_seed)

    # planning survives honest scoring iff, on the TRUE-env yardstick, it beats BOTH the flat
    # baseline and the shuffle control on EVERY seed (and rollout is predictable, the gate)
    survives = n_beats_flat == n_seeds and n_beats_shuffle == n_seeds and n_predictable == n_seeds

    agg_planner = _agg("mean_planner_terminal_dist_true")
    agg_flat = _agg("mean_flat_terminal_dist_true")
    agg_shuffle = _agg("mean_shuffle_terminal_dist_true")
    agg_red = _agg("goal_distance_reduction_true")
    agg_shuffle_gap = _agg("shuffle_gap_true")
    agg_optimism = _agg("planner_belief_minus_true")

    if survives:
        resolution = "survives"
        verdict = (
            "Planning survives honest scoring: on the TRUE synthetic dynamics the MPC planner still "
            "beats the flat-reactive baseline and the action-shuffle control on terminal distance to "
            "goal, on every seed (in-belief scoring was not the whole story)."
        )
    else:
        # distinguish a full collapse from a partial one
        if n_beats_flat == 0 or n_beats_shuffle == 0:
            resolution = "refuted"
        else:
            resolution = "reframed"
        verdict = (
            "Planning advantage does not hold up under honest same-yardstick scoring: on the TRUE "
            f"dynamics the planner beats the flat baseline on {n_beats_flat}/{n_seeds} seeds and the "
            f"shuffle control on {n_beats_shuffle}/{n_seeds} seeds, so the reported win was at least "
            "partly an in-belief self-assessment artifact."
        )

    out = {
        "id": "ex2_latent_planning",
        "contention": (
            "The planner's terminal distance was scored IN-BELIEF (against its own learned dynamics "
            "model), while the flat baseline was scored in the TRUE synthetic environment, so they "
            "were not on the same yardstick. This closure re-scores the planner AND the shuffle "
            "control on the TRUE dynamics (execute the selected action sequence through _step_true), "
            "grading all three identically."
        ),
        "fix": (
            "Search stays in-belief (legitimate planning). Only scoring changes: the SELECTED action "
            "sequence is executed through _step_true (noise=0.0) and the real terminal distance is "
            "measured, exactly as _run_flat_head_trial already scores the flat baseline."
        ),
        "seeds": SEEDS,
        "n_trials_per_seed": N_TRIALS,
        "scale": {
            "dim": DIM,
            "action_dim": ACTION_DIM,
            "horizon": HORIZON,
            "n_shooting_samples": N_SHOOTING_SAMPLES,
            "cem_iters": CEM_ITERS,
            "hidden": HIDDEN,
            "epochs": EPOCHS,
            "n_train_transitions": N_TRAIN_TRANSITIONS,
            "nonlinearity": NONLINEARITY,
            "noise": NOISE,
        },
        "gates": {
            "success_frac_of_start": SUCCESS_FRAC_OF_START,
            "planning_margin": PLANNING_MARGIN,
            "r2_floor": R2_FLOOR,
        },
        "true_env_terminal_distance": {
            "planner": agg_planner,
            "flat": agg_flat,
            "shuffle": agg_shuffle,
        },
        "true_env_goal_distance_reduction_planner_minus_flat": agg_red,
        "true_env_shuffle_gap_shuffle_minus_planner": agg_shuffle_gap,
        "in_belief_vs_true_optimism_planner": agg_optimism,
        "n_seeds_planner_beats_flat_true": n_beats_flat,
        "n_seeds_planner_beats_shuffle_true": n_beats_shuffle,
        "n_seeds_rollout_predictable": n_predictable,
        "n_seeds": n_seeds,
        "survives": survives,
        "resolution": resolution,
        "verdict": verdict,
        "per_seed": per_seed,
        "wall_clock_s": round(time.time() - t0, 3),
    }

    out_path = Path("runs/pre_studio/close_ex2_planning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[close_ex2_planning] wrote {out_path}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
