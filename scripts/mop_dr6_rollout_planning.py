#!/usr/bin/env python
"""DR6: internal simulation / rollout planning (the ex2 extension), H-ROLLOUT harness, WP-05.

THESIS: simulating futures with an action-conditioned learned dynamics model and acting on the
imagined endpoint beats a reactive policy AND an action-shuffle control at matched compute.

NULL (preregistered, in code before any result): planning ties max(reactive, action-shuffle) at
matched compute. Falsifier: planner true-env terminal distance is not lower than the BEST control
by PLANNING_MARGIN on every seed, or the rollout gate fails (rollout not licensed, EX2 stays weak).

Honest-scoring rule (the close_ex2_planning fix, applied from the start): the planner and the
shuffle SELECT their action sequence through the learned model (search in-belief is legitimate,
that is what planning is) but are SCORED by executing the selected sequence through the TRUE
dynamics, exactly as the reactive arms are scored. All arms share one yardstick.

Compute accounting: ALL rollouts are counted. The planner spends cem_iters * n_samples * horizon
dynamics-model forwards per trial; the matched-compute reactive control (greedy one-step search)
gets the SAME total forward budget per trial (cem_iters * n_samples candidates per step, horizon
steps); the shuffle control spends n_samples * horizon (reported, never more than the planner).
matched_within certifies planner vs greedy.

Controls: action-shuffle (candidate/action correspondence destroyed before rollout), amortized
flat-reactive head (ex2's trained policy, cheap by construction, reported), greedy one-step search
at matched FLOPs (the strong reactive), rollout_gate (diagnostics/sysid licensing gate) plus the
learned model's one-step and k-step R2 floors.

Reuses the shipped ex2 helpers (mop.experiments.ex2_latent_planning), does not edit them.
Writes runs/mot/dr6_rollout_planning.json.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: PYTHONPATH=. .venv/bin/python scripts/mop_dr6_rollout_planning.py --seeds 0-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.diagnostics.sysid import sysid_report
from mop.experiments.base import Experiment
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
# preregistered thresholds (in code before any result exists, per the honesty doctrine)
# ------------------------------------------------------------------------------------------------
PLANNING_MARGIN = 0.05  # planner must beat the BEST control's mean terminal distance by this
R2_FLOOR = 0.5  # learned-model one-step and k-step rollout R2 licensing floor
FLOP_TOL = 0.10  # planner vs greedy matched-compute tolerance (matched_within)
OUT_PATH = Path("runs/mot/dr6_rollout_planning.json")


def default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seeds": [0, 1, 2, 3, 4],
            "dim": 32,
            "action_dim": 8,
            "hidden": 256,
            "epochs": 300,
            "lr": 1e-3,
            "n_train_transitions": 4000,
            "nonlinearity": 0.5,
            "noise": 0.02,
            "n_trials": 100,
            "horizon": 8,
            "n_samples": 300,
            "cem_iters": 3,
            "cem_elite_frac": 0.1,
            "rollout_k": 8,
        }
    )


def parse_seeds(spec: str) -> list[int]:
    """'0-4' -> [0,1,2,3,4]; '0,2,5' -> [0,2,5]; '3' -> [3]."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def execute_true(
    seq: torch.Tensor, z0: torch.Tensor, goal: torch.Tensor, params, nonlin: float, g: torch.Generator
) -> float:
    """Execute an action sequence [H, A] through the TRUE dynamics (noise 0, matching how ex2 scores
    the flat head) and return the real terminal distance to goal: the one honest yardstick."""
    z = z0.clone()
    with torch.no_grad():
        for t in range(seq.shape[0]):
            z = _step_true(z.unsqueeze(0), seq[t].unsqueeze(0), params, nonlin, 0.0, g).squeeze(0)
    return float((z - goal).norm())


def shuffled_plan_seq(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_samples: int,
    action_dim: int,
    g: torch.Generator,
) -> torch.Tensor:
    """Action-shuffle control selection: mirror _mpc_plan_shuffled (permute the candidate/action
    correspondence before rollout, no CEM refit) and return the SELECTED shuffled sequence so it can
    be executed on the true dynamics like every other arm."""
    cand = torch.randn(n_samples, horizon, action_dim, generator=g).clamp(-2.0, 2.0)
    shuffled = cand[torch.randperm(n_samples, generator=g)]
    terminal = _rollout(model, z0, shuffled)
    dist = (terminal - goal.unsqueeze(0)).norm(dim=-1)
    return shuffled[int(torch.argmin(dist))]


def greedy_matched_trial(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_per_step: int,
    action_dim: int,
    params,
    nonlin: float,
    g: torch.Generator,
) -> float:
    """The matched-compute reactive control: at each of `horizon` steps sample n_per_step candidate
    one-step actions, forward each ONCE through the learned model, take the argmin distance to goal,
    execute that single action in the TRUE env. Total model forwards = horizon * n_per_step, set by
    the caller to equal the planner's budget exactly. No lookahead beyond one step: reactive."""
    z = z0.clone()
    with torch.no_grad():
        for _ in range(horizon):
            cand = torch.randn(n_per_step, action_dim, generator=g).clamp(-2.0, 2.0)
            znext = model(z.unsqueeze(0).expand(n_per_step, -1), cand)
            a = cand[int(torch.argmin((znext - goal.unsqueeze(0)).norm(dim=-1)))]
            z = _step_true(z.unsqueeze(0), a.unsqueeze(0), params, nonlin, 0.0, g).squeeze(0)
    return float((z - goal).norm())


def _run_seed(cfg: DictConfig, seed: int) -> dict:
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    dim, adim, hid = int(cfg.dim), int(cfg.action_dim), int(cfg.hidden)
    horizon, n_samples, cem_iters = int(cfg.horizon), int(cfg.n_samples), int(cfg.cem_iters)
    nonlin, noise = float(cfg.nonlinearity), float(cfg.noise)

    params = _true_dynamics_params(dim, adim, nonlin, g)
    z_tr = torch.randn(int(cfg.n_train_transitions), dim, generator=g)
    a_tr = _sample_actions(int(cfg.n_train_transitions), adim, g)
    zn_tr = _step_true(z_tr, a_tr, params, nonlin, noise, g)
    model = _DynamicsModel(dim, adim, hid)
    _train_dynamics(model, z_tr, a_tr, zn_tr, int(cfg.epochs), float(cfg.lr))

    # rollout_gate: linear sysid licensing gate on the SAME transitions (descriptive on the mildly
    # nonlinear generator, but actions-move-state and the Gramian rank are the licensing facts)
    gate = sysid_report(z_tr, a_tr, zn_tr, k=int(cfg.rollout_k), seed=seed)

    # learned-model predictability floors (one-step and k-step open-loop R2)
    n_val = max(64, int(cfg.n_train_transitions) // 4)
    z_val = torch.randn(n_val, dim, generator=g)
    a_val = _sample_actions(n_val, adim, g)
    zn_val = _step_true(z_val, a_val, params, nonlin, noise, g)
    with torch.no_grad():
        one_step_r2 = _r2(model(z_val, a_val), zn_val)
    k = int(cfg.rollout_k)
    ks = torch.randn(max(16, int(cfg.n_trials) // 4), dim, generator=g)
    ka = torch.randn(ks.shape[0], k, adim, generator=g).clamp(-2.0, 2.0)
    z_true, z_pred = ks.clone(), ks.clone()
    with torch.no_grad():
        for t in range(k):
            z_true = _step_true(z_true, ka[:, t, :], params, nonlin, 0.0, g)
            z_pred = model(z_pred, ka[:, t, :])
    kstep_r2 = _r2(z_pred, z_true)
    rollout_licensed = bool(one_step_r2 > R2_FLOOR and kstep_r2 > R2_FLOOR)

    # amortized flat-reactive head (ex2's policy baseline, trained through the learned model)
    flat = _FlatReactiveHead(dim, adim, hid)
    n_flat = int(cfg.n_train_transitions) // 2
    _train_flat_head(
        flat,
        model,
        torch.randn(n_flat, dim, generator=g),
        torch.randn(n_flat, dim, generator=g),
        adim,
        int(cfg.epochs),
        float(cfg.lr),
        32,
        g,
    )

    # compute accounting: ALL rollouts counted, per trial, in dynamics-model forwards and FLOPs
    fwd_flops = mlp_flops([dim + adim, hid, hid, dim])  # _DynamicsModel is a depth-2 mlp
    planner_fwds = cem_iters * n_samples * horizon
    n_per_step = cem_iters * n_samples  # greedy budget per step -> horizon * n_per_step == planner
    greedy_fwds = horizon * n_per_step
    shuffle_fwds = n_samples * horizon
    match = matched_within(planner_fwds * fwd_flops, greedy_fwds * fwd_flops, tol=FLOP_TOL)

    planner_d, flat_d, greedy_d, shuffle_d = [], [], [], []
    for _ in range(int(cfg.n_trials)):
        z0 = torch.randn(dim, generator=g)
        goal = torch.randn(dim, generator=g)
        seq, _ = _mpc_plan(model, z0, goal, horizon, n_samples, adim, cem_iters, float(cfg.cem_elite_frac), g)
        planner_d.append(execute_true(seq, z0, goal, params, nonlin, g))
        flat_d.append(_run_flat_head_trial(flat, model, z0, goal, horizon, params, nonlin, g))
        greedy_d.append(greedy_matched_trial(model, z0, goal, horizon, n_per_step, adim, params, nonlin, g))
        sseq = shuffled_plan_seq(model, z0, goal, horizon, n_samples, adim, g)
        shuffle_d.append(execute_true(sseq, z0, goal, params, nonlin, g))

    def _mean(v: list[float]) -> float:
        return sum(v) / max(1, len(v))

    mp, mf, mg, ms = _mean(planner_d), _mean(flat_d), _mean(greedy_d), _mean(shuffle_d)
    best_control = min(mf, mg, ms)
    advantage = best_control - mp  # positive = planner strictly closer than every control
    return {
        "seed": seed,
        "rollout_gate": gate,
        "one_step_r2": round(one_step_r2, 4),
        "kstep_r2": round(kstep_r2, 4),
        "rollout_licensed": rollout_licensed,
        "mean_terminal_dist_true": {
            "planner": round(mp, 4),
            "flat_reactive": round(mf, 4),
            "greedy_matched": round(mg, 4),
            "action_shuffle": round(ms, 4),
        },
        "best_control": round(best_control, 4),
        "planning_advantage": round(advantage, 4),
        "planner_beats_best_control": bool(advantage > PLANNING_MARGIN),
        "compute": {
            "dyn_forward_flops": fwd_flops,
            "planner_forwards_per_trial": planner_fwds,
            "greedy_forwards_per_trial": greedy_fwds,
            "shuffle_forwards_per_trial": shuffle_fwds,
            "planner_vs_greedy_matched": match,
        },
    }


class DR6RolloutPlanning(Experiment):
    id = "mop_dr6_rollout_planning"
    metric = ("true_env_terminal_distance", "planning_advantage", "rollout_gate")
    baseline = "max(flat-reactive head, greedy one-step search at matched FLOPs)"
    ablation = "action-shuffle control (candidate/action correspondence destroyed before rollout)"
    null_hypothesis = (
        "Planning ties max(reactive, action-shuffle) at matched compute: rollouts carry no usable "
        "future info beyond one step (planner advantage <= PLANNING_MARGIN on the true-env yardstick)."
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        t0 = time.time()
        per_seed = [_run_seed(cfg, int(s)) for s in cfg.seeds]
        advantages = [r["planning_advantage"] for r in per_seed]
        n_beats = sum(1 for r in per_seed if r["planner_beats_best_control"])
        n_licensed = sum(1 for r in per_seed if r["rollout_licensed"])
        n_seeds = len(per_seed)
        survives = n_beats == n_seeds and n_licensed == n_seeds
        out = {
            "contract": self.contract(),
            "preregistered": {
                "planning_margin": PLANNING_MARGIN,
                "r2_floor": R2_FLOOR,
                "flop_tol": FLOP_TOL,
                "survival_rule": "planner beats the BEST control by the margin on EVERY seed and the "
                "rollout gate is licensed on every seed",
            },
            "cfg": OmegaConf.to_container(cfg),
            "per_seed": per_seed,
            "advantage_ci": seed_ci(advantages),
            "advantage_sign_flips": sign_flip_report(advantages),
            "n_seeds_planner_beats_best_control": n_beats,
            "n_seeds_rollout_licensed": n_licensed,
            "n_seeds": n_seeds,
            "survives": survives,
            "null_supported": not survives,
            "wall_clock_s": round(time.time() - t0, 3),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dr6_rollout_planning.json").write_text(json.dumps(out, indent=2))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DR6 rollout planning (honest true-env scoring)")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    cfg = default_cfg()
    cfg.seeds = parse_seeds(args.seeds)
    out_path = Path(args.out)
    result = DR6RolloutPlanning().run(cfg, resolve("cpu"), out_path.parent)
    written = out_path.parent / "dr6_rollout_planning.json"
    if out_path != written:
        written.rename(out_path)
    print(f"[dr6] wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: result[k] for k in ("survives", "null_supported", "advantage_ci")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
