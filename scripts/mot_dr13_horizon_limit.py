#!/usr/bin/env python
"""DR13: planning-horizon limit, H-ROLLOUT harness, WP-05.

THESIS: learned-model rollout error compounds with horizon, so open-loop planning through the
model beats a matched-compute reactive policy at short horizons and stops beating it beyond a
crossover horizon: internal simulation has a bounded useful horizon on this substrate.

NULL (preregistered): planning never beats reactive at ANY horizon (contradicts ex2) OR beats it
only at horizon 1 (one-step lookahead, not really planning).

Per-horizon arms, all scored by executing in the TRUE dynamics (one yardstick):
  planner: open-loop random-shooting plan of length H through the learned model (n_samples
    candidates, n_samples * H model forwards), the selected sequence executed in the true env.
  reactive (matched compute): greedy one-step search re-planned at every step from the TRUE
    current state, n_samples candidates per step, H steps, so H * n_samples forwards: exactly the
    planner's budget. The reactive arm sees the true state at every step (its structural advantage)
    but never looks ahead; the planner looks ahead but commits open-loop (its structural cost).
    Error compounding is exactly the trade between those two.
The learned model's open-loop k-step R2 decay curve is reported alongside (the mechanism), plus
the diagnostics/sysid rollout_gate on the training transitions.

Reuses ex2 helpers, does not edit them. Writes runs/mot/dr13_horizon_limit.json.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: PYTHONPATH=. .venv/bin/python scripts/mot_dr13_horizon_limit.py --seeds 0-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from devsys.devices import DeviceInfo, resolve
from devsys.diagnostics.compute import matched_within, mlp_flops
from devsys.diagnostics.riskcov import seed_ci
from devsys.diagnostics.sysid import sysid_report
from devsys.experiments.base import Experiment
from devsys.experiments.ex2_latent_planning import (
    _DynamicsModel,
    _r2,
    _rollout,
    _sample_actions,
    _step_true,
    _train_dynamics,
    _true_dynamics_params,
)
from devsys.seeding import seed_everything

# ------------------------------------------------------------------------------------------------
# preregistered thresholds (in code before any result exists)
# ------------------------------------------------------------------------------------------------
PLANNING_MARGIN = 0.05  # planning "wins" at H iff reactive_mean - planner_mean > this
R2_FLOOR = 0.5  # one-step licensing floor for the learned model
FLOP_TOL = 0.10
OUT_PATH = Path("runs/mot/dr13_horizon_limit.json")


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
            "noise": 0.05,  # train AND execution process noise (the same world throughout)
            "horizons": [1, 2, 4, 8, 12, 16],
            "n_trials": 40,
            "n_samples": 200,
        }
    )


def parse_seeds(spec: str) -> list[int]:
    """'0-4' -> [0,1,2,3,4]; '0,2,5' -> [0,2,5]; '3' -> [3]."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def plan_open_loop(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_samples: int,
    action_dim: int,
    g: torch.Generator,
) -> torch.Tensor:
    """Plain random shooting (cem_iters=1 so the per-horizon budget is exactly n_samples * H
    forwards, linear in H, matching the reactive arm construction). Returns the selected [H, A]."""
    cand = torch.randn(n_samples, horizon, action_dim, generator=g).clamp(-2.0, 2.0)
    terminal = _rollout(model, z0, cand)
    dist = (terminal - goal.unsqueeze(0)).norm(dim=-1)
    return cand[int(torch.argmin(dist))]


def execute_open_loop(
    seq: torch.Tensor,
    z0: torch.Tensor,
    goal: torch.Tensor,
    params,
    nonlin: float,
    noise: float,
    g: torch.Generator,
) -> float:
    z = z0.clone()
    with torch.no_grad():
        for t in range(seq.shape[0]):
            z = _step_true(z.unsqueeze(0), seq[t].unsqueeze(0), params, nonlin, noise, g).squeeze(0)
    return float((z - goal).norm())


def reactive_matched_trial(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_samples: int,
    action_dim: int,
    params,
    nonlin: float,
    noise: float,
    g: torch.Generator,
) -> float:
    """Greedy one-step search from the TRUE state at every step, n_samples candidates per step:
    H * n_samples model forwards, the planner's exact budget."""
    z = z0.clone()
    with torch.no_grad():
        for _ in range(horizon):
            cand = torch.randn(n_samples, action_dim, generator=g).clamp(-2.0, 2.0)
            znext = model(z.unsqueeze(0).expand(n_samples, -1), cand)
            a = cand[int(torch.argmin((znext - goal.unsqueeze(0)).norm(dim=-1)))]
            z = _step_true(z.unsqueeze(0), a.unsqueeze(0), params, nonlin, noise, g).squeeze(0)
    return float((z - goal).norm())


def kstep_r2_curve(
    model: _DynamicsModel,
    params,
    nonlin: float,
    horizons: list[int],
    dim: int,
    adim: int,
    n: int,
    g: torch.Generator,
) -> dict[str, float]:
    """Open-loop model-vs-true R2 at each horizon under the same random action sequences (noise 0
    on the true side: this isolates MODEL error compounding, the mechanism DR13 is about)."""
    hmax = max(horizons)
    z0 = torch.randn(n, dim, generator=g)
    acts = torch.randn(n, hmax, adim, generator=g).clamp(-2.0, 2.0)
    z_true, z_pred = z0.clone(), z0.clone()
    curve: dict[str, float] = {}
    with torch.no_grad():
        for t in range(hmax):
            z_true = _step_true(z_true, acts[:, t, :], params, nonlin, 0.0, g)
            z_pred = model(z_pred, acts[:, t, :])
            if (t + 1) in horizons:
                curve[str(t + 1)] = round(_r2(z_pred, z_true), 4)
    return curve


def _run_seed(cfg: DictConfig, seed: int) -> dict:
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    dim, adim, hid = int(cfg.dim), int(cfg.action_dim), int(cfg.hidden)
    n_samples = int(cfg.n_samples)
    nonlin, noise = float(cfg.nonlinearity), float(cfg.noise)
    horizons = [int(h) for h in cfg.horizons]

    params = _true_dynamics_params(dim, adim, nonlin, g)
    z_tr = torch.randn(int(cfg.n_train_transitions), dim, generator=g)
    a_tr = _sample_actions(int(cfg.n_train_transitions), adim, g)
    zn_tr = _step_true(z_tr, a_tr, params, nonlin, noise, g)
    model = _DynamicsModel(dim, adim, hid)
    _train_dynamics(model, z_tr, a_tr, zn_tr, int(cfg.epochs), float(cfg.lr))

    gate = sysid_report(z_tr, a_tr, zn_tr, k=max(horizons), seed=seed)
    n_val = max(64, int(cfg.n_train_transitions) // 4)
    z_val = torch.randn(n_val, dim, generator=g)
    a_val = _sample_actions(n_val, adim, g)
    zn_val = _step_true(z_val, a_val, params, nonlin, noise, g)
    with torch.no_grad():
        one_step_r2 = _r2(model(z_val, a_val), zn_val)
    decay = kstep_r2_curve(model, params, nonlin, horizons, dim, adim, max(32, int(cfg.n_trials)), g)

    fwd_flops = mlp_flops([dim + adim, hid, hid, dim])
    per_horizon = {}
    for h in horizons:
        pd, rd = [], []
        for _ in range(int(cfg.n_trials)):
            z0 = torch.randn(dim, generator=g)
            goal = torch.randn(dim, generator=g)
            seq = plan_open_loop(model, z0, goal, h, n_samples, adim, g)
            pd.append(execute_open_loop(seq, z0, goal, params, nonlin, noise, g))
            rd.append(reactive_matched_trial(model, z0, goal, h, n_samples, adim, params, nonlin, noise, g))
        mp, mr = sum(pd) / len(pd), sum(rd) / len(rd)
        per_horizon[str(h)] = {
            "planner_mean_dist_true": round(mp, 4),
            "reactive_mean_dist_true": round(mr, 4),
            "planning_margin": round(mr - mp, 4),  # positive = planning wins at this horizon
            "planning_wins": bool(mr - mp > PLANNING_MARGIN),
            "compute": {
                "planner_forwards": n_samples * h,
                "reactive_forwards": n_samples * h,
                "matched": matched_within(n_samples * h * fwd_flops, n_samples * h * fwd_flops, tol=FLOP_TOL),
            },
        }

    wins = [h for h in horizons if per_horizon[str(h)]["planning_wins"]]
    losses = [h for h in horizons if not per_horizon[str(h)]["planning_wins"]]
    crossover_h = max(wins) if wins else 0
    return {
        "seed": seed,
        "rollout_gate": gate,
        "one_step_r2": round(one_step_r2, 4),
        "model_licensed": bool(one_step_r2 > R2_FLOOR),
        "kstep_r2_decay": decay,
        "per_horizon": per_horizon,
        "winning_horizons": wins,
        "crossover_horizon": crossover_h,
        "crossover_observed": bool(wins and any(h > max(wins) for h in losses)),
        "nontrivial_win": bool(any(h > 1 for h in wins)),
    }


class DR13HorizonLimit(Experiment):
    id = "mot_dr13_horizon_limit"
    metric = ("true_env_terminal_distance_vs_horizon", "crossover_horizon", "kstep_r2_decay")
    baseline = "greedy one-step reactive search at matched FLOPs, re-planned per step, per horizon"
    ablation = "horizon sweep (the planner's open-loop commitment length is the ablated variable)"
    null_hypothesis = (
        "Planning never beats reactive at any horizon (contradicts ex2) OR beats it only at "
        "horizon 1 (one-step lookahead, not planning): there is no nontrivial useful horizon."
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        t0 = time.time()
        per_seed = [_run_seed(cfg, int(s)) for s in cfg.seeds]
        n_seeds = len(per_seed)
        n_nontrivial = sum(1 for r in per_seed if r["nontrivial_win"])
        n_licensed = sum(1 for r in per_seed if r["model_licensed"])
        crossovers = [float(r["crossover_horizon"]) for r in per_seed]
        survives = n_nontrivial == n_seeds and n_licensed == n_seeds
        out = {
            "contract": self.contract(),
            "preregistered": {
                "planning_margin": PLANNING_MARGIN,
                "r2_floor": R2_FLOOR,
                "flop_tol": FLOP_TOL,
                "survival_rule": "planning beats matched reactive at some horizon > 1 on EVERY seed "
                "and the model clears the R2 floor on every seed; the crossover "
                "horizon (largest winning H) is the reported bound",
            },
            "cfg": OmegaConf.to_container(cfg),
            "per_seed": per_seed,
            "crossover_ci": seed_ci(crossovers),
            "n_seeds_nontrivial_win": n_nontrivial,
            "n_seeds_model_licensed": n_licensed,
            "n_seeds_crossover_observed": sum(1 for r in per_seed if r["crossover_observed"]),
            "n_seeds": n_seeds,
            "survives": survives,
            "null_supported": not survives,
            "wall_clock_s": round(time.time() - t0, 3),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dr13_horizon_limit.json").write_text(json.dumps(out, indent=2))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DR13 planning-horizon limit (matched-compute sweep)")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    cfg = default_cfg()
    cfg.seeds = parse_seeds(args.seeds)
    out_path = Path(args.out)
    result = DR13HorizonLimit().run(cfg, resolve("cpu"), out_path.parent)
    written = out_path.parent / "dr13_horizon_limit.json"
    if out_path != written:
        written.rename(out_path)
    print(f"[dr13] wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: result[k] for k in ("survives", "null_supported", "crossover_ci")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
