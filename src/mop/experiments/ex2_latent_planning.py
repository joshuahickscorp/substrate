
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..seeding import seed_everything
from ..shell.predictor import mlp
from .base import Experiment


def _true_dynamics_params(dim: int, action_dim: int, nonlinearity: float, generator: torch.Generator):
    A = torch.randn(dim, dim, generator=generator) * (0.7 / dim**0.5)
    B = torch.randn(dim, action_dim, generator=generator) * (1.0 / action_dim**0.5)
    C = torch.randn(dim, dim, generator=generator) * (1.0 / dim**0.5)
    D = torch.randn(dim, action_dim, generator=generator) * (1.0 / action_dim**0.5)
    return A, B, C, D


def _step_true(
    z: torch.Tensor, a: torch.Tensor, params, nonlinearity: float, noise: float, g: torch.Generator
):
    A, B, C, D = params
    lin = z @ A.T + a @ B.T
    cross = nonlinearity * torch.tanh(z @ C.T) * (a @ D.T)
    out = lin + cross
    if noise > 0:
        out = out + noise * torch.randn(out.shape, generator=g)
    return out


def _sample_actions(n: int, action_dim: int, g: torch.Generator) -> torch.Tensor:
    return torch.randn(n, action_dim, generator=g).clamp(-2.0, 2.0)


class _DynamicsModel(nn.Module):

    def __init__(self, dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(dim + action_dim, dim, hidden, depth=2, ln=True)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


def _train_dynamics(
    model: _DynamicsModel, z: torch.Tensor, a: torch.Tensor, znext: torch.Tensor, epochs: int, lr: float
) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(z, a)
        loss = F.mse_loss(pred, znext)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum() + 1e-9
    return float(1 - ss_res / ss_tot)


def _rollout(model: _DynamicsModel, z0: torch.Tensor, action_seqs: torch.Tensor) -> torch.Tensor:
    n_cand, horizon, _ = action_seqs.shape
    z = z0.unsqueeze(0).expand(n_cand, -1).clone()
    with torch.no_grad():
        for t in range(horizon):
            z = model(z, action_seqs[:, t, :])
    return z


def _mpc_plan(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_samples: int,
    action_dim: int,
    cem_iters: int,
    cem_elite_frac: float,
    g: torch.Generator,
) -> tuple[torch.Tensor, float]:
    mean = torch.zeros(horizon, action_dim)
    std = torch.ones(horizon, action_dim)
    n_elite = max(1, int(n_samples * cem_elite_frac))
    best_seq: torch.Tensor | None = None
    best_dist = float("inf")
    for _ in range(max(1, cem_iters)):
        noise = torch.randn(n_samples, horizon, action_dim, generator=g)
        cand = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-2.0, 2.0)
        terminal = _rollout(model, z0, cand)
        dist = (terminal - goal.unsqueeze(0)).norm(dim=-1)
        order = torch.argsort(dist)
        elite = cand[order[:n_elite]]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0).clamp(min=1e-3)
        if float(dist[order[0]]) < best_dist:
            best_dist = float(dist[order[0]])
            best_seq = cand[order[0]].clone()
    assert best_seq is not None  # cem_iters >= 1 guarantees at least one assignment
    return best_seq, best_dist


def _mpc_plan_shuffled(
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    n_samples: int,
    action_dim: int,
    g: torch.Generator,
) -> float:
    cand = torch.randn(n_samples, horizon, action_dim, generator=g).clamp(-2.0, 2.0)
    perm = torch.randperm(n_samples, generator=g)
    shuffled = cand[perm]
    terminal = _rollout(model, z0, shuffled)
    dist = (terminal - goal.unsqueeze(0)).norm(dim=-1)
    return float(dist.min())


class _FlatReactiveHead(nn.Module):

    def __init__(self, dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(2 * dim, action_dim, hidden, depth=1, ln=True)

    def forward(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, goal], dim=-1)).clamp(-2.0, 2.0)


def _best_one_step_action(
    model: _DynamicsModel,
    z: torch.Tensor,
    goal: torch.Tensor,
    action_dim: int,
    n_samples: int,
    g: torch.Generator,
) -> torch.Tensor:
    cand = torch.randn(n_samples, action_dim, generator=g).clamp(-2.0, 2.0)
    with torch.no_grad():
        znext = model(z.unsqueeze(0).expand(n_samples, -1), cand)
    dist = (znext - goal.unsqueeze(0)).norm(dim=-1)
    return cand[int(torch.argmin(dist))]


def _train_flat_head(
    head: _FlatReactiveHead,
    model: _DynamicsModel,
    z0s: torch.Tensor,
    goals: torch.Tensor,
    action_dim: int,
    epochs: int,
    lr: float,
    n_samples: int,
    g: torch.Generator,
) -> None:
    targets = torch.stack(
        [
            _best_one_step_action(model, z0s[i], goals[i], action_dim, n_samples, g)
            for i in range(z0s.shape[0])
        ]
    )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = head(z0s, goals)
        F.mse_loss(pred, targets).backward()
        opt.step()


def _run_flat_head_trial(
    head: _FlatReactiveHead,
    model: _DynamicsModel,
    z0: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    true_params,
    nonlinearity: float,
    g: torch.Generator,
) -> float:
    z = z0.clone()
    with torch.no_grad():
        for _ in range(horizon):
            a = head(z.unsqueeze(0), goal.unsqueeze(0)).squeeze(0)
            z = _step_true(z.unsqueeze(0), a.unsqueeze(0), true_params, nonlinearity, 0.0, g).squeeze(0)
    return float((z - goal).norm())


class EX2(Experiment):
    id = "ex2_latent_planning"
    metric = ("goal_success", "goal_distance_reduction", "rollout_error")
    baseline = "flat-reactive-head: single-step policy pi(z, goal) -> a, no rollout or search"
    ablation = "MPC/CEM rollout planner vs flat-reactive-head vs action-shuffled MPC control"
    null_hypothesis = (
        "the learned dynamics does not enable planning the flat shell cannot do, or rollout error is "
        "too high to plan against"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seed_everything(int(e.seed))
        g = torch.Generator().manual_seed(int(e.seed))

        dim = int(e.dim)
        action_dim = int(e.action_dim)
        horizon = int(e.horizon)
        n_trials = int(e.n_trials)
        n_train_transitions = int(e.n_train_transitions)
        n_shooting_samples = int(e.n_shooting_samples)
        cem_iters = int(e.cem_iters)
        cem_elite_frac = float(e.cem_elite_frac)
        nonlinearity = float(e.nonlinearity)
        noise = float(e.noise)
        hidden = int(e.hidden)
        epochs = int(e.epochs)
        lr = float(e.lr)
        success_frac_of_start = float(e.success_frac_of_start)
        planning_margin = float(e.planning_margin)
        r2_floor = float(e.r2_floor)
        rollout_k = int(e.rollout_k)

        t_start = time.time()

        true_params = _true_dynamics_params(dim, action_dim, nonlinearity, g)

        z_train = torch.randn(n_train_transitions, dim, generator=g)
        a_train = _sample_actions(n_train_transitions, action_dim, g)
        znext_train = _step_true(z_train, a_train, true_params, nonlinearity, noise, g)

        model = _DynamicsModel(dim, action_dim, hidden)
        train_losses = _train_dynamics(model, z_train, a_train, znext_train, epochs, lr)

        z_val = torch.randn(max(64, n_train_transitions // 4), dim, generator=g)
        a_val = _sample_actions(z_val.shape[0], action_dim, g)
        znext_val = _step_true(z_val, a_val, true_params, nonlinearity, noise, g)
        with torch.no_grad():
            pred_val = model(z_val, a_val)
        one_step_r2 = _r2(pred_val, znext_val)

        k_starts = torch.randn(max(32, n_trials // 4), dim, generator=g)
        k_actions = torch.randn(k_starts.shape[0], rollout_k, action_dim, generator=g).clamp(-2.0, 2.0)
        z_true, z_pred = k_starts.clone(), k_starts.clone()
        with torch.no_grad():
            for t in range(rollout_k):
                z_true = _step_true(z_true, k_actions[:, t, :], true_params, nonlinearity, 0.0, g)
                z_pred = model(z_pred, k_actions[:, t, :])
        kstep_r2 = _r2(z_pred, z_true)
        rollout_error = 1.0 - kstep_r2  # error framing of the same rollout quality number

        flat_head = _FlatReactiveHead(dim, action_dim, hidden)
        z0_train = torch.randn(n_train_transitions // 2, dim, generator=g)
        goal_train = torch.randn(n_train_transitions // 2, dim, generator=g)
        _train_flat_head(flat_head, model, z0_train, goal_train, action_dim, epochs, lr, n_samples=32, g=g)

        planner_dists, flat_dists, shuffle_dists = [], [], []
        planner_success, flat_success = 0, 0
        for _ in range(n_trials):
            z0 = torch.randn(dim, generator=g)
            goal = torch.randn(dim, generator=g)
            start_dist = float((z0 - goal).norm())
            success_threshold = success_frac_of_start * start_dist

            _, planner_dist = _mpc_plan(
                model, z0, goal, horizon, n_shooting_samples, action_dim, cem_iters, cem_elite_frac, g
            )
            planner_dists.append(planner_dist)
            if planner_dist < success_threshold:
                planner_success += 1

            flat_dist = _run_flat_head_trial(
                flat_head, model, z0, goal, horizon, true_params, nonlinearity, g
            )
            flat_dists.append(flat_dist)
            if flat_dist < success_threshold:
                flat_success += 1

            shuffle_dist = _mpc_plan_shuffled(model, z0, goal, horizon, n_shooting_samples, action_dim, g)
            shuffle_dists.append(shuffle_dist)

        def _mean(v):
            return sum(v) / max(1, len(v))

        goal_success = planner_success / max(1, n_trials)
        flat_success_rate = flat_success / max(1, n_trials)
        mean_planner_dist = _mean(planner_dists)
        mean_flat_dist = _mean(flat_dists)
        mean_shuffle_dist = _mean(shuffle_dists)
        goal_distance_reduction = mean_flat_dist - mean_planner_dist  # positive = planner gets closer
        shuffle_gap = mean_shuffle_dist - mean_planner_dist  # positive = real actions beat shuffled

        rollout_predictable = one_step_r2 > r2_floor and kstep_r2 > r2_floor
        planner_beats_flat = (
            goal_distance_reduction > planning_margin and (goal_success - flat_success_rate) > 0.0
        )
        planner_beats_shuffle = shuffle_gap > planning_margin
        planning_licensed = bool(rollout_predictable and planner_beats_flat and planner_beats_shuffle)

        null_supported = bool((not rollout_predictable) or not (planner_beats_flat and planner_beats_shuffle))

        wall_s = time.time() - t_start

        return {
            "goal_success": round(goal_success, 4),
            "goal_distance_reduction": round(goal_distance_reduction, 4),
            "rollout_error": round(rollout_error, 4),
            "one_step_rollout_r2": round(one_step_r2, 4),
            "kstep_rollout_r2": round(kstep_r2, 4),
            "rollout_k": rollout_k,
            "flat_head_goal_success": round(flat_success_rate, 4),
            "mean_planner_terminal_dist": round(mean_planner_dist, 4),
            "mean_flat_terminal_dist": round(mean_flat_dist, 4),
            "mean_shuffle_terminal_dist": round(mean_shuffle_dist, 4),
            "shuffle_gap": round(shuffle_gap, 4),
            "planner_beats_flat_head": bool(planner_beats_flat),
            "planner_beats_action_shuffle": bool(planner_beats_shuffle),
            "rollout_predictable": bool(rollout_predictable),
            "success_frac_of_start": success_frac_of_start,
            "r2_floor": r2_floor,
            "planning_licensed": planning_licensed,
            "final_dynamics_train_loss": round(train_losses[-1], 6) if train_losses else None,
            "n_trials": n_trials,
            "dim": dim,
            "action_dim": action_dim,
            "horizon": horizon,
            "n_shooting_samples": n_shooting_samples,
            "cem_iters": cem_iters,
            "seed": int(e.seed),
            "wall_clock_s": round(wall_s, 3),
            "null_supported": null_supported,
        }
