"""The previously registry-only F-series lanes that do not require unavailable hardware.

F6 and F15 use deterministic, locally generated intervention environments. They are not presented as
embodiment: their purpose is to test whether action and consequence Forms are load-bearing under exact
action-shuffle and action-blind controls. F7 reuses the project's growable-head mechanism and compares
developmental growth with both fixed-final capacity and a random growth schedule at identical final
parameters and update counts. F11 tests a deliberately small Form-token generator against stored-token,
raw-exemplar, and random-generator replay while pricing every arm by actual retained bytes.

F8 and F16 are different. Natural-data claims require licensed data, real encoder weights and inherited
features, and prior control receipts. Their default mode is a tiny mechanics-only smoke test. Scientific
mode fails closed when evidence is absent or inconsistent, then runs every compute-matched arm when the
package is valid. A fixture can validate that engine, but its provenance taint is irreversible and it can
never become a natural-data or promotable result.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import mlp_flops, param_count
from ..diagnostics.performance_density import density_block
from ..diagnostics.riskcov import seed_ci, sign_flip_report
from ..environments import bounded_trajectory_contract
from ..seeding import seed_everything
from ..shell.buffer import ReplayBuffer
from ..substrate.datasets import make_task_stream
from ..substrate.form import FormMeta, TensorFormAdapter, build_form_matrix, form_audit
from .b8_structural_growth import _GrowableHead
from .base import Experiment, _mean
from .form_rewrite_engine import ScientificExecutionRefused, run_gated_rewrite

__all__ = ("ScientificExecutionRefused", "F6", "F7", "F8", "F11", "F15", "F16")


def _fit_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    out_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
    classification: bool,
) -> nn.Sequential:
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = F.cross_entropy(pred, y.long()) if classification else F.mse_loss(pred, y.float())
        loss.backward()
        opt.step()
    return model


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    residual = float(((pred.detach() - target.detach()) ** 2).sum())
    centered = float(((target.detach() - target.detach().mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - residual / max(centered, 1.0e-12)


def _state_form(states: torch.Tensor, grid: int, dim: int, seed: int) -> torch.Tensor:
    """A deterministic state Form whose first two coordinates remain spatially interpretable."""
    xy = 2.0 * states.float() / max(1, grid - 1) - 1.0
    base = torch.cat(
        [
            xy,
            xy**2,
            torch.sin(math.pi * xy),
            torch.cos(math.pi * xy),
            (xy[:, :1] * xy[:, 1:2]),
        ],
        dim=1,
    )
    if dim < 2:
        raise ValueError("state Form dim must be at least two")
    if dim == 2:
        return xy
    g = torch.Generator().manual_seed(seed + 101)
    projection = torch.randn(base.shape[1], dim - 2, generator=g) / math.sqrt(base.shape[1])
    return torch.cat([xy, torch.tanh(base @ projection)], dim=1)


def _grid_step(states: torch.Tensor, actions: torch.Tensor, grid: int) -> torch.Tensor:
    delta = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long)
    return (states.long() + delta[actions.long()]).clamp(0, grid - 1)


def _action_form_audit(obs: torch.Tensor, actions: torch.Tensor, labels: torch.Tensor) -> dict:
    refs = [f"transition:{i:05d}" for i in range(obs.shape[0])]
    one_hot = F.one_hot(actions, num_classes=4).float()
    adapters = [
        TensorFormAdapter(
            FormMeta(
                tag="observation_state",
                kind="symbolic",
                feature_dim=obs.shape[1],
                source="deterministic-gridworld",
                objective="programmatic",
                referent_scheme="transition-id",
            ),
            obs,
            refs,
            factors={"next_state": labels},
        ),
        TensorFormAdapter(
            FormMeta(
                tag="action_trace",
                kind="control",
                feature_dim=4,
                source="deterministic-gridworld",
                objective="programmatic",
                referent_scheme="transition-id",
            ),
            one_hot,
            refs,
            factors={"next_state": labels},
        ),
    ]
    return form_audit(build_form_matrix(adapters), require_controls=False)


class F6(Experiment):
    id = "f6_sensorimotor_form_closure"
    metric = ("rollout_r2", "goal_success", "action_shuffle_delta")
    baseline = "action-blind and action-shuffled transition models with identical architecture and updates"
    ablation = "true action Form vs zeroed action Form vs referent-shuffled action Form"
    null_hypothesis = (
        "true action-conditioned Form closure fails to beat both action-blind and action-shuffled "
        "controls on held-out rollout prediction and deterministic goal reachability"
    )
    tier = "cpu-now"

    @staticmethod
    def _rollout_r2(
        model: nn.Module,
        *,
        grid: int,
        dim: int,
        horizon: int,
        n_rollouts: int,
        seed: int,
        action_mode: str,
    ) -> float:
        g = torch.Generator().manual_seed(seed + 303)
        state = torch.randint(0, grid, (n_rollouts, 2), generator=g)
        pred_form = _state_form(state, grid, dim, seed)
        true_state = state.clone()
        preds, targets = [], []
        for _ in range(horizon):
            action = torch.randint(0, 4, (n_rollouts,), generator=g)
            true_state = _grid_step(true_state, action, grid)
            target = _state_form(true_state, grid, dim, seed)
            if action_mode == "blind":
                supplied = torch.zeros(n_rollouts, 4)
            else:
                supplied = F.one_hot(action, num_classes=4).float()
            pred_form = model(torch.cat([pred_form, supplied], dim=1))
            preds.append(pred_form)
            targets.append(target)
        return _r2(torch.cat(preds), torch.cat(targets))

    @staticmethod
    def _goal_success(
        model: nn.Module,
        *,
        grid: int,
        dim: int,
        trials: int,
        max_steps: int,
        seed: int,
        action_mode: str,
    ) -> float:
        g = torch.Generator().manual_seed(seed + 707)
        successes = 0
        for _ in range(trials):
            state = torch.randint(0, grid, (1, 2), generator=g)
            goal = torch.randint(0, grid, (1, 2), generator=g)
            while bool(torch.equal(state, goal)):
                goal = torch.randint(0, grid, (1, 2), generator=g)
            goal_form = _state_form(goal, grid, dim, seed)
            for _step in range(max_steps):
                state_form = _state_form(state, grid, dim, seed).repeat(4, 1)
                if action_mode == "blind":
                    supplied = torch.zeros(4, 4)
                else:
                    supplied = torch.eye(4)
                with torch.no_grad():
                    predicted_next = model(torch.cat([state_form, supplied], dim=1))
                action = int(((predicted_next[:, :2] - goal_form[:, :2]) ** 2).sum(1).argmin())
                state = _grid_step(state, torch.tensor([action]), grid)
                if bool(torch.equal(state, goal)):
                    successes += 1
                    break
        return successes / max(1, trials)

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        del device, run_dir
        e = cfg.experiment
        seeds = [int(s) for s in e.seeds]
        grid, dim, hidden = int(e.grid), int(e.form_dim), int(e.hidden)
        epochs, lr = int(e.epochs), float(e.lr)
        rollout: dict[str, list[float]] = {"true": [], "blind": [], "shuffled": []}
        goals: dict[str, list[float]] = {"true": [], "blind": [], "shuffled": []}
        params = 0
        n_train = 0
        audit: dict[str, Any] = {}
        for seed in seeds:
            seed_everything(seed)
            states = torch.tensor(
                [(r, c) for r in range(grid) for c in range(grid) for _a in range(4)],
                dtype=torch.long,
            )
            actions = torch.tensor(
                [a for _r in range(grid) for _c in range(grid) for a in range(4)], dtype=torch.long
            )
            states = states.repeat_interleave(int(e.repeats), dim=0)
            actions = actions.repeat_interleave(int(e.repeats), dim=0)
            g = torch.Generator().manual_seed(seed + 11)
            obs = _state_form(states, grid, dim, seed)
            nxt_state = _grid_step(states, actions, grid)
            target = _state_form(nxt_state, grid, dim, seed)
            obs = obs + float(e.sensor_noise) * torch.randn(obs.shape, generator=g)
            target = target + float(e.sensor_noise) * torch.randn(target.shape, generator=g)
            perm = torch.randperm(obs.shape[0], generator=g)
            cut = int(obs.shape[0] * float(e.train_frac))
            tr, te = perm[:cut], perm[cut:]
            action_one_hot = F.one_hot(actions, num_classes=4).float()
            shuffled_actions = action_one_hot[torch.randperm(obs.shape[0], generator=g)]
            inputs = {
                "true": torch.cat([obs, action_one_hot], dim=1),
                "blind": torch.cat([obs, torch.zeros_like(action_one_hot)], dim=1),
                "shuffled": torch.cat([obs, shuffled_actions], dim=1),
            }
            models: dict[str, nn.Module] = {}
            for arm, x in inputs.items():
                models[arm] = _fit_mlp(
                    x[tr],
                    target[tr],
                    out_dim=dim,
                    hidden=hidden,
                    epochs=epochs,
                    lr=lr,
                    seed=seed + 101,
                    classification=False,
                )
                with torch.no_grad():
                    rollout[arm].append(_r2(models[arm](x[te]), target[te]))
                goals[arm].append(
                    self._goal_success(
                        models[arm],
                        grid=grid,
                        dim=dim,
                        trials=int(e.goal_trials),
                        max_steps=int(e.max_goal_steps),
                        seed=seed,
                        action_mode="blind" if arm == "blind" else "true",
                    )
                )
            # The autoregressive rollout is the harder dynamic check and replaces the one-step score.
            for arm, model in models.items():
                rollout[arm][-1] = self._rollout_r2(
                    model,
                    grid=grid,
                    dim=dim,
                    horizon=int(e.rollout_horizon),
                    n_rollouts=int(e.rollout_trials),
                    seed=seed,
                    action_mode="blind" if arm == "blind" else "true",
                )
            params = sum(param_count(m) for m in models.values())
            n_train = len(tr)
            audit = _action_form_audit(obs, actions, nxt_state)

        mean_rollout = {k: _mean(v) for k, v in rollout.items()}
        mean_goals = {k: _mean(v) for k, v in goals.items()}
        strongest_r2 = max(mean_rollout["blind"], mean_rollout["shuffled"])
        strongest_goal = max(mean_goals["blind"], mean_goals["shuffled"])
        deltas = [
            min(
                rollout["true"][i] - max(rollout["blind"][i], rollout["shuffled"][i]),
                goals["true"][i] - max(goals["blind"][i], goals["shuffled"][i]),
            )
            for i in range(len(seeds))
        ]
        margin = float(e.margin)
        action_shuffle_delta = mean_rollout["true"] - mean_rollout["shuffled"]
        rejects = bool(
            mean_rollout["true"] > strongest_r2 + margin and mean_goals["true"] > strongest_goal + margin
        )
        total_flops = len(seeds) * epochs * 3 * mlp_flops([dim + 4, hidden, dim], batch=n_train)
        return {
            "rollout_r2": round(mean_rollout["true"], 4),
            "goal_success": round(mean_goals["true"], 4),
            "action_shuffle_delta": round(action_shuffle_delta, 4),
            "rollout_r2_by_arm": {k: round(v, 4) for k, v in mean_rollout.items()},
            "goal_success_by_arm": {k: round(v, 4) for k, v in mean_goals.items()},
            "strongest_control_rollout_r2": round(strongest_r2, 4),
            "strongest_control_goal_success": round(strongest_goal, 4),
            "matched_architecture": True,
            "matched_updates": True,
            "form_audit": audit,
            "seeds": seeds,
            "seed_ci": seed_ci(deltas),
            "sign_flip": sign_flip_report(deltas),
            "null_supported": not rejects,
            "environment_contract": bounded_trajectory_contract(
                seed=seeds[0], episodes=4, horizon=max(6, int(e.rollout_horizon) + 2)
            ),
            "density": density_block(
                {"rollout_r2": mean_rollout["true"]},
                params=params,
                flops=total_flops,
                updates=len(seeds) * epochs * 3,
            ),
        }


def _curriculum_frontier(history: list[list[float]]) -> float:
    points = []
    for task_index, accuracies in enumerate(history):
        adaptation = accuracies[task_index]
        retention = _mean(accuracies[:task_index]) if task_index else adaptation
        points.append(2.0 * adaptation * retention / max(adaptation + retention, 1.0e-12))
    return _mean(points)


def _train_growth_arm(
    tasks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    mode: str,
    dim: int,
    classes: int,
    w_init: int,
    w_final: int,
    grow_add: int,
    epochs_per_task: int,
    lr: float,
    min_progress: float,
    seed: int,
    random_events: int | None = None,
) -> tuple[float, float, nn.Module, list[int], float]:
    width = w_final if mode == "fixed" else w_init
    gen = torch.Generator().manual_seed(seed + 919)
    head = _GrowableHead(dim, width, classes, gen)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    total_steps = len(tasks) * epochs_per_task
    if mode == "random":
        count = max(1, int(random_events or math.ceil((w_final - w_init) / grow_add)))
        rg = torch.Generator().manual_seed(seed + 929)
        candidates = torch.arange(1, max(2, total_steps - 1))
        selected = candidates[torch.randperm(len(candidates), generator=rg)[:count]]
        random_schedule = set(int(v) for v in selected.tolist())
    else:
        random_schedule = set()
    history: list[list[float]] = []
    growth_steps: list[int] = []
    active_params = 0.0
    global_step = 0
    recent_losses: list[float] = []
    needed_events = math.ceil(max(0, w_final - head.width) / max(1, grow_add))
    for task_index, (xtr, ytr, _xte, _yte) in enumerate(tasks):
        for _epoch in range(epochs_per_task):
            global_step += 1
            opt.zero_grad()
            loss = F.cross_entropy(head(xtr), ytr)
            loss.backward()
            opt.step()
            recent_losses.append(float(loss.detach()))
            active_params += param_count(head)
            should_grow = False
            if head.width < w_final and mode == "developmental" and len(recent_losses) >= 6:
                early = _mean(recent_losses[-6:-3])
                late = _mean(recent_losses[-3:])
                relative_progress = (early - late) / max(abs(early), 1.0e-8)
                steps_left = total_steps - global_step
                force = steps_left <= needed_events * 3
                should_grow = relative_progress < min_progress or force
            elif head.width < w_final and mode == "random":
                should_grow = global_step in random_schedule
            if should_grow:
                head.grow(min(grow_add, w_final - head.width))
                opt = torch.optim.Adam(head.parameters(), lr=lr)
                growth_steps.append(global_step)
                recent_losses.clear()
                needed_events = math.ceil(max(0, w_final - head.width) / max(1, grow_add))
        with torch.no_grad():
            history.append(
                [
                    float((head(tasks[j][2]).argmax(-1) == tasks[j][3]).float().mean())
                    for j in range(task_index + 1)
                ]
            )
    # Final capacity is an invariant, not an outcome. Any unscheduled remainder is installed only after
    # the last scored update, making the comparison conservative while keeping end-state parameters exact.
    if head.width < w_final:
        head.grow(w_final - head.width)
        growth_steps.append(total_steps)
    frontier = _curriculum_frontier(history)
    retained = _mean(history[-1][:-1]) if len(history[-1]) > 1 else history[-1][0]
    return frontier, retained, head, growth_steps, active_params / max(1, total_steps)


class F7(Experiment):
    id = "f7_developmental_form_growth"
    metric = ("frontier_auc_gain", "growth_efficiency", "retained_capacity")
    baseline = "fixed-final-capacity and random-growth controls at identical final parameters and updates"
    ablation = "learning-progress-triggered structural growth vs random growth timing vs fixed-final width"
    null_hypothesis = (
        "learning-progress growth fails to beat the stronger fixed-final or random-growth control on the "
        "adaptation-retention frontier at matched final parameters"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        del device, run_dir
        e = cfg.experiment
        seeds = [int(s) for s in e.seeds]
        dim, classes = int(e.dim), int(e.classes)
        frontiers: dict[str, list[float]] = {"developmental": [], "fixed_final": [], "random": []}
        retained: dict[str, list[float]] = {"developmental": [], "fixed_final": [], "random": []}
        growth_steps: dict[str, list[list[int]]] = {"developmental": [], "random": []}
        final_params: dict[str, list[int]] = {"developmental": [], "fixed_final": [], "random": []}
        active_params: dict[str, list[float]] = {"developmental": [], "fixed_final": [], "random": []}
        for seed in seeds:
            raw_tasks = make_task_stream(
                n_tasks=int(e.tasks),
                dim=dim,
                classes_per_task=classes,
                samples_per_task=int(e.samples_per_task),
                separation=float(e.separation),
                incremental="domain",
                seed=seed,
            )
            tasks = []
            for task in raw_tasks:
                cut = int(task.x.shape[0] * float(e.train_frac))
                tasks.append((task.x[:cut], task.y[:cut], task.x[cut:], task.y[cut:]))

            def run_arm(
                mode: str,
                task_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                arm_seed: int,
                random_events: int | None = None,
            ) -> tuple[float, float, nn.Module, list[int], float]:
                return _train_growth_arm(
                    task_data,
                    mode=mode,
                    dim=dim,
                    classes=classes,
                    w_init=int(e.w_init),
                    w_final=int(e.w_final),
                    grow_add=int(e.grow_add),
                    epochs_per_task=int(e.epochs_per_task),
                    lr=float(e.lr),
                    min_progress=float(e.min_progress),
                    seed=arm_seed,
                    random_events=random_events,
                )

            dev = run_arm("developmental", tasks, seed)
            fixed = run_arm("fixed", tasks, seed)
            random = run_arm("random", tasks, seed, random_events=len(dev[3]))
            for name, result in (("developmental", dev), ("fixed_final", fixed), ("random", random)):
                frontiers[name].append(result[0])
                retained[name].append(result[1])
                final_params[name].append(param_count(result[2]))
                active_params[name].append(result[4])
            growth_steps["developmental"].append(dev[3])
            growth_steps["random"].append(random[3])

        means = {name: _mean(values) for name, values in frontiers.items()}
        strongest = max(means["fixed_final"], means["random"])
        gain = means["developmental"] - strongest
        capacity_matched = (
            final_params["developmental"] == final_params["fixed_final"] == final_params["random"]
        )
        updates_matched = True
        dev_eff = means["developmental"] / max(_mean(active_params["developmental"]), 1.0)
        fixed_eff = means["fixed_final"] / max(_mean(active_params["fixed_final"]), 1.0)
        efficiency_ratio = dev_eff / max(fixed_eff, 1.0e-12)
        deltas = [
            frontiers["developmental"][i] - max(frontiers["fixed_final"][i], frontiers["random"][i])
            for i in range(len(seeds))
        ]
        rejects = capacity_matched and updates_matched and gain > float(e.margin)
        total_updates = len(seeds) * int(e.tasks) * int(e.epochs_per_task) * 3
        final_param_total = sum(v[-1] for v in final_params.values())
        return {
            "frontier_auc_gain": round(gain, 4),
            "growth_efficiency": round(efficiency_ratio, 4),
            "retained_capacity": round(_mean(retained["developmental"]), 4),
            "frontier_auc_by_arm": {k: round(v, 4) for k, v in means.items()},
            "retention_by_arm": {k: round(_mean(v), 4) for k, v in retained.items()},
            "growth_steps": growth_steps,
            "final_params": final_params,
            "average_active_params": {k: round(_mean(v), 2) for k, v in active_params.items()},
            "capacity_matched": capacity_matched,
            "updates_matched": updates_matched,
            "seeds": seeds,
            "seed_ci": seed_ci(deltas),
            "sign_flip": sign_flip_report(deltas),
            "null_supported": not rejects,
            "density": density_block(
                {"frontier_auc_gain": gain},
                params=final_param_total,
                updates=total_updates,
            ),
        }


class _DiagonalFormGenerator:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, classes: int):
        self.mean = torch.stack([x[y == c].mean(0) for c in range(classes)])
        self.std = torch.stack([x[y == c].std(0, unbiased=False).clamp_min(1.0e-3) for c in range(classes)])

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, self.mean.shape[0], (n,), generator=generator)
        noise = torch.randn(n, self.mean.shape[1], generator=generator)
        return self.mean[y] + self.std[y] * noise, y

    @property
    def bytes(self) -> int:
        return int((self.mean.numel() + self.std.numel()) * self.mean.element_size())


def _form_task(
    *, samples: int, classes: int, dim: int, separation: float, seed: int, domain: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 1009 * domain)
    y = torch.arange(samples) % classes
    y = y[torch.randperm(samples, generator=g)]
    centers = torch.randn(classes, dim, generator=g) * separation
    mode = torch.where(torch.arange(samples) % 2 == 0, -1.0, 1.0).unsqueeze(1)
    mode_dir = torch.randn(classes, dim, generator=g) * 0.7
    x = centers[y] + mode * mode_dir[y] + 0.45 * torch.randn(samples, dim, generator=g)
    return x.float(), y.long()


def _train_replay_classifier(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    replay: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    classes: int,
    hidden: int,
    epochs0: int,
    epochs1: int,
    lr: float,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(x0.shape[1], hidden), nn.GELU(), nn.Linear(hidden, classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs0):
        opt.zero_grad()
        F.cross_entropy(model(x0), y0).backward()
        opt.step()
    for _ in range(epochs1):
        opt.zero_grad()
        loss = F.cross_entropy(model(x1), y1)
        if replay is not None:
            loss = loss + F.cross_entropy(model(replay[0]), replay[1])
        loss.backward()
        opt.step()
    return model


def _manifold_validity(
    generated_x: torch.Tensor,
    generated_y: torch.Tensor,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    classes: int,
) -> float:
    centroids = torch.stack([real_x[real_y == c].mean(0) for c in range(classes)])
    nearest_label = torch.cdist(generated_x, centroids).argmin(1)
    thresholds = []
    for c in range(classes):
        xc = real_x[real_y == c]
        pair = torch.cdist(xc, xc)
        pair.fill_diagonal_(float("inf"))
        thresholds.append(torch.quantile(pair.min(1).values, 0.95))
    threshold = torch.stack(thresholds)[generated_y]
    nearest_real = torch.empty(generated_x.shape[0])
    for c in range(classes):
        mask = generated_y == c
        if bool(mask.any()):
            nearest_real[mask] = torch.cdist(generated_x[mask], real_x[real_y == c]).min(1).values
    valid = (nearest_label == generated_y) & (nearest_real <= threshold)
    return float(valid.float().mean())


class F11(Experiment):
    id = "f11_form_dream_replay"
    metric = ("retention_per_byte", "manifold_validity", "generated_vs_stored_gap")
    baseline = "stored Form-token and raw-exemplar replay under the same byte and replay-sample budgets"
    ablation = "diagonal Form generator vs stored tokens, raw exemplars, no replay, and random generator"
    null_hypothesis = (
        "generated Form replay falls outside the preregistered retention band of stored replay at the "
        "same memory ceiling or fails the held-out Form-manifold validity floor"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        del device, run_dir
        e = cfg.experiment
        seeds = [int(s) for s in e.seeds]
        classes, dim = int(e.classes), int(e.form_dim)
        memory_bytes, replay_samples = int(e.memory_bytes), int(e.replay_samples)
        retention: dict[str, list[float]] = {
            "stored_form": [],
            "raw_exemplar": [],
            "generated": [],
            "random_generator": [],
            "no_replay": [],
        }
        validity, random_validity = [], []
        actual_bytes: dict[str, list[int]] = {k: [] for k in retention if k != "no_replay"}
        params = 0
        for seed in seeds:
            x0, y0 = _form_task(
                samples=int(e.samples),
                classes=classes,
                dim=dim,
                separation=float(e.separation),
                seed=seed,
                domain=0,
            )
            x1, y1 = _form_task(
                samples=int(e.samples),
                classes=classes,
                dim=dim,
                separation=float(e.separation),
                seed=seed,
                domain=1,
            )
            cut = int(x0.shape[0] * float(e.train_frac))
            x0tr, y0tr, x0te, y0te = x0[:cut], y0[:cut], x0[cut:], y0[cut:]
            x1tr, y1tr = x1[:cut], y1[:cut]
            token_item_bytes = dim * x0.element_size() + y0.element_size()
            stored_count = min(len(x0tr), max(1, memory_bytes // token_item_bytes))
            buffer = ReplayBuffer(capacity=stored_count, dim=dim, prioritized=False, seed=seed)
            buffer.add(x0tr, y0tr)
            stored = buffer.sample(replay_samples)
            stored_replay = (stored["x"], stored["y"])
            actual_bytes["stored_form"].append(stored_count * token_item_bytes)

            raw_dim = int(e.raw_dim)
            raw_item_bytes = raw_dim * x0.element_size() + y0.element_size()
            raw_count = min(len(x0tr), max(1, memory_bytes // raw_item_bytes))
            rg = torch.Generator().manual_seed(seed + 401)
            raw_noise = torch.randn(len(x0tr), raw_dim - dim, generator=rg)
            raw = torch.cat([x0tr, raw_noise], dim=1)[:raw_count]
            raw_y = y0tr[:raw_count]
            pick = torch.randint(0, raw_count, (replay_samples,), generator=rg)
            raw_replay = (raw[pick, :dim], raw_y[pick])
            actual_bytes["raw_exemplar"].append(raw_count * raw_item_bytes)

            generator = _DiagonalFormGenerator(x0tr, y0tr, classes)
            generated = generator.sample(replay_samples, torch.Generator().manual_seed(seed + 501))
            actual_bytes["generated"].append(generator.bytes)
            validity.append(_manifold_validity(*generated, x0te, y0te, classes))

            random_generator = _DiagonalFormGenerator(x0tr, y0tr, classes)
            random_generator.mean = torch.randn(
                random_generator.mean.shape, generator=torch.Generator().manual_seed(seed + 601)
            )
            random_generator.std = torch.ones_like(random_generator.std)
            random_replay = random_generator.sample(replay_samples, torch.Generator().manual_seed(seed + 701))
            actual_bytes["random_generator"].append(random_generator.bytes)
            random_validity.append(_manifold_validity(*random_replay, x0te, y0te, classes))

            arms = {
                "stored_form": stored_replay,
                "raw_exemplar": raw_replay,
                "generated": generated,
                "random_generator": random_replay,
                "no_replay": None,
            }
            for name, replay in arms.items():
                model = _train_replay_classifier(
                    x0tr,
                    y0tr,
                    x1tr,
                    y1tr,
                    replay,
                    classes=classes,
                    hidden=int(e.hidden),
                    epochs0=int(e.epochs_task0),
                    epochs1=int(e.epochs_task1),
                    lr=float(e.lr),
                    seed=seed + 801,
                )
                retention[name].append(_accuracy(model, x0te, y0te))
                params = max(params, param_count(model))

        means = {k: _mean(v) for k, v in retention.items()}
        bytes_mean = {k: _mean([float(x) for x in v]) for k, v in actual_bytes.items()}
        gap = means["generated"] - means["stored_form"]
        valid = _mean(validity)
        retention_per_byte = means["generated"] / max(bytes_mean["generated"], 1.0)
        deltas = [retention["generated"][i] - retention["stored_form"][i] for i in range(len(seeds))]
        matches_stored = gap >= -float(e.retention_band)
        manifold_ok = valid >= float(e.manifold_floor)
        total_updates = len(seeds) * 5 * (int(e.epochs_task0) + int(e.epochs_task1))
        return {
            "retention_per_byte": retention_per_byte,
            "manifold_validity": round(valid, 4),
            "generated_vs_stored_gap": round(gap, 4),
            "retention_by_arm": {k: round(v, 4) for k, v in means.items()},
            "bytes_by_arm": {k: round(v, 1) for k, v in bytes_mean.items()},
            "retention_per_byte_by_arm": {k: means[k] / max(v, 1.0) for k, v in bytes_mean.items()},
            "random_generator_manifold_validity": round(_mean(random_validity), 4),
            "replay_samples_matched": True,
            "memory_ceiling_bytes": memory_bytes,
            "seeds": seeds,
            "seed_ci": seed_ci(deltas),
            "sign_flip": sign_flip_report(deltas),
            "null_supported": not (matches_stored and manifold_ok),
            "density": density_block(
                {"retention_per_byte": retention_per_byte},
                params=params,
                bytes=bytes_mean["generated"],
                updates=total_updates,
            ),
        }


def _affordance_dataset(
    *, samples: int, actions: int, obs_dim: int, train: bool, seed: int, confound: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + (0 if train else 10_000))
    y = torch.arange(samples) % actions
    y = y[torch.randperm(samples, generator=g)]
    if train:
        follows = torch.rand(samples, generator=g) < confound
        random_proxy = torch.randint(0, actions, (samples,), generator=g)
        proxy = torch.where(follows, y, random_proxy)
    else:
        # At test time appearance is independent rather than adversarially guaranteed wrong. The passive
        # arm should fall to chance under shortcut removal, not below chance because of a label inversion.
        proxy = torch.randint(0, actions, (samples,), generator=g)
    appearance = F.one_hot(proxy, num_classes=actions).float()
    projection = torch.randn(actions, obs_dim, generator=torch.Generator().manual_seed(seed + 17))
    obs = appearance @ projection + 0.35 * torch.randn(samples, obs_dim, generator=g)
    outcomes = F.one_hot(y, num_classes=actions).float()
    outcomes = outcomes + 0.03 * torch.randn(outcomes.shape, generator=g)
    return obs, outcomes, y.long()


class F15(Experiment):
    id = "f15_embodied_affordance_form"
    metric = ("affordance_decode_acc", "intervention_gain", "action_shuffle_delta")
    baseline = "passive observation and action-shuffled consequence Forms with identical heads and updates"
    ablation = "paired observation-action-outcome Form vs passive padded Form vs shuffled action outcomes"
    null_hypothesis = (
        "consequence-conditioned Form tokens fail to beat both passive observation and action-shuffled "
        "controls on held-out affordance decoding and action selection"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        del device, run_dir
        e = cfg.experiment
        seeds = [int(s) for s in e.seeds]
        actions, obs_dim = int(e.actions), int(e.obs_dim)
        scores: dict[str, list[float]] = {"intervention": [], "passive": [], "action_shuffle": []}
        params = 0
        audit: dict[str, Any] = {}
        for seed in seeds:
            obs_tr, outcomes_tr, ytr = _affordance_dataset(
                samples=int(e.train_samples),
                actions=actions,
                obs_dim=obs_dim,
                train=True,
                seed=seed,
                confound=float(e.appearance_confound),
            )
            obs_te, outcomes_te, yte = _affordance_dataset(
                samples=int(e.test_samples),
                actions=actions,
                obs_dim=obs_dim,
                train=False,
                seed=seed,
                confound=float(e.appearance_confound),
            )
            g = torch.Generator().manual_seed(seed + 211)
            train_perm = torch.randperm(len(ytr), generator=g)
            test_perm = torch.randperm(len(yte), generator=g)
            inputs = {
                "intervention": (torch.cat([obs_tr, outcomes_tr], 1), torch.cat([obs_te, outcomes_te], 1)),
                "passive": (
                    torch.cat([obs_tr, torch.zeros_like(outcomes_tr)], 1),
                    torch.cat([obs_te, torch.zeros_like(outcomes_te)], 1),
                ),
                "action_shuffle": (
                    torch.cat([obs_tr, outcomes_tr[train_perm]], 1),
                    torch.cat([obs_te, outcomes_te[test_perm]], 1),
                ),
            }
            models = {}
            for name, (xtr, xte) in inputs.items():
                model = _fit_mlp(
                    xtr,
                    ytr,
                    out_dim=actions,
                    hidden=int(e.hidden),
                    epochs=int(e.epochs),
                    lr=float(e.lr),
                    seed=seed + 301,
                    classification=True,
                )
                models[name] = model
                scores[name].append(_accuracy(model, xte, yte))
            params = sum(param_count(v) for v in models.values())
            refs = [f"object:{i:05d}" for i in range(len(ytr))]
            matrix = build_form_matrix(
                [
                    TensorFormAdapter(
                        FormMeta(
                            tag="observation",
                            kind="symbolic",
                            feature_dim=obs_dim,
                            source="deterministic-affordance-world",
                            objective="programmatic",
                            referent_scheme="object-id",
                        ),
                        obs_tr,
                        refs,
                        factors={"affordance": ytr},
                    ),
                    TensorFormAdapter(
                        FormMeta(
                            tag="action_consequence",
                            kind="control",
                            feature_dim=actions,
                            source="deterministic-affordance-world",
                            objective="programmatic",
                            referent_scheme="object-id",
                        ),
                        outcomes_tr,
                        refs,
                        factors={"affordance": ytr},
                    ),
                ]
            )
            audit = form_audit(matrix, require_controls=False)

        means = {k: _mean(v) for k, v in scores.items()}
        intervention_gain = means["intervention"] - means["passive"]
        shuffle_delta = means["intervention"] - means["action_shuffle"]
        deltas = [
            min(
                scores["intervention"][i] - scores["passive"][i],
                scores["intervention"][i] - scores["action_shuffle"][i],
            )
            for i in range(len(seeds))
        ]
        margin = float(e.margin)
        rejects = intervention_gain > margin and shuffle_delta > margin
        input_dim = obs_dim + actions
        flops = (
            len(seeds)
            * 3
            * int(e.epochs)
            * mlp_flops([input_dim, int(e.hidden), actions], batch=int(e.train_samples))
        )
        return {
            "affordance_decode_acc": round(means["intervention"], 4),
            "intervention_gain": round(intervention_gain, 4),
            "action_shuffle_delta": round(shuffle_delta, 4),
            "action_selection_success": round(means["intervention"], 4),
            "accuracy_by_arm": {k: round(v, 4) for k, v in means.items()},
            "matched_architecture": True,
            "matched_updates": True,
            "form_audit": audit,
            "seeds": seeds,
            "seed_ci": seed_ci(deltas),
            "sign_flip": sign_flip_report(deltas),
            "null_supported": not rejects,
            "environment_contract": bounded_trajectory_contract(seed=seeds[0], episodes=4, horizon=8),
            "density": density_block(
                {"affordance_decode_acc": means["intervention"]},
                params=params,
                flops=flops,
                updates=len(seeds) * 3 * int(e.epochs),
            ),
        }


class F8(Experiment):
    id = "f8_plastic_substrate_rewrite"
    metric = ("heldout_factor_acc", "representation_rewrite_delta", "matched_compute_delta")
    baseline = (
        "frozen inherited, larger frozen shell, and SSL-trained random-init controls at a matched "
        "estimated end-to-end FLOP budget"
    )
    ablation = "plastic self-supervised Form encoder vs inherited frozen encoder and shell-only controls"
    null_hypothesis = (
        "a plastic substrate fails to beat the frozen inherited, larger frozen-shell, or SSL-trained "
        "random-init controls on licensed held-out factors at a matched estimated end-to-end FLOP budget"
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        return run_gated_rewrite(
            variant="f8",
            experiment_id=self.id,
            metric_names=self.metric,
            cfg=cfg,
            device=device,
            run_dir=run_dir,
        )


class F16(Experiment):
    id = "f16_perfect_slate_null"
    metric = ("blank_vs_inherited_delta", "heldout_transfer", "matched_compute_delta")
    baseline = (
        "frozen inherited, larger-shell, and shared-initialization frozen-random controls at a matched "
        "estimated end-to-end FLOP budget"
    )
    ablation = (
        "self-supervised plastic training of a blank encoder vs the identical initialization kept frozen"
    )
    null_hypothesis = (
        "the blank substrate fails to beat inherited frozen features, the larger shell, or its identical "
        "frozen-random initialization on licensed held-out transfer at a matched estimated end-to-end "
        "FLOP budget"
    )
    tier = "env-later"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        return run_gated_rewrite(
            variant="f16",
            experiment_id=self.id,
            metric_names=self.metric,
            cfg=cfg,
            device=device,
            run_dir=run_dir,
        )


MISSING_F_EXPERIMENTS: tuple[type[Experiment], ...] = (F6, F7, F8, F11, F15, F16)
