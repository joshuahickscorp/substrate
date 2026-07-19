
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import param_count
from ..seeding import seed_everything
from ..shell.predictor import mlp
from .base import Experiment


def _head_net(dim: int, hidden: int, n_classes: int) -> nn.Module:
    return mlp(dim, n_classes, hidden, depth=1, ln=True)


def _clone(net: nn.Module) -> nn.Module:
    return copy.deepcopy(net)


def _sgd_steps(net: nn.Module, x, y, steps: int, lr: float) -> nn.Module:
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    return net


def _accuracy(net: nn.Module, x, y) -> float:
    with torch.no_grad():
        return float((net(x).argmax(-1) == y).float().mean())


def _reptile_meta_train(
    init: nn.Module,
    tasks: list,
    inner_steps: int,
    inner_lr: float,
    outer_lr: float,
    meta_iters: int,
    g: torch.Generator,
) -> nn.Module:
    meta = _clone(init)
    n = len(tasks)
    for _ in range(meta_iters):
        idx = int(torch.randint(0, n, (1,), generator=g).item())
        task = tasks[idx]
        adapted = _clone(meta)
        _sgd_steps(adapted, task.x, task.y, inner_steps, inner_lr)
        with torch.no_grad():
            for p_meta, p_adapt in zip(meta.parameters(), adapted.parameters(), strict=True):
                p_meta.add_(outer_lr * (p_adapt.detach() - p_meta))
    return meta


def _steps_to_target(
    init: nn.Module, task, target_acc: float, max_steps: int, lr: float
) -> tuple[int, float]:
    net = _clone(init)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    best = _accuracy(net, task.x, task.y)
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        F.cross_entropy(net(task.x), task.y).backward()
        opt.step()
        acc = _accuracy(net, task.x, task.y)
        best = max(best, acc)
        if acc >= target_acc:
            return step, acc
    return max_steps, best


def _fixed_budget_accuracy(init: nn.Module, task, adapt_steps: int, lr: float) -> float:
    net = _clone(init)
    _sgd_steps(net, task.x, task.y, adapt_steps, lr)
    return _accuracy(net, task.x, task.y)


class EX7(Experiment):
    id = "ex7_meta_learning"
    metric = (
        "adaptation_speed_meta",
        "adaptation_speed_random",
        "adaptation_speed_gain",
        "forward_transfer_meta",
        "forward_transfer_random",
        "forward_transfer_gain",
    )
    baseline = "random-init control adapted at matched total inner-step budget"
    ablation = (
        "Reptile meta-init vs random-init vs a second fixed-seed (hypernet-style) init, held-out tasks only"
    )
    null_hypothesis = (
        "the meta-learned init does not reduce adaptation steps vs a control init; the task family is "
        "too homogeneous or the shell too small"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        n_classes = int(e.n_classes)
        n_meta_tasks, n_holdout_tasks = int(e.n_meta_tasks), int(e.n_holdout_tasks)
        samples, separation = int(e.samples_per_task), float(e.separation)
        inner_steps, inner_lr = int(e.inner_steps), float(e.inner_lr)
        outer_lr, meta_iters = float(e.outer_lr), int(e.meta_iters)
        target_acc = float(e.target_acc)
        max_adapt_steps = int(e.max_adapt_steps)
        fixed_budget = int(e.fixed_budget_steps)
        margin = float(e.margin)

        speed_meta, speed_rand, speed_hyper = [], [], []
        xfer_meta, xfer_rand, xfer_hyper = [], [], []

        for s in seeds:
            seed_everything(s)
            g = torch.Generator().manual_seed(s)

            meta_tasks = make_task_stream(
                n_tasks=n_meta_tasks,
                dim=dim,
                classes_per_task=n_classes,
                samples_per_task=samples,
                separation=separation,
                incremental="task",
                seed=s * 1000 + 1,
            )
            holdout_tasks = make_task_stream(
                n_tasks=n_holdout_tasks,
                dim=dim,
                classes_per_task=n_classes,
                samples_per_task=samples,
                separation=separation,
                incremental="task",
                seed=s * 1000 + 2,
            )

            seed_everything(s)
            random_init = _head_net(dim, hidden, n_classes)

            seed_everything(s)
            meta_start = _head_net(dim, hidden, n_classes)
            meta_init = _reptile_meta_train(
                meta_start, meta_tasks, inner_steps, inner_lr, outer_lr, meta_iters, g
            )

            seed_everything(s + 777)
            hyper_init = _head_net(dim, hidden, n_classes)

            for task in holdout_tasks:
                sm, _ = _steps_to_target(meta_init, task, target_acc, max_adapt_steps, inner_lr)
                sr, _ = _steps_to_target(random_init, task, target_acc, max_adapt_steps, inner_lr)
                sh, _ = _steps_to_target(hyper_init, task, target_acc, max_adapt_steps, inner_lr)
                speed_meta.append(sm)
                speed_rand.append(sr)
                speed_hyper.append(sh)

                xfer_meta.append(_fixed_budget_accuracy(meta_init, task, fixed_budget, inner_lr))
                xfer_rand.append(_fixed_budget_accuracy(random_init, task, fixed_budget, inner_lr))
                xfer_hyper.append(_fixed_budget_accuracy(hyper_init, task, fixed_budget, inner_lr))

        def _mean(v: list[float] | list[int]) -> float:
            return sum(v) / len(v) if v else 0.0

        def _spread(v: list[float] | list[int]) -> float:
            return (max(v) - min(v)) / 2.0 if len(v) > 1 else 0.0

        speed_meta_mean, speed_rand_mean, speed_hyper_mean = (
            _mean(speed_meta),
            _mean(speed_rand),
            _mean(speed_hyper),
        )
        xfer_meta_mean, xfer_rand_mean, xfer_hyper_mean = (
            _mean(xfer_meta),
            _mean(xfer_rand),
            _mean(xfer_hyper),
        )

        speed_gain = speed_rand_mean - speed_meta_mean
        xfer_gain = xfer_meta_mean - xfer_rand_mean
        speed_gain_spread = max(_spread(speed_meta), _spread(speed_rand))
        xfer_gain_spread = max(_spread(xfer_meta), _spread(xfer_rand))

        meta_probe = _head_net(dim, hidden, n_classes)
        out = {
            "adaptation_speed_meta": round(speed_meta_mean, 3),
            "adaptation_speed_random": round(speed_rand_mean, 3),
            "adaptation_speed_hypernet": round(speed_hyper_mean, 3),
            "adaptation_speed_gain": round(speed_gain, 3),
            "forward_transfer_meta": round(xfer_meta_mean, 4),
            "forward_transfer_random": round(xfer_rand_mean, 4),
            "forward_transfer_hypernet": round(xfer_hyper_mean, 4),
            "forward_transfer_gain": round(xfer_gain, 4),
            "margin_steps": margin,
            "speed_gain_spread": round(speed_gain_spread, 3),
            "xfer_gain_spread": round(xfer_gain_spread, 4),
            "params": param_count(meta_probe),
            "n_holdout_tasks_total": len(speed_meta),
            "seeds": list(seeds),
            "null_supported": bool(speed_gain <= margin + speed_gain_spread),
            "meta_wins_adaptation_speed": bool(speed_gain > margin + speed_gain_spread),
            "meta_wins_forward_transfer": bool(xfer_gain > margin / max_adapt_steps + xfer_gain_spread),
        }
        return out
