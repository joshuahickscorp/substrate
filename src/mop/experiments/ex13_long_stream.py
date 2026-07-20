
from __future__ import annotations

import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.geometry import effective_rank
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..learning.backprop import Learner, TrainConfig
from ..seeding import seed_everything
from ..shell import Consolidation, ReplayBuffer
from ..shell.heads import ClassHead
from ..substrate.datasets import Task, make_task_stream
from .base import Experiment, _split


@torch.no_grad()
def _hidden_activations(model: ClassHead, x: torch.Tensor) -> torch.Tensor:
    net = model.net
    if isinstance(net, torch.nn.Sequential) and len(net) > 1:
        return net[:-1](x)
    return x


class EX13(Experiment):
    id = "ex13_long_stream"
    metric = ("forgetting_curve", "retention_threshold_length", "effective_rank")
    baseline = "naive-sequential (no replay, no consolidation, no plasticity scheduling)"
    ablation = (
        "naive-sequential vs replay+EWC-protected vs frozen-random-substrate control, over a long "
        "domain-incremental stream (hundreds of tasks)"
    )
    null_hypothesis = (
        "retention is flat in stream length within the seed spread, or every mechanism degrades "
        "identically (protection only delays the same collapse)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        s = e.stream
        seed = int(cfg.seed)
        seed_everything(seed)

        n_tasks = int(s.n_tasks)
        dim = int(s.dim)
        anchor_tasks = int(e.get("anchor_tasks", 3))
        eval_every = int(e.get("eval_every", 10))

        raw = make_task_stream(
            n_tasks=n_tasks,
            dim=dim,
            classes_per_task=int(s.classes_per_task),
            samples_per_task=int(s.samples_per_task),
            separation=float(s.separation),
            incremental=str(s.get("incremental", "domain")),
            seed=seed,
        )
        n_classes = raw[0].n_classes
        chance = 1.0 / n_classes
        train = [_split(t)[0] for t in raw]
        test = [_split(t)[1] for t in raw]
        anchors = test[: min(anchor_tasks, len(test))]

        arms_cfg = dict(e.configs)
        results: dict[str, dict] = {}
        wall: dict[str, float] = {}
        for arm, flags in arms_cfg.items():
            t0 = time.time()
            results[arm] = self._run_arm(
                arm, flags, train, anchors, n_classes, dim, cfg, device, project_random=False
            )
            wall[arm] = round(time.time() - t0, 3)

        control_n_tasks = int(e.get("n_tasks_control", n_tasks))
        if control_n_tasks == n_tasks:
            control_train, control_anchors = train, anchors
        else:
            raw_c = make_task_stream(
                n_tasks=control_n_tasks,
                dim=dim,
                classes_per_task=int(s.classes_per_task),
                samples_per_task=int(s.samples_per_task),
                separation=float(s.separation),
                incremental=str(s.get("incremental", "domain")),
                seed=seed,
            )
            control_train = [_split(t)[0] for t in raw_c]
            control_test = [_split(t)[1] for t in raw_c]
            control_anchors = control_test[: min(anchor_tasks, len(control_test))]
        t0 = time.time()
        protected_flags = arms_cfg.get("protected", {"replay": True, "ewc": True})
        results["frozen_random"] = self._run_arm(
            "frozen_random",
            protected_flags,
            control_train,
            control_anchors,
            n_classes,
            dim,
            cfg,
            device,
            project_random=True,
        )
        wall["frozen_random"] = round(time.time() - t0, 3)

        forgetting_curve = {arm: r["anchor_acc_by_task"] for arm, r in results.items()}
        retention_threshold_length = {arm: r["retention_threshold_length"] for arm, r in results.items()}
        effective_rank_trace = {arm: r["effective_rank_by_task"] for arm, r in results.items()}

        naive_bwt = self._anchor_bwt(results["naive"]["anchor_acc_by_task"], chance)
        protected_bwt = self._anchor_bwt(results["protected"]["anchor_acc_by_task"], chance)
        frozen_bwt = self._anchor_bwt(results["frozen_random"]["anchor_acc_by_task"], chance)

        divergence = protected_bwt - naive_bwt
        survives_frozen_random_control = (
            divergence > (protected_bwt - frozen_bwt) * 0.5 and (protected_bwt - frozen_bwt) > 0.05
        )
        meaningfully_different = abs(divergence) > float(e.get("divergence_margin", 0.1))

        null_supported = (not meaningfully_different) or (not survives_frozen_random_control)

        out = {
            "arms": {a: r["summary"] for a, r in results.items()},
            "forgetting_curve": forgetting_curve,
            "retention_threshold_length": retention_threshold_length,
            "effective_rank": effective_rank_trace,
            "n_tasks": n_tasks,
            "n_tasks_control": control_n_tasks,
            "anchor_tasks": len(anchors),
            "eval_every": eval_every,
            "chance": chance,
            "naive_anchor_bwt": round(naive_bwt, 4),
            "protected_anchor_bwt": round(protected_bwt, 4),
            "frozen_random_anchor_bwt": round(frozen_bwt, 4),
            "divergence_protected_minus_naive": round(divergence, 4),
            "survives_frozen_random_control": bool(survives_frozen_random_control),
            "wall_clock_seconds": wall,
            "null_supported": bool(null_supported),
        }
        return out

    def _run_arm(
        self,
        arm: str,
        flags,
        train: list[Task],
        anchors: list[Task],
        n_classes: int,
        dim: int,
        cfg: DictConfig,
        device: DeviceInfo,
        project_random: bool,
    ) -> dict:
        seed = int(cfg.seed)
        seed_everything(seed)
        e = cfg.experiment
        eval_every = int(e.get("eval_every", 10))
        threshold_margin = float(e.get("retention_threshold_margin", 0.1))
        chance = 1.0 / n_classes

        model = ClassHead(dim, n_classes, hidden=int(e.head.hidden), depth=int(e.head.depth))
        sh = cfg.shell
        buffer = (
            ReplayBuffer(
                capacity=int(sh.buffer.capacity),
                dim=dim,
                prioritized=bool(sh.buffer.prioritized),
                alpha=float(sh.buffer.alpha),
                beta=float(sh.buffer.beta),
                index=str(sh.buffer.index),
                eviction=str(sh.buffer.eviction),
                seed=seed,
            )
            if bool(flags.get("replay", False))
            else None
        )
        con = Consolidation(sh.consolidation) if bool(flags.get("ewc", False)) else None
        tc = TrainConfig(
            epochs_per_task=int(e.train.epochs_per_task),
            batch_size=int(e.train.batch_size),
            replay_batch=int(e.train.batch_size),
            base_lr=float(sh.plasticity.lr),
        )
        learner = Learner(model, device, tc, buffer=buffer, consolidation=con, seed=seed)

        def maybe_project(x: torch.Tensor) -> torch.Tensor:
            return frozen_random_projection(x, seed=seed) if project_random else x

        T = len(train)
        anchor_acc_by_task: list[list[float]] = []
        effective_rank_by_task: dict[int, float] = {}
        retention_threshold_length: int | None = None
        for i, task in enumerate(train):
            proj_task = (
                task
                if not project_random
                else Task(
                    task.name,
                    maybe_project(task.x),
                    task.y,
                    n_classes=task.n_classes,
                    task_id=task.task_id,
                )
            )
            learner.train_task(proj_task, i / T, (i + 1) / T)
            proj_anchors = (
                anchors
                if not project_random
                else [
                    Task(a.name, maybe_project(a.x), a.y, n_classes=a.n_classes, task_id=a.task_id)
                    for a in anchors
                ]
            )
            accs = learner.evaluate(proj_anchors)
            anchor_acc_by_task.append([round(a, 4) for a in accs])
            mean_anchor = sum(accs) / len(accs) if accs else 0.0
            if retention_threshold_length is None and mean_anchor < chance + threshold_margin:
                retention_threshold_length = i
            if i % eval_every == 0 or i == T - 1:
                with torch.no_grad():
                    x = proj_anchors[0].x if proj_anchors else task.x
                    hidden = _hidden_activations(model, x)
                    effective_rank_by_task[i] = round(effective_rank(hidden), 4)

        mean_anchor_final = (
            sum(anchor_acc_by_task[-1]) / len(anchor_acc_by_task[-1]) if anchor_acc_by_task else 0.0
        )
        mean_anchor_peak = max((sum(a) / len(a) for a in anchor_acc_by_task), default=0.0)
        summary = {
            "final_mean_anchor_acc": round(mean_anchor_final, 4),
            "peak_mean_anchor_acc": round(mean_anchor_peak, 4),
            "retention_threshold_length": retention_threshold_length,
            "final_effective_rank": effective_rank_by_task.get(max(effective_rank_by_task), 0.0)
            if effective_rank_by_task
            else 0.0,
        }
        return {
            "anchor_acc_by_task": anchor_acc_by_task,
            "effective_rank_by_task": effective_rank_by_task,
            "retention_threshold_length": retention_threshold_length,
            "summary": summary,
        }

    @staticmethod
    def _anchor_bwt(anchor_acc_by_task: list[list[float]], chance: float) -> float:
        if not anchor_acc_by_task:
            return 0.0
        means = [sum(a) / len(a) for a in anchor_acc_by_task if a]
        if not means:
            return 0.0
        peak = max(means)
        final = means[-1]
        return float(final - peak)
