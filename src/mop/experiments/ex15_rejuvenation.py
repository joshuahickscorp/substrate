
from __future__ import annotations

import copy
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

DEAD_UNIT_VAR_THRESHOLD = 1e-4


@torch.no_grad()
def _hidden_activations(model: ClassHead, x: torch.Tensor) -> torch.Tensor:
    net = model.net
    if isinstance(net, torch.nn.Sequential) and len(net) > 1:
        return net[:-1](x)
    return x


@torch.no_grad()
def _dead_unit_count(model: ClassHead, x: torch.Tensor, threshold: float = DEAD_UNIT_VAR_THRESHOLD) -> int:
    net = model.net
    if not (isinstance(net, torch.nn.Sequential) and len(net) > 1):
        return 0
    h = _hidden_activations(model, x)
    if h.shape[0] < 2:
        return 0
    var = h.var(dim=0, unbiased=False)
    return int((var < threshold).sum().item())


@torch.no_grad()
def _shrink_and_perturb(
    model: ClassHead,
    init_state: dict[str, torch.Tensor],
    shrink_factor: float,
    noise_std: float,
    seed: int,
) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        init = init_state[name].to(p.device)
        noise = torch.randn(p.shape, generator=g).to(p.device) * noise_std
        p.data.mul_(shrink_factor).add_(init, alpha=(1.0 - shrink_factor)).add_(noise)


class EX15(Experiment):
    id = "ex15_rejuvenation"
    metric = ("effective_rank", "dead_unit_count", "retained_accuracy")
    baseline = "protected (replay+EWC), no rejuvenation (the EX13 protected arm, standing control)"
    ablation = (
        "protected-no-rejuvenation vs protected+shrink-and-perturb-rejuvenation vs "
        "frozen-random-substrate+rejuvenation, over the same long domain-incremental stream"
    )
    null_hypothesis = (
        "rejuvenation does not restore plasticity, or it restores plasticity at the cost of "
        "retention; the frozen-latent shell does not suffer loss of plasticity at this scale"
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
        rejuv_interval = int(e.get("rejuvenation_interval", 25))
        shrink_factor = float(e.get("shrink_factor", 0.6))
        noise_std = float(e.get("noise_std", 0.02))

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

        results: dict[str, dict] = {}
        wall: dict[str, float] = {}

        t0 = time.time()
        results["protected"] = self._run_arm(
            "protected",
            train,
            anchors,
            n_classes,
            dim,
            cfg,
            device,
            project_random=False,
            rejuvenate=False,
        )
        wall["protected"] = round(time.time() - t0, 3)

        t0 = time.time()
        results["protected_rejuvenated"] = self._run_arm(
            "protected_rejuvenated",
            train,
            anchors,
            n_classes,
            dim,
            cfg,
            device,
            project_random=False,
            rejuvenate=True,
        )
        wall["protected_rejuvenated"] = round(time.time() - t0, 3)

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
        results["frozen_random_rejuvenated"] = self._run_arm(
            "frozen_random_rejuvenated",
            control_train,
            control_anchors,
            n_classes,
            dim,
            cfg,
            device,
            project_random=True,
            rejuvenate=True,
        )
        wall["frozen_random_rejuvenated"] = round(time.time() - t0, 3)

        effective_rank_trace = {a: r["effective_rank_by_task"] for a, r in results.items()}
        dead_unit_trace = {a: r["dead_units_by_task"] for a, r in results.items()}
        retained_accuracy = {a: r["anchor_acc_by_task"] for a, r in results.items()}

        base = results["protected"]["summary"]
        rej = results["protected_rejuvenated"]["summary"]
        ctrl = results["frozen_random_rejuvenated"]["summary"]

        base_rank_peak = max(results["protected"]["effective_rank_by_task"].values(), default=0.0)
        base_rank_final = base["final_effective_rank"]
        rank_collapse_margin = float(e.get("rank_collapse_margin", 0.1))
        plasticity_loss_observed = bool(
            base_rank_peak > 0 and (base_rank_peak - base_rank_final) / base_rank_peak > rank_collapse_margin
        ) or bool(base["final_dead_units"] > 0)

        rank_restore_margin = float(e.get("rank_restore_margin", 0.05))
        rank_restored = bool(rej["final_effective_rank"] > base_rank_final * (1.0 + rank_restore_margin))
        dead_units_reduced = bool(rej["final_dead_units"] < base["final_dead_units"])
        restores_plasticity = rank_restored or dead_units_reduced

        retention_cost_margin = float(e.get("retention_cost_margin", 0.1))
        retention_cost = base["final_mean_anchor_acc"] - rej["final_mean_anchor_acc"]
        retention_cost_paid = bool(retention_cost > retention_cost_margin)

        ctrl_rank_restored = bool(
            ctrl["final_effective_rank"]
            > results["frozen_random_rejuvenated"]["baseline_rank_reference"] * (1.0 + rank_restore_margin)
        )
        substrate_specific = restores_plasticity and not ctrl_rank_restored

        null_supported = (
            (not plasticity_loss_observed)
            or (not restores_plasticity)
            or (restores_plasticity and retention_cost_paid)
        )

        out = {
            "arms": {a: r["summary"] for a, r in results.items()},
            "effective_rank": effective_rank_trace,
            "dead_unit_count": dead_unit_trace,
            "retained_accuracy": retained_accuracy,
            "n_tasks": n_tasks,
            "n_tasks_control": control_n_tasks,
            "anchor_tasks": len(anchors),
            "eval_every": eval_every,
            "rejuvenation_interval": rejuv_interval,
            "shrink_factor": shrink_factor,
            "noise_std": noise_std,
            "chance": chance,
            "plasticity_loss_observed": plasticity_loss_observed,
            "restores_plasticity": restores_plasticity,
            "rank_restored": rank_restored,
            "dead_units_reduced": dead_units_reduced,
            "retention_cost": round(retention_cost, 4),
            "retention_cost_paid": retention_cost_paid,
            "substrate_specific": substrate_specific,
            "wall_clock_seconds": wall,
            "null_supported": bool(null_supported),
        }
        return out

    def _run_arm(
        self,
        arm: str,
        train: list[Task],
        anchors: list[Task],
        n_classes: int,
        dim: int,
        cfg: DictConfig,
        device: DeviceInfo,
        project_random: bool,
        rejuvenate: bool,
    ) -> dict:
        seed = int(cfg.seed)
        seed_everything(seed)
        e = cfg.experiment
        eval_every = int(e.get("eval_every", 10))
        rejuv_interval = int(e.get("rejuvenation_interval", 25))
        shrink_factor = float(e.get("shrink_factor", 0.6))
        noise_std = float(e.get("noise_std", 0.02))

        model = ClassHead(dim, n_classes, hidden=int(e.head.hidden), depth=int(e.head.depth))
        init_state = copy.deepcopy(model.state_dict())

        sh = cfg.shell
        buffer = ReplayBuffer(
            capacity=int(sh.buffer.capacity),
            dim=dim,
            prioritized=bool(sh.buffer.prioritized),
            alpha=float(sh.buffer.alpha),
            beta=float(sh.buffer.beta),
            index=str(sh.buffer.index),
            eviction=str(sh.buffer.eviction),
            seed=seed,
        )
        con = Consolidation(sh.consolidation)
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
        dead_units_by_task: dict[int, int] = {}
        rejuvenation_events: list[int] = []
        baseline_rank_reference = 0.0
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

            if rejuvenate and rejuv_interval > 0 and i > 0 and i % rejuv_interval == 0:
                _shrink_and_perturb(model, init_state, shrink_factor, noise_std, seed=seed + i)
                rejuvenation_events.append(i)

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
            if i % eval_every == 0 or i == T - 1:
                with torch.no_grad():
                    x = proj_anchors[0].x if proj_anchors else task.x
                    hidden = _hidden_activations(model, x)
                    er = round(effective_rank(hidden), 4)
                    effective_rank_by_task[i] = er
                    dead_units_by_task[i] = _dead_unit_count(model, x)
                    if i == 0:
                        baseline_rank_reference = er

        mean_anchor_final = (
            sum(anchor_acc_by_task[-1]) / len(anchor_acc_by_task[-1]) if anchor_acc_by_task else 0.0
        )
        mean_anchor_peak = max((sum(a) / len(a) for a in anchor_acc_by_task), default=0.0)
        last_key = max(effective_rank_by_task) if effective_rank_by_task else None
        summary = {
            "final_mean_anchor_acc": round(mean_anchor_final, 4),
            "peak_mean_anchor_acc": round(mean_anchor_peak, 4),
            "final_effective_rank": effective_rank_by_task.get(last_key, 0.0)
            if last_key is not None
            else 0.0,
            "final_dead_units": dead_units_by_task.get(last_key, 0) if last_key is not None else 0,
            "peak_dead_units": max(dead_units_by_task.values(), default=0),
            "n_rejuvenation_events": len(rejuvenation_events),
            "rejuvenation_events": rejuvenation_events,
        }
        return {
            "anchor_acc_by_task": anchor_acc_by_task,
            "effective_rank_by_task": effective_rank_by_task,
            "dead_units_by_task": dead_units_by_task,
            "baseline_rank_reference": baseline_rank_reference,
            "summary": summary,
        }
