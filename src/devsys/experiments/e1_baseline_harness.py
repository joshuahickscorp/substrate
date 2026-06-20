"""E1: the gate. A continual-learning harness over a class-incremental latent stream that
must demonstrate BOTH failure and success: a naive arm that visibly forgets, and a
replay+EWC arm that visibly retains, under fixed seeds, with saved plots. If E1 cannot show
forgetting, no continual-learning mechanism on top can be evaluated (the corpus rule).

The shared softmax head over fixed latents is the developmental substrate's downstream
probe; training only on the current task's classes degrades earlier-class logits (the
mechanism of catastrophic forgetting), which replay + EWC counteract.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ..devices import DeviceInfo  # noqa: E402
from ..learning.backprop import Learner, TrainConfig  # noqa: E402
from ..metrics import ContinualResult, FrontierPoint, frontier_auc, retention_from_bwt  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..shell import Consolidation, ReplayBuffer  # noqa: E402
from ..shell.heads import ClassHead  # noqa: E402
from ..substrate.datasets import Task, make_task_stream  # noqa: E402
from .base import Experiment  # noqa: E402


def _split(task: Task, frac: float = 0.8) -> tuple[Task, Task]:
    n = int(task.x.shape[0] * frac)
    tr = Task(task.name, task.x[:n], task.y[:n], n_classes=task.n_classes, task_id=task.task_id)
    te = Task(task.name, task.x[n:], task.y[n:], n_classes=task.n_classes, task_id=task.task_id)
    return tr, te


class E1(Experiment):
    id = "e1_baseline"
    metric = ("backward_transfer", "forward_transfer", "adaptation_speed", "frontier_auc")
    baseline = "ordinary sequential training (no replay, no EWC, fixed LR)"
    ablation = "predictor capacity x task separation; arms = {naive, replay+EWC}"
    null_hypothesis = (
        "with shuffled task labels (or matched capacity/difficulty) naive and protected show "
        "no retention gap; if no forgetting, tasks too easy; if no learning, target not "
        "decodable from the frozen latent"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        s = e.stream
        seed_everything(int(cfg.seed))
        raw = make_task_stream(
            n_tasks=int(s.n_tasks),
            dim=int(s.dim),
            classes_per_task=int(s.classes_per_task),
            samples_per_task=int(s.samples_per_task),
            separation=float(s.separation),
            incremental=str(s.get("incremental", "domain")),
            seed=int(cfg.seed),
        )
        n_classes = raw[0].n_classes
        train = [_split(t)[0] for t in raw]
        test = [_split(t)[1] for t in raw]
        chance = 1.0 / n_classes

        results: dict[str, ContinualResult] = {}
        for arm, flags in e.configs.items():
            results[arm] = self._run_arm(str(arm), flags, train, test, n_classes, int(s.dim), cfg, device)

        points = [
            FrontierPoint(arm, adaptation=_diag_mean(r), retention=retention_from_bwt(r.backward_transfer()))
            for arm, r in results.items()
        ]
        out = {
            "arms": {a: r.summary() for a, r in results.items()},
            "frontier_auc": frontier_auc(points),
            "chance": chance,
        }
        out["gate"] = self._gate(results, e.gate)
        self._plots(results, points, run_dir)
        return out

    def _run_arm(self, arm, flags, train, test, n_classes, dim, cfg, device) -> ContinualResult:
        seed_everything(int(cfg.seed))
        model = ClassHead(
            dim, n_classes, hidden=int(cfg.experiment.head.hidden), depth=int(cfg.experiment.head.depth)
        )
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
                seed=int(cfg.seed),
            )
            if bool(flags.replay)
            else None
        )
        con = Consolidation(sh.consolidation) if bool(flags.ewc) else None
        tc = TrainConfig(
            epochs_per_task=int(cfg.experiment.train.epochs_per_task),
            batch_size=int(cfg.experiment.train.batch_size),
            replay_batch=int(cfg.experiment.train.batch_size),
            base_lr=float(sh.plasticity.lr),
        )
        learner = Learner(model, device, tc, buffer=buffer, consolidation=con, seed=int(cfg.seed))
        T = len(train)
        R: list[list[float]] = []
        adapt: list[int] = []
        for i, task in enumerate(train):
            a = learner.train_task(task, i / T, (i + 1) / T)
            adapt.append(a)
            R.append(learner.evaluate(test))
        return ContinualResult(R=R, chance=1.0 / n_classes, adapt_steps=adapt)

    def _gate(self, results, g) -> dict:
        naive, prot = results["naive"], results["protected"]
        nb, pb = naive.backward_transfer(), prot.backward_transfer()
        forgets = nb <= float(g.forget_bwt)
        retains = pb >= nb + float(g.retain_margin)
        both_learn = naive.R[-1][-1] >= float(g.min_last_task) and prot.R[-1][-1] >= float(g.min_last_task)
        return {
            "naive_bwt": nb,
            "protected_bwt": pb,
            "naive_forgets": bool(forgets),
            "protected_retains": bool(retains),
            "both_learn_last_task": bool(both_learn),
            "passed": bool(forgets and retains and both_learn),
        }

    def _plots(self, results, points, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for arm, r in results.items():
            final = r.R[-1]
            ax[0].plot(range(len(final)), final, marker="o", label=f"{arm} (final)")
            ax[0].plot(
                [j for j in range(r.T)],
                [r.R[j][j] for j in range(r.T)],
                linestyle="--",
                alpha=0.5,
                label=f"{arm} (peak)",
            )
        ax[0].set_xlabel("task")
        ax[0].set_ylabel("accuracy")
        ax[0].set_title("per-task accuracy")
        ax[0].legend(fontsize=7)
        for p in points:
            ax[1].scatter(p.adaptation, p.retention)
            ax[1].annotate(p.name, (p.adaptation, p.retention))
        ax[1].set_xlabel("adaptation (mean peak acc)")
        ax[1].set_ylabel("retention (1+BWT)")
        ax[1].set_title("adaptation-retention frontier")
        ax[1].set_xlim(0, 1.05)
        ax[1].set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(run_dir / "e1_frontier.png", dpi=110)
        plt.close(fig)


def _diag_mean(r: ContinualResult) -> float:
    return float(sum(r.R[j][j] for j in range(r.T)) / r.T)
