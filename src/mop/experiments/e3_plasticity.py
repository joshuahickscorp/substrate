
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..diagnostics import critical_period_signature, fisher_trace_over_training  # noqa: E402
from ..learning.backprop import Learner, TrainConfig  # noqa: E402
from ..metrics import ContinualResult, FrontierPoint, frontier_auc, retention_from_bwt  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..shell import Neuromodulation, PlasticityController  # noqa: E402
from ..shell.heads import ClassHead  # noqa: E402
from ..substrate.datasets import make_task_stream  # noqa: E402
from .base import Experiment, _diag_mean, _split  # noqa: E402

ARMS = ("constant", "decay", "staged", "triggered")


class E3(Experiment):
    id = "e3_plasticity"
    metric = ("backward_transfer", "frontier_auc", "fisher_peak_index", "staged_beats_both")
    baseline = "constant LR (no plasticity controller, fixed base_lr)"
    ablation = "plasticity schedule arms = {constant, soft decay, hard staged, staged+triggered}"
    null_hypothesis = (
        "staged plasticity is just an LR trick: the hard critical-period schedule ties BOTH "
        "constant LR and a tuned soft decay on the adaptation-retention frontier, and on a "
        "frozen substrate there is no sensitivity window (no Fisher rise-then-fall) to shape"
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
        dim = int(s.dim)
        train = [_split(t)[0] for t in raw]
        test = [_split(t)[1] for t in raw]

        results: dict[str, ContinualResult] = {}
        for arm in ARMS:
            results[arm] = self._run_arm(arm, train, test, n_classes, dim, cfg, device)

        points = [
            FrontierPoint(a, adaptation=_diag_mean(r), retention=retention_from_bwt(r.backward_transfer()))
            for a, r in results.items()
        ]
        bwt = {a: r.backward_transfer() for a, r in results.items()}
        margin = float(e.get("tie_margin", 0.02))
        staged_beats_constant = bwt["staged"] > bwt["constant"] + margin
        staged_beats_decay = bwt["staged"] > bwt["decay"] + margin
        triggered_beats_staged = bwt["triggered"] > bwt["staged"] + margin

        fisher = self._fisher(train, n_classes, dim, cfg)

        out = {
            "arms": {a: r.summary() for a, r in results.items()},
            "bwt": bwt,
            "frontier_auc": frontier_auc(points),
            "frontier": {p.name: {"adaptation": p.adaptation, "retention": p.retention} for p in points},
            "chance": 1.0 / n_classes,
            "tie_margin": margin,
            "staged_beats_constant": bool(staged_beats_constant),
            "staged_beats_decay": bool(staged_beats_decay),
            "staged_beats_both": bool(staged_beats_constant and staged_beats_decay),
            "triggered_beats_staged": bool(triggered_beats_staged),
            "fisher_trace": fisher["trace"],
            "fisher_peak_index": int(fisher["peak_index"]),
            "fisher_rise_then_fall": bool(fisher["rise_then_fall"]),
        }
        self._plots(results, points, fisher["trace"], run_dir)
        return out

    def _controller(
        self, arm: str, base_lr: float
    ) -> tuple[PlasticityController | None, Neuromodulation | None]:
        if arm == "constant":
            return None, None  # no controller: schedule effectively stays at 1
        schedule = "soft" if arm == "decay" else "hard"
        pcfg = OmegaConf.create({"schedule": schedule, "lr": base_lr, "rigidity": 0.0, "pnn_fraction": 0.0})
        plast = PlasticityController(pcfg, seed=0)
        nm = None
        if arm == "triggered":
            ncfg = OmegaConf.create(
                {
                    "enabled": True,
                    "surprise_gain": 2.0,
                    "novelty_gain": 1.0,
                    "uncertainty_gain": 1.0,
                    "gate_floor": 0.5,
                    "gate_ceil": 3.0,
                }
            )
            nm = Neuromodulation(ncfg)
        return plast, nm

    def _run_arm(self, arm, train, test, n_classes, dim, cfg, device) -> ContinualResult:
        seed_everything(int(cfg.seed))
        model = ClassHead(
            dim, n_classes, hidden=int(cfg.experiment.head.hidden), depth=int(cfg.experiment.head.depth)
        )
        base_lr = float(cfg.shell.plasticity.lr)
        plast, nm = self._controller(arm, base_lr)
        tc = TrainConfig(
            epochs_per_task=int(cfg.experiment.train.epochs_per_task),
            batch_size=int(cfg.experiment.train.batch_size),
            replay_batch=int(cfg.experiment.train.batch_size),
            base_lr=base_lr,
        )
        learner = Learner(model, device, tc, plasticity=plast, neuromod=nm, seed=int(cfg.seed))
        T = len(train)
        R: list[list[float]] = []
        adapt: list[int] = []
        for i, task in enumerate(train):
            adapt.append(learner.train_task(task, i / T, (i + 1) / T))
            R.append(learner.evaluate(test))
        return ContinualResult(R=R, chance=1.0 / n_classes, adapt_steps=adapt)

    def _fisher(self, train, n_classes, dim, cfg) -> dict:
        seed_everything(int(cfg.seed))
        t0 = train[0]
        x = safe_to(t0.x, torch.device("cpu"))
        y = safe_to(t0.y, torch.device("cpu"))
        bs = int(cfg.experiment.train.batch_size)
        batches = [(x[i : i + bs], y[i : i + bs]) for i in range(0, x.shape[0], bs)]
        hidden = int(cfg.experiment.fisher.hidden)

        def make_model():
            torch.manual_seed(int(cfg.seed))
            return ClassHead(dim, n_classes, hidden=hidden, depth=2)

        def loss_fn_factory(m):
            return lambda b: F.cross_entropy(m(b[0]), b[1])

        trace = fisher_trace_over_training(
            make_model,
            batches,
            loss_fn_factory,
            checkpoints=int(cfg.experiment.fisher.checkpoints),
            steps_per_ckpt=int(cfg.experiment.fisher.steps_per_ckpt),
            lr=float(cfg.experiment.fisher.lr),
        )
        return critical_period_signature(trace)

    def _plots(self, results, points, trace, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for arm, r in results.items():
            ax[0].plot(range(len(r.R[-1])), r.R[-1], marker="o", label=arm)
        ax[0].set_xlabel("task")
        ax[0].set_ylabel("final accuracy")
        ax[0].set_title("per-task accuracy (after full stream)")
        ax[0].legend(fontsize=7)
        for p in points:
            ax[1].scatter(p.adaptation, p.retention)
            ax[1].annotate(p.name, (p.adaptation, p.retention))
        ax[1].set_xlabel("adaptation (mean peak acc)")
        ax[1].set_ylabel("retention (1+BWT)")
        ax[1].set_title("adaptation-retention frontier")
        ax[1].set_xlim(0, 1.05)
        ax[1].set_ylim(0, 1.05)
        ax[2].plot(range(len(trace)), trace, marker="s", color="purple")
        ax[2].set_xlabel("checkpoint")
        ax[2].set_ylabel("Fisher trace")
        ax[2].set_title("Fisher trace (critical-period signature)")
        fig.tight_layout()
        fig.savefig(run_dir / "e3_plasticity.png", dpi=110)
        plt.close(fig)
