"""E2: latent hippocampus (replay). Over a domain-incremental latent stream (low separation,
so the naive head forgets), compare four replay arms on the adaptation-retention frontier:
  (a) no-replay        : ordinary sequential training, the forgetting baseline
  (b) random replay    : a uniform ReplayBuffer (prioritized=False)
  (c) prioritized replay: a PER ReplayBuffer (prioritized=True), surprise-weighted
  (d) replay+EWC       : random replay plus weight-space Consolidation (method=ewc)

Each arm yields a ContinualResult (R from evaluate after each task), a FrontierPoint
(adaptation=mean diagonal acc, retention=retention_from_bwt(BWT)) and a per-arm BWT. The
explicit null (corpus E2): prioritized replay merely TIES random replay AND replay merely
TIES no-replay (stream too short / latents not distinct). We answer it directly with two
booleans (replay_beats_noreplay, prioritized_beats_random) and the BWT gaps that back them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from ..devices import DeviceInfo  # noqa: E402
from ..learning.backprop import Learner, TrainConfig  # noqa: E402
from ..metrics import ContinualResult, FrontierPoint, frontier_auc, retention_from_bwt  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..shell import Consolidation, ReplayBuffer  # noqa: E402
from ..shell.heads import ClassHead  # noqa: E402
from ..substrate.datasets import make_task_stream  # noqa: E402
from .base import Experiment, _diag_mean, _split  # noqa: E402

# arm spec: (uses_replay, prioritized, uses_ewc). None for prioritized => no buffer at all.
_ARMS = {
    "no_replay": (False, None, False),
    "random_replay": (True, False, False),
    "prioritized_replay": (True, True, False),
    "replay_ewc": (True, False, True),
}


class E2(Experiment):
    id = "e2_replay"
    metric = ("backward_transfer", "frontier_auc", "adaptation_speed", "retention")
    baseline = "no-replay sequential training over a domain-incremental latent stream"
    ablation = "replay scheme = {none, random, prioritized, replay+EWC} at matched buffer size"
    null_hypothesis = (
        "prioritized replay ties random replay AND replay ties no-replay (stream too short or "
        "latents not distinct enough for buffer content to matter)"
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
            incremental="domain",
            seed=int(cfg.seed),
        )
        n_classes = raw[0].n_classes
        train = [_split(t)[0] for t in raw]
        test = [_split(t)[1] for t in raw]
        chance = 1.0 / n_classes

        results = {
            arm: self._run_arm(spec, train, test, n_classes, int(s.dim), cfg, device)
            for arm, spec in _ARMS.items()
        }
        points = [
            FrontierPoint(arm, adaptation=_diag_mean(r), retention=retention_from_bwt(r.backward_transfer()))
            for arm, r in results.items()
        ]
        bwt = {arm: r.backward_transfer() for arm, r in results.items()}

        tol = float(e.get("tie_tol", 0.02))
        replay_gap = bwt["random_replay"] - bwt["no_replay"]
        prioritized_gap = bwt["prioritized_replay"] - bwt["random_replay"]
        replay_beats_noreplay = bool(replay_gap > tol)
        prioritized_beats_random = bool(prioritized_gap > tol)

        out = {
            "arms": {a: r.summary() for a, r in results.items()},
            "bwt": {a: float(v) for a, v in bwt.items()},
            "frontier_auc": frontier_auc(points),
            "chance": chance,
            "tie_tol": tol,
            "replay_gap": float(replay_gap),
            "prioritized_gap": float(prioritized_gap),
            # the explicit null answers: null holds when BOTH are False
            "replay_beats_noreplay": replay_beats_noreplay,
            "prioritized_beats_random": prioritized_beats_random,
            "null_holds": bool(not replay_beats_noreplay and not prioritized_beats_random),
        }
        self._plot(points, run_dir)
        return out

    def _run_arm(self, spec, train, test, n_classes, dim, cfg, device) -> ContinualResult:
        uses_replay, prioritized, uses_ewc = spec
        seed_everything(int(cfg.seed))
        model = ClassHead(
            dim, n_classes, hidden=int(cfg.experiment.head.hidden), depth=int(cfg.experiment.head.depth)
        )
        sh = cfg.shell
        buffer = (
            ReplayBuffer(
                capacity=int(sh.buffer.capacity),
                dim=dim,
                key_dim=dim,
                prioritized=bool(prioritized),
                alpha=float(sh.buffer.alpha),
                beta=float(sh.buffer.beta),
                index=str(sh.buffer.index),
                eviction=str(sh.buffer.eviction),
                seed=int(cfg.seed),
            )
            if uses_replay
            else None
        )
        con = None
        if uses_ewc:
            ccfg = OmegaConf.merge(sh.consolidation, OmegaConf.create({"method": "ewc"}))
            con = Consolidation(ccfg)
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
            adapt.append(learner.train_task(task, i / T, (i + 1) / T))
            R.append(learner.evaluate(test))
        return ContinualResult(R=R, chance=1.0 / n_classes, adapt_steps=adapt)

    def _plot(self, points, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        for p in points:
            ax.scatter(p.adaptation, p.retention, s=60)
            ax.annotate(p.name, (p.adaptation, p.retention), fontsize=8)
        ax.set_xlabel("adaptation (mean diagonal acc)")
        ax.set_ylabel("retention (1+BWT)")
        ax.set_title("E2 replay frontier")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(run_dir / "e2_frontier.png", dpi=110)
        plt.close(fig)
