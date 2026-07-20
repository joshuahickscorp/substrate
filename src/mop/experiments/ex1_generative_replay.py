
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..metrics import ContinualResult
from ..seeding import seed_everything
from ..shell.buffer import ReplayBuffer
from ..shell.heads import ClassHead
from .base import Experiment


class _DiagGaussianGenerator:

    def __init__(self, x: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = 1e-3):
        self.n_classes = n_classes
        self.dim = x.shape[1]
        self.mean = torch.zeros(n_classes, self.dim)
        self.std = torch.ones(n_classes, self.dim)
        for c in range(n_classes):
            xc = x[y == c]
            if xc.shape[0] < 2:
                continue
            self.mean[c] = xc.mean(0)
            self.std[c] = xc.std(0, unbiased=True).clamp_min(eps)

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.randint(0, self.n_classes, (n,), generator=generator)
        noise = torch.randn(n, self.dim, generator=generator)
        x = self.mean[y] + self.std[y] * noise
        return x, y


def _fit_eval_epochs(
    model: nn.Module,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    epochs: int,
    lr: float,
    replay_x: torch.Tensor | None,
    replay_y: torch.Tensor | None,
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(xtr), ytr)
        if replay_x is not None and replay_y is not None and replay_x.shape[0] > 0:
            loss = loss + F.cross_entropy(model(replay_x), replay_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((model(xte).argmax(-1) == yte).float().mean())


class EX1(Experiment):
    id = "ex1_generative_replay"
    metric = ("backward_transfer", "frontier_auc", "generator_probe_transfer")
    baseline = "stored-buffer replay: real cached task-0 latents via shell.buffer.ReplayBuffer"
    ablation = (
        "replay source = {none, stored-buffer, generated (per-class diagonal Gaussian)} "
        "at matched replay-sample budget"
    )
    null_hypothesis = (
        "generated replay does not match stored-buffer replay on BWT at matched budget; "
        "the distribution gap costs retention"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        n_classes_per_task = int(e.n_classes_per_task)
        samples = int(e.samples)
        separation = float(e.separation)
        epochs, lr = int(e.epochs), float(e.lr)
        replay_budget = int(e.replay_budget)
        tie_tol = float(e.tie_tol)

        arm_bwt: dict[str, list[float]] = {"no_replay": [], "stored_buffer": [], "generated": []}
        probe_transfer: list[float] = []

        for s in seeds:
            seed_everything(s)
            tasks = make_task_stream(
                n_tasks=2,
                dim=dim,
                classes_per_task=n_classes_per_task,
                samples_per_task=samples,
                separation=separation,
                incremental="task",  # shared label space across tasks 0 and 1
                seed=s,
            )
            t0, t1 = tasks[0], tasks[1]
            nc = n_classes_per_task
            cut0, cut1 = int(t0.x.shape[0] * 0.8), int(t1.x.shape[0] * 0.8)
            x0tr, y0tr, x0te, y0te = t0.x[:cut0], t0.y[:cut0], t0.x[cut0:], t0.y[cut0:]
            x1tr, y1tr = t1.x[:cut1], t1.y[:cut1]

            gen = _DiagGaussianGenerator(x0tr, y0tr, nc)
            g = torch.Generator().manual_seed(s)
            xgen, ygen = gen.sample(replay_budget, g)

            probe_real = _eval_linear_on(xgen, ygen, x0te, y0te, seed=s)
            probe_transfer.append(probe_real)

            buf = ReplayBuffer(capacity=max(replay_budget, x0tr.shape[0]), dim=dim, seed=s)
            buf.add(x0tr, y0tr)
            sampled = buf.sample(min(replay_budget, len(buf)))
            xbuf, ybuf = sampled["x"], sampled["y"]

            R: dict[str, list[list[float]]] = {}
            for arm in ("no_replay", "stored_buffer", "generated"):
                seed_everything(s)  # identical init across arms for a fair comparison
                model = ClassHead(dim, nc, hidden=hidden, depth=1)
                acc0_before = _eval(model, x0te, y0te)
                if arm == "no_replay":
                    rx, ry = None, None
                elif arm == "stored_buffer":
                    rx, ry = xbuf, ybuf
                else:
                    rx, ry = xgen, ygen
                _fit_eval_epochs(model, x1tr, y1tr, x0te, y0te, epochs, lr, rx, ry)
                acc0_after = _eval(model, x0te, y0te)
                R[arm] = [[acc0_before, 0.0], [acc0_after, acc0_after]]

            for arm in arm_bwt:
                r = ContinualResult(R=R[arm], chance=1.0 / nc)
                arm_bwt[arm].append(r.backward_transfer())

        def mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs))

        bwt_mean = {a: round(mean(v), 4) for a, v in arm_bwt.items()}
        gap_vs_stored = bwt_mean["generated"] - bwt_mean["stored_buffer"]
        gap_vs_noreplay = bwt_mean["generated"] - bwt_mean["no_replay"]
        frontier = _frontier_auc_from_bwt(bwt_mean)

        null_supported = bool(gap_vs_stored < -tie_tol or abs(gap_vs_stored) <= tie_tol)
        out = {
            "bwt_no_replay": bwt_mean["no_replay"],
            "bwt_stored_buffer": bwt_mean["stored_buffer"],
            "bwt_generated": bwt_mean["generated"],
            "backward_transfer": bwt_mean,
            "frontier_auc": frontier,
            "generator_probe_transfer": round(mean(probe_transfer), 4),
            "gap_generated_minus_stored": round(gap_vs_stored, 4),
            "gap_generated_minus_noreplay": round(gap_vs_noreplay, 4),
            "tie_tol": tie_tol,
            "replay_budget": replay_budget,
            "seeds": list(seeds),
            "generated_beats_noreplay": bool(gap_vs_noreplay > tie_tol),
            "generated_ties_stored": bool(abs(gap_vs_stored) <= tie_tol),
            "null_supported": null_supported,
        }
        return out


def _eval(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def _eval_linear_on(xtr, ytr, xte, yte, seed: int, epochs: int = 200, lr: float = 0.05) -> float:
    seed_everything(seed)
    n_classes = int(max(int(ytr.max()), int(yte.max())) + 1)
    head = nn.Linear(xtr.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((head(xte).argmax(-1) == yte).float().mean())


def _frontier_auc_from_bwt(bwt_mean: dict[str, float]) -> float:
    return round(float(sum(max(0.0, 1.0 + v) for v in bwt_mean.values()) / len(bwt_mean)), 4)
