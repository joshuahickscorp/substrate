
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..seeding import seed_everything
from .base import Experiment


class HyperShellHead(nn.Module):

    def __init__(self, dim: int, n_classes: int, hidden: int):
        super().__init__()
        self.dim, self.n_classes = dim, n_classes
        out = dim * n_classes + n_classes  # flat W (n_classes x dim) + b (n_classes)
        self.h = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def weights_from_context(self, context_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = context_x.mean(dim=0, keepdim=True)  # [1, dim], order-invariant context summary
        flat = self.h(ctx).squeeze(0)  # [dim*n_classes + n_classes]
        w = flat[: self.dim * self.n_classes].view(self.n_classes, self.dim)
        b = flat[self.dim * self.n_classes :]
        return w, b

    def forward(self, context_x: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        w, b = self.weights_from_context(context_x)
        return query_x @ w.t() + b


def _static_head(dim: int, n_classes: int) -> nn.Linear:
    return nn.Linear(dim, n_classes)


def _fit_static(head: nn.Module, x, y, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()


def _fit_hypernet(hyper: HyperShellHead, tasks_ctx_query: list[tuple], epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(hyper.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        losses = [F.cross_entropy(hyper(cx, qx), qy) for cx, _cy, qx, qy in tasks_ctx_query]
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def _gradient_tta_curve(static_init: nn.Linear, cx, cy, qx, qy, max_steps: int, lr: float) -> list[float]:
    head = _static_head(static_init.in_features, static_init.out_features)
    head.load_state_dict(static_init.state_dict())
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    curve = []
    for _ in range(max_steps):
        opt.zero_grad()
        F.cross_entropy(head(cx), cy).backward()
        opt.step()
        with torch.no_grad():
            curve.append(_acc(head(qx), qy))
    return curve


class EX4(Experiment):
    id = "ex4_fast_weights"
    metric = (
        "accuracy_held_out",
        "hypernet_acc",
        "static_acc",
        "gradient_tta_acc",
        "adaptation_speed",
        "collapse_cosine",
    )
    baseline = "static head trained once with no eval-time adaptation"
    ablation = "in-context zero-gradient hypernet vs gradient-TTA (same per-task context) and meta-init"
    null_hypothesis = (
        "in-context plasticity does not match gradient plasticity, or the hypernet collapses to a "
        "context-independent average shell"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        n_classes = int(e.n_classes)
        n_train_tasks, n_held_out_tasks = int(e.n_train_tasks), int(e.n_held_out_tasks)
        samples, separation = int(e.samples), float(e.separation)
        ctx_frac = float(e.ctx_frac)
        epochs, lr = int(e.epochs), float(e.lr)
        tta_lr, tta_max_steps = float(e.tta_lr), int(e.tta_max_steps)
        margin = float(e.margin)
        collapse_thresh = float(e.collapse_thresh)

        hyper_accs, static_accs, tta_final_accs, adapt_speeds, collapse_cos = [], [], [], [], []

        for s in seeds:
            seed_everything(s)
            stream = make_task_stream(
                n_tasks=n_train_tasks + n_held_out_tasks,
                dim=dim,
                classes_per_task=n_classes,
                samples_per_task=samples,
                separation=separation,
                incremental="task",
                seed=s,
            )
            train_tasks, held_tasks = stream[:n_train_tasks], stream[n_train_tasks:]

            def split_ctx_query(task):
                n = task.x.shape[0]
                cut = max(n_classes, int(n * ctx_frac))
                return task.x[:cut], task.y[:cut], task.x[cut:], task.y[cut:]

            hyper = HyperShellHead(dim, n_classes, hidden)
            tasks_ctx_query = [split_ctx_query(t) for t in train_tasks]
            _fit_hypernet(hyper, tasks_ctx_query, epochs, lr)

            static = _static_head(dim, n_classes)
            pooled_x = torch.cat([t.x for t in train_tasks], dim=0)
            pooled_y = torch.cat([t.y for t in train_tasks], dim=0)
            _fit_static(static, pooled_x, pooled_y, epochs, lr)

            for t in held_tasks:
                cx, cy, qx, qy = split_ctx_query(t)
                with torch.no_grad():
                    h_logits = hyper(cx, qx)
                    s_logits = static(qx)
                hyper_accs.append(_acc(h_logits, qy))
                static_accs.append(_acc(s_logits, qy))

                curve = _gradient_tta_curve(static, cx, cy, qx, qy, tta_max_steps, tta_lr)
                tta_final_accs.append(curve[-1])
                target = hyper_accs[-1]
                reached = next((i + 1 for i, a in enumerate(curve) if a >= target), None)
                adapt_speeds.append(reached if reached is not None else tta_max_steps)

            if len(held_tasks) >= 2:
                cx_a, _, _, _ = split_ctx_query(held_tasks[0])
                cx_b, _, _, _ = split_ctx_query(held_tasks[1])
                with torch.no_grad():
                    wa, ba = hyper.weights_from_context(cx_a)
                    wb, bb = hyper.weights_from_context(cx_b)
                    flat_a = torch.cat([wa.flatten(), ba.flatten()])
                    flat_b = torch.cat([wb.flatten(), bb.flatten()])
                    cos = float(F.cosine_similarity(flat_a, flat_b, dim=0))
                collapse_cos.append(cos)

        def mean(v):
            return sum(v) / len(v) if v else float("nan")

        hyper_m, static_m, tta_m = mean(hyper_accs), mean(static_accs), mean(tta_final_accs)
        collapse_m = mean(collapse_cos)
        gain_vs_static = hyper_m - static_m
        gain_vs_tta = hyper_m - tta_m
        collapsed = bool(collapse_m >= collapse_thresh)
        beats_static = bool(gain_vs_static > margin)

        out = {
            "hypernet_acc": round(hyper_m, 4),
            "static_acc": round(static_m, 4),
            "gradient_tta_acc": round(tta_m, 4),
            "accuracy_held_out": round(hyper_m, 4),
            "gain_vs_static": round(gain_vs_static, 4),
            "gain_vs_gradient_tta": round(gain_vs_tta, 4),
            "adaptation_speed": round(mean(adapt_speeds), 3),
            "tta_max_steps": tta_max_steps,
            "collapse_cosine": round(collapse_m, 4),
            "collapse_thresh": collapse_thresh,
            "margin": margin,
            "seeds": list(seeds),
            "n_held_out_evals": len(hyper_accs),
            "null_supported": bool((not beats_static) or collapsed),
            "hypernet_beats_static": beats_static,
            "hypernet_matches_gradient_tta": bool(gain_vs_tta >= -margin),
            "collapsed": collapsed,
        }
        return out
