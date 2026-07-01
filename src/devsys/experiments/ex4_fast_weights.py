"""EX4: fast-weight / hypernetwork shells. A hypernetwork h(context_latents) emits the weights of a
tiny linear head in ONE forward pass (in-context, zero-gradient adaptation: no backward pass at eval
time). Compared against two standing controls: a static head trained once and never adapted, and a
gradient-TTA arm that DOES take a few gradient steps at eval time on the same per-task context. A
context-independence (collapse) diagnostic checks whether h ignores its context and just emits a
fixed average shell (cosine similarity of h(context_A) vs h(context_B) across different-class
contexts; near 1.0 means collapse).

NULL: in-context plasticity does not match gradient plasticity, or the hypernet collapses to a
context-independent average shell (the forward pass buys nothing over a fixed head, or it is
indistinguishable from ignoring its input). Taxonomy slot 4 (predictor/hypernet too weak) or 9
(fast-weight generation collapses to a slow-weight average). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

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
    """Hypernetwork h(context_latents_mean) -> flat vector reshaped into (W, b) of a linear head
    dim -> n_classes. Adaptation is a single forward pass through h conditioned on a per-task
    context; no gradient step touches the shell weights at eval time."""

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
    """Meta-init: train h across tasks so a single forward pass, conditioned on a task's context,
    scores well on that task's query set. This is the 'meta-init' standing control folded into the
    hypernet's own training (it must learn to USE context, not just memorize one task)."""
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
    """A fresh copy of the static-head architecture, adapted from scratch by gradient steps on the
    task's CONTEXT set (labeled, same context the hypernet sees), evaluated on the query set after
    each step. Returns the per-step query accuracy curve (length max_steps)."""
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
            # task-incremental stream: every task reuses the same label space 0..n_classes-1 with
            # independent cluster geometry per task, so a context must be READ to solve a task.
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

            # meta-init: hypernet trained across TRAIN tasks only, held-out tasks are unseen geometry
            hyper = HyperShellHead(dim, n_classes, hidden)
            tasks_ctx_query = [split_ctx_query(t) for t in train_tasks]
            _fit_hypernet(hyper, tasks_ctx_query, epochs, lr)

            # static baseline: one head trained on the POOLED train-task data, never adapted per task
            static = _static_head(dim, n_classes)
            pooled_x = torch.cat([t.x for t in train_tasks], dim=0)
            pooled_y = torch.cat([t.y for t in train_tasks], dim=0)
            _fit_static(static, pooled_x, pooled_y, epochs, lr)

            # evaluate on held-out tasks (unseen cluster geometry, forces genuine context use)
            for t in held_tasks:
                cx, cy, qx, qy = split_ctx_query(t)
                with torch.no_grad():
                    h_logits = hyper(cx, qx)
                    s_logits = static(qx)
                hyper_accs.append(_acc(h_logits, qy))
                static_accs.append(_acc(s_logits, qy))

                # gradient-TTA control: fresh copy of the static init, adapted on the SAME context
                curve = _gradient_tta_curve(static, cx, cy, qx, qy, tta_max_steps, tta_lr)
                tta_final_accs.append(curve[-1])
                target = hyper_accs[-1]
                reached = next((i + 1 for i, a in enumerate(curve) if a >= target), None)
                adapt_speeds.append(reached if reached is not None else tta_max_steps)

            # collapse diagnostic: does h ignore its context. Feed contexts from two DIFFERENT
            # held-out tasks (different class geometry) and compare the emitted weight vectors.
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
            # the explicit null: in-context adaptation does not beat the static head within the
            # seed/margin spread, OR the hypernet has collapsed to a context-independent average shell
            "null_supported": bool((not beats_static) or collapsed),
            "hypernet_beats_static": beats_static,
            "hypernet_matches_gradient_tta": bool(gain_vs_tta >= -margin),
            "collapsed": collapsed,
        }
        return out
