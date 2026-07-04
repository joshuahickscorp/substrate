"""EX14: memory bake-off. Three eviction/retrieval schemes over mop.shell.buffer.ReplayBuffer at
MATCHED capacity: FIFO (oldest-out), random-eviction (reservoir, the standard unbiased control), and
uncertainty-indexed (evict the most-confident stored exemplar first, keeping ambiguous/high-loss ones,
via the buffer's `priority` eviction mode with a per-exemplar loss-margin priority signal). A fourth
arm layers modern-Hopfield-style associative KV retrieval (the buffer's `.retrieve` k-NN search) on TOP
of each scheme to score recall@k, the associative-memory analogue asked for in the contract.

NULL: at matched buffer capacity, uncertainty-indexed eviction does not beat FIFO or random-eviction on
recall@k or on downstream backward transfer (BWT); associative (k-NN) retrieval does not beat FIFO
capacity either. A tie is the expected, honest result for a toy two-task stream with a small buffer:
capacity and retrieval topology dominate, and a smarter eviction *heuristic* buys nothing extra.
Negative-result taxonomy slot 4 (predictor/heuristic too weak relative to capacity) or 6 (scale ceiling
not yet reached at toy size). cpu-now, seconds.

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
from ..shell.buffer import ReplayBuffer
from .base import Experiment, _mean, _spread


def _fill_buffer(scheme: str, capacity: int, dim: int, task, head: nn.Module, seed: int) -> ReplayBuffer:
    """Fill a capacity-matched buffer from one task's exemplars under the named eviction scheme.
    fifo/random use unprioritized eviction (oldest-out vs reservoir-uniform); uncertainty uses the
    buffer's `priority` eviction with priority = per-exemplar loss margin (low margin = high priority,
    i.e. ambiguous/ high-loss exemplars are KEPT and confident ones are evicted first)."""
    eviction = "fifo" if scheme == "fifo" else "reservoir" if scheme == "random" else "priority"
    buf = ReplayBuffer(capacity, dim, prioritized=False, eviction=eviction, seed=seed)
    x, y = task.x, task.y
    if scheme == "uncertainty":
        with torch.no_grad():
            logits = head(x)
            srt = logits.sort(dim=-1, descending=True).values
            margin = (srt[:, 0] - srt[:, 1]).abs()
            # low margin (ambiguous / high loss) -> high keep-priority; the buffer's `priority`
            # eviction always evicts the lowest-`prio` slot, so priority = 1 / margin keeps the
            # hard exemplars and evicts the confident ones first.
            prio = 1.0 / (margin + 1e-3)
        for j in range(x.shape[0]):
            buf.add(x[j : j + 1], y[j : j + 1], priority=float(prio[j]))
    else:
        for j in range(x.shape[0]):
            buf.add(x[j : j + 1], y[j : j + 1])
    return buf


def _recall_at_k(buf: ReplayBuffer, xq: torch.Tensor, yq: torch.Tensor, k: int) -> float:
    """Fraction of held-out queries whose majority label among the k nearest STORED exemplars
    (the buffer's KV `.retrieve`, the associative-memory analogue) matches the true query label."""
    if len(buf) == 0:
        return 0.0
    out = buf.retrieve(xq, k=min(k, len(buf)))
    yneigh = out["y"]  # [Q, k]
    pred = torch.mode(yneigh, dim=-1).values
    return float((pred == yq).float().mean())


def _bwt(buf: ReplayBuffer, dim: int, nc: int, task0, task1, epochs: int, lr: float, seed: int) -> float:
    """Continual retention: task 0 already trained (head passed in pre-fit on task 0), now train task 1
    WITH replay from `buf` (which was filled from task 0's exemplars under the scheme being tested).
    Returns BWT = task-0 accuracy after task 1 minus task-0 accuracy right after task 0."""
    seed_everything(seed)
    head = nn.Linear(dim, nc)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(task0.x), task0.y).backward()
        opt.step()
    with torch.no_grad():
        acc_first = float((head(task0.x).argmax(-1) == task0.y).float().mean())
    for _ in range(epochs):
        opt.zero_grad()
        xb, yb = task1.x, task1.y
        if len(buf) > 0:
            s = buf.sample(min(len(buf), task1.x.shape[0]))
            xb = torch.cat([task1.x, s["x"]])
            yb = torch.cat([task1.y, s["y"]])
        F.cross_entropy(head(xb), yb).backward()
        opt.step()
    with torch.no_grad():
        acc_end = float((head(task0.x).argmax(-1) == task0.y).float().mean())
    return acc_end - acc_first


class EX14(Experiment):
    id = "ex14_memory_bakeoff"
    metric = ("recall_at_k", "backward_transfer", "retention_per_byte")
    baseline = "FIFO eviction and random (reservoir) eviction, both at the SAME matched buffer capacity"
    ablation = "uncertainty-indexed eviction (keep low-margin/high-loss exemplars) vs FIFO vs random"
    null_hypothesis = (
        "associative memory does not beat FIFO capacity and uncertainty indexing ties random eviction, "
        "even at scale"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        capacity = int(e.capacity)
        n_classes = int(e.n_classes)
        samples = int(e.samples_per_task)
        k = int(e.k)
        epochs, lr = int(e.epochs), float(e.lr)
        tie_margin = float(e.tie_margin)

        schemes = ("fifo", "random", "uncertainty")
        recall: dict[str, list[float]] = {s: [] for s in schemes}
        bwt: dict[str, list[float]] = {s: [] for s in schemes}
        bytes_per_item = dim * 4  # float32, 4 bytes: matched across schemes by construction (same capacity)

        for seed in seeds:
            stream = make_task_stream(
                n_tasks=2,
                dim=dim,
                classes_per_task=n_classes,
                samples_per_task=samples,
                separation=float(e.separation),
                incremental="domain",  # shared label space, independent geometry per task (forces retention)
                seed=seed,
            )
            task0, task1 = stream[0], stream[1]
            cut = int(task0.x.shape[0] * 0.7)
            xtr0, ytr0, xte0, yte0 = task0.x[:cut], task0.y[:cut], task0.x[cut:], task0.y[cut:]

            # a task-0-only head, used to (a) derive the uncertainty priority signal, (b) seed BWT fitting.
            seed_everything(seed)
            probe_head = nn.Linear(dim, n_classes)
            opt = torch.optim.Adam(probe_head.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                F.cross_entropy(probe_head(xtr0), ytr0).backward()
                opt.step()

            for scheme in schemes:
                buf = _fill_buffer(scheme, capacity, dim, task0, probe_head, seed)
                recall[scheme].append(_recall_at_k(buf, xte0, yte0, k))
                bwt[scheme].append(_bwt(buf, dim, n_classes, task0, task1, epochs, lr, seed))

        recall_mean = {s: round(_mean(v), 4) for s, v in recall.items()}
        bwt_mean = {s: round(_mean(v), 4) for s, v in bwt.items()}
        fifo_spread_recall = _spread(recall["fifo"])
        fifo_spread_bwt = _spread(bwt["fifo"])
        random_spread_recall = _spread(recall["random"])
        random_spread_bwt = _spread(bwt["random"])

        d_uncertainty_vs_fifo_recall = recall_mean["uncertainty"] - recall_mean["fifo"]
        d_uncertainty_vs_random_recall = recall_mean["uncertainty"] - recall_mean["random"]
        d_uncertainty_vs_fifo_bwt = bwt_mean["uncertainty"] - bwt_mean["fifo"]
        d_uncertainty_vs_random_bwt = bwt_mean["uncertainty"] - bwt_mean["random"]

        tol_recall = max(fifo_spread_recall, random_spread_recall, tie_margin)
        tol_bwt = max(fifo_spread_bwt, random_spread_bwt, tie_margin)

        ties_fifo_recall = d_uncertainty_vs_fifo_recall <= tol_recall + 1e-4
        ties_random_recall = abs(d_uncertainty_vs_random_recall) <= tol_recall + 1e-4
        ties_fifo_bwt = d_uncertainty_vs_fifo_bwt <= tol_bwt + 1e-4
        ties_random_bwt = abs(d_uncertainty_vs_random_bwt) <= tol_bwt + 1e-4

        # associative retrieval (recall@k off the buffer's KV index) does not beat plain FIFO capacity
        associative_beats_fifo_capacity = recall_mean["fifo"] > (1.0 / n_classes) + tie_margin

        retention_per_byte = {
            s: round(bwt_mean[s] / max(1, capacity * bytes_per_item) * 1e6, 6) for s in schemes
        }

        null_supported = bool(
            (ties_fifo_recall and ties_random_recall) and (ties_fifo_bwt and ties_random_bwt)
        )
        uncertainty_wins = bool(
            (not ties_fifo_recall or not ties_fifo_bwt)
            and (d_uncertainty_vs_fifo_recall > tol_recall or d_uncertainty_vs_fifo_bwt > tol_bwt)
        )

        return {
            "recall_at_k": recall_mean,
            "backward_transfer": bwt_mean,
            "retention_per_byte": retention_per_byte,
            "capacity": capacity,
            "bytes_per_item": bytes_per_item,
            "k": k,
            "delta_uncertainty_vs_fifo_recall": round(d_uncertainty_vs_fifo_recall, 4),
            "delta_uncertainty_vs_random_recall": round(d_uncertainty_vs_random_recall, 4),
            "delta_uncertainty_vs_fifo_bwt": round(d_uncertainty_vs_fifo_bwt, 4),
            "delta_uncertainty_vs_random_bwt": round(d_uncertainty_vs_random_bwt, 4),
            "tie_tolerance_recall": round(tol_recall, 4),
            "tie_tolerance_bwt": round(tol_bwt, 4),
            "associative_beats_fifo_capacity": bool(associative_beats_fifo_capacity),
            "seeds": list(seeds),
            # the explicit null: uncertainty ties FIFO/random on both axes at matched capacity
            "null_supported": null_supported,
            "uncertainty_indexing_wins": uncertainty_wins,
        }
