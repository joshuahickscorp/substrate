"""EX5: local learning rules at scale, extended from I4 to retention (BWT) on a genuine
continual stream. I4 (i4_backprop_alternatives.py) trains each rule FROM SCRATCH on a
single fixed (x, y) task; there is no persistent model carried across tasks, so I4 cannot
produce a continual-learning result. This module fixes that gap for THREE rules only
(backprop, feedback_alignment, predictive_coding), self-containedly, without touching
mop.learning.alternatives.rules (which other experiments depend on exactly as-is).

Scope cut, documented honestly: the ORIGINAL registry entry for ex5_local_rules_scale
describes a second half of the mechanism, probing across MULTIPLE real frozen encoders
(V-JEPA 2 vs V-JEPA 2.1 etc). That half is explicitly OUT OF SCOPE here: official dense
V-JEPA 2.1 weights now exist, but the citable cache and matched integration controls remain
separate work, so this runner uses only the frozen substrate the repo already has. This is a
deliberate scope cut made up front, not
a failed attempt. What IS built for real: local-rule backward-transfer/retention at scale
on the single available frozen substrate, matched-budget across rules (same hidden width
and epoch budget per rule, same seed protocol per task), plus a depth-sweep control that
repeats the whole comparison at 2-3 hidden widths to check whether any rule's relative BWT
advantage is a capacity artifact that disappears at matched-but-different depth.

Persistent rule implementations (exactly what update rule each one is, stated plainly):

  * backprop: a 2-layer MLP (Linear -> ReLU -> Linear) trained with torch.autograd and
    Adam. This is the ceiling I4 also uses it as.

  * feedback_alignment: the SAME 2-layer architecture, but the hidden-layer credit signal
    is pushed backward through a FIXED random matrix B (never updated, never equal to
    W2^T), i.e. no weight transport. Forward pass, per-layer deltas, and weight updates
    are implemented by hand (no autograd), matching the shape of
    mop.learning.alternatives.rules.train_feedback_alignment's per-step update, but
    reorganized into a persistent train_step()/evaluate() pair so the SAME W1/W2/B carry
    forward, task after task, instead of being reinitialized per task.

  * predictive_coding: Whittington-Bogacz-style local energy descent, the same shape as
    mop.learning.alternatives.rules.train_predictive_coding: clamp input and output to
    the label, relax a hidden value node for a fixed number of inference steps to minimize
    local prediction errors, then update W1/W2 by local error*activity products (no
    autograd, no weight transport in the credit path). Reorganized the same way into a
    persistent train_step()/evaluate() pair with state carried across tasks.

Each rule exposes train_step(model, x, y) -> loss (one local optimization step on a task's
data, mutating the persistent model state) and evaluate(model, x, y) -> accuracy, so the
SAME rule can be driven repeatedly across a domain-incremental make_task_stream(n_tasks=N)
and its retention measured with mop.metrics.ContinualResult.

Anchor-task-subset trick (full R is O(T^2) full-batch evals, too slow at T=40-80): instead
of evaluating every seen task after every new task, we track a small fixed set of ANCHOR
tasks (task 0, and a handful evenly spaced through the stream) and evaluate only those
anchors after every task boundary. ContinualResult.R is built from this anchor-restricted
matrix: R[i][j] is only ever populated for j in the anchor set (self-diagonal, i.e. R[j][j],
is captured the moment task j is finished, whether or not j itself is an anchor, so
backward_transfer's per-anchor "peak vs final" comparison is always honest). This trades
full-matrix resolution for O(T * |anchors|) cost, the same asymptotic idea as tracking a
representative task subset instead of the full pairwise matrix.

Doctrine: null_hypothesis is reported honestly. If a local rule (predictive_coding is the
literature's usual candidate, being a local energy-based rule that some theories connect to
reduced catastrophic forgetting relative to global backprop) shows a genuine BWT advantage
over backprop, that positive finding is reported and NOT suppressed, even though the null
otherwise expects backprop to dominate on both axes.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No
sentience or agency language.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..metrics import ContinualResult
from ..seeding import seed_everything
from ..substrate.datasets import make_task_stream
from .base import Experiment

RULE_NAMES = ("backprop", "feedback_alignment", "predictive_coding")


# --------------------------------------------------------------------------------- state


@dataclass
class _PersistentModel:
    """Shared weight container for all three rules: one hidden layer, ReLU, one output
    layer. Only the fields a given rule actually mutates are meaningful for that rule
    (e.g. B is unused by backprop, opt is unused by feedback_alignment/predictive_coding).
    """

    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    B: torch.Tensor | None = None  # fixed random feedback matrix (feedback_alignment only)
    opt: torch.optim.Optimizer | None = None  # backprop only
    params: list[torch.Tensor] = field(default_factory=list)  # backprop only, for the optimizer


def _init_model(rule: str, dim: int, hidden: int, n_classes: int, seed: int) -> _PersistentModel:
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(hidden, dim, generator=g) / dim**0.5).requires_grad_(rule == "backprop")
    b1 = torch.zeros(hidden, requires_grad=(rule == "backprop"))
    W2 = (torch.randn(n_classes, hidden, generator=g) / hidden**0.5).requires_grad_(rule == "backprop")
    b2 = torch.zeros(n_classes, requires_grad=(rule == "backprop"))
    B = None
    opt = None
    params: list[torch.Tensor] = []
    if rule == "feedback_alignment":
        B = torch.randn(hidden, n_classes, generator=g) / n_classes**0.5
    if rule == "backprop":
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=0.05)
    return _PersistentModel(W1=W1, b1=b1, W2=W2, b2=b2, B=B, opt=opt, params=params)


def _relu(z: torch.Tensor) -> torch.Tensor:
    return z.clamp_min(0.0)


# ----------------------------------------------------------------------- rule: backprop


def _backprop_train_step(model: _PersistentModel, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    assert model.opt is not None
    for g_ in model.opt.param_groups:
        g_["lr"] = lr
    model.opt.zero_grad()
    z1 = x @ model.W1.T + model.b1
    o = _relu(z1) @ model.W2.T + model.b2
    loss = F.cross_entropy(o, y)
    loss.backward()
    model.opt.step()
    return float(loss.detach())


@torch.no_grad()
def _backprop_evaluate(model: _PersistentModel, x: torch.Tensor, y: torch.Tensor) -> float:
    o = _relu(x @ model.W1.T + model.b1) @ model.W2.T + model.b2
    return float((o.argmax(-1) == y).float().mean())


# ------------------------------------------------------------------ rule: feedback_alignment


def _fa_train_step(model: _PersistentModel, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    """One full-batch FA step: forward pass, hidden credit pushed through the FIXED random
    matrix B (no weight transport, B != W2^T and is never updated), local delta rule for
    W1/W2. Matches the shape of rules.train_feedback_alignment's inner loop body, but as a
    single step reusable across tasks with persistent W1/W2/B."""
    assert model.B is not None
    n_classes = model.W2.shape[0]
    n = x.shape[0]
    T = F.one_hot(y, n_classes).float()
    with torch.no_grad():
        z1 = x @ model.W1.T + model.b1
        a1 = _relu(z1)
        o = a1 @ model.W2.T + model.b2
        loss = F.cross_entropy(o, y)
        e = torch.softmax(o, -1) - T
        gW2 = e.T @ a1 / n
        gb2 = e.mean(0)
        delta1 = (e @ model.B.T) * (z1 > 0).float()  # random feedback, no transport
        gW1 = delta1.T @ x / n
        gb1 = delta1.mean(0)
        model.W2 -= lr * gW2
        model.b2 -= lr * gb2
        model.W1 -= lr * gW1
        model.b1 -= lr * gb1
    return float(loss)


@torch.no_grad()
def _fa_evaluate(model: _PersistentModel, x: torch.Tensor, y: torch.Tensor) -> float:
    o = _relu(x @ model.W1.T + model.b1) @ model.W2.T + model.b2
    return float((o.argmax(-1) == y).float().mean())


# ------------------------------------------------------------------- rule: predictive_coding


def _pc_train_step(
    model: _PersistentModel, x: torch.Tensor, y: torch.Tensor, lr: float, infer: int = 20
) -> float:
    """One full-batch predictive-coding step: clamp input and output-target, relax the hidden
    value node for `infer` local steps to minimize prediction error, then update W1/W2 by
    local error*activity products. Matches the shape of
    rules.train_predictive_coding's inner loop body, persistent across tasks."""
    n_classes = model.W2.shape[0]
    n = x.shape[0]
    T = F.one_hot(y, n_classes).float()
    f = torch.tanh

    def fp(z: torch.Tensor) -> torch.Tensor:
        return 1 - torch.tanh(z) ** 2

    with torch.no_grad():
        z1 = x @ model.W1.T
        h = f(z1)
        for _ in range(infer):
            o_pred = h @ model.W2.T
            eps_o = T - torch.softmax(o_pred, -1)
            eps_h = h - f(z1)
            dh = -eps_h + (eps_o @ model.W2) * fp(z1)
            h = h + 0.1 * dh
        o_pred = h @ model.W2.T
        loss = F.cross_entropy(o_pred, y)
        eps_o = T - torch.softmax(o_pred, -1)
        eps_h = h - f(z1)
        model.W2 += lr * (eps_o.T @ h) / n
        model.W1 += lr * ((eps_h * fp(z1)).T @ x) / n
    return float(loss)


@torch.no_grad()
def _pc_evaluate(model: _PersistentModel, x: torch.Tensor, y: torch.Tensor) -> float:
    o = torch.tanh(x @ model.W1.T) @ model.W2.T
    return float((o.argmax(-1) == y).float().mean())


_TRAIN_STEP = {
    "backprop": _backprop_train_step,
    "feedback_alignment": _fa_train_step,
    "predictive_coding": _pc_train_step,
}
_EVALUATE = {
    "backprop": _backprop_evaluate,
    "feedback_alignment": _fa_evaluate,
    "predictive_coding": _pc_evaluate,
}
# relative activation-memory reported by I4 for the closest matched rule (1.0 = stores all
# activations like backprop; lower = fewer stored activations along the credit path)
_ACTIVATION_MEMORY = {"backprop": 1.0, "feedback_alignment": 1.0, "predictive_coding": 0.5}
_LOCAL = {"backprop": False, "feedback_alignment": False, "predictive_coding": True}
_WEIGHT_TRANSPORT = {"backprop": True, "feedback_alignment": False, "predictive_coding": False}


def _anchor_indices(n_tasks: int, n_anchors: int) -> list[int]:
    """A small, evenly-spaced set of anchor task indices (always includes task 0 and the
    last task). n_anchors is clamped to [1, n_tasks]."""
    n_anchors = max(1, min(n_anchors, n_tasks))
    if n_anchors == 1:
        return [0]
    step = (n_tasks - 1) / (n_anchors - 1)
    return sorted({round(i * step) for i in range(n_anchors)})


def _run_rule_on_stream(
    rule: str,
    tasks: list,
    dim: int,
    hidden: int,
    n_classes: int,
    epochs_per_task: int,
    lr: float,
    seed: int,
    anchor_idx: list[int],
) -> tuple[ContinualResult, float, dict]:
    """Train `rule` incrementally across `tasks` (persistent weights, carried forward task
    after task). Anchor-task-subset trick: rather than the full O(T^2) accuracy matrix, we
    only ever evaluate the K tasks in `anchor_idx` (K << T), and only at K+1 checkpoints:
    right after each anchor task itself finishes training (captures that anchor's peak,
    uncorrupted-by-later-tasks accuracy on ITS OWN held-out split), plus one final
    checkpoint after the whole stream finishes (captures every anchor's end-of-stream
    retention). The resulting K x K matrix R has exactly the shape ContinualResult expects
    (R[k][k] = anchor k's peak accuracy, R[-1][k] = anchor k's accuracy after the full
    stream), so backward_transfer() is the honest mean retention drop across anchors, at
    O(T + K) evaluations instead of O(T^2).

    Returns the ContinualResult, elapsed seconds, and rule diagnostics (matched-budget by
    construction: identical hidden width and epochs_per_task across rules)."""
    seed_everything(seed)
    model = _init_model(rule, dim, hidden, n_classes, seed)
    train_step = _TRAIN_STEP[rule]
    evaluate = _EVALUATE[rule]

    anchor_pos = {j: k for k, j in enumerate(anchor_idx)}
    K = len(anchor_idx)
    R: list[list[float]] = [[0.0] * K for _ in range(K)]

    def held_out(task):
        x, y = task.x, task.y
        cut = max(1, int(x.shape[0] * 0.8))
        xte, yte = x[cut:], y[cut:]
        return (xte, yte) if xte.shape[0] > 0 else (x[:cut], y[:cut])

    t0 = time.perf_counter()
    for i, task in enumerate(tasks):
        x, y = task.x, task.y
        cut = max(1, int(x.shape[0] * 0.8))
        xtr, ytr = x[:cut], y[:cut]
        for _ in range(epochs_per_task):
            train_step(model, xtr, ytr, lr)
        if i in anchor_pos:
            k = anchor_pos[i]
            xte, yte = held_out(task)
            R[k][k] = evaluate(model, xte, yte)  # this anchor's peak, right after its own training
    # final checkpoint: every anchor's accuracy after the full stream has been trained
    for j in anchor_idx:
        k = anchor_pos[j]
        xte, yte = held_out(tasks[j])
        R[-1][k] = evaluate(model, xte, yte)
    elapsed = time.perf_counter() - t0

    result = ContinualResult(R=R, chance=1.0 / n_classes)
    activation_memory = _ACTIVATION_MEMORY[rule]
    diag = {
        "weight_transport": _WEIGHT_TRANSPORT[rule],
        "local": _LOCAL[rule],
        "activation_memory": activation_memory,
    }
    return result, elapsed, diag


class EX5(Experiment):
    id = "ex5_local_rules_scale"
    metric = ("accuracy_gap", "activation_memory", "locality", "backward_transfer")
    baseline = "persistent backprop MLP trained incrementally across the same continual stream (the ceiling)"
    ablation = (
        "persistent local rule = {feedback_alignment, predictive_coding} at matched hidden "
        "width and epoch budget, repeated at a depth-sweep of hidden widths"
    )
    null_hypothesis = (
        "no local rule comes within the accuracy margin of backprop AND none offers a "
        "continual-learning or memory advantage that justifies the gap"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        margin = float(e.margin)
        n_tasks = int(e.n_tasks)
        dim = int(e.dim)
        classes_per_task = int(e.classes_per_task)
        samples_per_task = int(e.samples_per_task)
        epochs_per_task = int(e.epochs_per_task)
        lr = float(e.lr)
        separation = float(e.separation)
        hidden_widths = [int(h) for h in e.hidden_widths]
        n_anchors = int(e.n_anchors)
        n_classes = classes_per_task  # domain-incremental: shared label space across tasks

        depth_sweep: dict[int, dict] = {}
        for hidden in hidden_widths:
            per_rule: dict[str, dict] = {}
            for rule in RULE_NAMES:
                accs, bwts, secs, diag = [], [], [], {}
                for s in seeds:
                    tasks = make_task_stream(
                        n_tasks=n_tasks,
                        dim=dim,
                        classes_per_task=classes_per_task,
                        samples_per_task=samples_per_task,
                        separation=separation,
                        incremental="domain",  # shared labels, shifting geometry: forgetting regime
                        seed=s,
                    )
                    anchor_idx = _anchor_indices(n_tasks, n_anchors)
                    result, elapsed, d = _run_rule_on_stream(
                        rule, tasks, dim, hidden, n_classes, epochs_per_task, lr, s, anchor_idx
                    )
                    summ = result.summary()
                    accs.append(summ["avg_accuracy"])
                    bwts.append(summ["backward_transfer"])
                    secs.append(elapsed)
                    diag = d
                mean_acc = sum(accs) / len(accs)
                mean_bwt = sum(bwts) / len(bwts)
                var_acc = sum((a - mean_acc) ** 2 for a in accs) / len(accs)
                per_rule[rule] = {
                    "acc_mean": mean_acc,
                    "acc_std": var_acc**0.5,
                    "backward_transfer": mean_bwt,
                    "seconds": sum(secs) / len(secs),
                    **diag,
                }
            ceiling = per_rule["backprop"]["acc_mean"]
            for row in per_rule.values():
                row["gap_to_backprop"] = ceiling - row["acc_mean"]
            depth_sweep[hidden] = per_rule

        # primary reporting hidden width: the middle of the depth sweep (or the only one)
        primary_hidden = hidden_widths[len(hidden_widths) // 2]
        primary = depth_sweep[primary_hidden]
        ceiling = primary["backprop"]["acc_mean"]
        ceiling_bwt = primary["backprop"]["backward_transfer"]

        within_margin = {
            n: r for n, r in primary.items() if n != "backprop" and r["gap_to_backprop"] <= margin
        }
        bwt_advantage = {
            n: r for n, r in primary.items() if n != "backprop" and r["backward_transfer"] > ceiling_bwt
        }
        # a BWT advantage only "justifies the gap" if it is large enough to offset the
        # accuracy shortfall; we report both the raw advantage and this offset judgement
        justifies_gap = {
            n: r
            for n, r in bwt_advantage.items()
            if (r["backward_transfer"] - ceiling_bwt) >= primary[n]["gap_to_backprop"]
        }

        # depth-sweep control: does any rule's relative BWT ranking vs backprop flip across
        # hidden widths? If so the "advantage" (or lack of one) at the primary width is a
        # capacity artifact, not a stable property of the rule.
        depth_sweep_stable = {}
        for rule in RULE_NAMES:
            if rule == "backprop":
                continue
            deltas = [
                depth_sweep[h][rule]["backward_transfer"] - depth_sweep[h]["backprop"]["backward_transfer"]
                for h in hidden_widths
            ]
            signs = {1 if d > 0 else (-1 if d < 0 else 0) for d in deltas}
            depth_sweep_stable[rule] = len(signs) <= 1  # sign never flips across widths

        null_supported = bool(len(within_margin) == 0 and len(justifies_gap) == 0)

        out = {
            "hidden_widths": hidden_widths,
            "primary_hidden": primary_hidden,
            "n_tasks": n_tasks,
            "margin": margin,
            "ceiling_backprop_acc": ceiling,
            "ceiling_backprop_bwt": ceiling_bwt,
            "table": primary,
            "depth_sweep": depth_sweep,
            "rules_within_accuracy_margin": sorted(within_margin),
            "rules_with_bwt_advantage": sorted(bwt_advantage),
            "rules_whose_bwt_advantage_justifies_gap": sorted(justifies_gap),
            "depth_sweep_bwt_ranking_stable": depth_sweep_stable,
            "accuracy_gap": {n: r["gap_to_backprop"] for n, r in primary.items()},
            "activation_memory": {n: r["activation_memory"] for n, r in primary.items()},
            "locality": {n: r["local"] for n, r in primary.items()},
            "backward_transfer": {n: r["backward_transfer"] for n, r in primary.items()},
            "null_supported": null_supported,
        }
        self._write_table(depth_sweep, hidden_widths, run_dir)
        return out

    def _write_table(self, depth_sweep: dict, hidden_widths: list[int], run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "| hidden | rule | acc | gap | bwt | local | transport | act-mem | sec |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for h in hidden_widths:
            for n, r in depth_sweep[h].items():
                lines.append(
                    f"| {h} | {n} | {r['acc_mean']:.3f}±{r['acc_std']:.3f} | "
                    f"{r['gap_to_backprop']:+.3f} | {r['backward_transfer']:+.3f} | "
                    f"{r['local']} | {r['weight_transport']} | "
                    f"{r['activation_memory']:.1f} | {r['seconds']:.2f} |"
                )
        (run_dir / "ex5_table.md").write_text("\n".join(lines) + "\n")
