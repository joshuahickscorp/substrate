"""E7: sparse / modular developmental predictor (interference reduction at matched params).

On a domain-incremental latent stream we train three HEAD variants of EQUAL parameter count
sequentially and read off forgetting (BWT): (a) a dense MLP, (b) a k-WTA head that masks all
but the top-k hidden units on the forward pass (sparse activation, lateral-inhibition analog),
(c) a tiny MoE with a few expert MLPs and a softmax router. The corpus control is explicit:
interference reduction must not be just more capacity, so the dense baseline is parameter
matched and the k-WTA head shares the dense head's exact shape (sparsity is the only change).

NULL (6.7 / Experiment 7): no sparse or modular variant beats the parameter-matched dense
baseline on interference; the reduction, if any, was just capacity. We return sparse_beats_dense
and every variant's BWT. Any speedup claim is reported SEPARATELY and marked gpu-later: on dense
hardware k-WTA and MoE do not run faster (the corpus says so), so wall-clock here is diagnostic
only and never a success criterion.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch import nn  # noqa: E402

from ..devices import DeviceInfo, safe_to  # noqa: E402
from ..metrics import ContinualResult  # noqa: E402
from ..seeding import seed_everything  # noqa: E402
from ..substrate.datasets import Task, make_task_stream  # noqa: E402
from .base import Experiment  # noqa: E402


class DenseHead(nn.Module):
    """latent -> hidden (GELU) -> classes. The parameter-matched reference."""

    def __init__(self, dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class KWTAHead(nn.Module):
    """Identical shape to DenseHead; the only change is a top-k mask on the hidden layer, so
    each input fires k of `hidden` units (sparse activation). Same parameter count as dense."""

    def __init__(self, dim: int, hidden: int, n_classes: int, k: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.k = max(1, min(k, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x))
        thresh = h.topk(self.k, dim=-1).values[..., -1:]
        return self.fc2(h * (h >= thresh))


class MoEHead(nn.Module):
    """A few expert linear maps (dim->classes) plus a softmax router (dim->experts). Expert
    widths are chosen so total params match the dense head; the router is the small overhead the
    corpus warns can dominate. Returns logits and the per-batch routing distribution."""

    def __init__(self, dim: int, n_classes: int, n_experts: int, expert_hidden: int):
        super().__init__()
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, expert_hidden), nn.GELU(), nn.Linear(expert_hidden, n_classes))
            for _ in range(n_experts)
        )
        self.last_gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.router(x), dim=-1)  # [B, E]
        self.last_gates = gates.detach()
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C]
        return (gates.unsqueeze(-1) * stacked).sum(dim=1)


def _split(task: Task, frac: float = 0.8) -> tuple[Task, Task]:
    n = int(task.x.shape[0] * frac)
    tr = Task(task.name, task.x[:n], task.y[:n], n_classes=task.n_classes, task_id=task.task_id)
    te = Task(task.name, task.x[n:], task.y[n:], n_classes=task.n_classes, task_id=task.task_id)
    return tr, te


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _routing_entropy(gates: torch.Tensor) -> float:
    """Mean entropy (nats) of the per-sample expert distribution. High and flat => the router
    never specialized (a corpus null signature); dropping => modules carved up the stream."""
    p = gates.clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean())


class E7(Experiment):
    id = "e7_sparse"
    metric = ("backward_transfer", "forgetting", "routing_entropy", "wall_clock_s")
    baseline = "dense MLP head of equal parameter count (interference must not be just capacity)"
    ablation = "head variant in {dense, k-WTA top-k activation, MoE experts + softmax router}"
    null_hypothesis = (
        "no sparse or modular head beats the parameter-matched dense head on interference "
        "(backward transfer); any reduction was just capacity. Speedup, reported separately, "
        "is not a success criterion and is not expected on dense hardware"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        s = e.stream
        dev = device.device
        seed_everything(int(cfg.seed))
        raw = make_task_stream(
            n_tasks=int(s.n_tasks),
            dim=int(s.dim),
            classes_per_task=int(s.classes_per_task),
            samples_per_task=int(s.samples_per_task),
            separation=float(s.separation),
            incremental="domain",  # shared labels, geometry rotates => reliable interference
            seed=int(cfg.seed),
        )
        dim = int(s.dim)
        n_classes = raw[0].n_classes
        train = [_split(t)[0] for t in raw]
        test = [_split(t)[1] for t in raw]
        chance = 1.0 / n_classes
        hidden = int(e.hidden)
        k = int(e.kwta_k)
        n_experts = int(e.n_experts)

        builders = {
            "dense": lambda: DenseHead(dim, hidden, n_classes),
            "kwta": lambda: KWTAHead(dim, hidden, n_classes, k),
            "moe": lambda: MoEHead(
                dim, n_classes, n_experts, self._matched_expert_hidden(dim, hidden, n_classes, n_experts)
            ),
        }

        arms: dict[str, dict] = {}
        for name, build in builders.items():
            arms[name] = self._run_arm(name, build, train, test, dim, chance, cfg, dev)

        dense_bwt = arms["dense"]["bwt"]
        sparse_arms = {n: a for n, a in arms.items() if n != "dense"}
        margin = float(e.margin)
        beats = {n: (a["bwt"] - dense_bwt) for n, a in sparse_arms.items()}
        best_sparse = max(beats, key=lambda n: beats[n])
        sparse_beats_dense = bool(beats[best_sparse] > margin)  # the explicit null check

        out = {
            "chance": chance,
            "n_params": {n: a["n_params"] for n, a in arms.items()},
            "param_matched": self._param_matched(arms),
            "bwt": {n: a["bwt"] for n, a in arms.items()},
            "forgetting": {n: -a["bwt"] for n, a in arms.items()},
            "avg_accuracy": {n: a["avg_accuracy"] for n, a in arms.items()},
            "routing_entropy": {
                "moe_first": arms["moe"]["routing_entropy_first"],
                "moe_last": arms["moe"]["routing_entropy_last"],
                "dropped": arms["moe"]["routing_entropy_first"] - arms["moe"]["routing_entropy_last"],
            },
            "dense_bwt": dense_bwt,
            "best_sparse": best_sparse,
            "bwt_gain_over_dense": beats,
            "margin": margin,
            "sparse_beats_dense": sparse_beats_dense,  # null answer: False => null supported
            "null_supported": not sparse_beats_dense,
            # speedup is a SEPARATE claim, gpu-later, NOT a success criterion (corpus 6.7)
            "speedup": {
                "wall_clock_s": {n: a["wall_clock_s"] for n, a in arms.items()},
                "claim_tier": "gpu-later",
                "required_for_success": False,
                "note": "dense hardware gives no sparse speedup; needs custom kernels",
            },
        }
        self._plot(arms, dense_bwt, run_dir)
        return out

    def _run_arm(self, name, build, train, test, dim, chance, cfg, dev) -> dict:
        seed_everything(int(cfg.seed))
        model = safe_to_module(build(), dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg.experiment.lr))
        loss_fn = nn.CrossEntropyLoss()
        epochs = int(cfg.experiment.epochs_per_task)
        R: list[list[float]] = []
        t0 = time.perf_counter()
        for task in train:
            x = safe_to(task.x, dev)
            y = safe_to(task.y, dev)
            model.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
            R.append(self._evaluate(model, test, dev))
        wall = time.perf_counter() - t0
        cr = ContinualResult(R=R, chance=chance)
        ent_first, ent_last = self._routing_entropy_endpoints(model, train, dev)
        return {
            "n_params": _n_params(model),
            "bwt": cr.backward_transfer(),
            "avg_accuracy": cr.avg_accuracy(),
            "R": R,
            "wall_clock_s": wall,
            "routing_entropy_first": ent_first,
            "routing_entropy_last": ent_last,
        }

    @torch.no_grad()
    def _evaluate(self, model, test, dev) -> list[float]:
        model.eval()
        accs = []
        for task in test:
            logits = model(safe_to(task.x, dev))
            accs.append(float((logits.argmax(-1).cpu() == task.y).float().mean()))
        return accs

    @torch.no_grad()
    def _routing_entropy_endpoints(self, model, train, dev) -> tuple[float, float]:
        if not isinstance(model, MoEHead):
            return 0.0, 0.0
        model.eval()
        model(safe_to(train[0].x, dev))
        first = _routing_entropy(model.last_gates)  # type: ignore[arg-type]
        model(safe_to(train[-1].x, dev))
        last = _routing_entropy(model.last_gates)  # type: ignore[arg-type]
        return first, last

    def _matched_expert_hidden(self, dim, hidden, n_classes, n_experts) -> int:
        """Pick per-expert width so the MoE total (router + experts) is close to the dense head
        param count. Dense ~= dim*hidden + hidden*n_classes; each expert ~= dim*w + w*n_classes,
        plus a dim*n_experts router. Solve for w, floor at 1."""
        dense = dim * hidden + hidden + n_classes * hidden + n_classes
        router = dim * n_experts + n_experts
        per_expert_budget = max(1.0, (dense - router) / n_experts)
        w = (per_expert_budget - n_classes) / (dim + n_classes + 1.0)
        return max(1, int(round(w)))

    def _param_matched(self, arms, tol: float = 0.15) -> bool:
        dense = arms["dense"]["n_params"]
        return all(abs(a["n_params"] - dense) <= tol * dense for a in arms.values())

    def _plot(self, arms, dense_bwt, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        names = list(arms)
        bwts = [arms[n]["bwt"] for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(range(len(names)), bwts)
        ax[0].axhline(dense_bwt, color="k", linestyle="--", alpha=0.6, label="dense BWT")
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names)
        ax[0].set_ylabel("backward transfer (higher = less forgetting)")
        ax[0].set_title("E7: interference at matched params")
        ax[0].legend(fontsize=8)
        for n in names:
            R = arms[n]["R"]
            final = R[-1]
            ax[1].plot(range(len(final)), final, marker="o", label=f"{n} (final)")
        ax[1].set_xlabel("task")
        ax[1].set_ylabel("final accuracy")
        ax[1].set_title("per-task accuracy after full stream")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run_dir / "e7_interference.png", dpi=110)
        plt.close(fig)


def safe_to_module(m: nn.Module, dev) -> nn.Module:
    try:
        return m.to(dev)
    except (RuntimeError, NotImplementedError):  # pragma: no cover - hardware dependent
        return m.to("cpu")
