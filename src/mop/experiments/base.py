from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from ..devices import DeviceInfo

if TYPE_CHECKING:
    from ..substrate.datasets import Task

CONTRACT = ("id", "metric", "baseline", "ablation", "null_hypothesis", "tier")
TIERS = {"cpu-now", "gpu-later", "env-later", "2.1-only"}


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((a - m) ** 2 for a in v) / (len(v) - 1)) ** 0.5


def _split(task: Task, frac: float = 0.8) -> tuple[Task, Task]:
    from ..substrate.datasets import Task

    n = int(task.x.shape[0] * frac)
    tr = Task(task.name, task.x[:n], task.y[:n], n_classes=task.n_classes, task_id=task.task_id)
    te = Task(task.name, task.x[n:], task.y[n:], n_classes=task.n_classes, task_id=task.task_id)
    return tr, te


def _split_xy(x, y, frac: float = 0.7):
    cut = int(x.shape[0] * frac)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def _spread(v) -> float:
    return (max(v) - min(v)) / 2.0 if len(v) > 1 else 0.0


def _diag_mean(r) -> float:
    return float(sum(r.R[j][j] for j in range(r.T)) / r.T)


def _fit_eval(backbone, head, x, y, xte, yte, epochs: int, lr: float) -> float:
    import torch
    import torch.nn.functional as F

    opt = torch.optim.Adam([*backbone.parameters(), *head.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = backbone(x)
        z = z[0] if isinstance(z, tuple) else z
        F.cross_entropy(head(z), y).backward()
        opt.step()
    with torch.no_grad():
        z = backbone(xte)
        z = z[0] if isinstance(z, tuple) else z
        return float((head(z).argmax(-1) == yte).float().mean())


class Experiment(ABC):
    id: str = ""
    metric: tuple[str, ...] = ()
    baseline: str = ""
    ablation: str = ""
    null_hypothesis: str = ""
    tier: str = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if getattr(cls, "__abstractmethods__", None):
            return  # still-abstract intermediate, not a concrete experiment yet
        missing = [a for a in CONTRACT if not getattr(cls, a, None)]
        if missing:
            raise TypeError(
                f"{cls.__name__} violates the doctrine contract: missing {missing}. "
                "Every experiment must declare a baseline, ablation, metric, and null."
            )
        if cls.tier not in TIERS:
            raise TypeError(f"{cls.__name__}.tier={cls.tier!r} not in {sorted(TIERS)}")

    @abstractmethod
    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        pass

    def contract(self) -> dict:
        return {a: getattr(self, a) for a in CONTRACT}
