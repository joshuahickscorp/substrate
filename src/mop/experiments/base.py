"""The doctrine contract, enforced in code. Every experiment carries a baseline, an
ablation, a metric, and an explicit null. A concrete Experiment that fails to declare its
null_hypothesis (or any contract field) raises at class-definition time: it cannot exist,
let alone run. This is the corpus rule ("'it just did not work' is not acceptable") made
unskippable.
"""

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
        """Run at the configured (toy) scale. Returns a metrics dict (json-serializable)."""

    def contract(self) -> dict:
        return {a: getattr(self, a) for a in CONTRACT}
