from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from ..devices import DeviceInfo

CONTRACT = ("id", "metric", "baseline", "ablation", "null_hypothesis", "tier")
TIERS = {"cpu-now", "gpu-later", "env-later", "2.1-only"}


class Experiment(ABC):
    id: str = ""
    metric: tuple[str, ...] = ()
    baseline: str = ""
    ablation: str = ""
    null_hypothesis: str = ""
    tier: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return
        missing = [field for field in CONTRACT if not getattr(cls, field, None)]
        if missing:
            raise TypeError(f"{cls.__name__} violates the experiment contract: missing {missing}")
        if cls.tier not in TIERS:
            raise TypeError(f"{cls.__name__}.tier={cls.tier!r} not in {sorted(TIERS)}")

    @abstractmethod
    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        raise NotImplementedError

    def contract(self) -> dict:
        return {field: getattr(self, field) for field in CONTRACT}
