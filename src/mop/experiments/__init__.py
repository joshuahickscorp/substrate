
from __future__ import annotations

from .base import Experiment
from .custom_substrate import CM7, CM8

REGISTRY: dict[str, type[Experiment]] = {CM7.id: CM7, CM8.id: CM8}


def register(cls: type[Experiment]) -> type[Experiment]:
    REGISTRY[cls.id] = cls
    return cls


def get_experiment(eid: str) -> Experiment:
    if eid not in REGISTRY:
        raise KeyError(f"unknown experiment {eid!r}; have {sorted(REGISTRY)}")
    return REGISTRY[eid]()


__all__ = [
    "Experiment",
    "CM7",
    "CM8",
    "REGISTRY",
    "register",
    "get_experiment",
]
