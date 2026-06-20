"""Experiment registry. id -> Experiment class. The doctrine contract is enforced at import
time (a subclass without a declared null cannot even be defined)."""

from __future__ import annotations

from .base import Experiment
from .e1_baseline_harness import E1
from .i4_backprop_alternatives import I4

REGISTRY: dict[str, type[Experiment]] = {E1.id: E1, I4.id: I4}


def register(cls: type[Experiment]) -> type[Experiment]:
    REGISTRY[cls.id] = cls
    return cls


def get_experiment(eid: str) -> Experiment:
    if eid not in REGISTRY:
        raise KeyError(f"unknown experiment {eid!r}; have {sorted(REGISTRY)}")
    return REGISTRY[eid]()


# scaffolds register themselves on import
from . import scaffolds  # noqa: E402,F401

__all__ = ["Experiment", "E1", "I4", "REGISTRY", "register", "get_experiment"]
