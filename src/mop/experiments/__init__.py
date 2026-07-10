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


# E2-E10 scaffolds, registered on import
from . import scaffolds  # noqa: E402

for _cls in scaffolds.SCAFFOLDS:
    REGISTRY[_cls.id] = _cls

# The custom-substrate lane is intentionally kept out of the generic scaffold bank: CM7 is a
# checkpointed local training workbench and CM8 is its fail-closed upstream-evidence preflight.
from .custom_substrate import CM7, CM8  # noqa: E402

REGISTRY[CM7.id] = CM7
REGISTRY[CM8.id] = CM8

# P4 rides the same lane: the class is a bounded single-cell smoke of the registered screen
# codepath; the screen itself runs through scripts/p4_capability_density.py.
from .p4_capability_density import P4Screen  # noqa: E402

REGISTRY[P4Screen.id] = P4Screen

__all__ = ["Experiment", "E1", "I4", "CM7", "CM8", "P4Screen", "REGISTRY", "register", "get_experiment"]
