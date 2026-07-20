
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


from . import scaffolds  # noqa: E402

for _cls in scaffolds.SCAFFOLDS:
    REGISTRY[_cls.id] = _cls

from .custom_substrate import CM7, CM8  # noqa: E402

REGISTRY[CM7.id] = CM7
REGISTRY[CM8.id] = CM8

from .p4_capability_density import P4Screen  # noqa: E402
from .p5_context_wrapper import P5Context  # noqa: E402

REGISTRY[P4Screen.id] = P4Screen
REGISTRY[P5Context.id] = P5Context

__all__ = [
    "Experiment",
    "E1",
    "I4",
    "CM7",
    "CM8",
    "P4Screen",
    "P5Context",
    "REGISTRY",
    "register",
    "get_experiment",
]
