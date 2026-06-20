"""E2-E10 scaffolds. Each is a self-contained Experiment subclass with its null baked in,
runnable at toy scale. Imported here so the registry can pick them up (see __init__)."""

from __future__ import annotations

from .base import Experiment
from .e2_replay import E2
from .e3_plasticity import E3
from .e4_neuromod import E4
from .e5_curiosity import E5
from .e6_relational import E6
from .e7_sparse import E7
from .e8_dendritic import E8
from .e9_local import E9
from .e10_openended import E10

SCAFFOLDS: list[type[Experiment]] = [E2, E3, E4, E5, E6, E7, E8, E9, E10]
