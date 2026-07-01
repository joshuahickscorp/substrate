"""The trainable shell around the frozen substrate. Everything here learns; the encoder
does not. Components map to corpus levers (see ARCHITECTURE.md)."""

from __future__ import annotations

from .buffer import KVIndex, ReplayBuffer
from .consolidation import EWC, SI, Consolidation
from .ensemble import Ensemble
from .heads import ClassHead, GaussianHead, build_head
from .modulation import Chunking, ContextGating, WorkingMemory, build_modulation
from .neuromod import Neuromodulation, RunningStat
from .plasticity import PlasticityController
from .predictor import Predictor, mlp
from .refine import IterativeRefiner

__all__ = [
    "Predictor",
    "mlp",
    "IterativeRefiner",
    "ClassHead",
    "GaussianHead",
    "build_head",
    "Ensemble",
    "ReplayBuffer",
    "KVIndex",
    "PlasticityController",
    "EWC",
    "SI",
    "Consolidation",
    "Neuromodulation",
    "RunningStat",
    "ContextGating",
    "WorkingMemory",
    "Chunking",
    "build_modulation",
]
