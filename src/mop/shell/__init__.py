"""The trainable shell around the frozen substrate. Everything here learns; the encoder
does not. Components map to corpus levers (see ARCHITECTURE.md)."""

from __future__ import annotations

from .buffer import KVIndex, ReplayBuffer
from .capmatch import fixed_total_params_sweep, matched_capacity, width_for_param_count
from .consolidation import EWC, SI, Consolidation
from .ensemble import Ensemble
from .heads import (
    ClassHead,
    GaussianHead,
    KWTAHead,
    MoEHead,
    build_head,
    moe_expert_hidden_for_dense,
    routing_entropy,
)
from .modulation import Chunking, ContextGating, WorkingMemory, build_modulation
from .neuromod import Neuromodulation, RunningStat
from .plasticity import PlasticityController
from .predictor import Predictor, mlp
from .refine import IterativeRefiner
from .workspace import WorkspaceShell

__all__ = [
    "Predictor",
    "mlp",
    "IterativeRefiner",
    "ClassHead",
    "GaussianHead",
    "build_head",
    # e7 sparse head family promoted into the shell (WP-02)
    "KWTAHead",
    "MoEHead",
    "routing_entropy",
    "moe_expert_hidden_for_dense",
    # workspace composition + matched-capacity constructors (WP-02)
    "WorkspaceShell",
    "width_for_param_count",
    "matched_capacity",
    "fixed_total_params_sweep",
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
