"""Small local environments used to test action-conditioned mechanics.

These environments are programmatic instruments.  They make action, consequence, and
counterfactual contracts executable; they are not evidence of natural embodiment.
"""

from .persistent_grid import (
    ACTION_NAMES,
    PersistentGridEnvironment,
    WorldSpec,
    bounded_trajectory_contract,
    collect_trajectory_bundle,
    make_world_spec,
    trajectory_tensors,
    verify_trajectory_bundle,
    write_trajectory_bundle,
)

__all__ = (
    "ACTION_NAMES",
    "PersistentGridEnvironment",
    "WorldSpec",
    "bounded_trajectory_contract",
    "collect_trajectory_bundle",
    "make_world_spec",
    "trajectory_tensors",
    "verify_trajectory_bundle",
    "write_trajectory_bundle",
)
