"""Deprecated Perspective import facade over the canonical Form substrate API.

All exported objects are identity aliases from `mop.substrate.form`. Keep this facade for one
deprecation phase so existing Studio scripts and notebooks remain import-compatible.
"""

from __future__ import annotations

from .adapter import (
    LatentStorePerspectiveAdapter,
    PerspectiveAdapter,
    PerspectiveBatch,
    PerspectiveMatrix,
    PerspectiveMeta,
    PerspectiveRegistry,
    SubstratePerspectiveAdapter,
    TensorPerspectiveAdapter,
    build_perspective_matrix,
    perspective_audit,
)

__all__ = [
    "PerspectiveMeta",
    "PerspectiveBatch",
    "PerspectiveMatrix",
    "PerspectiveAdapter",
    "TensorPerspectiveAdapter",
    "LatentStorePerspectiveAdapter",
    "SubstratePerspectiveAdapter",
    "PerspectiveRegistry",
    "build_perspective_matrix",
    "perspective_audit",
]
