"""Perspective adapters for Studio multi-arm evidence.

The substrate layer asks "which frozen encoder produced this representation". The perspective layer
asks the next question: are several views of the SAME referents paired, labeled, controlled, and ready
for cross-perspective tests.
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
