from __future__ import annotations

from ..substrate.form import (
    FormAdapter,
    FormBatch,
    FormMatrix,
    FormMeta,
    FormRegistry,
    LatentStoreFormAdapter,
    SubstrateFormAdapter,
    TensorFormAdapter,
    build_form_matrix,
    form_audit,
)

PerspectiveMeta = FormMeta
PerspectiveBatch = FormBatch
PerspectiveMatrix = FormMatrix
PerspectiveAdapter = FormAdapter
TensorPerspectiveAdapter = TensorFormAdapter
LatentStorePerspectiveAdapter = LatentStoreFormAdapter
SubstratePerspectiveAdapter = SubstrateFormAdapter
PerspectiveRegistry = FormRegistry
build_perspective_matrix = build_form_matrix
perspective_audit = form_audit

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
