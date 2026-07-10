"""Deprecated Perspective names for the canonical Form substrate contract.

`mop.substrate.form` is the sole implementation of referent-aligned metadata, batches, adapters,
registries, matrices, builders, and audits. These identity aliases preserve one import-compatible
deprecation phase for Studio callers and external notebooks; no Perspective implementation lives
here. Durable DR1 artifact names remain unchanged for receipt compatibility.
"""

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
