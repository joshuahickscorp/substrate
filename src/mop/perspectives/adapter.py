"""PerspectiveAdapter: pair heterogeneous views on identical referents.

Studio DR1 needs vision, motion, language, audio, code, math, and their controls to refer to the same
clips before any mixture-of-perspectives claim is meaningful. This module is a small data contract:
each perspective yields features plus referent ids, the matrix builder aligns by id, and the audit names
missing matched controls instead of letting positive language outrun the evidence.

No model is loaded by construction. Heavy encoders live behind `SubstrateAdapter` or cached stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ..substrate.adapter import SubstrateAdapter
from ..substrate.latent_store import LatentStore


def _referent_tuple(referents: Sequence[object]) -> tuple[str, ...]:
    out = tuple(str(r) for r in referents)
    if not out:
        raise ValueError("a perspective batch needs at least one referent")
    if len(set(out)) != len(out):
        raise ValueError("referent ids must be unique within a perspective")
    return out


def _factor_dict(factors: Mapping[str, Any] | None, n: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, values in (factors or {}).items():
        t = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if t.shape[0] != n:
            raise ValueError(f"factor {name!r} length {t.shape[0]} != referent count {n}")
        out[str(name)] = t.detach().clone()
    return out


@dataclass(frozen=True)
class PerspectiveMeta:
    """Machine-readable honesty metadata for one perspective arm."""

    tag: str
    modality: str
    feature_dim: int
    source: str
    referent_key: str = "clip_id"
    supervised: bool = False
    derived: bool = False
    control_for: str | None = None
    license: str = "unknown"
    factors: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("PerspectiveMeta.tag is required")
        if not self.modality:
            raise ValueError("PerspectiveMeta.modality is required")
        if self.feature_dim <= 0:
            raise ValueError("PerspectiveMeta.feature_dim must be positive")
        if self.control_for == self.tag:
            raise ValueError("a perspective cannot be its own control")


@dataclass(frozen=True)
class PerspectiveBatch:
    """One extracted perspective with referent ids and optional factor labels."""

    meta: PerspectiveMeta
    features: torch.Tensor
    referents: tuple[str, ...]
    factors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim < 2:
            raise ValueError("perspective features must have shape [N, ...]")
        if self.features.shape[0] != len(self.referents):
            raise ValueError(
                f"feature count {self.features.shape[0]} != referent count {len(self.referents)}"
            )
        if len(set(self.referents)) != len(self.referents):
            raise ValueError("referent ids must be unique within a perspective")
        flat_dim = int(self.features.flatten(1).shape[1])
        if flat_dim != self.meta.feature_dim:
            raise ValueError(f"feature_dim metadata {self.meta.feature_dim} != flattened dim {flat_dim}")
        for name, values in self.factors.items():
            if values.shape[0] != len(self.referents):
                raise ValueError(f"factor {name!r} length {values.shape[0]} != referent count")

    def flattened(self) -> torch.Tensor:
        """Return [N, D] float features for probes and alignment."""
        return self.features.flatten(1).float()


@dataclass(frozen=True)
class PerspectiveMatrix:
    """A referent-aligned set of perspectives."""

    referents: tuple[str, ...]
    features: dict[str, torch.Tensor]
    metadata: dict[str, PerspectiveMeta]
    factors: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)

    def tags(self) -> list[str]:
        return sorted(self.features)

    def controls(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for tag, meta in self.metadata.items():
            if meta.control_for is not None:
                out.setdefault(meta.control_for, []).append(tag)
        return {k: sorted(v) for k, v in sorted(out.items())}


class PerspectiveAdapter(ABC):
    """ABC: one view of a shared referent set."""

    meta: PerspectiveMeta

    @property
    def tag(self) -> str:
        return self.meta.tag

    @abstractmethod
    def extract(self) -> PerspectiveBatch:
        """Return a batch. Implementations may read cached stores, tensors, or a substrate adapter."""


class TensorPerspectiveAdapter(PerspectiveAdapter):
    """A lightweight cached-view adapter, useful for language/audio/code features already on disk."""

    def __init__(
        self,
        meta: PerspectiveMeta,
        features: torch.Tensor,
        referents: Sequence[object],
        *,
        factors: Mapping[str, Any] | None = None,
    ):
        self.meta = meta
        self._features = features.detach().clone()
        self._referents = _referent_tuple(referents)
        self._factors = _factor_dict(factors, len(self._referents))
        PerspectiveBatch(self.meta, self._features, self._referents, self._factors)

    def extract(self) -> PerspectiveBatch:
        return PerspectiveBatch(self.meta, self._features, self._referents, self._factors)


class LatentStorePerspectiveAdapter(PerspectiveAdapter):
    """Expose a LatentStore as a perspective without changing the store format."""

    def __init__(
        self,
        store: LatentStore,
        *,
        tag: str | None = None,
        modality: str = "vision",
        source: str | None = None,
        referents: Sequence[object] | None = None,
        factors: Mapping[str, Any] | None = None,
        supervised: bool = False,
        derived: bool = False,
        control_for: str | None = None,
        license: str = "unknown",
        notes: str = "",
    ):
        self.store = store
        lat = store.latents().flatten(1).float()
        refs = referents if referents is not None else [f"{store.meta.name}:{i}" for i in range(lat.shape[0])]
        self.meta = PerspectiveMeta(
            tag=tag or store.meta.name,
            modality=modality,
            feature_dim=int(lat.shape[1]),
            source=source or str(store.root),
            supervised=supervised,
            derived=derived,
            control_for=control_for,
            license=license,
            factors=tuple(sorted((factors or {}).keys())),
            notes=notes,
        )
        self._referents = _referent_tuple(refs)
        self._factors = _factor_dict(factors, len(self._referents))

    def extract(self) -> PerspectiveBatch:
        return PerspectiveBatch(self.meta, self.store.latents(), self._referents, self._factors)


class SubstratePerspectiveAdapter(PerspectiveAdapter):
    """Wrap a SubstrateAdapter when the perspective is computed from clips."""

    def __init__(
        self,
        substrate: SubstrateAdapter,
        clips: torch.Tensor,
        referents: Sequence[object],
        *,
        tag: str | None = None,
        modality: str = "vision",
        source: str = "clips",
        factors: Mapping[str, Any] | None = None,
        supervised: bool = False,
        derived: bool = False,
        control_for: str | None = None,
        license: str = "unknown",
        notes: str = "",
        batch: int = 8,
    ):
        self.substrate = substrate
        self.clips = clips
        self.batch = int(batch)
        self._referents = _referent_tuple(referents)
        self._factors = _factor_dict(factors, len(self._referents))
        self.meta = PerspectiveMeta(
            tag=tag or substrate.tag,
            modality=modality,
            feature_dim=substrate.meta.embed_dim,
            source=source,
            supervised=supervised,
            derived=derived,
            control_for=control_for,
            license=license,
            factors=tuple(sorted(self._factors)),
            notes=notes or substrate.meta.notes,
        )

    def extract(self) -> PerspectiveBatch:
        features = self.substrate.extract_batched(self.clips, batch=self.batch)
        return PerspectiveBatch(self.meta, features, self._referents, self._factors)


@dataclass
class PerspectiveRegistry:
    """Holds perspective adapters and extracts them sequentially."""

    adapters: dict[str, PerspectiveAdapter] = field(default_factory=dict)

    def register(self, adapter: PerspectiveAdapter) -> None:
        if adapter.tag in self.adapters:
            raise ValueError(f"duplicate perspective tag: {adapter.tag}")
        self.adapters[adapter.tag] = adapter

    def tags(self) -> list[str]:
        return sorted(self.adapters)

    def extract_all(self) -> dict[str, PerspectiveBatch]:
        return {tag: self.adapters[tag].extract() for tag in self.tags()}


def build_perspective_matrix(
    adapters: Sequence[PerspectiveAdapter] | PerspectiveRegistry,
) -> PerspectiveMatrix:
    """Extract adapters and align every arm to the first arm's referent order."""
    if isinstance(adapters, PerspectiveRegistry):
        batches = adapters.extract_all()
    else:
        batches = _extract_sequence(adapters)
    if not batches:
        raise ValueError("at least one perspective adapter is required")

    ordered = sorted(batches)
    master = batches[ordered[0]].referents
    features: dict[str, torch.Tensor] = {}
    metadata: dict[str, PerspectiveMeta] = {}
    factors: dict[str, dict[str, torch.Tensor]] = {}

    for tag in ordered:
        batch = _align_batch(batches[tag], master)
        features[tag] = batch.flattened()
        metadata[tag] = batch.meta
        factors[tag] = batch.factors
    return PerspectiveMatrix(master, features, metadata, factors)


def perspective_audit(matrix: PerspectiveMatrix) -> dict[str, Any]:
    """Summarize the control and provenance surface for a perspective matrix."""
    controls = matrix.controls()
    substantive = [t for t, m in matrix.metadata.items() if m.control_for is None]
    missing_controls = sorted(t for t in substantive if t not in controls)
    return {
        "n_referents": len(matrix.referents),
        "tags": matrix.tags(),
        "modalities": {t: matrix.metadata[t].modality for t in matrix.tags()},
        "feature_dims": {t: int(matrix.features[t].shape[1]) for t in matrix.tags()},
        "controls": controls,
        "missing_controls": missing_controls,
        "supervised": sorted(t for t, m in matrix.metadata.items() if m.supervised),
        "derived": sorted(t for t, m in matrix.metadata.items() if m.derived),
        "licenses": {t: matrix.metadata[t].license for t in matrix.tags()},
    }


def _extract_sequence(adapters: Sequence[PerspectiveAdapter]) -> dict[str, PerspectiveBatch]:
    out: dict[str, PerspectiveBatch] = {}
    for adapter in adapters:
        if adapter.tag in out:
            raise ValueError(f"duplicate perspective tag: {adapter.tag}")
        out[adapter.tag] = adapter.extract()
    return out


def _align_batch(batch: PerspectiveBatch, master: tuple[str, ...]) -> PerspectiveBatch:
    if batch.referents == master:
        return batch
    if set(batch.referents) != set(master):
        missing = sorted(set(master).difference(batch.referents))
        extra = sorted(set(batch.referents).difference(master))
        raise ValueError(f"referent mismatch for {batch.meta.tag}: missing={missing[:5]}, extra={extra[:5]}")
    idx = torch.tensor([batch.referents.index(r) for r in master], dtype=torch.long)
    factors = {name: values.index_select(0, idx) for name, values in batch.factors.items()}
    return replace(batch, features=batch.features.index_select(0, idx), referents=master, factors=factors)
