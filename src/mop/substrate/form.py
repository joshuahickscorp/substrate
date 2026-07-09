"""Form substrate interface: arbitrary observations behind one referent-aligned contract.

This is intentionally one level more abstract than `SubstrateAdapter`. A substrate adapter knows how to
turn clips into features. A form adapter knows how to present ANY observation family, vision, audio,
text-derived metadata, symbolic state, telemetry, code, action traces, or a future learned substrate, as
features over the same referents with honesty metadata and matched controls.

The module is pure data-plane scaffolding. It never loads a model, never assumes video shape, and never
claims that alignment is intelligence. It gives experiments a shared place to ask: did this representation
preserve a factor, align across forms, and survive its controls?
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .latent_store import LatentStore

FORM_SCHEMA = "mop-form-matrix/v1"

FORM_KINDS = (
    "vision",
    "audio",
    "text",
    "symbolic",
    "timeseries",
    "control",
    "code",
    "math",
    "latent",
    "mixed",
)

OBJECTIVE_FAMILIES = (
    "unknown",
    "inherited-frozen",
    "random-control",
    "handcrafted",
    "self-supervised",
    "supervised",
    "programmatic",
    "metadata",
    "learned-shell",
    "custom-substrate",
)


def _referent_tuple(referents: Sequence[object]) -> tuple[str, ...]:
    out = tuple(str(r) for r in referents)
    if not out:
        raise ValueError("a form batch needs at least one referent")
    if len(set(out)) != len(out):
        raise ValueError("referent ids must be unique within a form batch")
    return out


def _factor_dict(factors: Mapping[str, Any] | None, n: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, values in (factors or {}).items():
        t = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if t.shape[0] != n:
            raise ValueError(f"factor {name!r} length {t.shape[0]} != referent count {n}")
        out[str(name)] = t.detach().clone()
    return out


def referent_order(referents: Sequence[object], canonical: Sequence[object]) -> list[int] | None:
    """The single referent-alignment implementation for the whole project.

    Return the index list that reorders `referents` into `canonical` order, or None when they are
    already identical. Raise with missing/extra detail on a set mismatch. Both the form matrix and the
    perspective matrix build on this one function, so there is exactly one place referent alignment can
    be right or wrong (FORM_SUBSTRATE_CODEMAP.md section 0). The dict lookup is O(n) rather than the old
    per-arm `list.index` scan.
    """
    ref = tuple(str(r) for r in referents)
    can = tuple(str(r) for r in canonical)
    if ref == can:
        return None
    rset, cset = set(ref), set(can)
    if rset != cset:
        missing = sorted(cset - rset)
        extra = sorted(rset - cset)
        raise ValueError(f"referent mismatch: missing={missing[:5]}, extra={extra[:5]}")
    pos = {r: i for i, r in enumerate(ref)}
    return [pos[r] for r in can]


@dataclass(frozen=True)
class FormMeta:
    """Honesty metadata for one form arm.

    `kind` is the observation family. `objective` states how the features came into being. `control_for`
    links a control arm to the substantive arm it tests, for example `audio_random_control` controls
    `audio_ssl`. This keeps a result from quietly comparing a real arm to no floor.
    """

    tag: str
    kind: str
    feature_dim: int
    source: str
    objective: str = "unknown"
    token_shape: tuple[int, ...] = ()
    time_axis: bool = False
    trainable: bool = False
    control_for: str | None = None
    license: str = "unknown"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("FormMeta.tag is required")
        if self.kind not in FORM_KINDS:
            raise ValueError(f"FormMeta.kind={self.kind!r} not in {FORM_KINDS}")
        if self.feature_dim <= 0:
            raise ValueError("FormMeta.feature_dim must be positive")
        if self.objective not in OBJECTIVE_FAMILIES:
            raise ValueError(f"FormMeta.objective={self.objective!r} not in {OBJECTIVE_FAMILIES}")
        if self.control_for == self.tag:
            raise ValueError("a form arm cannot control itself")
        if any(int(v) <= 0 for v in self.token_shape):
            raise ValueError("token_shape entries must be positive")


@dataclass(frozen=True)
class FormBatch:
    """One extracted form, with referent ids and optional factor labels."""

    meta: FormMeta
    features: torch.Tensor
    referents: tuple[str, ...]
    factors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim < 2:
            raise ValueError("form features must have shape [N, ...]")
        if self.features.shape[0] != len(self.referents):
            raise ValueError(
                f"feature count {self.features.shape[0]} != referent count {len(self.referents)}"
            )
        if len(set(self.referents)) != len(self.referents):
            raise ValueError("referent ids must be unique within a form batch")
        flat_dim = int(self.features.flatten(1).shape[1])
        if flat_dim != self.meta.feature_dim:
            raise ValueError(f"feature_dim metadata {self.meta.feature_dim} != flattened dim {flat_dim}")
        for name, values in self.factors.items():
            if values.shape[0] != len(self.referents):
                raise ValueError(f"factor {name!r} length {values.shape[0]} != referent count")

    def flattened(self) -> torch.Tensor:
        """Return [N, D] float features for probes, alignment, and shell heads."""
        return self.features.flatten(1).float()


class FormAdapter(ABC):
    """ABC: one observation family over one referent set."""

    meta: FormMeta

    @property
    def tag(self) -> str:
        return self.meta.tag

    @abstractmethod
    def extract(self) -> FormBatch:
        """Return a referent-aligned feature batch."""


class TensorFormAdapter(FormAdapter):
    """A cached tensor form. Useful for tests, toy experiments, and feature stores already on disk."""

    def __init__(
        self,
        meta: FormMeta,
        features: torch.Tensor,
        referents: Sequence[object],
        *,
        factors: Mapping[str, Any] | None = None,
    ):
        self.meta = meta
        self._features = features.detach().clone()
        self._referents = _referent_tuple(referents)
        self._factors = _factor_dict(factors, len(self._referents))
        FormBatch(self.meta, self._features, self._referents, self._factors)

    def extract(self) -> FormBatch:
        return FormBatch(self.meta, self._features, self._referents, self._factors)


class LatentStoreFormAdapter(FormAdapter):
    """Present a cached `LatentStore` as one form arm, without changing the store format.

    This is the encode-once bridge: real V-JEPA (or any encoder) features are cached to a
    `LatentStore` once by the cache scripts, then read forever as a form. The adapter never loads a
    model and never encodes; a form arm that needs live clips-to-features encoding uses the existing
    `SubstratePerspectiveAdapter` (encode once, cache, then read here), never a re-encode in this
    data-plane class.

    Dense stores `[N, T, ...]` are flattened to `[N, D]` for probes and alignment; the original
    per-item geometry is recorded in `FormMeta.token_shape` so it stays recoverable. When the store
    carries a `factors.json` sidecar and no explicit `factors` are given, it is read as factor labels.
    """

    def __init__(
        self,
        store: LatentStore,
        *,
        tag: str | None = None,
        kind: str = "latent",
        source: str | None = None,
        objective: str = "inherited-frozen",
        referents: Sequence[object] | None = None,
        factors: Mapping[str, Any] | None = None,
        trainable: bool = False,
        control_for: str | None = None,
        license: str = "unknown",
        notes: str = "",
    ):
        self.store = store
        feat_shape = tuple(int(v) for v in store.meta.feat_shape)
        flat_dim = 1
        for v in feat_shape:
            flat_dim *= v
        self.meta = FormMeta(
            tag=tag or store.meta.name,
            kind=kind,
            feature_dim=int(flat_dim),
            source=source or str(store.root),
            objective=objective,
            token_shape=feat_shape if len(feat_shape) > 1 else (),
            trainable=trainable,
            control_for=control_for,
            license=license,
            notes=notes,
        )
        n = len(store)
        refs = referents if referents is not None else [f"{store.meta.name}:{i}" for i in range(n)]
        self._referents = _referent_tuple(refs)
        if len(self._referents) != n:
            raise ValueError(f"referent count {len(self._referents)} != store count {n}")
        loaded = factors if factors is not None else _read_store_factors(store)
        self._factors = _factor_dict(loaded, n)

    def extract(self) -> FormBatch:
        feats = self.store.latents().flatten(1).float()
        return FormBatch(self.meta, feats, self._referents, self._factors)


def _read_store_factors(store: LatentStore) -> dict[str, Any] | None:
    """Read a `factors.json` sidecar (column -> list) from the store root, or None if absent."""
    import json

    path = store.root / "factors.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a dict of column -> list of per-referent values")
    return data


@dataclass(frozen=True)
class FormMatrix:
    """A referent-aligned set of arbitrary forms."""

    referents: tuple[str, ...]
    features: dict[str, torch.Tensor]
    metadata: dict[str, FormMeta]
    factors: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    schema: str = FORM_SCHEMA

    def tags(self) -> list[str]:
        return sorted(self.features)

    def controls(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for tag, meta in self.metadata.items():
            if meta.control_for is not None:
                out.setdefault(meta.control_for, []).append(tag)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def kinds(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for tag, meta in self.metadata.items():
            out.setdefault(meta.kind, []).append(tag)
        return {k: sorted(v) for k, v in sorted(out.items())}


def build_form_matrix(adapters: Sequence[FormAdapter]) -> FormMatrix:
    """Extract and align forms by referent id.

    The first adapter's referent order becomes canonical. Every later form must contain exactly the same
    referents, but may arrive in a different order. Features are flattened because diagnostics and shell
    heads operate on `[N, D]`; `FormMeta.token_shape` preserves the original token geometry when needed.
    """
    batches = [adapter.extract() for adapter in adapters]
    if not batches:
        raise ValueError("build_form_matrix needs at least one form adapter")
    tags: set[str] = set()
    canonical = batches[0].referents
    features: dict[str, torch.Tensor] = {}
    metadata: dict[str, FormMeta] = {}
    factors: dict[str, dict[str, torch.Tensor]] = {}

    for batch in batches:
        tag = batch.meta.tag
        if tag in tags:
            raise ValueError(f"duplicate form tag: {tag}")
        tags.add(tag)
        idx = referent_order(batch.referents, canonical)
        if idx is None:
            features[tag] = batch.flattened()
            factors[tag] = dict(batch.factors)
        else:
            order = torch.tensor(idx, dtype=torch.long)
            features[tag] = batch.flattened().index_select(0, order)
            factors[tag] = {name: v.index_select(0, order) for name, v in batch.factors.items()}
        metadata[tag] = batch.meta

    return FormMatrix(canonical, features, metadata, factors)


def form_audit(matrix: FormMatrix, *, require_controls: bool = True) -> dict:
    """Summarize whether the form matrix is scientifically usable.

    A matrix can be mechanically valid but weak as evidence. The audit names missing controls, single-kind
    matrices, and trainable arms so downstream claims cannot pretend the substrate was a neutral input.
    """
    controls = matrix.controls()
    substantive = [
        tag
        for tag, meta in matrix.metadata.items()
        if meta.control_for is None and meta.objective != "random-control"
    ]
    missing_controls = [tag for tag in substantive if tag not in controls]
    trainable = sorted(tag for tag, meta in matrix.metadata.items() if meta.trainable)
    warnings: list[str] = []
    if len(matrix.kinds()) < 2:
        warnings.append("only one form kind is present")
    if require_controls and missing_controls:
        warnings.append("one or more substantive form arms lack a matched control")
    if trainable:
        warnings.append("trainable form arms are present; separate substrate effects from shell effects")
    return {
        "schema": matrix.schema,
        "n": len(matrix.referents),
        "tags": matrix.tags(),
        "kinds": matrix.kinds(),
        "controls": controls,
        "missing_controls": sorted(missing_controls),
        "trainable_tags": trainable,
        "warnings": warnings,
        "all_ok": bool((not require_controls or not missing_controls) and len(matrix.kinds()) >= 2),
    }


def fit_affine_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-3,
) -> torch.Tensor:
    """Fit an affine map `source -> target` from paired referents.

    This is the small, explicit alignment primitive used by toy form experiments. It is not a claim that
    cognition is linear alignment; it is the matched baseline a stronger form substrate must beat.
    """
    x = torch.as_tensor(source).detach().float().flatten(1)
    y = torch.as_tensor(target).detach().float().flatten(1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"source rows {x.shape[0]} != target rows {y.shape[0]}")
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=1)
    reg = torch.eye(x_aug.shape[1], dtype=x.dtype, device=x.device) * float(ridge)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def apply_affine_alignment(source: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a fitted affine alignment returned by `fit_affine_alignment`."""
    x = torch.as_tensor(source).detach().float().flatten(1)
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1) @ weight
