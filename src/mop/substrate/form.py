
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .adapter import SubstrateAdapter
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

    tag: str
    kind: str
    feature_dim: int
    source: str
    objective: str = "unknown"
    token_shape: tuple[int, ...] = ()
    time_axis: bool = False
    trainable: bool = False
    supervised: bool = False
    derived: bool = False
    control_for: str | None = None
    license: str = "unknown"
    notes: str = ""
    referent_scheme: str = "unknown"
    referents_explicit: bool = True
    manifest_verified: bool = False

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
        return self.features.flatten(1).float()


class FormAdapter(ABC):

    meta: FormMeta

    @property
    def tag(self) -> str:
        return self.meta.tag

    @abstractmethod
    def extract(self) -> FormBatch:
        pass


class TensorFormAdapter(FormAdapter):

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

    def __init__(
        self,
        store: LatentStore,
        *,
        tag: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        objective: str | None = None,
        referents: Sequence[object] | None = None,
        referent_scheme: str | None = None,
        factors: Mapping[str, Any] | None = None,
        trainable: bool = False,
        supervised: bool = False,
        derived: bool = False,
        control_for: str | None = None,
        license: str = "unknown",
        notes: str = "",
        require_explicit_referents: bool = False,
        require_manifest: bool = False,
    ):
        self.store = store
        manifest, manifest_verified = _read_store_manifest(store, required=require_manifest)
        declared_form = manifest.get("form") if isinstance(manifest, dict) else None
        declared_form = declared_form if isinstance(declared_form, dict) else {}
        feat_shape = tuple(int(v) for v in store.meta.feat_shape)
        flat_dim = 1
        for v in feat_shape:
            flat_dim *= v

        n = len(store)
        explicit_refs = referents is not None
        discovered_scheme: str | None = None
        refs: Sequence[object] | None = referents
        if refs is None:
            discovered = _read_store_referents(store)
            if discovered is not None:
                refs, discovered_scheme = discovered
                explicit_refs = True
        if refs is None:
            if require_explicit_referents:
                raise ValueError(
                    f"{store.root} has no explicit referent sidecar; expected referents.json or "
                    "clip_stems.json"
                )
            refs = [f"{store.meta.name}:{i}" for i in range(n)]
            discovered_scheme = "store-local-row-index"

        resolved_scheme = (
            referent_scheme
            or str(declared_form.get("referent_scheme") or "")
            or discovered_scheme
            or "explicit-argument"
        )
        self.meta = FormMeta(
            tag=tag or store.meta.name,
            kind=kind or str(declared_form.get("kind") or "latent"),
            feature_dim=int(flat_dim),
            source=source or str(store.root),
            objective=objective or str(declared_form.get("objective") or "inherited-frozen"),
            token_shape=feat_shape if len(feat_shape) > 1 else (),
            time_axis=len(feat_shape) > 1,
            trainable=trainable,
            supervised=supervised,
            derived=derived,
            control_for=control_for,
            license=license,
            notes=notes,
            referent_scheme=resolved_scheme,
            referents_explicit=explicit_refs,
            manifest_verified=manifest_verified,
        )
        self._referents = _referent_tuple(refs)
        if len(self._referents) != n:
            raise ValueError(f"referent count {len(self._referents)} != store count {n}")
        loaded = factors if factors is not None else _read_store_factors(store, n)
        self._factors = _factor_dict(loaded, n)

    def extract(self) -> FormBatch:
        feats = self.store.latents().float()
        return FormBatch(self.meta, feats, self._referents, self._factors)


class SubstrateFormAdapter(FormAdapter):

    def __init__(
        self,
        substrate: SubstrateAdapter,
        clips: torch.Tensor,
        referents: Sequence[object],
        *,
        tag: str | None = None,
        kind: str = "vision",
        source: str = "clips",
        objective: str | None = None,
        referent_scheme: str = "explicit-argument",
        factors: Mapping[str, Any] | None = None,
        trainable: bool = False,
        supervised: bool = False,
        derived: bool = False,
        control_for: str | None = None,
        license: str = "unknown",
        notes: str = "",
        batch: int = 8,
    ):
        if int(batch) <= 0:
            raise ValueError("batch must be positive")
        self.substrate = substrate
        self.clips = clips
        self.batch = int(batch)
        self._referents = _referent_tuple(referents)
        if int(clips.shape[0]) != len(self._referents):
            raise ValueError(f"clip count {int(clips.shape[0])} != referent count {len(self._referents)}")
        self._factors = _factor_dict(factors, len(self._referents))
        inherited_objective = "inherited-frozen" if substrate.meta.pretrained else "random-control"
        self.meta = FormMeta(
            tag=tag or substrate.tag,
            kind=kind,
            feature_dim=int(substrate.meta.embed_dim),
            source=source,
            objective=objective or inherited_objective,
            trainable=trainable,
            supervised=supervised,
            derived=derived,
            control_for=control_for,
            license=license,
            notes=notes or substrate.meta.notes,
            referent_scheme=referent_scheme,
            referents_explicit=True,
            manifest_verified=False,
        )
        self._cached_features: torch.Tensor | None = None

    def extract(self) -> FormBatch:
        if self._cached_features is None:
            features = self.substrate.extract_batched(self.clips, batch=self.batch)
            self._cached_features = features.detach().clone()
        return FormBatch(self.meta, self._cached_features, self._referents, self._factors)


def _read_store_factors(store: LatentStore, n: int) -> dict[str, Any] | None:
    import json

    path = store.root / "factors.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    if "columns" in data:
        columns = data["columns"]
        if not isinstance(columns, dict):
            raise ValueError(f"{path} columns must be a mapping")
        return columns

    columns: dict[str, Any] = {}
    for name, values in data.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            continue
        if len(values) != n:
            raise ValueError(f"factor {name!r} length {len(values)} != referent count {n}")
        columns[str(name)] = values
    return columns


def _read_store_referents(store: LatentStore) -> tuple[Sequence[object], str] | None:
    import json

    for filename, scheme in (("referents.json", "referent-id"), ("clip_stems.json", "clip-stem")):
        path = store.root / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("referents", data.get("ids"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a list or contain a referents/ids list")
        return data, scheme
    return None


def _read_store_manifest(store: LatentStore, *, required: bool) -> tuple[dict[str, Any], bool]:
    import json

    from .cache_manifest import DEFAULT_MANIFEST, validate_cache_manifest

    path = store.root / DEFAULT_MANIFEST
    if not path.exists():
        if required:
            raise ValueError(f"{path} missing for a manifest-required form")
        return {}, False
    problems = validate_cache_manifest(store.root)
    if problems:
        if required:
            raise ValueError(f"invalid {DEFAULT_MANIFEST}: {'; '.join(problems)}")
        return {}, False
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}, True


@dataclass(frozen=True)
class FormMatrix:

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


@dataclass
class FormRegistry:

    adapters: dict[str, FormAdapter] = field(default_factory=dict)

    def register(self, adapter: FormAdapter) -> None:
        if adapter.tag in self.adapters:
            raise ValueError(f"duplicate form tag: {adapter.tag}")
        self.adapters[adapter.tag] = adapter

    def tags(self) -> list[str]:
        return sorted(self.adapters)

    def extract_all(self) -> dict[str, FormBatch]:
        return {tag: self.adapters[tag].extract() for tag in self.tags()}


def build_form_matrix(adapters: Sequence[FormAdapter] | FormRegistry) -> FormMatrix:
    batches = (
        list(adapters.extract_all().values())
        if isinstance(adapters, FormRegistry)
        else [adapter.extract() for adapter in adapters]
    )
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


def form_audit(matrix: FormMatrix, *, require_controls: bool = True, require_citable: bool = False) -> dict:
    controls = matrix.controls()
    substantive = [
        tag
        for tag, meta in matrix.metadata.items()
        if meta.control_for is None and meta.objective != "random-control"
    ]
    missing_controls = [tag for tag in substantive if tag not in controls]
    trainable = sorted(tag for tag, meta in matrix.metadata.items() if meta.trainable)
    implicit_referents = sorted(tag for tag, meta in matrix.metadata.items() if not meta.referents_explicit)
    unverified_manifests = sorted(tag for tag, meta in matrix.metadata.items() if not meta.manifest_verified)
    uncitable = sorted(set(implicit_referents) | set(unverified_manifests))
    warnings: list[str] = []
    if len(matrix.kinds()) < 2:
        warnings.append("only one form kind is present")
    if require_controls and missing_controls:
        warnings.append("one or more substantive form arms lack a matched control")
    if trainable:
        warnings.append("trainable form arms are present; separate substrate effects from shell effects")
    if implicit_referents:
        warnings.append("one or more form arms use store-local fallback referents")
    if unverified_manifests:
        warnings.append("one or more form arms lack a verified cache manifest")
    return {
        "schema": matrix.schema,
        "n": len(matrix.referents),
        "n_referents": len(matrix.referents),
        "tags": matrix.tags(),
        "kinds": matrix.kinds(),
        "modalities": {tag: matrix.metadata[tag].kind for tag in matrix.tags()},
        "feature_dims": {tag: int(matrix.features[tag].shape[1]) for tag in matrix.tags()},
        "controls": controls,
        "missing_controls": sorted(missing_controls),
        "trainable_tags": trainable,
        "supervised": sorted(tag for tag, meta in matrix.metadata.items() if meta.supervised),
        "derived": sorted(tag for tag, meta in matrix.metadata.items() if meta.derived),
        "licenses": {tag: matrix.metadata[tag].license for tag in matrix.tags()},
        "objectives": {tag: matrix.metadata[tag].objective for tag in matrix.tags()},
        "referent_schemes": {tag: matrix.metadata[tag].referent_scheme for tag in matrix.tags()},
        "implicit_referent_tags": implicit_referents,
        "unverified_manifest_tags": unverified_manifests,
        "uncitable_tags": uncitable,
        "warnings": warnings,
        "all_ok": bool(
            (not require_controls or not missing_controls)
            and len(matrix.kinds()) >= 2
            and (not require_citable or not uncitable)
        ),
    }


def fit_affine_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-3,
) -> torch.Tensor:
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
    x = torch.as_tensor(source).detach().float().flatten(1)
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return torch.cat([x, ones], dim=1) @ weight
