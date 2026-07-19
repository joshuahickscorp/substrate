
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.substrate.events import atomic_write_bytes, canonical_bytes, canonical_sha256

from . import BED_ID
from .adapter import RealStarssAdapter, native_fold_split
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    DEFAULT_N_VAL_ROOMS,
    RealArtifactRefusal,
)
from .schema import Clip, ClipSplit

CACHE_SCHEMA = "mop-starss23-escs-feature-cache/v1"
SUPERFLUX_CACHE_SCHEMA = "mop-starss23-escs-superflux-feature-cache/v1"
SPATIAL_DOA_CACHE_SCHEMA = "mop-starss23-escs-feature-cache-spatial-doa/v1"
DEFAULT_CACHE_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/feature_cache")

_FEATURE_DTYPE = "<f8"
_FEATURE_SUFFIX = ".f8"
_FEATURES_SUBDIR = "features"
_MANIFEST_NAME = "manifest.json"


class FeatureCacheRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CachePolicy:

    name: str
    schema: str
    factory: Callable[[], Any]
    d_feat: int
    flops_per_frame: int
    config_identity: tuple[str, str] | None = None
    manifest_identity: tuple[str, str] | None = None
    charge_label: str = "featurizer"


def cache_policy(front_end: str = "base") -> CachePolicy:

    if front_end == "base":
        from .featurizer import D_FEAT, FLOPS_PER_FRAME, FrozenFeaturizer
        return CachePolicy("base", CACHE_SCHEMA, FrozenFeaturizer, D_FEAT, FLOPS_PER_FRAME)
    if front_end == "superflux":
        from .featurizer_superflux_spectral import D_FEAT, FLOPS_PER_FRAME, SuperfluxSpectralFeaturizer
        identity = ("featurizer_id", "superflux_spectral")
        return CachePolicy(
            "superflux", SUPERFLUX_CACHE_SCHEMA, SuperfluxSpectralFeaturizer, D_FEAT, FLOPS_PER_FRAME,
            config_identity=identity, manifest_identity=identity, charge_label="SuperFlux front-end",
        )
    if front_end == "spatial_doa":
        from .featurizer_spatial_doa import D_FEAT, FLOPS_PER_FRAME, SpatialDoaFeaturizer
        return CachePolicy(
            "spatial_doa", SPATIAL_DOA_CACHE_SCHEMA, SpatialDoaFeaturizer, D_FEAT, FLOPS_PER_FRAME,
            config_identity=("front_end", "spatial_doa_active_intensity"),
        )
    raise FeatureCacheRefusal(f"unknown STARSS23 cache front end {front_end!r}")


def _config_payload(
    policy: CachePolicy, featurizer: Any, foa_root: Path, metadata_root: Path,
    max_frames: int | None, n_val_rooms: int,
) -> dict[str, Any]:
    payload = {"schema": policy.schema, "bed_id": BED_ID}
    if policy.config_identity:
        payload[policy.config_identity[0]] = policy.config_identity[1]
    payload.update({
        "featurizer_parameter_digest": featurizer.parameter_digest(),
        "featurizer_n_params": featurizer.n_params(), "flops_per_frame": policy.flops_per_frame,
        "foa_root": str(foa_root), "metadata_root": str(metadata_root), "max_frames": max_frames,
        "n_val_rooms": int(n_val_rooms),
    })
    return payload


def cache_key(
    *, front_end: str = "base", featurizer: Any | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT, metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None, n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> str:

    policy = cache_policy(front_end)
    provider = featurizer or policy.factory()
    return canonical_sha256(_config_payload(
        policy, provider, Path(foa_root), Path(metadata_root), max_frames, n_val_rooms,
    ))


def cache_dir_for(
    *, front_end: str = "base", cache_root: str | Path = DEFAULT_CACHE_ROOT,
    featurizer: Any | None = None, foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT, max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> Path:
    return Path(cache_root) / cache_key(
        front_end=front_end, featurizer=featurizer, foa_root=foa_root, metadata_root=metadata_root,
        max_frames=max_frames, n_val_rooms=n_val_rooms,
    )


def _feature_bytes(features: np.ndarray) -> bytes:
    return np.ascontiguousarray(features, dtype=_FEATURE_DTYPE).tobytes()


@dataclass(frozen=True, slots=True)
class CachedCorpus:

    cache_key: str
    cache_dir: Path
    clips: tuple[Clip, ...]
    features_by_clip: dict[str, np.ndarray]
    split: ClipSplit
    featurizer_digest: str
    flops_per_frame: int
    truncations: tuple[dict[str, Any], ...]
    foa_root: str
    metadata_root: str
    max_frames: int | None
    n_val_rooms: int

    def train_onset_density(self) -> float:
        frames = sum(clip.n_frames for clip in self.split.train)
        return sum(len(clip.onsets) for clip in self.split.train) / frames if frames else 0.0

    def n_test_clips(self) -> int:
        return len(self.split.test)

    def n_test_onsets(self) -> int:
        return sum(len(clip.onsets) for clip in self.split.test)

    def n_test_frames(self) -> int:
        return int(sum(clip.n_frames for clip in self.split.test))

    def total_frames(self) -> int:
        return int(sum(clip.n_frames for clip in self.clips))

    def total_featurize_flops(self) -> int:
        return self.flops_per_frame * self.total_frames()


@dataclass(frozen=True, slots=True)
class BuildReport:

    front_end: str
    cache_key: str
    cache_dir: Path
    manifest_path: Path
    n_clips: int
    total_frames: int
    total_feature_bytes: int
    manifest_bytes: int
    featurize_seconds: float
    seconds_per_clip: float

    def numbers(self) -> dict[str, Any]:
        common = {
            "cache_key": self.cache_key, "cache_dir": str(self.cache_dir), "n_clips": self.n_clips,
            "total_frames": self.total_frames,
            "total_feature_megabytes": round(self.total_feature_bytes / 1e6, 3),
            "featurize_seconds": round(self.featurize_seconds, 3),
        }
        if self.front_end == "base":
            return {**common, "total_feature_bytes": self.total_feature_bytes,
                    "manifest_bytes": self.manifest_bytes,
                    "seconds_per_clip": round(self.seconds_per_clip, 4)}
        if self.front_end == "superflux":
            common["total_superflux_featurize_flops"] = (
                cache_policy("superflux").flops_per_frame * self.total_frames
            )
        return common


def build_feature_cache(
    *, front_end: str = "base", cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT, metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None, n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> BuildReport:

    policy = cache_policy(front_end)
    featurizer = policy.factory()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    split = native_fold_split(
        adapter, n_val_rooms, refusal=RealArtifactRefusal, refuse_empty=False
    )
    clips = adapter.clips()
    key = cache_key(
        front_end=front_end, featurizer=featurizer, foa_root=foa_root, metadata_root=metadata_root,
        max_frames=max_frames, n_val_rooms=n_val_rooms,
    )
    out_dir = Path(cache_root) / key
    features_dir = out_dir / _FEATURES_SUBDIR
    clip_manifest: list[dict[str, Any]] = []
    total_frames = total_feature_bytes = 0
    started = time.perf_counter()
    for clip in clips:
        features = featurizer.featurize(adapter.audio(clip.clip_id))
        expected = (clip.n_frames, policy.d_feat)
        if features.shape != expected:
            raise FeatureCacheRefusal(
                f"clip {clip.clip_id} featurized to {features.shape}, expected {expected}"
            )
        raw = _feature_bytes(features)
        atomic_write_bytes(features_dir / f"{clip.clip_id}{_FEATURE_SUFFIX}", raw)
        total_frames += clip.n_frames
        total_feature_bytes += len(raw)
        clip_manifest.append({
            "clip": clip.payload(), "onset_frames": list(clip.onset_frames),
            "feature_shape": [clip.n_frames, policy.d_feat], "feature_dtype": _FEATURE_DTYPE,
            "feature_sha256": featurizer.feature_digest(features), "feature_bytes": len(raw),
            "featurize_flops": policy.flops_per_frame * clip.n_frames,
        })
    elapsed = time.perf_counter() - started
    provider_payload = {
        "n_params": featurizer.n_params(), "parameter_digest": featurizer.parameter_digest(),
        "flops_per_frame": policy.flops_per_frame,
    }
    if policy.manifest_identity:
        provider_payload = {policy.manifest_identity[0]: policy.manifest_identity[1], **provider_payload}
    manifest = {
        "schema": policy.schema, "bed_id": BED_ID, "cache_key": key,
        "config": _config_payload(
            policy, featurizer, Path(foa_root), Path(metadata_root), max_frames, n_val_rooms,
        ),
        "featurizer": provider_payload,
        "featurize_charge_note": (
            "caching is a wall-clock optimization only; featurize_flops is charged per arm in the FLOP "
            f"ledger exactly as if the {policy.charge_label} had been recomputed, so the cache is not a "
            "budget cut"
        ),
        "split": {"train": [clip.clip_id for clip in split.train],
                  "val": [clip.clip_id for clip in split.val], "test": [clip.clip_id for clip in split.test],
                  "detail": dict(split.detail)},
        "truncation": [row.payload() for row in adapter.truncations()], "clips": clip_manifest,
        "totals": {"n_clips": len(clips), "total_frames": total_frames,
                   "total_feature_bytes": total_feature_bytes,
                   "total_featurize_flops": policy.flops_per_frame * total_frames},
    }
    raw_manifest = canonical_bytes(manifest)
    manifest_path = out_dir / _MANIFEST_NAME
    atomic_write_bytes(manifest_path, raw_manifest)
    return BuildReport(
        front_end, key, out_dir, manifest_path, len(clips), total_frames, total_feature_bytes,
        len(raw_manifest), elapsed, elapsed / len(clips) if clips else 0.0,
    )


def _read_manifest(cache_dir: Path, policy: CachePolicy) -> dict[str, Any]:
    path = cache_dir / _MANIFEST_NAME
    if not path.is_file():
        raise FeatureCacheRefusal(f"no {policy.name} feature-cache manifest at {path}; build the cache first")
    manifest = json.loads(path.read_bytes().decode("utf-8"))
    if manifest.get("schema") != policy.schema:
        raise FeatureCacheRefusal(f"unexpected {policy.name} cache schema {manifest.get('schema')!r}")
    return manifest


def _load_feature_block(
    cache_dir: Path, entry: dict[str, Any], featurizer: Any,
) -> np.ndarray:
    clip_id = entry["clip"]["clip_id"]
    n_frames, d_feat = entry["feature_shape"]
    path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
    if not path.is_file():
        raise FeatureCacheRefusal(f"cached feature block missing for {clip_id} at {path}")
    raw = path.read_bytes()
    if len(raw) != entry["feature_bytes"]:
        raise FeatureCacheRefusal(
            f"cached feature block for {clip_id} is {len(raw)} bytes, manifest says {entry['feature_bytes']}"
        )
    features = np.frombuffer(raw, dtype=_FEATURE_DTYPE).reshape(int(n_frames), int(d_feat)).copy()
    if featurizer.feature_digest(features) != entry["feature_sha256"]:
        raise FeatureCacheRefusal(f"cached feature block for {clip_id} failed its integrity digest")
    return features


def load_cached_corpus(
    *, front_end: str = "base", cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT, metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None, n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> CachedCorpus:

    policy = cache_policy(front_end)
    featurizer = policy.factory()
    cache_dir = cache_dir_for(
        front_end=front_end, cache_root=cache_root, featurizer=featurizer, foa_root=foa_root,
        metadata_root=metadata_root, max_frames=max_frames, n_val_rooms=n_val_rooms,
    )
    manifest = _read_manifest(cache_dir, policy)
    clips_by_id: dict[str, Clip] = {}
    features_by_clip: dict[str, np.ndarray] = {}
    ordered_ids: list[str] = []
    for entry in manifest["clips"]:
        clip = Clip.from_payload(entry["clip"])
        clips_by_id[clip.clip_id] = clip
        features_by_clip[clip.clip_id] = _load_feature_block(cache_dir, entry, featurizer)
        ordered_ids.append(clip.clip_id)

    def partition(ids: list[str]) -> tuple[Clip, ...]:
        missing = [clip_id for clip_id in ids if clip_id not in clips_by_id]
        if missing:
            raise FeatureCacheRefusal(f"split references clips absent from the cache: {missing}")
        return tuple(clips_by_id[clip_id] for clip_id in ids)

    split_ids = manifest["split"]
    split = ClipSplit(
        train=partition(split_ids["train"]), val=partition(split_ids["val"]),
        test=partition(split_ids["test"]), detail=dict(split_ids.get("detail", {})),
    )
    config = manifest["config"]
    return CachedCorpus(
        manifest["cache_key"], cache_dir, tuple(clips_by_id[clip_id] for clip_id in ordered_ids),
        features_by_clip, split, manifest["featurizer"]["parameter_digest"],
        int(manifest["featurizer"]["flops_per_frame"]), tuple(manifest.get("truncation", [])),
        str(config["foa_root"]), str(config["metadata_root"]), config["max_frames"],
        int(config["n_val_rooms"]),
    )


def load_or_build_cached_corpus(**kwargs: Any) -> CachedCorpus:

    directory = cache_dir_for(**kwargs)
    if not (directory / _MANIFEST_NAME).is_file():
        build_feature_cache(**kwargs)
    return load_cached_corpus(**kwargs)


@dataclass(frozen=True, slots=True)
class VerifyReport:
    checked_clips: tuple[str, ...]
    all_byte_identical: bool
    per_clip: dict[str, bool]

    def numbers(self) -> dict[str, Any]:
        return {"checked_clips": list(self.checked_clips), "all_byte_identical": self.all_byte_identical}


def verify_cache_bytes(
    *, front_end: str = "base", cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT, metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None, n_val_rooms: int = DEFAULT_N_VAL_ROOMS, n_clips: int = 2,
) -> VerifyReport:

    if n_clips <= 0:
        raise FeatureCacheRefusal("n_clips must be positive")
    policy = cache_policy(front_end)
    featurizer = policy.factory()
    directory = cache_dir_for(
        front_end=front_end, cache_root=cache_root, featurizer=featurizer, foa_root=foa_root,
        metadata_root=metadata_root, max_frames=max_frames, n_val_rooms=n_val_rooms,
    )
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    checked = [clip.clip_id for clip in adapter.clips()[:n_clips]]
    parity = {}
    for clip_id in checked:
        fresh = _feature_bytes(featurizer.featurize(adapter.audio(clip_id)))
        cached = directory / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
        if not cached.is_file():
            raise FeatureCacheRefusal(f"cached feature block missing for {clip_id} at {cached}")
        parity[clip_id] = cached.read_bytes() == fresh
    return VerifyReport(tuple(checked), all(parity.values()), parity)
