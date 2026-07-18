"""Net-new feature cache for the frozen spatial-DOA featurizer.

This is an additive component and changes no sealed scoring logic. It mirrors the shared frozen-featurizer
feature cache (``feature_cache.py``) but featurizes with ``SpatialDoaFeaturizer`` instead of the committed
``FrozenFeaturizer``. It is a SEPARATE net-new builder, not an edit to ``feature_cache.py``, precisely so
the sealed frozen-featurizer cache is never touched. Because the cache key digests the featurizer's own
``parameter_digest`` and per-frame FLOP count, the spatial-DOA cache lands in its own ``<cache_key>``
subdirectory and can never collide with or overwrite the frozen cache.

The frozen featurizer is the dominant per-run cost and is gate-independent, so this module featurizes the
real room-disjoint STARSS23 FOA subset EXACTLY ONCE with the spatial-DOA front-end and caches the per-clip
feature arrays, the ground-truth onsets, and the native fold-respecting split to a byte-reproducible
on-disk cache. Every downstream run (the producer and the independent verifier's re-scoring) then reuses
the cache. Caching is a wall-clock optimization only, never a budget cut: the manifest still records the
honest per-frame featurize FLOPs so the downstream FLOP ledger charges the featurizer per arm exactly as
if it had been recomputed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID
from .adapter import RealStarssAdapter
from .featurizer_spatial_doa import D_FEAT, FLOPS_PER_FRAME, SpatialDoaFeaturizer
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    DEFAULT_N_VAL_ROOMS,
    _fold_respecting_split,
)
from .schema import Clip, ClipSplit

CACHE_SCHEMA = "mop-starss23-escs-feature-cache-spatial-doa/v1"

DEFAULT_CACHE_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/feature_cache")

_FEATURE_DTYPE = "<f8"
_FEATURE_SUFFIX = ".f8"
_FEATURES_SUBDIR = "features"
_MANIFEST_NAME = "manifest.json"


class SpatialDoaCacheRefusal(ValueError):
    """Raised when the spatial-DOA feature cache cannot be built, keyed, or loaded coherently."""


def _config_payload(
    *,
    featurizer: SpatialDoaFeaturizer,
    foa_root: Path,
    metadata_root: Path,
    max_frames: int | None,
    n_val_rooms: int,
) -> dict[str, Any]:
    """The canonical config a cache directory is bound to. Any change here mints a fresh cache key."""

    return {
        "schema": CACHE_SCHEMA,
        "bed_id": BED_ID,
        "front_end": "spatial_doa_active_intensity",
        "featurizer_parameter_digest": featurizer.parameter_digest(),
        "featurizer_n_params": featurizer.n_params(),
        "flops_per_frame": FLOPS_PER_FRAME,
        "foa_root": str(Path(foa_root)),
        "metadata_root": str(Path(metadata_root)),
        "max_frames": max_frames,
        "n_val_rooms": int(n_val_rooms),
    }


def cache_key(
    *,
    featurizer: SpatialDoaFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> str:
    """Return the canonical cache key for the spatial-DOA featurizer plus config. Reads no audio."""

    featurizer = featurizer or SpatialDoaFeaturizer()
    payload = _config_payload(
        featurizer=featurizer,
        foa_root=Path(foa_root),
        metadata_root=Path(metadata_root),
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    return canonical_sha256(payload)


def cache_dir_for(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    featurizer: SpatialDoaFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> Path:
    """Resolve the cache subdirectory the spatial-DOA featurizer plus config lives in."""

    key = cache_key(
        featurizer=featurizer,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    return Path(cache_root) / key


def _feature_bytes(features: np.ndarray) -> bytes:
    return np.ascontiguousarray(features, dtype=_FEATURE_DTYPE).tobytes()


@dataclass(frozen=True, slots=True)
class CachedCorpus:
    """A featurize-free view of the real corpus under the spatial-DOA front-end."""

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
        onsets = sum(len(clip.onsets) for clip in self.split.train)
        frames = sum(clip.n_frames for clip in self.split.train)
        return onsets / frames if frames > 0 else 0.0

    def n_test_clips(self) -> int:
        return len(self.split.test)

    def n_test_onsets(self) -> int:
        return sum(len(clip.onsets) for clip in self.split.test)

    def n_test_frames(self) -> int:
        return int(sum(clip.n_frames for clip in self.split.test))

    def total_frames(self) -> int:
        return int(sum(clip.n_frames for clip in self.clips))

    def total_featurize_flops(self) -> int:
        return int(self.flops_per_frame * self.total_frames())


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Timing and size provenance of one cache build. All wall-clock, none of it sealed as budget."""

    cache_key: str
    cache_dir: Path
    manifest_path: Path
    n_clips: int
    total_frames: int
    total_feature_bytes: int
    featurize_seconds: float

    def numbers(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "cache_dir": str(self.cache_dir),
            "n_clips": self.n_clips,
            "total_frames": self.total_frames,
            "total_feature_megabytes": round(self.total_feature_bytes / 1e6, 3),
            "featurize_seconds": round(self.featurize_seconds, 3),
        }


def build_feature_cache(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> BuildReport:
    """Featurize the real room-disjoint subset once with the spatial-DOA front-end and seal the cache."""

    featurizer = SpatialDoaFeaturizer()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    split = _fold_respecting_split(adapter, n_val_rooms)

    clips = adapter.clips()
    key = cache_key(
        featurizer=featurizer,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    out_dir = Path(cache_root) / key
    features_dir = out_dir / _FEATURES_SUBDIR
    features_dir.mkdir(parents=True, exist_ok=True)

    clip_manifest: list[dict[str, Any]] = []
    total_frames = 0
    total_feature_bytes = 0
    started = time.perf_counter()
    for clip in clips:
        audio = adapter.audio(clip.clip_id)
        features = featurizer.featurize(audio)
        if features.shape != (clip.n_frames, D_FEAT):
            raise SpatialDoaCacheRefusal(
                f"clip {clip.clip_id} featurized to {features.shape}, expected {(clip.n_frames, D_FEAT)}"
            )
        raw = _feature_bytes(features)
        digest = featurizer.feature_digest(features)
        (features_dir / f"{clip.clip_id}{_FEATURE_SUFFIX}").write_bytes(raw)
        total_frames += clip.n_frames
        total_feature_bytes += len(raw)
        clip_manifest.append(
            {
                "clip": clip.payload(),
                "onset_frames": list(clip.onset_frames),
                "feature_shape": [clip.n_frames, D_FEAT],
                "feature_dtype": _FEATURE_DTYPE,
                "feature_sha256": digest,
                "feature_bytes": len(raw),
                "featurize_flops": int(FLOPS_PER_FRAME * clip.n_frames),
            }
        )
    featurize_seconds = time.perf_counter() - started

    manifest: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "bed_id": BED_ID,
        "cache_key": key,
        "config": _config_payload(
            featurizer=featurizer,
            foa_root=Path(foa_root),
            metadata_root=Path(metadata_root),
            max_frames=max_frames,
            n_val_rooms=n_val_rooms,
        ),
        "featurizer": {
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        },
        "featurize_charge_note": (
            "caching is a wall-clock optimization only; featurize_flops is charged per arm in the FLOP "
            "ledger exactly as if the featurizer had been recomputed, so the cache is not a budget cut"
        ),
        "split": {
            "train": [clip.clip_id for clip in split.train],
            "val": [clip.clip_id for clip in split.val],
            "test": [clip.clip_id for clip in split.test],
            "detail": dict(split.detail),
        },
        "truncation": [t.payload() for t in adapter.truncations()],
        "clips": clip_manifest,
        "totals": {
            "n_clips": len(clips),
            "total_frames": total_frames,
            "total_feature_bytes": total_feature_bytes,
            "total_featurize_flops": int(FLOPS_PER_FRAME * total_frames),
        },
    }
    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.write_bytes(canonical_bytes(manifest))

    return BuildReport(
        cache_key=key,
        cache_dir=out_dir,
        manifest_path=manifest_path,
        n_clips=len(clips),
        total_frames=total_frames,
        total_feature_bytes=total_feature_bytes,
        featurize_seconds=featurize_seconds,
    )


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise SpatialDoaCacheRefusal(
            f"no spatial-DOA feature-cache manifest at {manifest_path}; build the cache first"
        )
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    if manifest.get("schema") != CACHE_SCHEMA:
        raise SpatialDoaCacheRefusal(f"unexpected cache schema {manifest.get('schema')!r}")
    return manifest


def _load_feature_block(cache_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    clip_id = entry["clip"]["clip_id"]
    n_frames, d_feat = entry["feature_shape"]
    path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
    if not path.is_file():
        raise SpatialDoaCacheRefusal(f"cached feature block missing for {clip_id} at {path}")
    raw = path.read_bytes()
    if len(raw) != entry["feature_bytes"]:
        raise SpatialDoaCacheRefusal(
            f"cached feature block for {clip_id} is {len(raw)} bytes, manifest says {entry['feature_bytes']}"
        )
    features = np.frombuffer(raw, dtype=_FEATURE_DTYPE).reshape(int(n_frames), int(d_feat)).copy()
    digest = SpatialDoaFeaturizer().feature_digest(features)
    if digest != entry["feature_sha256"]:
        raise SpatialDoaCacheRefusal(
            f"cached feature block for {clip_id} failed its integrity digest (cache is corrupt)"
        )
    return features


def load_cached_corpus(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> CachedCorpus:
    """Load the featurize-free spatial-DOA corpus a run consumes: Clip objects, features, and the split.

    Every feature block is re-verified against its manifest sha256 on load, so a corrupt or stale cache is
    refused rather than silently scored.
    """

    cache_dir = cache_dir_for(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    manifest = _read_manifest(cache_dir)

    clips_by_id: dict[str, Clip] = {}
    features_by_clip: dict[str, np.ndarray] = {}
    ordered_ids: list[str] = []
    for entry in manifest["clips"]:
        clip = Clip.from_payload(entry["clip"])
        clips_by_id[clip.clip_id] = clip
        features_by_clip[clip.clip_id] = _load_feature_block(cache_dir, entry)
        ordered_ids.append(clip.clip_id)

    split_ids = manifest["split"]

    def _partition(ids: list[str]) -> tuple[Clip, ...]:
        missing = [cid for cid in ids if cid not in clips_by_id]
        if missing:
            raise SpatialDoaCacheRefusal(f"split references clips absent from the cache: {missing}")
        return tuple(clips_by_id[cid] for cid in ids)

    split = ClipSplit(
        train=_partition(split_ids["train"]),
        val=_partition(split_ids["val"]),
        test=_partition(split_ids["test"]),
        detail=dict(split_ids.get("detail", {})),
    )

    config = manifest["config"]
    return CachedCorpus(
        cache_key=manifest["cache_key"],
        cache_dir=cache_dir,
        clips=tuple(clips_by_id[cid] for cid in ordered_ids),
        features_by_clip=features_by_clip,
        split=split,
        featurizer_digest=manifest["featurizer"]["parameter_digest"],
        flops_per_frame=int(manifest["featurizer"]["flops_per_frame"]),
        truncations=tuple(manifest.get("truncation", [])),
        foa_root=str(config["foa_root"]),
        metadata_root=str(config["metadata_root"]),
        max_frames=config["max_frames"],
        n_val_rooms=int(config["n_val_rooms"]),
    )


def load_or_build_corpus(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> CachedCorpus:
    """Load the spatial-DOA cache, building it first if it is absent. Byte-reproducible either way."""

    cache_dir = cache_dir_for(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    if not (cache_dir / _MANIFEST_NAME).is_file():
        build_feature_cache(
            cache_root=cache_root,
            foa_root=foa_root,
            metadata_root=metadata_root,
            max_frames=max_frames,
            n_val_rooms=n_val_rooms,
        )
    return load_cached_corpus(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )


def _main(argv: list[str] | None = None) -> int:
    """Build the spatial-DOA cache and print the timing and size numbers."""

    import argparse

    parser = argparse.ArgumentParser(description="Build the STARSS23 ESCS spatial-DOA feature cache.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--foa", default=str(DEFAULT_FOA_ROOT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_ROOT))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--n-val-rooms", type=int, default=DEFAULT_N_VAL_ROOMS)
    args = parser.parse_args(argv)

    report = build_feature_cache(
        cache_root=args.cache_root,
        foa_root=args.foa,
        metadata_root=args.metadata,
        max_frames=args.max_frames,
        n_val_rooms=args.n_val_rooms,
    )
    print(json.dumps(report.numbers(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
