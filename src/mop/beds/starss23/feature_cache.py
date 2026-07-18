"""Component 9a: the shared frozen-featurizer feature cache for the STARSS23 ESCS bed.

This is a net-new, additive component. It changes no sealed scoring logic. The deterministic frozen
featurizer is the dominant per-run cost and is gate-independent: identical clip bytes yield identical
feature bytes regardless of which gate variant reads them. So this module featurizes the real, room
disjoint STARSS23 FOA subset EXACTLY ONCE (importing the frozen ``FrozenFeaturizer``, never
reimplementing it) and caches the per-clip per-frame feature arrays, the ground-truth onset frames, and
the native fold-respecting clip / room / split partition to a deterministic, byte-reproducible on-disk
cache. Every gate variant and every control then reuses the cache, so a variant run is featurize-free.

The cache is keyed by a canonical digest over the featurizer's frozen-parameter sha and the corpus
config (the FOA and metadata roots, the whole-frame cap, and the val-room count), so a cache directory
can only ever serve the exact featurizer and corpus that produced it. Feature blocks are stored as raw
little-endian float64 bytes (the featurizer's own output dtype) with their sha256 recorded in the
manifest, so a load re-verifies every block and ``load_cached_corpus`` reconstructs byte-for-byte what
the harness, gate, and referee consume: the ``ClipSplit`` of real ``Clip`` objects (carrying their
onsets) and the ``features_by_clip`` map.

Caching is a wall-clock optimization only, never a budget reduction. The manifest still records the
honest per-clip and total featurize FLOPs (``FLOPS_PER_FRAME`` times the frame count) so the downstream
FLOP ledger charges the featurizer per arm exactly as if it had been recomputed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.substrate.events import canonical_bytes, canonical_sha256

from . import BED_ID
from .adapter import RealStarssAdapter
from .featurizer import D_FEAT, FLOPS_PER_FRAME, FrozenFeaturizer
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    DEFAULT_N_VAL_ROOMS,
    _fold_respecting_split,
)
from .schema import Clip, ClipSplit

CACHE_SCHEMA = "mop-starss23-escs-feature-cache/v1"

# The deterministic on-disk cache root on this host. Every distinct featurizer-plus-config combination
# lands in its own ``<cache_key>`` subdirectory below this root.
DEFAULT_CACHE_ROOT = Path("/Users/scammermike/Downloads/mop-data/starss23/feature_cache")

# Per-clip feature blocks are stored raw so the on-disk bytes equal the featurizer output bytes exactly.
_FEATURE_DTYPE = "<f8"
_FEATURE_SUFFIX = ".f8"
_FEATURES_SUBDIR = "features"
_MANIFEST_NAME = "manifest.json"


class FeatureCacheRefusal(ValueError):
    """Raised when the feature cache cannot be built, keyed, or loaded coherently."""


# ---------------------------------------------------------------------------
# The cache key: a canonical digest over the featurizer sha and the corpus config.
# ---------------------------------------------------------------------------


def _config_payload(
    *,
    featurizer: FrozenFeaturizer,
    foa_root: Path,
    metadata_root: Path,
    max_frames: int | None,
    n_val_rooms: int,
) -> dict[str, Any]:
    """The canonical config a cache directory is bound to. Any change here mints a fresh cache key."""

    return {
        "schema": CACHE_SCHEMA,
        "bed_id": BED_ID,
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
    featurizer: FrozenFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> str:
    """Return the canonical cache key for a featurizer-plus-config combination. Reads no audio."""

    featurizer = featurizer or FrozenFeaturizer()
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
    featurizer: FrozenFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> Path:
    """Resolve the cache subdirectory a given featurizer-plus-config combination lives in."""

    key = cache_key(
        featurizer=featurizer,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    return Path(cache_root) / key


def _feature_bytes(features: np.ndarray) -> bytes:
    """Canonical little-endian float64 bytes of a feature block (the featurizer's own output dtype)."""

    return np.ascontiguousarray(features, dtype=_FEATURE_DTYPE).tobytes()


# ---------------------------------------------------------------------------
# The loaded corpus: exactly what the harness / gate / referee consume.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CachedCorpus:
    """A featurize-free view of the real corpus: real Clip objects, their features, and the split.

    ``split`` and ``clips`` carry real ``Clip`` objects with their onset labels, and ``features_by_clip``
    maps every clip id to its (n_frames, D_FEAT) float64 feature block, byte-identical to a fresh
    featurization. A variant producer consumes exactly these, with no audio decode and no featurize.
    """

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


# ---------------------------------------------------------------------------
# Build: featurize the corpus once and seal the cache.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildReport:
    """Timing and size provenance of one cache build. All wall-clock, none of it sealed as budget."""

    cache_key: str
    cache_dir: Path
    manifest_path: Path
    n_clips: int
    total_frames: int
    total_feature_bytes: int
    manifest_bytes: int
    featurize_seconds: float
    seconds_per_clip: float
    per_clip_seconds: dict[str, float]

    def numbers(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "cache_dir": str(self.cache_dir),
            "n_clips": self.n_clips,
            "total_frames": self.total_frames,
            "total_feature_bytes": self.total_feature_bytes,
            "total_feature_megabytes": round(self.total_feature_bytes / 1e6, 3),
            "manifest_bytes": self.manifest_bytes,
            "featurize_seconds": round(self.featurize_seconds, 3),
            "seconds_per_clip": round(self.seconds_per_clip, 4),
        }


def build_feature_cache(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> BuildReport:
    """Featurize the real room-disjoint subset once and write the byte-reproducible on-disk cache.

    Every clip is featurized exactly once with the frozen front-end. The split is the native
    fold-respecting split (test is exactly fold-4 dev-test), reused unchanged from the real producer.
    """

    featurizer = FrozenFeaturizer()
    adapter = RealStarssAdapter(
        foa_root, metadata_root, rights_clean=True, max_frames=max_frames
    )
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

    per_clip_seconds: dict[str, float] = {}
    clip_manifest: list[dict[str, Any]] = []
    total_frames = 0
    total_feature_bytes = 0
    for clip in clips:
        audio = adapter.audio(clip.clip_id)
        started = time.perf_counter()
        features = featurizer.featurize(audio)
        elapsed = time.perf_counter() - started
        per_clip_seconds[clip.clip_id] = elapsed

        if features.shape != (clip.n_frames, D_FEAT):
            raise FeatureCacheRefusal(
                f"clip {clip.clip_id} featurized to {features.shape}, expected "
                f"{(clip.n_frames, D_FEAT)}"
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

    featurize_seconds = float(sum(per_clip_seconds.values()))
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
    manifest_bytes = canonical_bytes(manifest)
    manifest_path = out_dir / _MANIFEST_NAME
    manifest_path.write_bytes(manifest_bytes)

    n_clips = len(clips)
    return BuildReport(
        cache_key=key,
        cache_dir=out_dir,
        manifest_path=manifest_path,
        n_clips=n_clips,
        total_frames=total_frames,
        total_feature_bytes=total_feature_bytes,
        manifest_bytes=len(manifest_bytes),
        featurize_seconds=featurize_seconds,
        seconds_per_clip=featurize_seconds / n_clips if n_clips else 0.0,
        per_clip_seconds=per_clip_seconds,
    )


# ---------------------------------------------------------------------------
# Load: reconstruct exactly what the harness / gate / referee consume.
# ---------------------------------------------------------------------------


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FeatureCacheRefusal(f"no feature-cache manifest at {manifest_path}; build the cache first")
    import json

    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    if manifest.get("schema") != CACHE_SCHEMA:
        raise FeatureCacheRefusal(f"unexpected cache schema {manifest.get('schema')!r}")
    return manifest


def _load_feature_block(cache_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    clip_id = entry["clip"]["clip_id"]
    n_frames, d_feat = entry["feature_shape"]
    path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
    if not path.is_file():
        raise FeatureCacheRefusal(f"cached feature block missing for {clip_id} at {path}")
    raw = path.read_bytes()
    if len(raw) != entry["feature_bytes"]:
        raise FeatureCacheRefusal(
            f"cached feature block for {clip_id} is {len(raw)} bytes, manifest says "
            f"{entry['feature_bytes']}"
        )
    features = np.frombuffer(raw, dtype=_FEATURE_DTYPE).reshape(int(n_frames), int(d_feat)).copy()
    digest = FrozenFeaturizer().feature_digest(features)
    if digest != entry["feature_sha256"]:
        raise FeatureCacheRefusal(
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
    """Load the featurize-free corpus a variant run consumes: Clip objects, features, and the split.

    Every feature block is re-verified against its manifest sha256 on load, so a corrupt or stale cache
    is refused rather than silently scored. The returned ``split`` and ``features_by_clip`` are exactly
    what ``real_artifact._run_seed_real`` reads, so a variant producer is a drop-in with no featurize.
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
            raise FeatureCacheRefusal(f"split references clips absent from the cache: {missing}")
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


# ---------------------------------------------------------------------------
# Verification: cached bytes must equal a fresh featurization on a sample of clips.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerifyReport:
    """Result of re-featurizing a sample of clips and comparing to the cached bytes."""

    checked_clips: tuple[str, ...]
    all_byte_identical: bool
    per_clip: dict[str, bool]

    def numbers(self) -> dict[str, Any]:
        return {
            "checked_clips": list(self.checked_clips),
            "all_byte_identical": self.all_byte_identical,
        }


def verify_cache_bytes(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
    n_clips: int = 2,
) -> VerifyReport:
    """Re-featurize the first ``n_clips`` clips from a fresh adapter and assert byte-identity to the cache.

    This proves the cache is a faithful stand-in for the frozen featurizer: a variant that reads the
    cache scores byte-identically to one that recomputes the front-end.
    """

    if n_clips <= 0:
        raise FeatureCacheRefusal("n_clips must be positive")
    cache_dir = cache_dir_for(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    featurizer = FrozenFeaturizer()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    check_ids = [clip.clip_id for clip in adapter.clips()[:n_clips]]

    per_clip: dict[str, bool] = {}
    for clip_id in check_ids:
        fresh = _feature_bytes(featurizer.featurize(adapter.audio(clip_id)))
        cached_path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
        if not cached_path.is_file():
            raise FeatureCacheRefusal(f"cached feature block missing for {clip_id} at {cached_path}")
        per_clip[clip_id] = cached_path.read_bytes() == fresh

    return VerifyReport(
        checked_clips=tuple(check_ids),
        all_byte_identical=all(per_clip.values()),
        per_clip=per_clip,
    )


def _main(argv: list[str] | None = None) -> int:
    """Build the cache, verify two clips byte-for-byte, and print the timing and size numbers."""

    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build the STARSS23 ESCS feature cache.")
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
    verify = verify_cache_bytes(
        cache_root=args.cache_root,
        foa_root=args.foa,
        metadata_root=args.metadata,
        max_frames=args.max_frames,
        n_val_rooms=args.n_val_rooms,
        n_clips=2,
    )
    out = {"build": report.numbers(), "verify": verify.numbers()}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if verify.all_byte_identical else 1


if __name__ == "__main__":
    raise SystemExit(_main())
