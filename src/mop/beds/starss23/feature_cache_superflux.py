"""A net-new feature cache for the frozen SuperFlux front-end (STARSS23 ESCS bed).

This is an additive component. It changes no sealed scoring logic and it never touches the committed
frozen-featurizer cache: it is a faithful twin of ``feature_cache.py`` bound to
``SuperfluxSpectralFeaturizer`` instead of ``FrozenFeaturizer``. The SuperFlux front-end is deterministic
and gate-independent, so this module featurizes the real, room-disjoint STARSS23 FOA subset EXACTLY ONCE
with the SuperFlux front-end and caches the per-clip feature arrays, the onset labels, and the native
fold-respecting split to a byte-reproducible on-disk cache. The producer then reads this cache, so the
run is featurize-free on repeat.

The cache is keyed by a canonical digest over the SuperFlux front-end's ``parameter_digest`` (which
differs from the base front-end because MU and the frequency max filter are new fixed DSP) and its
``FLOPS_PER_FRAME``, so this cache lands in its own ``<cache_key>`` subdirectory and can never be served
the base front-end's bytes, nor serve them. Feature blocks are raw little-endian float64 with a recorded
sha256, re-verified on load against the SuperFlux front-end's own ``feature_digest``.

Caching is a wall-clock optimization only, never a budget reduction. The manifest records the honest
SuperFlux per-clip and total featurize FLOPs so the downstream FLOP ledger charges the front-end per arm
exactly as if it had been recomputed.

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
from .feature_cache import DEFAULT_CACHE_ROOT, CachedCorpus, FeatureCacheRefusal
from .featurizer_superflux_spectral import D_FEAT, FLOPS_PER_FRAME, SuperfluxSpectralFeaturizer
from .real_artifact import (
    DEFAULT_FOA_ROOT,
    DEFAULT_METADATA_ROOT,
    DEFAULT_N_VAL_ROOMS,
    _fold_respecting_split,
)
from .schema import Clip, ClipSplit

SUPERFLUX_CACHE_SCHEMA = "mop-starss23-escs-superflux-feature-cache/v1"

_FEATURE_DTYPE = "<f8"
_FEATURE_SUFFIX = ".f8"
_FEATURES_SUBDIR = "features"
_MANIFEST_NAME = "manifest.json"


def _config_payload(
    *,
    featurizer: SuperfluxSpectralFeaturizer,
    foa_root: Path,
    metadata_root: Path,
    max_frames: int | None,
    n_val_rooms: int,
) -> dict[str, Any]:
    """The canonical config a SuperFlux cache directory is bound to. Any change mints a fresh key."""

    return {
        "schema": SUPERFLUX_CACHE_SCHEMA,
        "bed_id": BED_ID,
        "featurizer_id": "superflux_spectral",
        "featurizer_parameter_digest": featurizer.parameter_digest(),
        "featurizer_n_params": featurizer.n_params(),
        "flops_per_frame": FLOPS_PER_FRAME,
        "foa_root": str(Path(foa_root)),
        "metadata_root": str(Path(metadata_root)),
        "max_frames": max_frames,
        "n_val_rooms": int(n_val_rooms),
    }


def superflux_cache_key(
    *,
    featurizer: SuperfluxSpectralFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> str:
    """Return the canonical cache key for the SuperFlux front-end plus config. Reads no audio."""

    featurizer = featurizer or SuperfluxSpectralFeaturizer()
    payload = _config_payload(
        featurizer=featurizer,
        foa_root=Path(foa_root),
        metadata_root=Path(metadata_root),
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    return canonical_sha256(payload)


def superflux_cache_dir_for(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    featurizer: SuperfluxSpectralFeaturizer | None = None,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> Path:
    """Resolve the cache subdirectory the SuperFlux front-end plus config lives in."""

    key = superflux_cache_key(
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
class BuildReport:
    """Timing and size provenance of one SuperFlux cache build. All wall-clock, none sealed as budget."""

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
            "total_superflux_featurize_flops": int(FLOPS_PER_FRAME * self.total_frames),
            "featurize_seconds": round(self.featurize_seconds, 3),
        }


def build_superflux_feature_cache(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> BuildReport:
    """Featurize the real room-disjoint subset once with the SuperFlux front-end and seal the cache.

    The split is the native fold-respecting split (test is exactly fold-4 dev-test), reused unchanged from
    the real producer, so this cache is a drop-in for the harness / gate / referee with no featurize.
    """

    featurizer = SuperfluxSpectralFeaturizer()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    split = _fold_respecting_split(adapter, n_val_rooms)

    clips = adapter.clips()
    key = superflux_cache_key(
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
            raise FeatureCacheRefusal(
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
        "schema": SUPERFLUX_CACHE_SCHEMA,
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
            "featurizer_id": "superflux_spectral",
            "n_params": featurizer.n_params(),
            "parameter_digest": featurizer.parameter_digest(),
            "flops_per_frame": FLOPS_PER_FRAME,
        },
        "featurize_charge_note": (
            "caching is a wall-clock optimization only; featurize_flops is charged per arm in the FLOP "
            "ledger exactly as if the SuperFlux front-end had been recomputed, so the cache is not a "
            "budget cut"
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
        raise FeatureCacheRefusal(
            f"no SuperFlux feature-cache manifest at {manifest_path}; build the cache first"
        )
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    if manifest.get("schema") != SUPERFLUX_CACHE_SCHEMA:
        raise FeatureCacheRefusal(f"unexpected SuperFlux cache schema {manifest.get('schema')!r}")
    return manifest


def _load_feature_block(cache_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    clip_id = entry["clip"]["clip_id"]
    n_frames, d_feat = entry["feature_shape"]
    path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
    if not path.is_file():
        raise FeatureCacheRefusal(f"cached SuperFlux feature block missing for {clip_id} at {path}")
    raw = path.read_bytes()
    if len(raw) != entry["feature_bytes"]:
        raise FeatureCacheRefusal(
            f"cached feature block for {clip_id} is {len(raw)} bytes, manifest says {entry['feature_bytes']}"
        )
    features = np.frombuffer(raw, dtype=_FEATURE_DTYPE).reshape(int(n_frames), int(d_feat)).copy()
    digest = SuperfluxSpectralFeaturizer().feature_digest(features)
    if digest != entry["feature_sha256"]:
        raise FeatureCacheRefusal(
            f"cached SuperFlux feature block for {clip_id} failed its integrity digest (cache is corrupt)"
        )
    return features


def load_superflux_cached_corpus(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> CachedCorpus:
    """Load the featurize-free SuperFlux corpus: real Clip objects, their SuperFlux features, the split."""

    cache_dir = superflux_cache_dir_for(
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


def load_or_build_superflux_cached_corpus(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
) -> CachedCorpus:
    """Load the SuperFlux cache, building it once from the real subset if it is not present yet."""

    cache_dir = superflux_cache_dir_for(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    if not (cache_dir / _MANIFEST_NAME).is_file():
        build_superflux_feature_cache(
            cache_root=cache_root,
            foa_root=foa_root,
            metadata_root=metadata_root,
            max_frames=max_frames,
            n_val_rooms=n_val_rooms,
        )
    return load_superflux_cached_corpus(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )


def verify_cache_bytes(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    foa_root: str | Path = DEFAULT_FOA_ROOT,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    max_frames: int | None = None,
    n_val_rooms: int = DEFAULT_N_VAL_ROOMS,
    n_clips: int = 2,
) -> dict[str, Any]:
    """Re-featurize the first ``n_clips`` clips and assert byte-identity to the cached SuperFlux bytes."""

    if n_clips <= 0:
        raise FeatureCacheRefusal("n_clips must be positive")
    cache_dir = superflux_cache_dir_for(
        cache_root=cache_root,
        foa_root=foa_root,
        metadata_root=metadata_root,
        max_frames=max_frames,
        n_val_rooms=n_val_rooms,
    )
    featurizer = SuperfluxSpectralFeaturizer()
    adapter = RealStarssAdapter(foa_root, metadata_root, rights_clean=True, max_frames=max_frames)
    check_ids = [clip.clip_id for clip in adapter.clips()[:n_clips]]
    per_clip: dict[str, bool] = {}
    for clip_id in check_ids:
        fresh = _feature_bytes(featurizer.featurize(adapter.audio(clip_id)))
        cached_path = cache_dir / _FEATURES_SUBDIR / f"{clip_id}{_FEATURE_SUFFIX}"
        if not cached_path.is_file():
            raise FeatureCacheRefusal(f"cached feature block missing for {clip_id} at {cached_path}")
        per_clip[clip_id] = cached_path.read_bytes() == fresh
    return {"checked_clips": check_ids, "all_byte_identical": all(per_clip.values()), "per_clip": per_clip}


def _main(argv: list[str] | None = None) -> int:
    """Build the SuperFlux cache, verify two clips byte-for-byte, and print the timing and size numbers."""

    import argparse

    parser = argparse.ArgumentParser(description="Build the STARSS23 ESCS SuperFlux feature cache.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--foa", default=str(DEFAULT_FOA_ROOT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_ROOT))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--n-val-rooms", type=int, default=DEFAULT_N_VAL_ROOMS)
    args = parser.parse_args(argv)

    report = build_superflux_feature_cache(
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
    out = {"build": report.numbers(), "verify": {k: verify[k] for k in ("checked_clips", "all_byte_identical")}}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if verify["all_byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
