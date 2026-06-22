"""Real-video ingestion. The missing piece between cached synthetic latents and real natural
video: decode video files -> clips shaped for V-JEPA (pixel_values_videos = [B,T,3,H,W]) ->
the existing cache_latents pipeline -> the memmap store the whole shell already consumes.

The PREPROCESSING core (sample frames, resize, normalize) is torch-only and fully tested. The
video DECODE is a lazy backend (torchvision.io, then decord) so the package adds no hard dep;
install one via `uv pip install -e ".[video]"`. On the Studio: drop class-foldered clips and run
scripts/cache_video.py. Today (no codec / no clips) the preprocessing path is exercised on
synthetic frames, so it is verified before any real video exists.

HARDENING: validate_source checks the class-folder layout before a single byte is decoded;
caching persists a sorted-folder->index label_map.json so labels are reproducible; every clip
gets a sha256 content hash so duplicates are detectable (warned + counted); short clips pad to
frames_per_clip; corrupt/undecodable files are skipped (logged + counted), never crashing the
whole cache; a pre-existing partial store/label_map is detected and reported. None of this adds
a dependency: the decode backend stays lazy and everything else is torch+stdlib.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

from ..logging_utils import get_logger

log = get_logger("substrate.video")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
FIXTURE_EXTS = (".npy",)  # codec-free mocked clips (the rehearsal corpus); same pipeline
INGEST_EXTS = VIDEO_EXTS + FIXTURE_EXTS
LABEL_MAP_NAME = "label_map.json"


def preprocess_clip(
    frames: torch.Tensor,
    frames_per_clip: int = 64,
    res: int = 256,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
) -> torch.Tensor:
    """frames [T,H,W,3] (uint8 or float) or [T,3,H,W] -> normalized clip [frames_per_clip,3,res,res].
    Uniform temporal subsample/pad to frames_per_clip, bilinear resize, ImageNet normalization.
    SHORT clips (T < frames_per_clip) are handled here: linspace over [0,T-1] repeats frames to
    fill the budget, so a 2-frame clip still yields frames_per_clip frames (verified by test)."""
    f = torch.as_tensor(frames)
    if f.dim() != 4:
        raise ValueError(f"frames must be 4D [T,H,W,3] or [T,3,H,W], got {tuple(f.shape)}")
    if f.shape[0] == 0:
        raise ValueError("empty clip: 0 frames decoded")
    if f.shape[-1] == 3 and f.shape[1] != 3:  # [T,H,W,3] -> [T,3,H,W]
        f = f.permute(0, 3, 1, 2)
    f = f.float()
    if f.max() > 1.5:  # uint8-range -> [0,1]
        f = f / 255.0
    t = f.shape[0]
    idx = torch.linspace(0, max(0, t - 1), frames_per_clip).round().long().clamp(0, t - 1)
    f = f[idx]  # [frames_per_clip,3,H,W]; short clips pad by repeating sampled frames
    f = F.interpolate(f, size=(res, res), mode="bilinear", align_corners=False)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (f - m) / s


def read_video(path: str | Path) -> torch.Tensor:
    """Decode a clip to frames [T,H,W,3] uint8. A .npy fixture is a codec-free MOCKED decode (the
    rehearsal corpus); real video uses a lazy backend (torchvision.io, then decord) and raises a
    clear unblock if none is installed."""
    p = str(path)
    if p.endswith(".npy"):  # mocked decode: the rehearsal corpus, no codec needed
        import numpy as np

        return torch.from_numpy(np.load(p))
    try:
        from torchvision.io import read_video

        v, _, _ = read_video(p, output_format="THWC", pts_unit="sec")
        return v
    except ImportError:
        pass
    try:
        import decord

        return torch.from_numpy(decord.VideoReader(p)[:].asnumpy())
    except ImportError:
        pass
    raise RuntimeError(
        'no video backend installed; run `uv pip install -e ".[video]"` (torchvision) '
        "or `uv pip install decord` to decode real video files"
    )


def clip_hash(clip: torch.Tensor) -> str:
    """sha256 over a clip's decoded+preprocessed bytes. Identifies content so duplicate clips
    (same pixels under different filenames) are detectable. Cast to a fixed dtype first so the
    digest is stable across the uint8/float input paths preprocess_clip accepts."""
    c = torch.as_tensor(clip).detach().to(torch.float32).contiguous().cpu()
    return hashlib.sha256(c.numpy().tobytes()).hexdigest()


def list_class_files(source: str | Path) -> tuple[list[str], list[tuple[Path, int]]]:
    """(sorted class names, [(file, label)]) for a class-foldered source. Label is the sorted
    class-folder index, the single source of truth shared by validate_source and iter_video_clips."""
    root = Path(source)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    files = [
        (f, cidx[c])
        for c in classes
        for f in sorted((root / c).iterdir())
        if f.is_file() and f.suffix.lower() in INGEST_EXTS
    ]
    return classes, files


def validate_source(source: str | Path) -> dict:
    """Check the class-folder layout BEFORE decoding anything, and return a manifest:
    {classes, n_clips, per_class:{name:count}, label_map:{name:index}}.
    Raises ValueError with a clear, actionable message on: missing dir, a path that is not a
    directory, no class subfolders, classes that contain no video files (lists the empties)."""
    root = Path(source)
    if not root.exists():
        raise ValueError(f"source {root} does not exist (expected a dir of <class>/<clip>.mp4)")
    if not root.is_dir():
        raise ValueError(f"source {root} is not a directory (expected a dir of <class>/<clip>.mp4)")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise ValueError(
            f"source {root} has no class subfolders; expected <source>/<class>/<clip>.mp4 "
            f"(one folder per label)"
        )
    label_map = {c: i for i, c in enumerate(classes)}
    per_class = {
        c: sum(1 for f in sorted((root / c).iterdir()) if f.is_file() and f.suffix.lower() in INGEST_EXTS)
        for c in classes
    }
    empty = [c for c, n in per_class.items() if n == 0]
    if empty:
        raise ValueError(
            f"class folder(s) with no {list(INGEST_EXTS)} clips: {empty} under {root}; "
            f"every class needs at least one clip"
        )
    n_clips = sum(per_class.values())
    if n_clips == 0:  # defensive: per-class all-empty would already have raised
        raise ValueError(f"source {root} contains no video files in any class folder")
    return {"classes": classes, "n_clips": n_clips, "per_class": per_class, "label_map": label_map}


def write_label_map(cache_dir: str | Path, label_map: dict[str, int]) -> Path:
    """Persist the sorted-folder->index label_map beside the store so labels are reproducible
    across cache runs. Returns the path written."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / LABEL_MAP_NAME
    p.write_text(json.dumps(label_map, indent=2, sort_keys=True))
    return p


def read_label_map(cache_dir: str | Path) -> dict[str, int] | None:
    """Read a persisted label_map if present, else None."""
    p = Path(cache_dir) / LABEL_MAP_NAME
    if not p.exists():
        return None
    return json.loads(p.read_text())


def detect_partial_cache(cache_dir: str | Path, name: str) -> dict:
    """Look for an existing (possibly partial) store/label_map at <cache_dir>/<name> so a re-run
    is not silently destructive. Returns {exists, store_dir, has_meta, count, has_label_map}.
    Caching is not auto-resumed (cache_latents rewrites the memmap); this gives the caller a clean
    report + a clear message to act on rather than a half-overwritten store."""
    store_dir = Path(cache_dir) / name
    meta_p = store_dir / "meta.json"
    out = {
        "exists": store_dir.exists(),
        "store_dir": str(store_dir),
        "has_meta": meta_p.exists(),
        "count": 0,
        "has_label_map": (store_dir / LABEL_MAP_NAME).exists() or (Path(cache_dir) / LABEL_MAP_NAME).exists(),
    }
    if meta_p.exists():
        try:
            out["count"] = int(json.loads(meta_p.read_text()).get("count", 0))
        except Exception:
            out["count"] = 0
    if out["exists"]:
        log.warning(
            "existing cache at %s (meta=%s count=%d label_map=%s); cache_latents will OVERWRITE "
            "the memmap from position 0 (not an incremental resume). Move/clear it to start clean.",
            store_dir,
            out["has_meta"],
            out["count"],
            out["has_label_map"],
        )
    return out


def iter_clip_records(
    source: str | Path,
    frames_per_clip: int = 64,
    res: int = 256,
    limit: int | None = None,
) -> Iterator[tuple[torch.Tensor, int, str]]:
    """Per-clip generator under the batching layer: yields (clip [T,3,res,res], label, sha256).
    CORRUPT/undecodable files are skipped with a logged warning (never crash the whole cache);
    SHORT clips are padded by preprocess_clip. Tracks duplicate content hashes and logs a summary.
    iter_video_clips wraps this for the (clip_batch, label_batch) contract cache_latents consumes."""
    _classes, files = list_class_files(source)
    if limit:
        files = files[:limit]
    seen: dict[str, str] = {}
    n_ok = n_corrupt = n_dup = 0
    for f, label in files:
        try:
            frames = read_video(f)
            clip = preprocess_clip(frames, frames_per_clip, res)
        except RuntimeError:
            raise  # no-backend unblock must surface, not be swallowed as a corrupt clip
        except Exception as e:  # undecodable / truncated / empty: skip, keep the cache going
            n_corrupt += 1
            log.warning("skip undecodable clip %s (%s: %s)", f, type(e).__name__, e)
            continue
        h = clip_hash(clip)
        if h in seen:
            n_dup += 1
            log.warning("duplicate clip content: %s repeats %s (hash %s)", f, seen[h], h[:12])
        else:
            seen[h] = str(f)
        n_ok += 1
        yield clip, label, h
    log.info(
        "clips decoded: %d ok, %d duplicate-content, %d skipped (corrupt/undecodable)",
        n_ok,
        n_dup,
        n_corrupt,
    )


def iter_video_clips(
    source: str | Path,
    frames_per_clip: int = 64,
    res: int = 256,
    batch: int = 2,
    limit: int | None = None,
    hashes_out: list[str] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Walk `source` as class-foldered video files (source/<class>/<clip>.mp4) and yield
    (clip_batch [B,T,3,res,res], label_batch [B]). Labels are the sorted class-folder index.
    This is the iterator cache_latents consumes to build a real-video latent store. Corrupt files
    are skipped and short clips padded (see iter_clip_records). If `hashes_out` is given, each
    kept clip's sha256 is appended to it (so the caller can record/inspect content hashes)."""
    buf_x, buf_y = [], []
    for clip, label, h in iter_clip_records(source, frames_per_clip, res, limit):
        buf_x.append(clip)
        buf_y.append(label)
        if hashes_out is not None:
            hashes_out.append(h)
        if len(buf_x) == batch:
            yield torch.stack(buf_x), torch.tensor(buf_y)
            buf_x, buf_y = [], []
    if buf_x:  # partial final batch
        yield torch.stack(buf_x), torch.tensor(buf_y)
