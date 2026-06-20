"""Real-video ingestion. The missing piece between cached synthetic latents and real natural
video: decode video files -> clips shaped for V-JEPA (pixel_values_videos = [B,T,3,H,W]) ->
the existing cache_latents pipeline -> the memmap store the whole shell already consumes.

The PREPROCESSING core (sample frames, resize, normalize) is torch-only and fully tested. The
video DECODE is a lazy backend (torchvision.io, then decord) so the package adds no hard dep;
install one via `uv pip install -e ".[video]"`. On the Studio: drop class-foldered clips and run
scripts/cache_video.py. Today (no codec / no clips) the preprocessing path is exercised on
synthetic frames, so it is verified before any real video exists.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def preprocess_clip(
    frames: torch.Tensor,
    frames_per_clip: int = 64,
    res: int = 256,
    mean: tuple = IMAGENET_MEAN,
    std: tuple = IMAGENET_STD,
) -> torch.Tensor:
    """frames [T,H,W,3] (uint8 or float) or [T,3,H,W] -> normalized clip [frames_per_clip,3,res,res].
    Uniform temporal subsample/pad to frames_per_clip, bilinear resize, ImageNet normalization."""
    f = torch.as_tensor(frames)
    if f.dim() != 4:
        raise ValueError(f"frames must be 4D [T,H,W,3] or [T,3,H,W], got {tuple(f.shape)}")
    if f.shape[-1] == 3 and f.shape[1] != 3:  # [T,H,W,3] -> [T,3,H,W]
        f = f.permute(0, 3, 1, 2)
    f = f.float()
    if f.max() > 1.5:  # uint8-range -> [0,1]
        f = f / 255.0
    t = f.shape[0]
    idx = torch.linspace(0, max(0, t - 1), frames_per_clip).round().long().clamp(0, t - 1)
    f = f[idx]  # [frames_per_clip,3,H,W]
    f = F.interpolate(f, size=(res, res), mode="bilinear", align_corners=False)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (f - m) / s


def read_video(path: str | Path) -> torch.Tensor:
    """Decode a video file to frames [T,H,W,3] uint8 via a lazy backend. Raises a clear unblock
    if no backend is installed."""
    p = str(path)
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


def iter_video_clips(
    source: str | Path,
    frames_per_clip: int = 64,
    res: int = 256,
    batch: int = 2,
    limit: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Walk `source` as class-foldered video files (source/<class>/<clip>.mp4) and yield
    (clip_batch [B,T,3,res,res], label_batch [B]). Labels are the sorted class-folder index.
    This is the iterator cache_latents consumes to build a real-video latent store."""
    root = Path(source)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    cidx = {c: i for i, c in enumerate(classes)}
    files = [
        (f, cidx[c]) for c in classes for f in sorted((root / c).iterdir()) if f.suffix.lower() in VIDEO_EXTS
    ]
    if limit:
        files = files[:limit]
    buf_x, buf_y = [], []
    for f, label in files:
        buf_x.append(preprocess_clip(read_video(f), frames_per_clip, res))
        buf_y.append(label)
        if len(buf_x) == batch:
            yield torch.stack(buf_x), torch.tensor(buf_y)
            buf_x, buf_y = [], []
    if buf_x:
        yield torch.stack(buf_x), torch.tensor(buf_y)
