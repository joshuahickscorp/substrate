"""Tiny deterministic video-corpus generator for the Studio rehearsal capsule. Writes a
class-foldered corpus of CLIPS to <root>/<class>/<clip>.npy (uint8 [T,H,W,3]) with NO external
dataset and NO codec: a .npy clip is the honest mocked equivalent of an .mp4 on a machine
without a video backend (the same validate -> decode -> preprocess -> cache contract runs over
it; only the decode is mocked, swapped for real .mp4 decode on the Studio).

Deterministic by seed. Optionally injects a duplicate clip (identical content, different name)
and a short clip (fewer frames) so the rehearsal exercises duplicate-detection and short-clip
padding, not just the happy path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _class_clip(rng: np.random.Generator, c: int, frames: int, h: int, w: int) -> np.ndarray:
    """A class-conditioned clip: a per-class base color + a moving bright block, plus light
    noise, so distinct classes are distinguishable (the frozen-random encoder still separates
    them enough for a rehearsal). uint8 [T,H,W,3]."""
    base = np.zeros((frames, h, w, 3), dtype=np.float32)
    base[..., c % 3] = 0.4  # per-class color channel
    for t in range(frames):
        x = (t * 3 + c * 5) % max(1, w - 4)
        base[t, h // 2 : h // 2 + 4, x : x + 4, :] = 1.0  # moving block, class-shifted
    base += 0.05 * rng.standard_normal(base.shape).astype(np.float32)
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def generate_video_corpus(
    root: str | Path,
    n_classes: int = 2,
    clips_per_class: int = 3,
    frames: int = 8,
    h: int = 32,
    w: int = 32,
    seed: int = 0,
    inject_duplicate: bool = True,
    inject_short: bool = True,
) -> dict:
    """Write <root>/<class>/<clip>.npy and return a manifest:
    {root, classes, n_clips, per_class, ext, duplicate, short, mocked: True}."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    classes = [f"class{c}" for c in range(n_classes)]
    per_class: dict[str, int] = {}
    duplicate = short = None
    for c, name in enumerate(classes):
        (root / name).mkdir(parents=True, exist_ok=True)
        clips = []
        for j in range(clips_per_class):
            clip = _class_clip(rng, c, frames, h, w)
            p = root / name / f"clip{j}.npy"
            np.save(p, clip)
            clips.append((p, clip))
        # a SHORT clip (fewer frames) in the first class, to exercise temporal padding
        if inject_short and c == 0:
            sclip = _class_clip(rng, c, max(1, frames // 2), h, w)
            np.save(root / name / "clip_short.npy", sclip)
            short = str(root / name / "clip_short.npy")
        # a DUPLICATE of clip0 (identical content, different name) to exercise dup detection
        if inject_duplicate and c == 0:
            np.save(root / name / "clip_dup.npy", clips[0][1])
            duplicate = str(root / name / "clip_dup.npy")
        per_class[name] = len(list((root / name).glob("*.npy")))
    return {
        "root": str(root),
        "classes": classes,
        "n_clips": sum(per_class.values()),
        "per_class": per_class,
        "ext": ".npy",
        "duplicate": duplicate,
        "short": short,
        "mocked": True,
    }
