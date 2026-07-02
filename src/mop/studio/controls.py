"""Synthetic control expansion (Frontier 7). Deterministic generated video packs for rehearsal,
controlled gates, and regression fixtures. NOT science claims: these are controls (a known cause
with a known expected effect), rehearsal data (exercise the real validate/decode/cache contract
without a download), and fixtures (regenerate identically by seed).

Each family is a class-foldered corpus of .npy clips (uint8 [T,H,W,3]) written under
<out>/<family>/<class>/<clip>.npy, the same layout validate_source + iter_video_clips consume, so a
control corpus flows through the exact pipeline a real .mp4 dataset will. The two classes per
family encode one controlled contrast (object permanence: persists vs vanishes; occlusion: revealed
vs stays hidden; and so on). Generation honors a byte budget (the profile fixture budget) and stops
cleanly when hit, reporting truncation rather than overrunning disk.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

Clip = np.ndarray  # uint8 [T,H,W,3]


def _blank(frames: int, h: int, w: int) -> np.ndarray:
    return np.zeros((frames, h, w, 3), dtype=np.float32)


def _put(canvas: np.ndarray, t: int, y: int, x: int, sz: int, color: tuple[float, float, float]) -> None:
    """Draw a small filled square of `color` at (y,x) on frame t, clipped to bounds."""
    H, W = canvas.shape[1], canvas.shape[2]
    y0, y1 = max(0, y), min(H, y + sz)
    x0, x1 = max(0, x), min(W, x + sz)
    if y1 > y0 and x1 > x0:
        canvas[t, y0:y1, x0:x1, :] = color


def _finish(base: np.ndarray, rng: np.random.Generator, noise: float = 0.04) -> Clip:
    base = base + noise * rng.standard_normal(base.shape).astype(np.float32)
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


# ---- control families. Each: (rng, frames, h, w, class_idx) -> uint8 clip [T,H,W,3] ----
# class 0 vs class 1 encode the controlled contrast named in the docstring.


def _moving_object(rng, frames, h, w, c) -> Clip:
    """Direction of motion: class 0 moves left->right, class 1 moves right->left."""
    base = _blank(frames, h, w)
    sz = max(2, w // 8)
    for t in range(frames):
        frac = t / max(1, frames - 1)
        x = int(frac * (w - sz)) if c == 0 else int((1 - frac) * (w - sz))
        _put(base, t, h // 2, x, sz, (0.9, 0.9, 0.2))
    return _finish(base, rng)


def _object_permanence(rng, frames, h, w, c) -> Clip:
    """Behind an occluder: class 0 the object reappears (permanence holds), class 1 it never
    returns (a permanence violation). Same first half, different second half."""
    base = _blank(frames, h, w)
    sz = max(2, w // 8)
    occ_x = w // 2
    for t in range(frames):
        frac = t / max(1, frames - 1)
        x = int(frac * (w - sz))
        hidden = occ_x - sz <= x <= occ_x + sz
        # occluder bar
        _put(base, t, 0, occ_x, max(2, w // 12), (0.2, 0.2, 0.5))
        if not hidden and (c == 0 or x <= occ_x):
            _put(base, t, h // 2, x, sz, (0.2, 0.9, 0.3))
    return _finish(base, rng)


def _occlusion_reveal(rng, frames, h, w, c) -> Clip:
    """Occluder lifts: class 0 reveals the object behind it, class 1 stays occluded the whole clip."""
    base = _blank(frames, h, w)
    sz = max(2, w // 6)
    _put_obj_y = h // 2
    for t in range(frames):
        _put(base, t, _put_obj_y, w // 2, sz, (0.9, 0.4, 0.2))  # the object, always present
        frac = t / max(1, frames - 1)
        bar_h = h if c == 1 else int(h * (1 - frac))  # class 0 the bar shrinks (reveals)
        if bar_h > 0:
            base[t, 0:bar_h, w // 2 - sz : w // 2 + sz, :] = (0.3, 0.3, 0.3)
    return _finish(base, rng)


def _relation_change(rng, frames, h, w, c) -> Clip:
    """Spatial relation: class 0 two objects swap vertical order (above<->below), class 1 keep it."""
    base = _blank(frames, h, w)
    sz = max(2, w // 8)
    x = w // 2
    for t in range(frames):
        frac = t / max(1, frames - 1)
        if c == 0:
            ya = int((h - sz) * frac)
            yb = int((h - sz) * (1 - frac))
        else:
            ya, yb = int(h * 0.25), int(h * 0.6)
        _put(base, t, ya, x - sz, sz, (0.9, 0.2, 0.2))
        _put(base, t, yb, x + 1, sz, (0.2, 0.4, 0.9))
    return _finish(base, rng)


def _containment(rng, frames, h, w, c) -> Clip:
    """Containment vs contact: class 0 the object enters a container (stays inside), class 1 it
    bounces off the container wall (contact, then away)."""
    base = _blank(frames, h, w)
    sz = max(2, w // 10)
    cont_y = h // 2
    for t in range(frames):
        # container (open box drawn as two side bars)
        _put(base, t, cont_y, w // 2 - 3 * sz, sz, (0.3, 0.5, 0.3))
        _put(base, t, cont_y, w // 2 + 2 * sz, sz, (0.3, 0.5, 0.3))
        frac = t / max(1, frames - 1)
        if c == 0:
            x = int((w // 2 - 2 * sz) * frac)  # moves in and stays
        else:
            x = int((w // 2 - 2 * sz) * (frac if frac < 0.5 else (1 - frac)))  # in then back out
        _put(base, t, cont_y, x, sz, (0.95, 0.85, 0.1))
    return _finish(base, rng)


def _noisy_tv(rng, frames, h, w, c) -> Clip:
    """Noisy-TV distractor: class 0 a structured moving signal, class 1 pure unpredictable noise
    (aleatoric; a correct curiosity/uncertainty mechanism must NOT chase it)."""
    if c == 1:
        return (rng.integers(0, 256, size=(frames, h, w, 3))).astype(np.uint8)
    base = _blank(frames, h, w)
    sz = max(2, w // 8)
    for t in range(frames):
        x = int((t / max(1, frames - 1)) * (w - sz))
        _put(base, t, h // 3, x, sz, (0.2, 0.9, 0.9))
    return _finish(base, rng)


def _navigation(rng, frames, h, w, c) -> Clip:
    """Navigation-like camera transition: class 0 forward translation (a block scales up, dolly),
    class 1 rotation (the block pans across, yaw)."""
    base = _blank(frames, h, w)
    for t in range(frames):
        frac = t / max(1, frames - 1)
        if c == 0:
            sz = max(2, int(w * (0.1 + 0.6 * frac)))  # grows (approach)
            _put(base, t, h // 2 - sz // 2, w // 2 - sz // 2, sz, (0.8, 0.8, 0.8))
        else:
            sz = max(2, w // 6)
            x = int(frac * (w - sz))  # sweeps across (pan)
            _put(base, t, h // 2 - sz // 2, x, sz, (0.8, 0.8, 0.8))
    return _finish(base, rng)


# class/domain-incremental families parameterize the SAME generator across more classes/domains.
def _class_incremental(rng, frames, h, w, c) -> Clip:
    """One class of a class-incremental stream: a block whose color + start position is class-keyed,
    so adding classes grows the label space (use n_classes > 2)."""
    base = _blank(frames, h, w)
    sz = max(2, w // 8)
    color = ((c * 53 % 256) / 255.0, (c * 97 % 256) / 255.0, (c * 151 % 256) / 255.0)
    for t in range(frames):
        x = (int((t / max(1, frames - 1)) * (w - sz)) + c * 3) % max(1, w - sz)
        _put(base, t, (h // 2 + c * 2) % max(1, h - sz), x, sz, color)
    return _finish(base, rng)


def _domain_incremental(rng, frames, h, w, c) -> Clip:
    """Domain-incremental shift: same two motion classes, but a domain tint/contrast shift is baked
    in via the seed offset upstream; here class 0/1 are the shared labels under shifted statistics."""
    base = _moving_object(rng, frames, h, w, c).astype(np.float32) / 255.0
    base = 0.6 * base + 0.2  # contrast/brightness domain shift
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def _aleatoric_tv(rng, frames, h, w, c) -> Clip:
    """The genuine noisy-TV trap for the CURRICULUM engine: BOTH classes are pure unpredictable noise
    drawn from the same distribution, so the class label is UNDECODABLE from content (aleatoric). A
    learning-progress chooser must reject it; an error-seeking chooser would chase it forever. (This
    differs from _noisy_tv, whose signal-vs-noise classes ARE separable; that one is a distractor for
    uncertainty gating, this one is an undecodable-label trap for curriculum.)"""
    del c  # label carries no information: identical noise distribution for every class
    return (rng.integers(0, 256, size=(frames, h, w, 3))).astype(np.uint8)


def _hard_motion(rng, frames, h, w, c) -> Clip:
    """Learnable-but-hard: a low-contrast block with a small class-dependent vertical offset and heavy
    noise, so classes are decodable above chance yet not trivially mastered (the curriculum sweet
    spot: most remaining room among the decodable candidates)."""
    base = _blank(frames, h, w)
    sz = max(2, w // 10)
    for t in range(frames):
        x = int((t / max(1, frames - 1)) * (w - sz))
        y = h // 2 + (1 if c == 0 else -1) * (h // 8)  # small class-dependent offset
        _put(base, t, y, x, sz, (0.30, 0.30, 0.30))  # low contrast
    base += 0.30 * rng.standard_normal(base.shape).astype(np.float32)  # heavy noise overlay
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


# family registry: name -> (generator, default n_classes)
FAMILIES: dict[str, tuple[Callable, int]] = {
    "moving_object": (_moving_object, 2),
    "object_permanence": (_object_permanence, 2),
    "occlusion_reveal": (_occlusion_reveal, 2),
    "relation_change": (_relation_change, 2),
    "containment": (_containment, 2),
    "noisy_tv": (_noisy_tv, 2),
    "navigation": (_navigation, 2),
    "class_incremental": (_class_incremental, 4),
    "domain_incremental": (_domain_incremental, 2),
    "aleatoric_tv": (_aleatoric_tv, 2),
    "hard_motion": (_hard_motion, 2),
}
DEFAULT_FAMILIES = ("moving_object", "object_permanence", "occlusion_reveal", "noisy_tv", "navigation")


def generate_controls(
    out: str | Path,
    families: list[str] | None = None,
    clips_per_class: int = 3,
    frames: int = 8,
    h: int = 32,
    w: int = 32,
    seed: int = 0,
    budget_gb: float = 2.0,
) -> dict:
    """Write the requested control families as class-foldered .npy corpora under `out`, honoring a
    total byte budget. Returns {families:[{name,root,classes,n_clips,bytes}], total_bytes, budget_bytes,
    truncated, mocked:True}. Deterministic by seed; safe to regenerate. truncated is True if the
    budget stopped generation early (reported, never silently overrun)."""
    out = Path(out)
    fams = list(families or DEFAULT_FAMILIES)
    for f in fams:
        if f not in FAMILIES:
            raise ValueError(f"unknown control family {f!r}; known: {sorted(FAMILIES)}")
    budget_bytes = int(budget_gb * 1e9)
    bytes_per_clip = frames * h * w * 3  # uint8
    written: list[dict] = []
    total = 0
    truncated = False
    for fam in fams:
        gen, n_classes = FAMILIES[fam]
        # STABLE per-family offset: Python's hash() of a str is salted (PYTHONHASHSEED), so it differs
        # across processes and would make corpora non-reproducible; sha256 is stable across runs.
        fam_offset = int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 100_000
        rng = np.random.default_rng(seed + fam_offset)
        fam_root = out / fam
        classes: list[str] = []
        n_clips = 0
        for c in range(n_classes):
            cname = f"class{c}"
            (fam_root / cname).mkdir(parents=True, exist_ok=True)
            classes.append(cname)
            for j in range(clips_per_class):
                if total + bytes_per_clip > budget_bytes:
                    truncated = True
                    break
                clip = gen(rng, frames, h, w, c)
                np.save(fam_root / cname / f"clip{j}.npy", clip)
                total += int(clip.nbytes)
                n_clips += 1
            if truncated:
                break
        written.append(
            {"name": fam, "root": str(fam_root), "classes": classes, "n_clips": n_clips, "bytes": None}
        )
        if truncated:
            break
    # fill per-family byte sizes from disk (authoritative)
    for w_rec in written:
        root = Path(w_rec["root"])
        w_rec["bytes"] = int(sum(p.stat().st_size for p in root.rglob("*.npy"))) if root.exists() else 0
    return {
        "families": written,
        "total_bytes": total,
        "budget_bytes": budget_bytes,
        "truncated": truncated,
        "bytes_per_clip": bytes_per_clip,
        "mocked": True,
        "kind": "structured-synthetic",
    }
