#!/usr/bin/env python
"""AT4 handcrafted-descriptor featurizer (WP-01): HOG + hue histogram + frame-difference flow stats on
the SAME nuisance clips as every encoder cache (clip identity rule: torch regeneration seed for seed via
featurize_programmatic.iter_bound_nuisance_clips, streamed one clip at a time as uint8, RAM <1GB).

This column is the classical-vision reference substrate for AT4: if a perceptual encoder TIES it on an
atlas factor, the tie reads test-too-easy; only when the handcrafted/programmatic columns exceed the
perceptual ones does a tie read substrate-bounded. The random control for this arm (a random read-head
over the descriptors) is applied by the consuming experiment (WP-09), not here.

Queue class: light post-encode (Q1.0b). Guarded against running during the in-flight encode.

Usage: python scripts/featurize_handcrafted.py --out runs/mot/at4_handcrafted_features.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.featurize_programmatic import (  # noqa: E402
    CACHE_ROOT,
    CLIPSET,
    clip_checksum,
    guard_post_encode,
    iter_bound_nuisance_clips,
    read_encode_params,
    to_uint8,
    write_factors_sidecar,
)

from mop.diagnostics import linear_probe  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

HOG_CELLS = 8  # cells per side
HOG_BINS = 9  # unsigned orientation bins over [0, pi)
HUE_BINS = 16
TSUB = 8  # frames featurized per clip
FLOW_STATS = 12  # 6 per-pair stats, mean and std over pairs
DESCRIPTOR_DIM = HOG_CELLS * HOG_CELLS * HOG_BINS + HUE_BINS + FLOW_STATS


def subsample_frames(frames_u8: torch.Tensor, tsub: int = TSUB) -> torch.Tensor:
    """[T,3,H,W] uint8 -> [t<=tsub,3,H,W] uint8, same stride rule as the encoder caches."""
    t = frames_u8.shape[0]
    if t <= tsub:
        return frames_u8
    return frames_u8[:: (t // tsub)][:tsub]


def hog_features(frames_u8: torch.Tensor, cells: int = HOG_CELLS, bins: int = HOG_BINS) -> torch.Tensor:
    """Unsigned-gradient HOG on grayscale frames: per-cell orientation histograms, magnitude weighted,
    L2 normalized per frame, averaged over frames. [T,3,H,W] uint8 -> [cells*cells*bins]."""
    gray = frames_u8.float().mean(dim=1) / 255.0  # [T,H,W]
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, 1:-1] = (gray[:, :, 2:] - gray[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (gray[:, 2:, :] - gray[:, :-2, :]) / 2
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    ang = torch.atan2(gy, gx) % math.pi  # unsigned orientation in [0, pi)
    idx = (ang / math.pi * bins).long().clamp(0, bins - 1)
    h, w = gray.shape[1], gray.shape[2]
    kh, kw = max(1, h // cells), max(1, w // cells)
    per_bin = []
    for b in range(bins):
        m = mag * (idx == b).float()
        per_bin.append(F.avg_pool2d(m.unsqueeze(1), kernel_size=(kh, kw))[:, 0, :cells, :cells])
    hist = torch.stack(per_bin, dim=1).reshape(gray.shape[0], -1)  # [T, bins*cells*cells]
    hist = hist / (hist.norm(dim=1, keepdim=True) + 1e-8)
    return hist.mean(dim=0)


def hue_histogram(frames_u8: torch.Tensor, bins: int = HUE_BINS) -> torch.Tensor:
    """Saturation-weighted hue histogram averaged over frames. [T,3,H,W] uint8 -> [bins], sums to ~1
    per frame when any saturation exists."""
    rgb = frames_u8.float() / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, _ = rgb.max(dim=1)
    mn, _ = rgb.min(dim=1)
    sat = mx - mn  # [T,H,W]
    d = sat + 1e-8
    hr = (((g - b) / d) % 6.0) * (mx == r).float()
    hg = ((b - r) / d + 2.0) * ((mx == g) & (mx != r)).float()
    hb = ((r - g) / d + 4.0) * ((mx == b) & (mx != r) & (mx != g)).float()
    hue = (hr + hg + hb) / 6.0  # [0,1)
    idx = (hue * bins).long().clamp(0, bins - 1)
    hists = []
    for t in range(frames_u8.shape[0]):
        h = torch.zeros(bins).scatter_add_(0, idx[t].reshape(-1), sat[t].reshape(-1))
        hists.append(h / (sat[t].sum() + 1e-8))
    return torch.stack(hists).mean(dim=0)


def flow_stats(frames_u8: torch.Tensor) -> torch.Tensor:
    """Frame-difference motion statistics over consecutive subsampled frames: per pair the diff mean,
    std, max, moving fraction, and diff-mass centroid (x, y), then mean and std across pairs -> [12]."""
    gray = frames_u8.float().mean(dim=1) / 255.0  # [T,H,W]
    if gray.shape[0] < 2:
        return torch.zeros(FLOW_STATS)
    d = (gray[1:] - gray[:-1]).abs()  # [T-1,H,W]
    h, w = d.shape[1], d.shape[2]
    ys = torch.linspace(0, 1, h)[None, :, None]
    xs = torch.linspace(0, 1, w)[None, None, :]
    mass = d.sum(dim=(1, 2)) + 1e-8
    per_pair = torch.stack(
        [
            d.mean(dim=(1, 2)),
            d.std(dim=(1, 2)),
            d.amax(dim=(1, 2)),
            (d > 0.1).float().mean(dim=(1, 2)),
            (d * xs).sum(dim=(1, 2)) / mass,
            (d * ys).sum(dim=(1, 2)) / mass,
        ],
        dim=1,
    )  # [T-1, 6]
    std = per_pair.std(dim=0) if per_pair.shape[0] > 1 else torch.zeros(6)
    return torch.cat([per_pair.mean(dim=0), std])


def handcrafted_descriptor(frames_u8: torch.Tensor) -> torch.Tensor:
    """[T,3,H,W] uint8 (already subsampled) -> [DESCRIPTOR_DIM] float descriptor."""
    return torch.cat([hog_features(frames_u8), hue_histogram(frames_u8), flow_stats(frames_u8)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT4 handcrafted HOG/hue/flow featurizer (clip-identity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/at4_handcrafted_features.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    params = read_encode_params(seed=a.seed)
    n = params["n_shape"] * params["n_color"] * params["per"]

    store = LatentStore.create(
        Path(a.cache_root),
        "handcrafted_descriptors",
        feat_shape=(DESCRIPTOR_DIM,),
        capacity=n,
        key_dim=DESCRIPTOR_DIM,
        has_labels=True,
    )
    shapes, colors, checks = [], [], {}
    t0 = time.perf_counter()
    for i, s, c, clip, _ in iter_bound_nuisance_clips(
        params["n_shape"], params["n_color"], params["per"], params["seed"]
    ):
        if i == 0:
            checks["first"] = clip_checksum(clip)
        if i == n - 1:
            checks["last"] = clip_checksum(clip)
        frames = subsample_frames(to_uint8(clip))
        del clip  # drop the float clip immediately, keep only uint8 (RAM rule)
        z = handcrafted_descriptor(frames)
        store.write_batch(i, z[None].numpy(), z[None].numpy(), torch.tensor([s]).numpy())
        shapes.append(s)
        colors.append(c)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"featurized {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    store.finalize()
    write_factors_sidecar(store.root, params, shapes, colors, checks, source="featurize_handcrafted")

    x = store.latents()
    probe_s = linear_probe(x, torch.tensor(shapes), classification=True, epochs=300, seed=params["seed"])
    probe_c = linear_probe(x, torch.tensor(colors), classification=True, epochs=300, seed=params["seed"])
    out = {
        "experiment": "at4_handcrafted_features",
        "clipset": CLIPSET,
        "params": params,
        "n_clips": n,
        "feature_dim": DESCRIPTOR_DIM,
        "store": str(store.root),
        "clip_checksums": checks,
        "shape_probe_acc": round(probe_s["score"], 4),
        "color_probe_acc": round(probe_c["score"], 4),
        "chance_shape": round(probe_s["chance"], 4),
        "seconds": round(time.perf_counter() - t0, 1),
        "note": "handcrafted HOG + hue histogram + frame-diff flow reference column for AT4; the random "
        "read-head control over these descriptors is applied by the consuming experiment (WP-09)",
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
