#!/usr/bin/env python
"""The CORRECTED substrate test. The corpus control (frozen_random_projection) is a square full-rank
linear map, hence vacuous for probes (any probe absorbs its inverse; real ties frozen-random by
construction). That never tested whether V-JEPA's TRAINED features are special. This does, with the right
control: real V-JEPA features vs RANDOM-PIXEL features (a random projection of the downsampled raw pixels,
i.e. untrained-network-style features), on content with heavy NUISANCE variation.

Content: a shape identity is the class (circle/square/triangle/cross/...), but each clip randomizes
position, scale, rotation, color, background clutter, and motion. Within a shape class the raw pixels vary
wildly, so random-pixel features (no learned invariance) should struggle to decode shape, while V-JEPA
(trained to be invariant to position/scale/appearance) should decode it robustly. If real V-JEPA beats
random-pixel features here, the encoder does real perceptual work (the substrate IS special, and the fork
reopens toward keeping it). If they tie even here, the substrate adds nothing over random features.

This is the honest "does pretraining beat random features" test, on the axis pretraining is supposed to
help: invariance to nuisance. It is NOT ceiling-prone by construction (nuisance makes raw-pixel decoding
hard) and NOT tautological (the two feature spaces are genuinely different, not linear maps of each other).

Usage: python scripts/substrate_vs_random_features.py --n-shape 6 --per 16 \
    --out runs/pre_studio/substrate_vs_random.json   (device=cpu)

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from devsys.config import compose
from devsys.devices import resolve, safe_to
from devsys.diagnostics import linear_probe
from devsys.substrate import load_encoder

FRAMES, RES = 64, 256


def _hue(c: float) -> torch.Tensor:
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * c),
            0.5 + 0.5 * math.cos(2 * math.pi * (c + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (c + 2 / 3)),
        ]
    )


def _shape_mask(shape, cx, cy, r, rot, yy, xx):
    # rotate the coordinate frame by rot around (cx, cy) so asymmetric shapes carry orientation nuisance
    dx0, dy0 = xx - cx, yy - cy
    ca, sa = math.cos(rot), math.sin(rot)
    dx, dy = ca * dx0 - sa * dy0, sa * dx0 + ca * dy0
    if shape == 0:  # circle
        return (dx * dx + dy * dy) <= r * r
    if shape == 1:  # square
        return (dx.abs() <= r) & (dy.abs() <= r)
    if shape == 2:  # triangle
        return (dy <= r) & (dy >= -r) & (dx.abs() <= (r - dy) / 2 + r / 2)
    if shape == 3:  # cross
        return ((dx.abs() <= r) & (dy.abs() <= r / 3)) | ((dy.abs() <= r) & (dx.abs() <= r / 3))
    if shape == 4:  # diamond
        return (dx.abs() + dy.abs()) <= r
    # ring
    d2 = dx * dx + dy * dy
    return (d2 <= r * r) & (d2 >= (0.5 * r) ** 2)


def make_nuisance_clip(shape: int, g: torch.Generator) -> torch.Tensor:
    """A clip whose CLASS is the shape identity, with heavy nuisance: random position, scale, rotation,
    hue, background clutter, and motion. Raw pixels vary wildly within a shape class."""
    lin = torch.linspace(-1, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    r = 0.15 + 0.2 * float(torch.rand(1, generator=g))  # random scale
    rot = float(torch.rand(1, generator=g)) * 2 * math.pi  # random rotation
    hue = _hue(float(torch.rand(1, generator=g)))  # random color
    x0 = -0.5 + float(torch.rand(1, generator=g))  # random start position
    y0 = -0.5 + float(torch.rand(1, generator=g))
    vx = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)  # random motion
    vy = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)
    # background clutter: a few random faint blobs (nuisance texture)
    bg = 0.3 + 0.1 * torch.randn(3, RES, RES, generator=g)
    for _ in range(3):
        bcx, bcy = 2 * float(torch.rand(1, generator=g)) - 1, 2 * float(torch.rand(1, generator=g)) - 1
        br = 0.1 + 0.15 * float(torch.rand(1, generator=g))
        bm = (((xx - bcx) ** 2 + (yy - bcy) ** 2) <= br * br).float()
        bg = bg * (1 - 0.4 * bm)[None]
    frames = []
    for t in range(FRAMES):
        cx, cy = x0 + vx * (t / FRAMES), y0 + vy * (t / FRAMES)
        mask = _shape_mask(shape, cx, cy, r, rot, yy, xx).float()
        frame = bg.clone()
        frame = frame * (1 - mask)[None] + hue[:, None, None] * mask[None]
        frames.append(frame)
    clip = torch.stack(frames) + 0.03 * torch.randn(FRAMES, 3, RES, RES, generator=g)
    return clip.clamp(0, 1)


def random_pixel_features(
    clip: torch.Tensor, proj: torch.Tensor, ds: int = 32, tsub: int = 8
) -> torch.Tensor:
    """Untrained 'random features' baseline: downsample the clip (avg-pool space + subsample time),
    flatten, and apply a FIXED random projection to the V-JEPA embed dim, renormalized. This is what
    random (untrained) features of the raw pixels look like, the honest 'does pretraining help' control."""
    c = clip  # [T,3,H,W]
    c = c[:: (FRAMES // tsub)][:tsub]  # subsample time -> [tsub,3,H,W]
    c = F.avg_pool2d(c, kernel_size=RES // ds)  # -> [tsub,3,ds,ds]
    flat = c.reshape(-1)  # [tsub*3*ds*ds]
    z = flat @ proj  # [D]
    return z / (z.norm() + 1e-6) * math.sqrt(proj.shape[1])


def run(n_shape: int, per: int, seed: int) -> dict:
    cfg = compose(["encoder=vjepa2_vitl_fpc64_256", "device=cpu", "encoder.prefer_real=true"])
    dev = resolve("cpu")
    enc = load_encoder(cfg.encoder).to(dev.device)
    backend = enc.spec.backend
    D = int(cfg.encoder.embed_dim)
    g = torch.Generator().manual_seed(seed)
    ds, tsub = 32, 8
    pin = tsub * 3 * ds * ds
    pg = torch.Generator().manual_seed(seed + 1)
    proj = torch.randn(pin, D, generator=pg) / math.sqrt(pin)  # fixed random-pixel projection

    labels = [s for s in range(n_shape) for _ in range(per)]
    order = labels[:]
    Xv, Xr = [], []
    t0 = time.perf_counter()
    for i, s in enumerate(order):
        clip = make_nuisance_clip(s, g)
        Xv.append(enc.encode(safe_to(clip.unsqueeze(0), dev.device)).reshape(-1).float().cpu())
        Xr.append(random_pixel_features(clip, proj, ds, tsub))
        if (i + 1) % 8 == 0 or i + 1 == len(order):
            print(f"encoded {i + 1}/{len(order)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    xv = torch.stack(Xv)
    xr = torch.stack(Xr)
    y = torch.tensor(labels)

    pv = linear_probe(xv, y, classification=True, epochs=400, seed=seed)
    pr = linear_probe(xr, y, classification=True, epochs=400, seed=seed)
    delta = pv["score"] - pr["score"]
    out = {
        "backend": backend,
        "n_clips": len(order),
        "n_shape": n_shape,
        "per_class": per,
        "chance": round(pv["chance"], 4),
        "seconds": round(time.perf_counter() - t0, 1),
        "vjepa_shape_acc": round(pv["score"], 4),
        "random_pixel_shape_acc": round(pr["score"], 4),
        "vjepa_minus_random_pixel": round(delta, 4),
        "note": "shape identity under heavy nuisance (position/scale/rotation/color/clutter/motion); "
        "random-pixel = random projection of downsampled raw pixels (untrained features)",
    }
    if backend != "vjepa_hf":
        out["verdict"] = "INVALID: encoder ran as frozen_random, rerun with real weights"
    elif delta > 0.15:
        out["verdict"] = (
            "SUBSTRATE IS SPECIAL: real V-JEPA decodes shape under heavy nuisance far better than "
            "random-pixel features, so pretraining buys real invariance. The fork reopens toward keeping "
            "the frozen encoder; the earlier 'not special' read was an artifact of the vacuous "
            "latent-projection control."
        )
    elif delta > 0.05:
        out["verdict"] = (
            "substrate modestly special: V-JEPA beats random-pixel features by a real but small margin"
        )
    elif pv["score"] < out["chance"] + 0.1 and pr["score"] < out["chance"] + 0.1:
        out["verdict"] = (
            "TOO HARD: both near chance, nuisance overwhelmed both feature spaces, ease the nuisance"
        )
    else:
        out["verdict"] = (
            "substrate NOT special even on nuisance-heavy content: random-pixel features decode shape as "
            "well as V-JEPA, so the encoder adds no invariance here. Strong substrate-bounded evidence, "
            "fork points custom."
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="real V-JEPA vs random-pixel features on nuisance-heavy content")
    ap.add_argument("--n-shape", type=int, default=6)
    ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run(a.n_shape, a.per, a.seed)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
