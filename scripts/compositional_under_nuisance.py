#!/usr/bin/env python
"""Compositional binding, finally OFF the ceiling. Every prior compositional probe (dense-vs-pooled,
compositional_binding) ceilinged: 4 shapes x 4 colors on clean backgrounds are trivially linearly
separable in 1024-d, so seen AND held-out combinations both hit 1.0 and real ties frozen-random. The arc
(DOCTRINE_SYNTHESIS 3c) concluded the binding constraint is TEST DIFFICULTY, and the fix is content with
genuinely bound factors under heavy NUISANCE so decoding is not free.

This test binds two labeled factors into one object, SHAPE (factor A) and COLOR (factor B), and buries
them under nuisance (random position, scale, rotation, background clutter, motion) that varies within
every (shape, color) cell. It then runs the held-out-combination gate: train a probe to decode SHAPE on a
diagonal-held-out set of (shape, color) cells, test SHAPE on the unseen cells. A substrate that represents
shape compositionally (factored from color) decodes held-out shapes above chance; one that memorizes
(shape, color) conjunctions collapses to chance on unseen cells.

The control is the CORRECT one (not the vacuous latent projection): random-PIXEL features (a fixed random
projection of the downsampled raw pixels, i.e. untrained-network features). If real V-JEPA extrapolates to
held-out (shape, color) combinations under nuisance AND random-pixel features cannot, the frozen substrate
carries a compositional shape code that raw pixels do not. If they tie, compositionality here is generic
pixel geometry, not a substrate gift.

Usage: python scripts/compositional_under_nuisance.py --n-shape 5 --n-color 5 --per 8 \
    --out runs/pre_studio/compositional_under_nuisance.json   (device=cpu)

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import torch
from scripts.substrate_vs_random_features import FRAMES, RES, _hue, _shape_mask, random_pixel_features

from devsys.config import compose
from devsys.devices import resolve, safe_to
from devsys.diagnostics.held_out_combo import held_out_combination
from devsys.substrate import load_encoder


def make_bound_nuisance_clip(shape: int, color: int, n_color: int, g: torch.Generator) -> torch.Tensor:
    """A clip whose two LABELED factors are shape identity and color hue, bound into one moving object,
    with heavy nuisance (random position, scale, rotation, clutter, motion) varying within every cell."""
    lin = torch.linspace(-1, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    r = 0.15 + 0.2 * float(torch.rand(1, generator=g))  # nuisance scale
    rot = float(torch.rand(1, generator=g)) * 2 * math.pi  # nuisance rotation
    hue = _hue(color / max(1, n_color))  # LABELED color
    x0 = -0.5 + float(torch.rand(1, generator=g))  # nuisance start position
    y0 = -0.5 + float(torch.rand(1, generator=g))
    vx = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)  # nuisance motion
    vy = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)
    bg = 0.3 + 0.1 * torch.randn(3, RES, RES, generator=g)  # nuisance clutter
    for _ in range(3):
        bcx, bcy = 2 * float(torch.rand(1, generator=g)) - 1, 2 * float(torch.rand(1, generator=g)) - 1
        br = 0.1 + 0.15 * float(torch.rand(1, generator=g))
        bm = (((xx - bcx) ** 2 + (yy - bcy) ** 2) <= br * br).float()
        bg = bg * (1 - 0.4 * bm)[None]
    frames = []
    for t in range(FRAMES):
        cx, cy = x0 + vx * (t / FRAMES), y0 + vy * (t / FRAMES)
        mask = _shape_mask(shape, cx, cy, r, rot, yy, xx).float()  # LABELED shape
        frame = bg.clone()
        frame = frame * (1 - mask)[None] + hue[:, None, None] * mask[None]
        frames.append(frame)
    clip = torch.stack(frames) + 0.03 * torch.randn(FRAMES, 3, RES, RES, generator=g)
    return clip.clamp(0, 1)


def run(n_shape: int, n_color: int, per: int, seed: int) -> dict:
    cfg = compose(["encoder=vjepa2_vitl_fpc64_256", "device=cpu", "encoder.prefer_real=true"])
    dev = resolve("cpu")
    enc = load_encoder(cfg.encoder).to(dev.device)
    backend = enc.spec.backend
    D = int(cfg.encoder.embed_dim)
    g = torch.Generator().manual_seed(seed)
    ds, tsub = 32, 8
    pin = tsub * 3 * ds * ds
    pg = torch.Generator().manual_seed(seed + 1)
    proj = torch.randn(pin, D, generator=pg) / (pin**0.5)

    cells = [(s, c) for s in range(n_shape) for c in range(n_color) for _ in range(per)]
    Xv, Xr, ys, yc = [], [], [], []
    t0 = time.perf_counter()
    for i, (s, c) in enumerate(cells):
        clip = make_bound_nuisance_clip(s, c, n_color, g)
        Xv.append(enc.encode(safe_to(clip.unsqueeze(0), dev.device)).reshape(-1).float().cpu())
        Xr.append(random_pixel_features(clip, proj, ds, tsub))
        ys.append(s)
        yc.append(c)
        if (i + 1) % 8 == 0 or i + 1 == len(cells):
            print(f"encoded {i + 1}/{len(cells)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    xv = torch.stack(Xv)
    xr = torch.stack(Xr)
    y_shape = torch.tensor(ys)
    y_color = torch.tensor(yc)

    hoc_v = held_out_combination(xv, y_shape, y_color, seed=seed)
    hoc_r = held_out_combination(xr, y_shape, y_color, seed=seed)
    v_held = hoc_v.get("heldout_acc", 0.0)
    r_held = hoc_r.get("heldout_acc", 0.0)
    ch = hoc_v.get("chance", 1.0 / n_shape)
    delta = round(v_held - r_held, 4)
    out = {
        "backend": backend,
        "n_clips": len(cells),
        "n_shape": n_shape,
        "n_color": n_color,
        "per_cell": per,
        "seconds": round(time.perf_counter() - t0, 1),
        "chance": ch,
        "vjepa": hoc_v,
        "random_pixel": hoc_r,
        "heldout_shape_delta_vjepa_minus_random_pixel": delta,
        "note": "held-out (shape,color) combinations under heavy nuisance; control is random-pixel features",
    }
    if backend != "vjepa_hf":
        out["verdict"] = "INVALID: encoder ran as frozen_random, rerun with real weights"
    elif hoc_v.get("seen_acc", 0.0) > 0.97 and v_held > 0.97 and hoc_r.get("seen_acc", 0.0) > 0.97:
        out["verdict"] = "CEILING again: nuisance did not bite, both feature spaces still trivially separable"
    elif v_held > ch + 0.15 and delta > 0.1:
        out["verdict"] = (
            "COMPOSITIONAL AND SUBSTRATE-SPECIFIC: real V-JEPA extrapolates SHAPE to held-out "
            "(shape, color) combinations under nuisance where random-pixel features cannot. The frozen "
            "substrate carries a compositional shape code that raw pixels do not. First non-ceiling "
            "compositional positive in the program."
        )
    elif v_held > ch + 0.15:
        out["verdict"] = (
            "compositional but NOT substrate-specific: V-JEPA extrapolates to held-out combinations, but "
            "random-pixel features do about as well, so it is generic pixel geometry, not a substrate gift"
        )
    elif v_held <= ch + 0.1:
        out["verdict"] = (
            "NON-COMPOSITIONAL / MEMORIZED: real V-JEPA held-out shape collapses to chance, the pooled "
            "substrate binds shape and color into conjunctions it cannot factor under nuisance (taxonomy 3, "
            "ship the bound, retarget dense 2.1)"
        )
    else:
        out["verdict"] = "ambiguous"
    return out


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(
        description="compositional held-out (shape,color) under nuisance, real vs random-pixel"
    )
    ap.add_argument("--n-shape", type=int, default=5)
    ap.add_argument("--n-color", type=int, default=5)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run(a.n_shape, a.n_color, a.per, a.seed)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
