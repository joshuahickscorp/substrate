#!/usr/bin/env python
"""The compositional-binding probe: the one synthetic test that can actually bite (per the phase-transition
finding, additive/independent factors ceiling; only genuinely BOUND factors that still SHARE structure
across cells make held-out-combination informative). Gratings failed because hue and orientation are
independent global properties. Here the two factors are bound into a single OBJECT: a moving colored SHAPE
(factor A = shape: circle/square/triangle/cross; factor B = color hue). Shape and color are bound in the
same object, but structure is shared across cells (every square shares squareness, every red shares
redness), so compositional extrapolation to held-out (shape, color) combinations is possible IN PRINCIPLE:
a model that represents shape and color compositionally can decode "red square" having seen "blue square"
and "red circle", while a model that only memorizes conjunctions cannot.

The test asks, on REAL V-JEPA pooled geometry: train a probe to decode SHAPE while holding out a diagonal
set of (shape, color) cells, then test on the unseen combinations, vs a frozen-random control. This is the
first compositional test in the program that is neither ceiling-trivial (factors are bound, not additive)
nor chance-by-construction (structure is shared, not arbitrary per cell).

Usage: python scripts/compositional_binding_probe.py --n-shape 4 --n-color 4 --per 6 \
    --out runs/pre_studio/compositional_binding.json    (device=cpu; MPS overflows the ViT-L buffer)

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from mop.config import compose
from mop.devices import resolve, safe_to
from mop.diagnostics import linear_probe
from mop.diagnostics.held_out_combo import held_out_combination
from mop.diagnostics.substrate_ablation import frozen_random_projection
from mop.substrate import load_encoder

FRAMES, RES = 64, 256


def _hue(c: int, n: int) -> torch.Tensor:
    h = c / max(1, n)
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * h),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 2 / 3)),
        ]
    )


def _shape_mask(
    shape: int, cx: float, cy: float, r: float, yy: torch.Tensor, xx: torch.Tensor
) -> torch.Tensor:
    """A filled shape mask centered at (cx, cy), radius r. shape: 0 circle, 1 square, 2 triangle, 3 cross.
    Shape is a spatial-form factor (needs the arrangement), color is applied to the SAME mask, so the two
    are bound into one object but share form/color structure across cells."""
    dx, dy = xx - cx, yy - cy
    if shape == 0:  # circle
        return (dx * dx + dy * dy) <= r * r
    if shape == 1:  # square
        return (dx.abs() <= r) & (dy.abs() <= r)
    if shape == 2:  # triangle (pointing up)
        return (dy <= r) & (dy >= -r) & (dx.abs() <= (r - dy) / 2 + r / 2)
    # cross
    return ((dx.abs() <= r) & (dy.abs() <= r / 3)) | ((dy.abs() <= r) & (dx.abs() <= r / 3))


def make_object_clip(shape: int, color: int, n_shape: int, n_color: int, g: torch.Generator) -> torch.Tensor:
    """A [T,3,RES,RES] clip: a colored shape drifting across the frame (motion for V-JEPA). Shape sets the
    form, color sets the hue, both realized on the SAME object (bound), with a moving position so the
    encoder sees temporal structure. Background is a fixed neutral gray."""
    lin = torch.linspace(-1, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    tint = _hue(color, n_color)
    r = 0.28
    x0 = -0.4 + 0.05 * float(torch.randn(1, generator=g))
    y0 = 0.0 + 0.1 * float(torch.randn(1, generator=g))
    frames = []
    for t in range(FRAMES):
        cx = x0 + 0.8 * (t / FRAMES)  # drift left to right
        cy = y0 + 0.15 * math.sin(2 * math.pi * t / FRAMES)
        mask = _shape_mask(shape, cx, cy, r, yy, xx).float()  # [RES,RES]
        frame = 0.4 * torch.ones(3, RES, RES)  # gray background
        frame = frame * (1 - mask)[None] + tint[:, None, None] * mask[None]
        frames.append(frame)
    clip = torch.stack(frames)  # [T,3,RES,RES]
    clip = clip + 0.03 * torch.randn(clip.shape, generator=g)
    return clip.clamp(0, 1)


def run(n_shape: int, n_color: int, per: int, seed: int) -> dict:
    cfg = compose(
        [
            "encoder=vjepa2_vitl_fpc64_256",
            "device=cpu",
            "encoder.prefer_real=true",
            "+encoder.require_real=true",
        ]
    )
    dev = resolve("cpu")
    enc = load_encoder(cfg.encoder).to(dev.device)
    backend = enc.spec.backend
    g = torch.Generator().manual_seed(seed)

    cells = [(s, c) for s in range(n_shape) for c in range(n_color) for _ in range(per)]
    X, ys, yc = [], [], []
    t0 = time.perf_counter()
    for i, (s, c) in enumerate(cells):
        clip = make_object_clip(s, c, n_shape, n_color, g).unsqueeze(0)
        z = enc.encode(safe_to(clip, dev.device)).reshape(-1).float().cpu()  # pooled [D]
        X.append(z)
        ys.append(s)
        yc.append(c)
        if (i + 1) % 8 == 0 or i + 1 == len(cells):
            print(f"encoded {i + 1}/{len(cells)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    x = torch.stack(X)
    y_shape = torch.tensor(ys)
    y_color = torch.tensor(yc)

    def probe_pair(xx, y):
        r = linear_probe(xx, y, classification=True, epochs=300, seed=seed)
        fr = linear_probe(frozen_random_projection(xx, seed), y, classification=True, epochs=300, seed=seed)
        return {
            "real": round(r["score"], 4),
            "frozen_random": round(fr["score"], 4),
            "chance": round(r["chance"], 4),
            "delta": round(r["score"] - fr["score"], 4),
        }

    # single-factor decodability (sanity: are shape and color each decodable at all)
    shape_dec = probe_pair(x, y_shape)
    color_dec = probe_pair(x, y_color)
    # THE compositional test: held-out (shape, color) combinations, real vs frozen-random
    hoc_real = held_out_combination(x, y_shape, y_color, seed=seed)
    hoc_fr = held_out_combination(frozen_random_projection(x, seed), y_shape, y_color, seed=seed)

    heldout_delta = round(hoc_real["heldout_acc"] - hoc_fr["heldout_acc"], 4)
    seen = hoc_real["seen_acc"]
    held = hoc_real["heldout_acc"]
    ch = hoc_real["chance"]
    out = {
        "backend": backend,
        "n_clips": len(cells),
        "n_shape": n_shape,
        "n_color": n_color,
        "per_cell": per,
        "seconds": round(time.perf_counter() - t0, 1),
        "shape_decodable": shape_dec,
        "color_decodable": color_dec,
        "held_out_combination": {
            "real": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hoc_real.items()},
            "frozen_random": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hoc_fr.items()},
            "heldout_delta_real_minus_fr": heldout_delta,
        },
    }
    if backend != "vjepa_hf":
        out["verdict"] = "INVALID: encoder ran as frozen_random, rerun with real weights"
    elif seen > 0.95 and held > 0.95:
        out["verdict"] = (
            "CEILING: both seen and held-out combinations decode near-perfectly, the shapes "
            "are still too easily separable, need harder content or more classes"
        )
    elif held <= ch + 0.1:
        out["verdict"] = (
            "NON-COMPOSITIONAL / MEMORIZED: seen combinations decode but held-out combinations "
            "collapse to chance, V-JEPA does not represent shape and color compositionally here"
        )
    elif held > ch + 0.1 and heldout_delta > 0.05:
        out["verdict"] = (
            "COMPOSITIONAL AND SUBSTRATE-SPECIFIC: held-out combinations decode above chance AND "
            "above the frozen-random floor, real evidence the substrate factorizes shape and color"
        )
    elif held > ch + 0.1:
        out["verdict"] = (
            "compositional but projection-invariant: held-out combinations decode above chance, "
            "but a frozen-random projection does it equally, so it is generic linear geometry"
        )
    else:
        out["verdict"] = "ambiguous"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="compositional-binding probe on real V-JEPA (colored shapes)")
    ap.add_argument("--n-shape", type=int, default=4)
    ap.add_argument("--n-color", type=int, default=4)
    ap.add_argument("--per", type=int, default=6)
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
