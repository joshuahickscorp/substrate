#!/usr/bin/env python
"""AXIS: abstraction-across-perspectives, pushed to its laptop ceiling.

Build a RICHER synthetic clipset than the shape+color-only instrument by adding NEW attribute SLOTS:
  shape (6), color (6), count (1..3 objects), relation (primary object left/right of the other objects,
  defined only for count>=2), size (small/large primary object).
All buried under the usual nuisance (random position, scale, per-object rotation, background clutter,
motion). Then encode each clip with TWO perspectives and probe every slot from both:
  IMAGE : facebook/dinov2-small, middle frame, mean-pooled patch tokens -> 384d.
  TEXT  : LABEL-FREE pixel-derived textification -> Qwen2.5-0.5B mid-layer mean -> 896d.

LABEL-FREE RULE (hard): the text perspective reads ONLY pixels (a coarse palette-quantized grid + a count
of distinct bright blobs + their horizontal ordering + a coarse size estimate). It NEVER sees any slot
label, so decodability in the text substrate was carried through the rendering, not injected. This keeps a
random-init control valid.

PREREGISTERED (in code, below): a slot is 'instrumented' from a perspective iff its linear-probe accuracy
beats chance with a seed-CI lower bound (over N_SEEDS probe seeds) strictly above chance + MARGIN. A TIE
(CI lo <= chance + MARGIN) is a NULL. We do not fake scores.

Run:
  OMP_NUM_THREADS=4 .venv/bin/python scripts/mop_abstraction_richer_slots.py

No em dashes or en dashes (house rule).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from mop.diagnostics.riskcov import seed_ci

# ------------------------------------------------------------------ preregistered constants
OUT = Path(__file__).resolve().parents[1] / "runs" / "mot"
OUT.mkdir(parents=True, exist_ok=True)
DINO_ID = "facebook/dinov2-small"
QWEN_ID = "Qwen/Qwen2.5-0.5B"

FRAMES, RES = 16, 224  # fewer frames + smaller canvas than the vjepa clipset: dinov2 uses one frame only
N_SHAPE, N_COLOR = 6, 6
COUNTS = (1, 2, 3)
SIZES = (0, 1)  # 0=small, 1=large primary object
N_PER = 4  # replicates per (shape,color,count,size) would be huge; we SAMPLE instead (see build_clipset)
N_CLIPS = 112  # modest N target (aim 96-128); actual set is sampled to balance every slot
SEED = 0
N_PROBE_SEEDS = 5
MARGIN = 0.03  # CI lo must clear chance by this to count as instrumented (a tie is a null)


# ------------------------------------------------------------------ render primitives (reused shapes)
def _hue(c: float) -> torch.Tensor:
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * c),
            0.5 + 0.5 * math.cos(2 * math.pi * (c + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (c + 2 / 3)),
        ]
    )


def _shape_mask(shape, cx, cy, r, rot, yy, xx):
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
    d2 = dx * dx + dy * dy  # ring
    return (d2 <= r * r) & (d2 >= (0.5 * r) ** 2)


def make_rich_clip(shape, color, count, size, g) -> tuple[torch.Tensor, int]:
    """One clip with LABELED slots (shape,color,count,size) bound into the PRIMARY object, plus (count-1)
    filler objects (each a random shape/color) to realize the object count, all under nuisance. Returns
    the clip [FRAMES,3,RES,RES] and the derived relation label (0=primary left of fillers' mean-x,
    1=primary right; -1 if count==1)."""
    lin = torch.linspace(-1, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")

    # primary object: LABELED size drives its radius band; nuisance jitter within the band
    r_small, r_large = 0.10, 0.20
    base_r = (r_small if size == 0 else r_large) + 0.03 * float(torch.rand(1, generator=g))
    hue = _hue(color / N_COLOR)  # LABELED color
    px0 = -0.5 + float(torch.rand(1, generator=g))  # nuisance start position
    py0 = -0.5 + float(torch.rand(1, generator=g))
    pvx = 0.3 * (float(torch.rand(1, generator=g)) - 0.5)  # nuisance motion
    pvy = 0.3 * (float(torch.rand(1, generator=g)) - 0.5)
    prot = float(torch.rand(1, generator=g)) * 2 * math.pi  # nuisance rotation

    # filler objects (count-1 of them): random shape/color/pos, so COUNT is a real slot but shape/color
    # of fillers is nuisance. Their radius is drawn independent of the primary's size label.
    fillers = []
    for _ in range(count - 1):
        fshape = int(torch.randint(0, N_SHAPE, (1,), generator=g))
        fcol = float(torch.rand(1, generator=g))
        fr = 0.08 + 0.10 * float(torch.rand(1, generator=g))
        fx0 = -0.6 + 1.2 * float(torch.rand(1, generator=g))
        fy0 = -0.6 + 1.2 * float(torch.rand(1, generator=g))
        fvx = 0.3 * (float(torch.rand(1, generator=g)) - 0.5)
        fvy = 0.3 * (float(torch.rand(1, generator=g)) - 0.5)
        frot = float(torch.rand(1, generator=g)) * 2 * math.pi
        fillers.append((fshape, _hue(fcol), fr, fx0, fy0, fvx, fvy, frot))

    # relation label: primary's x0 vs mean filler x0 (a scene-geometry fact, not injected into pixels)
    relation = -1
    if count >= 2:
        mean_fx = sum(f[3] for f in fillers) / len(fillers)
        relation = 0 if px0 < mean_fx else 1

    bg = 0.3 + 0.08 * torch.randn(3, RES, RES, generator=g)  # nuisance clutter
    for _ in range(3):
        bcx, bcy = 2 * float(torch.rand(1, generator=g)) - 1, 2 * float(torch.rand(1, generator=g)) - 1
        br = 0.1 + 0.12 * float(torch.rand(1, generator=g))
        bm = (((xx - bcx) ** 2 + (yy - bcy) ** 2) <= br * br).float()
        bg = bg * (1 - 0.35 * bm)[None]

    frames = []
    for t in range(FRAMES):
        frame = bg.clone()
        # fillers first (drawn under the primary so the primary stays identifiable)
        for fshape, fhue, fr, fx0, fy0, fvx, fvy, frot in fillers:
            fcx, fcy = fx0 + fvx * (t / FRAMES), fy0 + fvy * (t / FRAMES)
            fm = _shape_mask(fshape, fcx, fcy, fr, frot, yy, xx).float()
            frame = frame * (1 - fm)[None] + fhue[:, None, None] * fm[None]
        pcx, pcy = px0 + pvx * (t / FRAMES), py0 + pvy * (t / FRAMES)
        pm = _shape_mask(shape, pcx, pcy, base_r, prot, yy, xx).float()  # LABELED shape+size
        frame = frame * (1 - pm)[None] + hue[:, None, None] * pm[None]
        frames.append(frame)
    clip = torch.stack(frames) + 0.03 * torch.randn(FRAMES, 3, RES, RES, generator=g)
    return clip.clamp(0, 1), relation


def build_clipset(seed: int):
    """Sample N_CLIPS clips with balanced-ish marginals over every slot. We enumerate a shuffled product
    and truncate, so each slot's classes stay roughly uniform (verified by the printed marginals)."""
    g = torch.Generator().manual_seed(seed)
    combos = [(s, c, n, z) for s in range(N_SHAPE) for c in range(N_COLOR) for n in COUNTS for z in SIZES]
    idx = torch.randperm(len(combos), generator=g).tolist()
    combos = [combos[i] for i in idx][:N_CLIPS]
    specs = []
    for k, (s, c, n, z) in enumerate(combos):
        cg = torch.Generator().manual_seed(seed * 100003 + k)  # per-clip generator, reproducible
        clip, rel = make_rich_clip(s, c, n, z, cg)
        specs.append((clip, dict(shape=s, color=c, count=n, size=z, relation=rel)))
    return specs


# ------------------------------------------------------------------ IMAGE perspective (dinov2-small)
def encode_image(specs):
    # torchvision is absent, so we replicate BitImageProcessor manually: our frames are already RES=224
    # (== dinov2 crop size), so resize/center-crop are no-ops; we only apply ImageNet normalization.
    from transformers import AutoModel

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model = AutoModel.from_pretrained(DINO_ID, local_files_only=True, dtype=torch.float32).eval()
    feats = []
    t0 = time.perf_counter()
    for i, (clip, _) in enumerate(specs):
        mid = clip[FRAMES // 2][None]  # [1,3,RES,RES] in [0,1]
        inp = (mid - mean) / std
        with torch.no_grad():
            out = model(pixel_values=inp)
        # mean-pool patch tokens (drop CLS at index 0)
        z = out.last_hidden_state[0, 1:, :].mean(dim=0).float()
        feats.append(z)
        if (i + 1) % 16 == 0 or i + 1 == len(specs):
            print(f"[image] {i + 1}/{len(specs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    del model
    return torch.stack(feats)


# ------------------------------------------------------------------ TEXT perspective (label-free)
PALETTE = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.45, 0.45, 0.45),
    "red": (0.9, 0.15, 0.15),
    "green": (0.15, 0.8, 0.2),
    "blue": (0.15, 0.25, 0.9),
    "yellow": (0.9, 0.85, 0.15),
    "magenta": (0.85, 0.2, 0.8),
    "cyan": (0.2, 0.8, 0.85),
}


def _blob_stats(frame):
    """LABEL-FREE pixel analysis of one frame: find bright/saturated foreground blobs on the dark-ish bg by
    a connected-component sweep on a coarse grid. Per blob it returns (x-centroid, area, nearest-palette
    colour name of the blob's mean pixel). Returns (n_blobs, per-blob list ordered left->right, total fg
    area fraction). Pure pixels in, numbers/names out; no slot label is consulted."""
    g = 28  # coarse grid
    names = list(PALETTE)
    pal = torch.tensor([PALETTE[k] for k in names])
    h, w = frame.shape[1], frame.shape[2]
    cells = frame[:, : (h // g) * g, : (w // g) * g]
    cells = cells.reshape(3, g, h // g, g, w // g).mean(dim=(2, 4))  # [3,g,g]
    lum = cells.mean(dim=0)  # [g,g]
    sat = cells.max(dim=0).values - cells.min(dim=0).values  # colourful foreground pops in saturation
    fg = ((sat > 0.20) | (lum > 0.75)).int()  # foreground = saturated or very bright
    rgb = cells.permute(1, 2, 0)  # [g,g,3] for per-cell colour lookup
    lab = torch.full((g, g), -1, dtype=torch.long)
    comps = []
    nxt = 0
    for i in range(g):
        for j in range(g):
            if fg[i, j] == 1 and lab[i, j] == -1:
                stack = [(i, j)]
                lab[i, j] = nxt
                xs, cols, area = [], [], 0
                while stack:
                    a, b = stack.pop()
                    xs.append(b)
                    cols.append(rgb[a, b])
                    area += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < g and 0 <= nb < g and fg[na, nb] == 1 and lab[na, nb] == -1:
                            lab[na, nb] = nxt
                            stack.append((na, nb))
                if area >= 2:  # drop 1-cell speckle
                    mean_rgb = torch.stack(cols).mean(0)
                    cname = names[int(((mean_rgb[None] - pal) ** 2).sum(-1).argmin())]
                    comps.append((sum(xs) / len(xs) / g, area, cname))
                nxt += 1
    comps.sort(key=lambda t: t[0])  # order left->right
    fg_frac = round(float(fg.float().mean()), 3)
    return len(comps), comps, fg_frac


def textify_clip(clip):
    """LABEL-FREE prose from pixels: for a few subsampled frames, a coarse palette grid plus derived blob
    count / left-right x-ordering / foreground-area (proxies for count, relation, size). No labels."""
    tsub, grid = 4, 4
    t = clip.shape[0]
    frames = clip[:: max(1, t // tsub)][:tsub]  # [tsub,3,RES,RES]
    names = list(PALETTE)
    pal = torch.tensor([PALETTE[k] for k in names])
    lines = []
    for f in range(frames.shape[0]):
        fr = frames[f]
        h, w = fr.shape[1], fr.shape[2]
        cells = fr[:, : (h // grid) * grid, : (w // grid) * grid]
        cells = cells.reshape(3, grid, h // grid, grid, w // grid).mean(dim=(2, 4))
        flat = cells.permute(1, 2, 0).reshape(-1, 3)
        idx = ((flat[:, None, :] - pal[None]) ** 2).sum(-1).argmin(dim=1)
        rows = [" ".join(names[int(idx[r * grid + c])] for c in range(grid)) for r in range(grid)]
        nb, comps, ff = _blob_stats(fr)
        # per-blob descriptor: colour, x-position, coarse size bucket (all pixel-derived, label-free)
        descs = []
        for x, area, cname in comps:
            szbucket = "big" if area >= 12 else ("mid" if area >= 5 else "small")
            descs.append(f"{cname}@x{x:.2f}/{szbucket}")
        order = " ".join(descs) if descs else "none"
        lines.append(
            f"frame {f}: "
            + " | ".join(rows)
            + f" ; blobs {nb} ; objects left-to-right {order} ; foreground {ff}"
        )
    return "scene report.\n" + "\n".join(lines)


def encode_text(specs):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_ID, local_files_only=True)
    model = AutoModel.from_pretrained(QWEN_ID, local_files_only=True, dtype=torch.float32).eval()
    feats = []
    t0 = time.perf_counter()
    for i, (clip, _) in enumerate(specs):
        text = textify_clip(clip)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        z = hs[len(hs) // 2][0].mean(dim=0).float()  # mid-layer mean over tokens
        feats.append(z)
        if (i + 1) % 16 == 0 or i + 1 == len(specs):
            print(f"[text] {i + 1}/{len(specs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    del model
    return torch.stack(feats)


# ------------------------------------------------------------------ probing (seed-CI vs chance)
def probe_slot(x, y, mask=None):
    """Linear probe with N_PROBE_SEEDS train/test splits; returns seed-CI dict + chance. Optional mask
    selects a subset of clips (for relation, which is only defined when count>=2)."""
    from mop.diagnostics.linear_probe import linear_probe as lp

    if mask is not None:
        x, y = x[mask], y[mask]
    # remap labels to contiguous 0..K-1 so chance = 1/K is honest for the present classes
    uniq = sorted(set(int(v) for v in y.tolist()))
    remap = {u: k for k, u in enumerate(uniq)}
    yy = torch.tensor([remap[int(v)] for v in y.tolist()])
    n_classes = len(uniq)
    if n_classes < 2 or x.shape[0] < 12:
        return None
    accs = []
    for s in range(N_PROBE_SEEDS):
        r = lp(x, yy, classification=True, seed=s, test_frac=0.3)
        accs.append(r["score"])
    ci = seed_ci(accs)
    chance = 1.0 / n_classes
    return {
        "n": int(x.shape[0]),
        "n_classes": n_classes,
        "chance": round(chance, 4),
        "acc_mean": ci["mean"],
        "acc_sd": ci["sd"],
        "acc_lo": ci["lo"],
        "acc_hi": ci["hi"],
        "instrumented": bool(ci["lo"] > chance + MARGIN),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("building clipset...", flush=True)
    specs = build_clipset(SEED)
    labels = {
        k: torch.tensor([sp[1][k] for sp in specs]) for k in ("shape", "color", "count", "size", "relation")
    }

    # marginals sanity
    marg = {}
    for k, v in labels.items():
        vals = [int(x) for x in v.tolist()]
        marg[k] = {str(u): vals.count(u) for u in sorted(set(vals))}
    print("marginals:", json.dumps(marg), flush=True)

    print(f"encoding IMAGE ({DINO_ID})...", flush=True)
    x_img = encode_image(specs)
    print(f"encoding TEXT ({QWEN_ID}, label-free)...", flush=True)
    x_txt = encode_text(specs)
    print(f"features: image {tuple(x_img.shape)}, text {tuple(x_txt.shape)}", flush=True)

    # standardize features (probe-friendly)
    def z(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-6)

    xi, xt = z(x_img), z(x_txt)

    rel_mask = labels["relation"] >= 0  # relation only defined for count>=2
    results = {}
    for slot in ("shape", "color", "count", "size", "relation"):
        m = rel_mask if slot == "relation" else None
        results[slot] = {
            "image": probe_slot(xi, labels[slot], m),
            "text": probe_slot(xt, labels[slot], m),
        }

    # save features + labels + nuisance-free slot record
    torch.save(
        {
            "x_image": x_img,
            "x_text": x_txt,
            "labels": {k: v for k, v in labels.items()},
        },
        OUT / "richer_slots_features.pt",
    )

    out = {
        "n_clips": len(specs),
        "frames": FRAMES,
        "res": RES,
        "image_dim": int(x_img.shape[1]),
        "text_dim": int(x_txt.shape[1]),
        "image_encoder": DINO_ID,
        "text_encoder": QWEN_ID,
        "n_probe_seeds": N_PROBE_SEEDS,
        "margin": MARGIN,
        "prereg": "slot instrumented iff probe seed-CI lo > chance + margin (a tie is a null)",
        "marginals": marg,
        "seconds_total": round(time.perf_counter() - t0, 1),
        "slots": results,
    }
    (OUT / "richer_slots.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
