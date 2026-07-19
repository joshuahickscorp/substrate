#!/usr/bin/env python

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_qwen_textified import PALETTE, resolve_model_id  # noqa: E402
from scripts.cache_randominit_vitl_features import assert_encoder_lane_free  # noqa: E402
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

STORE_PREFIX = "qwen05b_shapecap"
TSUB, GRID = 8, 4
FG_THRESHOLD = 0.28  # foreground segmentation threshold on max(color-distance-to-bg, colorfulness)
CC_ITERS = 128  # label-propagation sweeps for the largest-connected-component pass
KILLSWITCH_SHAPE_ACC = 0.30  # <= this on the real arm means the caption failed to carry shape


def _largest_cc(mask: torch.Tensor) -> torch.Tensor:
    h, w = mask.shape
    idx = torch.arange(h * w).reshape(h, w)
    big = h * w + 1
    lab = torch.where(mask, idx, torch.full_like(idx, -1))
    for _ in range(CC_ITERS):
        up = torch.roll(lab, 1, 0)
        up[0, :] = -1
        dn = torch.roll(lab, -1, 0)
        dn[-1, :] = -1
        lf = torch.roll(lab, 1, 1)
        lf[:, 0] = -1
        rt = torch.roll(lab, -1, 1)
        rt[:, -1] = -1
        stack = torch.stack([lab, up, dn, lf, rt])
        stack = torch.where(stack < 0, torch.full_like(stack, big), stack)
        new = stack.min(dim=0).values
        new = torch.where(mask, new, torch.full_like(new, -1))
        if torch.equal(new, lab):
            break
        lab = new
    labs = lab[mask]
    if labs.numel() == 0:
        return mask
    vals, counts = torch.unique(labs, return_counts=True)
    return lab == vals[counts.argmax()]


def _foreground_mask(frame_u8: torch.Tensor) -> torch.Tensor:
    f = frame_u8.float() / 255.0
    corners = torch.stack([f[:, 0, 0], f[:, 0, -1], f[:, -1, 0], f[:, -1, -1]])  # [4,3]
    bg = corners.mean(dim=0)
    diff = (f - bg[:, None, None]).pow(2).sum(dim=0).sqrt()  # color distance to bg
    sat = f.max(dim=0).values - f.min(dim=0).values  # colorfulness
    return _largest_cc(torch.maximum(diff, sat) > FG_THRESHOLD)


def _shape_descriptors(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if xs.numel() < 8:
        return torch.zeros(13)
    x = xs.float()
    y = ys.float()
    xb, yb = x.mean(), y.mean()
    area = float(mask.sum())

    def mu(p: int, q: int) -> torch.Tensor:
        return (((x - xb) ** p) * ((y - yb) ** q)).sum()

    m00 = area

    def eta(p: int, q: int) -> torch.Tensor:
        return mu(p, q) / (m00 ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    h = torch.zeros(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11**2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n21 - n03) * (
        n21 + n03
    ) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - (n30 - 3 * n12) * (
        n21 + n03
    ) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    hlog = torch.sign(h) * torch.log10(h.abs() + 1e-30)

    m = mask.float()
    nb = torch.zeros_like(m)
    nb[1:, :] += m[:-1, :]
    nb[:-1, :] += m[1:, :]
    nb[:, 1:] += m[:, :-1]
    nb[:, :-1] += m[:, 1:]
    perim = float(((m > 0) & (nb < 4)).sum())
    circularity = 4.0 * math.pi * area / (perim**2 + 1e-9)

    cov = torch.tensor(
        [
            [float(mu(2, 0) / m00), float(mu(1, 1) / m00)],
            [float(mu(1, 1) / m00), float(mu(0, 2) / m00)],
        ]
    )
    ev = torch.linalg.eigvalsh(cov)
    ecc = float(ev.max() / (ev.min() + 1e-6))

    hgt = float(ys.max() - ys.min() + 1)
    wid = float(xs.max() - xs.min() + 1)
    extent = area / (hgt * wid + 1e-9)
    log_area = math.log10(area + 1.0)
    aspect_log = math.log10((wid + 1.0) / (hgt + 1.0))
    return torch.cat(
        [hlog, torch.tensor([circularity, ecc, extent, log_area, aspect_log], dtype=torch.float32)]
    )


def _color_grid_lines(frames: torch.Tensor, grid: int = GRID) -> list[str]:
    names = list(PALETTE)
    pal = torch.tensor([PALETTE[k] for k in names])
    lines = []
    for f in range(frames.shape[0]):
        h, w = frames.shape[2], frames.shape[3]
        cells = frames[f, :, : (h // grid) * grid, : (w // grid) * grid]
        cells = cells.reshape(3, grid, h // grid, grid, w // grid).mean(dim=(2, 4))
        flat = cells.permute(1, 2, 0).reshape(-1, 3)
        idx = ((flat[:, None, :] - pal[None]) ** 2).sum(-1).argmin(dim=1)
        rows = [" ".join(names[int(idx[r * grid + c])] for c in range(grid)) for r in range(grid)]
        lines.append(f"frame {f} colors: " + " | ".join(rows))
    return lines


def shapecap_clip(frames_u8: torch.Tensor, tsub: int = TSUB) -> str:
    t = frames_u8.shape[0]
    frames = frames_u8[:: max(1, t // tsub)][:tsub].float() / 255.0
    sub = frames_u8[:: max(1, t // tsub)][:tsub]
    descs = torch.stack([_shape_descriptors(_foreground_mask(fr)) for fr in sub]).mean(dim=0)
    hu = descs[:7]
    circ, ecc, ext, larea, alog = (float(v) for v in descs[7:])
    hu_str = " ".join(f"{float(v):.2f}" for v in hu)
    cap = (
        "shape report. "
        f"compactness {circ:.2f}. elongation {ecc:.2f}. filledness {ext:.2f}. "
        f"logsize {larea:.2f}. aspect {alog:.2f}. "
        f"invariants: {hu_str}."
    )
    return cap + "\nscene colors.\n" + "\n".join(_color_grid_lines(frames))


def load_arm(model_id: str, arm: str, seed: int):
    from transformers import AutoConfig, AutoModel

    if arm == "real":
        model = AutoModel.from_pretrained(model_id, local_files_only=True, dtype=torch.float32)
    else:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
    return model.eval()


def mid_layer_mean(model, tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    return hs[len(hs) // 2][0].mean(dim=0).float()


def encode_arm(model_id: str, arm: str, params: dict, cache_root: Path, seed: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = load_arm(model_id, arm, seed)
    dim = int(model.config.hidden_size)
    n = params["n_shape"] * params["n_color"] * params["per"]
    store = LatentStore.create(
        cache_root, f"{STORE_PREFIX}_{arm}", feat_shape=(dim,), capacity=n, key_dim=dim, has_labels=True
    )
    shapes, colors, checks = [], [], {}
    sample_caption = ""
    t0 = time.perf_counter()
    for i, s, c, clip, _ in iter_bound_nuisance_clips(
        params["n_shape"], params["n_color"], params["per"], params["seed"]
    ):
        if i == 0:
            checks["first"] = clip_checksum(clip)
        if i == n - 1:
            checks["last"] = clip_checksum(clip)
        frames = to_uint8(clip)
        del clip  # stream as uint8, one clip in memory at a time
        cap = shapecap_clip(frames)
        if i == 0:
            sample_caption = cap
        z = mid_layer_mean(model, tokenizer, cap)
        store.write_batch(i, z[None].numpy(), z[None].numpy(), torch.tensor([s]).numpy())
        shapes.append(s)
        colors.append(c)
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"[{arm}] encoded {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    store.finalize()
    write_factors_sidecar(
        store.root,
        params,
        shapes,
        colors,
        checks,
        source="cache_qwen_shapecap",
        model_id=model_id,
        arm=arm,
        tsub=TSUB,
        grid=GRID,
        fg_threshold=FG_THRESHOLD,
    )
    x = store.latents()
    probe_s = linear_probe(x, torch.tensor(shapes), classification=True, epochs=300, seed=seed)
    probe_c = linear_probe(x, torch.tensor(colors), classification=True, epochs=300, seed=seed)
    del model, tokenizer
    gc.collect()
    return {
        "store": str(store.root),
        "embed_dim": dim,
        "shape_probe_acc": round(probe_s["score"], 4),
        "color_probe_acc": round(probe_c["score"], 4),
        "chance": round(probe_s["chance"], 4),
        "sample_caption": sample_caption,
        "clip_checksums": checks,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="small-LLM shape-caption nuisance cache (encoder lane)")
    ap.add_argument("--model-id", default=None, help="override; default reads the staging manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/cache_qwen_shapecap.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    assert_encoder_lane_free()  # live-process mutex: one encoder at a time, never skipped by --force
    model_id = resolve_model_id(a.model_id)
    params = read_encode_params(seed=a.seed)
    arms = {
        arm: encode_arm(model_id, arm, params, Path(a.cache_root), a.seed) for arm in ("real", "randominit")
    }
    real_acc = arms["real"]["shape_probe_acc"]
    killswitch_fired = real_acc <= KILLSWITCH_SHAPE_ACC
    out = {
        "experiment": "cache_qwen_shapecap",
        "clipset": CLIPSET,
        "model_id": model_id,
        "params": params,
        "tsub": TSUB,
        "grid": GRID,
        "fg_threshold": FG_THRESHOLD,
        "killswitch_shape_acc_threshold": KILLSWITCH_SHAPE_ACC,
        "killswitch_fired": killswitch_fired,
        "arms": arms,
        "note": "label-free SHAPE caption (Hu invariants + circularity/eccentricity/extent on a "
        "foreground mask, pixels only, labels never touch the text) + palette-color grid; the claim "
        "scope is pretraining-over-random-init within this rendering, nothing more. Kill-switch: if the "
        "real-arm shape probe was not meaningfully above chance, the caption failed and this is reported "
        "as an honest null, not tuned toward a positive.",
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
