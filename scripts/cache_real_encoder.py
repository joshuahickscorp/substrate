#!/usr/bin/env python

from __future__ import annotations

import math
import sys

import torch

from mop.config import REPO_ROOT, compose
from mop.devices import resolve, safe_to
from mop.logging_utils import get_logger
from mop.substrate import LatentStore, load_encoder
from mop.substrate.datasets import make_task_stream  # noqa: F401 (kept for parity)

log = get_logger("cache_real")
FRAMES, RES = 64, 256


def _class_color(c: int, k: int) -> torch.Tensor:
    h = c / max(1, k)
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * h),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 2 / 3)),
        ]
    )


def make_class_clip(c: int, k: int, g: torch.Generator) -> torch.Tensor:
    lin = torch.linspace(0, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    freq = 3.0 + 2.5 * c
    theta = c * math.pi / max(1, k)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    t = torch.arange(FRAMES).float() / FRAMES
    grating = torch.sin(2 * math.pi * (freq * proj[None] + (1.0 + 0.3 * c) * t[:, None, None]))
    grating = (grating + 1) / 2  # [T,RES,RES] in [0,1]
    tint = _class_color(c, k)
    clip = grating[:, None, :, :] * tint[None, :, None, None]  # [T,3,RES,RES]
    clip = clip + 0.03 * torch.randn(clip.shape, generator=g)
    return clip.clamp(0, 1)


def main(argv: list[str] | None = None) -> int:
    cfg = compose(
        list(sys.argv[1:] if argv is None else argv)
        + ["encoder.prefer_real=true", "+encoder.require_real=true"]
    )
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder).to(dev.device)
    if enc.spec.backend == "frozen_random":
        log.warning("real weights unavailable; this store will be frozen-random, not real-encoder")
    k = int(cfg.get("classes", 8))
    per = int(cfg.get("per_class", 16))
    total = k * per
    batch = int(cfg.get("batch", 4))
    g = torch.Generator().manual_seed(int(cfg.seed))

    name = f"{cfg.encoder.name}_real"
    store = LatentStore.create(
        REPO_ROOT / cfg.data_dir / "cache",
        name,
        feat_shape=(int(cfg.encoder.embed_dim),),
        capacity=total,
        key_dim=int(cfg.encoder.embed_dim),
        has_labels=True,
    )
    labels = torch.arange(total) % k  # interleave classes
    import time

    pos, t0 = 0, time.perf_counter()
    while pos < total:
        bs = min(batch, total - pos)
        clips = torch.stack(
            [make_class_clip(int(labels[pos + j]), k, g) for j in range(bs)]
        )  # [bs,T,3,RES,RES]
        z = enc.encode(safe_to(clips, dev.device)).reshape(bs, -1).float().cpu()
        store.write_batch(pos, z.numpy(), z.numpy(), labels[pos : pos + bs].numpy())
        pos += bs
        log.info("encoded %d/%d (%.1fs)", pos, total, time.perf_counter() - t0)
    store.finalize()
    secs = time.perf_counter() - t0
    log.info(
        "cached %d real-encoder latents backend=%s in %.1fs (%.2fs/clip)",
        len(store),
        enc.spec.backend,
        secs,
        secs / total,
    )

    from mop.diagnostics import linear_probe

    probe = linear_probe(store.latents(), store.labels(), classification=True, epochs=300)
    print(
        f"OK real-encoder cache: {len(store)} latents, backend={enc.spec.backend}, {secs / total:.2f}s/clip"
    )
    print(
        f"   linear-probe (REAL-ENCODER) acc={probe['score']:.3f} chance={probe['chance']:.3f} "
        f"decodable={probe['decodable']} (n={len(store)}, underpowered if small)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
