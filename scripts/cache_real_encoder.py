#!/usr/bin/env python
"""Cache a small REAL-ENCODER latent store. Real V-JEPA 2 ViT-L weights are loaded and run
over STRUCTURED synthetic video (per-class color / orientation / spatial-frequency / motion),
chunked on MPS. There is no natural-video dataset here, so the content is synthetic, but the
PERCEPTUAL GEOMETRY is the real encoder's: this is the real-encoder answer to "is class info
linearly decodable from V-JEPA latents". Pooled latents + labels land in a memmap store.

Usage: python scripts/cache_real_encoder.py [device=mps] [+classes=8] [+per_class=16]
"""

from __future__ import annotations

import math
import sys

import torch

from devsys.config import REPO_ROOT, compose
from devsys.devices import resolve, safe_to
from devsys.logging_utils import get_logger
from devsys.substrate import LatentStore, load_encoder
from devsys.substrate.datasets import make_task_stream  # noqa: F401 (kept for parity)

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
    """One [T,3,RES,RES] clip: a moving oriented grating with class-specific frequency, angle,
    motion speed, and color tint. Distinct classes -> distinct V-JEPA latents (if the encoder
    preserves them, which the linear probe then measures)."""
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
    cfg = compose(list(sys.argv[1:] if argv is None else argv) + ["encoder.prefer_real=true"])
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

    # immediate real-encoder diagnostic: is class linearly decodable from the real latents?
    from devsys.diagnostics import linear_probe

    probe = linear_probe(store.latents(), store.labels(), classification=True, epochs=300)
    print(
        f"OK real-encoder cache: {len(store)} latents, backend={enc.spec.backend}, {secs / total:.2f}s/clip"
    )
    print(f"   linear-probe (REAL-ENCODER) acc={probe['score']:.3f} chance={probe['chance']:.3f} "
          f"decodable={probe['decodable']} (n={len(store)}, underpowered if small)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
