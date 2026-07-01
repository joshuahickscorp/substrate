#!/usr/bin/env python
"""Cache a FACTORIZED real-encoder latent store: structured synthetic video in which TWO independent
visual factors vary (color hue = factor A, motion/orientation = factor B), run through the REAL frozen
V-JEPA 2 encoder. The content is synthetic but the perceptual geometry is the real encoder's. Unlike the
single-factor cache_real_encoder.py (where one class index entangles freq/angle/motion/color together),
this gives two SEPARATELY-DECODABLE factors, which is what held-out-combination and compositionality
probes need: train on a subset of (A, B) pairs, test whether unseen (A, B) pairs decode above the
shuffled/frozen-random floor on real encoder features.

Storage: LatentStore has one label channel, so the two factors are encoded as a composite label
y = a*n_b + b, plus a factors.json sidecar recording n_a/n_b. devsys.substrate.real_latent.factorized_arrays
reverses that into (x, y_a, y_b).

Usage: python scripts/cache_factorized_encoder.py [device=cpu] [+n_a=6] [+n_b=6] [+per=8] [+batch=1]
  MPS overflows the ViT-L attention buffer at 64-frame/256px even at batch=1, so use device=cpu.

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import json
import math
import sys
import time

import torch

from devsys.config import REPO_ROOT, compose
from devsys.devices import resolve, safe_to
from devsys.logging_utils import get_logger
from devsys.substrate import LatentStore, load_encoder

log = get_logger("cache_factorized")
FRAMES, RES = 64, 256


def _hue_tint(a: int, n_a: int) -> torch.Tensor:
    """Factor A: color hue, evenly spaced around the wheel. Independent of factor B."""
    h = a / max(1, n_a)
    return torch.tensor(
        [
            0.5 + 0.5 * math.cos(2 * math.pi * h),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 1 / 3)),
            0.5 + 0.5 * math.cos(2 * math.pi * (h + 2 / 3)),
        ]
    )


def make_factorized_clip(a: int, b: int, n_a: int, n_b: int, g: torch.Generator) -> torch.Tensor:
    """One [T,3,RES,RES] clip. Factor A (a) sets the COLOR HUE only; factor B (b) sets the grating
    ORIENTATION and DRIFT DIRECTION only. The spatial frequency is held fixed so the two factors are
    visually independent: any (hue, orientation) combination is realizable, which is what makes
    held-out-combination decoding a real test rather than an artifact of an entangled 1-D family."""
    lin = torch.linspace(0, 1, RES)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    theta = b * math.pi / max(1, n_b)  # factor B: orientation
    drift = 1.0 if (b % 2 == 0) else -1.0  # factor B: drift direction sign
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    freq = 5.0  # FIXED across both factors (not a hidden third factor)
    t = torch.arange(FRAMES).float() / FRAMES
    grating = torch.sin(2 * math.pi * (freq * proj[None] + drift * t[:, None, None]))
    grating = (grating + 1) / 2  # [T,RES,RES] in [0,1]
    tint = _hue_tint(a, n_a)  # factor A
    clip = grating[:, None, :, :] * tint[None, :, None, None]  # [T,3,RES,RES]
    clip = clip + 0.03 * torch.randn(clip.shape, generator=g)
    return clip.clamp(0, 1)


def main(argv: list[str] | None = None) -> int:
    cfg = compose(list(sys.argv[1:] if argv is None else argv) + ["encoder.prefer_real=true"])
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder).to(dev.device)
    if enc.spec.backend == "frozen_random":
        log.warning("real weights unavailable; this store will be frozen-random, not real-encoder")

    n_a = int(cfg.get("n_a", 6))
    n_b = int(cfg.get("n_b", 6))
    per = int(cfg.get("per", 8))  # clips per (a, b) cell
    batch = int(cfg.get("batch", 1))
    total = n_a * n_b * per
    g = torch.Generator().manual_seed(int(cfg.seed))

    name = f"{cfg.encoder.name}_factorized"
    store = LatentStore.create(
        REPO_ROOT / cfg.data_dir / "cache",
        name,
        feat_shape=(int(cfg.encoder.embed_dim),),
        capacity=total,
        key_dim=int(cfg.encoder.embed_dim),
        has_labels=True,
    )
    # composite label y = a*n_b + b, interleaved over cells then repeats
    cells = [(a, b) for a in range(n_a) for b in range(n_b)]
    order = [cells[i % len(cells)] for i in range(total)]
    labels = torch.tensor([a * n_b + b for (a, b) in order], dtype=torch.long)

    pos, t0 = 0, time.perf_counter()
    while pos < total:
        bs = min(batch, total - pos)
        clips = torch.stack(
            [make_factorized_clip(order[pos + j][0], order[pos + j][1], n_a, n_b, g) for j in range(bs)]
        )
        z = enc.encode(safe_to(clips, dev.device)).reshape(bs, -1).float().cpu()
        store.write_batch(pos, z.numpy(), z.numpy(), labels[pos : pos + bs].numpy())
        pos += bs
        log.info("encoded %d/%d (%.1fs)", pos, total, time.perf_counter() - t0)
    store.finalize()

    (store.root / "factors.json").write_text(
        json.dumps({"n_a": n_a, "n_b": n_b, "factor_a": "hue", "factor_b": "orientation_drift"}, indent=2)
    )
    secs = time.perf_counter() - t0
    log.info(
        "cached %d factorized real-encoder latents backend=%s in %.1fs (%.2fs/clip), n_a=%d n_b=%d",
        len(store),
        enc.spec.backend,
        secs,
        secs / total,
        n_a,
        n_b,
    )

    # immediate diagnostics: are BOTH factors independently decodable from the real latents?
    from devsys.diagnostics import linear_probe
    from devsys.substrate import factorized_arrays

    x, ya, yb = factorized_arrays(store)
    pa = linear_probe(x, ya, classification=True, epochs=300)
    pb = linear_probe(x, yb, classification=True, epochs=300)
    print(f"OK factorized cache: {len(store)} latents, backend={enc.spec.backend}, {secs / total:.2f}s/clip")
    print(f"   factor A (hue)   linear-probe acc={pa['score']:.3f} chance={pa['chance']:.3f}")
    print(f"   factor B (orient) linear-probe acc={pb['score']:.3f} chance={pb['chance']:.3f}")
    print(f"   (n={len(store)}, underpowered if small; both should decode if the encoder keeps both factors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
