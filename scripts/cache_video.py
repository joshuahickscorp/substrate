#!/usr/bin/env python
"""Cache REAL natural-video latents. Decodes class-foldered video clips through the real frozen
V-JEPA encoder into a memmap store the whole shell consumes. This is the Studio day-one path:
drop clips under <source>/<class>/*.mp4 and run this.

Usage:
  python scripts/cache_video.py +source=/path/to/clips encoder=vjepa2_vitl_fpc64_256 device=mps +total=512
Needs a video backend: uv pip install -e ".[video]" (torchvision) or `uv pip install decord`.
On this laptop the 64-frame ViT-L forward is faster on cpu (Metal compiler limit); the Studio
should run it on mps. Latents are tagged backend=vjepa_hf (real) in the store meta.
"""

from __future__ import annotations

import sys

from devsys.config import REPO_ROOT, compose
from devsys.devices import resolve
from devsys.logging_utils import get_logger
from devsys.substrate import cache_latents, iter_video_clips, load_encoder

log = get_logger("cache_video")


def main(argv: list[str] | None = None) -> int:
    cfg = compose(list(sys.argv[1:] if argv is None else argv) + ["encoder.prefer_real=true"])
    source = cfg.get("source")
    if not source:
        print("FAIL: pass +source=/path/to/clips (a dir of <class>/<clip>.mp4)")
        return 1
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder).to(dev.device)
    if enc.spec.backend == "frozen_random":
        log.warning("real V-JEPA weights unavailable; latents will be frozen-random, not real")
    fpc = int(cfg.encoder.frames_per_clip)
    res = int(cfg.encoder.resolution)
    total = int(cfg.get("total", 512))
    clips = iter_video_clips(
        source, frames_per_clip=fpc, res=res, batch=int(cfg.get("batch", 2)), limit=total
    )
    name = f"{cfg.encoder.name}_video"
    store = cache_latents(enc, clips, REPO_ROOT / cfg.data_dir / "cache", name, total=total, device=dev)
    print(f"OK cached {len(store)} real-video latents backend={enc.spec.backend} -> {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
