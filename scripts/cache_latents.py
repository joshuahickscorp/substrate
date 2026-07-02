#!/usr/bin/env python
"""Cache frozen-encoder latents to a memmap store, ONCE. After this the shell trains on the
cache and never touches the encoder again.

Usage:
  python scripts/cache_latents.py encoder=vjepa2_vitl_fpc64_256 device=mps +total=256
Real video weights load lazily (uv pip install -e ".[encoder]" + network). With no weights
this caches through the frozen-random substrate over synthetic clips: the round-trip is fully
exercised and the store records backend=frozen_random so it is not mistaken for V-JEPA latents.
"""

from __future__ import annotations

import sys

from mop.config import REPO_ROOT, compose
from mop.devices import resolve
from mop.logging_utils import get_logger
from mop.substrate import cache_latents, load_encoder, synthetic_clips

log = get_logger("cache_latents")


def main(argv: list[str] | None = None) -> int:
    cfg = compose(list(sys.argv[1:] if argv is None else argv))
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder)
    total = int(cfg.get("total", 256))
    name = f"{cfg.encoder.name}_synthetic"
    clips = synthetic_clips(n=total, batch=16, shape=(3, 4, 16, 16), n_classes=int(cfg.get("n_classes", 8)))
    store = cache_latents(enc, clips, REPO_ROOT / cfg.data_dir / "cache", name, total=total, device=dev)
    log.info("cached %d latents backend=%s dim=%d", len(store), enc.spec.backend, store.meta.key_dim)
    print(f"OK cached {len(store)} latents backend={enc.spec.backend} -> {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
