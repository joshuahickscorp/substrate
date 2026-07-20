#!/usr/bin/env python

from __future__ import annotations

import json
import sys

from mop.config import REPO_ROOT, compose
from mop.devices import resolve
from mop.logging_utils import get_logger
from mop.substrate import cache_latents, iter_video_clips, load_encoder
from mop.substrate.video import detect_partial_cache, validate_source, write_label_map

log = get_logger("cache_video")


def main(argv: list[str] | None = None) -> int:
    cfg = compose(
        list(sys.argv[1:] if argv is None else argv)
        + ["encoder.prefer_real=true", "+encoder.require_real=true"]
    )
    source = cfg.get("source")
    if not source:
        print("FAIL: pass +source=/path/to/clips (a dir of <class>/<clip>.mp4)")
        return 1
    try:
        manifest = validate_source(source)
    except ValueError as e:
        print(f"FAIL: {e}")
        return 1
    log.info(
        "source ok: %d classes, %d clips, per_class=%s",
        len(manifest["classes"]),
        manifest["n_clips"],
        manifest["per_class"],
    )
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder).to(dev.device)
    fpc = int(cfg.encoder.frames_per_clip)
    res = int(cfg.encoder.resolution)
    total = int(cfg.get("total", 512))
    cache_root = REPO_ROOT / cfg.data_dir / "cache"
    name = f"{cfg.encoder.name}_video"
    detect_partial_cache(cache_root, name)  # report any pre-existing store before we overwrite it
    hashes: list[str] = []
    clips = iter_video_clips(
        source, frames_per_clip=fpc, res=res, batch=int(cfg.get("batch", 2)), limit=total, hashes_out=hashes
    )
    store = cache_latents(enc, clips, cache_root, name, total=total, device=dev)
    label_map_path = write_label_map(store.root, manifest["label_map"])
    n_dup = len(hashes) - len(set(hashes))
    (store.root / "clip_hashes.json").write_text(json.dumps(hashes, indent=2))
    if n_dup:
        log.warning("%d duplicate-content clip(s) cached (see clip_hashes.json)", n_dup)
    print(
        f"OK cached {len(store)} real-video latents backend={enc.spec.backend} "
        f"(duplicates={n_dup}) -> {store.root} ; label_map -> {label_map_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
