#!/usr/bin/env python
"""WP-01 cache pass (Q2.3, ENCODER LANE ONLY): DINOv2-S real AND same-arch from_config random-init over
the shared nuisance clips. 8 subsampled frames per clip at 224px, CLS token per frame, mean pooled over
frames, into data/cache/dinov2s_nuisance_{real,randominit}. Identical preprocessing for both arms: the
random-init column is the non-vacuous control (never a square latent projection).

HARD RULES: this is a torch encode job. It queues strictly AFTER the in-flight V-JEPA encode, serially,
one model in memory at a time (the two arms run sequentially, clips regenerated per arm, which is free
because regeneration is deterministic). Clips stream one at a time as uint8 (clip identity rule); no
batching of clips, ever. Weights must already be staged by stage_small_substrates.py; loading is
local_files_only so this job can never hit the network.

Usage: python scripts/cache_dinov2s_nuisance.py   -> runs/mot/cache_dinov2s.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

from mop.config import REPO_ROOT  # noqa: E402
from mop.diagnostics import linear_probe  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

MODEL_ID = "facebook/dinov2-small"
STAGING_MANIFEST = REPO_ROOT / "runs" / "mot" / "staging_manifest.json"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
TSUB, RES_IN = 8, 224


def guard_staged() -> None:
    if not STAGING_MANIFEST.exists():
        raise SystemExit(f"{STAGING_MANIFEST} missing: run scripts/stage_small_substrates.py first")


def preprocess_frames(frames_u8: torch.Tensor, res: int = RES_IN) -> torch.Tensor:
    """[t,3,H,W] uint8 -> [t,3,res,res] float, bilinear resize + ImageNet normalization. The ONE
    preprocessing path, shared verbatim by the real and random-init arms."""
    x = frames_u8.float() / 255.0
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)
    return (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]


def load_arm(arm: str, seed: int):
    """Load exactly one model into memory. real = staged pretrained weights; randominit = the same
    architecture from_config with seeded random weights (the honest control)."""
    from transformers import AutoConfig, AutoModel

    if arm == "real":
        model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    else:
        cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
        torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
    return model.eval()


def encode_arm(arm: str, params: dict, cache_root: Path, seed: int) -> dict:
    model = load_arm(arm, seed)
    dim = int(model.config.hidden_size)
    n = params["n_shape"] * params["n_color"] * params["per"]
    store = LatentStore.create(
        cache_root, f"dinov2s_nuisance_{arm}", feat_shape=(dim,), capacity=n, key_dim=dim, has_labels=True
    )
    shapes, colors, checks = [], [], {}
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, s, c, clip, _ in iter_bound_nuisance_clips(
            params["n_shape"], params["n_color"], params["per"], params["seed"]
        ):
            if i == 0:
                checks["first"] = clip_checksum(clip)
            if i == n - 1:
                checks["last"] = clip_checksum(clip)
            frames = to_uint8(clip)[:: max(1, clip.shape[0] // TSUB)][:TSUB]
            del clip  # stream as uint8, one clip in memory at a time
            x = preprocess_frames(frames)
            z = model(pixel_values=x).last_hidden_state[:, 0].mean(dim=0).float()  # CLS, frame mean
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
        source="cache_dinov2s_nuisance",
        model_id=MODEL_ID,
        arm=arm,
        frames=TSUB,
        resolution=RES_IN,
    )
    probe = linear_probe(store.latents(), torch.tensor(shapes), classification=True, epochs=300, seed=seed)
    del model
    gc.collect()
    return {
        "store": str(store.root),
        "embed_dim": dim,
        "shape_probe_acc": round(probe["score"], 4),
        "chance": round(probe["chance"], 4),
        "clip_checksums": checks,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DINOv2-S real + random-init nuisance cache (encoder lane)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/cache_dinov2s.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    assert_encoder_lane_free()  # live-process mutex: one encoder at a time, never skipped by --force
    guard_staged()
    params = read_encode_params(seed=a.seed)
    arms = {arm: encode_arm(arm, params, Path(a.cache_root), a.seed) for arm in ("real", "randominit")}
    out = {
        "experiment": "cache_dinov2s_nuisance",
        "clipset": CLIPSET,
        "model_id": MODEL_ID,
        "params": params,
        "frames": TSUB,
        "resolution": RES_IN,
        "arms": arms,
        "note": "real and random-init share one preprocessing path; the delta over random-init is the "
        "only licensed substrate reading (delta over any square projection is banned as vacuous)",
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
