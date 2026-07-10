#!/usr/bin/env python
"""WP-11 encoder-lane job Q2.2: the token/frame-matched SINGLE-FRAME V-JEPA pass for AT3 (time-axis
ablation), writing `data/cache/vjepa2_vitl_singleframe`. Each canonical nuisance clip is reduced to its
middle frame TILED to the full 64-frame extent, so the encoder consumes the identical token grid
(64 x 256 x 256) with ZERO temporal variation: any full-clip advantage over this arm is a temporal-axis
currency, not a raw-token-count advantage.

Because AT3's comparison (and PR2's real arm) needs full-clip V-JEPA features on the SAME clips, and no
other queue row encodes them, this pass also writes `data/cache/vjepa2_vitl_nuisance` (the full-clip
pooled column) in the same serial pass by default: the model is loaded once, each clip gets two forwards.
Disable with --no-full-clip if that column already exists.

Encoder-lane rule: refuses to start while any other encoder process is alive (pgrep guard). Streams one
clip at a time. Uses the same clip regeneration and cache helpers as cache_randominit_vitl_features.py
(clip identity rule: torch RNG, seed for seed).

Usage: python scripts/cache_vjepa_single_frame.py \
    --out-dir data/cache/vjepa2_vitl_singleframe --log runs/mot/cache_singleframe.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import (  # noqa: E402
    CANONICAL,
    NUISANCE_COLS,
    assert_encoder_lane_free,
    iter_nuisance_clips,
    write_feature_cache,
)

from mop.config import compose  # noqa: E402
from mop.devices import resolve, safe_to  # noqa: E402
from mop.substrate import load_encoder  # noqa: E402


def single_frame_clip(clip: torch.Tensor) -> torch.Tensor:
    """[T, 3, H, W] -> the middle frame repeated T times: identical token/frame count, no temporal
    information. The matched control that makes a full-clip advantage a TIME claim."""
    t = clip.shape[0]
    return clip[t // 2].unsqueeze(0).expand(t, -1, -1, -1).contiguous()


def run_cache(
    n_shape: int,
    n_color: int,
    per: int,
    seed: int,
    out_dir: str,
    full_dir: str,
    log_path: str,
    also_full_clip: bool = True,
) -> dict:
    """One serial pass: load real V-JEPA once, per clip encode the single-frame-tiled version (and the
    full clip when requested), write both caches plus the run log."""
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
    feats_sf, feats_full, ys, yc, nuis = [], [], [], [], []
    t0 = time.perf_counter()
    n_total = n_shape * n_color * per
    with torch.no_grad():
        for i, s, c, draws, clip in iter_nuisance_clips(n_shape, n_color, per, seed):
            sf = single_frame_clip(clip)
            feats_sf.append(enc.encode(safe_to(sf.unsqueeze(0), dev.device)).reshape(-1).float().cpu())
            if also_full_clip:
                feats_full.append(
                    enc.encode(safe_to(clip.unsqueeze(0), dev.device)).reshape(-1).float().cpu()
                )
            ys.append(s)
            yc.append(c)
            nuis.append([draws[k] for k in NUISANCE_COLS])
            del clip, sf
            if (i + 1) % 8 == 0 or i + 1 == n_total:
                print(f"encoded {i + 1}/{n_total} ({time.perf_counter() - t0:.0f}s)", flush=True)

    def meta_for(tag: str, notes: str, distinct_frames: int) -> dict:
        return {
            "tag": tag,
            "embed_dim": int(feats_sf[0].numel()),
            "input_resolution": 256,
            "frames": 64,
            "distinct_frames": distinct_frames,
            "pretrained": True,
            "family": "vjepa_hf",
            "backend": backend,
            "n_shape": n_shape,
            "n_color": n_color,
            "per_cell": per,
            "seed": seed,
            "count": len(feats_sf),
            "nuisance_cols": list(NUISANCE_COLS),
            "notes": notes,
        }

    write_feature_cache(
        out_dir,
        features=torch.stack(feats_sf),
        labels_shape=ys,
        labels_color=yc,
        nuisance=nuis,
        meta=meta_for(
            "vjepa2_vitl_singleframe",
            "middle frame tiled to 64 frames: token/frame-matched, zero temporal variation (AT3 arm)",
            1,
        ),
    )
    if also_full_clip:
        write_feature_cache(
            full_dir,
            features=torch.stack(feats_full),
            labels_shape=ys,
            labels_color=yc,
            nuisance=nuis,
            meta=meta_for(
                "vjepa2_vitl_nuisance",
                "full-clip pooled V-JEPA on the canonical nuisance clips (AT3 full arm, PR2 real arm)",
                64,
            ),
        )
    log = {
        "single_frame_cache": str(out_dir),
        "full_clip_cache": str(full_dir) if also_full_clip else None,
        "backend": backend,
        "valid": backend == "vjepa_hf",
        "n_clips": len(feats_sf),
        "seconds": round(time.perf_counter() - t0, 1),
        "seed": seed,
        "note": "INVALID: encoder ran as frozen_random, rerun with real weights"
        if backend != "vjepa_hf"
        else "ok",
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(log, indent=2, default=str))
    return log


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="single-frame (token/frame-matched) V-JEPA pass for AT3")
    ap.add_argument("--n-shape", type=int, default=CANONICAL["n_shape"])
    ap.add_argument("--n-color", type=int, default=CANONICAL["n_color"])
    ap.add_argument("--per", type=int, default=CANONICAL["per"])
    ap.add_argument("--seed", type=int, default=CANONICAL["seed"])
    ap.add_argument("--out-dir", default="data/cache/vjepa2_vitl_singleframe")
    ap.add_argument("--full-dir", default="data/cache/vjepa2_vitl_nuisance")
    ap.add_argument("--no-full-clip", action="store_true", help="skip the full-clip companion cache")
    ap.add_argument("--log", default="runs/mot/cache_singleframe.json")
    a = ap.parse_args(argv)
    assert_encoder_lane_free()
    log = run_cache(a.n_shape, a.n_color, a.per, a.seed, a.out_dir, a.full_dir, a.log, not a.no_full_clip)
    print(json.dumps(log, indent=2, default=str))
    return 0 if log["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
