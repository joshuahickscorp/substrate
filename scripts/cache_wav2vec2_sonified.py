#!/usr/bin/env python
"""WP-01 cache pass (Q2.5, ENCODER LANE ONLY): a PREREGISTERED deterministic SONIFICATION of the shared
nuisance clips through wav2vec2-base, real AND from_config random-init, mid-layer mean hidden state,
into data/cache/wav2vec2_sonified_{real,randominit}.

PREREGISTERED MAPPING (fixed before any result exists): each of 16 subsampled frames becomes a 0.1s
segment at 16kHz; mean luminance sets the carrier frequency, saturation-weighted mean hue sets a second
oscillator, spatial gradient energy sets a deterministic chirp amplitude, and the brightness centroid
sets the segment gain. The waveform is zero-mean unit-variance normalized (the wav2vec2 convention).

PREREGISTERED CAVEAT (per the WP-01 null block): the sonification INJECTS factor structure (hue maps to
frequency by construction), so decodability from the audio substrate is NOT evidence the factor is
substrate-universal. The only licensed claim from this cache is pretraining-over-random-init WITHIN this
rendering. This caveat ships in the output json.

HARD RULES: encoder lane only, strictly after the in-flight V-JEPA encode, one model in memory at a
time (arms sequential), clips streamed one at a time as uint8 (clip identity rule), local_files_only.

Usage: python scripts/cache_wav2vec2_sonified.py   -> runs/mot/cache_wav2vec2.json

No em dashes or en dashes (BLACKHOLE.md).
"""

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

MODEL_ID = "facebook/wav2vec2-base"
STAGING_MANIFEST = REPO_ROOT / "runs" / "mot" / "staging_manifest.json"
STORE_PREFIX = "wav2vec2_sonified"
SR, SEG_SECONDS, TSUB = 16000, 0.1, 16
CAVEAT = (
    "the sonification mapping injects factor structure (hue -> frequency by construction); the only "
    "licensed claim from this cache is pretraining-over-random-init within this rendering"
)


def sonify_clip(
    frames_u8: torch.Tensor, sr: int = SR, seg: float = SEG_SECONDS, tsub: int = TSUB
) -> torch.Tensor:
    """Deterministic clip -> waveform mapping (no RNG anywhere). [T,3,H,W] uint8 -> [tsub*seg*sr]
    float waveform, zero-mean unit-variance."""
    t = frames_u8.shape[0]
    frames = frames_u8[:: max(1, t // tsub)][:tsub].float() / 255.0  # [t,3,H,W]
    n_seg = int(sr * seg)
    ts = torch.arange(n_seg).float() / sr
    segments = []
    for f in range(frames.shape[0]):
        rgb = frames[f]  # [3,H,W]
        lum = rgb.mean()
        mx, _ = rgb.max(dim=0)
        mn, _ = rgb.min(dim=0)
        sat = (mx - mn).mean()
        gray = rgb.mean(dim=0)
        grad = (gray[:, 1:] - gray[:, :-1]).abs().mean() + (gray[1:, :] - gray[:-1, :]).abs().mean()
        ys = torch.linspace(0, 1, gray.shape[0])[:, None]
        cy = float((gray * ys).sum() / (gray.sum() + 1e-8))
        f1 = 220.0 + 660.0 * float(lum)  # luminance carrier
        f2 = 880.0 + 440.0 * float(sat)  # saturation-weighted second oscillator
        chirp_amp = 0.3 * min(1.0, 4.0 * float(grad))  # texture as a deterministic chirp, not noise
        gain = 0.4 + 0.5 * cy  # brightness centroid height as segment gain
        seg_wave = gain * (
            0.6 * torch.sin(2 * math.pi * f1 * ts)
            + 0.4 * torch.sin(2 * math.pi * f2 * ts)
            + chirp_amp * torch.sin(2 * math.pi * (2000.0 + 1500.0 * float(grad)) * ts + 40.0 * ts * ts)
        )
        segments.append(seg_wave)
    wave = torch.cat(segments)
    return (wave - wave.mean()) / (wave.std() + 1e-8)


def load_arm(arm: str, seed: int):
    from transformers import AutoConfig, AutoModel

    if arm == "real":
        model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    else:
        cfg = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
        torch.manual_seed(seed)
        model = AutoModel.from_config(cfg)
    return model.eval()


def mid_layer_mean(model, wave: torch.Tensor) -> torch.Tensor:
    """Mid-layer mean hidden state over the sonified waveform -> [D]."""
    with torch.no_grad():
        hs = model(input_values=wave[None], output_hidden_states=True).hidden_states
    return hs[len(hs) // 2][0].mean(dim=0).float()


def encode_arm(arm: str, params: dict, cache_root: Path, seed: int) -> dict:
    model = load_arm(arm, seed)
    dim = int(model.config.hidden_size)
    n = params["n_shape"] * params["n_color"] * params["per"]
    store = LatentStore.create(
        cache_root, f"{STORE_PREFIX}_{arm}", feat_shape=(dim,), capacity=n, key_dim=dim, has_labels=True
    )
    shapes, colors, checks = [], [], {}
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
        z = mid_layer_mean(model, sonify_clip(frames))
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
        source="cache_wav2vec2_sonified",
        model_id=MODEL_ID,
        arm=arm,
        sr=SR,
        tsub=TSUB,
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
    ap = argparse.ArgumentParser(description="wav2vec2 sonified nuisance cache (encoder lane)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/cache_wav2vec2.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    assert_encoder_lane_free()  # live-process mutex: one encoder at a time, never skipped by --force
    if not STAGING_MANIFEST.exists():
        raise SystemExit(f"{STAGING_MANIFEST} missing: run scripts/stage_small_substrates.py first")
    params = read_encode_params(seed=a.seed)
    arms = {arm: encode_arm(arm, params, Path(a.cache_root), a.seed) for arm in ("real", "randominit")}
    out = {
        "experiment": "cache_wav2vec2_sonified",
        "clipset": CLIPSET,
        "model_id": MODEL_ID,
        "params": params,
        "sr": SR,
        "tsub": TSUB,
        "arms": arms,
        "preregistered_caveat": CAVEAT,
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
