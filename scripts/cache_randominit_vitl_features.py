#!/usr/bin/env python
"""WP-11 encoder-lane job Q2.1: ONE logged encode pass of the canonical 200 nuisance clips through a
random-init same-arch ViT-L at 256px, writing `data/cache/randominit_vitl_nuisance`. This is the
non-vacuous control column for PR2 (plasticity real vs random-encoder), the DR8 typing arm, and the
AT1/AL2 grids: identical architecture, identical resolution, identical clips, UNTRAINED weights, so any
real-minus-randominit delta isolates pretraining (never a square latent projection).

Clip identity rule (EXECUTION_MANIFEST WP-01): the nuisance clips exist only in memory, drawn by torch
RNG inside scripts/compositional_under_nuisance.py. This script regenerates them seed for seed by
importing the SAME generator function, streams ONE clip at a time (never a batched float32 tensor), and
records the per-clip nuisance draws (scale, rotation, position, motion) by replaying the generator
stream from a cloned state, so AT3 can decode temporal factors without re-deriving anything.

Encoder-lane rule: this script REFUSES to start while any other encoder process is alive (pgrep guard),
per the one-encoder-at-a-time law. The random-init build reuses `_random_init_vit` from
scripts/substrate_vs_random_init_vit.py, it does not duplicate the encode logic.

Cache layout (shared by every WP-11 consumer via `load_feature_cache`):
  <dir>/features.npy [N, D] float32, labels_shape.npy [N] int64, labels_color.npy [N] int64,
  nuisance.npy [N, 6] float32 (columns r, rot, x0, y0, vx, vy), meta.json.

Usage: python scripts/cache_randominit_vitl_features.py \
    --out-dir data/cache/randominit_vitl_nuisance --log runs/mot/cache_randominit_vitl.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compositional_under_nuisance import make_bound_nuisance_clip  # noqa: E402
from scripts.substrate_vs_random_init_vit import _random_init_vit  # noqa: E402

from mop.config import compose  # noqa: E402
from mop.devices import resolve, safe_to  # noqa: E402

# Canonical nuisance population: the exact grid the in-flight compositional_under_nuisance run encodes
# (5 shapes x 5 colors x 8 clips per cell, generator seed 0, cell-major order). Every WP-11/WP-01 cache
# must use these numbers or the cross-column comparisons silently diverge.
CANONICAL = {"n_shape": 5, "n_color": 5, "per": 8, "seed": 0}

NUISANCE_COLS = ("r", "rot", "x0", "y0", "vx", "vy")

# Any process whose command line matches one of these owns (or would own) the encoder lane.
ENCODER_PROC_PATTERNS = (
    "compositional_under_nuisance",
    "substrate_vs_random_features",
    "substrate_vs_random_init_vit",
    "cache_randominit_vitl_features",
    "cache_vjepa_single_frame",
    "cache_dinov2s_nuisance",
    "cache_qwen_textified",
    "cache_wav2vec2_sonified",
    "cache_real_encoder",
    "cache_latents",
    "cache_factorized_encoder",
)


def assert_encoder_lane_free(patterns: tuple[str, ...] = ENCODER_PROC_PATTERNS) -> None:
    """Refuse to start if any other encoder-lane process is alive (one encoder at a time, 18GB pool).
    Own pid and parent pid are excluded (the launching shell's command line echoes the script name)."""
    mine = {os.getpid(), os.getppid()}
    alive: list[tuple[int, str]] = []
    for pat in patterns:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        for tok in r.stdout.split():
            pid = int(tok)
            if pid not in mine:
                alive.append((pid, pat))
    if alive:
        raise SystemExit(
            f"encoder lane BUSY, refusing to start: {alive}. One encoder at a time; rerun after the "
            "in-flight encode exits."
        )


def extract_nuisance_draws(g: torch.Generator) -> dict[str, float]:
    """Replay the first six torch.rand draws of make_bound_nuisance_clip from a CLONED generator state
    (consuming nothing), returning the transformed nuisance values in source order: scale r, rotation,
    start position x0/y0, motion vx/vy. Must stay in lockstep with the source draw order; the unit test
    checks the extracted motion against the rendered clip's actual centroid drift."""
    clone = torch.Generator()
    clone.set_state(g.get_state())
    u = [float(torch.rand(1, generator=clone)) for _ in range(6)]
    return {
        "r": 0.15 + 0.2 * u[0],
        "rot": u[1] * 2 * math.pi,
        "x0": -0.5 + u[2],
        "y0": -0.5 + u[3],
        "vx": 0.4 * (u[4] - 0.5),
        "vy": 0.4 * (u[5] - 0.5),
    }


def iter_nuisance_clips(n_shape: int, n_color: int, per: int, seed: int):
    """Yield (index, shape, color, draws, clip) for the canonical population, one clip at a time from a
    single sequential generator, exactly replicating compositional_under_nuisance.run's cell order."""
    g = torch.Generator().manual_seed(seed)
    idx = 0
    for s in range(n_shape):
        for c in range(n_color):
            for _ in range(per):
                draws = extract_nuisance_draws(g)
                clip = make_bound_nuisance_clip(s, c, n_color, g)
                yield idx, s, c, draws, clip
                idx += 1


def write_feature_cache(
    root: Path | str,
    *,
    features: torch.Tensor,
    labels_shape,
    labels_color=None,
    nuisance=None,
    meta: dict | None = None,
) -> Path:
    """Write the WP-11 pooled-feature cache layout. `features` [N, D]; labels int64; nuisance [N, 6]
    float32 in NUISANCE_COLS order; meta is the honest-confound record (resolution, frames, pretrained)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "features.npy", features.detach().cpu().to(torch.float32).numpy())
    np.save(root / "labels_shape.npy", np.asarray(labels_shape, dtype="int64"))
    if labels_color is not None:
        np.save(root / "labels_color.npy", np.asarray(labels_color, dtype="int64"))
    if nuisance is not None:
        np.save(root / "nuisance.npy", np.asarray(nuisance, dtype="float32"))
    (root / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
    return root


def load_feature_cache(root: Path | str) -> dict:
    """Load a pooled-feature cache tolerantly. Accepts the WP-11 layout (features.npy + labels_shape),
    the LatentStore layout (latents.npy + labels.npy), or a features.pt torch dict, so WP-01 caches from
    the other lane remain readable whatever layout they chose. Returns features [N, D] float32,
    labels_shape [N] (int64 or None), labels_color, nuisance [N, 6] (or None), meta dict."""
    root = Path(root)
    out: dict = {"meta": {}, "labels_shape": None, "labels_color": None, "nuisance": None}
    if (root / "meta.json").exists():
        out["meta"] = json.loads((root / "meta.json").read_text())
    if (root / "features.npy").exists():
        feats = np.load(root / "features.npy")
    elif (root / "latents.npy").exists():
        feats = np.asarray(np.load(root / "latents.npy", mmap_mode="r"))
    elif (root / "features.pt").exists():
        blob = torch.load(root / "features.pt", map_location="cpu")
        if isinstance(blob, dict):
            feats = blob.get("features", blob.get("latents"))
            for src, dst in (
                ("labels_shape", "labels_shape"),
                ("labels", "labels_shape"),
                ("labels_color", "labels_color"),
                ("nuisance", "nuisance"),
            ):
                if out[dst] is None and src in blob:
                    out[dst] = torch.as_tensor(blob[src])
        else:
            feats = blob
        feats = torch.as_tensor(feats).cpu().numpy()
    else:
        raise FileNotFoundError(f"no feature cache at {root} (features.npy / latents.npy / features.pt)")
    feats = np.asarray(feats)
    out["features"] = torch.as_tensor(feats, dtype=torch.float32).reshape(feats.shape[0], -1)
    for name in ("labels_shape", "labels_color", "nuisance"):
        p = root / f"{name}.npy"
        if out[name] is None and p.exists():
            out[name] = torch.as_tensor(np.load(p))
    if out["labels_shape"] is None and (root / "labels.npy").exists():
        out["labels_shape"] = torch.as_tensor(np.load(root / "labels.npy"))
    for name in ("labels_shape", "labels_color"):
        if out[name] is not None:
            out[name] = out[name].to(torch.int64).reshape(-1)
    if out["nuisance"] is not None:
        out["nuisance"] = out["nuisance"].to(torch.float32)
    return out


def run_cache(n_shape: int, n_color: int, per: int, seed: int, out_dir: str, log_path: str) -> dict:
    """The single logged encode pass: build the random-init ViT-L, stream the canonical clips through
    it one at a time, write the cache and the run log. RAM: one model + one clip at a time."""
    cfg = compose(["encoder=vjepa2_vitl_fpc64_256", "device=cpu"])
    dev = resolve("cpu")
    rand = _random_init_vit(cfg.encoder)
    if rand is None:
        raise SystemExit("could not build the random-init ViT offline (transformers config unavailable)")
    rand = rand.to(dev.device)
    feats, ys, yc, nuis = [], [], [], []
    t0 = time.perf_counter()
    n_total = n_shape * n_color * per
    with torch.no_grad():
        for i, s, c, draws, clip in iter_nuisance_clips(n_shape, n_color, per, seed):
            z = rand.encode(safe_to(clip.unsqueeze(0), dev.device)).reshape(-1).float().cpu()
            feats.append(z)
            ys.append(s)
            yc.append(c)
            nuis.append([draws[k] for k in NUISANCE_COLS])
            del clip
            if (i + 1) % 8 == 0 or i + 1 == n_total:
                print(f"encoded {i + 1}/{n_total} ({time.perf_counter() - t0:.0f}s)", flush=True)
    x = torch.stack(feats)
    meta = {
        "tag": "randominit_vitl_nuisance",
        "embed_dim": int(x.shape[1]),
        "input_resolution": 256,
        "frames": 64,
        "pretrained": False,
        "family": "random_init_vit",
        "backend": rand.spec.backend,
        "n_shape": n_shape,
        "n_color": n_color,
        "per_cell": per,
        "seed": seed,
        "count": int(x.shape[0]),
        "nuisance_cols": list(NUISANCE_COLS),
        "notes": "same-arch untrained ViT-L at matched 256px on the canonical nuisance clips (PR2/AT1 "
        "control column)",
    }
    write_feature_cache(out_dir, features=x, labels_shape=ys, labels_color=yc, nuisance=nuis, meta=meta)
    log = dict(meta)
    log.update(
        {
            "cache_dir": str(out_dir),
            "seconds": round(time.perf_counter() - t0, 1),
            "feature_norm_mean": round(float(x.norm(dim=-1).mean()), 4),
            "max_cache_clips_note": "200 pooled vectors (~800KB), clamp override logged: the clamp "
            "guards dense real-video caches, this cache is pooled synthetic nuisance features",
        }
    )
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(json.dumps(log, indent=2, default=str))
    return log


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="random-init ViT-L feature pass over the nuisance clips")
    ap.add_argument("--n-shape", type=int, default=CANONICAL["n_shape"])
    ap.add_argument("--n-color", type=int, default=CANONICAL["n_color"])
    ap.add_argument("--per", type=int, default=CANONICAL["per"])
    ap.add_argument("--seed", type=int, default=CANONICAL["seed"])
    ap.add_argument("--out-dir", default="data/cache/randominit_vitl_nuisance")
    ap.add_argument("--log", default="runs/mot/cache_randominit_vitl.json")
    a = ap.parse_args(argv)
    assert_encoder_lane_free()
    log = run_cache(a.n_shape, a.n_color, a.per, a.seed, a.out_dir, a.log)
    print(json.dumps(log, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
