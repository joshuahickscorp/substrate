#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compositional_under_nuisance import make_bound_nuisance_clip  # noqa: E402

from mop.config import REPO_ROOT  # noqa: E402
from mop.diagnostics import linear_probe  # noqa: E402
from mop.substrate import LatentStore  # noqa: E402

ENCODE_JSON = REPO_ROOT / "runs" / "pre_studio" / "compositional_under_nuisance.json"
CACHE_ROOT = REPO_ROOT / "data" / "cache"
CLIPSET = "bound_nuisance_v1"
N_SCALAR_DRAWS = 15  # r, rot, x0, y0, vx, vy, then (bcx, bcy, br) x 3 clutter blobs, in draw order
DEFAULT_PARAMS = {"n_shape": 5, "n_color": 5, "per": 8, "seed": 0}


def guard_post_encode(force: bool = False) -> None:
    if force or ENCODE_JSON.exists():
        return
    raise SystemExit(
        f"refusing to run: {ENCODE_JSON} not found, so the in-flight V-JEPA encode has not exited. "
        "This featurizer is a torch-CPU job (clip identity rule) and must queue after the encode "
        "(Q1.0a/Q1.0b). Pass --force only if you know the encoder lane is free."
    )


def read_encode_params(path: Path = ENCODE_JSON, seed: int = 0) -> dict:
    if path.exists():
        d = json.loads(path.read_text())
        return {
            "n_shape": int(d["n_shape"]),
            "n_color": int(d["n_color"]),
            "per": int(d["per_cell"]),
            "seed": seed,
            "params_source": str(path),
        }
    return dict(DEFAULT_PARAMS, seed=seed, params_source="defaults (encode json absent)")


class _RandRecorder:

    def __init__(self, generator: torch.Generator):
        self.generator = generator
        self.draws: list[float] = []

    def __enter__(self) -> _RandRecorder:
        self._orig = torch.rand

        def wrapped(*args, **kwargs):
            out = self._orig(*args, **kwargs)
            if kwargs.get("generator") is self.generator and out.numel() == 1:
                self.draws.append(float(out))
            return out

        torch.rand = wrapped
        return self

    def __exit__(self, *exc) -> bool:
        torch.rand = self._orig
        return False


def iter_bound_nuisance_clips(
    n_shape: int, n_color: int, per: int, seed: int, record: bool = False
) -> Iterator[tuple[int, int, int, torch.Tensor, list[float] | None]]:
    g = torch.Generator().manual_seed(seed)
    cells = [(s, c) for s in range(n_shape) for c in range(n_color) for _ in range(per)]
    for i, (s, c) in enumerate(cells):
        if record:
            with _RandRecorder(g) as rec:
                clip = make_bound_nuisance_clip(s, c, n_color, g)
            yield i, s, c, clip, rec.draws
        else:
            yield i, s, c, make_bound_nuisance_clip(s, c, n_color, g), None


def to_uint8(clip: torch.Tensor) -> torch.Tensor:
    return (clip.clamp(0, 1) * 255.0).round().to(torch.uint8)


def clip_checksum(clip: torch.Tensor) -> str:
    return hashlib.sha256(to_uint8(clip).numpy().tobytes()).hexdigest()


def write_factors_sidecar(
    store_dir: Path, params: dict, shapes: list[int], colors: list[int], checksums: dict[str, str], **extra
) -> None:
    payload = {
        "clipset": CLIPSET,
        "seed": params["seed"],
        "n_shape": params["n_shape"],
        "n_color": params["n_color"],
        "per_cell": params["per"],
        "count": len(shapes),
        "checksum_first": checksums["first"],
        "checksum_last": checksums["last"],
        "shape": shapes,
        "color": colors,
    }
    payload.update(extra)
    (store_dir / "factors.json").write_text(json.dumps(payload, indent=2))


def scene_program_vector(
    shape: int, color: int, n_shape: int, n_color: int, draws: list[float]
) -> torch.Tensor:
    if len(draws) != N_SCALAR_DRAWS:
        raise ValueError(f"expected {N_SCALAR_DRAWS} scalar draws, got {len(draws)}")
    v = torch.zeros(n_shape + n_color + N_SCALAR_DRAWS)
    v[shape] = 1.0
    v[n_shape + color] = 1.0
    v[n_shape + n_color :] = torch.tensor(draws)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT4 programmatic scene-program featurizer (clip-identity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/at4_programmatic_features.json")
    ap.add_argument("--force", action="store_true", help="skip the post-encode guard (encoder lane free)")
    a = ap.parse_args(argv)
    guard_post_encode(a.force)
    params = read_encode_params(seed=a.seed)
    n = params["n_shape"] * params["n_color"] * params["per"]
    dim = params["n_shape"] + params["n_color"] + N_SCALAR_DRAWS

    store = LatentStore.create(
        Path(a.cache_root),
        "programmatic_reference",
        feat_shape=(dim,),
        capacity=n,
        key_dim=dim,
        has_labels=True,
    )
    shapes, colors, checks = [], [], {}
    t0 = time.perf_counter()
    for i, s, c, clip, draws in iter_bound_nuisance_clips(
        params["n_shape"], params["n_color"], params["per"], params["seed"], record=True
    ):
        if i == 0:
            checks["first"] = clip_checksum(clip)
        if i == n - 1:
            checks["last"] = clip_checksum(clip)
        z = scene_program_vector(s, c, params["n_shape"], params["n_color"], draws)
        store.write_batch(i, z[None].numpy(), z[None].numpy(), torch.tensor([s]).numpy())
        shapes.append(s)
        colors.append(c)
        del clip
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"featurized {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    store.finalize()
    write_factors_sidecar(store.root, params, shapes, colors, checks, source="featurize_programmatic")

    y_shape, y_color = torch.tensor(shapes), torch.tensor(colors)
    x = store.latents()
    probe_s = linear_probe(x, y_shape, classification=True, epochs=200, seed=params["seed"])
    probe_c = linear_probe(x, y_color, classification=True, epochs=200, seed=params["seed"])
    out = {
        "experiment": "at4_programmatic_features",
        "clipset": CLIPSET,
        "params": params,
        "n_clips": n,
        "feature_dim": dim,
        "store": str(store.root),
        "clip_checksums": checks,
        "shape_probe_acc": round(probe_s["score"], 4),
        "color_probe_acc": round(probe_c["score"], 4),
        "chance_shape": round(probe_s["chance"], 4),
        "seconds": round(time.perf_counter() - t0, 1),
        "note": "ground-truth scene program, the AT4 ceiling column: separability here is by "
        "construction and never a substrate claim",
    }
    text = json.dumps(out, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
