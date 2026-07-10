#!/usr/bin/env python
"""THE decisive substrate experiment: does pooling destroy compositional/spatial structure that the
dense pre-pool tokens keep? The whole corpus lands most abstraction/binding tests as taxonomy-3 bounds,
but we never separated two explanations: (B) the frozen V-JEPA 2 substrate is genuinely bounded, vs
(C) MEAN-POOLING throws the structure away before the shell ever sees it. This probe collapses that
ambiguity on real encoder geometry.

One real encoder forward per clip yields the full [8192, 1024] pre-pool tokens. From the SAME tokens we
form two representations of identical information content upstream:
  - POOLED: mean over all tokens -> [1024] (what the whole corpus used)
  - GRID: mean over G contiguous token groups -> [G, 1024] flattened (keeps G centroids, not 1)
We decode two independent factors of the synthetic-but-real-geometry content:
  - HUE (factor A): a GLOBAL appearance factor; mean-pooling should preserve it.
  - ORIENTATION (factor B): a SPATIAL factor; the spatial mean of a full-field grating is near-constant
    regardless of angle, so if the encoder stores orientation only in the spatial ARRANGEMENT of tokens,
    mean-pooling destroys it while the grid keeps it.
Each factor is probed on POOLED and GRID, each against a frozen-random projection control.

Verdict logic:
  - if ORIENTATION decodes much better from GRID than POOLED (and hue decodes from both): POOLING IS THE
    BOTTLENECK. The fork answer is a dense/token-level substrate (pre-pool tokens or V-JEPA 2.1 dense).
  - if ORIENTATION fails on BOTH grid and pooled: the substrate itself does not carry it, bounded, and a
    custom/plastic representation is the honest direction.
  - if both decode from POOLED already: pooling is not the bottleneck for these factors.

Usage: python scripts/dense_vs_pooled_probe.py --n-a 4 --n-b 4 --per 3 --grid 64 \
    --out runs/pre_studio/dense_vs_pooled.json
  device=cpu (MPS overflows the ViT-L attention buffer at 64-frame/256px).

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_factorized_encoder import make_factorized_clip  # noqa: E402

from mop.config import compose  # noqa: E402
from mop.devices import resolve, safe_to  # noqa: E402
from mop.diagnostics import linear_probe  # noqa: E402
from mop.diagnostics.substrate_ablation import frozen_random_projection  # noqa: E402
from mop.substrate import load_encoder  # noqa: E402


def _grid_pool(tokens: torch.Tensor, g: int) -> torch.Tensor:
    """tokens [N, D] -> [g*D] by averaging N tokens into g contiguous groups, then flattening. Keeps g
    local centroids instead of the single global mean, so any structure that survives at coarser
    granularity is retained. Robust to the exact token ordering: g contiguous groups keep SOME locality
    regardless of whether the layout is temporal-major or spatial-major."""
    n, d = tokens.shape
    per = max(1, n // g)
    groups = [tokens[i * per : (i + 1) * per].mean(0) for i in range(g)]
    return torch.stack(groups).reshape(-1)


def _probe_pair(x: torch.Tensor, y: torch.Tensor, seed: int) -> dict:
    real = linear_probe(x, y, classification=True, epochs=300, seed=seed)
    fr = linear_probe(frozen_random_projection(x, seed), y, classification=True, epochs=300, seed=seed)
    return {
        "real_acc": round(real["score"], 4),
        "frozen_random_acc": round(fr["score"], 4),
        "chance": round(real["chance"], 4),
        "delta_real_minus_fr": round(real["score"] - fr["score"], 4),
    }


def run(n_a: int, n_b: int, per: int, grid: int, seed: int) -> dict:
    cfg = compose(
        [
            "encoder=vjepa2_vitl_fpc64_256",
            "device=cpu",
            "encoder.dense=true",
            "encoder.prefer_real=true",
            "+encoder.require_real=true",
        ]
    )
    dev = resolve("cpu")
    enc = load_encoder(cfg.encoder).to(dev.device)
    backend = enc.spec.backend
    g = torch.Generator().manual_seed(seed)

    cells = [(a, b) for a in range(n_a) for b in range(n_b) for _ in range(per)]
    pooled, gridded, hue, orient = [], [], [], []
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(cells):
        clip = make_factorized_clip(a, b, n_a, n_b, g).unsqueeze(0)  # [1,T,3,H,W]
        tok = enc.encode(safe_to(clip, dev.device))  # dense: [1, N, D]
        tok = tok[0].float().cpu()  # [N, D]
        if tok.dim() == 1:
            tok = tok.unsqueeze(0)
        pooled.append(tok.mean(0))  # [D]
        gridded.append(_grid_pool(tok, grid))  # [g*D]
        hue.append(a)
        orient.append(b)
        if (i + 1) % 8 == 0 or i + 1 == len(cells):
            print(f"encoded {i + 1}/{len(cells)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    xp = torch.stack(pooled)
    xg = torch.stack(gridded)
    ya = torch.tensor(hue)
    yb = torch.tensor(orient)
    n_tokens = tok.shape[0]

    out = {
        "backend": backend,
        "n_clips": len(cells),
        "n_a_hue": n_a,
        "n_b_orientation": n_b,
        "per_cell": per,
        "n_tokens_dense": int(n_tokens),
        "grid_groups": grid,
        "pooled_dim": int(xp.shape[1]),
        "grid_dim": int(xg.shape[1]),
        "seconds": round(time.perf_counter() - t0, 1),
        "hue_global_factor": {
            "pooled": _probe_pair(xp, ya, seed),
            "grid": _probe_pair(xg, ya, seed),
        },
        "orientation_spatial_factor": {
            "pooled": _probe_pair(xp, yb, seed),
            "grid": _probe_pair(xg, yb, seed),
        },
    }

    # decisive verdict on the orientation (spatial) factor: does the grid recover it where pooling loses it
    o_pooled = out["orientation_spatial_factor"]["pooled"]["real_acc"]
    o_grid = out["orientation_spatial_factor"]["grid"]["real_acc"]
    chance = out["orientation_spatial_factor"]["pooled"]["chance"]
    grid_recovers = bool(o_grid - o_pooled > 0.1)
    pooled_already = bool(o_pooled > chance + 0.1)
    bounded = bool(o_grid <= chance + 0.1)
    out["orientation_grid_minus_pooled"] = round(o_grid - o_pooled, 4)
    if backend != "vjepa_hf":
        out["verdict"] = (
            "INVALID: encoder ran as frozen_random (real weights not loaded), rerun with real weights"
        )
    elif pooled_already:
        out["verdict"] = (
            "pooling is NOT the bottleneck for orientation: it already decodes from the pooled vector"
        )
    elif grid_recovers:
        out["verdict"] = (
            "POOLING IS THE BOTTLENECK: orientation decodes from the dense grid but not the pooled vector. "
            "The fork points to a dense/token-level substrate (pre-pool tokens or V-JEPA 2.1 dense)."
        )
    elif bounded:
        out["verdict"] = (
            "SUBSTRATE-BOUNDED: orientation fails on BOTH grid and pooled, so the frozen encoder does not "
            "carry it at all here, not just a pooling loss. Points toward a custom/plastic representation."
        )
    else:
        out["verdict"] = "ambiguous: grid gives a small but sub-threshold gain over pooled on orientation"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dense-vs-pooled decodability probe on real V-JEPA tokens")
    ap.add_argument("--n-a", type=int, default=4, help="hue values (global factor)")
    ap.add_argument("--n-b", type=int, default=4, help="orientation values (spatial factor)")
    ap.add_argument("--per", type=int, default=3, help="clips per (hue, orientation) cell")
    ap.add_argument("--grid", type=int, default=64, help="dense token groups kept (G)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = run(a.n_a, a.n_b, a.per, a.grid, a.seed)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
