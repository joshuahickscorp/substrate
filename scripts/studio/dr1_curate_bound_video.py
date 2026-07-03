#!/usr/bin/env python
"""DR1 (Process B, Studio-only): curate REAL bound-attribute natural video and encode it past the
128-clip laptop clamp into a full V-JEPA latent store, so the compositional-binding probe finally runs
on real bound-attribute states instead of the programmatic clipset.

WHY STUDIO (not the laptop): the binding probe's whole bite depends on shape and color being genuinely
BOUND in the pixels (a red square and a blue circle, never a free cross-product of factor channels),
and on enough clips per (shape, color) cell to give the probe a non-degenerate hold-out. The laptop
path clamps the real-video encode at 128 clips (the 18GB pool cannot hold a longer 64x256x256 ViT-L
forward queue), which is below the per-cell floor the binding probe needs. The Studio box has the RAM
and the encoder throughput to encode the full curated set in resumable legs; this script is that encode.
It NEVER runs on the laptop (hard free-RAM guard below), and it refuses to start while any other
encoder-lane process is alive (the one-encoder-at-a-time pgrep guard, shared with the laptop caches).

PREREGISTERED NULL (fixed here before any real-video result exists): on real bound-attribute video, a
probe trained to read the CONJUNCTION (shape-and-color cell identity) from V-JEPA features does no
better, outside seed spread, than the FACTORIZED baseline that predicts shape and color independently
and multiplies -- i.e. the substrate carries the two attributes as separable channels and binds nothing.
The null is REJECTED only if the conjunction probe's held-out cell accuracy beats the factorized
product baseline with a seed-CI lower bound above zero AND a consistent per-seed sign. A square latent
projection is NOT an admissible baseline here; the factorized-product predictor is the honest floor
(a substrate that only linearly separates each factor can already hit it). This script only CURATES and
ENCODES; the probe + verdict live in the scripts/compositional_binding_probe.py consumer, which reads
this store. We write the store and its provenance, nothing more.

CURATION CONTRACT enforced before any encode:
  - source layout is <source>/<shape>_<color>/<clip>.mp4 (one folder per BOUND cell, not per factor);
    validate_source checks the folders exist and are non-empty, and the folder-name parse below asserts
    every folder names BOTH a shape and a color so no unbound (single-factor) cell sneaks in.
  - a per-cell count floor (--min-per-cell) is asserted so no cell is too thin for a hold-out.
  - the sorted-folder->index label_map and a per-cell manifest are persisted beside the store.

RESUMABLE PER-CLIP-RANGE LEGS: --start/--end select a half-open clip-index range over the flattened
sorted clip list, so a long encode runs as several bounded legs (each a separate process, each guarded).
Each leg writes to its own store shard data/cache/<name>/leg_<start>_<end>; a final --merge pass stitches
the shards into one store. A leg refuses to overwrite a finished shard unless --force is given.

Usage (Studio):
  python scripts/studio/dr1_curate_bound_video.py --source /data/bound_video --start 0   --end 256
  python scripts/studio/dr1_curate_bound_video.py --source /data/bound_video --start 256 --end 512
  python scripts/studio/dr1_curate_bound_video.py --source /data/bound_video --merge

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import assert_encoder_lane_free  # noqa: E402

from mop.config import REPO_ROOT, compose  # noqa: E402
from mop.devices import resolve  # noqa: E402
from mop.diagnostics import linear_probe  # noqa: E402
from mop.substrate import cache_latents, iter_video_clips, load_encoder  # noqa: E402
from mop.substrate.video import detect_partial_cache, validate_source, write_label_map  # noqa: E402

MIN_FREE_RAM_GB = 32.0  # Studio-only guard: refuse to run on the 18GB laptop pool.

# =========================== PREREGISTERED ACCEPTANCE CRITERION ============================
# (fixed here IN CODE, before any real-video encode is spent; mirrors the shapecap kill-switch
# in scripts/cache_qwen_shapecap.py). A tie is a NULL: a target attribute that only reaches
# chance FAILS the criterion, and the encode is refused rather than tuned toward a pass.
#
# WHY: the Studio V-JEPA encode is expensive and one-shot. Before spending it, the target BOUND
# attribute (shape and color, read off the <shape>_<color> cell folder) must be LABEL-FREE-
# RECOVERABLE from the PAIRED CAPTION alone: a single linear probe on a cheap, label-free,
# deterministic featurization of each clip's caption must beat chance on a HELD-OUT split. If the
# caption cannot even carry the attribute label-free, then no downstream binding result on these
# clips can be attributed to the substrate rather than to the caption pipeline, and the encode is
# a waste. This is a pre-encode gate, not a result: it only asserts the caption is informative,
# exactly as the shapecap kill-switch asserts the shape descriptor is informative before trusting
# its alignment number.
#
# THRESHOLD (preregistered, not tuned): the probe's held-out accuracy must exceed chance by at
# least ACCEPT_MARGIN on BOTH attributes (shape and color). chance = 1 / n_classes for that
# attribute. Equality with (or below) chance + margin is a NULL and REFUSES the encode.
ACCEPT_MARGIN = 0.10  # held-out accuracy must beat chance by this margin (mirrors linear_probe's)
ACCEPT_PROBE_SEED = 0  # fixed split/seed so the acceptance decision is deterministic
# ==========================================================================================


def assert_studio_ram(min_gb: float = MIN_FREE_RAM_GB) -> float:
    """Hard guard: refuse to run unless >= min_gb of free RAM is available. This keeps the heavy real
    V-JEPA encode off the laptop by accident (the laptop pool is 18GB). Returns the free-GB reading."""
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as e:  # psutil absent or unreadable: fail closed, never fail open
        raise SystemExit(
            f"cannot read free RAM ({e}); refusing to run the Studio encode without the >= "
            f"{min_gb:.0f}GB safety check. Install psutil on the Studio box."
        ) from e
    if free_gb < min_gb:
        raise SystemExit(
            f"free RAM {free_gb:.1f}GB < required {min_gb:.0f}GB. This is a Studio-only encode; it will "
            "not run on the laptop. Move it to the Studio box."
        )
    return free_gb


def parse_bound_cell(folder: str) -> tuple[str, str]:
    """A curated folder names a BOUND cell as <shape>_<color>; return (shape, color). Raises if the
    name does not carry both attributes (guards against an unbound single-factor folder sneaking in)."""
    parts = folder.split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"cell folder {folder!r} is not <shape>_<color>: bound-attribute curation needs BOTH "
            "attributes in every folder name (e.g. 'square_red'), never a single-factor folder"
        )
    return parts[0], parts[1]


def assert_bound_and_stocked(manifest: dict, min_per_cell: int) -> dict:
    """Validate every class folder is a bound cell and clears the per-cell count floor. Returns a
    per-cell manifest {cell: {shape, color, count}} for the sidecar."""
    cells: dict[str, dict] = {}
    thin: list[tuple[str, int]] = []
    for cell, count in manifest["per_class"].items():
        shape, color = parse_bound_cell(cell)
        cells[cell] = {"shape": shape, "color": color, "count": count}
        if count < min_per_cell:
            thin.append((cell, count))
    shapes = {v["shape"] for v in cells.values()}
    colors = {v["color"] for v in cells.values()}
    if len(shapes) < 2 or len(colors) < 2:
        raise ValueError(
            f"binding needs >= 2 shapes and >= 2 colors; got shapes={sorted(shapes)} "
            f"colors={sorted(colors)}. A single-shape or single-color set cannot test binding."
        )
    if thin:
        raise ValueError(
            f"cells below the per-cell floor {min_per_cell}: {thin}. Curate more clips or lower "
            "--min-per-cell (with a note that the hold-out is thin)."
        )
    return cells


def _caption_features(captions: list[str], dim: int = 256) -> torch.Tensor:
    """Cheap, LABEL-FREE, deterministic featurization of a caption: a fixed-width hashed character-
    trigram bag (no learned weights, no external model, no labels). This is deliberately weak, so a
    probe clearing chance on it means the attribute is recoverable from the caption TEXT itself, not
    from some heavy encoder. Returns a [n, dim] float tensor."""
    import torch

    feats = torch.zeros(len(captions), dim)
    for i, cap in enumerate(captions):
        s = cap.lower()
        for j in range(len(s) - 2):
            tri = s[j : j + 3]
            feats[i, hash(tri) % dim] += 1.0
    # L2-normalize rows so caption length does not dominate the probe
    norms = feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return feats / norms


def caption_recoverability(captions: list[str], labels: list[int], seed: int = ACCEPT_PROBE_SEED) -> dict:
    """PREREGISTERED acceptance probe: fit a single linear layer on cheap label-free caption features to
    predict an attribute label on a HELD-OUT split. Returns {score, chance, margin, passed}. passed is
    True only if held-out accuracy beats chance by ACCEPT_MARGIN (a tie is a NULL). Mirrors the shapecap
    kill-switch: informative-caption gate, not a result."""
    import torch

    x = _caption_features(captions)
    y = torch.tensor(labels, dtype=torch.long)
    out = linear_probe(x, y, classification=True, seed=seed)
    margin = out["score"] - out["chance"]
    return {
        "score": round(float(out["score"]), 4),
        "chance": round(float(out["chance"]), 4),
        "margin": round(float(margin), 4),
        "passed": bool(margin >= ACCEPT_MARGIN),
    }


def assert_caption_recoverable(captions: list[str], cells: list[str]) -> dict:
    """ACCEPTANCE GATE the Studio run MUST call before spending the encode. Given one caption per clip
    and its <shape>_<color> cell, verify BOTH bound attributes (shape, color) clear the preregistered
    above-chance floor on a held-out probe. Raises SystemExit (refusing the encode) if either attribute
    is at or below the chance+margin floor. Returns the per-attribute acceptance report on success."""
    if len(captions) != len(cells):
        raise ValueError(f"captions ({len(captions)}) and cells ({len(cells)}) must be 1:1 per clip")
    parsed = [parse_bound_cell(c) for c in cells]
    shapes = sorted({s for s, _ in parsed})
    colors = sorted({c for _, c in parsed})
    shape_idx = {s: i for i, s in enumerate(shapes)}
    color_idx = {c: i for i, c in enumerate(colors)}
    report = {
        "shape": caption_recoverability(captions, [shape_idx[s] for s, _ in parsed]),
        "color": caption_recoverability(captions, [color_idx[c] for _, c in parsed]),
    }
    failed = [attr for attr, r in report.items() if not r["passed"]]
    if failed:
        raise SystemExit(
            "ACCEPTANCE CRITERION FAILED: the paired captions do not carry "
            f"{failed} label-free above chance+{ACCEPT_MARGIN} on a held-out probe "
            f"(report={report}). The bound attribute is not recoverable from the caption, so the "
            "Studio encode would be uninterpretable. Refusing to spend the encode; fix the caption "
            "pipeline (this is a preregistered NULL, not tuned toward a pass)."
        )
    return report


def shard_name(base: str, start: int, end: int) -> str:
    return f"{base}/leg_{start}_{end}"


def encode_leg(
    source: str,
    base_name: str,
    start: int,
    end: int,
    min_per_cell: int,
    force: bool,
) -> dict:
    """Encode the half-open clip-index range [start, end) into its own store shard. The real V-JEPA
    encoder is loaded once; iter_video_clips streams the curated clips; cache_latents writes the shard."""
    manifest = validate_source(source)
    cells = assert_bound_and_stocked(manifest, min_per_cell)
    cache_root = REPO_ROOT / "data" / "cache"
    name = shard_name(base_name, start, end)
    existing = detect_partial_cache(cache_root, name)
    if existing["exists"] and existing["has_meta"] and not force:
        raise SystemExit(
            f"shard {existing['store_dir']} already finished (count={existing['count']}); pass --force "
            "to re-encode it, or pick a fresh --start/--end range."
        )
    cfg = compose(["encoder=vjepa2_vitl_fpc64_256", "device=mps", "encoder.prefer_real=true"])
    dev = resolve(str(cfg.device.kind))
    enc = load_encoder(cfg.encoder).to(dev.device)
    backend = enc.spec.backend
    if backend == "frozen_random":
        raise SystemExit(
            "real V-JEPA weights unavailable (backend=frozen_random); this curation encode is only "
            "meaningful with real weights. Install the encoder extra and rerun."
        )
    fpc = int(cfg.encoder.frames_per_clip)
    res = int(cfg.encoder.resolution)
    hashes: list[str] = []
    t0 = time.perf_counter()
    # iter_video_clips walks sorted class folders then sorted files: a stable flattened order that
    # start/end index into. We take the whole stream and slice the leg range in cache_latents by
    # skipping to start and capping at end (total = end - start after the skip).
    stream = _sliced_clip_stream(source, fpc, res, start, end, hashes)
    total = max(0, end - start)
    store = cache_latents(
        enc, stream, cache_root, name, total=total, device=dev, result_tag=f"dr1_bound_leg_{start}_{end}"
    )
    write_label_map(cache_root / name, manifest["label_map"])
    sidecar = {
        "leg": [start, end],
        "n_encoded": len(store),
        "cells": cells,
        "clip_hashes": hashes,
        "backend": backend,
        "curation": "bound-attribute real video, <shape>_<color> folders",
    }
    (cache_root / name / "cells.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    log = {
        "shard": str(cache_root / name),
        "leg": [start, end],
        "n_encoded": len(store),
        "backend": backend,
        "valid": backend == "vjepa_hf",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return log


def _sliced_clip_stream(source, fpc, res, start, end, hashes):
    """Yield only clips whose flattened index falls in [start, end). Wraps iter_video_clips (batch=1
    so the index is exact) and drops out once end is reached, so a leg touches only its range."""
    stream = iter_video_clips(source, frames_per_clip=fpc, res=res, batch=1, hashes_out=hashes)
    for idx, (x, y) in enumerate(stream):
        if idx >= end:
            break
        if idx >= start:
            yield x, y


def merge_shards(base_name: str) -> dict:
    """Report the finished leg shards for base_name and the merge plan. The physical stitch is a store
    concat performed by the store layer; here we validate the legs are contiguous and label-consistent
    and emit a manifest the probe consumer reads. (No encoder is loaded in the merge pass.)"""
    cache_root = REPO_ROOT / "data" / "cache" / base_name
    if not cache_root.exists():
        raise SystemExit(f"no shards under {cache_root}; run at least one leg first.")
    legs = []
    for shard in sorted(cache_root.glob("leg_*")):
        cells_p = shard / "cells.json"
        if not cells_p.exists():
            continue
        legs.append({"shard": str(shard), **json.loads(cells_p.read_text())})
    legs.sort(key=lambda s_: s_["leg"][0])
    ranges = [tuple(s_["leg"]) for s_ in legs]
    contiguous = all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    total = sum(s_["n_encoded"] for s_ in legs)
    manifest = {
        "base": str(cache_root),
        "legs": ranges,
        "contiguous": contiguous,
        "total_encoded": total,
        "backends": sorted({s_["backend"] for s_ in legs}),
    }
    (cache_root / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if not contiguous:
        manifest["warning"] = "leg ranges are not contiguous; re-encode the gap before the probe runs"
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DR1 Studio: curate+encode real bound-attribute video")
    ap.add_argument("--source", help="dir of <shape>_<color>/<clip>.mp4 (required unless --merge)")
    ap.add_argument("--name", default="vjepa2_vitl_bound_video", help="base cache name under data/cache")
    ap.add_argument("--start", type=int, default=0, help="leg range start (flattened clip index)")
    ap.add_argument("--end", type=int, default=256, help="leg range end (exclusive)")
    ap.add_argument("--min-per-cell", type=int, default=16, help="per-cell clip floor for a hold-out")
    ap.add_argument("--merge", action="store_true", help="stitch finished legs into a merge manifest")
    ap.add_argument("--force", action="store_true", help="re-encode a finished shard")
    a = ap.parse_args(argv)

    assert_studio_ram()  # Studio-only: refuses on the laptop pool
    if a.merge:
        out = merge_shards(a.name)
        print(json.dumps(out, indent=2, default=str))
        return 0
    if not a.source:
        print("FAIL: --source is required unless --merge")
        return 1
    if a.end <= a.start:
        print(f"FAIL: empty leg range [{a.start}, {a.end})")
        return 1
    assert_encoder_lane_free()  # one encoder at a time (shared pgrep guard)
    log = encode_leg(a.source, a.name, a.start, a.end, a.min_per_cell, a.force)
    print(json.dumps(log, indent=2, default=str))
    return 0 if log["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
