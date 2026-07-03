#!/usr/bin/env python
"""DR1 (Process B, Studio-only): curate REAL bound-attribute natural video and encode it past the
128-clip laptop clamp into a full V-JEPA latent store, so the compositional-binding and abstraction
probes finally run on real bound-attribute states with MANY genuine composable factors instead of the
two-factor (shape, color) programmatic clipset.

HARDENED (carries the five rounds of laptop learnings). The Studio session should only need to point
--source at the curated corpus and scale the leg ranges. Every acceptance gate and control the laptop
mandated is wired here and RUNS before any encode is spent:

  1. MULTI-FACTOR COMPOSABLE CURATION (expand plan Track B). The laptop's abstraction ceiling (6) was
     the two-composable-factor wall: the clipset had exactly one categorical factor (shape) plus color.
     This script curates the four composable factors the plan names: object identity, count, relation,
     action. A cell folder names EVERY factor: <object>-<count>-<relation>-<action> (a configurable
     schema via --factors). assert_bound_and_stocked asserts every folder carries the full factor tuple
     (no single-factor cell), each factor has >= 2 values, and every cell clears the per-cell floor.

  2. PAIRED REAL-CAPTION ACCEPTANCE GATE (expand plan, mandated pre-encode). The laptop killed the
     cross-family (vision<->language) abstraction because the synthetic pixel-text was SHAPE-BLIND (it
     carried color, not shape; see mop_a6_residual_alignment.py provenance). Here the paired perspective
     is a REAL caption per clip (captions.json sidecar) that must CARRY each composable factor. The
     preregistered acceptance gate assert_caption_recoverable fits a single linear probe on a cheap,
     label-free caption featurization and REFUSES the encode unless EVERY factor is recoverable above
     chance+ACCEPT_MARGIN on a held-out split. This is now WIRED into main() as a hard pre-encode gate
     (the pre-authored version defined the gate but never called it: a dead gate, fixed here). A tie is a
     NULL: the encode is refused rather than tuned toward a pass.

  3. A6 RESIDUALIZED-ALIGNMENT GUARD (expand plan; mop_a6_residual_alignment.py). A cross-modal
     caption<->vision topology "win" must not be the retired nuisance-geometry confound. The --a6-guard
     pass, run AFTER the encode over the merged store + captions, projects out the named factors and the
     spatiotemporal nuisance (train-fit only) and tests whether the residual vision<->caption neighbor
     topology still clears a permutation floor with a stable sign. A survivor there is genuine shared
     structure; a tie is a null. This guard needs the encoded store, so it is a Studio-time pass, but the
     preregistered rule and the residualization machinery ship here so the Studio only flips the flag.

  4. RAM + ENCODER-LANE GUARDS (unchanged, validated). A hard >= 32GB free-RAM guard refuses the 18GB
     laptop pool by accident; the one-encoder-at-a-time pgrep guard refuses to start while another
     encoder lane is alive.

WHY STUDIO (not the laptop): the probes' bite depends on the factors being genuinely BOUND in the
pixels and on enough clips per cell for a non-degenerate hold-out. The laptop path clamps the real-video
encode at 128 clips (the 18GB pool cannot hold a longer 64x256x256 ViT-L forward queue), below the
per-cell floor. The Studio box has the RAM and encoder throughput to encode the full curated set in
resumable legs; this script is that encode. It NEVER runs on the laptop.

PREREGISTERED NULL (fixed here before any real-video result exists): on real composable-factor video, a
probe trained to read a CONJUNCTION (full cell identity) from V-JEPA features does no better, outside seed
spread, than the FACTORIZED baseline that predicts each factor independently and multiplies. The null is
REJECTED only if the conjunction probe's held-out cell accuracy beats the factorized product baseline
with a seed-CI lower bound above zero AND a consistent per-seed sign. A square latent projection is NOT an
admissible baseline; the factorized-product predictor is the honest floor. This script only CURATES and
ENCODES (and runs the pre-encode gates + the post-encode A6 guard); the conjunction/abstraction verdicts
live in the scripts/compositional_binding_probe.py and scripts/mop_abstraction_systematicity.py consumers,
which read this store.

CURATION CONTRACT enforced before any encode:
  - source layout is <source>/<cell>/<clip>.mp4 where <cell> names EVERY composable factor joined by
    CELL_DELIM (default '-'), e.g. dog-2-left_of-running. validate_source checks the folders exist and
    are non-empty; the folder-name parse asserts the full factor tuple so no unbound cell sneaks in.
  - a per-cell count floor (--min-per-cell) is asserted so no cell is too thin for a hold-out.
  - a per-clip captions.json ({clip_stem: caption}) must exist and cover every clip, and every factor
    must be label-free-recoverable from the captions before the encode is spent.
  - the sorted-folder->index label_map and a per-cell manifest are persisted beside the store.

RESUMABLE PER-CLIP-RANGE LEGS: --start/--end select a half-open clip-index range over the flattened
sorted clip list, so a long encode runs as several bounded legs (each a separate guarded process). Each
leg writes to its own store shard data/cache/<name>/leg_<start>_<end>; a final --merge pass stitches the
shards. A leg refuses to overwrite a finished shard unless --force is given.

Usage (Studio):
  # 1. pre-encode gate + legs (gate runs automatically inside each leg before any byte is decoded)
  python scripts/studio/dr1_curate_bound_video.py --source /data/comp_video --start 0   --end 256
  python scripts/studio/dr1_curate_bound_video.py --source /data/comp_video --start 256 --end 512
  # 2. stitch
  python scripts/studio/dr1_curate_bound_video.py --source /data/comp_video --merge
  # 3. cross-modal nuisance guard over the merged store (Studio-time; needs the encoded latents)
  python scripts/studio/dr1_curate_bound_video.py --source /data/comp_video --a6-guard

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

# The composable factors the expand plan needs (Track B). A cell folder names EVERY factor in this order,
# joined by CELL_DELIM. Override with --factors to curate a different composable schema.
DEFAULT_FACTORS = ("object", "count", "relation", "action")
CELL_DELIM = "-"  # <object>-<count>-<relation>-<action>; '-' never appears inside a factor value.
CAPTIONS_NAME = "captions.json"  # per-clip paired real captions: {clip_stem: caption}

# =========================== PREREGISTERED ACCEPTANCE CRITERION ============================
# (fixed here IN CODE, before any real-video encode is spent; mirrors the shapecap kill-switch
# in scripts/cache_qwen_shapecap.py). A tie is a NULL: a factor that only reaches chance FAILS the
# criterion, and the encode is refused rather than tuned toward a pass.
#
# WHY: the Studio V-JEPA encode is expensive and one-shot. Before spending it, EVERY composable factor
# (object, count, relation, action, read off the <cell> folder) must be LABEL-FREE-RECOVERABLE from the
# PAIRED REAL CAPTION alone: a single linear probe on a cheap, label-free, deterministic featurization of
# each clip's caption must beat chance on a HELD-OUT split. If the caption cannot even carry a factor
# label-free, then no downstream cross-modal (vision<->language) result on these clips can be attributed
# to the substrate rather than to the caption pipeline, and the encode is a waste. This is exactly the
# wall the laptop hit: the synthetic pixel-text was shape-blind, so the cross-family abstraction result
# was uninterpretable. This gate REFUSES to repeat that with real video.
#
# THRESHOLD (preregistered, not tuned): the probe's held-out accuracy must exceed chance by at least
# ACCEPT_MARGIN on EVERY factor. chance = 1 / n_classes for that factor. Equality with (or below)
# chance + margin is a NULL and REFUSES the encode.
ACCEPT_MARGIN = 0.10  # held-out accuracy must beat chance by this margin (mirrors linear_probe's)
ACCEPT_PROBE_SEED = 0  # fixed split/seed so the acceptance decision is deterministic
# ==========================================================================================

# =========================== A6 RESIDUAL-ALIGNMENT GUARD (preregistered) ===================
# Ported from scripts/mop_a6_residual_alignment.py. A cross-modal caption<->vision topology alignment is
# only genuine shared structure if it SURVIVES projecting out the named composable factors AND the
# spatiotemporal nuisance (train-fit only, no test leakage). Verdict rule (fixed here, a tie is a null):
#   a pair "shares structure" in the minus_all condition iff the learned-vs-permuted neighbor-recall
#   delta has a seed-CI lower bound strictly > 0 AND a consistent per-seed sign (no flip). Anything at or
#   below the permutation floor COLLAPSES (nuisance-geometry confound, exactly the retired failure mode).
A6_RANK = 32  # primary rank (al2 PRIMARY_RANK)
A6_KNN_K = 10  # al2 KNN_K
A6_N_PERM = 20  # permuted-pairing floor draws per seed (al2 N_PERM)
A6_SEEDS = tuple(range(8))  # 8 seeds, matched to the laptop abstraction runs
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


def parse_cell(folder: str, factors: tuple[str, ...]) -> dict[str, str]:
    """A curated folder names a composable cell as <f0>-<f1>-...-<fN> in `factors` order; return
    {factor: value}. Raises if the name does not carry EVERY factor (guards against an unbound
    single-factor folder sneaking in, the two-factor wall the laptop hit)."""
    parts = folder.split(CELL_DELIM)
    if len(parts) != len(factors) or not all(parts):
        raise ValueError(
            f"cell folder {folder!r} does not name all {len(factors)} composable factors "
            f"{factors!r} joined by {CELL_DELIM!r} (e.g. 'dog-2-left_of-running'). Bound composable "
            "curation needs EVERY factor in every folder name, never a single-factor folder. Use "
            "underscores inside a factor value (e.g. 'left_of'); the delimiter is reserved."
        )
    return dict(zip(factors, parts, strict=True))


def assert_bound_and_stocked(manifest: dict, min_per_cell: int, factors: tuple[str, ...]) -> dict:
    """Validate every class folder is a full composable cell and clears the per-cell count floor. Every
    factor must take >= 2 values (else that factor cannot be tested). Returns a per-cell manifest
    {cell: {**factor_values, count}} for the sidecar."""
    cells: dict[str, dict] = {}
    thin: list[tuple[str, int]] = []
    values: dict[str, set] = {f: set() for f in factors}
    for cell, count in manifest["per_class"].items():
        parsed = parse_cell(cell, factors)
        cells[cell] = {**parsed, "count": count}
        for f in factors:
            values[f].add(parsed[f])
        if count < min_per_cell:
            thin.append((cell, count))
    degenerate = [f for f in factors if len(values[f]) < 2]
    if degenerate:
        raise ValueError(
            f"composable factors with < 2 values cannot be tested: {degenerate} "
            f"(value counts { {f: len(values[f]) for f in factors} }). Curate at least 2 values per "
            "factor, or drop the degenerate factor from --factors."
        )
    if thin:
        raise ValueError(
            f"cells below the per-cell floor {min_per_cell}: {thin}. Curate more clips or lower "
            "--min-per-cell (with a note that the hold-out is thin)."
        )
    return cells


def load_captions(source: str | Path) -> dict[str, str]:
    """Load the paired real captions sidecar {clip_stem: caption} from <source>/captions.json. Raises a
    clear SystemExit if it is missing (the paired-language perspective is mandatory: the cross-family
    abstraction gate cannot run without it, and the laptop's cross-modal null came from a caption side
    that did not carry the attribute)."""
    p = Path(source) / CAPTIONS_NAME
    if not p.exists():
        raise SystemExit(
            f"missing paired-caption sidecar {p}. DR1 needs a real caption per clip that CARRIES the "
            "composable factors (unlike the shape-blind synthetic pixel-text). Provide "
            f"{CAPTIONS_NAME} as a JSON map {{clip_stem: caption}} covering every clip."
        )
    caps = json.loads(p.read_text())
    if not isinstance(caps, dict) or not caps:
        raise SystemExit(f"{p} must be a non-empty JSON object {{clip_stem: caption}}")
    return {str(k): str(v) for k, v in caps.items()}


def _caption_features(captions: list[str], dim: int = 256) -> torch.Tensor:
    """Cheap, LABEL-FREE, deterministic featurization of a caption: a fixed-width hashed character-
    trigram bag (no learned weights, no external model, no labels). Deliberately weak, so a probe
    clearing chance on it means the factor is recoverable from the caption TEXT itself, not from some
    heavy encoder. Returns a [n, dim] float tensor.

    Determinism note: Python's builtin hash() is salted per-process (PYTHONHASHSEED), which would make
    the acceptance decision non-reproducible across runs. We use a fixed, salt-free trigram hash instead
    so ACCEPT_PROBE_SEED alone pins the decision."""
    import torch

    feats = torch.zeros(len(captions), dim)
    for i, cap in enumerate(captions):
        s = cap.lower()
        for j in range(len(s) - 2):
            tri = s[j : j + 3]
            feats[i, _stable_hash(tri) % dim] += 1.0
    norms = feats.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return feats / norms


def _stable_hash(s: str) -> int:
    """Deterministic, process-salt-free string hash (FNV-1a 32-bit). Replaces builtin hash() so the
    acceptance decision is reproducible regardless of PYTHONHASHSEED."""
    h = 2166136261
    for ch in s.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def caption_recoverability(captions: list[str], labels: list[int], seed: int = ACCEPT_PROBE_SEED) -> dict:
    """PREREGISTERED acceptance probe: fit a single linear layer on cheap label-free caption features to
    predict a factor label on a HELD-OUT split. Returns {score, chance, margin, passed}. passed is True
    only if held-out accuracy beats chance by ACCEPT_MARGIN (a tie is a NULL)."""
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


def assert_caption_recoverable(captions: list[str], cells: list[str], factors: tuple[str, ...]) -> dict:
    """ACCEPTANCE GATE the Studio run MUST pass before spending the encode. Given one caption per clip and
    its composable <cell>, verify EVERY factor clears the preregistered above-chance floor on a held-out
    probe. Raises SystemExit (refusing the encode) if any factor is at or below the chance+margin floor.
    Returns the per-factor acceptance report on success."""
    if len(captions) != len(cells):
        raise ValueError(f"captions ({len(captions)}) and cells ({len(cells)}) must be 1:1 per clip")
    parsed = [parse_cell(c, factors) for c in cells]
    report: dict[str, dict] = {}
    for f in factors:
        vals = sorted({p[f] for p in parsed})
        idx = {v: i for i, v in enumerate(vals)}
        report[f] = caption_recoverability(captions, [idx[p[f]] for p in parsed])
    failed = [f for f, r in report.items() if not r["passed"]]
    if failed:
        raise SystemExit(
            "ACCEPTANCE CRITERION FAILED: the paired real captions do not carry "
            f"{failed} label-free above chance+{ACCEPT_MARGIN} on a held-out probe "
            f"(report={report}). Those composable factors are not recoverable from the caption, so a "
            "downstream cross-modal (vision<->language) result would be uninterpretable, exactly the "
            "wall the laptop hit with shape-blind synthetic text. Refusing to spend the encode; fix the "
            "caption pipeline (this is a preregistered NULL, not tuned toward a pass)."
        )
    return report


def _clip_stems_in_leg(source, factors, min_per_cell, start, end) -> tuple[list[str], list[str]]:
    """Return (clip_stems, cells) for the clips whose flattened sorted index falls in [start, end). The
    order MUST match iter_video_clips' walk (sorted class folders, then sorted files). Used to pair
    captions and cells to the exact clips a leg will encode, so the acceptance gate probes the same
    subset (not a mismatched superset). No decode happens here; it is a filesystem walk only."""
    from mop.substrate.video import list_class_files

    _classes, files = list_class_files(source)  # sorted class folders, then sorted files
    stems: list[str] = []
    cells: list[str] = []
    for idx, (f, _label) in enumerate(files):
        if idx >= end:
            break
        if idx >= start:
            stems.append(Path(f).stem)
            cells.append(Path(f).parent.name)
    return stems, cells


def run_acceptance_gate(
    source: str, factors: tuple[str, ...], min_per_cell: int, start: int, end: int
) -> dict:
    """Wire the preregistered caption-recoverability gate: validate curation, load the paired captions,
    pair them to the leg's clips, and REFUSE the encode unless every composable factor is recoverable.
    Returns the acceptance report (also persisted by the caller)."""
    manifest = validate_source(source)
    assert_bound_and_stocked(manifest, min_per_cell, factors)  # curation contract
    caps_map = load_captions(source)
    stems, cells = _clip_stems_in_leg(source, factors, min_per_cell, start, end)
    missing = [s for s in stems if s not in caps_map]
    if missing:
        raise SystemExit(
            f"{len(missing)} clip(s) in leg [{start},{end}) have no caption in {CAPTIONS_NAME} "
            f"(e.g. {missing[:5]}). Every clip needs a paired real caption before the encode."
        )
    captions = [caps_map[s] for s in stems]
    report = assert_caption_recoverable(captions, cells, factors)
    print(
        "ACCEPTANCE GATE PASSED: every composable factor is label-free-recoverable from the paired "
        f"real captions above chance+{ACCEPT_MARGIN}. "
        + " ".join(f"{f}={report[f]['score']}(chance {report[f]['chance']})" for f in factors),
        flush=True,
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
    factors: tuple[str, ...],
    force: bool,
) -> dict:
    """Encode the half-open clip-index range [start, end) into its own store shard. The preregistered
    caption-recoverability acceptance gate runs FIRST and refuses the encode on a NULL. The real V-JEPA
    encoder is then loaded once; iter_video_clips streams the curated clips; cache_latents writes the
    shard."""
    # PRE-ENCODE GATE (wired here, hard): curation contract + paired-caption recoverability. Refuses the
    # encode before any byte is decoded if a composable factor is not carried by the caption.
    accept_report = run_acceptance_gate(source, factors, min_per_cell, start, end)

    manifest = validate_source(source)
    cells = assert_bound_and_stocked(manifest, min_per_cell, factors)
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
    stream = _sliced_clip_stream(source, fpc, res, start, end, hashes)
    total = max(0, end - start)
    store = cache_latents(
        enc, stream, cache_root, name, total=total, device=dev, result_tag=f"dr1_comp_leg_{start}_{end}"
    )
    write_label_map(cache_root / name, manifest["label_map"])
    # The store's clip order (walk = sorted class folders then sorted files), so the A6 cross-modal guard
    # can pair captions and residualize named factors 1:1 against the latent rows. clip_stems is the row
    # order; clip_cells is the parallel per-clip cell folder.
    clip_stems, clip_cells = _clip_stems_in_leg(source, factors, min_per_cell, start, end)
    sidecar = {
        "leg": [start, end],
        "n_encoded": len(store),
        "factors": list(factors),
        "cells": cells,
        "clip_stems": clip_stems,
        "clip_cells": clip_cells,
        "clip_hashes": hashes,
        "backend": backend,
        "acceptance_report": accept_report,
        "curation": f"composable real video, <{CELL_DELIM.join(factors)}> folders",
    }
    (cache_root / name / "cells.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    # Persist the two consumer-shaped sidecars beside THIS leg store (a leg dir IS a LatentStore), so a
    # per-leg A6 guard works; merge_shards concatenates them across legs for a merged store. clip_stems.json
    # is a list (store-row order); clip_cells.json is a {stem: cell} map, exactly what the guard reads.
    (cache_root / name / "clip_stems.json").write_text(json.dumps(clip_stems, indent=2))
    (cache_root / name / "clip_cells.json").write_text(
        json.dumps(dict(zip(clip_stems, clip_cells, strict=True)), indent=2, sort_keys=True)
    )
    log = {
        "shard": str(cache_root / name),
        "leg": [start, end],
        "n_encoded": len(store),
        "backend": backend,
        "acceptance_passed": True,
        "valid": backend == "vjepa_hf",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return log


def _sliced_clip_stream(source, fpc, res, start, end, hashes):
    """Yield only clips whose flattened index falls in [start, end). Wraps iter_video_clips (batch=1 so
    the index is exact) and drops out once end is reached, so a leg touches only its range."""
    stream = iter_video_clips(source, frames_per_clip=fpc, res=res, batch=1, hashes_out=hashes)
    for idx, (x, y) in enumerate(stream):
        if idx >= end:
            break
        if idx >= start:
            yield x, y


def merge_shards(base_name: str) -> dict:
    """Report the finished leg shards for base_name and the merge plan. Validate the legs are contiguous
    and label-consistent and emit a manifest the probe consumer reads. (No encoder is loaded here.)"""
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
    # Concatenate the per-leg clip order (in sorted-leg order = merged store-row order) into store-root
    # sidecars, so the A6 cross-modal guard can run against the merged store. Both consumer-shaped:
    # clip_stems.json a list, clip_cells.json a {stem: cell} map.
    order_ok = bool(legs) and all("clip_stems" in s_ and "clip_cells" in s_ for s_ in legs)
    if order_ok:
        all_stems: list[str] = []
        stem_to_cell: dict[str, str] = {}
        for s_ in legs:
            for st, ce in zip(s_["clip_stems"], s_["clip_cells"], strict=True):
                all_stems.append(st)
                stem_to_cell[st] = ce
        (cache_root / "clip_stems.json").write_text(json.dumps(all_stems, indent=2))
        (cache_root / "clip_cells.json").write_text(json.dumps(stem_to_cell, indent=2, sort_keys=True))
    manifest = {
        "base": str(cache_root),
        "legs": ranges,
        "contiguous": contiguous,
        "total_encoded": total,
        "backends": sorted({s_["backend"] for s_ in legs}),
        "factors": legs[0]["factors"] if legs else [],
        "clip_order_persisted": order_ok,
    }
    (cache_root / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if not order_ok:
        manifest["clip_order_note"] = (
            "some legs predate the stem-sidecar wiring; re-encode them so clip_stems.json / clip_cells.json "
            "cover every clip before the A6 guard runs"
        )
    if not contiguous:
        manifest["warning"] = "leg ranges are not contiguous; re-encode the gap before the probe runs"
    return manifest


# ------------------------------------------------------------------------------------------
# A6 RESIDUALIZED-ALIGNMENT GUARD (ported from scripts/mop_a6_residual_alignment.py). Nuisance-geometry
# control for a cross-modal caption<->vision result. Needs the encoded store, so it is a Studio-time
# pass; the machinery ships here so the Studio only flips --a6-guard.
# ------------------------------------------------------------------------------------------
def _standardize(train, other):
    import torch  # noqa: F401

    mu, sd = train.mean(dim=0), train.std(dim=0) + 1e-6
    return (train - mu) / sd, (other - mu) / sd


def _ridge_fit(a, b, lam: float = 1.0):
    import torch

    n, d = a.shape
    reg = lam * torch.eye(d)
    return torch.linalg.solve(a.T @ a + reg, a.T @ b)


def _rank_truncate(w, rank: int):
    import torch

    u, s, vt = torch.linalg.svd(w, full_matrices=False)
    r = min(rank, s.shape[0])
    return (u[:, :r] * s[:r]) @ vt[:r]


def _knn_recall(pred, target, k: int) -> float:
    import torch

    def _neighbors(z):
        d = torch.cdist(z, z)
        d.fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).indices

    np_, nt = _neighbors(pred), _neighbors(target)
    hits = sum(len(set(np_[i].tolist()) & set(nt[i].tolist())) for i in range(pred.shape[0]))
    return hits / (pred.shape[0] * k)


def _project_out(x, design, tr):
    import torch

    beta = torch.linalg.pinv(design[tr]) @ x[tr]
    return x - design @ beta


def a6_residual_guard(store_dir: Path, captions_map: dict, factors: tuple[str, ...]) -> dict:
    """Preregistered cross-modal nuisance-geometry control. Load the encoded vision latents and the
    paired caption features (same clips, same order), project out the named composable factors AND the
    spatiotemporal nuisance (train-fit only), then test whether the residual vision<->caption neighbor
    topology still clears a permutation floor with a stable sign. A survivor is genuine shared structure;
    a tie/below-floor COLLAPSES (nuisance-geometry confound). Returns the verdict report.

    STUDIO-TIME NOTE: this reads the encoded store, so it runs after the merge. If a per-clip nuisance
    array is not persisted with the store (the real-video path does not synthesize the 6-factor nuisance
    the programmatic clipset had), the guard runs the label-partition conditions (raw / minus_factors)
    and flags that the explicit nuisance-design condition needs the nuisance sidecar. It never fakes the
    nuisance condition."""
    import numpy as np
    import torch

    from mop.substrate import LatentStore

    if not (store_dir / "meta.json").exists():
        raise SystemExit(
            f"no encoded store at {store_dir}; run the legs + --merge first, then --a6-guard. The "
            "residual-alignment guard needs the encoded vision latents."
        )
    store = LatentStore.open(store_dir)
    xv = store.latents().float()

    # Pair captions to the store's clip order via the persisted label_map + cells sidecar. The store rows
    # are in the flattened sorted-clip order; we reconstruct the (stem, cell) order the same way the
    # encode walked them so captions align 1:1. If a stems sidecar is present we use it; else we rebuild.
    stems_p = store_dir / "clip_stems.json"
    if stems_p.exists():
        stems = json.loads(stems_p.read_text())
    else:
        raise SystemExit(
            f"no clip_stems.json beside {store_dir}; the merge pass must persist the store's clip order "
            "for the A6 guard to pair captions to latents. Re-run --merge with this hardened script "
            "(which writes clip_stems.json), or provide the stem order."
        )
    if len(stems) != xv.shape[0]:
        raise SystemExit(
            f"store has {xv.shape[0]} rows but {len(stems)} clip stems; cannot pair captions to latents."
        )
    captions = [captions_map[s] for s in stems if s in captions_map]
    if len(captions) != xv.shape[0]:
        raise SystemExit("some store clips have no caption; the A6 guard needs a caption per latent.")
    xt = _caption_features(captions)

    nuis_p = store_dir / "nuisance.npy"
    has_nuisance = nuis_p.exists()
    nuis = torch.tensor(np.load(nuis_p)).float() if has_nuisance else None

    # We need the named-factor labels per clip for the minus_factors / minus_all designs. Reconstruct
    # them from the label_map + cells sidecar of the store.
    factor_labels = _factor_labels_for_store(store_dir, stems, factors)

    conditions = ["raw", "minus_factors"] + (["minus_nuisance", "minus_all"] if has_nuisance else [])
    out = {}
    for mode in conditions:
        deltas = []
        for seed in A6_SEEDS:
            deltas.append(_a6_one(xv, xt, factor_labels, nuis, mode, seed))
        from mop.diagnostics.riskcov import seed_ci, sign_flip_report

        ci = seed_ci(deltas)
        flips = sign_flip_report(deltas)
        survives = bool(ci["lo"] > 0.0 and flips["consistent_sign"] == 1)
        out[mode] = {"delta_ci": ci, "sign_flips": flips, "survives": survives}

    decisive = out.get("minus_all", out["minus_factors"])
    verdict = (
        "GENUINE-SHARED-STRUCTURE: cross-modal caption<->vision topology survives projecting out the "
        "named factors" + (" and the nuisance" if has_nuisance else "") + " (seed-CI lo > 0, no sign "
        "flip). Not a nuisance-geometry artifact."
        if decisive["survives"]
        else "COLLAPSES: cross-modal topology does not clear the permutation floor after residualization; "
        "any raw alignment is the nuisance-geometry / label-partition confound, not shared abstract code."
    )
    return {
        "guard": "a6_residual_alignment (cross-modal caption<->vision nuisance control)",
        "store": str(store_dir),
        "has_nuisance_sidecar": has_nuisance,
        "conditions": out,
        "decisive_condition": "minus_all" if has_nuisance else "minus_factors",
        "verdict": verdict,
        "studio_todo": None
        if has_nuisance
        else (
            "no nuisance.npy beside the store: the explicit spatiotemporal-nuisance conditions "
            "(minus_nuisance / minus_all) were SKIPPED. For the full A6 guard, persist a per-clip "
            "nuisance array (camera motion, object position/scale, etc.) with the encode, or accept the "
            "label-partition-only conditions as a weaker control."
        ),
    }


def _factor_labels_for_store(store_dir: Path, stems: list[str], factors: tuple[str, ...]):
    """Reconstruct per-clip factor integer labels {factor: LongTensor[N]} from the store's cells sidecar
    and clip stems, so the A6 designs can one-hot them."""
    import torch

    # clip_cells.json maps stem -> cell folder name (persisted by the merge pass alongside clip_stems).
    cc_p = store_dir / "clip_cells.json"
    if not cc_p.exists():
        raise SystemExit(
            f"no clip_cells.json beside {store_dir}; the merge pass must persist stem->cell so the A6 "
            "guard can residualize the named factors. Re-run --merge with this hardened script."
        )
    stem_to_cell = json.loads(cc_p.read_text())
    parsed = [parse_cell(stem_to_cell[s], factors) for s in stems]
    labels = {}
    for f in factors:
        vals = sorted({p[f] for p in parsed})
        idx = {v: i for i, v in enumerate(vals)}
        labels[f] = torch.tensor([idx[p[f]] for p in parsed], dtype=torch.long)
    return labels


def _a6_one(xv, xt, factor_labels, nuis, mode, seed) -> float:
    """One seed of the residual-alignment delta (learned neighbor-recall minus permuted-pairing floor)
    for the vision->caption map under a residualization mode."""
    import torch
    import torch.nn.functional as F

    n = xv.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]

    def _design():
        cols = []
        if mode in ("minus_factors", "minus_all"):
            for lab in factor_labels.values():
                cols.append(F.one_hot(lab, int(lab.max()) + 1).float())
        if mode in ("minus_nuisance", "minus_all") and nuis is not None:
            cols.append(torch.cat([torch.ones(n, 1), nuis], dim=1))
        return torch.cat(cols, dim=1) if cols else None

    a, b = xv.clone(), xt.clone()
    design = _design()
    if design is not None:
        a = _project_out(a, design, tr)
        b = _project_out(b, design, tr)

    atr, ate = _standardize(a[tr], a[te])
    btr, bte = _standardize(b[tr], b[te])
    w = _rank_truncate(_ridge_fit(atr, btr), A6_RANK)
    learned = _knn_recall(ate @ w, bte, k=A6_KNN_K)

    floors = []
    for _ in range(A6_N_PERM):
        shuf = torch.randperm(cut, generator=g)
        w_s = _rank_truncate(_ridge_fit(atr, btr[shuf]), A6_RANK)
        floors.append(_knn_recall(ate @ w_s, bte, k=A6_KNN_K))
    permuted = sum(floors) / len(floors)
    return learned - permuted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DR1 Studio: curate+encode real composable-factor video")
    ap.add_argument("--source", help="dir of <cell>/<clip>.mp4 + captions.json (required unless --merge)")
    ap.add_argument("--name", default="vjepa2_vitl_comp_video", help="base cache name under data/cache")
    ap.add_argument("--factors", default=",".join(DEFAULT_FACTORS), help="comma-list composable factors")
    ap.add_argument("--start", type=int, default=0, help="leg range start (flattened clip index)")
    ap.add_argument("--end", type=int, default=256, help="leg range end (exclusive)")
    ap.add_argument("--min-per-cell", type=int, default=16, help="per-cell clip floor for a hold-out")
    ap.add_argument("--merge", action="store_true", help="stitch finished legs into a merge manifest")
    ap.add_argument("--a6-guard", action="store_true", help="run the cross-modal nuisance guard (Studio)")
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="run ONLY the pre-encode caption-recoverability acceptance gate (no encode; dry-run check)",
    )
    ap.add_argument("--force", action="store_true", help="re-encode a finished shard")
    a = ap.parse_args(argv)
    factors = tuple(f for f in a.factors.split(",") if f)

    if a.gate_only:
        # The gate is a filesystem + text probe only; it does NOT need Studio RAM, so it can dry-run
        # anywhere (including the laptop) to confirm the caption pipeline before the Studio encode.
        if not a.source:
            print("FAIL: --source is required for --gate-only")
            return 1
        report = run_acceptance_gate(a.source, factors, a.min_per_cell, a.start, a.end)
        print(json.dumps({"acceptance_gate": "PASSED", "report": report}, indent=2))
        return 0

    assert_studio_ram()  # Studio-only: refuses on the laptop pool (encode + guard are heavy)
    if a.merge:
        out = merge_shards(a.name)
        print(json.dumps(out, indent=2, default=str))
        return 0
    if a.a6_guard:
        if not a.source:
            print("FAIL: --source is required for --a6-guard (to load captions.json)")
            return 1
        caps = load_captions(a.source)
        store_dir = REPO_ROOT / "data" / "cache" / a.name
        out = a6_residual_guard(store_dir, caps, factors)
        print(json.dumps(out, indent=2, default=str))
        return 0
    if not a.source:
        print("FAIL: --source is required unless --merge")
        return 1
    if a.end <= a.start:
        print(f"FAIL: empty leg range [{a.start}, {a.end})")
        return 1
    assert_encoder_lane_free()  # one encoder at a time (shared pgrep guard)
    log = encode_leg(a.source, a.name, a.start, a.end, a.min_per_cell, factors, a.force)
    print(json.dumps(log, indent=2, default=str))
    return 0 if log["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
