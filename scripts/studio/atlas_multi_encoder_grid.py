#!/usr/bin/env python
"""ATLAS (Process B, Studio-only): the at-scale AT1 + AL2 grid over the FULL multi-encoder cache set. The
laptop pilots (mop_at1_grid_pilot.py, mop_al2_alignment_pilot.py) type only whatever few columns the
18GB pool could cache; this Studio run types the whole atlas -- every real encoder against its own
random-init control (AT1's nine-verdict nuisance grid) AND every real-encoder pair for shared-latent
alignment (AL2's permutation-floor topology test) -- in one pass, so the 06_cognitive_currencies_atlas
scope claim is made over the registered column set, not a laptop subset.

WHY STUDIO (not the laptop): the honest scope verdict (modality-specific vs universal) is only meaningful
once EVERY registered encoder column is present and count-matched on the SAME shared clipset. Building
all of those caches (V-JEPA ViT-L full-clip and single-frame, DINOv2, Qwen textified, wav2vec2 sonified,
plus the at-scale additions) and holding them all in memory for the pairwise alignment grid exceeds the
laptop pool. This orchestrator does NOT encode; it CONSUMES the finished caches the Studio encode jobs
wrote and runs the two audited evaluators over the full set. It carries a hard >= 32GB free-RAM guard so
it cannot run on the laptop by accident, and it refuses to start a run whose column set is not the full
registered atlas unless --allow-partial is passed (so a partial Studio run cannot masquerade as the
at-scale claim).

PREREGISTERED NULLS (both inherited verbatim from the audited pilots, NOT re-invented here):
  - AT1 null: every substrate's linear-probe decodability delta over its OWN matched random-init control
    is within seed spread (random-control-artifact); pretraining bought no nuisance invariance. Cleared
    per column only by a seed-CI lower bound above zero that ALSO survives the capacity-capped MLP probe
    (probe-specific guard) with no per-seed sign flip. The random-init control is the honest floor; a
    square latent projection is NOT admissible and is not used.
  - AL2 null: a rank-k linear map fit on a ROW-SHUFFLED (permuted) target pairing preserves neighbor
    structure as well as the learned map (alignment-artifact) for every real-encoder pair. Cleared per
    pair only by a kNN neighbor-recall delta (learned minus permuted) with a seed-CI lower bound strictly
    above zero and no sign flip. Ridge R^2 is descriptive only, never the win criterion (the A1 regrade).

The at-scale verdict is the SCOPE typing over the full column/pair set: random-control-artifact grid-wide,
modality-specific, or (only if survivors span >= 2 modalities on the full registered set) universal. Any
column/pair whose cache is missing is REPORTED missing and EXCLUDES the universal claim; it is never
silently substituted. This orchestrator loads no encoder.

Usage (Studio):
  python scripts/studio/atlas_multi_encoder_grid.py --seeds 0-9   -> runs/mot/atlas_multi_encoder_grid.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.featurize_programmatic import CACHE_ROOT  # noqa: E402
from scripts.mop_al2_alignment_pilot import evaluate_pairs  # noqa: E402
from scripts.mop_at1_grid_pilot import COLUMNS as AT1_PILOT_COLUMNS  # noqa: E402
from scripts.mop_at1_grid_pilot import REFERENCE_COLUMNS, evaluate_grid, parse_seeds  # noqa: E402

MIN_FREE_RAM_GB = 32.0

# The FULL registered atlas: every real-encoder column the at-scale grid must carry (superset of the
# laptop pilot's COLUMNS). A run that is missing any of these cannot make the universal-scope claim.
REGISTERED_COLUMNS = list(AT1_PILOT_COLUMNS) + [
    {
        "tag": "vjepa_bound_video",
        "modality": "video_bound",
        "family": "vit_l_vjepa",
        "real": "vjepa2_vitl_bound_video",
        "randinit": "randominit_vitl_nuisance",
        "note": "DR1 curated real bound-attribute video; control is the matched random-init ViT-L",
    },
    {
        "tag": "dinov2b",
        "modality": "image",
        "family": "vit_b_dinov2",
        "real": "dinov2b_nuisance_real",
        "randinit": "dinov2b_nuisance_randominit",
        "note": "at-scale larger DINOv2 column (Studio encode)",
    },
]

# Real arms only for the alignment grid: mapping a pretrained substrate onto a random-init one is a
# context row, not a claim. These are the real-encoder cache names AL2 pairs.
REGISTERED_ALIGNMENT_ARMS = [
    "vjepa2_vitl_nuisance",
    "vjepa2_vitl_singleframe",
    "vjepa2_vitl_bound_video",
    "dinov2s_nuisance_real",
    "dinov2b_nuisance_real",
    "qwen05b_textified_real",
    "wav2vec2_sonified_real",
]


def assert_studio_ram(min_gb: float = MIN_FREE_RAM_GB) -> float:
    """Hard guard: refuse to run unless >= min_gb of free RAM. Keeps the at-scale grid off the laptop."""
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024**3)
    except Exception as e:
        raise SystemExit(
            f"cannot read free RAM ({e}); refusing to run without the >= {min_gb:.0f}GB safety check. "
            "Install psutil on the Studio box."
        ) from e
    if free_gb < min_gb:
        raise SystemExit(
            f"free RAM {free_gb:.1f}GB < required {min_gb:.0f}GB. This is the at-scale Studio grid; it "
            "will not run on the laptop pool (18GB)."
        )
    return free_gb


def _present_columns(cache_root: Path, columns: list[dict]) -> tuple[list[dict], list[str]]:
    """Split the registered columns into those whose real+randinit caches both exist and those missing.
    A missing cache excludes the column and, downstream, the universal-scope claim."""
    present, missing = [], []
    for col in columns:
        real_ok = (cache_root / col["real"] / "meta.json").exists()
        rand_ok = (cache_root / col["randinit"] / "meta.json").exists()
        if real_ok and rand_ok:
            present.append(col)
        else:
            missing.append(col["tag"])
    return present, missing


def _present_arms(cache_root: Path, arms: list[str]) -> tuple[list[str], list[str]]:
    present = [a for a in arms if (cache_root / a / "meta.json").exists()]
    missing = [a for a in arms if a not in present]
    return present, missing


def atlas_scope(at1: dict, al2: dict, full_grid: bool, full_pairs: bool) -> dict:
    """The at-scale scope typing. Survivors are AT1 columns that cleared the nine-verdict order AND, for
    the alignment axis, pairs that cleared the permutation floor. The UNIVERSAL claim is only admissible
    when the full registered column/pair set was present (no missing cache silently narrowed the set)."""
    at1_survivors = [c for c in at1.get("grid", []) if c.get("verdict") == "genuine-substrate-signal"]
    al2_survivors = [
        name for name, p in al2.get("pairs", {}).items() if p.get("verdict") == "genuine-shared-structure"
    ]
    survivor_mods = sorted({c["modality"] for c in at1_survivors})
    if not at1_survivors and not al2_survivors:
        scope = "random-control-artifact (atlas-wide): no registered substrate beat its own control"
    elif len(survivor_mods) >= 2 and full_grid and full_pairs:
        scope = (
            f"universal across {survivor_mods} on the FULL registered atlas (both nuisance and "
            "alignment axes, no missing cache narrowed the set)"
        )
    elif len(survivor_mods) >= 2:
        scope = (
            f"multi-modality ({survivor_mods}) but the registered set was INCOMPLETE; the universal "
            "claim is withheld until every registered cache is present"
        )
    else:
        mod = survivor_mods[0] if survivor_mods else "alignment-only"
        scope = f"modality-specific ({mod}) on the registered atlas"
    return {
        "scope": scope,
        "at1_survivors": [c["tag"] for c in at1_survivors],
        "al2_survivors": al2_survivors,
        "survivor_modalities": survivor_mods,
        "full_registered_grid": full_grid,
        "full_registered_pairs": full_pairs,
    }


def run(cache_root: Path, seeds: list[int], allow_partial: bool) -> dict:
    t0 = time.perf_counter()
    present_cols, missing_cols = _present_columns(cache_root, REGISTERED_COLUMNS)
    present_arms, missing_arms = _present_arms(cache_root, REGISTERED_ALIGNMENT_ARMS)
    full_grid = not missing_cols
    full_pairs = not missing_arms
    if (not full_grid or not full_pairs) and not allow_partial:
        raise SystemExit(
            f"registered atlas incomplete (missing columns={missing_cols} arms={missing_arms}); this "
            "at-scale grid refuses to run partial without --allow-partial, so a partial run cannot "
            "masquerade as the at-scale claim. Finish the Studio encodes or pass --allow-partial "
            "(verdict will withhold universal)."
        )
    at1 = evaluate_grid(cache_root, seeds, columns=present_cols, reference_columns=REFERENCE_COLUMNS)
    al2 = evaluate_pairs(cache_root, seeds, arms=present_arms)
    scope = atlas_scope(at1, al2, full_grid, full_pairs)
    # Null is supported when NEITHER axis rejects its inherited null; not evaluable (None) only when
    # neither axis had anything to type.
    at1_null = at1.get("null_supported")
    al2_null = al2.get("null_supported")
    if at1_null is None and al2_null is None:
        null_supported = None
        verdict = "NO COLUMNS/PAIRS: nothing on disk to type on either axis"
    else:
        null_supported = (at1_null in (True, None)) and (al2_null in (True, None))
        verdict = scope["scope"]
    return {
        "experiment": {
            "id": "atlas_multi_encoder_grid",
            "metric": "at-scale AT1 nine-verdict nuisance grid + AL2 permutation-floor alignment grid",
            "baseline": "each column's own matched random-init (AT1); permuted-pairing rank-k map (AL2)",
            "null_hypothesis": (
                "no registered substrate beats its own random-init control (AT1) and no real-encoder "
                "pair beats its permutation floor (AL2): the atlas is a random-control/alignment artifact"
            ),
            "tier": "studio (at-scale, full registered atlas)",
        },
        "seeds": seeds,
        "registered_columns_present": [c["tag"] for c in present_cols],
        "registered_columns_missing": missing_cols,
        "registered_arms_present": present_arms,
        "registered_arms_missing": missing_arms,
        "full_registered_grid": full_grid,
        "full_registered_pairs": full_pairs,
        "at1_nuisance_grid": at1,
        "al2_alignment_grid": al2,
        "atlas_scope": scope,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": verdict,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Atlas at-scale AT1+AL2 multi-encoder grid (Studio)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="run with a partial registered set (universal-scope claim is then withheld)",
    )
    ap.add_argument("--out", default="runs/mot/atlas_multi_encoder_grid.json")
    a = ap.parse_args(argv)

    assert_studio_ram()  # Studio-only guard
    result = run(Path(a.cache_root), parse_seeds(a.seeds), a.allow_partial)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=str))
    print(
        json.dumps(
            {k: result[k] for k in ("atlas_scope", "null_supported", "verdict")}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
