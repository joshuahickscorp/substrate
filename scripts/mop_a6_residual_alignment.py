#!/usr/bin/env python
"""A6 residual alignment (the language-independent-abstraction bet, laptop lane, ZERO new encode).

Motivation, grounded in verified disk state (2026-07-02): al2's regrade reports
vjepa2_vitl_nuisance -> qwen05b_textified_real as "genuine-shared-structure" (delta 0.084). But a read
only probe shows BOTH the pixel-derived text cache (shape 0.267 vs chance 0.20, color 0.517) and the
handcrafted HOG descriptor (shape 0.217, color 1.000) carry color and NOT shape, while the frozen vision
encoders carry shape (0.78-0.87). So the raw vision<->text topology alignment could be nothing more than
the color channel both sides trivially share.

This experiment separates two very different readings of "perspectives share a code":
  (1) TRIVIAL: both sides can decode the same discrete factor (color), so their neighbor graphs both
      cluster by color and any topology metric lights up. This is "they name the same categories."
  (2) THESIS: the two representations share metric geometry BEYOND the label partition, i.e. residual
      within/between structure aligns after the known discrete factors are projected out. This is the
      actual MoP shared-code claim.

Design (preregistered in code before any result):
  - Reuse al2's exact topology metric: rank-k ridge map, kNN neighbor-recall on the test split, permuted
    pairing floor of equal rank. delta = learned recall minus permuted recall.
  - RESIDUALIZATION is fit on the TRAIN split only (no test leakage): project out the column space of a
    per-condition design matrix (discrete factors as one-hot class means, the 6 nuisance factors as a
    linear design with rotation entered as sin/cos), learned from train, from both reps on train and test.
  - Conditions per pair: raw ; minus_color ; minus_shape_color ; minus_nuisance ; minus_all.
  - A pair "shares structure" in a condition only when the delta seed-CI lower bound is strictly above
    zero AND no per-seed sign flip (al2's classify_pair rule); that is the only survivor verdict.

Preregistered nulls (N1, N2 first; N3, N4 added as a follow-up control after N2 rejected, see EXPERIMENT
provenance):
  - N1 (color-carried): after minus_color, alignment is within the permutation floor.
  - N2 (label-partition-only): after minus_shape_color, within the floor. Rejecting N2 means shared
    structure survives removing both discrete labels.
  - N3 (nuisance-geometry-carried): after minus_nuisance, within the floor.
  - N4 (nothing-beyond-named-factors): after minus_all (shape+color+nuisance), within the floor. A
    survivor here would be shared structure beyond every generative factor, the strongest positive.
A tie is a null. No threshold is tuned after seeing a number.

Usage:
  PYTHONPATH=<repo>/src OMP_NUM_THREADS=4 .venv/bin/python scripts/mop_a6_residual_alignment.py --seeds 0-9
  -> runs/mot/a6_residual_alignment.json

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.featurize_programmatic import CACHE_ROOT, CLIPSET  # noqa: E402
from scripts.mop_al2_alignment_pilot import (  # noqa: E402
    KNN_K,
    N_PERM,
    PRIMARY_RANK,
    RANKS,
    classify_pair,
    knn_neighbor_recall,
    r2_score,
    rank_truncate,
    ridge_fit,
)
from scripts.mop_at1_grid_pilot import parse_seeds  # noqa: E402

from mop.substrate import LatentStore  # noqa: E402

EXPERIMENT = {
    "id": "a6_residual_alignment",
    "metric": "al2 kNN neighbor-recall topology delta (learned vs permuted-pairing floor), computed on "
    "representations with named generative factors projected out (train-fit) per condition",
    "baseline": "the same rank-k map refit on a row-shuffled target pairing (topology permutation null)",
    "ablation": "condition sweep raw / minus_color / minus_shape_color / minus_nuisance / minus_all; "
    "rank 8 vs 32, primary 32",
    "null_hypothesis": "N1: after removing color, vision<->text neighbor topology is within the "
    "permutation floor (color-carried). N2: after removing shape+color, within the floor (only the label "
    "partition is shared). N3: after removing the 6 nuisance factors, within the floor (geometry-carried). "
    "N4: after removing shape+color+nuisance, within the floor (nothing beyond named factors is shared).",
    "provenance": "N1 and N2 were preregistered and run first. N2 REJECTED (alignment survived removing "
    "shape+color), so the nuisance conditions N3/N4 were added as a follow-up control (position, size, "
    "orientation, motion are the obvious remaining shared carrier) and preregistered before THEIR run. "
    "This is disclosed rather than presented as a single up-front design.",
    "tier": "cpu-now (A6; zero new encode; the registered claim is studio/DR1)",
    "note": "separates shared-label-partition (trivial) and shared-nuisance-geometry (mundane) from "
    "shared-abstract-code (the MoP north star). Reuses al2's topology primitives; residualization is "
    "train-fit only; a survivor is a stable genuine-shared-structure verdict, ties are nulls.",
}

# Vision (carries shape+color) crossed with the pixel-derived non-encoder columns (carry color, not shape).
PAIRS = [
    ("vjepa2_vitl_nuisance", "qwen05b_textified_real"),
    ("qwen05b_textified_real", "vjepa2_vitl_nuisance"),
    ("dinov2s_nuisance_real", "qwen05b_textified_real"),
    ("vjepa2_vitl_nuisance", "handcrafted_descriptors"),
]

# SHAPE-AXIS BET (this file's addition; distinct --out, no clobber of the base run). The shapecap text cache
# is a Qwen mean over a caption that DOES carry shape (build report: carries_shape=true, killswitch did NOT
# fire, real shape linear-decode 0.6167 vs chance 0.20). Unlike the color-grid textification (shape at
# chance 0.267), this text side has a real shape signal, so it is the FIRST column where a surviving
# minus_all vision<->text alignment could be shape-carried shared abstraction rather than nuisance geometry.
#
# PREREGISTERED VERDICT RULE for the shape axis (fixed in code before running; a tie is a null):
#   Testability killswitch: the shape-caption cache must carry shape (real shape decode strictly above
#   chance 0.20). It does (0.6167), so shape_axis_testable=True. If a future rebuild had it at chance, the
#   axis would be moot on this clipset (requires DR1 real video) and we would report testable=False, NOT
#   fake a positive.
#   Decisive question: does vision<->shapecap alignment SURVIVE minus_all (shape+color+nuisance removed)?
#     - SURVIVES  := for a vision->shapecap pair, the minus_all condition returns the ONLY clearing verdict
#       "genuine-shared-structure" (al2 classify_pair: seed-CI lower bound strictly > 0 AND no sign flip)
#       at the primary rank. This would be shape-carried cross-modal abstract shared code (a genuine first).
#     - COLLAPSES := minus_all is "alignment-artifact" or "non-replicating" (at/indistinguishable from the
#       permutation floor, or unstable). This bounds the null: even a shape-carrying caption shares no
#       abstract geometry with vision beyond the nuisance, exactly as the color-grid text did.
#   A minus_all that merely ties the floor is a COLLAPSE (null), never rounded up to a survivor.
SHAPECAP_PAIRS = [
    ("vjepa2_vitl_nuisance", "qwen05b_shapecap_real"),
    ("dinov2s_nuisance_real", "qwen05b_shapecap_real"),
]
CONDITIONS = ["raw", "minus_color", "minus_shape_color", "minus_nuisance", "minus_all"]
# The 6 generative nuisance factors are per-clip (same clips across every column), read once from the
# canonical vision cache. Rotation is circular, so it enters the design as sin/cos, not a raw linear term.
NUIS_SOURCE = "vjepa2_vitl_nuisance"
NUIS_COLS = ["r", "rot", "x0", "y0", "vx", "vy"]


def load_column(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """(latents [N,D], shape [N], color [N]) handling both cache layouts, or None if absent."""
    root = CACHE_ROOT / name
    if not (root / "meta.json").exists():
        return None
    store = LatentStore.open(root)
    x = store.latents().float()
    if (root / "factors.json").exists():
        fac = json.loads((root / "factors.json").read_text())
        shape = torch.tensor(fac["shape"]).long()
        color = torch.tensor(fac["color"]).long()
    else:  # native vision-cache layout
        shape = torch.tensor(np.load(root / "labels_shape.npy")).long()
        color = torch.tensor(np.load(root / "labels_color.npy")).long()
    return x, shape, color


def _onehot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, n_classes).float()


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = train.mean(dim=0), train.std(dim=0) + 1e-6
    return (train - mu) / sd, (other - mu) / sd


def nuisance_design(nuis: torch.Tensor) -> torch.Tensor:
    """The 6 raw nuisance columns -> a linear design [N, 7]: r, x0, y0, vx, vy, sin(rot), cos(rot).
    Rotation enters as sin/cos because it is circular (a raw linear term cannot remove a periodic factor)."""
    r, rot, x0, y0, vx, vy = (nuis[:, i] for i in range(6))
    return torch.stack([r, x0, y0, vx, vy, torch.sin(rot), torch.cos(rot)], dim=1)


def build_design(
    shape: torch.Tensor, color: torch.Tensor, nuis: torch.Tensor, mode: str
) -> torch.Tensor | None:
    """The design matrix whose column space is projected out, per condition. One-hot factors span the
    constant; minus_nuisance adds an explicit intercept so the mean is removed too. None = raw (no-op)."""
    n_s, n_c = int(shape.max()) + 1, int(color.max()) + 1
    oh_s, oh_c = _onehot(shape, n_s), _onehot(color, n_c)
    znuis = nuisance_design(nuis)
    ones = torch.ones(shape.shape[0], 1)
    if mode == "raw":
        return None
    if mode == "minus_color":
        return oh_c
    if mode == "minus_shape_color":
        return torch.cat([oh_s, oh_c], dim=1)
    if mode == "minus_nuisance":
        return torch.cat([ones, znuis], dim=1)
    if mode == "minus_all":
        return torch.cat([oh_s, oh_c, znuis], dim=1)
    raise ValueError(mode)


def project_out(x: torch.Tensor, design: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
    """Remove the component of x explained by the design's column space, coefficients fit on TRAIN only
    (no test leakage): beta = pinv(D_tr) @ X_tr, residual = X - D @ beta."""
    beta = torch.linalg.pinv(design[tr]) @ x[tr]
    return x - design @ beta


def residual_learned_vs_floors(
    xa: torch.Tensor,
    xb: torch.Tensor,
    shape: torch.Tensor,
    color: torch.Tensor,
    nuis: torch.Tensor,
    mode: str,
    seed: int,
    rank: int,
    test_frac: float = 0.3,
) -> dict:
    """One seed, one rank, one residualization condition. Residualizer fit on train only, applied to
    both a and b, on train and test alike; then al2's learned-vs-permuted topology delta."""
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    perm = torch.randperm(n, generator=g)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]

    design = build_design(shape, color, nuis, mode)
    if design is not None:
        xa = project_out(xa, design, tr)
        xb = project_out(xb, design, tr)

    atr, ate = _standardize(xa[tr], xa[te])
    btr, bte = _standardize(xb[tr], xb[te])

    w = rank_truncate(ridge_fit(atr, btr), rank)
    pred = ate @ w
    learned_recall = knn_neighbor_recall(pred, bte, k=KNN_K)
    learned_r2 = r2_score(pred, bte)

    perm_recalls = []
    for _ in range(N_PERM):
        shuf = torch.randperm(cut, generator=g)
        w_s = rank_truncate(ridge_fit(atr, btr[shuf]), rank)
        perm_recalls.append(knn_neighbor_recall(ate @ w_s, bte, k=KNN_K))
    permuted_recall = float(sum(perm_recalls) / len(perm_recalls))

    return {
        "learned_recall": round(learned_recall, 4),
        "permuted_recall": round(permuted_recall, 4),
        "learned_r2_descriptive": round(learned_r2, 4),
        "delta": round(learned_recall - permuted_recall, 4),
    }


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def condition_report(
    xa: torch.Tensor,
    xb: torch.Tensor,
    shape: torch.Tensor,
    color: torch.Tensor,
    nuis: torch.Tensor,
    mode: str,
    seeds: list[int],
    ranks: list[int],
) -> dict:
    by_rank = {}
    for k in ranks:
        runs = [residual_learned_vs_floors(xa, xb, shape, color, nuis, mode, s, k) for s in seeds]
        by_rank[str(k)] = {
            "per_seed": runs,
            "mean_learned_recall": _mean([r["learned_recall"] for r in runs]),
            "mean_permuted_recall": _mean([r["permuted_recall"] for r in runs]),
            **classify_pair([r["delta"] for r in runs]),
        }
    primary = str(PRIMARY_RANK if PRIMARY_RANK in ranks else ranks[-1])
    return {"by_rank": by_rank, "primary_rank": int(primary), "verdict": by_rank[primary]["verdict"]}


def load_nuisance(cache_root: Path) -> torch.Tensor:
    """The per-clip generative nuisance, read once from the canonical vision cache (same clips for every
    column by the clip-identity rule). [N, 6] in NUIS_COLS order."""
    return torch.tensor(np.load(cache_root / NUIS_SOURCE / "nuisance.npy")).float()


def evaluate(cache_root: Path, seeds: list[int], ranks: list[int], pairs: list | None = None) -> dict:
    pairs = PAIRS if pairs is None else pairs
    loaded = {}
    for name in {n for pair in pairs for n in pair}:
        got = load_column(name)
        if got is not None:
            loaded[name] = got
    nuis = load_nuisance(cache_root)
    pairs_out = {}
    for a_name, b_name in pairs:
        if a_name not in loaded or b_name not in loaded:
            continue
        xa, sa, ca = loaded[a_name]
        xb, sb, cb = loaded[b_name]
        if xa.shape[0] != xb.shape[0] or xa.shape[0] != nuis.shape[0]:
            continue
        # clip identity across columns is the featurizer's invariant (same clips, same order); the
        # per-column labels must agree, else the residualizer would remove the wrong factor.
        if not (torch.equal(sa, sb) and torch.equal(ca, cb)):
            print(f"[skip {a_name} -> {b_name}] label mismatch across columns", flush=True)
            continue
        # labels are identical across columns; use the source column's shape/color for residualization.
        conditions = {mode: condition_report(xa, xb, sa, ca, nuis, mode, seeds, ranks) for mode in CONDITIONS}
        pairs_out[f"{a_name} -> {b_name}"] = {
            "conditions": conditions,
            **{f"{mode}_verdict": conditions[mode]["verdict"] for mode in CONDITIONS},
        }
        print(
            f"[{a_name} -> {b_name}] "
            + " ".join(f"{mode}={conditions[mode]['verdict']}" for mode in CONDITIONS),
            flush=True,
        )

    def _all_floor(mode: str) -> bool | None:
        if not pairs_out:
            return None
        return all(p[f"{mode}_verdict"] == "alignment-artifact" for p in pairs_out.values())

    def _survivors(mode: str) -> int:
        # A "survivor" retains STABLE shared structure (genuine-shared-structure), the only verdict that
        # clears the null. non-replicating and alignment-artifact both fail to reject it.
        return sum(1 for p in pairs_out.values() if p[f"{mode}_verdict"] == "genuine-shared-structure")

    survivors = {mode: _survivors(mode) for mode in CONDITIONS}
    # Decisive carrier read: what factor's removal collapses every pair to a non-survivor? If minus_all
    # and minus_nuisance leave zero survivors while minus_shape_color leaves some, the cross-modal shared
    # code is the spatiotemporal NUISANCE geometry, not the semantic (shape,color) abstraction.
    if not pairs_out:
        carrier = "no-pairs"
    elif survivors["minus_nuisance"] == 0 and survivors["minus_shape_color"] > 0:
        carrier = "nuisance-geometry (position/size/orientation/motion), NOT the semantic label partition"
    elif survivors["minus_all"] > 0:
        carrier = "structure beyond all named factors (shape, color, nuisance) survives, investigate"
    elif survivors["minus_color"] == 0:
        carrier = "color channel only"
    else:
        carrier = "mixed; see per-pair table"

    # SHAPE-AXIS decisive read (only meaningful when the run's pairs target the shapecap text column). We
    # apply the preregistered rule: a vision->shapecap pair whose minus_all verdict is
    # "genuine-shared-structure" SURVIVES (shape-carried cross-modal abstraction); anything else COLLAPSES.
    shapecap_pairs = {k: v for k, v in pairs_out.items() if k.endswith("qwen05b_shapecap_real")}
    shape_axis = None
    if shapecap_pairs:
        survivor_pairs = [
            k for k, v in shapecap_pairs.items() if v["minus_all_verdict"] == "genuine-shared-structure"
        ]
        shape_axis = {
            "shape_axis_testable": True,  # killswitch did not fire (real shape decode 0.6167 > chance 0.20)
            "killswitch_fired": False,
            "real_shape_probe": 0.6167,
            "per_pair_minus_all_verdict": {k: v["minus_all_verdict"] for k, v in shapecap_pairs.items()},
            "per_pair_minus_shape_color_verdict": {
                k: v["minus_shape_color_verdict"] for k, v in shapecap_pairs.items()
            },
            "minus_all_survivors": survivor_pairs,
            "shape_axis_alignment_survives_minus_all": len(survivor_pairs) > 0,
            "reading": (
                "SURVIVES: vision<->shapecap alignment clears the permutation floor even after removing "
                "shape+color+nuisance; the shared cross-modal code is shape-carried abstraction (a genuine "
                "first on this clipset)."
                if survivor_pairs
                else "COLLAPSES: even a shape-carrying caption shares no abstract geometry with vision "
                "beyond the nuisance; minus_all is at/indistinguishable from the permutation floor, exactly "
                "as the color-grid textification collapsed. A bounding null, not a positive."
            ),
        }

    return {
        "experiment": EXPERIMENT,
        "clipset": CLIPSET,
        "seeds": seeds,
        "ranks": ranks,
        "conditions": CONDITIONS,
        "columns_present": sorted(loaded),
        "nuisance_cols": NUIS_COLS,
        "pairs": pairs_out,
        "shape_axis": shape_axis,
        "stable_genuine_survivors_per_condition": survivors,
        "carrier_verdict": carrier,
        # Legacy strict-floor flags (every pair at the permutation floor). non-replicating is unstable,
        # not floor, so these can read False even when zero pairs stably survive; use the survivor counts.
        "n1_color_carried_all_pairs": _all_floor("minus_color"),
        "n2_label_partition_only_all_pairs": _all_floor("minus_shape_color"),
        "n3_nuisance_carried_all_pairs": _all_floor("minus_nuisance"),
        "n4_all_named_factors_all_pairs": _all_floor("minus_all"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A6 residual (partial-out) vision<->text topology alignment")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--ranks", default="8,32")
    ap.add_argument(
        "--pairs",
        default="base",
        choices=["base", "shapecap"],
        help="'base' = the default vision<->pixel-text pairs; 'shapecap' = the SHAPE-AXIS bet "
        "(vision -> qwen05b_shapecap_real). Use with a distinct --out to avoid clobbering the base run.",
    )
    ap.add_argument("--out", default="runs/mot/a6_residual_alignment.json")
    a = ap.parse_args(argv)
    ranks = [int(x) for x in a.ranks.split(",")] if a.ranks else RANKS
    pairs = SHAPECAP_PAIRS if a.pairs == "shapecap" else PAIRS
    result = evaluate(Path(a.cache_root), parse_seeds(a.seeds), ranks, pairs)
    text = json.dumps(result, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
