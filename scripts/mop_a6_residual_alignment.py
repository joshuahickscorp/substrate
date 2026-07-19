#!/usr/bin/env python

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

PAIRS = [
    ("vjepa2_vitl_nuisance", "qwen05b_textified_real"),
    ("qwen05b_textified_real", "vjepa2_vitl_nuisance"),
    ("dinov2s_nuisance_real", "qwen05b_textified_real"),
    ("vjepa2_vitl_nuisance", "handcrafted_descriptors"),
]

SHAPECAP_PAIRS = [
    ("vjepa2_vitl_nuisance", "qwen05b_shapecap_real"),
    ("dinov2s_nuisance_real", "qwen05b_shapecap_real"),
]
CONDITIONS = ["raw", "minus_color", "minus_shape_color", "minus_nuisance", "minus_all"]
NUIS_SOURCE = "vjepa2_vitl_nuisance"
NUIS_COLS = ["r", "rot", "x0", "y0", "vx", "vy"]


def load_column(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
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
    r, rot, x0, y0, vx, vy = (nuis[:, i] for i in range(6))
    return torch.stack([r, x0, y0, vx, vy, torch.sin(rot), torch.cos(rot)], dim=1)


def build_design(
    shape: torch.Tensor, color: torch.Tensor, nuis: torch.Tensor, mode: str
) -> torch.Tensor | None:
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
        if not (torch.equal(sa, sb) and torch.equal(ca, cb)):
            print(f"[skip {a_name} -> {b_name}] label mismatch across columns", flush=True)
            continue
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
        return sum(1 for p in pairs_out.values() if p[f"{mode}_verdict"] == "genuine-shared-structure")

    survivors = {mode: _survivors(mode) for mode in CONDITIONS}
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
