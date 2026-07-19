#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.featurize_programmatic import CACHE_ROOT, CLIPSET  # noqa: E402
from scripts.mop_at1_grid_pilot import clip_identity_check, load_store, parse_seeds  # noqa: E402

from mop.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

EXPERIMENT = {
    "id": "al2_alignment_regrade",
    "metric": "cross-substrate kNN neighbor-recall of a rank-k linear map over a permuted-pairing floor",
    "baseline": "rank-k map refit on a row-shuffled target-side pairing (topology/permutation null)",
    "ablation": "rank sweep (8 vs 32); primary rank preregistered as 32; R^2 kept as descriptive scalar",
    "null_hypothesis": "a rank-k map fit on a permuted pairing preserves neighbor structure as well as "
    "the learned map (alignment-artifact) for every substrate pair",
    "tier": "cpu-now (A1 re-grade of al2; the registered claim is studio)",
    "regrade_note": "A1: win criterion switched from ridge R^2 over a random-map floor to a kNN-topology "
    "permutation null; real V-JEPA arm discovery fixed to include the vjepa_hf cache "
    "'vjepa2_vitl_nuisance' (was omitted as 'vjepa2_vitl_nuisance_real', a name mismatch).",
}

DEFAULT_ARMS = [
    "vjepa2_vitl_nuisance",
    "vjepa2_vitl_singleframe",
    "dinov2s_nuisance_real",
    "qwen05b_textified_real",
    "wav2vec2_sonified_real",
    "handcrafted_descriptors",
]
RANKS = [8, 32]
PRIMARY_RANK = 32
RIDGE_LAMBDA = 1e-2
KNN_K = 10  # neighborhood size for the topology metric
N_PERM = 5  # target-side shuffles averaged for the permutation floor


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = train.mean(dim=0), train.std(dim=0) + 1e-6
    return (train - mu) / sd, (other - mu) / sd


def ridge_fit(xa: torch.Tensor, xb: torch.Tensor, lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    da = xa.shape[1]
    return torch.linalg.solve(xa.T @ xa + lam * torch.eye(da), xa.T @ xb)


def rank_truncate(w: torch.Tensor, k: int) -> torch.Tensor:
    if k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :k] @ torch.diag(s[:k]) @ vh[:k]


def r2_score(pred: torch.Tensor, y: torch.Tensor) -> float:
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum() + 1e-8
    return float(1 - ss_res / ss_tot)


def _knn_index(x: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    kk = min(k, x.shape[0] - 1)
    return d.topk(kk, largest=False).indices


def knn_neighbor_recall(pred_b: torch.Tensor, true_b: torch.Tensor, k: int = KNN_K) -> float:
    if pred_b.shape[0] <= k:
        return 0.0
    nn_pred = _knn_index(pred_b, k)
    nn_true = _knn_index(true_b, k)
    kk = nn_true.shape[1]
    hits = 0
    for i in range(pred_b.shape[0]):
        pset = set(nn_pred[i].tolist())
        hits += sum(1 for j in nn_true[i].tolist() if j in pset)
    return float(hits / (pred_b.shape[0] * kk))


def learned_vs_floors(
    xa: torch.Tensor, xb: torch.Tensor, seed: int, rank: int, test_frac: float = 0.3
) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    perm = torch.randperm(n, generator=g)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    atr, ate = _standardize(xa[tr], xa[te])
    btr, bte = _standardize(xb[tr], xb[te])

    w = rank_truncate(ridge_fit(atr, btr), rank)
    pred = ate @ w
    learned_recall = knn_neighbor_recall(pred, bte)
    learned_r2 = r2_score(pred, bte)  # descriptive only, never the win criterion

    perm_recalls = []
    for _ in range(N_PERM):
        shuf = torch.randperm(cut, generator=g)
        w_s = rank_truncate(ridge_fit(atr, btr[shuf]), rank)
        perm_recalls.append(knn_neighbor_recall(ate @ w_s, bte))
    permuted_recall = float(sum(perm_recalls) / len(perm_recalls))

    return {
        "learned_recall": round(learned_recall, 4),
        "permuted_recall": round(permuted_recall, 4),
        "learned_r2": round(learned_r2, 4),
        "delta": round(learned_recall - permuted_recall, 4),
    }


def classify_pair(deltas: list[float]) -> dict:
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    if ci["lo"] <= 0:
        verdict = "alignment-artifact"
    elif flips["any_flip"]:
        verdict = "non-replicating"
    else:
        verdict = "genuine-shared-structure"
    return {"verdict": verdict, "delta_ci": ci, "sign_flips": flips}


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def pair_report(xa: torch.Tensor, xb: torch.Tensor, seeds: list[int], ranks: list[int]) -> dict:
    by_rank = {}
    for k in ranks:
        runs = [learned_vs_floors(xa, xb, s, k) for s in seeds]
        by_rank[k] = {
            "per_seed": runs,
            "mean_learned_recall": _mean([r["learned_recall"] for r in runs]),
            "mean_permuted_recall": _mean([r["permuted_recall"] for r in runs]),
            "mean_learned_r2_descriptive": _mean([r["learned_r2"] for r in runs]),
            **classify_pair([r["delta"] for r in runs]),
        }
    primary = PRIMARY_RANK if PRIMARY_RANK in by_rank else ranks[-1]
    return {"by_rank": by_rank, "primary_rank": primary, "verdict": by_rank[primary]["verdict"]}


def evaluate_pairs(
    cache_root: Path, seeds: list[int], arms: list[str] | None = None, ranks: list[int] | None = None
) -> dict:
    arms = DEFAULT_ARMS if arms is None else arms
    ranks = RANKS if ranks is None else ranks
    t0 = time.perf_counter()
    loaded, sidecars = {}, {}
    for name in arms:
        got = load_store(cache_root / name)
        if got is None:
            continue
        loaded[name] = got[0]
        sidecars[name] = got[2]
    pairs = {}
    for a_name, b_name in itertools.permutations(sorted(loaded), 2):
        xa, xb = loaded[a_name], loaded[b_name]
        if xa.shape[0] != xb.shape[0]:
            continue
        rep = pair_report(xa, xb, seeds, ranks)
        pairs[f"{a_name} -> {b_name}"] = rep
        print(f"[{a_name} -> {b_name}] verdict={rep['verdict']}", flush=True)
    return {
        "experiment": EXPERIMENT,
        "clipset": CLIPSET,
        "seeds": seeds,
        "ranks": ranks,
        "arms_present": sorted(loaded),
        "arms_missing": sorted(set(arms) - set(loaded)),
        "clip_identity": clip_identity_check(sidecars),
        "pairs": pairs,
        "null_supported": (
            all(p["verdict"] == "alignment-artifact" for p in pairs.values()) if pairs else None
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AL2 shared-latent alignment vs random-map floor (pilot)")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/al2_alignment_regrade.json")
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun marker, recorded in the output")
    a = ap.parse_args(argv)
    result = evaluate_pairs(Path(a.cache_root), parse_seeds(a.seeds))
    result["rerun"] = a.rerun
    if not result["pairs"]:
        result["verdict"] = "NO PAIRS: fewer than two count-matched substrate caches on disk, nothing typed"
    text = json.dumps(result, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
