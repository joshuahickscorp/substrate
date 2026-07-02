#!/usr/bin/env python
"""AL2 shared-latent alignment pilot (WP-14, Q3.9): does a THIN linear map between substrate pairs on
the shared nuisance clips predict one substrate's latents from another ABOVE the random-map-of-equal-rank
floor? The alignment-artifact verdict (nine-verdict order, step 4) says any apparent agreement is the
map's capacity, not shared structure; this pilot measures exactly that, per pair, per rank.

Preregistered construction (in code before any result exists):
  - learned map: ridge least squares train-split fit, SVD-truncated to rank k, R^2 on the test split
  - floor A (random-map-of-equal-rank): a FIXED seeded Gaussian rank-k map with only a scalar gain fit
    on train (the map itself never sees the pairing)
  - floor B (shuffled-fit): the identical ridge+truncation fit on row-shuffled train pairs
    (correspondence destroyed, capacity intact)
  - per-seed delta = learned R^2 minus max(floor A, floor B); primary rank preregistered as 32.
Preregistered null (registry AL2): a random map of equal rank predicts the target as well as the
learned map (alignment-artifact) for every substrate pair.

Usage: python scripts/mot_al2_alignment_pilot.py --seeds 0-4   -> runs/mot/al2_alignment_pilot.json

No em dashes or en dashes (BLACKHOLE.md).
"""

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
from scripts.mot_at1_grid_pilot import clip_identity_check, load_store, parse_seeds  # noqa: E402

from devsys.diagnostics.riskcov import seed_ci, sign_flip_report  # noqa: E402

EXPERIMENT = {
    "id": "al2_alignment_pilot",
    "metric": "cross-substrate prediction R^2 of a rank-k linear map over the random-map floor",
    "baseline": "random map of equal rank (scalar gain only) and shuffled-pairing fit of equal capacity",
    "ablation": "rank sweep (8 vs 32); primary rank preregistered as 32",
    "null_hypothesis": "a random map of equal rank predicts the target as well as the learned map "
    "(alignment-artifact) for every substrate pair",
    "tier": "cpu-now (pilot; the registered claim is studio)",
}

# Real arms only: mapping a pretrained substrate onto a random-init one is a context row, not a claim.
DEFAULT_ARMS = [
    "vjepa2_vitl_nuisance_real",
    "vjepa2_vitl_singleframe",
    "dinov2s_nuisance_real",
    "qwen05b_textified_real",
    "wav2vec2_sonified_real",
    "handcrafted_descriptors",
]
RANKS = [8, 32]
PRIMARY_RANK = 32
RIDGE_LAMBDA = 1e-2


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = train.mean(dim=0), train.std(dim=0) + 1e-6
    return (train - mu) / sd, (other - mu) / sd


def ridge_fit(xa: torch.Tensor, xb: torch.Tensor, lam: float = RIDGE_LAMBDA) -> torch.Tensor:
    """W = argmin ||xa W - xb||^2 + lam ||W||^2, closed form. [Da,Db]."""
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


def learned_vs_floors(
    xa: torch.Tensor, xb: torch.Tensor, seed: int, rank: int, test_frac: float = 0.3
) -> dict:
    """One seed, one rank: learned rank-k map R^2 vs the two preregistered floors."""
    g = torch.Generator().manual_seed(seed)
    n = xa.shape[0]
    perm = torch.randperm(n, generator=g)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]
    atr, ate = _standardize(xa[tr], xa[te])
    btr, bte = _standardize(xb[tr], xb[te])

    w = rank_truncate(ridge_fit(atr, btr), rank)
    learned = r2_score(ate @ w, bte)

    gmap = torch.randn(xa.shape[1], rank, generator=g) @ torch.randn(rank, xb.shape[1], generator=g)
    gmap = gmap / (xa.shape[1] * rank) ** 0.5
    ptr = atr @ gmap
    alpha = float((ptr * btr).sum() / ((ptr * ptr).sum() + 1e-8))  # scalar gain, the only fit
    random_map = r2_score(alpha * (ate @ gmap), bte)

    shuf = torch.randperm(cut, generator=g)
    w_s = rank_truncate(ridge_fit(atr[shuf], btr), rank)
    shuffled_fit = r2_score(ate @ w_s, bte)

    floor = max(random_map, shuffled_fit)
    return {
        "learned_r2": round(learned, 4),
        "random_map_r2": round(random_map, 4),
        "shuffled_fit_r2": round(shuffled_fit, 4),
        "delta": round(learned - floor, 4),
    }


def classify_pair(deltas: list[float]) -> dict:
    """Decision order for one pair: within seed spread of the floor -> alignment-artifact; sign flips
    -> non-replicating; else genuine shared structure."""
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    if ci["lo"] <= 0:
        verdict = "alignment-artifact"
    elif flips["any_flip"]:
        verdict = "non-replicating"
    else:
        verdict = "genuine-shared-structure"
    return {"verdict": verdict, "delta_ci": ci, "sign_flips": flips}


def pair_report(xa: torch.Tensor, xb: torch.Tensor, seeds: list[int], ranks: list[int]) -> dict:
    by_rank = {}
    for k in ranks:
        runs = [learned_vs_floors(xa, xb, s, k) for s in seeds]
        by_rank[k] = {
            "per_seed": runs,
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
        # not evaluable with no pairs on disk: None, never a rejected null
        "null_supported": (
            all(p["verdict"] == "alignment-artifact" for p in pairs.values()) if pairs else None
        ),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AL2 shared-latent alignment vs random-map floor (pilot)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/al2_alignment_pilot.json")
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
