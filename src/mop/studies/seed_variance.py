"""Leg 11B: seed-variance and statistical power for the headline E1 retention claim.

E1 (the gate) is rerun across seeds; the quantity that carries the headline is the
protected-vs-naive backward-transfer gap (protected_bwt - naive_bwt, positive => the
replay+EWC arm retains better). With one seed that gap is a coin flip dressed as a result.
This leg measures how the gap's mean and standard-error-of-the-mean (SEM) behave as the
seed count S grows over {1,2,3,5,8}, and derives the smallest S at which the claim is
trustworthy at three strengths (headline, ranking, sanity) FROM the measured SEM curve,
not from a hardcoded budget. That recommendation sets the Studio seed budget.

A claim is sound at seed-count S when BOTH hold: (a) the sign of the per-seed gap is stable
(every seed agrees the protected arm retains better, so the direction is not seed-luck), and
(b) the mean has stabilized, meaning the SEM has fallen below a per-claim tolerance OR the
successive-S mean has stopped moving. Headline needs the tightest tolerance, sanity the
loosest. Synthetic latents => provenance "provisional".
"""

from __future__ import annotations

import math
from pathlib import Path

from ..config import compose
from ..devices import resolve
from ..experiments.e1_baseline_harness import E1

SEED_GRID = (1, 2, 3, 5, 8)  # representative power ladder for the rerun
# SEM tolerance per claim strength: headline must pin the magnitude, sanity only the sign.
CLAIM_TOL = {"headline_claim": 0.02, "ranking_claim": 0.05, "sanity_claim": 0.10}
# fallback stabilization signal: successive-S mean moves less than this fraction of the mean.
MEAN_DELTA_FRAC = 0.10

# Toy E1 scale that still produces the forgetting(naive)/retention(protected) signal with
# real cross-seed variance, in well under a second per seed on cpu. Low separation +
# domain-incremental forces interference so the gap is nonzero and seed-dependent.
TOY_OVERRIDES = [
    "experiment.stream.dim=48",
    "experiment.stream.n_tasks=5",
    "experiment.stream.samples_per_task=96",
    "experiment.stream.classes_per_task=4",
    "experiment.stream.separation=0.4",
    "experiment.train.epochs_per_task=4",
    "experiment.train.batch_size=32",
]


def _e1_gap(seed: int, toy: bool, run_dir: Path) -> dict:
    """One E1 run at a fixed seed. Returns the per-seed BWT for each arm and their gap."""
    overrides = ["experiment=e1_baseline", "device=cpu", f"seed={seed}"]
    if toy:
        overrides += TOY_OVERRIDES
    cfg = compose(overrides)
    out = E1().run(cfg, resolve("cpu"), run_dir)
    g = out["gate"]
    naive, prot = float(g["naive_bwt"]), float(g["protected_bwt"])
    return {"seed": seed, "naive_bwt": naive, "protected_bwt": prot, "gap": prot - naive}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sem(xs: list[float]) -> float:
    """Standard error of the mean: sample-std / sqrt(n). Zero for a single seed (no spread
    is measurable yet, which is exactly why one seed cannot support a claim)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def _aggregate(per_seed: list[dict], grid: tuple[int, ...]) -> list[dict]:
    """For each S in the grid, aggregate the first S seeds' gaps into mean/SEM/sign-stable."""
    rows = []
    for S in grid:
        gaps = [r["gap"] for r in per_seed[:S]]
        m, se = _mean(gaps), _sem(gaps)
        # sign stable: the protected arm retains better in aggregate AND that direction
        # survives sampling noise (mean is at least one SEM clear of zero, so a tie/dip on a
        # single seed cannot have flipped it). A single seed is never sign-stable: SEM is
        # undefined for n=1, so one observation can never establish a direction.
        sign_stable = S >= 2 and m > 0.0 and (m - se) > 0.0
        rows.append({"S": S, "mean_gap": m, "sem": se, "sign_stable": bool(sign_stable), "n_gaps": S})
    return rows


def _recommend(per_S: list[dict], tol: float) -> int:
    """Smallest S that is sign-stable AND has its mean stabilized at this tolerance.

    Stabilized := SEM below tol, OR the step from the previous grid point moved the mean by
    less than MEAN_DELTA_FRAC of its magnitude (the SEM-curve has flattened). Derived from the
    MEASURED curve; if nothing in the grid clears the bar, recommend the largest S tested.
    """
    prev = None
    for row in per_S:
        S, m, se, sign_ok = row["S"], row["mean_gap"], row["sem"], row["sign_stable"]
        sem_ok = se <= tol
        delta_ok = prev is not None and abs(m - prev) <= MEAN_DELTA_FRAC * max(abs(m), 1e-9)
        if sign_ok and (sem_ok or delta_ok):
            return int(S)
        prev = m
    return int(per_S[-1]["S"])


def seed_variance(max_seeds: int = 8, toy: bool = True) -> dict:
    """Rerun E1 across seeds 0..max_seeds-1, then characterize the protected-vs-naive BWT gap
    as a function of seed-count S. Returns per-S mean/SEM/sign-stability and a measured-curve
    recommendation of the seed budget for three claim strengths."""
    if max_seeds < 1:
        raise ValueError("max_seeds must be >= 1")
    grid = tuple(S for S in SEED_GRID if max_seeds >= S) or (max_seeds,)
    needed = max(grid)
    run_dir = Path("runs/seed_variance")
    per_seed = [_e1_gap(s, toy, run_dir) for s in range(needed)]
    per_S = _aggregate(per_seed, grid)
    recommended = {claim: _recommend(per_S, tol) for claim, tol in CLAIM_TOL.items()}
    return {
        "per_S": per_S,
        "per_seed": per_seed,
        "recommended_seeds": recommended,
        "claim_tolerances": CLAIM_TOL,
        "seed_grid": list(grid),
        "provenance": "provisional",
    }


def main() -> int:
    import json
    import sys

    print("[seed_variance] rerunning E1 across the seed ladder (toy cpu scale)", file=sys.stderr)
    out = seed_variance()
    for row in out["per_S"]:
        print(
            f"[seed_variance] S={row['S']} mean_gap={row['mean_gap']:+.4f} "
            f"sem={row['sem']:.4f} sign_stable={row['sign_stable']}",
            file=sys.stderr,
        )
    print(f"[seed_variance] recommended_seeds={out['recommended_seeds']}", file=sys.stderr)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
