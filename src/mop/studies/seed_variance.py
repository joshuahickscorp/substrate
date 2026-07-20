
from __future__ import annotations

import math
from pathlib import Path

from ..config import compose
from ..devices import resolve
from ..experiments.e1_baseline_harness import E1

SEED_GRID = (1, 2, 3, 5, 8)  # representative power ladder for the rerun
CLAIM_TOL = {"headline_claim": 0.02, "ranking_claim": 0.05, "sanity_claim": 0.10}
MEAN_DELTA_FRAC = 0.10

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
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) / math.sqrt(n)


def _aggregate(per_seed: list[dict], grid: tuple[int, ...]) -> list[dict]:
    rows = []
    for S in grid:
        gaps = [r["gap"] for r in per_seed[:S]]
        m, se = _mean(gaps), _sem(gaps)
        sign_stable = S >= 2 and m > 0.0 and (m - se) > 0.0
        rows.append({"S": S, "mean_gap": m, "sem": se, "sign_stable": bool(sign_stable), "n_gaps": S})
    return rows


def _recommend(per_S: list[dict], tol: float) -> int:
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
