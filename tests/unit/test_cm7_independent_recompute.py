"""Independent recomputation gate for the CM7 objective-selection null.

Born from the 2026-07-10 independent re-verification (docs/INDEPENDENT_REVERIFICATION_2026_07_10.md):
the CM7 branch-retiring verdict must never rest on the verifier that froze it. This test re-derives
every verdict-determining statistic from the RAW workbench receipt using independently implemented
math (exact closed-form Student-t CDF for df=4, textbook Holm step-down, bisection ppf for the
simultaneous Bonferroni-t critical value; no repo statistics code imported) and fails if the
canonical chain ever drifts from what the raw data supports.

Pure json plus math over committed receipts: milliseconds, no model, no randomness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "proof/CUSTOM_SUBSTRATE_PILOT_CHAIN/raw_workbench_receipt.json"
VER = ROOT / "proof/CUSTOM_SUBSTRATE_PILOT_CHAIN/independent_verifier.json"

CANDIDATES = ("predictive", "invariance", "reconstruction")
CONTROLS = ("random_target", "frozen_random")
MARGIN = 0.03
ALPHA = 0.05

pytestmark = pytest.mark.skipif(not (RAW.exists() and VER.exists()), reason="CM7 chain receipts not present")


def _t_cdf_df4(t: float) -> float:
    """Exact Student-t CDF for df=4: I_x(2, 1/2) has the closed form below (direct integration)."""
    df = 4.0
    x = df / (df + t * t)
    ib = 1.0 - 1.5 * x * math.sqrt(1.0 - x) - (1.0 - x) ** 1.5
    tail = 0.5 * ib
    return 1.0 - tail if t >= 0 else tail


def _t_ppf_df4(p: float) -> float:
    lo, hi = 0.0, 1.0e6
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf_df4(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _holm(raw_p: dict[str, float]) -> dict[str, float]:
    m = len(raw_p)
    ordered = sorted(raw_p, key=lambda k: raw_p[k])
    adj: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * raw_p[key]))
        adj[key] = running
    return adj


def _recompute() -> tuple[dict, dict, list[str]]:
    """Return (recomputed, canonical_verifier, raw_scan_problems)."""
    raw = json.loads(RAW.read_text())
    ver = json.loads(VER.read_text())
    seeds = sorted(raw["seed_results"], key=int)
    arms = CANDIDATES + CONTROLS

    problems: list[str] = []
    if len(seeds) != 5:
        problems.append(f"seed count {len(seeds)} != 5")
    scores: dict[str, list[float]] = {a: [] for a in arms}
    for s in seeds:
        rec = raw["seed_results"][s]
        for a in arms:
            v = rec[a]["evaluation"].get("heldout_combo_score")
            if v is None or math.isnan(float(v)) or math.isinf(float(v)):
                problems.append(f"seed {s} arm {a} bad heldout score {v!r}")
                v = 0.0
            scores[a].append(float(v))
            tr = rec[a].get("training")
            if a != "frozen_random" and tr is not None:
                if tr.get("complete") is not True:
                    problems.append(f"seed {s} arm {a} training incomplete")
                if tr.get("completed_steps") != tr.get("requested_steps"):
                    problems.append(f"seed {s} arm {a} truncated")

    means = {a: sum(v) / len(v) for a, v in scores.items()}
    raw_winner = max(CANDIDATES, key=lambda a: means[a])

    comparisons: dict[str, dict] = {}
    for cand in CANDIDATES:
        for opp in (*CONTROLS, *(a for a in CANDIDATES if a != cand)):
            deltas = [x - y for x, y in zip(scores[cand], scores[opp], strict=True)]
            n = len(deltas)
            mean = sum(deltas) / n
            sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
            se = sd / math.sqrt(n)
            if se == 0.0:
                t = math.inf if mean > MARGIN else -math.inf
                p = 0.0 if mean > MARGIN else 1.0
            else:
                t = (mean - MARGIN) / se
                p = 1.0 - _t_cdf_df4(t)
            comparisons[f"{cand}_vs_{opp}"] = {
                "mean_delta": mean,
                "se": se,
                "raw_one_sided_p": p,
            }

    family_size = len(comparisons)
    adj = _holm({k: v["raw_one_sided_p"] for k, v in comparisons.items()})
    crit = _t_ppf_df4(1.0 - ALPHA / family_size)
    for key, row in comparisons.items():
        row["holm_adjusted_p"] = adj[key]
        row["simultaneous_lower_bound"] = row["mean_delta"] - crit * row["se"]
        row["clears_margin"] = row["simultaneous_lower_bound"] > MARGIN and row["holm_adjusted_p"] < ALPHA

    winner_clears = all(
        row["clears_margin"] for key, row in comparisons.items() if key.startswith(raw_winner)
    )
    recomputed = {
        "raw_winner": raw_winner,
        "means": means,
        "family_size": family_size,
        "simultaneous_t_critical": crit,
        "comparisons": comparisons,
        "winner_clears_all_corrected_comparisons": winner_clears,
    }
    return recomputed, ver, problems


def test_raw_receipt_is_clean() -> None:
    _, _, problems = _recompute()
    assert problems == [], f"raw workbench receipt scan problems: {problems}"


def test_recomputed_family_matches_canonical_verifier() -> None:
    mine, ver, _ = _recompute()
    sel = ver["selection"]
    assert sel["raw_winner"] == mine["raw_winner"]
    assert sel["family_size"] == mine["family_size"]
    assert abs(sel["simultaneous_t_critical"] - mine["simultaneous_t_critical"]) < 1.0e-7
    vcomp = ver["paired_comparisons"]
    assert set(vcomp) == set(mine["comparisons"])
    for key, row in mine["comparisons"].items():
        theirs = vcomp[key]
        assert abs(row["mean_delta"] - theirs["mean_delta"]) < 1.0e-9, key
        assert abs(row["holm_adjusted_p"] - theirs["holm_adjusted_p"]) < 1.0e-7, key
        assert abs(row["simultaneous_lower_bound"] - theirs["simultaneous_lower_bound"]) < 1.0e-7, key
        assert row["clears_margin"] == theirs["clears_margin"], key


def test_not_promoted_verdict_reproduces_from_raw_data() -> None:
    mine, ver, _ = _recompute()
    # the three killing facts: winner clears nothing, and both untrained controls beat it
    assert mine["winner_clears_all_corrected_comparisons"] is False
    assert mine["comparisons"][f"{mine['raw_winner']}_vs_random_target"]["mean_delta"] < 0.0
    assert mine["comparisons"][f"{mine['raw_winner']}_vs_frozen_random"]["mean_delta"] < 0.0
    assert ver["promotion"] is False
    assert ver["verdict"] == "not-promoted"
    assert ver["gates"]["winner_clears_all_corrected_comparisons"] is False
    assert ver["verification_complete"] is True
    assert ver["null_valid"] is True
    assert ver["all_ok"] is False


def test_compute_match_reproduces() -> None:
    raw = json.loads(RAW.read_text())
    seeds = sorted(raw["seed_results"], key=int)
    flops = {}
    for a in (*CANDIDATES, "random_target"):
        per_seed = [raw["seed_results"][s][a]["training"]["compute"]["estimated_total_flops"] for s in seeds]
        flops[a] = sum(per_seed) / len(per_seed)
    ref = sum(flops.values()) / len(flops)
    max_dev = max(abs(v - ref) / ref for v in flops.values())
    assert abs(max_dev - raw["compute_match"]["max_fractional_deviation"]) < 1.0e-12
    assert max_dev <= raw["compute_match"]["tolerance_fraction"]
