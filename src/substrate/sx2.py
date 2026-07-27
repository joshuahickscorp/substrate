"""SX2: does a set of perspectives beat the strongest single one at matched compute.

The perspective set is the 76 cells of the temporal factorial. Each is a different architecture, capacity,
readout, horizon and reset schedule, each already ran on the same held out units, and each has a sealed per
unit score. Nobody built them to be a perspective set, which is the property that makes this admissible
where a synthetic bed would not be.

The bar is severe and it is the whole point. Selecting among k cells costs k times one cell, so the honest
comparison is a set of k against the best single cell given k times its budget. In this factorial that
alternative exists on the shelf: the cells differ in capacity and training budget, so a bigger single cell
is a real compute matched competitor rather than a hypothetical one.

If oracle selection over the set does not beat the strongest matched single alternative by the SESOI, SX2
closes. No selector is trained, no arbitration canary runs, and the closure is the result.

House style: no dashes.
"""

from __future__ import annotations

import itertools
import json
import random
import statistics
import sys
from collections import defaultdict

from substrate import evidence as io
from substrate import historical

SESOI = 0.05
K_VALUES = (1, 2, 4, 8, 16)
TEMPORAL_RUNS = historical.root("temporal_receipts")
BEDS = ("har_stream", "harth_stream")


class Refused(RuntimeError):
    """An SX2 comparison the design will not make."""


def cell_scores(bed: str) -> dict:
    """{cell: {unit: accuracy}} from the sealed principal receipts, plus the cost of each cell."""
    per_unit: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    cost: dict[str, list] = defaultdict(list)
    for path in sorted((TEMPORAL_RUNS / "e2_principal").glob(f"{bed}_*.json")):
        doc = json.loads(path.read_text())
        for row in doc["runs"]:
            for unit, acc in (row.get("per_unit_accuracy") or {}).items():
                per_unit[row["cell"]][unit].append(float(acc))
            params = (row.get("params") or {}).get("total")
            if params:
                cost[row["cell"]].append(float(params) * float(row.get("updates") or 0))
    scores = {c: {u: statistics.fmean(v) for u, v in units.items()} for c, units in per_unit.items()}
    costs = {c: statistics.fmean(v) for c, v in cost.items() if v}
    return {"scores": scores, "costs": costs}


def diversity_statistics(scores: dict) -> dict:
    """Correlation, complementarity and disagreement over the shared units."""
    cells = sorted(scores)
    units = sorted(set.intersection(*(set(scores[c]) for c in cells)))
    if len(units) < 3:
        raise Refused(f"only {len(units)} shared units, too few for any correlation")
    vectors = {c: [scores[c][u] for u in units] for c in cells}

    pairs, complementary, disagreement = [], 0, []
    for a, b in itertools.combinations(cells, 2):
        x, y = vectors[a], vectors[b]
        try:
            r = statistics.correlation(x, y)
        except statistics.StatisticsError:
            r = float("nan")
        pairs.append(r)
        # complementary means each wins on some unit the other loses
        a_wins = sum(1 for i in range(len(units)) if x[i] > y[i])
        b_wins = sum(1 for i in range(len(units)) if y[i] > x[i])
        if a_wins and b_wins:
            complementary += 1
        disagreement.append(statistics.fmean(abs(x[i] - y[i]) for i in range(len(units))))

    finite = [r for r in pairs if r == r]
    return {
        "n_cells": len(cells),
        "n_units": len(units),
        "mean_pairwise_correlation": round(statistics.fmean(finite), 6) if finite else None,
        "min_pairwise_correlation": round(min(finite), 6) if finite else None,
        "complementary_pairs": complementary,
        "total_pairs": len(pairs),
        "complementary_fraction": round(complementary / max(len(pairs), 1), 6),
        "mean_pairwise_disagreement": round(statistics.fmean(disagreement), 6),
    }


def _mean_over_units(scores: dict, cell: str, units: list) -> float:
    return statistics.fmean(scores[cell][u] for u in units)


def set_value(scores: dict, cells: list, units: list, rule: str) -> float:
    """What a set of cells is worth under a combination rule, per unit."""
    if rule == "oracle":
        return statistics.fmean(max(scores[c][u] for c in cells) for u in units)
    if rule == "mean_vote":
        return statistics.fmean(statistics.fmean(scores[c][u] for c in cells) for u in units)
    if rule == "reliability_weighted":
        weights = {c: _mean_over_units(scores, c, units) for c in cells}
        total = sum(weights.values()) or 1.0
        return statistics.fmean(sum(scores[c][u] * weights[c] for c in cells) / total for u in units)
    raise Refused(f"unknown combination rule {rule!r}")


def matched_single(scores: dict, costs: dict, units: list, budget: float) -> dict:
    """The strongest single cell that costs no more than the set. This is the bar."""
    affordable = [c for c in scores if costs.get(c, float("inf")) <= budget]
    if not affordable:
        return {"cell": None, "value": None, "n_affordable": 0}
    best = max(affordable, key=lambda c: _mean_over_units(scores, c, units))
    return {
        "cell": best,
        "value": round(_mean_over_units(scores, best, units), 6),
        "n_affordable": len(affordable),
    }


def run(bed: str = "harth_stream", seed: int = 0) -> dict:
    data = cell_scores(bed)
    scores, costs = data["scores"], data["costs"]
    if len(scores) < 16:
        raise Refused(f"only {len(scores)} cells with per unit scores")
    units = sorted(set.intersection(*(set(v) for v in scores.values())))
    cells = sorted(scores)
    stats = diversity_statistics(scores)

    ranked = sorted(cells, key=lambda c: -_mean_over_units(scores, c, units))
    best_single = {
        "cell": ranked[0],
        "value": round(_mean_over_units(scores, ranked[0], units), 6),
        "cost": round(costs.get(ranked[0], 0.0), 1),
    }
    rng = random.Random(seed)

    rows = {}
    max_k = min(len(cells), 32)
    for k in [k for k in K_VALUES if k <= len(cells)] + ([max_k] if max_k not in K_VALUES else []):
        top = ranked[:k]
        # a diversity set: greedily add the cell least correlated with what is already chosen
        diverse = [ranked[0]]
        while len(diverse) < k:
            rest = [c for c in cells if c not in diverse]
            pick = min(
                rest,
                key=lambda c: statistics.fmean(abs(_mean_over_units(scores, c, units) - _mean_over_units(scores, d, units)) for d in diverse) * -1,
            )
            diverse.append(pick)
        randomised = rng.sample(cells, k)

        budget = sum(costs.get(c, 0.0) for c in top)
        matched = matched_single(scores, costs, units, budget)
        rows[k] = {
            "k": k,
            "set_cost": round(budget, 1),
            "oracle_top_k": round(set_value(scores, top, units, "oracle"), 6),
            "oracle_diverse_k": round(set_value(scores, diverse, units, "oracle"), 6),
            "oracle_random_k": round(set_value(scores, randomised, units, "oracle"), 6),
            "mean_vote_top_k": round(set_value(scores, top, units, "mean_vote"), 6),
            "reliability_weighted_top_k": round(set_value(scores, top, units, "reliability_weighted"), 6),
            "best_single_unmatched": best_single["value"],
            "strongest_compute_matched_single": matched,
        }
        oracle = rows[k]["oracle_top_k"]
        bar = matched["value"] if matched["value"] is not None else best_single["value"]
        rows[k]["margin_over_matched_single"] = round(oracle - bar, 6)
        rows[k]["clears_sesoi"] = (oracle - bar) > SESOI

    marginal = {}
    ks = sorted(rows)
    for a, b in zip(ks, ks[1:], strict=False):
        marginal[f"{a}_to_{b}"] = round(rows[b]["oracle_top_k"] - rows[a]["oracle_top_k"], 6)

    clearing = [k for k, r in rows.items() if r["clears_sesoi"]]
    verdict = "headroom_exists" if clearing else "closed_no_headroom"
    return {
        "schema": "substrate-sx2-diversity/v1",
        "bed": bed,
        "perspective_set": "the 76 cells of the temporal core factorial",
        "why_admissible": (
            "each cell is a different architecture, capacity, readout, horizon and reset "
            "schedule, each already scored on the same held out units, and none was built "
            "to be a perspective set"
        ),
        "n_cells": len(cells),
        "n_units": len(units),
        "units": units,
        "diversity_statistics": stats,
        "best_single_unmatched": best_single,
        "k_rows": rows,
        "marginal_value_of_k": marginal,
        "sesoi": SESOI,
        "k_clearing_sesoi": clearing,
        "verdict": verdict,
        "compute_matching": (
            "selecting among k cells costs k times one cell, so every comparison is "
            "against the strongest single cell affordable at the same total cost. The "
            "factorial contains larger and longer trained cells, so that alternative is "
            "real rather than hypothetical"
        ),
        "reading": (
            "oracle selection is an upper bound that no selector can exceed. If the upper bound "
            "does not clear the SESOI over a compute matched single cell, no selector built on "
            "this set can, and SX2 closes without training one"
            if not clearing
            else "the upper bound clears the SESOI, so one bounded selector and arbitration canary is licensed"
        ),
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    path = io.seal("SUBSTRATE_SX2_DIVERSITY.json", doc)
    print(
        json.dumps(
            {
                "sealed": path.relative_to(io.ROOT).as_posix(),
                "n_cells": doc["n_cells"],
                "n_units": doc["n_units"],
                "mean_correlation": doc["diversity_statistics"]["mean_pairwise_correlation"],
                "complementary_fraction": doc["diversity_statistics"]["complementary_fraction"],
                "best_single": doc["best_single_unmatched"]["value"],
                "margins": {k: r["margin_over_matched_single"] for k, r in doc["k_rows"].items()},
                "verdict": doc["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
