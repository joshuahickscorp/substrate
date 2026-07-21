"""Experiment value of information.

Selection by architectural appeal is how this program spent four campaigns on premises that were already
closed. The queue below is a transparent prioritization aid, not a truth claim: the arithmetic is simple on
purpose so that a wrong ranking is visibly wrong rather than authoritative.

The one rule that is not advisory: an experiment that repeats a closed premise is refused, whatever it
scores.

House style: no dashes.
"""

from __future__ import annotations

# every dimension is scored 0 to 1 by the proposer and must be justified in the record
DIMENSIONS = (
    "expected_information_gain",
    "probability_of_changing_the_substrate_decision",
    "compute_cost",
    "duration_cost",
    "instrumentation_risk",
    "baseline_uncertainty",
    "oracle_headroom",
    "independent_unit_quality",
    "reusability_of_implementation",
    "reusability_of_data",
    "discriminates_competing_hypotheses",
    "risk_of_repeating_a_closed_premise",
)

BENEFIT = (
    "expected_information_gain",
    "probability_of_changing_the_substrate_decision",
    "discriminates_competing_hypotheses",
    "oracle_headroom",
    "independent_unit_quality",
    "reusability_of_implementation",
    "reusability_of_data",
)
COST = ("compute_cost", "duration_cost", "instrumentation_risk", "baseline_uncertainty")


def score(candidate: dict) -> dict:
    s = candidate["scores"]
    missing = [d for d in DIMENSIONS if d not in s]
    if missing:
        raise ValueError(f"{candidate['id']}: unscored dimensions {missing}")
    # decision information is dominated by whether the result can move the decision and separate hypotheses
    decision_information = (
        0.4 * s["expected_information_gain"]
        + 0.35 * s["probability_of_changing_the_substrate_decision"]
        + 0.25 * s["discriminates_competing_hypotheses"]
    )
    reuse = 0.5 * (s["reusability_of_implementation"] + s["reusability_of_data"])
    cost = (
        0.4 * s["compute_cost"]
        + 0.2 * s["duration_cost"]
        + 0.25 * s["instrumentation_risk"]
        + 0.15 * s["baseline_uncertainty"]
    )
    feasibility = 0.5 * (s["oracle_headroom"] + s["independent_unit_quality"])
    closed = s["risk_of_repeating_a_closed_premise"]
    priority = round(decision_information * (0.6 + 0.4 * feasibility) * (0.7 + 0.3 * reuse) / max(0.05, cost), 3)
    refused = closed >= 0.5 or s["oracle_headroom"] <= 0.0
    return {
        "id": candidate["id"],
        "title": candidate["title"],
        "question": candidate["question"],
        "scores": s,
        "decision_information": round(decision_information, 3),
        "cost_index": round(cost, 3),
        "feasibility": round(feasibility, 3),
        "reuse": round(reuse, 3),
        "priority": 0.0 if refused else priority,
        "raw_priority": priority,
        "status": "refused_closed_premise" if refused else "eligible",
        "refusal_reason": (
            "risk of repeating a closed premise is at or above one half"
            if closed >= 0.5
            else ("no oracle headroom" if s["oracle_headroom"] <= 0.0 else "")
        ),
        "hypotheses_separated": candidate.get("hypotheses_separated", []),
        "justification": candidate.get("justification", {}),
    }


def queue(candidates: list[dict], select: int = 2) -> dict:
    scored = [score(c) for c in candidates]
    scored.sort(key=lambda r: (-r["priority"], r["cost_index"]))
    eligible = [r for r in scored if r["status"] == "eligible"]
    chosen = eligible[:select]
    # a pair that separates the same hypotheses twice buys less than a pair that spans them
    covered = sorted({h for r in chosen for h in r["hypotheses_separated"]})
    return {
        "schema": "mop-experiment-value-queue/v1",
        "dimensions": list(DIMENSIONS),
        "formula": (
            "priority = decision_information * (0.6 + 0.4 * feasibility) * (0.7 + 0.3 * reuse) / cost_index, "
            "refused when the closed premise risk is at or above one half or the oracle headroom is zero"
        ),
        "candidates": scored,
        "selected": [r["id"] for r in chosen],
        "selected_detail": chosen,
        "refused": [r["id"] for r in scored if r["status"] != "eligible"],
        "hypotheses_covered_by_selection": covered,
        "caveat": "the arithmetic is a prioritization aid and is not itself scientific evidence",
    }
