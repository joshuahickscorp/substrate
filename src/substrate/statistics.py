"""Fixed-horizon, history-clustered analysis. Never treat 10,000 episodes as 10,000 minds.

Bootstrap intervals here are ordinary fixed-sample percentile intervals; they
are NOT time-uniform confidence sequences. Adaptive pilots remain exploratory.
No significance claim, alpha-spending or automatic early stopping is invented.
"""
from __future__ import annotations

import math
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .core import Cost, Trust, clone, digest, fields, integer, numeric, read_json, require, sha
from .odyssey import validate_plan


def quantile(values: list[float], probability: float) -> float:
    require(bool(values), "quantile requires observations")
    numeric(probability, 0, 1)
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_history_interval(differences: list[float], *, seed: int, repetitions: int = 5000) -> dict:
    require(len(differences) >= 2, "paired inference requires at least two independent histories")
    integer(repetitions, 100, 100000)
    for value in differences:
        numeric(value, -1, 1)
    rng = random.Random(seed)
    draws = [stats.fmean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)]
    return {"difference": stats.fmean(differences), "interval": [quantile(draws, .025), quantile(draws, .975)],
            "interval_kind": "fixed-horizon-paired-history-percentile-bootstrap", "histories": len(differences),
            "repetitions": repetitions, "seed": seed, "not_optional_stopping_valid": True,
            "small_sample_warning": len(differences) < 20}


def verify_scores(plan: dict, signed: dict, shards: dict[str, list], trust: Trust) -> list[dict]:
    validate_plan(plan)
    require(plan["status"] == "locked", "analysis requires the preregistered locked plan")
    fields(signed, {"document", "certificate"})
    document = signed["document"]
    fields(document, {"schema", "plan", "shards", "expected", "unit_of_replication"})
    require(document["schema"] == "odyssey-scores-v2" and document["plan"] == sha(plan), "score report belongs to another plan")
    require(document["unit_of_replication"] == "independent-developmental-history", "incorrect replication unit")
    findings = trust.verify(signed["certificate"], purpose="report", domain="micro-world", subject=document,
                            producer="odyssey-runner")
    require(findings.get("complete") is True, "evaluator did not release a complete report")
    require(len(document["shards"]) == len(set(document["shards"])) and set(shards) == set(document["shards"]),
            "missing, duplicate or extra score shards")
    rows = []
    for key in document["shards"]:
        digest(key)
        require(sha(shards[key]) == key, "score shard digest mismatch")
        require(type(shards[key]) is list and 1 <= len(shards[key]) <= 128, "score shard size bound")
        rows.extend(shards[key])
    expected = {(unit, arm, family, split, index) for unit in plan["units"] for arm in plan["arms"]
                for family in plan["families"] for split, count in plan["episodes"].items() if split != "train"
                for index in range(count)}
    actual, ids = set(), set()
    for row in rows:
        fields(row, {"id", "unit", "arm", "family", "split", "index", "correct", "confidence", "mechanism", "cost"})
        key = (row["unit"], row["arm"], row["family"], row["split"], row["index"])
        require(key not in actual and row["id"] not in ids, "duplicate statistical observation")
        require(type(row["correct"]) is bool, "missing outcomes cannot be silently dropped")
        numeric(row["confidence"], 0, 1)
        Cost(**row["cost"])
        actual.add(key)
        ids.add(digest(row["id"]))
    require(actual == expected and len(rows) == document["expected"] == findings["rows"], "full intention-to-treat matrix required")
    return rows


def summarize(rows: list[dict]) -> dict:
    require(bool(rows), "summary requires observations")
    successes = sum(row["correct"] for row in rows)
    answered = [row for row in rows if row["mechanism"] != "abstain"]
    wall = [row["cost"]["elapsed_ns"] for row in rows]
    known_usage = [row for row in rows if row["cost"]["source"] != "usage-unknown-reservation-retained"]
    energy = [row["cost"]["energy_j"] for row in rows]
    memory = [row["cost"]["peak_memory_bytes"] for row in rows]
    return {"trials": len(rows), "correct": successes, "accuracy": successes / len(rows),
            "brier": stats.fmean((row["confidence"] - int(row["correct"]))**2 for row in rows),
            "coverage": len(answered) / len(rows), "selective_accuracy": stats.fmean(r["correct"] for r in answered) if answered else None,
            "native_fraction": sum(r["mechanism"] == "native-skill" for r in rows) / len(rows),
            "cache_fraction": sum(r["mechanism"] == "answer-cache" for r in rows) / len(rows),
            "model_calls": sum(row["cost"]["model_calls"] for row in rows),
            "wall_ns_total": sum(wall), "wall_ns_p50": quantile(wall, .5), "wall_ns_p95": quantile(wall, .95),
            "verified_answers_per_second": successes * 1e9 / sum(wall) if sum(wall) else None,
            "tokens_known_subset": sum(r["cost"]["input_tokens"] + r["cost"]["output_tokens"] for r in known_usage),
            "usage_missing_trials": len(rows) - len(known_usage),
            "energy_j": sum(energy) if all(v is not None for v in energy) else None,
            "peak_memory_bytes": max(memory) if all(v is not None for v in memory) else None,
            "timing_scope": "answer path only; acquisition, oracle and qualification costs reported separately"}


def pareto(summaries: dict[str, dict]) -> list[str]:
    """Accuracy, model calls and wall time remain separate dimensions."""
    result = []
    for name, row in summaries.items():
        dominated = False
        for other_name, other in summaries.items():
            if name == other_name:
                continue
            no_worse = (other["accuracy"] >= row["accuracy"] and other["model_calls"] <= row["model_calls"]
                        and other["wall_ns_total"] <= row["wall_ns_total"])
            strictly = (other["accuracy"] > row["accuracy"] or other["model_calls"] < row["model_calls"]
                        or other["wall_ns_total"] < row["wall_ns_total"])
            dominated |= no_worse and strictly
        if not dominated:
            result.append(name)
    return sorted(result)


def analyze(plan: dict, signed: dict, shards: dict[str, list], trust: Trust, *, acquisition: dict | None = None) -> dict:
    rows = verify_scores(plan, signed, shards, trust)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["arm"], row["family"])].append(row)
    summaries = [{"split": split, "arm": arm, "family": family, **summarize(group),
                  "finite_input_domain": family == "logic"} for (split, arm, family), group in sorted(groups.items())]
    left, right = plan["primary_contrast"]
    units = defaultdict(list)
    for row in rows:
        if row["split"] == "evaluation" and row["arm"] in {left, right}:
            units[(row["unit"], row["arm"])].append(int(row["correct"]))
    differences = [stats.fmean(units[(unit, left)]) - stats.fmean(units[(unit, right)]) for unit in plan["units"]]
    primary = paired_history_interval(differences, seed=int(sha({"plan": sha(plan), "analysis": "bootstrap-v2"})[:16], 16))
    aggregate = {arm: summarize([r for r in rows if r["split"] == "evaluation" and r["arm"] == arm]) for arm in plan["arms"]}
    return {"schema": "odyssey-analysis-v2", "plan": sha(plan), "study_kind": plan["kind"],
            "research_target": plan.get("cognition", {}).get("target", "legacy-mechanism-study"),
            "program": plan.get("cognition", {}).get("program"),
            "target_attainment": "not-determined-by-this-study-analysis",
            "primary": {"contrast": [left, right], "outcome": "evaluation_accuracy", **primary},
            "stratified": summaries, "evaluation_aggregate": aggregate, "answer_path_pareto": pareto(aggregate),
            "acquisition": clone(acquisition), "all_planned_final_trials_retained": True,
            "interpretation": {"pilot_is_exploratory": plan["kind"] == "pilot", "AGI": "not-inferred",
                "philosophical_identity": "not-inferred", "memory_vs_structure": "requires-ablation-and-cost-interpretation",
                "secondary_contrasts": "descriptive; no unregistered multiple-testing significance claims",
                "small_finite_domains": "logic tests exhaustive reuse, not unseen-input generalization",
                "hardware_claims": "unknown energy/memory never reported as zero"}}


def read_score_directory(directory: Path) -> tuple[dict, dict[str, list]]:
    signed = read_json(directory / "manifest.json")
    shards = {key: read_json(directory / "shards" / (digest(key) + ".json")) for key in signed["document"]["shards"]}
    return signed, shards
