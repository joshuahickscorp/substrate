"""Deterministic paired statistics for developmental histories.

"""

from __future__ import annotations

import hashlib
import math
import random
import statistics

from substrate import v2config as C


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), count) for count in range(tail + 1)) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def paired(values: list[float], endpoint: str) -> dict:
    if not values:
        raise ValueError(f"paired endpoint {endpoint!r} has no independent units")
    digest = int(hashlib.sha256(endpoint.encode()).hexdigest()[:8], 16)
    rng = random.Random(C.STATISTICS["bootstrap_seed"] + digest)
    bootstraps = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(C.STATISTICS["bootstrap_repetitions"])
    ]
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standardized = statistics.fmean(values) / deviation if deviation else (math.inf if statistics.fmean(values) else 0.0)
    return {
        "endpoint": endpoint,
        "n": len(values),
        "raw_paired_effects": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "bootstrap_95_ci": [_percentile(bootstraps, 0.025), _percentile(bootstraps, 0.975)],
        "exact_sign_p": exact_sign_p(values),
        "standardized_effect": standardized,
        "sesoi": C.SESOI,
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    rows = {}
    still_rejecting = True
    for index, name in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        rejected = still_rejecting and p_values[name] <= threshold
        if not rejected:
            still_rejecting = False
        rows[name] = {
            "raw_p": p_values[name],
            "holm_threshold": threshold,
            "reject_zero": rejected,
        }
    return {
        "family": ordered,
        "alpha": alpha,
        "method": "Holm",
        "rows": rows,
    }
