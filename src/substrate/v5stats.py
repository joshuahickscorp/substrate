"""Deterministic statistical utilities for the Substrate v5 campaign.

The developmental history is always the independent unit.  These helpers keep
raw paired effects in every result so an independent verifier never has to
trust a precomputed summary.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Iterable, Mapping


class Refused(RuntimeError):
    """A statistical claim could not be evaluated from valid independent units."""


def _seed(identity: str) -> int:
    return int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Refused("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise Refused("percentile probability must be in [0, 1]")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def exact_sign_p(values: Iterable[float]) -> float:
    nonzero = [float(value) for value in values if float(value) != 0.0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0.0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(math.comb(len(nonzero), count) for count in range(tail + 1))
    probability /= 2 ** len(nonzero)
    return min(1.0, 2.0 * probability)


def paired_effect(
    values: Iterable[float],
    endpoint: str,
    *,
    sesoi: float,
    bootstraps: int = 2_000,
) -> dict[str, object]:
    raw = [float(value) for value in values]
    if not raw:
        raise Refused(f"paired endpoint {endpoint!r} has no histories")
    if sesoi < 0:
        raise Refused("SESOI must be non-negative")
    if bootstraps < 100:
        raise Refused("at least 100 bootstrap draws are required")
    rng = random.Random(_seed(endpoint))
    draws = [
        statistics.fmean(raw[rng.randrange(len(raw))] for _ in raw)
        for _ in range(bootstraps)
    ]
    mean = statistics.fmean(raw)
    deviation = statistics.stdev(raw) if len(raw) > 1 else 0.0
    confidence_interval = [percentile(draws, 0.025), percentile(draws, 0.975)]
    return {
        "endpoint": endpoint,
        "n": len(raw),
        "independent_unit": "developmental_history",
        "raw_paired_effects": raw,
        "mean": mean,
        "median": statistics.median(raw),
        "bootstrap_95_ci": confidence_interval,
        "exact_sign_p": exact_sign_p(raw),
        "standardized_effect": (
            mean / deviation if deviation else ("infinity" if mean else 0.0)
        ),
        "sesoi": sesoi,
        "clears_sesoi": mean >= sesoi and confidence_interval[0] > 0.0,
        "activation": False,
    }


def paired_contrast(
    full: Mapping[int, float],
    controls: Mapping[str, Mapping[int, float]],
    endpoint: str,
    *,
    sesoi: float,
) -> dict[str, object]:
    if not controls:
        raise Refused(f"endpoint {endpoint!r} has no controls")
    histories = sorted(full)
    if any(set(control) != set(histories) for control in controls.values()):
        raise Refused(f"endpoint {endpoint!r} has unmatched developmental histories")
    raw = []
    strongest = []
    for history in histories:
        values = {name: float(rows[history]) for name, rows in controls.items()}
        strongest_name = max(values, key=lambda name: (values[name], name))
        strongest.append(strongest_name)
        raw.append(float(full[history]) - values[strongest_name])
    result = paired_effect(raw, endpoint, sesoi=sesoi)
    result.update(
        {
            "full_arm": "full_v5",
            "controls": sorted(controls),
            "strongest_control_by_history": strongest,
        }
    )
    return result


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, object]:
    if not p_values:
        raise Refused("Holm correction requires at least one hypothesis")
    ordered = sorted(p_values, key=lambda name: (float(p_values[name]), name))
    rows: dict[str, dict[str, float | bool]] = {}
    rejecting = True
    for index, name in enumerate(ordered):
        raw = float(p_values[name])
        if not 0.0 <= raw <= 1.0:
            raise Refused(f"invalid p-value for {name}")
        threshold = alpha / (len(ordered) - index)
        rejected = rejecting and raw <= threshold
        if not rejected:
            rejecting = False
        rows[name] = {
            "raw_p": raw,
            "holm_threshold": threshold,
            "reject_zero": rejected,
        }
    return {
        "family": ordered,
        "alpha": alpha,
        "method": "Holm",
        "rows": rows,
        "activation": False,
    }
