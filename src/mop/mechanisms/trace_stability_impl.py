"""Deterministic seeded reference implementation of the cross-seed trace-stability metric.

This module computes the metric the trace_stability epoch measures: cross-seed agreement of a
candidate cognitive trace under the scaffold's declared rank-correlation metric. It reuses the
scaffold's metric computation (cross_seed_agreement) and its declared metric vocabulary; it never
edits or re-declares the contract or the bar.

Two regimes are synthesized deterministically from a seed and never from the wall clock or any
unseeded source. The favorable regime encodes a real cross-seed structure: a shared canonical
score vector presented under a per-seed coordinate permutation (a per-seed labeling nuisance, like
the arbitrary sign or axis order of an embedding). Decoding each seed by its own declared key
recovers the same canonical ranking, so cross-seed agreement is high. The null regime carries no
shared structure, so recovered rankings vary across seeds and agreement stays at chance.

The four declared controls each destroy the structure in a distinct, honest way, so on a genuinely
stable favorable trace every control falls to chance:
  single-seed: only one distinct seed, so there is no cross-seed pair to compare, and agreement
    falls to the chance floor (the exact few-seed regime the null indicts).
  shuffled-seed: decode each seed with a neighbor seed's key, breaking the seed-to-measurement
    binding, so recovered rankings scatter across seeds.
  permuted-trace: permute each seed's presented values before decoding, so recovered rankings
    scatter across seeds.
  label-shuffled: relabel each seed's recovered items independently, destroying the identity to
    value alignment across seeds.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..substrate.events import canonical_sha256
from .trace_stability_scaffold import (
    STABILITY_METRICS,
    TraceRecord,
    cross_seed_agreement,
)

MECHANISM_ID = "trace_stability"
STABILITY_METRIC = "rank-correlation"
TRACE_ID = "trace.candidate"

# The number of ranked items in a per-seed trace. Kept fixed so the metric is deterministic.
K_ITEMS = 8
# The toy agreement threshold a favorable regime must clear. Chance sits near 0.5.
THRESHOLD = 0.9
# Agreement returned when no cross-seed pair exists; equals the rank-correlation chance level.
CHANCE_FLOOR = 0.5

# A fixed, distinct canonical score vector. Its induced ranking is the shared cross-seed structure.
CANONICAL_SCORES: tuple[float, ...] = (3.0, 7.0, 1.0, 5.0, 0.0, 6.0, 2.0, 4.0)

_TRACE_SHA = canonical_sha256({"trace_id": TRACE_ID})

if STABILITY_METRIC not in STABILITY_METRICS:  # fail closed on scaffold metric drift
    raise ValueError("trace stability metric is not declared by the scaffold vocabulary")
if len(CANONICAL_SCORES) != K_ITEMS:
    raise ValueError("canonical scores must declare exactly K_ITEMS entries")


@dataclass(frozen=True, slots=True)
class SeedMeasurement:
    """One seed's deterministic trace: presented values plus the seed's declared coordinate key."""

    seed: int
    values: tuple[float, ...]
    key: tuple[int, ...]


def _digest_int(parts: Sequence[object]) -> int:
    """A deterministic nonnegative integer derived from the parts, with no unseeded randomness."""

    return int(canonical_sha256(list(parts))[:16], 16)


def _unit(parts: Sequence[object]) -> float:
    """A deterministic fraction in [0, 1] derived from the parts, with no unseeded randomness."""

    return int(canonical_sha256(list(parts))[:8], 16) / 0xFFFFFFFF


def _permutation(n: int, parts: Sequence[object]) -> tuple[int, ...]:
    """A deterministic permutation of range(n), seeded only by the parts (Fisher and Yates)."""

    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = _digest_int([*parts, "swap", i]) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


def favorable_measurement(seed: int) -> SeedMeasurement:
    """A seeded trace WITH real cross-seed structure: canonical scores in a per-seed coordinate map."""

    key = _permutation(K_ITEMS, ["favorable-key", seed])
    values = [0.0] * K_ITEMS
    for i in range(K_ITEMS):
        values[key[i]] = CANONICAL_SCORES[i]
    return SeedMeasurement(seed=seed, values=tuple(values), key=key)


def null_measurement(seed: int) -> SeedMeasurement:
    """A seeded trace with NO cross-seed structure: independent per-seed values keep agreement at chance."""

    key = _permutation(K_ITEMS, ["null-key", seed])
    values = tuple(_unit(["null-value", seed, j]) for j in range(K_ITEMS))
    return SeedMeasurement(seed=seed, values=values, key=key)


def _decode(values: Sequence[float], key: Sequence[int]) -> tuple[float, ...]:
    """Recover canonical-coordinate values by reading each canonical slot through the declared key."""

    return tuple(values[key[i]] for i in range(len(key)))


def _ranking(values: Sequence[float]) -> tuple[int, ...]:
    """Return the rank of each item as a permutation of range(len(values)); ties break by index."""

    order = sorted(range(len(values)), key=lambda t: (values[t], t))
    rank = [0] * len(values)
    for position, item in enumerate(order):
        rank[item] = position
    return tuple(rank)


def recovered_ranking(measurement: SeedMeasurement) -> tuple[int, ...]:
    """The canonical ranking recovered from a measurement by decoding with its own declared key."""

    return _ranking(_decode(measurement.values, measurement.key))


def rank_agreement(rankings: Sequence[tuple[int, ...]]) -> float:
    """Cross-seed agreement of the recovered rankings, via the scaffold's rank-correlation metric.

    A single ranking carries no cross-seed pair, so agreement falls to the chance floor.
    """

    if len(rankings) < 2:
        return CHANCE_FLOOR
    records = tuple(
        TraceRecord(
            trace_id=TRACE_ID,
            seed=index,
            session_id="session-0",
            effect=0.0,
            trace_sha256=_TRACE_SHA,
            ranking=ranking,
        )
        for index, ranking in enumerate(rankings)
    )
    return round(cross_seed_agreement(records, STABILITY_METRIC), 9)


def derived_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Deterministically derive count distinct nonnegative seeds from a base seed."""

    return tuple(abs(base_seed) * count + offset for offset in range(count))


def control_rankings(
    measurements: Sequence[SeedMeasurement], control: str, *, leak: bool
) -> list[tuple[int, ...]]:
    """Recovered rankings for the favorable measurements under one control arm.

    When leak is True the control fails to scramble and the clean cross-seed structure survives; this
    is only for exercising the fail-closed path and never happens on a real control run.
    """

    if leak:
        return [recovered_ranking(measurement) for measurement in measurements]
    if control == "single-seed":
        return [recovered_ranking(measurements[0])]
    if control == "shuffled-seed":
        count = len(measurements)
        return [
            _ranking(_decode(measurements[j].values, measurements[(j + 1) % count].key))
            for j in range(count)
        ]
    if control == "permuted-trace":
        permuted_rankings: list[tuple[int, ...]] = []
        for measurement in measurements:
            perm = _permutation(K_ITEMS, ["permuted-trace", measurement.seed])
            permuted = tuple(measurement.values[perm[t]] for t in range(K_ITEMS))
            permuted_rankings.append(_ranking(_decode(permuted, measurement.key)))
        return permuted_rankings
    if control == "label-shuffled":
        relabeled_rankings: list[tuple[int, ...]] = []
        for measurement in measurements:
            base = recovered_ranking(measurement)
            perm = _permutation(K_ITEMS, ["label-shuffled", measurement.seed])
            relabeled_rankings.append(tuple(base[perm[i]] for i in range(K_ITEMS)))
        return relabeled_rankings
    raise ValueError(f"unsupported control arm {control!r}")
