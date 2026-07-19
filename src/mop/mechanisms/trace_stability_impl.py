
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

K_ITEMS = 8
THRESHOLD = 0.9
CHANCE_FLOOR = 0.5

CANONICAL_SCORES: tuple[float, ...] = (3.0, 7.0, 1.0, 5.0, 0.0, 6.0, 2.0, 4.0)

_TRACE_SHA = canonical_sha256({"trace_id": TRACE_ID})

if STABILITY_METRIC not in STABILITY_METRICS:  # fail closed on scaffold metric drift
    raise ValueError("trace stability metric is not declared by the scaffold vocabulary")
if len(CANONICAL_SCORES) != K_ITEMS:
    raise ValueError("canonical scores must declare exactly K_ITEMS entries")


@dataclass(frozen=True, slots=True)
class SeedMeasurement:

    seed: int
    values: tuple[float, ...]
    key: tuple[int, ...]


def _digest_int(parts: Sequence[object]) -> int:

    return int(canonical_sha256(list(parts))[:16], 16)


def _unit(parts: Sequence[object]) -> float:

    return int(canonical_sha256(list(parts))[:8], 16) / 0xFFFFFFFF


def _permutation(n: int, parts: Sequence[object]) -> tuple[int, ...]:

    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = _digest_int([*parts, "swap", i]) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


def favorable_measurement(seed: int) -> SeedMeasurement:

    key = _permutation(K_ITEMS, ["favorable-key", seed])
    values = [0.0] * K_ITEMS
    for i in range(K_ITEMS):
        values[key[i]] = CANONICAL_SCORES[i]
    return SeedMeasurement(seed=seed, values=tuple(values), key=key)


def null_measurement(seed: int) -> SeedMeasurement:

    key = _permutation(K_ITEMS, ["null-key", seed])
    values = tuple(_unit(["null-value", seed, j]) for j in range(K_ITEMS))
    return SeedMeasurement(seed=seed, values=values, key=key)


def _decode(values: Sequence[float], key: Sequence[int]) -> tuple[float, ...]:

    return tuple(values[key[i]] for i in range(len(key)))


def _ranking(values: Sequence[float]) -> tuple[int, ...]:

    order = sorted(range(len(values)), key=lambda t: (values[t], t))
    rank = [0] * len(values)
    for position, item in enumerate(order):
        rank[item] = position
    return tuple(rank)


def recovered_ranking(measurement: SeedMeasurement) -> tuple[int, ...]:

    return _ranking(_decode(measurement.values, measurement.key))


def rank_agreement(rankings: Sequence[tuple[int, ...]]) -> float:

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

    return tuple(abs(base_seed) * count + offset for offset in range(count))


def control_rankings(
    measurements: Sequence[SeedMeasurement], control: str, *, leak: bool
) -> list[tuple[int, ...]]:

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
