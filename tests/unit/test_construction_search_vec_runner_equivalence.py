
from __future__ import annotations

import struct
import time

from mop.mechanisms.construction_search_bed import ConstructionSearchBed
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner
from mop.mechanisms.construction_search_vec_runner import ConstructionSearchVecRunner
from mop.studies.generation1_consolidated_final_campaign import (
    MECHANICS_CYCLE_STRIDE,
    MECHANICS_FRESH_BASE,
)
from mop.studies.generation1_full_generations_wave import EPOCH_CYCLES
from mop.studies.generation1_successor_mechanics_queue import CANARY_SEEDS, LANES

_RUNG_SAMPLE = (0, 240, 479)
_OFFSET_SAMPLE = (0, 1, 1024, 2047)
_SPEEDUP_MIN_SEEDS = 200


def _g1g1_lane():

    return next(lane for lane in LANES if lane.lane_id == "G1-G1")


def _cycle_shifted_seed(base_start: int, rung: int, offset: int, cycle: int) -> int:

    lane = _g1g1_lane()
    source_seed = base_start + rung * lane.seeds_per_rung + offset
    return source_seed + MECHANICS_FRESH_BASE + cycle * MECHANICS_CYCLE_STRIDE


def _cycle_band_seeds() -> list[int]:

    lane = _g1g1_lane()
    seeds: list[int] = []
    for cycle in EPOCH_CYCLES:
        for base in (lane.producer_start, lane.challenge_start):
            for rung in _RUNG_SAMPLE:
                for offset in _OFFSET_SAMPLE:
                    seeds.append(_cycle_shifted_seed(base, rung, offset, cycle))
    return seeds


def _default_bed_seeds() -> list[int]:

    lane = _g1g1_lane()
    seeds: set[int] = set(range(1000))
    seeds.update(range(lane.canary_start, lane.canary_start + CANARY_SEEDS))
    seeds.update(_cycle_band_seeds())
    return sorted(seeds)


def _float_bits(value: float) -> bytes:

    return struct.pack("<d", float(value))


def _first_field_diff(scalar_obj, vec_obj, path: str = ""):

    if isinstance(scalar_obj, dict) and isinstance(vec_obj, dict):
        for key in sorted(set(scalar_obj) | set(vec_obj)):
            found = _first_field_diff(scalar_obj.get(key), vec_obj.get(key), f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(scalar_obj, list) and isinstance(vec_obj, list):
        if len(scalar_obj) != len(vec_obj):
            return (path, scalar_obj, vec_obj)
        for index, (left, right) in enumerate(zip(scalar_obj, vec_obj, strict=True)):
            found = _first_field_diff(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(scalar_obj, float) or isinstance(vec_obj, float):
        if _float_bits(scalar_obj) != _float_bits(vec_obj):
            return (path, scalar_obj, vec_obj)
        return None
    if scalar_obj != vec_obj:
        return (path, scalar_obj, vec_obj)
    return None


def _compare_receipts(scalar_runner, vec_runner, bed, seed):

    scalar_receipt = scalar_runner.mint(scalar_runner.run(bed, seed))
    vec_receipt = vec_runner.mint(vec_runner.run(bed, seed))
    digests_equal = scalar_receipt.digest() == vec_receipt.digest()
    scalar_payload = scalar_receipt.payload()
    vec_payload = vec_receipt.payload()
    field_diff = _first_field_diff(scalar_payload, vec_payload)
    matched = digests_equal and scalar_payload == vec_payload and field_diff is None
    return matched, scalar_receipt.verdict, len(scalar_receipt.controls_cleared), field_diff


def test_vec_runner_receipts_are_byte_identical_to_scalar_over_wide_sweep():

    scalar_runner = ConstructionSearchRunner()
    vec_runner = ConstructionSearchVecRunner()

    default_bed = ConstructionSearchBed()
    default_seeds = _default_bed_seeds()
    assert len(default_seeds) >= 1500, f"default-bed sweep must exceed 1500 seeds, got {len(default_seeds)}"

    degenerate_beds = (
        ("zero_synergy", ConstructionSearchBed(synergy_bonus=0.0)),
        ("punitive_cost", ConstructionSearchBed(per_eval_cost=0.05)),
    )
    degenerate_seeds = list(range(48))

    mismatches: list[tuple] = []
    seeds_tested = 0
    verdicts_seen: set[str] = set()
    controls_lengths_seen: set[int] = set()

    for seed in default_seeds:
        matched, verdict, controls_len, field_diff = _compare_receipts(
            scalar_runner, vec_runner, default_bed, seed
        )
        seeds_tested += 1
        verdicts_seen.add(verdict)
        controls_lengths_seen.add(controls_len)
        if not matched:
            mismatches.append(("default", seed, field_diff))

    for label, bed in degenerate_beds:
        for seed in degenerate_seeds:
            matched, verdict, controls_len, field_diff = _compare_receipts(
                scalar_runner, vec_runner, bed, seed
            )
            seeds_tested += 1
            verdicts_seen.add(verdict)
            controls_lengths_seen.add(controls_len)
            if not matched:
                mismatches.append((label, seed, field_diff))

    print(
        f"vec-runner receipt sweep: seeds_tested={seeds_tested} "
        f"default_bed_seeds={len(default_seeds)} cycle_band_seeds={len(_cycle_band_seeds())} "
        f"mismatches={len(mismatches)} verdicts={sorted(verdicts_seen)} "
        f"controls_cleared_lengths={sorted(controls_lengths_seen)}"
    )

    assert not mismatches, (
        f"{len(mismatches)} receipt mismatches over {seeds_tested} seeds; "
        f"first diverging (bed, seed, field_path/scalar/vec): {mismatches[0]}"
    )
    assert {"mechanics-ok", "null"} <= verdicts_seen, f"both verdict branches must run: {verdicts_seen}"
    assert {0, 3} <= controls_lengths_seen, (
        f"both empty and fully cleared controls must run: {controls_lengths_seen}"
    )


def test_vec_runner_matches_scalar_on_real_g1g1_cycle_bands_19_to_32():

    scalar_runner = ConstructionSearchRunner()
    vec_runner = ConstructionSearchVecRunner()
    bed = ConstructionSearchBed()

    assert tuple(EPOCH_CYCLES) == tuple(range(19, 33)), (
        f"full-generations cycle offsets must be 19..32, got {EPOCH_CYCLES}"
    )
    cycle_seeds = _cycle_band_seeds()
    assert cycle_seeds, "cycle band seed set must be non-empty"

    mismatches: list[tuple] = []
    for seed in cycle_seeds:
        matched, _verdict, _controls_len, field_diff = _compare_receipts(
            scalar_runner, vec_runner, bed, seed
        )
        if not matched:
            mismatches.append((seed, field_diff))
    assert not mismatches, (
        f"{len(mismatches)} cycle-band receipt mismatches over {len(cycle_seeds)} seeds; "
        f"first diverging (seed, field_path/scalar/vec): {mismatches[0]}"
    )


def test_vec_runner_is_faster_than_scalar_runner_over_the_canary_band():

    scalar_runner = ConstructionSearchRunner()
    vec_runner = ConstructionSearchVecRunner()
    bed = ConstructionSearchBed()
    lane = _g1g1_lane()
    seeds = list(range(lane.canary_start, lane.canary_start + CANARY_SEEDS))
    assert len(seeds) >= _SPEEDUP_MIN_SEEDS, f"speedup band must cover >= {_SPEEDUP_MIN_SEEDS} seeds"

    scalar_runner.run(bed, seeds[0])
    vec_runner.run(bed, seeds[0])

    start = time.perf_counter()
    for seed in seeds:
        scalar_runner.run(bed, seed)
    scalar_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for seed in seeds:
        vec_runner.run(bed, seed)
    vec_seconds = time.perf_counter() - start

    speedup = scalar_seconds / vec_seconds if vec_seconds > 0 else float("inf")
    print(
        f"vec-runner speedup: seeds={len(seeds)} scalar_s={scalar_seconds:.3f} "
        f"vec_s={vec_seconds:.3f} speedup={speedup:.2f}x"
    )
    assert speedup > 1.0, f"vectorized runner must beat the scalar runner, got {speedup:.2f}x"
