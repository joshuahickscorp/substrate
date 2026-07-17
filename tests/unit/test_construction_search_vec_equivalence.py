"""Wide-sweep proof that the vectorized construction search is bitwise identical to the scalar bed.

The vectorized reimplementation in ``construction_search_vec_impl`` is the frontier speed lever for
the construction lane. It may only ship if it is byte-for-byte identical to the sealed scalar
authority ``construction_search_impl`` for every seed the wave will ever run, because the receipt
digest folded into the sealed proof is a pure deterministic function of the five arm results, so a
single differing bit in any raw_score, evaluation count, or member tuple would silently change the
construction lane's evidence class. This proof is the quality guard.

The sweep covers, for both the null and the favorable regime:

- seeds 0..999 (the dense low-seed floor), and
- the construction lane's real seed bands from ``generation1_successor_mechanics_queue`` LANES for
  G1-G1: the full 256-seed canary gate band, plus a stratified cross section of the producer and
  challenge bands built with the exact per-rung offset math ``base + rung*seeds_per_rung + k`` that
  ``build_work_items`` uses, including the first and last executed seed of each band.

Every seed asserts full-structure bitwise identity across all five arms on both regimes. The test
counts mismatches and fails if any is nonzero, reporting the exact diverging seed and value. A
separate case reproduces the sealed receipt digest end to end through the real runner.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import struct

from mop.mechanisms import construction_search_impl as scalar
from mop.mechanisms.construction_search_bed import (
    CHEAP_CONTROLS,
    ORACLE_REFERENCE,
    SEARCH_ARM,
    ConstructionSearchBed,
)
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner, RunResults
from mop.mechanisms.construction_search_vec_impl import (
    _lcg_states_flat,
    vec_evaluate_regime,
)
from mop.studies.generation1_successor_mechanics_queue import CANARY_SEEDS, LANES

_MANTISSA_MASK = (1 << 53) - 1
_TWO_POW_53 = float(1 << 53)
_RUNG_SAMPLE_STRIDE = 15


def _float_bits(value: float) -> bytes:
    """The raw 8 IEEE-754 bytes of a float, so any bit difference (including sign of zero) is caught."""

    return struct.pack("<d", float(value))


def _g1g1_lane():
    """The G1-G1 construction_search lane spec, the authority for the real executed seed bands."""

    return next(lane for lane in LANES if lane.lane_id == "G1-G1")


def _band_seeds() -> list[int]:
    """The real construction lane seeds: full canary band plus stratified producer and challenge."""

    lane = _g1g1_lane()
    seeds: list[int] = list(range(lane.canary_start, lane.canary_start + CANARY_SEEDS))
    for base in (lane.producer_start, lane.challenge_start):
        for rung in range(0, lane.rungs_per_phase, _RUNG_SAMPLE_STRIDE):
            rung_start = base + rung * lane.seeds_per_rung  # exact build_work_items offset math
            for offset in (0, 1, lane.seeds_per_rung // 2, lane.seeds_per_rung - 1):
                seeds.append(rung_start + offset)
        seeds.append(base)  # first executed seed of the band
        seeds.append(base + lane.rungs_per_phase * lane.seeds_per_rung - 1)  # last executed seed
    return seeds


def _sweep_seeds() -> list[int]:
    """Seeds 0..999 (required floor) plus the construction lane's real G1-G1 seed bands."""

    return list(range(1000)) + _band_seeds()


def _arm_mismatches(scalar_arms, vec_arms, seed: int, regime: str) -> list[tuple]:
    """Every field-level bitwise difference between the scalar and vectorized arm dicts for one seed."""

    diffs: list[tuple] = []
    if set(vec_arms) != set(scalar_arms):
        diffs.append((seed, regime, "arm-keys", sorted(scalar_arms), sorted(vec_arms)))
        return diffs
    for arm in scalar_arms:
        sref = scalar_arms[arm]
        vref = vec_arms[arm]
        if vref.arm != sref.arm:
            diffs.append((seed, regime, arm, "name", sref.arm, vref.arm))
        if _float_bits(vref.raw_score) != _float_bits(sref.raw_score):
            diffs.append(
                (seed, regime, arm, "raw_score", sref.raw_score.hex(), float(vref.raw_score).hex())
            )
        if int(vref.evaluations) != int(sref.evaluations):
            diffs.append((seed, regime, arm, "evaluations", sref.evaluations, vref.evaluations))
        if tuple(vref.members) != tuple(sref.members):
            diffs.append((seed, regime, arm, "members", sref.members, vref.members))
    return diffs


def test_lcg_stream_is_bitwise_identical_to_scalar_deterministic_stream():
    """The vectorized uint64 LCG reproduces the scalar next_float sequence bit for bit."""

    lane = _g1g1_lane()
    stream_seeds = [
        0,
        1,
        2,
        255,
        999,
        lane.canary_start,
        lane.producer_start,
        lane.challenge_start,
        (17 ^ 0xA24BAED4963EE407) & ((1 << 64) - 1),  # search-salted, exceeds 2^63
        (17 ^ 0x2545F4914F6CDD1D) & ((1 << 64) - 1),  # random-salted, exceeds 2^63
        (lane.producer_start ^ 0xA24BAED4963EE407) & ((1 << 64) - 1),
    ]
    length = 517
    mismatches: list[tuple] = []
    for stream_seed in stream_seeds:
        reference = scalar.DeterministicStream(stream_seed)
        ref_floats = [reference.next_float() for _ in range(length)]
        states = _lcg_states_flat(stream_seed, length)
        mantissa = (states >> 11) & _MANTISSA_MASK
        vec_floats = [int(m) / _TWO_POW_53 for m in mantissa]
        for index, (ref, got) in enumerate(zip(ref_floats, vec_floats, strict=True)):
            if _float_bits(ref) != _float_bits(got):
                mismatches.append((stream_seed, index, ref, got))
    assert not mismatches, f"stream bit mismatch; first diverging: {mismatches[0]}"


def test_vectorized_construction_is_bitwise_identical_to_scalar_over_wide_sweep():
    """Full five-arm evaluate_regime is bitwise identical, scalar vs vectorized, on every sweep seed."""

    bed = ConstructionSearchBed()
    seeds = _sweep_seeds()
    mismatches: list[tuple] = []
    for seed in seeds:
        for regime, spec in (("favorable", bed.favorable_regime(seed)), ("null", bed.null_regime(seed))):
            scalar_arms = scalar.evaluate_regime(spec, seed)
            vec_arms = vec_evaluate_regime(spec, seed)
            mismatches.extend(_arm_mismatches(scalar_arms, vec_arms, seed, regime))
    assert not mismatches, (
        f"{len(mismatches)} bitwise mismatches over {len(seeds)} seeds x 2 regimes; "
        f"first diverging: {mismatches[0]}"
    )


def _run_results_from_arms(bed, runner, seed, favorable_arms, null_arms) -> str:
    """Reproduce the sealed receipt digest from a supplied arm dict, mirroring the runner formulas.

    This is fed both the scalar arms (validating the mirror against the real runner) and the
    vectorized arms (proving the vectorized path reproduces the exact sealed receipt digest).
    """

    favorable_spec = bed.favorable_regime(seed)
    null_spec = bed.null_regime(seed)
    per_eval_cost = favorable_spec.per_eval_cost
    favorable_nets = {a: r.raw_score - per_eval_cost * r.evaluations for a, r in favorable_arms.items()}
    null_nets = {a: r.raw_score - per_eval_cost * r.evaluations for a, r in null_arms.items()}
    search_favorable = favorable_nets[SEARCH_ARM]
    search_null = null_nets[SEARCH_ARM]
    favorable_margins = {c: search_favorable - favorable_nets[c] for c in CHEAP_CONTROLS}
    null_margins = {c: search_null - null_nets[c] for c in CHEAP_CONTROLS}
    favorable_beats_all = all(favorable_margins[c] > 0.0 for c in CHEAP_CONTROLS)
    null_holds = not all(null_margins[c] > 0.0 for c in CHEAP_CONTROLS)
    controls_cleared = tuple(c for c in CHEAP_CONTROLS if favorable_margins[c] > 0.0)
    headroom_gap = favorable_arms[ORACLE_REFERENCE].raw_score - favorable_arms[SEARCH_ARM].raw_score
    results = RunResults(
        seed=seed,
        per_eval_cost=per_eval_cost,
        favorable_objective_digest=favorable_spec.digest(),
        null_objective_digest=null_spec.digest(),
        favorable_nets=favorable_nets,
        null_nets=null_nets,
        favorable_margins=favorable_margins,
        null_margins=null_margins,
        favorable_headroom_gap=max(0.0, headroom_gap),
        favorable_beats_all=favorable_beats_all,
        null_holds=null_holds,
        controls_cleared=controls_cleared,
    )
    return runner.mint(results).digest()


def test_vectorized_arms_reproduce_the_sealed_receipt_digest_end_to_end():
    """The vectorized arms reproduce the real runner receipt digest, the folded sealed artifact."""

    bed = ConstructionSearchBed()
    runner = ConstructionSearchRunner()
    lane = _g1g1_lane()
    seeds = list(range(100)) + [lane.canary_start, lane.producer_start, lane.challenge_start]
    helper_validated = 0
    digest_mismatches: list[tuple] = []
    for seed in seeds:
        real_digest = runner.mint(runner.run(bed, seed)).digest()
        scalar_arms_fav = scalar.evaluate_regime(bed.favorable_regime(seed), seed)
        scalar_arms_null = scalar.evaluate_regime(bed.null_regime(seed), seed)
        if _run_results_from_arms(bed, runner, seed, scalar_arms_fav, scalar_arms_null) == real_digest:
            helper_validated += 1
        vec_arms_fav = vec_evaluate_regime(bed.favorable_regime(seed), seed)
        vec_arms_null = vec_evaluate_regime(bed.null_regime(seed), seed)
        vec_digest = _run_results_from_arms(bed, runner, seed, vec_arms_fav, vec_arms_null)
        if vec_digest != real_digest:
            digest_mismatches.append((seed, real_digest, vec_digest))
    assert helper_validated == len(seeds), "digest mirror did not match the real runner on every seed"
    assert not digest_mismatches, f"receipt digest mismatch; first: {digest_mismatches[0]}"
