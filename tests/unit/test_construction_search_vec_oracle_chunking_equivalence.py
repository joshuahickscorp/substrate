"""Byte-identity and peak-RAM proof for the cache-chunked vectorized oracle kernel.

``construction_search_vec_impl.vec_run_oracle`` scores the exhaustive ``2**num_members`` subset
enumeration in cache-fitting chunks over the subset axis instead of one monolithic
``(2**num_members, num_members, num_tasks)`` float64 array (about 1 GB at the 20-member oracle
ceiling). That chunking is a frontier speed and safety lever: it bounds the peak transient RSS to a
few megabytes so a numpy allocation spike can never cross the ``-P kill`` line mid-rung, and it is
faster on any host whose cache the monolithic temporary overflows. It may only ship if it is
byte-for-byte identical to the path it replaces, because the oracle raw_score folds into the
construction lane's ``favorable_headroom_gap`` and therefore into the sealed receipt digest, so a
single differing bit would change the evidence class.

This module proves that identity against BOTH references the lever must match:

- the pre-change monolithic vectorized path, reproduced verbatim below as
  ``_reference_monolithic_oracle`` (a single full-batch ``_score_subsets`` over every subset plus one
  ``np.argmax``), which is exactly the body ``vec_run_oracle`` had before the chunking edit, and
- the sealed scalar authority ``construction_search_impl.run_oracle``.

The wide sweep covers seeds 0..999, the full 256-seed G1-G1 canary band, and the real G1-G1 producer
and challenge bands carried through the full-generations fresh cycles 19..32, on both the favorable and
null regime, which is well over one thousand distinct seeds. A ceiling case exercises the chunk-boundary
tie-break directly at member counts that split into many chunks, a receipt case proves the full minted
``RunReceipt.digest()`` through the vectorized runner is identical for the chunked, the monolithic, and
the scalar oracle, an isolated-subprocess case measures the peak RSS before and after the change at the
20-member ceiling, and a wall-time case measures the speedup over more than two hundred seeds. Every
identity case counts mismatches and fails if any is nonzero.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import textwrap

import numpy as np

from mop.config import REPO_ROOT
from mop.mechanisms import construction_search_impl as scalar
from mop.mechanisms import construction_search_vec_impl as vec
from mop.mechanisms.construction_search_bed import ConstructionSearchBed
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner
from mop.mechanisms.construction_search_vec_impl import VecArmResult, vec_run_oracle
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
_SPEEDUP_MEMBERS = 18  # a member count whose monolithic temp (~113 MB) overflows cache on this lever
_CEILING_MEMBERS = 20  # the oracle headroom ceiling: the ~1 GB monolithic temp the chunking removes


def _reference_monolithic_oracle(spec) -> VecArmResult:
    """Verbatim pre-change monolithic ``vec_run_oracle``: one full-batch score plus one argmax.

    This is the exact body the vectorized oracle had before the cache-chunking edit. It builds the
    whole ``2**num_members`` subset table at once, scores it in a single ``_score_subsets`` call, and
    takes a single ``np.argmax``. It reuses the module's own unchanged ``_affinity_array`` and
    ``_score_subsets``, so it reproduces the monolithic path bit for bit and is the reference the
    chunked path must equal.
    """

    num_members = spec.num_members
    if num_members > vec._MAX_ORACLE_MEMBERS:
        raise vec.ConstructionSearchVecRefusal("oracle headroom is only defined for small member counts")
    masks = np.arange(1 << num_members, dtype=np.uint64)
    bit_index = np.arange(num_members, dtype=np.uint64)
    incl = ((masks[:, None] >> bit_index[None, :]) & np.uint64(1)).astype(bool)
    affinity = vec._affinity_array(spec)
    scores = vec._score_subsets(
        affinity, incl, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks
    )
    best = int(np.argmax(scores))
    best_members = tuple(int(m) for m in range(num_members) if bool(incl[best, m]))
    return VecArmResult(
        arm=vec._ORACLE_ARM,
        raw_score=float(scores[best]),
        evaluations=1 << num_members,
        members=best_members,
    )


def _float_bits(value: float) -> bytes:
    """The raw 8 IEEE-754 bytes of a float, so any bit difference (sign of zero included) is caught."""

    return struct.pack("<d", float(value))


def _g1g1_lane():
    """The G1-G1 construction_search lane spec: the authority for the real executed seed bands."""

    return next(lane for lane in LANES if lane.lane_id == "G1-G1")


def _cycle_shifted_seed(base_start: int, rung: int, offset: int, cycle: int) -> int:
    """One real executed G1-G1 band seed carried into a fresh cycle, exact fresh_mechanics_item math."""

    lane = _g1g1_lane()
    source_seed = base_start + rung * lane.seeds_per_rung + offset
    return source_seed + MECHANICS_FRESH_BASE + cycle * MECHANICS_CYCLE_STRIDE


def _cycle_band_seeds() -> list[int]:
    """The real G1-G1 producer and challenge seed bands across the full-generations cycles 19..32."""

    lane = _g1g1_lane()
    seeds: list[int] = []
    for cycle in EPOCH_CYCLES:
        for base in (lane.producer_start, lane.challenge_start):
            for rung in _RUNG_SAMPLE:
                for offset in _OFFSET_SAMPLE:
                    seeds.append(_cycle_shifted_seed(base, rung, offset, cycle))
    return seeds


def _wide_sweep_seeds() -> list[int]:
    """Dense floor 0..999, the full real canary band, and the cycle-shifted G1-G1 bands, deduped."""

    lane = _g1g1_lane()
    seeds: list[int] = list(range(1000))
    seeds.extend(range(lane.canary_start, lane.canary_start + CANARY_SEEDS))
    seeds.extend(_cycle_band_seeds())
    seen: set[int] = set()
    unique: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique.append(seed)
    return unique


def _oracle_mismatch(left: VecArmResult, right, seed: int, regime: str, tag: str):
    """Return a mismatch tuple if two oracle arm results differ in any receipt-bearing field, else None."""

    if _float_bits(left.raw_score) != _float_bits(right.raw_score):
        return (seed, regime, tag, "raw_score", right.raw_score.hex(), float(left.raw_score).hex())
    if tuple(left.members) != tuple(right.members):
        return (seed, regime, tag, "members", tuple(right.members), tuple(left.members))
    if int(left.evaluations) != int(right.evaluations):
        return (seed, regime, tag, "evaluations", right.evaluations, left.evaluations)
    return None


def test_chunked_oracle_is_bit_identical_to_monolithic_and_scalar_over_wide_sweep() -> None:
    """The chunked oracle equals BOTH the monolithic vec path and the scalar authority, bit for bit."""

    bed = ConstructionSearchBed()
    seeds = _wide_sweep_seeds()
    assert len(seeds) >= 1000, f"wide sweep must exceed 1000 seeds, got {len(seeds)}"
    cycle_seeds = set(_cycle_band_seeds())
    assert cycle_seeds & set(seeds), "the sweep must include the real G1-G1 cycle bands 19..32"
    assert tuple(EPOCH_CYCLES) == tuple(range(19, 33)), f"cycles must be 19..32, got {EPOCH_CYCLES}"

    mismatches: list[tuple] = []
    for seed in seeds:
        for regime, spec in (
            ("favorable", bed.favorable_regime(seed)),
            ("null", bed.null_regime(seed)),
        ):
            chunked = vec_run_oracle(spec)
            monolithic = _reference_monolithic_oracle(spec)
            scalar_result = scalar.run_oracle(spec)
            for reference, tag in ((monolithic, "monolithic"), (scalar_result, "scalar")):
                diff = _oracle_mismatch(chunked, reference, seed, regime, tag)
                if diff is not None:
                    mismatches.append(diff)
    assert not mismatches, (
        f"{len(mismatches)} oracle mismatches over {len(seeds)} seeds x 2 regimes; "
        f"first diverging: {mismatches[0]}"
    )


def test_chunked_oracle_matches_references_at_multi_chunk_ceiling_member_counts() -> None:
    """Where the enumeration splits into many chunks, the boundary tie-break stays byte-identical.

    The default bed enumerates 4096 subsets as a single chunk, so the chunk-boundary reduction is not
    exercised there. This drives member counts whose enumeration splits into dozens or hundreds of
    chunks and asserts the chunked oracle still equals the monolithic vec path exactly, plus the scalar
    authority at the counts small enough to brute force cheaply.
    """

    mismatches: list[tuple] = []
    for num_members, seed_count, check_scalar in ((14, 12, True), (16, 12, True), (18, 8, False), (20, 4, False)):
        bed = ConstructionSearchBed(num_members=num_members)
        chunk_rows = vec._oracle_chunk_rows(num_members, bed.num_tasks, 1 << num_members)
        assert chunk_rows < (1 << num_members), f"nm={num_members} must split into more than one chunk"
        for seed in range(seed_count):
            for regime, spec in (
                ("favorable", bed.favorable_regime(seed)),
                ("null", bed.null_regime(seed)),
            ):
                chunked = vec_run_oracle(spec)
                monolithic = _reference_monolithic_oracle(spec)
                diff = _oracle_mismatch(chunked, monolithic, seed, f"nm{num_members}:{regime}", "monolithic")
                if diff is not None:
                    mismatches.append(diff)
                if check_scalar:
                    scalar_result = scalar.run_oracle(spec)
                    diff = _oracle_mismatch(
                        chunked, scalar_result, seed, f"nm{num_members}:{regime}", "scalar"
                    )
                    if diff is not None:
                        mismatches.append(diff)
    assert not mismatches, f"{len(mismatches)} ceiling-count oracle mismatches; first: {mismatches[0]}"


def test_chunked_oracle_receipt_digest_matches_monolithic_and_scalar_through_the_vec_runner(
    monkeypatch,
) -> None:
    """The full minted RunReceipt digest and payload are identical for chunked, monolithic, and scalar.

    The vectorized runner mints receipts from ``vec_evaluate_regime``, which resolves ``vec_run_oracle``
    from the impl module at call time. Swapping in the monolithic reference therefore yields the
    monolithic-path receipt, and the unpatched runner yields the chunked-path receipt; both must equal
    the sealed scalar runner's receipt to the byte.
    """

    bed = ConstructionSearchBed()
    lane = _g1g1_lane()
    seeds = list(range(60)) + [lane.canary_start, lane.producer_start, lane.challenge_start]
    seeds += _cycle_band_seeds()[:24]

    scalar_runner = ConstructionSearchRunner()
    vec_runner = ConstructionSearchVecRunner()

    # Chunked-path receipts first (module unpatched), captured as (digest, payload) per seed.
    chunked = {seed: vec_runner.mint(vec_runner.run(bed, seed)) for seed in seeds}
    scalar_receipts = {seed: scalar_runner.mint(scalar_runner.run(bed, seed)) for seed in seeds}

    # Now route the vectorized oracle arm through the monolithic reference and re-mint.
    monkeypatch.setattr(vec, "vec_run_oracle", _reference_monolithic_oracle)
    monolithic = {seed: vec_runner.mint(vec_runner.run(bed, seed)) for seed in seeds}

    mismatches: list[tuple] = []
    for seed in seeds:
        c, m, s = chunked[seed], monolithic[seed], scalar_receipts[seed]
        if not (c.digest() == m.digest() == s.digest()):
            mismatches.append((seed, "digest", s.digest(), c.digest(), m.digest()))
        if not (c.payload() == m.payload() == s.payload()):
            mismatches.append((seed, "payload"))
    assert not mismatches, f"{len(mismatches)} receipt mismatches; first: {mismatches[0]}"


_RSS_PROGRAM = textwrap.dedent(
    """
    import sys, resource
    import numpy as np
    from mop.mechanisms.construction_search_bed import ConstructionSearchBed
    from mop.mechanisms import construction_search_vec_impl as vec

    mode, num_members, ncalls = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    def monolithic(spec):
        n = spec.num_members
        masks = np.arange(1 << n, dtype=np.uint64)
        bit = np.arange(n, dtype=np.uint64)
        incl = ((masks[:, None] >> bit[None, :]) & np.uint64(1)).astype(bool)
        aff = vec._affinity_array(spec)
        sc = vec._score_subsets(aff, incl, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks)
        return float(sc[int(np.argmax(sc))])

    def rss_mb():
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1e6 if sys.platform == "darwin" else raw / 1024.0

    bed = ConstructionSearchBed(num_members=num_members)
    specs = [bed.favorable_regime(seed) for seed in range(ncalls)]
    base_mb = rss_mb()
    for spec in specs:
        if mode == "chunked":
            vec.vec_run_oracle(spec)
        else:
            monolithic(spec)
    peak_mb = rss_mb()
    print(f"{peak_mb:.1f} {base_mb:.1f}")
    """
)


def _subprocess_peak_rss(mode: str, num_members: int, ncalls: int) -> tuple[float, float]:
    """Run one oracle path in an isolated interpreter and return its (peak_mb, baseline_mb) RSS."""

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", _RSS_PROGRAM, mode, str(num_members), str(ncalls)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    peak_mb, base_mb = (float(token) for token in completed.stdout.split())
    return peak_mb, base_mb


def test_chunked_oracle_peak_rss_is_bounded_far_below_monolithic() -> None:
    """At the 20-member ceiling the chunked oracle's peak RSS is hundreds of MB below the monolithic."""

    monolithic_peak, _monolithic_base = _subprocess_peak_rss("monolithic", _CEILING_MEMBERS, 6)
    chunked_peak, _chunked_base = _subprocess_peak_rss("chunked", _CEILING_MEMBERS, 6)

    print(
        f"oracle peak RSS at nm={_CEILING_MEMBERS}: "
        f"monolithic(before)={monolithic_peak:.1f} MB chunked(after)={chunked_peak:.1f} MB "
        f"reduction={monolithic_peak - chunked_peak:.1f} MB"
    )
    # The monolithic 2**20 x 20 x 3 float64 temporary is about 0.5 GB (near 1 GB peak with the np.where
    # intermediate); the chunked path holds only a ~2 MB chunk. The gap is hundreds of MB with wide
    # margin. A modest floor keeps this robust to interpreter and numpy baseline drift across hosts.
    assert monolithic_peak - chunked_peak > 300.0, (
        f"chunking must cut peak RSS by >300 MB at the ceiling: "
        f"monolithic={monolithic_peak:.1f} MB chunked={chunked_peak:.1f} MB"
    )
    assert chunked_peak < monolithic_peak * 0.5, (
        f"chunked peak RSS must be well under half the monolithic peak: "
        f"monolithic={monolithic_peak:.1f} MB chunked={chunked_peak:.1f} MB"
    )


def test_chunked_oracle_wall_speedup_over_wide_seed_run() -> None:
    """The chunked oracle is not slower than the monolithic path over more than two hundred seeds.

    On a host whose cache the ~113 MB monolithic temporary at ``_SPEEDUP_MEMBERS`` overflows, chunking
    is modestly faster; on a very large cache it is roughly neutral. The hard ship gate is byte-identity
    (asserted above), so this case only guards against a real regression and prints the measured ratio.
    """

    import time

    bed = ConstructionSearchBed(num_members=_SPEEDUP_MEMBERS)
    specs = [bed.favorable_regime(seed) for seed in range(_SPEEDUP_MIN_SEEDS)]
    assert len(specs) >= _SPEEDUP_MIN_SEEDS

    # Warm both paths so first-call numpy costs are not charged to the measurement.
    _reference_monolithic_oracle(specs[0])
    vec_run_oracle(specs[0])

    start = time.perf_counter()
    for spec in specs:
        _reference_monolithic_oracle(spec)
    monolithic_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for spec in specs:
        vec_run_oracle(spec)
    chunked_seconds = time.perf_counter() - start

    speedup = monolithic_seconds / chunked_seconds if chunked_seconds > 0 else float("inf")
    print(
        f"chunked oracle wall speedup: seeds={len(specs)} nm={_SPEEDUP_MEMBERS} "
        f"monolithic_s={monolithic_seconds:.3f} chunked_s={chunked_seconds:.3f} speedup={speedup:.3f}x"
    )
    assert speedup > 0.7, f"chunked oracle must not materially regress wall time, got {speedup:.3f}x"
