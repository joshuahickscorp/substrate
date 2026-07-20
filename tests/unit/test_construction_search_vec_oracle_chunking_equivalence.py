
from __future__ import annotations

import struct
import subprocess
import sys
import textwrap

import numpy as np

from mop.config import REPO_ROOT
from mop.mechanisms import construction_search_vec_impl as vec
from mop.mechanisms.construction_search_bed import ConstructionSearchBed
from mop.mechanisms.construction_search_runner import ConstructionSearchRunner
from mop.mechanisms.construction_search_vec_impl import VecArmResult, vec_run_oracle
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

    return struct.pack("<d", float(value))


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


def _wide_sweep_seeds() -> list[int]:

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

    if _float_bits(left.raw_score) != _float_bits(right.raw_score):
        return (seed, regime, tag, "raw_score", right.raw_score.hex(), float(left.raw_score).hex())
    if tuple(left.members) != tuple(right.members):
        return (seed, regime, tag, "members", tuple(right.members), tuple(left.members))
    if int(left.evaluations) != int(right.evaluations):
        return (seed, regime, tag, "evaluations", right.evaluations, left.evaluations)
    return None


def test_chunked_oracle_is_bit_identical_to_monolithic_over_wide_sweep() -> None:

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
            diff = _oracle_mismatch(chunked, monolithic, seed, regime, "monolithic")
            if diff is not None:
                mismatches.append(diff)
    assert not mismatches, (
        f"{len(mismatches)} oracle mismatches over {len(seeds)} seeds x 2 regimes; "
        f"first diverging: {mismatches[0]}"
    )


def test_chunked_oracle_matches_references_at_multi_chunk_ceiling_member_counts() -> None:

    mismatches: list[tuple] = []
    for num_members, seed_count in ((14, 12), (16, 12), (18, 8), (20, 4)):
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
    assert not mismatches, f"{len(mismatches)} ceiling-count oracle mismatches; first: {mismatches[0]}"


def test_chunked_oracle_receipt_digest_matches_monolithic_through_the_runner(
    monkeypatch,
) -> None:

    bed = ConstructionSearchBed()
    lane = _g1g1_lane()
    seeds = list(range(60)) + [lane.canary_start, lane.producer_start, lane.challenge_start]
    seeds += _cycle_band_seeds()[:24]

    runner = ConstructionSearchRunner()

    chunked = {seed: runner.mint(runner.run(bed, seed)) for seed in seeds}

    monkeypatch.setattr(vec, "vec_run_oracle", _reference_monolithic_oracle)
    monolithic = {seed: runner.mint(runner.run(bed, seed)) for seed in seeds}

    mismatches: list[tuple] = []
    for seed in seeds:
        c, m = chunked[seed], monolithic[seed]
        if c.digest() != m.digest():
            mismatches.append((seed, "digest", c.digest(), m.digest()))
        if c.payload() != m.payload():
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

    monolithic_peak, _monolithic_base = _subprocess_peak_rss("monolithic", _CEILING_MEMBERS, 6)
    chunked_peak, _chunked_base = _subprocess_peak_rss("chunked", _CEILING_MEMBERS, 6)

    print(
        f"oracle peak RSS at nm={_CEILING_MEMBERS}: "
        f"monolithic(before)={monolithic_peak:.1f} MB chunked(after)={chunked_peak:.1f} MB "
        f"reduction={monolithic_peak - chunked_peak:.1f} MB"
    )
    assert monolithic_peak - chunked_peak > 300.0, (
        f"chunking must cut peak RSS by >300 MB at the ceiling: "
        f"monolithic={monolithic_peak:.1f} MB chunked={chunked_peak:.1f} MB"
    )
    assert chunked_peak < monolithic_peak * 0.5, (
        f"chunked peak RSS must be well under half the monolithic peak: "
        f"monolithic={monolithic_peak:.1f} MB chunked={chunked_peak:.1f} MB"
    )


def test_chunked_oracle_wall_speedup_over_wide_seed_run() -> None:

    import time

    bed = ConstructionSearchBed(num_members=_SPEEDUP_MEMBERS)
    specs = [bed.favorable_regime(seed) for seed in range(_SPEEDUP_MIN_SEEDS)]
    assert len(specs) >= _SPEEDUP_MIN_SEEDS

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
