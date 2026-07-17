"""Independent adversarial re-proof that the vectorized construction search is bitwise identical.

This test does not trust the prior equivalence proof in
``tests/unit/test_construction_search_vec_equivalence.py``. It re-derives the identity from scratch
over a deliberately DIFFERENT and WIDER sweep than the prior proof used, so that a single differing
bit anywhere in the vectorized path (``construction_search_vec_impl``) versus the sealed scalar
authority (``construction_search_impl``) is surfaced with its exact diverging seed and value.

The prior proof swept seeds 0..999 plus the G1-G1 lane bands. This proof deliberately avoids that
seed set and instead uses three independent, fully deterministic seed sources:

- the disjoint dense range 1000..3000 (no overlap with the prior floor),
- 64 large seeds drawn from a fixed sha256 walk seeded by a constant literal (no rng, no OS entropy),
  which land uniformly across the full 64-bit seed space the dense floor never reaches, and
- explicit boundary seeds: 0, 1, 2, powers-of-two edges up to 2**64-1, and the two salts themselves
  (which force the salted LCG stream seed to exactly zero, a stream boundary the prior proof skipped).

For every seed the full five-arm ``evaluate_regime`` structure is compared field for field on BOTH
regimes: arm name, raw_score compared as raw IEEE-754 bytes (so sign-of-zero and any mantissa bit is
caught), evaluation count as an exact integer, and the member tuple. A separate case stresses the
vectorized LCG block decomposition directly at many adversarial stream lengths, and a third case
reproduces the sealed runner receipt digest end to end from the vectorized arms.

The stance is refute-by-default: identity is asserted only if zero mismatches are found. The salts,
LCG constants, and control policy values used here are reproduced as local literals and then
cross-checked against the scalar authority, so this test carries no hidden coupling to the module
under test.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
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

# Reproduced by value from the scalar authority and cross-checked in test_local_constants_match_authority
# below. Defining them locally (rather than importing the private module constants) keeps this proof
# from silently inheriting a wrong salt or mask from the module it is meant to audit.
_MASK64 = (1 << 64) - 1
_LCG_MULT = 6364136223846793005
_LCG_ADD = 1442695040888963407
_SEARCH_SALT = 0xA24BAED4963EE407
_RANDOM_SALT = 0x2545F4914F6CDD1D
_INCLUSION_PROB = 0.5
_MANTISSA_MASK = (1 << 53) - 1
_TWO_POW_53 = float(1 << 53)

_EXPECTED_ARMS = frozenset({*CHEAP_CONTROLS, SEARCH_ARM, ORACLE_REFERENCE})
_ARM_FIELDS = ("no-search", "greedy-only", "random-construction", SEARCH_ARM, ORACLE_REFERENCE)


def _float_bits(value: float) -> bytes:
    """The raw 8 IEEE-754 bytes of a float, so any bit difference (sign of zero included) is caught."""

    return struct.pack("<d", float(value))


def _sha_walk_seeds(count: int) -> list[int]:
    """``count`` deterministic 64-bit seeds from a fixed sha256 walk. No rng, no OS entropy.

    The walk is seeded from a constant literal, then each step hashes the previous digest and reads
    the leading 8 bytes as a big-endian uint64. Successive seeds spread across the whole 64-bit space,
    so many exceed 2**63, exercising the high-seed region the dense floor 1000..3000 never touches.
    """

    seeds: list[int] = []
    state = hashlib.sha256(b"mop-adversarial-construction-search-vec-equivalence/v1").digest()
    for _ in range(count):
        state = hashlib.sha256(state).digest()
        seeds.append(int.from_bytes(state[:8], "big") & _MASK64)
    return seeds


def _boundary_seeds() -> list[int]:
    """Explicit edge seeds, including the two salts that force a salted stream seed of exactly zero."""

    return [
        0,
        1,
        2,
        3,
        255,
        256,
        (1 << 16),
        (1 << 31) - 1,
        (1 << 31),
        (1 << 31) + 1,
        (1 << 32) - 1,
        (1 << 32),
        (1 << 32) + 1,
        (1 << 53) - 1,
        (1 << 53),
        (1 << 53) + 1,
        (1 << 63) - 1,
        (1 << 63),
        (1 << 63) + 1,
        _MASK64 - 1,
        _MASK64,  # 2**64-1
        _SEARCH_SALT,  # seed ^ _SEARCH_SALT == 0, so the search stream starts from stream seed 0
        _RANDOM_SALT,  # seed ^ _RANDOM_SALT == 0, so the random stream starts from stream seed 0
        _SEARCH_SALT ^ _RANDOM_SALT,
        (_SEARCH_SALT + 1) & _MASK64,
        (_RANDOM_SALT - 1) & _MASK64,
    ]


def _sweep_seeds() -> list[int]:
    """The independent adversarial seed set: dense 1000..3000, sha walk, and boundary edges."""

    seeds = list(range(1000, 3001)) + _sha_walk_seeds(64) + _boundary_seeds()
    # De-duplicate while preserving order, so an accidental overlap does not inflate the count.
    seen: set[int] = set()
    unique: list[int] = []
    for seed in seeds:
        if seed not in seen:
            seen.add(seed)
            unique.append(seed)
    return unique


def _arm_mismatches(scalar_arms, vec_arms, seed: int, regime: str) -> list[tuple]:
    """Every field-level bitwise difference between the scalar and vectorized arm dicts for one seed."""

    diffs: list[tuple] = []
    if frozenset(vec_arms) != _EXPECTED_ARMS:
        diffs.append((seed, regime, "vec-arm-keys", sorted(_EXPECTED_ARMS), sorted(vec_arms)))
        return diffs
    if frozenset(scalar_arms) != _EXPECTED_ARMS:
        diffs.append((seed, regime, "scalar-arm-keys", sorted(_EXPECTED_ARMS), sorted(scalar_arms)))
        return diffs
    for arm in _ARM_FIELDS:
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


def test_local_constants_match_authority():
    """The literals this proof uses must equal the sealed scalar authority; otherwise the proof lies."""

    assert _MASK64 == scalar._MASK64
    assert _LCG_MULT == scalar._LCG_MULT
    assert _LCG_ADD == scalar._LCG_ADD
    assert _SEARCH_SALT == scalar._SEARCH_SALT
    assert _RANDOM_SALT == scalar._RANDOM_SALT
    assert _INCLUSION_PROB == scalar._INCLUSION_PROB


def test_adversarial_lcg_block_math_matches_scalar_stream():
    """The vectorized uint64 LCG reproduces DeterministicStream bit for bit at many stress lengths.

    The vectorized states pass through a sqrt-sized block decomposition whose behavior depends on the
    requested count, so identity is checked at counts spanning perfect squares, primes, block-size
    boundaries, and the two real arm lengths (480 restarts-worth and 30000 random-worth), for stream
    seeds including the salted seeds this bed actually produces and the exact-zero stream boundary.
    """

    short_lengths = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 24, 25, 26, 35, 36, 37,
        48, 49, 50, 63, 64, 65, 80, 99, 100, 101, 120, 168, 169, 170,
        173, 195, 224, 255, 256, 288, 289, 480, 483, 528, 529, 530,
    ]
    long_lengths = [9973, 30000]

    base_seeds = [0, 1, 2, 3, _MASK64, (1 << 63), (1 << 63) - 1, (1 << 32) - 1]
    salted = []
    for raw in [0, 1, 17, 1234, (1 << 63), _MASK64, _SEARCH_SALT, _RANDOM_SALT]:
        salted.append((raw ^ _SEARCH_SALT) & _MASK64)
        salted.append((raw ^ _RANDOM_SALT) & _MASK64)
    short_seeds = base_seeds + salted + _sha_walk_seeds(8)
    long_seeds = [0, (1 << 63), (_SEARCH_SALT), (_MASK64 ^ _RANDOM_SALT) & _MASK64]

    mismatches: list[tuple] = []

    def _ref_floats(stream_seed: int, length: int) -> list[float]:
        stream = scalar.DeterministicStream(stream_seed)
        return [stream.next_float() for _ in range(length)]

    def _vec_floats(stream_seed: int, length: int) -> list[float]:
        states = _lcg_states_flat(stream_seed, length)
        mantissa = (states >> 11) & _MANTISSA_MASK
        return [int(m) / _TWO_POW_53 for m in mantissa]

    for stream_seed in short_seeds:
        reference = _ref_floats(stream_seed, max(short_lengths))
        for length in short_lengths:
            got = _vec_floats(stream_seed, length)
            if len(got) != length:
                mismatches.append((stream_seed, length, "length", length, len(got)))
                continue
            for index in range(length):
                if _float_bits(reference[index]) != _float_bits(got[index]):
                    mismatches.append((stream_seed, length, index, reference[index], got[index]))
                    break

    for stream_seed in long_seeds:
        reference = _ref_floats(stream_seed, max(long_lengths))
        for length in long_lengths:
            got = _vec_floats(stream_seed, length)
            if len(got) != length:
                mismatches.append((stream_seed, length, "length", length, len(got)))
                continue
            for index in range(length):
                if _float_bits(reference[index]) != _float_bits(got[index]):
                    mismatches.append((stream_seed, length, index, reference[index], got[index]))
                    break

    assert not mismatches, f"{len(mismatches)} stream bit mismatches; first diverging: {mismatches[0]}"


def test_adversarial_full_structure_bitwise_identical_over_independent_sweep():
    """Full five-arm evaluate_regime is bitwise identical, scalar vs vectorized, on every sweep seed.

    Refute-by-default: the assertion passes only when zero field-level mismatches are found across the
    entire independent sweep on both the favorable and the null regime.
    """

    bed = ConstructionSearchBed()
    seeds = _sweep_seeds()
    mismatches: list[tuple] = []
    for seed in seeds:
        for regime, spec in (
            ("favorable", bed.favorable_regime(seed)),
            ("null", bed.null_regime(seed)),
        ):
            scalar_arms = scalar.evaluate_regime(spec, seed)
            vec_arms = vec_evaluate_regime(spec, seed)
            mismatches.extend(_arm_mismatches(scalar_arms, vec_arms, seed, regime))
    assert not mismatches, (
        f"{len(mismatches)} bitwise mismatches over {len(seeds)} independent seeds x 2 regimes; "
        f"first diverging: {mismatches[0]}"
    )


def _run_results_from_arms(bed, seed, favorable_arms, null_arms) -> RunResults:
    """Reproduce the sealed RunResults from a supplied arm dict, mirroring the runner formulas.

    Fed the scalar arms it validates the mirror against the real runner; fed the vectorized arms it
    proves the vectorized path reproduces the exact sealed receipt digest bit for bit.
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
    return RunResults(
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


def test_adversarial_vectorized_arms_reproduce_sealed_receipt_digest():
    """The vectorized arms reproduce the real runner receipt digest, the folded sealed artifact.

    For each seed the real runner (which runs the scalar arms internally) is minted end to end, then
    the same digest is rebuilt from the vectorized arms through the runner formulas. The scalar mirror
    is validated against the real runner on every seed so the reproduction formula is known faithful.
    """

    bed = ConstructionSearchBed()
    runner = ConstructionSearchRunner()
    seeds = list(range(1000, 1100)) + _sha_walk_seeds(32) + _boundary_seeds()
    seen: set[int] = set()
    unique_seeds = [s for s in seeds if not (s in seen or seen.add(s))]

    helper_validated = 0
    evidence_mismatches: list[tuple] = []
    digest_mismatches: list[tuple] = []
    for seed in unique_seeds:
        real_results = runner.run(bed, seed)
        real_digest = runner.mint(real_results).digest()
        real_evidence = real_results.evidence_digest()

        scalar_results = _run_results_from_arms(
            bed,
            seed,
            scalar.evaluate_regime(bed.favorable_regime(seed), seed),
            scalar.evaluate_regime(bed.null_regime(seed), seed),
        )
        if (
            runner.mint(scalar_results).digest() == real_digest
            and scalar_results.evidence_digest() == real_evidence
        ):
            helper_validated += 1

        vec_results = _run_results_from_arms(
            bed,
            seed,
            vec_evaluate_regime(bed.favorable_regime(seed), seed),
            vec_evaluate_regime(bed.null_regime(seed), seed),
        )
        if vec_results.evidence_digest() != real_evidence:
            evidence_mismatches.append((seed, real_evidence, vec_results.evidence_digest()))
        if runner.mint(vec_results).digest() != real_digest:
            digest_mismatches.append((seed, real_digest, runner.mint(vec_results).digest()))

    assert helper_validated == len(unique_seeds), (
        "digest mirror did not reproduce the real runner on every seed; "
        f"validated {helper_validated} of {len(unique_seeds)}"
    )
    assert not evidence_mismatches, f"evidence digest mismatch; first: {evidence_mismatches[0]}"
    assert not digest_mismatches, f"receipt digest mismatch; first: {digest_mismatches[0]}"
