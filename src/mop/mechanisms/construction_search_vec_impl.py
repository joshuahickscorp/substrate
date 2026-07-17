"""Net-new numpy-vectorized reimplementation of the construction_search stream and arms.

This module is an independent, bitwise-faithful reimplementation of the deterministic
construction search defined by the sealed authority ``construction_search_impl.py``. It exists
to run the same evidence far faster (the vectorized frontier lever), and it is only ever allowed
to ship once a wide seed sweep proves it is byte-identical to the scalar authority. The proof is
the quality guard: a single differing bit would change the construction lane's raw_score, member
tuples, or receipt digest, and therefore the evidence class, so this module must never diverge.

It reproduces, for any seed:

- the exact 64-bit LCG float sequence of the scalar ``DeterministicStream`` (numpy uint64 with
  explicit mod-2^64 wraparound),
- the exact ``run_random`` and ``run_construction_search`` outputs (the two stream-consuming arms),
- the exact ``run_oracle``, ``run_greedy``, and ``run_no_search`` outputs,
- and thus the exact five-arm ``evaluate_regime`` structure the scalar path returns.

The LCG constants, salts, and control policy constants below are reproduced locally by value and
mirror the sealed authority; this module imports nothing from ``construction_search_impl.py`` so
its correctness is not coupled to that module's source hash. Equivalence is asserted, not assumed,
by ``tests/unit/test_construction_search_vec_equivalence.py``.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .construction_search_bed import RegimeSpec

# These constants mirror the sealed authority construction_search_impl.py by value. They are
# reproduced here (not imported) so this module carries no import-time coupling to that source.
_MASK64 = (1 << 64) - 1
_LCG_MULT = 6364136223846793005
_LCG_ADD = 1442695040888963407
_SEARCH_SALT = 0xA24BAED4963EE407
_RANDOM_SALT = 0x2545F4914F6CDD1D
_EPS = 1e-12
_MAX_PASSES = 4
_INCLUSION_PROB = 0.5
_MAX_ORACLE_MEMBERS = 20

# Arm names mirror construction_search_bed.SEARCH_ARM and ORACLE_REFERENCE by value.
_NO_SEARCH_ARM = "no-search"
_GREEDY_ARM = "greedy-only"
_RANDOM_ARM = "random-construction"
_SEARCH_ARM = "construction-search"
_ORACLE_ARM = "oracle-headroom"

_TWO_POW_53 = float(1 << 53)
_MANTISSA_MASK = np.uint64((1 << 53) - 1)
_SHIFT_11 = np.uint64(11)
_AU = np.uint64(_LCG_MULT)
_CU = np.uint64(_LCG_ADD)


class ConstructionSearchVecRefusal(ValueError):
    """Raised when the vectorized path is asked to run outside its declared, deterministic scope."""


@dataclass(frozen=True, slots=True)
class VecArmResult:
    """One arm's outcome, field-for-field comparable to the scalar impl's ArmResult.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    arm: str
    raw_score: float
    evaluations: int
    members: tuple[int, ...]

    def as_tuple(self) -> tuple[str, float, int, tuple[int, ...]]:
        return (self.arm, self.raw_score, self.evaluations, self.members)


def _lcg_states_flat(stream_seed: int, count: int) -> np.ndarray:
    """Return states state_1..state_count of the scalar LCG as uint64, vectorized over blocks.

    The scalar DeterministicStream(seed) sets state_0 = (seed*MULT + ADD) & MASK in its constructor,
    then next_float advances state_k = (state_{k-1}*MULT + ADD) & MASK and derives its float from
    state_k. So the k-th float (k = 1..count) comes from state_k. This computes those states with a
    sqrt-sized block decomposition: the block start states are advanced with exact Python integer
    arithmetic, and each block's interior is advanced with uint64 numpy (mod 2^64 by construction).
    """

    if count < 0:
        raise ConstructionSearchVecRefusal("stream length must be nonnegative")
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    state0 = ((int(stream_seed) & _MASK64) * _LCG_MULT + _LCG_ADD) & _MASK64

    block_len = max(1, math.isqrt(count))
    n_blocks = (count + block_len - 1) // block_len

    # Compose the single LCG step block_len times: x -> a_l*x + c_l (mod 2^64).
    a_l, c_l = 1, 0
    for _ in range(block_len):
        a_l = (a_l * _LCG_MULT) & _MASK64
        c_l = (c_l * _LCG_MULT + _LCG_ADD) & _MASK64

    # Block start states are state_{b*block_len} for b = 0..n_blocks-1, with block 0 starting at
    # state_0 (the constructor state). Built with exact Python ints, then handed to numpy.
    pre_states = [0] * n_blocks
    state = state0
    for b in range(n_blocks):
        pre_states[b] = state
        state = (a_l * state + c_l) & _MASK64
    cur = np.array(pre_states, dtype=np.uint64)

    out = np.empty((n_blocks, block_len), dtype=np.uint64)
    with np.errstate(over="ignore", under="ignore"):
        for j in range(block_len):
            cur = cur * _AU + _CU
            out[:, j] = cur
    flat = out.reshape(-1)[:count]
    return flat


def _lcg_float_matrix(stream_seed: int, rows: int, cols: int) -> np.ndarray:
    """Return a (rows, cols) float64 matrix equal to the first rows*cols scalar next_float values.

    Row-major: entry (r, c) equals the (r*cols + c + 1)-th float of DeterministicStream(stream_seed),
    matching the scalar consumers that draw cols floats per outer step in member order.
    """

    total = rows * cols
    states = _lcg_states_flat(stream_seed, total)
    mantissa = (states >> _SHIFT_11) & _MANTISSA_MASK
    floats = mantissa.astype(np.float64) / _TWO_POW_53
    return floats.reshape(rows, cols)


def _affinity_array(spec: RegimeSpec) -> np.ndarray:
    """The regime affinity as a (num_members, num_tasks) float64 array. Values are exact copies."""

    return np.asarray(spec.affinity, dtype=np.float64)


def _score_subsets(
    affinity: np.ndarray,
    incl: np.ndarray,
    size_penalty: float,
    synergy_pair: tuple[int, ...],
    synergy_bonus: float,
    num_tasks: int,
) -> np.ndarray:
    """Vectorized score_coalition over a batch of subsets, bitwise identical to the scalar formula.

    For every subset row: base is the per-task coverage mean (max affinity among members per task,
    zero for an empty subset), minus the size penalty, plus the synergy bonus when the synergy pair
    is fully present. The task sum is a left fold to match the scalar ``total += max(...)`` order,
    and every arithmetic op is IEEE double, so results match the scalar path bit for bit.
    """

    masked = np.where(incl[:, :, None], affinity[None, :, :], -np.inf)
    per_task_max = masked.max(axis=1)
    total = per_task_max[:, 0].astype(np.float64, copy=True)
    for task in range(1, num_tasks):
        total = total + per_task_max[:, task]
    sizes = incl.sum(axis=1)
    base = np.where(sizes > 0, total / num_tasks, 0.0)
    score = base - size_penalty * sizes
    synergy_term = np.zeros(incl.shape[0], dtype=np.float64)
    if synergy_bonus > 0.0 and synergy_pair:
        has_pair = np.ones(incl.shape[0], dtype=bool)
        for member in synergy_pair:
            has_pair = has_pair & incl[:, member]
        synergy_term = np.where(has_pair, synergy_bonus, 0.0)
    return score + synergy_term


def _score_coalition_scalar(spec: RegimeSpec, members: Any) -> float:
    """Exact pure-Python mirror of the scalar score_coalition, for the non-stream arms.

    Used by the greedy and no-search arms, whose control flow (first-improvement running maxima
    with an epsilon margin) is cheap and reproduced verbatim rather than vectorized.
    """

    member_list = list(members)
    if not member_list:
        base = 0.0
    else:
        total = 0.0
        for task in range(spec.num_tasks):
            total += max(spec.affinity[m][task] for m in member_list)
        base = total / spec.num_tasks
    penalty = spec.size_penalty * len(member_list)
    synergy = 0.0
    if spec.synergy_bonus > 0.0 and spec.synergy_pair and set(spec.synergy_pair) <= set(member_list):
        synergy = spec.synergy_bonus
    return base - penalty + synergy


def vec_run_no_search(spec: RegimeSpec) -> VecArmResult:
    """Reuse the formation default coalition. One evaluation, no search. Mirrors run_no_search."""

    members = tuple(sorted(m for m in spec.formation_default if 0 <= m < spec.num_members))
    score = _score_coalition_scalar(spec, members)
    return VecArmResult(arm=_NO_SEARCH_ARM, raw_score=score, evaluations=1, members=members)


def vec_run_greedy(spec: RegimeSpec) -> VecArmResult:
    """One greedy add pass with no restarts, reproduced verbatim from run_greedy."""

    members: set[int] = set()
    current = _score_coalition_scalar(spec, members)
    evaluations = 0
    for _ in range(spec.num_members):
        best_member = -1
        best_score = current
        for candidate in range(spec.num_members):
            if candidate in members:
                continue
            evaluations += 1
            trial = _score_coalition_scalar(spec, members | {candidate})
            if trial > best_score + _EPS:
                best_score = trial
                best_member = candidate
        if best_member < 0:
            break
        members.add(best_member)
        current = best_score
    return VecArmResult(
        arm=_GREEDY_ARM, raw_score=current, evaluations=evaluations, members=tuple(sorted(members))
    )


def vec_run_random(spec: RegimeSpec, seed: int) -> VecArmResult:
    """Vectorized equivalent of run_random: sample subsets under the budget, keep the best score."""

    num_members = spec.num_members
    stream_seed = (seed ^ _RANDOM_SALT) & _MASK64
    floats = _lcg_float_matrix(stream_seed, spec.random_samples, num_members)
    incl = floats < _INCLUSION_PROB
    affinity = _affinity_array(spec)
    scores = _score_subsets(
        affinity, incl, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks
    )
    best = int(np.argmax(scores))
    best_members = tuple(int(m) for m in range(num_members) if bool(incl[best, m]))
    return VecArmResult(
        arm=_RANDOM_ARM,
        raw_score=float(scores[best]),
        evaluations=spec.random_samples,
        members=best_members,
    )


def _hill_climb_batch(
    affinity: np.ndarray, incl: np.ndarray, spec: RegimeSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched first-improvement hill climb over all restarts at once.

    Each restart is a row of ``incl``. Within a pass the members are visited in index order; a
    single-member flip that strictly improves the current score (by more than eps) is accepted
    immediately, so later members in the same pass see the updated set and score, exactly as the
    scalar per-restart climb does. A restart stops after any full pass with no accepted flip, and
    a restart accrues num_members evaluations for every pass it was active at the start of. This
    batches the identical scalar decisions; it does not change any of them.
    """

    n_restarts, num_members = incl.shape
    incl = incl.copy()
    score = _score_subsets(
        affinity, incl, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks
    )
    active = np.ones(n_restarts, dtype=bool)
    evaluations = np.zeros(n_restarts, dtype=np.int64)
    for _ in range(_MAX_PASSES):
        improved = np.zeros(n_restarts, dtype=bool)
        for member in range(num_members):
            trial = incl.copy()
            trial[:, member] = ~incl[:, member]
            trial_score = _score_subsets(
                affinity, trial, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks
            )
            accept = active & (trial_score > score + _EPS)
            incl[:, member] = np.where(accept, trial[:, member], incl[:, member])
            score = np.where(accept, trial_score, score)
            improved = improved | accept
            evaluations = evaluations + active.astype(np.int64)
        active = active & improved
        if not active.any():
            break
    return incl, score, evaluations


def vec_run_construction_search(spec: RegimeSpec, seed: int) -> VecArmResult:
    """Vectorized equivalent of run_construction_search: seeded random-restart hill climbing."""

    num_members = spec.num_members
    n_restarts = spec.search_restarts
    stream_seed = (seed ^ _SEARCH_SALT) & _MASK64
    start_floats = _lcg_float_matrix(stream_seed, n_restarts, num_members)
    incl = start_floats < _INCLUSION_PROB
    affinity = _affinity_array(spec)
    polished, scores, climb_evals = _hill_climb_batch(affinity, incl, spec)
    evaluations = n_restarts + int(climb_evals.sum())
    best = int(np.argmax(scores))
    best_members = tuple(int(m) for m in range(num_members) if bool(polished[best, m]))
    return VecArmResult(
        arm=_SEARCH_ARM,
        raw_score=float(scores[best]),
        evaluations=evaluations,
        members=best_members,
    )


def vec_run_oracle(spec: RegimeSpec) -> VecArmResult:
    """Vectorized equivalent of run_oracle: the exhaustive best over all subsets."""

    num_members = spec.num_members
    if num_members > _MAX_ORACLE_MEMBERS:
        raise ConstructionSearchVecRefusal("oracle headroom is only defined for small member counts")
    masks = np.arange(1 << num_members, dtype=np.uint64)
    bit_index = np.arange(num_members, dtype=np.uint64)
    incl = ((masks[:, None] >> bit_index[None, :]) & np.uint64(1)).astype(bool)
    affinity = _affinity_array(spec)
    scores = _score_subsets(
        affinity, incl, spec.size_penalty, spec.synergy_pair, spec.synergy_bonus, spec.num_tasks
    )
    best = int(np.argmax(scores))
    best_members = tuple(int(m) for m in range(num_members) if bool(incl[best, m]))
    return VecArmResult(
        arm=_ORACLE_ARM,
        raw_score=float(scores[best]),
        evaluations=1 << num_members,
        members=best_members,
    )


def vec_evaluate_regime(spec: RegimeSpec, seed: int) -> dict[str, VecArmResult]:
    """Run every arm on one regime and return arm name to VecArmResult, mirroring evaluate_regime.

    The dict keys are byte-identical to the scalar evaluate_regime keys, so the two structures are
    directly comparable field by field.
    """

    return {
        _NO_SEARCH_ARM: vec_run_no_search(spec),
        _GREEDY_ARM: vec_run_greedy(spec),
        _RANDOM_ARM: vec_run_random(spec, seed),
        _SEARCH_ARM: vec_run_construction_search(spec, seed),
        _ORACLE_ARM: vec_run_oracle(spec),
    }
