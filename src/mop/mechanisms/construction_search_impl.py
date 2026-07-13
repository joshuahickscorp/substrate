"""Deterministic construction search and its control policies over toy shadow coalitions.

Every arm here is a pure, seeded, CPU-only routine that scores subsets of candidate members against
a sealed regime objective. The arms are:

- no-search: reuse the G0 formation default coalition; one evaluation, no search.
- greedy-only: one greedy add pass with no restarts; adds the best improving member until none helps.
- random-construction: sample subsets under the matched budget and keep the best raw score.
- construction-search: seeded random-restart hill climbing over subsets, the mechanism under test.
- oracle-headroom: the exhaustive best over all subsets; an uncharged ceiling reference, not a claim.

Each arm reports its raw objective and the number of objective evaluations it spent. The runner
charges the per-evaluation cost against that count, so an arm that spends more evaluations pays more.
Greedy is cheap and deterministic, the search costs more than greedy, and random construction, which
must brute force, costs the most. The construction search wins on the favorable regime by finding a
synergy the greedy pass structurally cannot reach, and it wins net of cost because it is far more
sample efficient than random construction.

Claim scope for the whole module: deterministic programmatic mechanics only; no capability or
natural-data claim. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .construction_search_bed import (
    CHEAP_CONTROLS,
    ORACLE_REFERENCE,
    SEARCH_ARM,
    RegimeSpec,
)

_EPS = 1e-12
_MAX_PASSES = 4
_INCLUSION_PROB = 0.5
_MAX_ORACLE_MEMBERS = 20
_MASK64 = (1 << 64) - 1
_LCG_MULT = 6364136223846793005
_LCG_ADD = 1442695040888963407
_SEARCH_SALT = 0xA24BAED4963EE407
_RANDOM_SALT = 0x2545F4914F6CDD1D


class ConstructionSearchImplRefusal(ValueError):
    """Raised when an arm is asked to run outside its declared, deterministic scope."""


class DeterministicStream:
    """A seeded linear congruential stream of floats in [0, 1). Reproducible, no OS entropy."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ConstructionSearchImplRefusal("stream seed must be nonnegative")
        self._state = ((seed & _MASK64) * _LCG_MULT + _LCG_ADD) & _MASK64

    def next_float(self) -> float:
        self._state = (self._state * _LCG_MULT + _LCG_ADD) & _MASK64
        return ((self._state >> 11) & ((1 << 53) - 1)) / float(1 << 53)


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm's outcome on one regime: the best coalition, its raw score, and evaluations spent.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    arm: str
    raw_score: float
    evaluations: int
    members: tuple[int, ...]

    def charged_net(self, per_eval_cost: float) -> float:
        """The objective net of the charged search cost: raw score minus cost times evaluations."""

        return self.raw_score - per_eval_cost * self.evaluations


def score_coalition(spec: RegimeSpec, members: Iterable[int]) -> float:
    """Multi-task coalition score: mean best-per-task coverage, minus a size penalty, plus synergy."""

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


def run_no_search(spec: RegimeSpec) -> ArmResult:
    """Reuse the G0 formation default coalition. One evaluation, no search."""

    members = tuple(sorted(m for m in spec.formation_default if 0 <= m < spec.num_members))
    score = score_coalition(spec, members)
    return ArmResult(arm="no-search", raw_score=score, evaluations=1, members=members)


def run_greedy(spec: RegimeSpec) -> ArmResult:
    """One greedy add pass with no restarts. Adds the best strictly improving member until none help."""

    members: set[int] = set()
    current = score_coalition(spec, members)
    evaluations = 0
    for _ in range(spec.num_members):
        best_member = -1
        best_score = current
        for candidate in range(spec.num_members):
            if candidate in members:
                continue
            evaluations += 1
            trial = score_coalition(spec, members | {candidate})
            if trial > best_score + _EPS:
                best_score = trial
                best_member = candidate
        if best_member < 0:
            break
        members.add(best_member)
        current = best_score
    return ArmResult(
        arm="greedy-only", raw_score=current, evaluations=evaluations, members=tuple(sorted(members))
    )


def run_random(spec: RegimeSpec, seed: int) -> ArmResult:
    """Sample subsets under the matched budget and keep the best raw score. No hill climbing."""

    stream = DeterministicStream(seed ^ _RANDOM_SALT)
    best_score = float("-inf")
    best_members: tuple[int, ...] = ()
    for _ in range(spec.random_samples):
        subset = {m for m in range(spec.num_members) if stream.next_float() < _INCLUSION_PROB}
        candidate = score_coalition(spec, subset)
        if candidate > best_score:
            best_score = candidate
            best_members = tuple(sorted(subset))
    return ArmResult(
        arm="random-construction",
        raw_score=best_score,
        evaluations=spec.random_samples,
        members=best_members,
    )


def _hill_climb(spec: RegimeSpec, members: set[int]) -> tuple[set[int], float, int]:
    """First-improvement hill climb by single-member flips. Returns the polished set and evals."""

    score = score_coalition(spec, members)
    evaluations = 0
    for _ in range(_MAX_PASSES):
        improved = False
        for member in range(spec.num_members):
            trial = set(members)
            if member in trial:
                trial.discard(member)
            else:
                trial.add(member)
            evaluations += 1
            trial_score = score_coalition(spec, trial)
            if trial_score > score + _EPS:
                members = trial
                score = trial_score
                improved = True
        if not improved:
            break
    return members, score, evaluations


def run_construction_search(spec: RegimeSpec, seed: int) -> ArmResult:
    """Seeded random-restart hill climbing over subsets. The mechanism under test."""

    stream = DeterministicStream(seed ^ _SEARCH_SALT)
    best_score = float("-inf")
    best_members: tuple[int, ...] = ()
    evaluations = 0
    for _ in range(spec.search_restarts):
        start = {m for m in range(spec.num_members) if stream.next_float() < _INCLUSION_PROB}
        evaluations += 1
        polished, score, climb_evals = _hill_climb(spec, start)
        evaluations += climb_evals
        if score > best_score:
            best_score = score
            best_members = tuple(sorted(polished))
    return ArmResult(
        arm=SEARCH_ARM, raw_score=best_score, evaluations=evaluations, members=best_members
    )


def run_oracle(spec: RegimeSpec) -> ArmResult:
    """Exhaustive best over all subsets. An uncharged ceiling reference, never a value claim."""

    if spec.num_members > _MAX_ORACLE_MEMBERS:
        raise ConstructionSearchImplRefusal("oracle headroom is only defined for small member counts")
    best_score = float("-inf")
    best_members: tuple[int, ...] = ()
    for mask in range(1 << spec.num_members):
        subset = tuple(i for i in range(spec.num_members) if (mask >> i) & 1)
        candidate = score_coalition(spec, subset)
        if candidate > best_score:
            best_score = candidate
            best_members = subset
    return ArmResult(
        arm=ORACLE_REFERENCE,
        raw_score=best_score,
        evaluations=1 << spec.num_members,
        members=best_members,
    )


def evaluate_regime(spec: RegimeSpec, seed: int) -> dict[str, ArmResult]:
    """Run every arm on one regime deterministically and return arm name to result."""

    results: dict[str, ArmResult] = {
        "no-search": run_no_search(spec),
        "greedy-only": run_greedy(spec),
        "random-construction": run_random(spec, seed),
        SEARCH_ARM: run_construction_search(spec, seed),
        ORACLE_REFERENCE: run_oracle(spec),
    }
    expected = {*CHEAP_CONTROLS, SEARCH_ARM, ORACLE_REFERENCE}
    if set(results) != expected:
        raise ConstructionSearchImplRefusal("evaluate_regime must cover every arm exactly")
    return results
