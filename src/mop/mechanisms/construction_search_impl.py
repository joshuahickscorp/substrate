
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
    pass


class DeterministicStream:

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

    arm: str
    raw_score: float
    evaluations: int
    members: tuple[int, ...]

    def charged_net(self, per_eval_cost: float) -> float:

        return self.raw_score - per_eval_cost * self.evaluations


def score_coalition(spec: RegimeSpec, members: Iterable[int]) -> float:

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

    members = tuple(sorted(m for m in spec.formation_default if 0 <= m < spec.num_members))
    score = score_coalition(spec, members)
    return ArmResult(arm="no-search", raw_score=score, evaluations=1, members=members)


def run_greedy(spec: RegimeSpec) -> ArmResult:

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
