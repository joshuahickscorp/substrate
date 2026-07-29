"""Build one developmental history from many sealed challenge units.

A challenge unit is a short episode with a single held-out probe. A
developmental history is what the science is actually about: a long ordered
experience in which earlier learning must change later cognition. This module
concatenates units of one family into that history and partitions them into the
three disjoint roles the harness requires.

The partition is by unit, not by probe, and the scoring units are drawn from a
disjoint identifier band so that nothing scored at the end was used to admit a
rewrite along the way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from substrate import genesis_challenge as CH
from substrate import genesis_harness as H
from substrate import genesis_material as M
from substrate import genesis_tournament as T

# Units per developmental history, split by role. Development units admit
# rewrites, retention units guard against forgetting, scoring units are the
# reported measure and are touched by nothing else.
DEVELOPMENT_UNITS = 5
RETENTION_UNITS = 4
SCORING_UNITS = 5
UNITS_PER_HISTORY = DEVELOPMENT_UNITS + RETENTION_UNITS + SCORING_UNITS

# Identifier bands keep the three roles from ever sharing a generated unit.
BAND = 1_000_000
DEVELOPMENT_BAND = 0
RETENTION_BAND = BAND
SCORING_BAND = 2 * BAND
ALTERNATIVE_BAND = 3 * BAND


class HistoryRefused(RuntimeError):
    """A developmental history was assembled in a way that would void it."""


@dataclass(frozen=True, slots=True)
class _Role:
    name: str
    band: int
    count: int


ROLES = (
    _Role(H.DEVELOPMENT, DEVELOPMENT_BAND, DEVELOPMENT_UNITS),
    _Role(H.RETENTION, RETENTION_BAND, RETENTION_UNITS),
    _Role(H.SCORING, SCORING_BAND, SCORING_UNITS),
)


class _SplitScorer:
    """Scores exactly the probes of one role, and holds nothing else."""

    def __init__(self, entries: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        self._expected = dict(entries)
        if not self._expected:
            raise HistoryRefused("a split scorer needs at least one sealed probe")

    def __call__(self, answers: Sequence[M.Answer]) -> float:
        by_index = {int(answer.probe_index): answer for answer in answers}
        correct = 0
        for probe_index, expected in self._expected.items():
            answer = by_index.get(probe_index)
            if answer is None or answer.abstained:
                continue
            if tuple(int(value) for value in answer.value) == expected:
                correct += 1
        return correct / len(self._expected)

    def probe_indices(self) -> frozenset[int]:
        return frozenset(self._expected)


def _units_for_role(family: str, split: str, history_id: int, role: _Role, seed_namespace: str) -> list[CH.Unit]:
    base = role.band + history_id * UNITS_PER_HISTORY
    return [CH.generate(family, split, base + offset, seed_namespace=seed_namespace) for offset in range(role.count)]


def build_history(
    *,
    family: str,
    split: str,
    history_id: int,
    seed_namespace: str,
) -> T.Unit:
    """Assemble one developmental history and its three disjoint probe roles."""
    observations: list[M.Observation] = []
    probes_by_role: dict[str, list[M.Probe]] = {role.name: [] for role in ROLES}
    entries_by_role: dict[str, list[tuple[int, tuple[int, ...]]]] = {role.name: [] for role in ROLES}

    observation_index = 0
    probe_index = 0
    elapsed = 0
    for role in ROLES:
        for unit in _units_for_role(family, split, history_id, role, seed_namespace):
            local_to_global: dict[int, int] = {}
            for observation in unit.observations:
                elapsed += max(1, observation.elapsed_ms)
                observations.append(
                    M.Observation(
                        index=observation_index,
                        channel=observation.channel,
                        payload=observation.payload,
                        elapsed_ms=observation.elapsed_ms or 1,
                        teaching=observation.teaching,
                        modality=observation.modality,
                    )
                )
                observation_index += 1
            for probe in unit.probes:
                local_to_global[probe.index] = probe_index
                probes_by_role[role.name].append(
                    M.Probe(
                        index=probe_index,
                        family=probe.family,
                        channel=probe.channel,
                        probe=probe.probe,
                        arity=probe.arity,
                    )
                )
                probe_index += 1
            for local_index, expected in unit.sealed.entries():
                if local_index not in local_to_global:
                    raise HistoryRefused(f"{family}: sealed answer refers to a probe the unit did not publish")
                entries_by_role[role.name].append((local_to_global[local_index], expected))

    split_probes = H.ProbeSplit(
        development=tuple(probes_by_role[H.DEVELOPMENT]),
        retention=tuple(probes_by_role[H.RETENTION]),
        scoring=tuple(probes_by_role[H.SCORING]),
    )

    scorers = {role.name: _SplitScorer(entries_by_role[role.name]) for role in ROLES}
    for left in ROLES:
        for right in ROLES:
            if left.name >= right.name:
                continue
            shared = scorers[left.name].probe_indices() & scorers[right.name].probe_indices()
            if shared:
                raise HistoryRefused(f"{family}: sealed probe indices shared between {left.name} and {right.name}")

    alternative = _alternative_observations(family, split, history_id, seed_namespace, len(observations))

    # The generating structure, keyed the way the oracle addresses it. This is
    # the upper reference: without it there is no headroom measurement and no
    # way to tell a valid null from a task nothing can solve. It is returned
    # separately from the history and never enters the observation stream.
    expected_by_probe: dict[int, tuple[int, ...]] = {}
    for entries in entries_by_role.values():
        expected_by_probe.update(dict(entries))
    oracle_structure: dict[str, tuple[int, ...]] = {}
    for probe in split_probes.development + split_probes.retention + split_probes.scoring:
        expected = expected_by_probe.get(probe.index)
        if expected is None:
            raise HistoryRefused(f"{family}: probe {probe.index} has no sealed answer")
        address = f"{probe.family}|{probe.channel}|{','.join(str(int(value)) for value in probe.probe)}"
        oracle_structure[address] = expected

    return T.Unit(
        history_id=history_id,
        family=family,
        observations=tuple(observations),
        alternative_observations=alternative,
        probes=split_probes,
        judge=H.Judge(
            score_development=scorers[H.DEVELOPMENT],
            score_retention=scorers[H.RETENTION],
            score_scoring=scorers[H.SCORING],
        ),
        oracle_structure=oracle_structure,
    )


def _alternative_observations(
    family: str,
    split: str,
    history_id: int,
    seed_namespace: str,
    length: int,
) -> tuple[M.Observation, ...]:
    """A genuinely different history of the same length, for the wrong-history control."""
    collected: list[M.Observation] = []
    offset = 0
    index = 0
    while len(collected) < length:
        unit = CH.generate(family, split, ALTERNATIVE_BAND + history_id * UNITS_PER_HISTORY + offset, seed_namespace=seed_namespace)
        for observation in unit.observations:
            if len(collected) >= length:
                break
            collected.append(
                M.Observation(
                    index=index,
                    channel=observation.channel,
                    payload=observation.payload,
                    elapsed_ms=observation.elapsed_ms or 1,
                    teaching=observation.teaching,
                    modality=observation.modality,
                )
            )
            index += 1
        offset += 1
        if offset > length + UNITS_PER_HISTORY:
            raise HistoryRefused(f"{family}: could not assemble an alternative history of length {length}")
    return tuple(collected)


def provider(split: str, seed_namespace: str) -> Any:
    """A tournament unit provider bound to one split and seed namespace."""

    def build(family: str, history_id: int) -> T.Unit:
        return build_history(family=family, split=split, history_id=history_id, seed_namespace=seed_namespace)

    return build


def demo() -> None:
    unit = build_history(family="unseen_concept_acquisition", split="principal", history_id=0, seed_namespace="demo")
    assert len(unit.probes.development) == DEVELOPMENT_UNITS
    assert len(unit.probes.retention) == RETENTION_UNITS
    assert len(unit.probes.scoring) == SCORING_UNITS
    assert len(unit.observations) > 100, len(unit.observations)
    assert len(unit.alternative_observations) == len(unit.observations)

    identity = M._digest([observation.digest() for observation in unit.observations])
    alternative = M._digest([observation.digest() for observation in unit.alternative_observations])
    assert identity != alternative, "the wrong-history control would have received the real history"

    # Determinism: the same coordinates rebuild the same history exactly.
    again = build_history(family="unseen_concept_acquisition", split="principal", history_id=0, seed_namespace="demo")
    assert M._digest([observation.digest() for observation in again.observations]) == identity

    # A different split must not reuse the same probes.
    replication = build_history(family="unseen_concept_acquisition", split="replication", history_id=0, seed_namespace="demo")
    principal_probes = {probe.probe for probe in unit.probes.scoring}
    replication_probes = {probe.probe for probe in replication.probes.scoring}
    assert principal_probes != replication_probes

    # An arm that abstains everywhere scores zero on every role.
    empty: list[M.Answer] = []
    assert unit.judge.score_development(empty) == 0.0
    assert unit.judge.score_scoring(empty) == 0.0
    print(f"genesis history self-check passed: {len(unit.observations)} observations, {UNITS_PER_HISTORY} units per history")


if __name__ == "__main__":
    demo()
