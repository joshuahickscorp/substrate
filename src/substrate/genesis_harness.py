"""One developmental history, run identically for every arm.

The harness owns three things the arms and the evaluator must not own:

1. Delivery. Every arm receives the same observations in the same order from
   the same object, so ``information`` parity is a fact rather than a claim.
   The wrong-history and shuffled-history controls are fed a transformed stream
   BY the harness, so a control cannot quietly un-transform its own input.

2. The three-way probe split. Proposals are verified against the development
   split, retention is checked against an earlier split, and the reported score
   comes from a scoring split that nothing in the run has touched. Without this
   separation, plasticity reads the held-out outcome and every result is void.

3. Time. Continuous-time arms are advanced by the harness clock. A material
   that advanced its own clock would be running a declared mutation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_material as M
from substrate import genesis_statistics as S

# Probe roles. Disjoint by construction.
DEVELOPMENT = "development"
RETENTION = "retention"
SCORING = "scoring"
PROBE_ROLES = (DEVELOPMENT, RETENTION, SCORING)


class HarnessRefused(RuntimeError):
    """The run violated a condition that would invalidate the result."""


@dataclass(frozen=True, slots=True)
class ProbeSplit:
    """Three disjoint probe sets drawn from one unit."""

    development: tuple[M.Probe, ...]
    retention: tuple[M.Probe, ...]
    scoring: tuple[M.Probe, ...]

    def __post_init__(self) -> None:
        indices = [
            {probe.index for probe in self.development},
            {probe.index for probe in self.retention},
            {probe.index for probe in self.scoring},
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                overlap = indices[left] & indices[right]
                if overlap:
                    raise HarnessRefused(f"probe splits overlap on indices {sorted(overlap)}")
        if not self.scoring:
            raise HarnessRefused("a unit with no scoring probes cannot be measured")


def split_probes(probes: Sequence[M.Probe]) -> ProbeSplit:
    """Deterministic three-way split when a generator does not supply one.

    Ordering is by probe index so the split is stable across processes and
    identical for every arm.
    """
    ordered = sorted(probes, key=lambda probe: probe.index)
    if len(ordered) < 3:
        raise HarnessRefused("a unit needs at least three probes to split")
    development, retention, scoring = [], [], []
    for position, probe in enumerate(ordered):
        (development, retention, scoring)[position % 3].append(probe)
    return ProbeSplit(tuple(development), tuple(retention), tuple(scoring))


@dataclass(frozen=True, slots=True)
class Stream:
    """The observation stream an arm actually receives."""

    observations: tuple[M.Observation, ...]
    transform: str = "identity"

    def digest(self) -> str:
        return M._digest([observation.digest() for observation in self.observations])


def identity_stream(observations: Sequence[M.Observation]) -> Stream:
    return Stream(tuple(observations), "identity")


def shuffled_stream(observations: Sequence[M.Observation], *, seed: int) -> Stream:
    """Deterministically reorder. The arm never learns the original order."""
    ordered = list(observations)
    count = len(ordered)
    state = (seed * 0x9E3779B97F4A7C15 + 0x1234567) & 0xFFFFFFFFFFFFFFFF
    for position in range(count - 1, 0, -1):
        state ^= (state >> 33) & 0xFFFFFFFFFFFFFFFF
        state = (state * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
        swap = state % (position + 1)
        ordered[position], ordered[swap] = ordered[swap], ordered[position]
    if count > 1 and all(left is right for left, right in zip(ordered, observations, strict=True)):
        raise HarnessRefused("the shuffled history control received the ordered history")
    return Stream(tuple(ordered), "shuffled")


def wrong_stream(observations: Sequence[M.Observation], other: Sequence[M.Observation]) -> Stream:
    """A different history of the same shape, from a different unit."""
    if len(other) != len(observations):
        raise HarnessRefused("the wrong-history control needs a same-length alternative history")
    if M._digest([row.digest() for row in other]) == M._digest([row.digest() for row in observations]):
        raise HarnessRefused("the wrong history control received the correct history")
    return Stream(tuple(other), "wrong_history")


@dataclass
class ArmRun:
    """Everything one arm produced on one developmental history."""

    arm: str
    mechanism: str
    stream_transform: str
    stream_digest: str
    score: float
    retention_score: float
    development_score: float
    receipts: list[M.Receipt] = field(default_factory=list)
    cost: dict[str, int] = field(default_factory=dict)
    opportunity: dict[str, Any] = field(default_factory=dict)
    deprived: tuple[str, ...] = ()
    wall_clock_seconds: float = 0.0
    peak_resident_bytes: int = 0
    exhausted: str | None = None


@dataclass(frozen=True, slots=True)
class Judge:
    """The evaluator's two scalars, supplied by whoever holds the sealed answers.

    ``score`` maps answers to a scalar in [0, 1]. The harness never sees an
    expected value and never passes one to an arm.
    """

    score_development: Callable[[Sequence[M.Answer]], float]
    score_retention: Callable[[Sequence[M.Answer]], float]
    score_scoring: Callable[[Sequence[M.Answer]], float]


def _answer_all(material: M.CognitiveMaterial, probes: Sequence[M.Probe]) -> list[M.Answer]:
    return [material.answer(probe) for probe in probes]


def run_arm(
    *,
    arm: str,
    factory: Callable[[M.Opportunity], M.CognitiveMaterial],
    stream: Stream,
    probes: ProbeSplit,
    judge: Judge,
    envelope: str,
    operation_budget: int,
    durable_write_budget: int,
    deprived: Sequence[str] = (),
    consolidation_every: int = 8,
    advance_ms: int = 0,
) -> ArmRun:
    """Run one arm over one history under an enforced equal budget."""
    opportunity = M.equal_opportunity(
        envelope=envelope,
        observations=stream.observations,
        sensor_channels=tuple(sorted({observation.channel for observation in stream.observations})),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        deprived=deprived,
    )
    material = factory(opportunity)
    started = time.monotonic()
    exhausted: str | None = None
    receipts: list[M.Receipt] = []

    try:
        for position, observation in enumerate(stream.observations, start=1):
            material.observe(observation)
            if advance_ms and hasattr(material, "advance"):
                material.advance(advance_ms)
            if position % consolidation_every:
                continue
            before = judge.score_development(_answer_all(material, probes.development))
            retention_before = judge.score_retention(_answer_all(material, probes.retention))
            proposals = material.propose()
            if not proposals:
                continue
            verdicts = []
            for proposal in proposals:
                after = judge.score_development(_answer_all(material, probes.development))
                retention_after = judge.score_retention(_answer_all(material, probes.retention))
                improvement = after - before
                retention = retention_after - retention_before
                verdicts.append(
                    M.Verdict(
                        proposal_id=proposal.proposal_id,
                        admitted=improvement > 0.0 and retention >= 0.0,
                        improvement=improvement,
                        retention=retention,
                    )
                )
            receipts.extend(material.apply(verdicts))
    except M.ResourceExhausted as error:
        exhausted = str(error)

    development_score = judge.score_development(_answer_all(material, probes.development))
    retention_score = judge.score_retention(_answer_all(material, probes.retention))
    scoring = judge.score_scoring(_answer_all(material, probes.scoring))
    elapsed = time.monotonic() - started

    return ArmRun(
        arm=arm,
        mechanism=material.mechanism,
        stream_transform=stream.transform,
        stream_digest=stream.digest(),
        score=scoring,
        retention_score=retention_score,
        development_score=development_score,
        receipts=receipts,
        cost=material.cost(),
        opportunity=opportunity.vector(),
        deprived=tuple(deprived),
        wall_clock_seconds=elapsed,
        peak_resident_bytes=opportunity.ledger.peak_resident_bytes,
        exhausted=exhausted,
    )


def run_history(
    *,
    history_id: int,
    family: str,
    arms: Mapping[str, Callable[[M.Opportunity], M.CognitiveMaterial]],
    observations: Sequence[M.Observation],
    alternative_observations: Sequence[M.Observation],
    probes: ProbeSplit,
    judge: Judge,
    envelope: str,
    operation_budget: int,
    durable_write_budget: int,
) -> dict[str, Any]:
    """Run every arm over one developmental history under identical conditions."""
    runs: dict[str, ArmRun] = {}
    for arm, factory in sorted(arms.items()):
        canonical = C.S2_ALIASES.get(arm, arm)
        deprived = C.BASELINE_DEPRIVATION.get(canonical, ())
        if "correct_history" in deprived:
            stream = wrong_stream(observations, alternative_observations)
        elif "history_order" in deprived:
            stream = shuffled_stream(observations, seed=history_id)
        else:
            stream = identity_stream(observations)
        runs[arm] = run_arm(
            arm=arm,
            factory=factory,
            stream=stream,
            probes=probes,
            judge=judge,
            envelope=envelope,
            operation_budget=operation_budget,
            durable_write_budget=durable_write_budget,
            deprived=deprived,
        )

    delivered = {arm: run.stream_digest for arm, run in runs.items()}
    undeprived = {
        arm: digest
        for arm, digest in delivered.items()
        if not set(C.BASELINE_DEPRIVATION.get(C.S2_ALIASES.get(arm, arm), ())) & {"correct_history", "history_order"}
    }
    if len(set(undeprived.values())) > 1:
        raise HarnessRefused(f"arms received different histories: {undeprived}")

    return {
        "history_id": history_id,
        "family": family,
        "envelope": envelope,
        "runs": runs,
        "delivered_stream_digests": delivered,
        "scores": [S.HistoryScore(history_id, arm, run.score) for arm, run in runs.items()],
        "activation": False,
    }


def demo() -> None:
    """Runnable self-check for the guarantees the harness is responsible for."""
    probes = [M.Probe(index, "f", "c", (index % 3 - 1,), 1) for index in range(9)]
    split = split_probes(probes)
    assert len(split.development) == 3 and len(split.retention) == 3 and len(split.scoring) == 3
    everything = {probe.index for probe in split.development + split.retention + split.scoring}
    assert everything == set(range(9))

    observations = [M.Observation(index, "c", (index % 3 - 1,)) for index in range(16)]
    shuffled = shuffled_stream(observations, seed=7)
    assert shuffled.digest() != identity_stream(observations).digest()

    other = [M.Observation(index, "c", ((index + 1) % 3 - 1,)) for index in range(16)]
    wrong = wrong_stream(observations, other)
    assert wrong.digest() != identity_stream(observations).digest()

    try:
        wrong_stream(observations, observations)
    except HarnessRefused:
        pass
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("wrong-history guard did not fire")

    try:
        ProbeSplit((probes[0],), (probes[0],), (probes[1],))
    except HarnessRefused:
        pass
    else:  # pragma: no cover
        raise AssertionError("overlapping probe splits were accepted")

    print("genesis harness self-check passed")


if __name__ == "__main__":
    demo()
