"""One developmental history under the Genesis II update economy.

This is a separate harness from ``genesis_harness`` rather than an edit to it,
because the Genesis I harness is frozen evidence: its result has to stay
byte-reproducible.  What changes here is one thing, and it is an instrument
repair rather than a threshold relaxation.

The Genesis I harness verified every proposal individually against the
development probes and admitted it only if the score strictly rose.  A
developmental history has five development probes, so the smallest observable
improvement is 0.2.  A single exact associative write essentially never moves a
five-probe score by a fifth, which means the instrument could not admit a
micro-write *at all*.  Running the tournament that way measures retrieval
heuristics over a nearly empty durable store: in a direct trace of three
families, the arms had committed one, zero and four associations by the end of
the history.

Genesis II is a program about the economy of small writes, so an instrument
that cannot admit a small write cannot measure it.  The repair is the one the
update economy already implies: proposals are verified in granularity batches.
A batch of cheap micro-writes is measured as one unit, because that is what it
is -- one unit of learning delivered in many small pieces.  An expensive
structural or topological change is still verified alone and still has to
strictly improve the score, because that is a single large claim.

The repair is applied identically to every arm, including S2, and it does not
touch the probe split, the sealing, the budgets or the deprivations.  A batch
that harms the score is rolled back whole.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from substrate import genesis2_config as C2
from substrate import genesis2_ledger as GL
from substrate import genesis_material as M
from substrate.genesis_harness import (  # reused unchanged: delivery, splits, streams
    DEVELOPMENT,
    RETENTION,
    SCORING,
    HarnessRefused,
    Judge,
    ProbeSplit,
    Stream,
    identity_stream,
    shuffled_stream,
    split_probes,
    wrong_stream,
)

__all__ = [
    "DEVELOPMENT",
    "RETENTION",
    "SCORING",
    "ArmRun",
    "HarnessRefused",
    "Judge",
    "ProbeSplit",
    "Stream",
    "identity_stream",
    "run_arm",
    "run_history",
    "shuffled_stream",
    "split_probes",
    "wrong_stream",
]

#: Granularities cheap enough to be verified as a batch.  A batch is admitted
#: when it does not harm: recording a fact that has not yet paid is still
#: learning, and its cost is charged to the ledger and to the memory envelope
#: rather than forgiven.
BATCH_GRANULARITIES = frozenset({"micro_association", "local_low_bit_adjustment", "association_promotion"})

#: Granularities expensive enough to answer for themselves.  These must strictly
#: improve the development score, alone, or they are reversed.
SINGLE_GRANULARITIES = frozenset({"structural_consolidation", "topology_revision"})

#: Legacy proposal kinds from the Genesis I arms, which name their own write
#: rather than a granularity.  They are batched: an S2 association write is a
#: micro-write whatever it is called.
_LEGACY_BATCH_KINDS = frozenset({"monolith_association_write"})

# Every arm receives the same maximum number of micro-writes in one admission
# decision. Without this cap, S2's unbounded proposal surface was evaluated as
# one enormous all-or-nothing batch while Genesis II candidates were capped.
MICRO_BATCH_LIMIT = 24


def _granularity_of(kind: str) -> str:
    if kind in C2.GRANULARITY_COST_WEIGHT:
        return kind
    if kind in _LEGACY_BATCH_KINDS:
        return "micro_association"
    return "micro_association"


def _batched(kind: str) -> bool:
    granularity = _granularity_of(kind)
    return granularity not in SINGLE_GRANULARITIES


def _micro_batches(proposals: Sequence[M.Proposal]) -> list[list[M.Proposal]]:
    return [list(proposals[start : start + MICRO_BATCH_LIMIT]) for start in range(0, len(proposals), MICRO_BATCH_LIMIT)]


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
    rolled_back: int = 0
    ledger: dict[str, Any] = field(default_factory=dict)
    mechanisms: dict[str, Any] = field(default_factory=dict)


def _answer_all(material: M.CognitiveMaterial, probes: Sequence[M.Probe]) -> list[M.Answer]:
    return [material.answer(probe) for probe in probes]


def _verify_group(
    material: M.CognitiveMaterial,
    group: Sequence[M.Proposal],
    judge: Judge,
    probes: ProbeSplit,
    ledger: GL.UpdateLedger,
    *,
    strict: bool,
) -> tuple[list[M.Receipt], int]:
    """Tentatively commit a group, measure it once, keep or reverse it whole."""
    before = judge.score_development(_answer_all(material, probes.development))
    retention_before = judge.score_retention(_answer_all(material, probes.retention))

    tentative: list[M.Receipt] = []
    for proposal in group:
        granularity = _granularity_of(proposal.kind)
        ledger.record(
            GL.UpdateRecord(
                proposal_id=proposal.proposal_id,
                arm=material.name,
                granularity=granularity,
                bytes_written=int(proposal.cost_bytes),
                information_introduced=len(proposal.delta),
                compute=1,
                scope=str(proposal.target),
                durability="durable",
            )
        )
        emitted = material.apply([M.Verdict(proposal.proposal_id, True, 0.0, 0.0)])
        tentative.extend(emitted)

    if not tentative:
        return [], 0

    improvement = judge.score_development(_answer_all(material, probes.development)) - before
    retention = judge.score_retention(_answer_all(material, probes.retention)) - retention_before
    admitted = (improvement > 0.0) if strict else (improvement >= 0.0)
    admitted = admitted and retention >= 0.0

    rolled_back = 0
    if not admitted:
        for receipt in reversed(tentative):
            cast(M.MaterialBase, material).rollback(receipt)
            rolled_back += 1
    elif hasattr(material, "finalize_receipt"):
        for receipt in tentative:
            cast(Any, material).finalize_receipt(receipt)

    receipts: list[M.Receipt] = []
    share = improvement / len(tentative) if tentative else 0.0
    for receipt in tentative:
        ledger.settle(
            receipt.proposal_id,
            committed=admitted,
            future_utility=share if admitted else 0.0,
            rollback_cost=0 if admitted else int(receipt.cost_bytes),
        )
        receipts.append(
            M.Receipt(
                proposal_id=receipt.proposal_id,
                kind=receipt.kind,
                target=receipt.target,
                committed=admitted,
                improvement=improvement,
                retention=retention,
                durable_state_digest_before=receipt.durable_state_digest_before,
                durable_state_digest_after=(receipt.durable_state_digest_after if admitted else receipt.durable_state_digest_before),
                cost_bytes=receipt.cost_bytes,
                mechanism=receipt.mechanism,
            )
        )
    return receipts, rolled_back


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
    byte_budget: int | None = None,
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
    if byte_budget is not None:
        # Calibrated relative envelopes bind in bytes, not in envelope names.
        opportunity.ledger.byte_budget = int(byte_budget)
    material = factory(opportunity)
    ledger = GL.UpdateLedger(arm=arm)
    started = time.monotonic()
    exhausted: str | None = None
    receipts: list[M.Receipt] = []
    rolled_back = 0

    try:
        for position, observation in enumerate(stream.observations, start=1):
            material.observe(observation)
            if advance_ms and hasattr(material, "advance"):
                material.advance(advance_ms)
            if position % consolidation_every:
                continue
            proposals = list(material.propose())
            if not proposals:
                continue
            batch = [proposal for proposal in proposals if _batched(proposal.kind)]
            singles = [proposal for proposal in proposals if not _batched(proposal.kind)]
            for micro_batch in _micro_batches(batch):
                emitted, reversed_count = _verify_group(material, micro_batch, judge, probes, ledger, strict=False)
                receipts.extend(emitted)
                rolled_back += reversed_count
            for proposal in singles:
                emitted, reversed_count = _verify_group(material, [proposal], judge, probes, ledger, strict=True)
                receipts.extend(emitted)
                rolled_back += reversed_count
    except M.ResourceExhausted as error:
        exhausted = str(error)

    development_score = judge.score_development(_answer_all(material, probes.development))
    retention_score = judge.score_retention(_answer_all(material, probes.retention))
    scoring = judge.score_scoring(_answer_all(material, probes.scoring))
    elapsed = time.monotonic() - started
    ledger.wall_clock_seconds = elapsed

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
        rolled_back=rolled_back,
        ledger=ledger.report(),
        mechanisms=material.mechanism_report() if hasattr(material, "mechanism_report") else {},
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
    byte_budget: int | None = None,
) -> dict[str, Any]:
    """Run every arm over one developmental history under identical conditions."""
    runs: dict[str, ArmRun] = {}
    for arm, factory in sorted(arms.items()):
        canonical = C2.S2_ALIASES.get(arm, arm)
        deprived = C2.BASELINE_DEPRIVATION.get(canonical, ())
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
            byte_budget=byte_budget,
        )

    delivered = {arm: run.stream_digest for arm, run in runs.items()}
    undeprived = {
        arm: digest
        for arm, digest in delivered.items()
        if not set(C2.BASELINE_DEPRIVATION.get(C2.S2_ALIASES.get(arm, arm), ())) & {"correct_history", "history_order"}
    }
    if len(set(undeprived.values())) > 1:
        raise HarnessRefused(f"arms received different histories: {undeprived}")

    return {
        "history_id": history_id,
        "family": family,
        "envelope": envelope,
        "runs": runs,
        "delivered_stream_digests": delivered,
        "activation": False,
    }


def demo() -> None:
    """Runnable self-check for the one property this harness changes."""
    probes = [M.Probe(index, "f", "c", (index,), 1) for index in range(9)]
    split = split_probes(probes)

    # A batched group is admitted when it does not harm; a single expensive
    # change is refused unless it strictly improves.
    assert _batched("micro_association") is True
    assert _batched("monolith_association_write") is True, "S2's write must batch like any micro-write"
    assert _batched("structural_consolidation") is False
    assert _batched("topology_revision") is False

    # Granularity resolution never silently invents a free update unit.
    assert _granularity_of("structural_consolidation") == "structural_consolidation"
    assert _granularity_of("something_unknown") == "micro_association"

    # The probe split guarantees inherited from the Genesis I harness still hold.
    everything = {probe.index for probe in split.development + split.retention + split.scoring}
    assert everything == set(range(9))
    try:
        ProbeSplit((probes[0],), (probes[0],), (probes[1],))
    except HarnessRefused:
        pass
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("overlapping probe splits were accepted")

    print("genesis2 harness self-check passed: batched micro-writes, single structural verification")


if __name__ == "__main__":
    demo()
