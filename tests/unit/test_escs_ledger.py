from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from mop.escs.accounting import WorkVector
from mop.escs.events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    EvidenceClass,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
)
from mop.escs.ledger import EventLedger
from mop.substrate.events import BranchRef, EventRef, canonical_sha256

SOURCE = {"producer": "ledger-fixture"}


def make_observation(name: str, tick: int = 0) -> ObservationEvent:
    return ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:{name}",),
        adapter_version="v1",
        sensor_scope={"sensor": name},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
    )


def make_hypothesis(
    parents: tuple[ObservationEvent | HypothesisEvent | ConsequenceEvent, ...],
    *,
    tick: int,
    status: EpistemicStatus = EpistemicStatus.INFERRED,
    branch: BranchRef = BranchRef("branch:factual"),
) -> HypothesisEvent:
    ids = tuple(parent.event_id for parent in parents)
    return HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=status,
        referent_hypotheses={"r": 1.0},
        factor_change_distribution={"f": 1.0},
        decision_relevance_distribution={"d": 1.0},
        reducibility_distribution={"r": 1.0},
        supporting_event_ids=ids,
        calibrated_confidence=0.6,
        abstention_reason=None,
        predicted_value_of_further_computation=1.0,
        causal_parent_ids=ids,
        counterfactual_branch_id=branch,
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
    )


def make_commitment(
    parents: tuple[HypothesisEvent, ...], *, tick: int, branch: BranchRef = BranchRef("branch:factual")
) -> CommitmentEvent:
    return CommitmentEvent.create(
        coalition_id="coalition:test",
        commitment_kind=CommitmentKind.EXTERNAL_ACTION,
        committed_payload={"action": "act"},
        decision_distribution={"act": 1.0},
        deadline_tick=tick + 5,
        predicted_utility_vector={"reward": 1.0},
        predicted_full_cost=WorkVector(actor_execution=1),
        causal_parent_ids=tuple(parent.event_id for parent in parents),
        counterfactual_branch_id=branch,
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
    )


def make_consequence(commitment: CommitmentEvent, *, tick: int) -> ConsequenceEvent:
    return ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={"state": tick},
        realized_utility_vector={"reward": 1.0},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(actor_execution=1),
        causal_parent_ids=(commitment.event_id,),
        counterfactual_branch_id=commitment.branch_id,
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
    )


def test_strict_plural_lifecycle_replays_deterministically() -> None:
    first = make_observation("one")
    second = make_observation("two")
    inferred = make_hypothesis((first, second), tick=1)
    commitment = make_commitment((inferred,), tick=2)
    consequence = make_consequence(commitment, tick=3)
    ledger = EventLedger()
    for event in (first, second, inferred, commitment, consequence):
        ledger.append(event)

    replay = EventLedger.replay(ledger.payload())

    assert replay.payload() == ledger.payload()
    assert replay.sha256 == ledger.sha256
    assert replay.verify() == []
    assert replay.consequences_for(commitment.event_id) == (consequence,)
    assert replay.commitments_for(inferred.event_id) == (commitment,)


def test_ledger_rejects_missing_parents_and_invalid_stage_transitions() -> None:
    observed = make_observation("one")
    inferred = make_hypothesis((observed,), tick=1)
    ledger = EventLedger()
    with pytest.raises(ValueError, match="missing causal parent"):
        ledger.append(inferred)
    ledger.append(observed)
    ledger.append(inferred)

    illegal_observation = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:illegal",),
        adapter_version="v1",
        sensor_scope={"sensor": "illegal"},
        transport_and_detection_cost=WorkVector(),
        causal_parent_ids=(inferred.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=SOURCE,
    )
    with pytest.raises(ValueError, match="observations may descend"):
        ledger.append(illegal_observation)


def test_counterfactual_root_is_explicit_and_cannot_authorize_factual_state() -> None:
    observed = make_observation("one")
    branch = BranchRef("branch:simulation-one")
    simulated = make_hypothesis((observed,), tick=1, status=EpistemicStatus.SIMULATED, branch=branch)
    simulated_commitment = make_commitment((simulated,), tick=2, branch=branch)
    ledger = EventLedger()
    for event in (observed, simulated, simulated_commitment):
        ledger.append(event)

    factual_commitment = make_commitment((simulated,), tick=3)
    with pytest.raises(ValueError, match="cannot authorize a factual event"):
        ledger.append(factual_commitment)
    assert ledger.events_on_branch(branch) == (simulated, simulated_commitment)


def test_counterfactual_branch_cannot_start_with_commitment_or_cross_later() -> None:
    observed = make_observation("one")
    inferred = make_hypothesis((observed,), tick=1)
    branch = BranchRef("branch:simulation-two")
    bad_root = make_commitment((inferred,), tick=2, branch=branch)
    ledger = EventLedger()
    ledger.append(observed)
    ledger.append(inferred)
    with pytest.raises(ValueError, match="begin with a simulated hypothesis"):
        ledger.append(bad_root)


def test_consequence_binding_rejects_wrong_commitment_and_time_travel() -> None:
    observed = make_observation("one")
    inferred = make_hypothesis((observed,), tick=1)
    first = make_commitment((inferred,), tick=2)
    second = CommitmentEvent.create(
        coalition_id="coalition:other",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={"reason": "uncertain"},
        decision_distribution={"abstain": 1.0},
        deadline_tick=7,
        predicted_utility_vector={"reward": 0.0},
        predicted_full_cost=WorkVector(),
        causal_parent_ids=(inferred.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=SOURCE,
    )
    wrong = ConsequenceEvent.create(
        commitment_event_id=first.event_id,
        observed_outcome={"state": 1},
        realized_utility_vector={"reward": 0.0},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(),
        causal_parent_ids=(first.event_id, second.event_id),
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance=SOURCE,
    )
    ledger = EventLedger()
    for event in (observed, inferred, first, second):
        ledger.append(event)
    with pytest.raises(ValueError, match="exactly its bound commitment"):
        ledger.append(wrong)

    early = make_consequence(first, tick=1)
    with pytest.raises(ValueError, match="begins before causal parent"):
        ledger.append(early)


def test_ledger_replay_fails_closed_on_hash_and_schema_drift() -> None:
    ledger = EventLedger()
    ledger.append(make_observation("one"))
    payload = deepcopy(ledger.payload())
    payload["entries"][0]["event"]["body"]["adapter_version"] = "tampered"
    with pytest.raises(ValueError, match="payload digest mismatch"):
        EventLedger.from_payload(payload)

    payload = deepcopy(ledger.payload())
    payload["entries"][0]["unknown"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        EventLedger.from_payload(payload)


def test_ledger_unknown_lookup_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown event"):
        EventLedger().get(EventRef("event:absent"))


def test_oracle_ancestry_cannot_be_laundered_to_a_weaker_evidence_class() -> None:
    observed = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:oracle-parent",),
        adapter_version="v1",
        sensor_scope={"sensor": "oracle-fixture"},
        transport_and_detection_cost=WorkVector(),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    laundered = HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"r": 1.0},
        factor_change_distribution={"f": 1.0},
        decision_relevance_distribution={"d": 1.0},
        reducibility_distribution={"r": 1.0},
        supporting_event_ids=(observed.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.0,
        causal_parent_ids=(observed.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )
    retained = replace(
        laundered,
        envelope=replace(
            laundered.envelope,
            evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
            event_id=EventRef(
                "event:"
                + canonical_sha256(
                    {
                        **laundered.envelope.identity_payload(),
                        "evidence_class": EvidenceClass.ORACLE_NONPROMOTABLE.value,
                    }
                )
            ),
        ),
    )
    ledger = EventLedger()
    ledger.append(observed)
    with pytest.raises(ValueError, match="cannot downgrade causal-parent taint"):
        ledger.append(laundered)
    ledger.append(retained)

    replay = EventLedger.replay(ledger.payload())
    assert replay.events == (observed, retained)
    assert replay.verify() == []


def test_batch_append_is_incremental_atomic_and_rolls_back_injected_second_failure() -> None:
    observed = make_observation("atomic")
    inferred = make_hypothesis((observed,), tick=1)

    class FailingSecondAppend(EventLedger):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def append(self, event):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected second append failure")
            return super().append(event)

    ledger = FailingSecondAppend()
    with pytest.raises(RuntimeError, match="second append"):
        ledger.append_batch((observed, inferred))

    assert ledger.events == ()
    assert ledger.entries == ()
    assert ledger.head_sha256 is None
    assert ledger._children_by_parent == {}
    assert ledger._consequences_by_commitment == {}
    with pytest.raises(TypeError, match="immutable tuple"):
        EventLedger().append_batch([observed])  # type: ignore[arg-type]


def test_batch_append_does_not_iterate_or_copy_historical_entries() -> None:
    class NoHistoryIteration(list):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("batch append scanned historical entries")

    ledger = EventLedger()
    first = make_observation("history-one")
    second = make_observation("history-two", tick=1)
    ledger.append(first)
    ledger._entries = NoHistoryIteration(ledger._entries)

    appended = ledger.append_batch((second,))

    assert len(appended) == 1
    assert ledger.get(second.event_id) == second


def test_verify_detects_derived_event_and_branch_cache_mutation() -> None:
    observed = make_observation("cache-mutation")
    ledger = EventLedger()
    ledger.append(observed)
    ledger._events.clear()
    assert "event lookup cache drift from deterministic replay" in ledger.verify()

    branch_ledger = EventLedger()
    branch_ledger.append(observed)
    branch_ledger._branch_ids.clear()
    assert "branch index cache drift from deterministic replay" in branch_ledger.verify()

    parent_ledger = EventLedger()
    inferred = make_hypothesis((observed,), tick=1)
    parent_ledger.append_batch((observed, inferred))
    parent_ledger._children_by_parent.clear()
    assert "parent-child index cache drift from deterministic replay" in parent_ledger.verify()

    consequence_ledger = EventLedger()
    commitment = make_commitment((inferred,), tick=2)
    consequence = make_consequence(commitment, tick=3)
    consequence_ledger.append_batch((observed, inferred, commitment, consequence))
    consequence_ledger._consequences_by_commitment.clear()
    assert "consequence index cache drift from deterministic replay" in consequence_ledger.verify()


def test_commitment_and_consequence_queries_do_not_scan_event_history() -> None:
    class NoHistoryIteration(list):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("indexed query scanned historical entries")

    observed = make_observation("indexed-query")
    inferred = make_hypothesis((observed,), tick=1)
    commitment = make_commitment((inferred,), tick=2)
    consequence = make_consequence(commitment, tick=3)
    ledger = EventLedger()
    ledger.append_batch((observed, inferred, commitment, consequence))
    ledger._entries = NoHistoryIteration(ledger._entries)

    assert ledger.entry_count == 4
    assert ledger.commitments_for(inferred.event_id) == (commitment,)
    assert ledger.consequences_for(commitment.event_id) == (consequence,)
