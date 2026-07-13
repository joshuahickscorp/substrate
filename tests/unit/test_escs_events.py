from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from mop.escs.accounting import FACTUAL_BRANCH, WorkVector
from mop.escs.events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    EventKind,
    EvidenceClass,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
    event_from_payload,
    state_version_for_parents,
)
from mop.substrate.events import BranchRef

SOURCE = {"producer": "unit-fixture", "oracle_fields_visible": False}


def observation(index: int = 0, *, tick: int = 0) -> ObservationEvent:
    return ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:{index}",),
        adapter_version="adapter-v1",
        sensor_scope={"sensor": "camera", "channel": index},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=2),
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
        measured_creation_cost=WorkVector(event_formation=1),
    )


def hypothesis(*observations: ObservationEvent, tick: int = 1) -> HypothesisEvent:
    parents = tuple(row.event_id for row in observations)
    return HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"candidate:one": 0.7, "candidate:two": 0.3},
        factor_change_distribution={"motion": 0.4},
        decision_relevance_distribution={"relevant": 0.6},
        reducibility_distribution={"reducible": 0.5},
        supporting_event_ids=parents,
        calibrated_confidence=0.7,
        abstention_reason=None,
        predicted_value_of_further_computation=2.0,
        causal_parent_ids=parents,
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance=SOURCE,
    )


def test_events_are_content_addressed_immutable_and_round_trip_exactly() -> None:
    first = observation()
    same = observation()

    assert first.event_id == same.event_id
    assert first.sha256 == str(first.event_id).removeprefix("event:")
    assert event_from_payload(first.payload()) == first
    with pytest.raises(FrozenInstanceError):
        first.adapter_version = "changed"  # type: ignore[misc]
    decoded = first.sensor_scope.value()
    decoded["sensor"] = "mutated"
    assert first.sensor_scope.value()["sensor"] == "camera"


def test_direct_constructors_reject_mutable_tuple_aliases() -> None:
    observed = observation()
    inferred = hypothesis(observed)

    with pytest.raises(ValueError, match="causal_parent_ids must be an immutable tuple"):
        replace(observed.envelope, causal_parent_ids=[])
    with pytest.raises(ValueError, match="supersedes_event_ids must be an immutable tuple"):
        replace(observed.envelope, supersedes_event_ids=[])
    with pytest.raises(ValueError, match="raw packet or delta references must be an immutable tuple"):
        replace(observed, raw_packet_or_delta_refs=["packet:0"])
    with pytest.raises(ValueError, match="supporting_event_ids must be an immutable tuple"):
        replace(inferred, supporting_event_ids=[observed.event_id])


def test_plural_parent_order_is_canonicalized_without_changing_identity() -> None:
    first = observation(1)
    second = observation(2)

    left = hypothesis(first, second)
    right = hypothesis(second, first)

    assert left.event_id == right.event_id
    assert left.envelope.causal_parent_ids == tuple(sorted((first.event_id, second.event_id), key=str))
    assert left.envelope.producer_state_version == state_version_for_parents(
        (first.event_id, second.event_id)
    )


def test_derived_state_version_rejects_reuse_across_different_parent_sets() -> None:
    first = observation(1)
    second = observation(2)
    first_version = state_version_for_parents((first.event_id,))

    with pytest.raises(ValueError, match="complete causal-parent set"):
        HypothesisEvent.create(
            origin=HypothesisOrigin.EVENT_FORMER,
            epistemic_status=EpistemicStatus.INFERRED,
            referent_hypotheses={"candidate": 1.0},
            factor_change_distribution={},
            decision_relevance_distribution={},
            reducibility_distribution={},
            supporting_event_ids=(second.event_id,),
            calibrated_confidence=0.5,
            abstention_reason=None,
            predicted_value_of_further_computation=0.0,
            causal_parent_ids=(second.event_id,),
            clock_start_tick=1,
            clock_end_tick=1,
            source_and_provenance=SOURCE,
            producer_state_version=first_version,
        )


def test_observations_reject_future_and_evaluator_fields() -> None:
    with pytest.raises(ValueError, match="forbidden future or evaluator"):
        ObservationEvent.create(
            raw_packet_or_delta_refs=("packet:leaky",),
            adapter_version="adapter-v1",
            sensor_scope={"observed_outcome": "future"},
            transport_and_detection_cost=WorkVector(),
            clock_start_tick=0,
            clock_end_tick=0,
            source_and_provenance=SOURCE,
        )
    with pytest.raises(ValueError, match="forbidden future or evaluator"):
        observation_payload = ObservationEvent.create(
            raw_packet_or_delta_refs=("packet:hidden",),
            adapter_version="adapter-v1",
            sensor_scope={"sensor": "camera"},
            transport_and_detection_cost=WorkVector(),
            clock_start_tick=0,
            clock_end_tick=0,
            source_and_provenance={"ground_truth": "secret"},
        )
        assert observation_payload  # pragma: no cover


def test_simulation_requires_nonfactual_branch_and_cannot_launder_status() -> None:
    parent = observation()
    common = dict(
        origin=HypothesisOrigin.ACTOR,
        referent_hypotheses={"candidate": 1.0},
        factor_change_distribution={},
        decision_relevance_distribution={"relevant": 1.0},
        reducibility_distribution={"reducible": 1.0},
        supporting_event_ids=(parent.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=1.0,
        causal_parent_ids=(parent.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance=SOURCE,
    )
    with pytest.raises(ValueError, match="explicit counterfactual branch"):
        HypothesisEvent.create(epistemic_status=EpistemicStatus.SIMULATED, **common)
    with pytest.raises(ValueError, match="retain simulated epistemic status"):
        HypothesisEvent.create(
            epistemic_status=EpistemicStatus.INFERRED,
            counterfactual_branch_id=BranchRef("branch:counterfactual"),
            **common,
        )


def test_commitment_and_consequence_fields_are_temporally_separate() -> None:
    observed = observation()
    inferred = hypothesis(observed)
    commitment = CommitmentEvent.create(
        coalition_id="coalition:one",
        commitment_kind=CommitmentKind.EXTERNAL_ACTION,
        committed_payload={"action": "left"},
        decision_distribution={"left": 0.8, "right": 0.2},
        deadline_tick=5,
        predicted_utility_vector={"reward": 0.4},
        predicted_full_cost=WorkVector(actor_execution=3),
        causal_parent_ids=(inferred.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=SOURCE,
    )
    consequence = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={"position": 2},
        realized_utility_vector={"reward": 1.0},
        delayed_or_partial=False,
        observation_uncertainty=0.1,
        realized_full_cost=WorkVector(actor_execution=4),
        causal_parent_ids=(commitment.event_id,),
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance=SOURCE,
    )

    assert commitment.kind is EventKind.COMMITMENT
    assert consequence.kind is EventKind.CONSEQUENCE
    assert consequence.commitment_event_id == commitment.event_id
    assert "observed_outcome" not in commitment.body_payload()
    assert "predicted_utility_vector" not in consequence.body_payload()
    assert commitment.branch_id == consequence.branch_id == FACTUAL_BRANCH


def test_all_event_factories_forward_canonical_evidence_class() -> None:
    observed = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:oracle",),
        adapter_version="adapter-v1",
        sensor_scope={"sensor": "camera"},
        transport_and_detection_cost=WorkVector(),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    inferred = HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"candidate": 1.0},
        factor_change_distribution={},
        decision_relevance_distribution={},
        reducibility_distribution={},
        supporting_event_ids=(observed.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.0,
        causal_parent_ids=(observed.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    commitment = CommitmentEvent.create(
        coalition_id="coalition:oracle",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={"reason": "fixture"},
        decision_distribution={"abstain": 1.0},
        deadline_tick=3,
        predicted_utility_vector={"reward": 0.0},
        predicted_full_cost=WorkVector(),
        causal_parent_ids=(inferred.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    consequence = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={"abstained": True},
        realized_utility_vector={"reward": 0.0},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector(),
        causal_parent_ids=(commitment.event_id,),
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance=SOURCE,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )

    assert all(
        event.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE
        for event in (observed, inferred, commitment, consequence)
    )
    assert event_from_payload(consequence.payload()) == consequence


def test_event_parser_rejects_payload_drift_and_unknown_fields() -> None:
    payload = deepcopy(observation().payload())
    payload["body"]["adapter_version"] = "tampered"
    with pytest.raises(ValueError, match="payload digest mismatch"):
        event_from_payload(payload)

    payload = deepcopy(observation().payload())
    payload["body"]["future"] = "field"
    with pytest.raises(ValueError, match="fields mismatch"):
        event_from_payload(payload)
