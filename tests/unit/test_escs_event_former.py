from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from mop.escs.accounting import LifecycleLedger, WorkVector
from mop.escs.event_former import (
    ChargedEventFormer,
    EventFormerConfig,
    EventFormerContractError,
    EventFormerDecision,
    EventFormerDescriptor,
    EventFormerResult,
    EventProposal,
    EvidenceClass,
    RawPacket,
)
from mop.escs.events import EpistemicStatus, EventKind, ObservationEvent
from mop.escs.ledger import EventLedger
from mop.substrate.events import EventRef


def _packet(
    *,
    tick: int = 2,
    payload: bytes = b"camera-delta",
    evidence_class: EvidenceClass = EvidenceClass.LEARNED_UNVERIFIED,
) -> RawPacket:
    return RawPacket.create(
        sensor_id="sensor:camera/0",
        capture_start_tick=tick - 1,
        capture_end_tick=tick,
        arrival_tick=tick,
        payload_bytes=payload,
        sensor_scope={"modality": "vision", "channel": 0},
        source_and_provenance={"adapter": "fixture-v1"},
        transport_operations=len(payload),
        adaptation_operations=3,
        detection_operations=4,
        evidence_class=evidence_class,
    )


def _proposal(*, operations: int = 2) -> EventProposal:
    return EventProposal.create(
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"object:0": 0.7, "object:1": 0.3},
        factor_change_distribution={"position": 0.8, "appearance": 0.2},
        decision_relevance_distribution={"relevant": 0.6, "irrelevant": 0.4},
        reducibility_distribution={"reducible": 0.55, "irreducible": 0.45},
        calibrated_confidence=0.7,
        predicted_value_of_further_computation=0.25,
        formation_operations=operations,
    )


class _Policy:
    def __init__(
        self,
        decision: EventFormerDecision | Exception | object,
        *,
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        oracle: bool = False,
    ) -> None:
        self._decision = decision
        self._descriptor = EventFormerDescriptor(
            policy_id="event-former:fixture/v1",
            evidence_class=evidence_class,
            oracle_access=oracle,
        )
        self._state_version = hashlib.sha256(b"event-former-fixture-state").hexdigest()
        if isinstance(decision, EventFormerDecision):
            self._retained_state_bytes = decision.retained_state_bytes
        else:
            self._retained_state_bytes = 0

    @property
    def descriptor(self) -> EventFormerDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return self._state_version

    @property
    def retained_state_bytes(self) -> int:
        return self._retained_state_bytes

    def evaluate(self, packet: RawPacket) -> object:
        assert isinstance(packet, RawPacket)
        if isinstance(self._decision, Exception):
            raise self._decision
        return self._decision


def _admit_decision(
    *,
    proposals: tuple[EventProposal, ...] | None = None,
    evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
) -> EventFormerDecision:
    rows = proposals if proposals is not None else (_proposal(),)
    return EventFormerDecision(
        proposals=tuple(sorted(rows, key=lambda row: row.proposal_id)),
        discarded_candidates=1,
        policy_operations=3,
        retained_state_bytes=10,
        idle_operations=0,
        abstention_reason=None,
        evidence_class=evidence_class,
    )


def _abstain_decision() -> EventFormerDecision:
    return EventFormerDecision(
        proposals=(),
        discarded_candidates=0,
        policy_operations=1,
        retained_state_bytes=5,
        idle_operations=3,
        abstention_reason="no-calibrated-value",
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )


def _former(
    policy: _Policy,
    *,
    config: EventFormerConfig | None = None,
    events: EventLedger | None = None,
) -> tuple[ChargedEventFormer, EventLedger, LifecycleLedger]:
    events = events or EventLedger()
    lifecycle = LifecycleLedger()
    former = ChargedEventFormer(
        policy=policy,
        config=config or EventFormerConfig(),
        event_ledger=events,
        lifecycle_ledger=lifecycle,
    )
    return former, events, lifecycle


def test_raw_packet_identity_and_hidden_truth_guards_are_recursive() -> None:
    packet = _packet()
    assert packet.packet_id.startswith("packet:")
    assert packet.payload_sha256
    assert _packet().packet_id == packet.packet_id

    with pytest.raises(ValueError, match="evaluator-only"):
        RawPacket.create(
            sensor_id="sensor:camera/0",
            capture_start_tick=0,
            capture_end_tick=0,
            arrival_tick=0,
            payload_bytes=b"raw",
            sensor_scope={"nested": [{"ground_truth": "object:7"}]},
            source_and_provenance={"adapter": "fixture"},
            transport_operations=1,
            adaptation_operations=1,
            detection_operations=1,
        )

    with pytest.raises(ValueError, match="evaluator-only"):
        _packet(payload=b'{"metadata":{"hidden_change_point":4}}')


def test_event_proposal_is_content_addressed_and_cannot_simulate_from_raw_input() -> None:
    proposal = _proposal()
    assert proposal.proposal_id == _proposal().proposal_id
    with pytest.raises(ValueError, match="cannot emit simulated"):
        EventProposal.create(
            epistemic_status=EpistemicStatus.SIMULATED,
            referent_hypotheses={},
            factor_change_distribution={},
            decision_relevance_distribution={},
            reducibility_distribution={},
            calibrated_confidence=0.5,
            predicted_value_of_further_computation=0.0,
            formation_operations=1,
        )


def test_admitted_packet_publishes_exact_four_stage_inputs_and_charges_work() -> None:
    former, events, lifecycle = _former(_Policy(_admit_decision()))

    result = former.process(_packet())

    assert result.admitted is True
    assert result.observation_event_id is not None
    assert len(result.hypothesis_event_ids) == 1
    assert [event.kind for event in events.events] == [EventKind.OBSERVATION, EventKind.HYPOTHESIS]
    assert events.verify() == []
    assert lifecycle.total.raw_transport_and_adapters == len(b"camera-delta") + 7
    assert lifecycle.total.event_formation == 7
    assert lifecycle.total.indexing_and_graph_maintenance == 3
    assert lifecycle.verify(event_ids=events.event_ids) == []

    former.finalize(end_tick=5)
    assert lifecycle.total.retained_byte_time == 30
    assert former.retained_state_bytes == 0
    assert former.finalized is True
    with pytest.raises(EventFormerContractError, match="finalized"):
        former.process(_packet(tick=6))


def test_abstention_charges_idle_and_retention_without_fabricating_an_event() -> None:
    former, events, lifecycle = _former(_Policy(_abstain_decision()))

    result = former.process(_packet(tick=3))

    assert result.admitted is False
    assert events.events == ()
    assert lifecycle.total.event_formation == 2
    assert lifecycle.total.idle_floor == 3
    former.finalize(end_tick=5)
    assert lifecycle.total.retained_byte_time == 10


def test_oracle_policy_is_explicitly_nonpromotable_and_refused_by_promotable_mode() -> None:
    oracle_decision = replace(_admit_decision(), evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE)
    oracle = _Policy(
        oracle_decision,
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
        oracle=True,
    )
    former, _, _ = _former(oracle)
    assert former.process(_packet()).evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE

    with pytest.raises(ValueError, match="only learned-unverified"):
        _former(oracle, config=EventFormerConfig(promotable_mode=True))


def test_cap_failure_leaves_event_ledger_unchanged_but_charges_attempted_work() -> None:
    decision = _admit_decision(proposals=(_proposal(operations=1), _proposal(operations=2)))
    former, events, lifecycle = _former(
        _Policy(decision),
        config=EventFormerConfig(max_hypothesis_fanout=1),
    )

    with pytest.raises(EventFormerContractError, match="hypothesis-fanout cap"):
        former.process(_packet())

    assert events.events == ()
    assert lifecycle.total.raw_transport_and_adapters == len(b"camera-delta") + 7
    assert lifecycle.total.event_formation > 0


def test_policy_exception_and_invalid_result_fail_closed_after_minimum_charge() -> None:
    former, events, lifecycle = _former(
        _Policy(RuntimeError("policy failed")),
        config=EventFormerConfig(minimum_policy_operations=2),
    )
    with pytest.raises(RuntimeError, match="policy failed"):
        former.process(_packet())
    assert events.events == ()
    assert lifecycle.total.event_formation >= 2

    invalid, invalid_events, invalid_lifecycle = _former(_Policy(object()))
    with pytest.raises(EventFormerContractError, match="invalid EventFormerDecision"):
        invalid.process(_packet())
    assert invalid_events.events == ()
    assert invalid_lifecycle.total.event_formation > 0


def test_invalid_parent_preflight_does_not_partially_publish_events() -> None:
    former, events, lifecycle = _former(_Policy(_admit_decision()))

    with pytest.raises(ValueError, match="missing causal parent"):
        former.process(_packet(), observation_parent_ids=(EventRef("event:missing"),))

    assert events.events == ()
    assert lifecycle.total.total_work > 0


def test_oversized_packet_rejection_is_charged_before_fail_closed() -> None:
    packet = _packet(payload=b"12345678")
    former, events, lifecycle = _former(
        _Policy(_admit_decision()),
        config=EventFormerConfig(max_packet_bytes=4),
    )

    with pytest.raises(EventFormerContractError, match="payload-byte cap"):
        former.process(packet)

    assert events.events == ()
    assert lifecycle.total.raw_transport_and_adapters == 15
    assert lifecycle.total.event_formation == 1


def test_no_packet_poll_has_bounded_nonzero_cost_and_no_event() -> None:
    former, events, lifecycle = _former(_Policy(_abstain_decision()))

    first = former.poll(tick=2, polling_operations=2)
    second = former.poll(tick=4, polling_operations=3, deadline_operations=4)

    assert first == (0, 1)
    assert second == (1, 3)
    assert events.events == ()
    assert lifecycle.total.raw_transport_and_adapters == 5
    assert lifecycle.total.idle_floor == 5
    assert lifecycle.total.retained_byte_time == 10
    former.finalize(end_tick=5)
    assert lifecycle.total.retained_byte_time == 15
    assert former.retained_state_bytes == 0


class _NoHistoryIteration(list):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("history was iterated")


def test_poll_sequence_accounting_does_not_scan_lifecycle_history() -> None:
    former, _, lifecycle = _former(_Policy(_abstain_decision()))
    lifecycle._entries = _NoHistoryIteration(lifecycle._entries)  # type: ignore[attr-defined]

    assert former.poll(tick=1, polling_operations=1) == (0, 1)
    assert lifecycle.entry_count == 1


def test_oracle_packet_cannot_be_laundered_through_learned_promotable_policy() -> None:
    learned_decision = _admit_decision(evidence_class=EvidenceClass.LEARNED_UNVERIFIED)
    policy = _Policy(learned_decision, evidence_class=EvidenceClass.LEARNED_UNVERIFIED)
    former, events, lifecycle = _former(
        policy,
        config=EventFormerConfig(promotable_mode=True),
    )

    with pytest.raises(EventFormerContractError, match="promotable boundary"):
        former.process(_packet(evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE))

    assert events.events == ()
    assert lifecycle.total.total_work > 0

    with pytest.raises(ValueError, match="non-learned packet"):
        EventFormerConfig(promotable_mode=True).validate_packet(
            _packet(evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE)
        )


def test_oracle_parent_taint_is_propagated_to_observation_and_hypothesis() -> None:
    events = EventLedger()
    parent = ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:{'0' * 64}",),
        adapter_version="fixture-v1",
        sensor_scope={"modality": "vision"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"fixture": True},
        measured_creation_cost=WorkVector(event_formation=1),
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    events.append(parent)
    learned_decision = _admit_decision(evidence_class=EvidenceClass.LEARNED_UNVERIFIED)
    former, events, _ = _former(
        _Policy(learned_decision, evidence_class=EvidenceClass.LEARNED_UNVERIFIED),
        events=events,
    )

    result = former.process(_packet(), observation_parent_ids=(parent.event_id,))

    assert result.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE
    assert all(event.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE for event in events.events[1:])


def test_event_former_contract_caps_mutable_and_unbounded_outputs() -> None:
    with pytest.raises(ValueError, match="immutable tuple"):
        EventFormerDecision(
            proposals=[_proposal()],  # type: ignore[arg-type]
            discarded_candidates=0,
            policy_operations=2,
            retained_state_bytes=0,
            idle_operations=0,
            abstention_reason=None,
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
    with pytest.raises(ValueError, match="positive integer"):
        replace(_proposal(), formation_operations=1.5)
    with pytest.raises(ValueError, match="boolean"):
        EventFormerDescriptor(
            policy_id="event-former:fixture/v1",
            evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
            oracle_access=1,  # type: ignore[arg-type]
        )

    idle_decision = replace(_abstain_decision(), idle_operations=4)
    idle_former, _, _ = _former(
        _Policy(idle_decision),
        config=EventFormerConfig(max_idle_operations=3),
    )
    with pytest.raises(EventFormerContractError, match="idle-operation cap"):
        idle_former.process(_packet())

    reason_decision = replace(_abstain_decision(), abstention_reason="ééé")
    reason_former, _, _ = _former(
        _Policy(reason_decision),
        config=EventFormerConfig(max_abstention_reason_bytes=5),
    )
    with pytest.raises(EventFormerContractError, match="reason byte cap"):
        reason_former.process(_packet())

    parent_former, _, _ = _former(
        _Policy(_admit_decision()),
        config=EventFormerConfig(max_observation_parents=1),
    )
    parents = (EventRef("event:a"), EventRef("event:b"))
    with pytest.raises(EventFormerContractError, match="parent count"):
        parent_former.process(_packet(), observation_parent_ids=parents)

    poll_former, _, poll_lifecycle = _former(
        _Policy(_abstain_decision()),
        config=EventFormerConfig(max_poll_operations=3),
    )
    with pytest.raises(EventFormerContractError, match="poll work"):
        poll_former.poll(tick=1, polling_operations=4, deadline_operations=0)
    assert poll_lifecycle.total.raw_transport_and_adapters == 3
    assert poll_lifecycle.total.idle_floor == 1


class _FailSecondAppendLedger(EventLedger):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def append(self, event):  # type: ignore[no-untyped-def]
        self.attempts += 1
        if self.attempts == 2:
            raise RuntimeError("injected second append failure")
        return super().append(event)


def test_publication_batch_failure_rolls_back_all_events_but_keeps_attempt_charge() -> None:
    events = _FailSecondAppendLedger()
    former, _, lifecycle = _former(_Policy(_admit_decision()), events=events)

    with pytest.raises(RuntimeError, match="injected second append"):
        former.process(_packet())

    assert events.events == ()
    assert events.verify() == []
    assert lifecycle.total.indexing_and_graph_maintenance == 3


class _MutatingPolicy(_Policy):
    def evaluate(self, packet: RawPacket) -> object:
        self._state_version = hashlib.sha256(b"mutated-state").hexdigest()
        return super().evaluate(packet)


def test_policy_state_mutation_poisons_and_releases_the_former() -> None:
    former, events, lifecycle = _former(_MutatingPolicy(_admit_decision()))

    with pytest.raises(EventFormerContractError, match="mutated"):
        former.process(_packet())

    assert events.events == ()
    assert lifecycle.total.event_formation > 0
    assert former.retained_state_bytes == 0
    assert former.finalized is True


def test_out_of_band_policy_state_bytes_and_descriptor_drift_fail_closed() -> None:
    state_policy = _Policy(_admit_decision())
    state_former, state_events, _ = _former(state_policy)
    state_policy._state_version = hashlib.sha256(b"out-of-band").hexdigest()
    with pytest.raises(EventFormerContractError, match="outside its transaction"):
        state_former.process(_packet())
    assert state_events.events == ()
    assert state_former.finalized is True

    bytes_policy = _Policy(_abstain_decision())
    bytes_former, _, _ = _former(bytes_policy)
    bytes_policy._retained_state_bytes += 1
    with pytest.raises(EventFormerContractError, match="outside its transaction"):
        bytes_former.poll(tick=1, polling_operations=1)
    assert bytes_former.retained_state_bytes == 0

    descriptor_policy = _Policy(_abstain_decision())
    descriptor_former, _, _ = _former(descriptor_policy)
    descriptor_policy._descriptor = EventFormerDescriptor(
        policy_id="event-former:changed/v1",
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )
    with pytest.raises(EventFormerContractError, match="outside its transaction"):
        descriptor_former.finalize(end_tick=1)


class _NoInputIteration(list):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("input sequence was iterated")


class _NoTupleIteration(tuple):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("oversized tuple was iterated")


def test_parent_boundary_caps_before_iteration_and_charges_lookup_attempts() -> None:
    mutable_former, _, mutable_lifecycle = _former(_Policy(_admit_decision()))
    with pytest.raises(EventFormerContractError, match="immutable tuple"):
        mutable_former.process(
            _packet(),
            observation_parent_ids=_NoInputIteration([EventRef("event:a")]),
        )
    assert mutable_lifecycle.total.indexing_and_graph_maintenance == 1

    oversized_former, _, oversized_lifecycle = _former(
        _Policy(_admit_decision()),
        config=EventFormerConfig(max_observation_parents=1),
    )
    oversized = _NoTupleIteration((EventRef("event:a"), EventRef("event:b")))
    with pytest.raises(EventFormerContractError, match="parent count"):
        oversized_former.process(_packet(), observation_parent_ids=oversized)
    assert oversized_lifecycle.total.indexing_and_graph_maintenance == 2

    missing_former, _, missing_lifecycle = _former(_Policy(_admit_decision()))
    with pytest.raises(ValueError, match="missing causal parent"):
        missing_former.process(_packet(), observation_parent_ids=(EventRef("event:missing"),))
    assert missing_lifecycle.total.indexing_and_graph_maintenance == 2


def test_result_ids_are_content_addressed_unique_and_canonical() -> None:
    packet_id = f"packet:{'0' * 64}"
    event_a = f"event:{'1' * 64}"
    event_b = f"event:{'2' * 64}"
    event_c = f"event:{'3' * 64}"

    with pytest.raises(ValueError, match="packet namespace"):
        EventFormerResult(
            packet_id="bad",
            observation_event_id=None,
            hypothesis_event_ids=(),
            evidence_class=EvidenceClass.LEARNED_UNVERIFIED,
            admitted=False,
            lifecycle_start_sequence=0,
            lifecycle_end_sequence=0,
            retained_state_bytes=0,
        )
    with pytest.raises(ValueError, match="canonical order"):
        EventFormerResult(
            packet_id=packet_id,
            observation_event_id=event_c,
            hypothesis_event_ids=(event_b, event_a),
            evidence_class=EvidenceClass.LEARNED_UNVERIFIED,
            admitted=True,
            lifecycle_start_sequence=0,
            lifecycle_end_sequence=1,
            retained_state_bytes=0,
        )


def test_backward_time_attempts_are_rejected_with_bounded_charges() -> None:
    former, _, lifecycle = _former(_Policy(_abstain_decision()))
    former.poll(tick=2, polling_operations=1)
    before = lifecycle.entry_count

    with pytest.raises(EventFormerContractError, match="backwards"):
        former.poll(tick=1, polling_operations=1)

    assert lifecycle.entry_count == before + 1
    assert lifecycle.entries[-1].reason == "event-former-poll-time-order-rejection"


def test_explicit_deployment_start_charges_pre_first_packet_residency() -> None:
    events = EventLedger()
    lifecycle = LifecycleLedger()
    former = ChargedEventFormer(
        policy=_Policy(_abstain_decision()),
        config=EventFormerConfig(),
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        deployment_start_tick=1,
    )

    former.process(_packet(tick=3))

    assert lifecycle.total.retained_byte_time == 10
