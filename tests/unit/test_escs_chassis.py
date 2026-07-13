from __future__ import annotations

import copy
import hashlib

import pytest

from mop.escs.accounting import LifecycleLedger, WorkVector
from mop.escs.actors import (
    ActionIntent,
    ActorActivationContext,
    ActorActivationResult,
    ActorDescriptor,
    ActorUpdateContext,
    ActorUpdatePlan,
    ReadinessEstimate,
)
from mop.escs.chassis import (
    CHASSIS_COMMITMENT_SCHEMA,
    CHASSIS_CONSEQUENCE_SCHEMA,
    EFFECT_AUTHORITY_SCHEMA,
    ChassisContractError,
    ChassisFailpoint,
    ChassisStatus,
    EffectOutcome,
    EffectRequest,
    EventSourcedCoalitionChassis,
    InjectedChassisFailure,
)
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
from mop.escs.messages import SchemaRegistry
from mop.escs.runtime import (
    CandidateMode,
    CoalitionRuntime,
    RuntimeCaps,
    RuntimeConfig,
    ScriptedDispatchPolicy,
)
from mop.substrate.events import BranchRef, EventRef, canonical_bytes, canonical_sha256


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class ChassisActor:
    def __init__(
        self,
        *,
        event_ledger: EventLedger,
        order: list[str],
        emit_action: bool = True,
        counter: int = 0,
    ) -> None:
        self._event_ledger = event_ledger
        self._order = order
        self._emit_action = emit_action
        self._counter = counter
        self._descriptor = ActorDescriptor(
            actor_id="actor:chassis",
            subscribed_event_types=("hypothesis",),
        )
        self.stage_calls = 0
        self.activation_payloads: list[bytes] = []

    @property
    def descriptor(self) -> ActorDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return digest(f"chassis-actor:{self._counter}")

    @property
    def retained_state_bytes(self) -> int:
        return 32

    def readiness(self, header):  # type: ignore[no-untyped-def]
        return ReadinessEstimate(
            actor_id=self.descriptor.actor_id,
            state_version=self.state_version,
            compatible=True,
            expected_decision_value=1.0,
            predicted_operations=4,
            predicted_message_bytes=0,
            estimation_operations=1,
        )

    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        self.activation_payloads.append(context.event_payload_bytes)
        actions = ()
        if self._emit_action:
            actions = (
                ActionIntent.create(
                    source_event_id=context.event_header.event_id,
                    branch_id=context.event_header.branch_id,
                    referent_hypotheses=context.event_header.referent_hypotheses,
                    epistemic_status=context.event_header.epistemic_status,
                    evidence_class=context.event_header.evidence_class,
                    producer_actor_id=self.descriptor.actor_id,
                    producer_state_version=self.state_version,
                    created_tick=context.event_header.created_tick,
                    expiry_tick=context.event_header.expiry_tick,
                    producer_operations=4,
                    payload_form="motor-program",
                    payload_bytes=b"turn-left",
                ),
            )
        return ActorActivationResult(action_intents=actions, executed_operations=5)

    def stage_update(self, context: ActorUpdateContext) -> ActorUpdatePlan:
        consequence = self._event_ledger.get(EventRef(context.consequence_event_id))
        assert isinstance(consequence, ConsequenceEvent)
        self._order.append("update")
        self.stage_calls += 1
        replacement = copy.copy(self)
        replacement._counter += 1
        return ActorUpdatePlan(
            actor_id=self.descriptor.actor_id,
            prior_state_version=self.state_version,
            next_state_version=replacement.state_version,
            idempotency_key=context.idempotency_key,
            executed_operations=3,
            replacement_actor=replacement,
        )


class RecordingEffect:
    def __init__(
        self,
        *,
        event_ledger: EventLedger,
        order: list[str],
        fail: bool = False,
    ) -> None:
        self._event_ledger = event_ledger
        self._order = order
        self._fail = fail
        self.requests: list[EffectRequest] = []

    def execute(self, request: EffectRequest) -> EffectOutcome:
        commitment = self._event_ledger.get(EventRef(request.commitment_event_id))
        assert isinstance(commitment, CommitmentEvent)
        self._order.append("effect")
        self.requests.append(request)
        if self._fail:
            raise RuntimeError("scripted effect failure")
        return EffectOutcome.create(
            observed_outcome={"position": "left"},
            realized_utility_vector={"reward": 1.0},
            realized_full_cost=WorkVector(actor_execution=2),
        )


def make_hypothesis(
    ledger: EventLedger,
    *,
    evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
    branch: BranchRef = BranchRef("branch:factual"),
    status: EpistemicStatus = EpistemicStatus.INFERRED,
) -> HypothesisEvent:
    observation = ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:{branch.value.split(':')[-1]}",),
        adapter_version="fixture-v1",
        sensor_scope={"sensor": "fixture"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"source": "sensor:fixture"},
        evidence_class=evidence_class,
    )
    hypothesis = HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=status,
        referent_hypotheses={"referent:item/1": 1.0},
        factor_change_distribution={"factor:motion": 1.0},
        decision_relevance_distribution={"relevant": 1.0},
        reducibility_distribution={"reducible": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.8,
        abstention_reason=None,
        predicted_value_of_further_computation=1.0,
        causal_parent_ids=(observation.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={"producer": "fixture:event-former"},
        evidence_class=evidence_class,
    )
    ledger.append_batch((observation, hypothesis))
    return hypothesis


def make_system(
    *,
    evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
    emit_action: bool = True,
    branch: BranchRef = BranchRef("branch:factual"),
    status: EpistemicStatus = EpistemicStatus.INFERRED,
):
    events = EventLedger()
    lifecycle = LifecycleLedger()
    hypothesis = make_hypothesis(
        events,
        evidence_class=evidence_class,
        branch=branch,
        status=status,
    )
    order: list[str] = []
    actor = ChassisActor(event_ledger=events, order=order, emit_action=emit_action)
    policy = ScriptedDispatchPolicy({}, default=(actor.descriptor.actor_id,))
    runtime = CoalitionRuntime(
        actors=(actor,),
        policy=policy,
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0, A=1),
        ),
        ledger=lifecycle,
        event_ledger=events,
        clock_ns=lambda: 1,
    )
    chassis = EventSourcedCoalitionChassis(
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        runtime=runtime,
    )
    effect = RecordingEffect(event_ledger=events, order=order)
    return events, lifecycle, hypothesis, actor, runtime, chassis, effect, order


def chassis_commitments(events: EventLedger) -> tuple[CommitmentEvent, ...]:
    return tuple(
        event
        for event in events.events
        if isinstance(event, CommitmentEvent)
        and isinstance(event.committed_payload.value(), dict)
        and event.committed_payload.value().get("schema") == CHASSIS_COMMITMENT_SCHEMA
    )


def test_dispatch_is_derived_only_from_ledger_hypothesis_and_uses_canonical_body() -> None:
    events, _, hypothesis, actor, _, chassis, _, _ = make_system()

    dispatch = chassis.dispatch_from_hypothesis(hypothesis.event_id)

    assert dispatch.header.event_id == str(hypothesis.event_id)
    assert dispatch.header.branch_id == str(hypothesis.branch_id)
    assert dispatch.header.evidence_class is hypothesis.evidence_class
    assert dispatch.header.producer_state_version == hypothesis.envelope.producer_state_version
    assert dispatch.header.source_event_ids == tuple(
        str(row) for row in hypothesis.envelope.causal_parent_ids
    )
    assert dispatch.payload_bytes == canonical_bytes(hypothesis.body_payload())
    assert dispatch.header.payload_digest == hypothesis.envelope.payload_digest
    with pytest.raises(ValueError, match="unknown event"):
        chassis.dispatch_from_hypothesis(EventRef("event:unknown"))
    assert actor.activation_payloads == []
    assert events.verify() == []


def test_commitment_precedes_effect_and_consequence_precedes_atomic_actor_update() -> None:
    events, lifecycle, hypothesis, actor, runtime, chassis, effect, order = make_system()

    result = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert result.status is ChassisStatus.COMPLETED
    assert order == ["effect", "update"]
    assert result.effect_invoked and result.updated_actor_ids == (actor.descriptor.actor_id,)
    commitment = events.get(EventRef(result.commitment_event_id))
    consequence = events.get(EventRef(result.consequence_event_id))
    assert isinstance(commitment, CommitmentEvent)
    assert isinstance(consequence, ConsequenceEvent)
    assert events.events.index(commitment) < events.events.index(consequence)
    committed = commitment.committed_payload.value()
    assert committed["action_record"]["action_id"] == result.action_id
    assert committed["trace_authority_id"] == result.trace_authority_id
    assert committed["full_trace_sha256"] == result.full_trace_sha256
    assert effect.requests[0].effect_id == result.effect_id
    assert runtime.actor_state_versions[actor.descriptor.actor_id] != actor.state_version
    assert result.lifecycle_start_sequence < result.lifecycle_end_sequence == len(lifecycle.entries)
    reasons = {row.reason for row in lifecycle.entries[result.lifecycle_start_sequence :]}
    assert {
        "chassis-ledger-to-dispatch-adaptation",
        "chassis-full-trace-binding",
        "chassis-commitment-formation-and-indexing",
        "chassis-effect-attempt",
        "chassis-effect-realized-cost",
        "chassis-consequence-formation-and-indexing",
    } <= reasons


@pytest.mark.parametrize(
    ("failpoint", "commitments", "consequences", "effect_calls", "updates"),
    (
        (ChassisFailpoint.AFTER_DISPATCH, 0, 0, 0, 0),
        (ChassisFailpoint.AFTER_COMMITMENT, 1, 0, 0, 0),
        (ChassisFailpoint.AFTER_EFFECT, 1, 0, 1, 0),
        (ChassisFailpoint.AFTER_CONSEQUENCE, 1, 1, 1, 0),
    ),
)
def test_failpoints_preserve_declared_publication_order(
    failpoint: ChassisFailpoint,
    commitments: int,
    consequences: int,
    effect_calls: int,
    updates: int,
) -> None:
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system()

    with pytest.raises(InjectedChassisFailure, match=failpoint.value):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect, failpoint=failpoint)

    rows = chassis_commitments(events)
    assert len(rows) == commitments
    assert sum(len(events.consequences_for(row.event_id)) for row in rows) == consequences
    assert len(effect.requests) == effect_calls
    assert actor.stage_calls == updates


def test_commitment_crash_window_is_at_most_once_but_can_omit_external_effect() -> None:
    events, _, hypothesis, _, _, chassis, effect, _ = make_system()
    with pytest.raises(InjectedChassisFailure):
        chassis.execute_hypothesis(
            hypothesis.event_id,
            effect=effect,
            failpoint=ChassisFailpoint.AFTER_COMMITMENT,
        )

    resumed = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert resumed.status is ChassisStatus.COMMITTED_EFFECT_NOT_REPLAYED
    assert resumed.resumed_from_ledger and not resumed.effect_invoked
    assert effect.requests == []
    assert len(chassis_commitments(events)) == 1
    assert not events.consequences_for(EventRef(resumed.commitment_event_id))


def test_effect_failure_is_charged_and_never_reinvoked_from_existing_commitment() -> None:
    events, lifecycle, hypothesis, _, _, chassis, _, order = make_system()
    failing = RecordingEffect(event_ledger=events, order=order, fail=True)

    failed = chassis.execute_hypothesis(hypothesis.event_id, effect=failing)
    resumed = chassis.execute_hypothesis(hypothesis.event_id, effect=failing)

    assert failed.status is ChassisStatus.EFFECT_FAILED and failed.effect_invoked
    assert resumed.status is ChassisStatus.COMMITTED_EFFECT_NOT_REPLAYED
    assert len(failing.requests) == 1
    reasons = [row.reason for row in lifecycle.entries]
    assert reasons.count("chassis-effect-attempt") == 1
    assert reasons.count("chassis-effect-exception") == 1


def test_recorded_consequence_resumes_update_without_reinvoking_effect() -> None:
    _, _, hypothesis, actor, _, chassis, effect, _ = make_system()
    with pytest.raises(InjectedChassisFailure):
        chassis.execute_hypothesis(
            hypothesis.event_id,
            effect=effect,
            failpoint=ChassisFailpoint.AFTER_CONSEQUENCE,
        )

    resumed = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert resumed.status is ChassisStatus.COMPLETED
    assert resumed.updated_actor_ids == (actor.descriptor.actor_id,)
    assert not resumed.effect_invoked and len(effect.requests) == 1
    assert actor.stage_calls == 1


def test_duplicate_and_restart_do_not_duplicate_effect_commitment_or_consequence() -> None:
    events, lifecycle, hypothesis, actor, runtime, chassis, effect, _ = make_system()
    completed = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    duplicate = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)
    restarted = EventSourcedCoalitionChassis(
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        runtime=runtime,
    ).execute_hypothesis(hypothesis.event_id, effect=effect)

    assert duplicate.status is ChassisStatus.ALREADY_COMPLETED
    assert restarted.status is ChassisStatus.CONSEQUENCE_RECORDED_UPDATE_UNAVAILABLE
    assert len(effect.requests) == 1 and actor.stage_calls == 1
    assert len(chassis_commitments(events)) == 1
    assert len(events.consequences_for(EventRef(completed.commitment_event_id))) == 1


def test_oracle_action_is_persistently_abstained_and_never_reaches_effect() -> None:
    events, _, hypothesis, _, _, chassis, effect, _ = make_system(
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE
    )

    result = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert result.status is ChassisStatus.ABSTAINED
    assert not result.effect_invoked and not effect.requests
    commitment = events.get(EventRef(result.commitment_event_id))
    assert isinstance(commitment, CommitmentEvent)
    assert commitment.commitment_kind.value == "abstention"
    assert commitment.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE
    assert commitment.committed_payload.value()["blocked_action_id"] is not None


def test_chassis_refuses_a_different_lifecycle_ledger_than_runtime() -> None:
    events, _, _, _, runtime, _, _, _ = make_system()
    with pytest.raises(ValueError, match="share one authoritative LifecycleLedger"):
        EventSourcedCoalitionChassis(
            event_ledger=events,
            lifecycle_ledger=LifecycleLedger(),
            runtime=runtime,
        )


def test_chassis_refuses_a_different_event_ledger_than_runtime() -> None:
    _, lifecycle, _, _, runtime, _, _, _ = make_system()
    with pytest.raises(ValueError, match="share one authoritative EventLedger"):
        EventSourcedCoalitionChassis(
            event_ledger=EventLedger(),
            lifecycle_ledger=lifecycle,
            runtime=runtime,
        )


def _append_distinct_hypothesis(events: EventLedger) -> HypothesisEvent:
    observation = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:foreign-parent",),
        adapter_version="fixture-v1",
        sensor_scope={"sensor": "foreign"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"source": "sensor:foreign"},
    )
    hypothesis = HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=EpistemicStatus.INFERRED,
        referent_hypotheses={"referent:foreign": 1.0},
        factor_change_distribution={"factor:foreign": 1.0},
        decision_relevance_distribution={"relevant": 1.0},
        reducibility_distribution={"reducible": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.0,
        causal_parent_ids=(observation.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={"producer": "fixture:event-former"},
    )
    events.append_batch((observation, hypothesis))
    return hypothesis


def test_foreign_parent_commitment_cannot_poison_restart_fence() -> None:
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system()
    foreign = _append_distinct_hypothesis(events)
    runtime_id = digest("forged-runtime")
    authority_sequence = 0
    trace_id = canonical_sha256(
        {
            "schema": "mop-escs-runtime-trace-authority/v1",
            "runtime_id": runtime_id,
            "authority_sequence": authority_sequence,
        }
    )
    forged = CommitmentEvent.create(
        coalition_id=f"coalition:{trace_id}",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={
            "schema": CHASSIS_COMMITMENT_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "runtime_id": runtime_id,
            "trace_authority_sequence": authority_sequence,
            "trace_authority_id": trace_id,
            "full_trace_sha256": digest("forged-full-trace"),
            "effect_id": digest("forged-effect"),
            "decision_reason": "runtime-produced-no-factual-action",
            "action_record": None,
            "blocked_action_id": None,
        },
        decision_distribution={"abstention": 1.0},
        deadline_tick=2,
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(foreign.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": trace_id,
        },
    )
    events.append(forged)

    result = chassis.execute_hypothesis(hypothesis.event_id, effect=effect, now_tick=1)

    assert result.status is ChassisStatus.COMPLETED
    assert len(effect.requests) == 1
    assert len(actor.activation_payloads) == 1


def test_foreign_runtime_commitment_cannot_poison_restart_fence() -> None:
    events, lifecycle, hypothesis, actor, runtime, chassis, effect, _ = make_system()
    runtime_id = digest("foreign-runtime-authority")
    authority_sequence = 0
    trace_id = canonical_sha256(
        {
            "schema": "mop-escs-runtime-trace-authority/v1",
            "runtime_id": runtime_id,
            "authority_sequence": authority_sequence,
        }
    )
    effect_id = canonical_sha256(
        {
            "schema": EFFECT_AUTHORITY_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "action_id": None,
            "trace_authority_id": trace_id,
        }
    )
    forged = CommitmentEvent.create(
        coalition_id=f"coalition:{trace_id}",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={
            "schema": CHASSIS_COMMITMENT_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "runtime_id": runtime_id,
            "trace_authority_sequence": authority_sequence,
            "trace_authority_id": trace_id,
            "full_trace_sha256": digest("foreign-full-trace"),
            "effect_id": effect_id,
            "decision_reason": "runtime-produced-no-factual-action",
            "action_record": None,
            "blocked_action_id": None,
        },
        decision_distribution={"abstention": 1.0},
        deadline_tick=2,
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(hypothesis.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": trace_id,
        },
        evidence_class=hypothesis.evidence_class,
    )
    events.append(forged)
    before = lifecycle.entry_count

    with pytest.raises(ChassisContractError, match="different runtime authority"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect, now_tick=2)

    assert runtime.runtime_id != runtime_id
    assert lifecycle.entry_count == before
    assert actor.stage_calls == 0
    assert effect.requests == []


def test_mismatched_consequence_authority_cannot_update_actor() -> None:
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system()
    with pytest.raises(InjectedChassisFailure):
        chassis.execute_hypothesis(
            hypothesis.event_id,
            effect=effect,
            failpoint=ChassisFailpoint.AFTER_COMMITMENT,
        )
    commitment = chassis_commitments(events)[0]
    committed = commitment.committed_payload.value()
    action_id = committed["action_record"]["action_id"]
    forged = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome={
            "schema": CHASSIS_CONSEQUENCE_SCHEMA,
            "effect_id": digest("wrong-effect"),
            "hypothesis_event_id": committed["hypothesis_event_id"],
            "trace_authority_id": committed["trace_authority_id"],
            "full_trace_sha256": committed["full_trace_sha256"],
            "action_id": action_id,
            "outcome": {"forged": True},
        },
        realized_utility_vector={"reward": 1.0},
        delayed_or_partial=False,
        observation_uncertainty=0.0,
        realized_full_cost=WorkVector.zero(),
        causal_parent_ids=(commitment.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance={
            "producer": "escs.chassis",
            "effect_id": digest("wrong-effect"),
        },
        evidence_class=commitment.evidence_class,
    )
    events.append(forged)

    with pytest.raises(ChassisContractError, match="authority does not match"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect, now_tick=2)

    assert actor.stage_calls == 0
    assert effect.requests == []


def test_replayed_ledger_preserves_effect_fence_but_not_runtime_pending_update() -> None:
    events, _, hypothesis, _, original_runtime, chassis, effect, _ = make_system()
    completed = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)
    replayed = EventLedger.replay(events.payload())
    lifecycle = LifecycleLedger()
    order: list[str] = []
    actor = ChassisActor(event_ledger=replayed, order=order)
    runtime = CoalitionRuntime(
        actors=(actor,),
        policy=ScriptedDispatchPolicy({}, default=(actor.descriptor.actor_id,)),
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0, A=1),
        ),
        ledger=lifecycle,
        event_ledger=replayed,
        clock_ns=lambda: 1,
    )
    restarted = EventSourcedCoalitionChassis(
        event_ledger=replayed,
        lifecycle_ledger=lifecycle,
        runtime=runtime,
        trusted_replay_runtime_ids=(original_runtime.runtime_id,),
    )
    replay_effect = RecordingEffect(event_ledger=replayed, order=order)

    result = restarted.execute_hypothesis(EventRef(completed.hypothesis_event_id), effect=replay_effect)

    assert result.status is ChassisStatus.CONSEQUENCE_RECORDED_UPDATE_UNAVAILABLE
    assert result.resumed_from_ledger and not result.effect_invoked
    assert replay_effect.requests == []
    assert replayed.verify() == []


def test_replayed_default_tick_advances_to_recorded_authority_before_charging() -> None:
    events, _, hypothesis, _, original_runtime, chassis, effect, _ = make_system(emit_action=False)
    recorded = chassis.execute_hypothesis(hypothesis.event_id, effect=effect, now_tick=5)
    assert recorded.status is ChassisStatus.ABSTAINED
    replayed = EventLedger.replay(events.payload())
    lifecycle = LifecycleLedger()
    actor = ChassisActor(event_ledger=replayed, order=[], emit_action=False)
    runtime = CoalitionRuntime(
        actors=(actor,),
        policy=ScriptedDispatchPolicy({}, default=(actor.descriptor.actor_id,)),
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0, A=1),
        ),
        ledger=lifecycle,
        event_ledger=replayed,
        clock_ns=lambda: 1,
    )
    restarted = EventSourcedCoalitionChassis(
        event_ledger=replayed,
        lifecycle_ledger=lifecycle,
        runtime=runtime,
        trusted_replay_runtime_ids=(original_runtime.runtime_id,),
    )
    before = lifecycle.entry_count

    result = restarted.execute_hypothesis(
        hypothesis.event_id,
        effect=RecordingEffect(event_ledger=replayed, order=[]),
    )

    assert result.status is ChassisStatus.ALREADY_COMPLETED
    new_charges = lifecycle.entries[before:]
    assert new_charges
    assert all(charge.start_tick >= 5 for charge in new_charges)


def test_chassis_hot_path_does_not_materialize_event_or_lifecycle_history() -> None:
    class NoHistoryIteration(list):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("hot path scanned ledger history")

    events, lifecycle, hypothesis, _, _, chassis, effect, _ = make_system()
    events._entries = NoHistoryIteration(events._entries)
    lifecycle._entries = NoHistoryIteration(lifecycle._entries)

    result = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert result.status is ChassisStatus.COMPLETED
    assert result.lifecycle_end_sequence == lifecycle.entry_count


def test_chassis_rejects_trace_from_a_different_runtime_authority(monkeypatch) -> None:
    events, lifecycle, hypothesis, actor, runtime, chassis, effect, _ = make_system()
    other_actor = ChassisActor(event_ledger=events, order=[])
    other_runtime = CoalitionRuntime(
        actors=(other_actor,),
        policy=ScriptedDispatchPolicy({}, default=(other_actor.descriptor.actor_id,)),
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0, A=1),
        ),
        ledger=lifecycle,
        event_ledger=events,
        clock_ns=lambda: 1,
    )
    dispatch = chassis.dispatch_from_hypothesis(hypothesis.event_id)
    foreign_trace = other_runtime.run(dispatch, now_tick=1)
    monkeypatch.setattr(runtime, "run", lambda *_args, **_kwargs: foreign_trace)

    with pytest.raises(ChassisContractError, match="different runtime authority"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert effect.requests == []
    assert chassis_commitments(events) == ()
    assert actor.activation_payloads == []


def test_chassis_rejects_time_before_runtime_accounting_frontier_without_charge() -> None:
    _, lifecycle, hypothesis, _, runtime, chassis, effect, _ = make_system()
    completed = chassis.execute_hypothesis(hypothesis.event_id, effect=effect)
    assert completed.status is ChassisStatus.COMPLETED
    runtime.run(None, now_tick=10)
    before = lifecycle.entry_count

    with pytest.raises(ChassisContractError, match="authority frontier"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect, now_tick=2)

    assert lifecycle.entry_count == before


def test_actionless_commitment_cannot_claim_an_accepted_factual_action() -> None:
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system()
    runtime_id = digest("foreign-runtime")
    authority_sequence = 0
    trace_id = canonical_sha256(
        {
            "schema": "mop-escs-runtime-trace-authority/v1",
            "runtime_id": runtime_id,
            "authority_sequence": authority_sequence,
        }
    )
    effect_id = canonical_sha256(
        {
            "schema": EFFECT_AUTHORITY_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "action_id": None,
            "trace_authority_id": trace_id,
        }
    )
    forged = CommitmentEvent.create(
        coalition_id=f"coalition:{trace_id}",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={
            "schema": CHASSIS_COMMITMENT_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "runtime_id": runtime_id,
            "trace_authority_sequence": authority_sequence,
            "trace_authority_id": trace_id,
            "full_trace_sha256": digest("foreign-full-trace"),
            "effect_id": effect_id,
            "decision_reason": "accepted-factual-action",
            "action_record": None,
            "blocked_action_id": None,
        },
        decision_distribution={"abstention": 1.0},
        deadline_tick=1,
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(hypothesis.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": trace_id,
        },
        evidence_class=hypothesis.evidence_class,
    )
    events.append(forged)

    with pytest.raises(ChassisContractError, match="impossible decision reason"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert effect.requests == []
    assert actor.activation_payloads == []
    assert events.consequences_for(forged.event_id) == ()


def test_stored_action_must_remain_authorized_by_its_hypothesis() -> None:
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system()
    with pytest.raises(InjectedChassisFailure):
        chassis.execute_hypothesis(
            hypothesis.event_id,
            effect=effect,
            failpoint=ChassisFailpoint.AFTER_COMMITMENT,
        )
    legitimate = chassis_commitments(events)[0]
    payload = copy.deepcopy(legitimate.committed_payload.value())
    record = payload["action_record"]
    record["identity"]["referent_hypotheses"] = ["referent:unauthorized"]
    record["action_id"] = canonical_sha256(
        {
            "header": record["identity"],
            "payload_base64": record["payload_base64"],
        }
    )
    payload["effect_id"] = canonical_sha256(
        {
            "schema": EFFECT_AUTHORITY_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "action_id": record["action_id"],
            "trace_authority_id": payload["trace_authority_id"],
        }
    )
    forged = CommitmentEvent.create(
        coalition_id=legitimate.coalition_id,
        commitment_kind=CommitmentKind.EXTERNAL_ACTION,
        committed_payload=payload,
        decision_distribution={"external_action": 1.0},
        deadline_tick=record["identity"]["expiry_tick"],
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(hypothesis.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": payload["trace_authority_id"],
        },
        evidence_class=legitimate.evidence_class,
    )
    events.append(forged)

    with pytest.raises(ChassisContractError, match="unauthorized referent"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert effect.requests == []
    assert actor.stage_calls == 0


def test_actionless_restart_reason_respects_simulated_then_oracle_precedence() -> None:
    branch = BranchRef("branch:simulated-precedence")
    events, _, hypothesis, actor, _, chassis, effect, _ = make_system(
        branch=branch,
        status=EpistemicStatus.SIMULATED,
        emit_action=False,
    )
    runtime_id = digest("precedence-runtime")
    authority_sequence = 0
    trace_id = canonical_sha256(
        {
            "schema": "mop-escs-runtime-trace-authority/v1",
            "runtime_id": runtime_id,
            "authority_sequence": authority_sequence,
        }
    )
    effect_id = canonical_sha256(
        {
            "schema": EFFECT_AUTHORITY_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "action_id": None,
            "trace_authority_id": trace_id,
        }
    )
    forged = CommitmentEvent.create(
        coalition_id=f"coalition:{trace_id}",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={
            "schema": CHASSIS_COMMITMENT_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "runtime_id": runtime_id,
            "trace_authority_sequence": authority_sequence,
            "trace_authority_id": trace_id,
            "full_trace_sha256": digest("precedence-full-trace"),
            "effect_id": effect_id,
            "decision_reason": "runtime-produced-no-factual-action",
            "action_record": None,
            "blocked_action_id": None,
        },
        decision_distribution={"abstention": 1.0},
        deadline_tick=1,
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(hypothesis.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": trace_id,
        },
        evidence_class=hypothesis.evidence_class,
    )
    events.append(forged)

    with pytest.raises(ChassisContractError, match="simulated abstention authority drift"):
        chassis.execute_hypothesis(hypothesis.event_id, effect=effect)

    assert effect.requests == []
    assert actor.activation_payloads == []
