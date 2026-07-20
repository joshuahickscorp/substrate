from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import replace

import pytest

from mop.escs.accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from mop.escs.actors import (
    ActionIntent,
    ActorActivationContext,
    ActorActivationResult,
    ActorDescriptor,
    ActorUpdateContext,
    ActorUpdatePlan,
    DispatchEvent,
    DispatchEventHeader,
    EndogenousHypothesisProposal,
    OutboundClaim,
    ReadinessEstimate,
)
from mop.escs.events import HypothesisEvent, HypothesisOrigin, ObservationEvent, state_version_for_parents
from mop.escs.ledger import EventLedger
from mop.escs.messages import (
    ClaimFault,
    ClaimMessage,
    ClaimSchema,
    EpistemicStatus,
    EvidenceClass,
    SchemaRegistry,
)
from mop.escs.runtime import (
    ActionFault,
    CandidateMode,
    CoalitionRuntime,
    DispatchDecision,
    RuntimeCapExceeded,
    RuntimeCaps,
    RuntimeConfig,
    RuntimeContractError,
    ScriptedDispatchPolicy,
)
from mop.substrate.events import EventRef, canonical_bytes, canonical_sha256


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


CLAIM_SCHEMA = ClaimSchema(
    schema_id="mop.test.runtime-claim",
    version=1,
    claim_types=frozenset({"factor_distribution"}),
    payload_forms=frozenset({"probability-table"}),
)


def make_event(
    *,
    event_id: str = "event:hypothesis/root",
    status: EpistemicStatus = EpistemicStatus.INFERRED,
    routing_shards: tuple[str, ...] = ("shard:one",),
    evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
) -> DispatchEvent:
    return DispatchEvent.create(
        event_id=event_id,
        event_kind="hypothesis",
        branch_id="branch:factual",
        producer_state_version=state_version_for_parents((EventRef("event:observation/root"),)),
        epistemic_status=status,
        evidence_class=evidence_class,
        referent_hypotheses=("referent:item/1",),
        factor_scope=("factor:motion",),
        routing_shards=routing_shards,
        source_event_ids=("event:observation/root",),
        created_tick=3,
        expiry_tick=20,
        payload_bytes=b"private-event-payload",
    )


def make_ledger_event(
    *,
    status: EpistemicStatus = EpistemicStatus.INFERRED,
    routing_shards: tuple[str, ...] = ("shard:one",),
    evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
) -> tuple[DispatchEvent, EventLedger]:

    ledger = EventLedger()
    observation = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:test/root",),
        adapter_version="test-adapter/v1",
        sensor_scope={"sensor": "test"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={"source": "runtime-test"},
        measured_creation_cost=WorkVector(event_formation=1),
        evidence_class=evidence_class,
    )
    ledger.append(observation)
    hypothesis = HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=status,
        referent_hypotheses={"hypotheses": ["referent:item/1"]},
        factor_change_distribution={"factor_scope": ["factor:motion"]},
        decision_relevance_distribution={"values": []},
        reducibility_distribution={"values": []},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.6,
        abstention_reason=None,
        predicted_value_of_further_computation=0.2,
        causal_parent_ids=(observation.event_id,),
        counterfactual_branch_id=FACTUAL_BRANCH,
        clock_start_tick=3,
        clock_end_tick=3,
        source_and_provenance={"source": "runtime-test-event-former"},
        measured_creation_cost=WorkVector(event_formation=1),
        evidence_class=evidence_class,
    )
    ledger.append(hypothesis)
    payload = b"private-event-payload"
    header = DispatchEventHeader(
        event_id=str(hypothesis.event_id),
        event_kind=hypothesis.kind.value,
        branch_id=str(hypothesis.branch_id),
        producer_state_version=hypothesis.envelope.producer_state_version,
        epistemic_status=hypothesis.epistemic_status,
        evidence_class=hypothesis.evidence_class,
        referent_hypotheses=("referent:item/1",),
        factor_scope=("factor:motion",),
        routing_shards=routing_shards,
        source_event_ids=(str(observation.event_id),),
        created_tick=hypothesis.envelope.clock_end_tick,
        expiry_tick=20,
        payload_digest=hypothesis.envelope.payload_digest,
        representation_payload_digest=hashlib.sha256(payload).hexdigest(),
    )
    return DispatchEvent(header, payload), ledger


class ScriptedActor:

    def __init__(
        self,
        actor_id: str,
        *,
        factor_scopes: tuple[str, ...] = (),
        shards: tuple[str, ...] = (),
        peers: tuple[str, ...] = (),
        peer_entry: bool = False,
        nominations: tuple[str, ...] = (),
        recipients: tuple[str, ...] = (),
        emit_action: bool = False,
        endogenous_depth: int = 0,
        claim_status: EpistemicStatus | None = None,
        output_evidence_class: EvidenceClass | None = None,
        claim_expiry_tick: int | None = None,
        fail_stage: bool = False,
        on_activate: Callable[[ActorActivationContext], None] | None = None,
    ) -> None:
        self._descriptor = ActorDescriptor(
            actor_id=actor_id,
            subscribed_event_types=("hypothesis",),
            factor_scopes=factor_scopes,
            shard_ids=shards,
            declared_peer_ids=peers,
            peer_entry=peer_entry,
        )
        self._private_counter = 0
        self._nominations = nominations
        self._recipients = recipients
        self._emit_action = emit_action
        self._endogenous_depth = endogenous_depth
        self._claim_status = claim_status
        self._output_evidence_class = output_evidence_class
        self._claim_expiry_tick = claim_expiry_tick
        self._fail_stage = fail_stage
        self._on_activate = on_activate
        self.readiness_headers = []
        self.activation_payloads: list[bytes] = []
        self.incoming_counts: list[int] = []
        self.update_calls = 0
        self.stage_log: list[str] = []

    @property
    def descriptor(self) -> ActorDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return digest(f"{self._descriptor.actor_id}:{self._private_counter}")

    @property
    def retained_state_bytes(self) -> int:
        return 64

    def readiness(self, header):
        self.readiness_headers.append(header)
        return ReadinessEstimate(
            actor_id=self.descriptor.actor_id,
            state_version=self.state_version,
            compatible=self.descriptor.header_compatible(header),
            expected_decision_value=0.2,
            predicted_operations=7,
            predicted_message_bytes=64,
            nominated_actor_ids=self._nominations,
            estimation_operations=3,
        )

    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        self.activation_payloads.append(context.event_payload_bytes)
        self.incoming_counts.append(len(context.incoming_claims))
        if self._on_activate is not None:
            self._on_activate(context)
        claims = ()
        if self._recipients and context.event_header.reasoning_depth == 0:
            status = self._claim_status or context.event_header.epistemic_status
            message = ClaimMessage.create(
                schema=CLAIM_SCHEMA,
                source_hypothesis_event_ids=(context.event_header.event_id,),
                referent_hypotheses=context.event_header.referent_hypotheses,
                branch_id=context.event_header.branch_id,
                factor_scope=context.event_header.factor_scope,
                claim_type="factor_distribution",
                epistemic_status=status,
                evidence_class=self._output_evidence_class or context.event_header.evidence_class,
                supporting_event_ids=(),
                producer_actor_id=self.descriptor.actor_id,
                producer_state_version=self.state_version,
                calibrated_confidence=0.6,
                created_tick=context.event_header.created_tick,
                expiry_tick=(
                    context.event_header.expiry_tick
                    if self._claim_expiry_tick is None
                    else self._claim_expiry_tick
                ),
                predicted_utility=(0.1,),
                producer_operations=10,
                payload_form="probability-table",
                payload_bytes=b"{candidate:0.6}",
            )
            claims = (OutboundClaim(message, self._recipients),)

        actions = ()
        if self._emit_action and context.event_header.reasoning_depth == 0:
            actions = (
                ActionIntent.create(
                    source_event_id=context.event_header.event_id,
                    branch_id=context.event_header.branch_id,
                    referent_hypotheses=context.event_header.referent_hypotheses,
                    epistemic_status=context.event_header.epistemic_status,
                    evidence_class=self._output_evidence_class or context.event_header.evidence_class,
                    producer_actor_id=self.descriptor.actor_id,
                    producer_state_version=self.state_version,
                    created_tick=context.event_header.created_tick,
                    expiry_tick=context.event_header.expiry_tick,
                    producer_operations=8,
                    payload_form="motor-program",
                    payload_bytes=b"turn-left",
                ),
            )

        endogenous = ()
        if context.event_header.reasoning_depth < self._endogenous_depth:
            next_depth = context.event_header.reasoning_depth + 1
            endogenous = (
                EndogenousHypothesisProposal.create(
                    producer_actor_id=self.descriptor.actor_id,
                    producer_actor_state_version=self.state_version,
                    source_event_ids=(context.event_header.event_id,),
                    supporting_event_ids=(context.event_header.event_id,),
                    branch_id=context.event_header.branch_id,
                    epistemic_status=context.event_header.epistemic_status,
                    evidence_class=self._output_evidence_class or context.event_header.evidence_class,
                    referent_hypotheses=context.event_header.referent_hypotheses,
                    factor_scope=context.event_header.factor_scope,
                    routing_shards=context.event_header.routing_shards,
                    created_tick=context.event_header.created_tick + next_depth,
                    expiry_tick=context.event_header.expiry_tick,
                    calibrated_confidence=0.55,
                    predicted_value_of_further_computation=0.1,
                    producer_operations=10,
                    payload_form="test-reasoning-state",
                    payload_bytes=(f"{self.descriptor.actor_id}:reasoning-depth:{next_depth}".encode()),
                ),
            )
        return ActorActivationResult(
            outbound_claims=claims,
            action_intents=actions,
            endogenous_proposals=endogenous,
            executed_operations=20,
        )

    def stage_update(self, context: ActorUpdateContext) -> ActorUpdatePlan:
        assert context.consequence_payload_bytes
        self.stage_log.append(context.idempotency_key)
        if self._fail_stage:
            raise RuntimeError("scripted stage failure")
        replacement = copy.copy(self)
        replacement._private_counter += 1
        return ActorUpdatePlan(
            actor_id=self.descriptor.actor_id,
            prior_state_version=self.state_version,
            next_state_version=replacement.state_version,
            idempotency_key=context.idempotency_key,
            executed_operations=5,
            replacement_actor=replacement,
        )


def simulated_child(
    context: ActorActivationContext,
    *,
    producer_actor_id: str,
    producer_actor_state_version: str,
    payload_label: str,
    branch_id: str,
    created_tick: int,
) -> EndogenousHypothesisProposal:
    return EndogenousHypothesisProposal.create(
        producer_actor_id=producer_actor_id,
        producer_actor_state_version=producer_actor_state_version,
        source_event_ids=(context.event_header.event_id,),
        supporting_event_ids=(context.event_header.event_id,),
        branch_id=branch_id,
        epistemic_status=EpistemicStatus.SIMULATED,
        evidence_class=context.event_header.evidence_class,
        referent_hypotheses=context.event_header.referent_hypotheses,
        factor_scope=context.event_header.factor_scope,
        routing_shards=context.event_header.routing_shards,
        created_tick=created_tick,
        expiry_tick=30,
        calibrated_confidence=0.5,
        predicted_value_of_further_computation=0.1,
        producer_operations=10,
        payload_form="counterfactual-test-state",
        payload_bytes=payload_label.encode(),
    )


class SiblingMessageActor(ScriptedActor):
    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        base = super().activate(context)
        if context.event_header.reasoning_depth == 0:
            return ActorActivationResult(
                endogenous_proposals=(
                    simulated_child(
                        context,
                        producer_actor_id=self.descriptor.actor_id,
                        producer_actor_state_version=self.state_version,
                        payload_label="counterfactual-one",
                        branch_id="branch:cf/one",
                        created_tick=4,
                    ),
                    simulated_child(
                        context,
                        producer_actor_id=self.descriptor.actor_id,
                        producer_actor_state_version=self.state_version,
                        payload_label="counterfactual-two",
                        branch_id="branch:cf/two",
                        created_tick=5,
                    ),
                ),
                executed_operations=base.executed_operations,
            )
        if context.event_header.branch_id == "branch:cf/one":
            message = ClaimMessage.create(
                schema=CLAIM_SCHEMA,
                source_hypothesis_event_ids=(context.event_header.event_id,),
                referent_hypotheses=context.event_header.referent_hypotheses,
                branch_id=context.event_header.branch_id,
                factor_scope=context.event_header.factor_scope,
                claim_type="factor_distribution",
                epistemic_status=EpistemicStatus.SIMULATED,
                evidence_class=context.event_header.evidence_class,
                supporting_event_ids=(),
                producer_actor_id=self.descriptor.actor_id,
                producer_state_version=self.state_version,
                calibrated_confidence=0.5,
                created_tick=context.event_header.created_tick,
                expiry_tick=20,
                predicted_utility=(0.1,),
                producer_operations=5,
                payload_form="probability-table",
                payload_bytes=b"cf-one-only",
            )
            return ActorActivationResult(
                outbound_claims=(OutboundClaim(message, ("actor:beta",)),),
                executed_operations=20,
            )
        return base


class FactualAndCounterfactualActor(ScriptedActor):
    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        base = super().activate(context)
        if context.event_header.reasoning_depth == 0:
            return ActorActivationResult(
                action_intents=base.action_intents,
                endogenous_proposals=(
                    simulated_child(
                        context,
                        producer_actor_id=self.descriptor.actor_id,
                        producer_actor_state_version=self.state_version,
                        payload_label="counterfactual-action",
                        branch_id="branch:cf/action",
                        created_tick=4,
                    ),
                ),
                executed_operations=base.executed_operations,
            )
        return base


class EveryRoundActionActor(ScriptedActor):
    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        base = super().activate(context)
        action = ActionIntent.create(
            source_event_id=context.event_header.event_id,
            branch_id=context.event_header.branch_id,
            referent_hypotheses=context.event_header.referent_hypotheses,
            epistemic_status=context.event_header.epistemic_status,
            evidence_class=context.event_header.evidence_class,
            producer_actor_id=self.descriptor.actor_id,
            producer_state_version=self.state_version,
            created_tick=context.event_header.created_tick,
            expiry_tick=context.event_header.expiry_tick,
            producer_operations=8,
            payload_form="motor-program",
            payload_bytes=b"counterfactual-action",
        )
        return ActorActivationResult(
            outbound_claims=base.outbound_claims,
            action_intents=(action,),
            endogenous_proposals=base.endogenous_proposals,
            executed_operations=20,
        )


class RoundDispatchPolicy:
    def __init__(self, selections: dict[int, tuple[str, ...]]) -> None:
        self._decisions = {
            reasoning_round: DispatchDecision.select(*actor_ids)
            for reasoning_round, actor_ids in selections.items()
        }
        self.request_count = 0
        self.last_request = None

    def select(self, request):
        self.request_count += 1
        self.last_request = request
        return self._decisions.get(request.reasoning_round, DispatchDecision.select())

    @property
    def authority_id(self) -> str:
        return canonical_sha256(
            {
                "schema": "test-round-dispatch-policy/v1",
                "decisions": {
                    str(reasoning_round): list(decision.selected_actor_ids)
                    for reasoning_round, decision in sorted(self._decisions.items())
                },
            }
        )

    @property
    def state_version(self) -> str:
        return canonical_sha256(
            {
                "authority_id": self.authority_id,
                "request_count": self.request_count,
                "last_event_id": (
                    None if self.last_request is None else self.last_request.event_header.event_id
                ),
                "last_reasoning_round": (
                    None if self.last_request is None else self.last_request.reasoning_round
                ),
            }
        )

    @property
    def retained_state_bytes(self) -> int:
        return len(
            canonical_bytes(
                {
                    "decisions": {
                        str(reasoning_round): list(decision.selected_actor_ids)
                        for reasoning_round, decision in sorted(self._decisions.items())
                    },
                    "request_count": self.request_count,
                    "last_event_id": (
                        None if self.last_request is None else self.last_request.event_header.event_id
                    ),
                    "last_reasoning_round": (
                        None if self.last_request is None else self.last_request.reasoning_round
                    ),
                }
            )
        )


class StaticDispatchPolicy:
    def __init__(self, decision: DispatchDecision) -> None:
        self._decision = decision

    @property
    def authority_id(self) -> str:
        beam = [
            list(row) if type(row) is tuple else {"opaque_test_type": type(row).__qualname__}
            for row in self._decision.considered_coalitions
        ]
        return canonical_sha256(
            {
                "schema": "test-static-dispatch-policy/v1",
                "selected": list(self._decision.selected_actor_ids),
                "beam": beam,
            }
        )

    @property
    def state_version(self) -> str:
        return self.authority_id

    @property
    def retained_state_bytes(self) -> int:
        return 1

    def select(self, request):
        del request
        return self._decision


def make_runtime(
    actors,
    policy,
    *,
    mode: CandidateMode = CandidateMode.BOUNDED_CENTRAL,
    caps: RuntimeCaps = RuntimeCaps(K=4, C=2, B=2, H=1, M=4, R=1),
    event_ledger: EventLedger | None = None,
    **config_overrides,
):
    ledger = LifecycleLedger()
    runtime = CoalitionRuntime(
        actors=actors,
        policy=policy,
        schemas=SchemaRegistry((CLAIM_SCHEMA,)),
        config=RuntimeConfig(mode=mode, caps=caps, **config_overrides),
        ledger=ledger,
        event_ledger=event_ledger,
    )
    return runtime, ledger


def test_policy_and_readiness_are_header_only_and_inactive_actor_stays_private_and_frozen():
    alpha = ScriptedActor("actor:alpha", emit_action=True)
    inactive = ScriptedActor("actor:inactive")
    event = make_event()
    policy = ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha",)})
    runtime, ledger = make_runtime(
        (alpha, inactive),
        policy,
        caps=RuntimeCaps(K=2, C=1, B=1, H=0, M=1, R=0),
    )

    trace = runtime.run(event)

    assert policy.request_count == 1
    assert policy.last_request is not None
    assert not hasattr(policy.last_request, "event_payload_bytes")
    assert not hasattr(policy, "seen_requests")
    assert alpha.readiness_headers == [event.header]
    assert inactive.readiness_headers == [event.header]
    assert alpha.activation_payloads == [b"private-event-payload"]
    assert inactive.activation_payloads == []
    assert trace.active_actor_ids == ("actor:alpha",)
    assert len(trace.action_intents) == 1 and not trace.rejected_actions
    assert all(not hasattr(row, "_private_counter") for row in runtime.actor_descriptors)

    updated = runtime.apply_consequence(
        trace_id=trace.trace_id,
        consequence_event_id="event:consequence/1",
        authorization_id=trace.action_intents[0].action_id,
        branch_id="branch:factual",
        consequence_payload_bytes=b"utility:+1",
        tick=10,
    )
    assert updated == ("actor:alpha",)
    assert len(alpha.stage_log) == 1 and not inactive.stage_log
    assert ledger.total.dispatch_and_exploration > 0
    assert ledger.total.actor_execution > 0
    assert ledger.total.learning > 0
    assert ledger.verify() == []
    reasons = {row.reason for row in ledger.entries}
    assert "dispatch-policy-selection" in reasons
    assert "actor-activation:actor:alpha" in reasons
    assert "action-intent:actor:alpha" in reasons


def test_claim_delivery_requires_a_bounded_endogenous_round_and_every_edge_is_charged():
    alpha = ScriptedActor("actor:alpha", recipients=("actor:beta",), endogenous_depth=1)
    beta = ScriptedActor("actor:beta")
    event, event_ledger = make_ledger_event()
    policy = ScriptedDispatchPolicy({}, default=("actor:alpha", "actor:beta"))
    runtime, ledger = make_runtime(
        (alpha, beta),
        policy,
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )

    trace = runtime.run(event)

    assert trace.quiescent and trace.halt_reason == "quiescent"
    assert trace.endogenous_rounds == 1 and len(trace.rounds) == 2
    assert len(trace.message_deliveries) == 1 and not trace.rejected_claims
    assert beta.incoming_counts == [0, 1]
    delivered = trace.message_deliveries[0].message
    assert ledger.total.messages == 2 * (delivered.encoded_bytes + 1)
    assert trace.message_deliveries[0].produced_round == 0
    assert trace.message_deliveries[0].consumed_round == 1
    assert any(row.reason == "message-edge:actor:alpha->actor:beta" for row in ledger.entries)
    endogenous_id = trace.rounds[0].admitted_endogenous_event_ids[0]
    resident = event_ledger.get(EventRef(endogenous_id))
    assert isinstance(resident, HypothesisEvent)
    assert resident.origin is HypothesisOrigin.ACTOR
    assert trace.rounds[1].event_header.payload_digest == resident.envelope.payload_digest
    provenance = resident.envelope.source_and_provenance.value()
    representation_sha = hashlib.sha256(b"actor:alpha:reasoning-depth:1").hexdigest()
    assert provenance["representation_payload_digest"] == representation_sha
    assert trace.rounds[1].event_header.representation_payload_digest == representation_sha


def test_epistemically_laundered_claim_is_charged_but_not_delivered():
    alpha = ScriptedActor(
        "actor:alpha",
        recipients=("actor:beta",),
        claim_status=EpistemicStatus.OBSERVED_CANDIDATE,
    )
    beta = ScriptedActor("actor:beta")
    event = make_event(status=EpistemicStatus.INFERRED)
    policy = ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha", "actor:beta")})
    runtime, ledger = make_runtime(
        (alpha, beta),
        policy,
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=0),
    )

    trace = runtime.run(event)

    assert not trace.message_deliveries
    assert len(trace.rejected_claims) == 1
    assert ClaimFault.EPISTEMIC_LAUNDERING in trace.rejected_claims[0].faults
    assert ledger.total.messages > 0


def test_k_bounds_dormant_population_and_c_b_m_caps_fail_closed_after_charging():
    actors = tuple(ScriptedActor(f"actor:{index}") for index in range(8))
    event = make_event()
    runtime, _ = make_runtime(
        actors,
        ScriptedDispatchPolicy({}, default=()),
        caps=RuntimeCaps(K=3, C=2, B=1, H=0, M=1, R=0),
    )
    trace = runtime.run(event)
    assert len(trace.rounds[0].candidate_actor_ids) == 3
    assert sum(bool(actor.readiness_headers) for actor in actors) == 3

    for decision, match in (
        (DispatchDecision.select("actor:0", "actor:1", "actor:2"), "coalition exceeded C"),
        (
            DispatchDecision(
                selected_actor_ids=("actor:0",),
                considered_coalitions=(("actor:0",), ("actor:1",)),
            ),
            "beam exceeded B",
        ),
    ):
        failing, ledger = make_runtime(
            tuple(ScriptedActor(f"actor:{index}") for index in range(3)),
            ScriptedDispatchPolicy({(event.header.event_id, 0): decision}),
            caps=RuntimeCaps(K=3, C=2, B=1, H=0, M=1, R=0),
        )
        with pytest.raises(RuntimeCapExceeded, match=match):
            failing.run(event)
        assert any(row.reason == "dispatch-policy-shape-rejection" for row in ledger.entries)

    sender = ScriptedActor("actor:sender", recipients=("actor:r1", "actor:r2"))
    recipients = (ScriptedActor("actor:r1"), ScriptedActor("actor:r2"))
    message_runtime, ledger = make_runtime(
        (sender, *recipients),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:sender", "actor:r1", "actor:r2")}),
        caps=RuntimeCaps(K=3, C=3, B=1, H=0, M=1, R=0),
    )
    with pytest.raises(RuntimeCapExceeded, match="message edges exceeded M"):
        message_runtime.run(event)
    assert ledger.total.messages > 0
    assert sum("message-edge-cap-rejection" in row.reason for row in ledger.entries) == 1


def test_sharded_and_peer_modes_use_only_their_bounded_neighborhoods():
    shard_one = ScriptedActor("actor:one", shards=("shard:one",))
    shard_two = ScriptedActor("actor:two", shards=("shard:two",))
    event = make_event(routing_shards=("shard:two",))
    sharded, _ = make_runtime(
        (shard_one, shard_two),
        ScriptedDispatchPolicy({}, default=()),
        mode=CandidateMode.SHARDED_SUBSCRIPTION,
        caps=RuntimeCaps(K=2, C=1, B=1, H=0, M=1, R=0),
        shard_cap=1,
        bids_per_shard=1,
    )
    trace = sharded.run(event)
    assert trace.rounds[0].candidate_actor_ids == ("actor:two",)
    assert not shard_one.readiness_headers and shard_two.readiness_headers

    entry = ScriptedActor(
        "actor:entry",
        peers=("actor:middle",),
        peer_entry=True,
        nominations=("actor:middle",),
    )
    middle = ScriptedActor(
        "actor:middle",
        peers=("actor:far",),
        nominations=("actor:far",),
    )
    far = ScriptedActor("actor:far")
    peer, _ = make_runtime(
        (entry, middle, far),
        ScriptedDispatchPolicy({}, default=()),
        mode=CandidateMode.PEER_NOMINATION,
        caps=RuntimeCaps(K=3, C=1, B=1, H=1, M=1, R=0),
        peer_entry_cap=1,
        peer_nomination_degree=1,
    )
    trace = peer.run(make_event())
    assert trace.rounds[0].candidate_actor_ids == ("actor:entry", "actor:middle")
    assert not far.readiness_headers


def test_r_is_an_exact_endogenous_round_cap_and_empty_queue_has_a_charged_idle_floor():
    emitter = ScriptedActor("actor:emitter", endogenous_depth=2)
    event, event_ledger = make_ledger_event()
    policy = ScriptedDispatchPolicy({}, default=("actor:emitter",))
    runtime, ledger = make_runtime(
        (emitter,),
        policy,
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )
    trace = runtime.run(event)
    assert len(trace.rounds) == 2 and trace.endogenous_rounds == 1
    assert not trace.quiescent and trace.halt_reason == "endogenous-round-cap"

    idle_before = ledger.total.idle_floor
    idle = runtime.run(None, now_tick=30)
    assert idle.quiescent and idle.halt_reason == "quiescent-empty-queue"
    assert ledger.total.idle_floor == idle_before + 1
    with pytest.raises(Exception, match="unknown, idle"):
        runtime.apply_consequence(
            trace_id=idle.trace_id,
            consequence_event_id="event:consequence/idle",
            authorization_id="a" * 64,
            branch_id="branch:factual",
            consequence_payload_bytes=b"none",
            tick=31,
        )


def test_messages_are_next_round_only_and_revalidated_for_expiry_state_and_branch():
    sender = ScriptedActor("actor:sender", recipients=("actor:beta",))
    beta = ScriptedActor("actor:beta")
    event = make_event()
    runtime, _ = make_runtime(
        (sender, beta),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:sender", "actor:beta")}),
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=0),
    )
    dropped = runtime.run(event)
    assert beta.incoming_counts == [0]
    assert not dropped.message_deliveries
    assert dropped.rejected_claims[0].phase == "unconsumed"

    expiring = ScriptedActor(
        "actor:alpha",
        recipients=("actor:beta",),
        endogenous_depth=1,
        claim_expiry_tick=3,
    )
    beta = ScriptedActor("actor:beta")
    event, event_ledger = make_ledger_event()
    policy = ScriptedDispatchPolicy({}, default=("actor:alpha", "actor:beta"))
    runtime, _ = make_runtime(
        (expiring, beta),
        policy,
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )
    expired = runtime.run(event)
    assert not expired.message_deliveries
    assert ClaimFault.EXPIRED in expired.rejected_claims[0].faults
    assert expired.rejected_claims[0].phase == "consumption"

    producer = ScriptedActor("actor:alpha", recipients=("actor:beta",), endogenous_depth=1)
    beta = ScriptedActor("actor:beta")
    event, event_ledger = make_ledger_event()

    class MutatingPolicy(ScriptedDispatchPolicy):
        def select(self, request):
            if request.reasoning_round == 1:
                producer._private_counter += 1
            return super().select(request)

    policy = MutatingPolicy({}, default=("actor:alpha", "actor:beta"))
    runtime, _ = make_runtime(
        (producer, beta),
        policy,
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )
    stale = runtime.run(event)
    assert not stale.message_deliveries
    assert ClaimFault.PRODUCER_STATE in stale.rejected_claims[0].faults

    alpha = SiblingMessageActor("actor:alpha")
    beta = ScriptedActor("actor:beta")
    event, event_ledger = make_ledger_event()
    policy = ScriptedDispatchPolicy({}, default=("actor:alpha", "actor:beta"))
    runtime, _ = make_runtime(
        (alpha, beta),
        policy,
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=2),
        endogenous_fanout_cap=2,
        event_ledger=event_ledger,
    )
    crossed = runtime.run(event)
    assert beta.incoming_counts == [0, 0, 0]
    assert not crossed.message_deliveries
    consumed_rejection = next(row for row in crossed.rejected_claims if row.phase == "consumption")
    assert ClaimFault.BRANCH in consumed_rejection.faults


def test_pending_updates_are_branch_partitioned_validate_before_consume_and_atomic():
    root = FactualAndCounterfactualActor("actor:root", emit_action=True)
    counterfactual = EveryRoundActionActor("actor:cf")
    event, event_ledger = make_ledger_event()
    policy = RoundDispatchPolicy({0: ("actor:root",), 1: ("actor:cf",)})
    runtime, _ = make_runtime(
        (root, counterfactual),
        policy,
        caps=RuntimeCaps(K=2, C=1, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )
    trace = runtime.run(event)
    factual_action = next(row for row in trace.action_intents if row.branch_id == "branch:factual")
    cf_action = next(row for row in trace.action_intents if row.branch_id == "branch:cf/action")

    assert runtime.apply_consequence(
        trace_id=trace.trace_id,
        consequence_event_id="event:consequence/factual",
        authorization_id=factual_action.action_id,
        branch_id="branch:factual",
        consequence_payload_bytes=b"factual",
        tick=10,
    ) == ("actor:root",)
    assert len(root.stage_log) == 1 and not counterfactual.stage_log

    with pytest.raises(Exception, match="no pending|already-consumed|unknown"):
        runtime.apply_consequence(
            trace_id=trace.trace_id,
            consequence_event_id="event:consequence/cf",
            authorization_id=cf_action.action_id,
            branch_id="branch:cf/action",
            consequence_payload_bytes=b"cf",
            tick=11,
        )
    assert not counterfactual.stage_log

    alpha = ScriptedActor("actor:alpha", emit_action=True)
    failing = ScriptedActor("actor:failing", fail_stage=True)
    runtime, _ = make_runtime(
        (alpha, failing),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha", "actor:failing")}),
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=0),
    )
    trace = runtime.run(event)
    before = dict(runtime.actor_state_versions)
    action = trace.action_intents[0]
    with pytest.raises(Exception, match="not bound"):
        runtime.apply_consequence(
            trace_id=trace.trace_id,
            consequence_event_id="event:consequence/atomic",
            authorization_id="b" * 64,
            branch_id="branch:factual",
            consequence_payload_bytes=b"atomic",
            tick=12,
        )
    with pytest.raises(RuntimeError, match="scripted stage failure"):
        runtime.apply_consequence(
            trace_id=trace.trace_id,
            consequence_event_id="event:consequence/atomic",
            authorization_id=action.action_id,
            branch_id="branch:factual",
            consequence_payload_bytes=b"atomic",
            tick=12,
        )
    assert dict(runtime.actor_state_versions) == before
    failing._fail_stage = False
    assert runtime.apply_consequence(
        trace_id=trace.trace_id,
        consequence_event_id="event:consequence/atomic",
        authorization_id=action.action_id,
        branch_id="branch:factual",
        consequence_payload_bytes=b"atomic",
        tick=12,
    ) == ("actor:alpha", "actor:failing")

    solo = ScriptedActor("actor:solo", emit_action=True)
    runtime, _ = make_runtime(
        (solo,),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:solo",)}),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=0, P=2),
    )
    older = runtime.run(event)
    newer = runtime.run(event)
    assert runtime.pending_update_count == 2
    runtime.apply_consequence(
        trace_id=newer.trace_id,
        consequence_event_id="event:consequence/newer",
        authorization_id=newer.action_intents[0].action_id,
        branch_id="branch:factual",
        consequence_payload_bytes=b"newer",
        tick=13,
    )
    assert runtime.pending_update_count == 0
    with pytest.raises(Exception, match="unknown|already-consumed"):
        runtime.apply_consequence(
            trace_id=older.trace_id,
            consequence_event_id="event:consequence/older",
            authorization_id=older.action_intents[0].action_id,
            branch_id="branch:factual",
            consequence_payload_bytes=b"older",
            tick=14,
        )

    two_actions, _ = make_runtime(
        (
            ScriptedActor("actor:left", emit_action=True),
            ScriptedActor("actor:right", emit_action=True),
        ),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:left", "actor:right")}),
        caps=RuntimeCaps(K=2, C=2, B=1, H=0, M=1, R=0),
    )
    with pytest.raises(Exception, match="one accepted action"):
        two_actions.run(event)


@pytest.mark.parametrize(
    ("caps", "match"),
    (
        (RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=0, A=0), "action intent count"),
        (
            RuntimeCaps(
                K=1,
                C=1,
                B=1,
                H=0,
                M=1,
                R=0,
                max_action_encoded_bytes=1,
            ),
            "action encoded bytes",
        ),
        (
            RuntimeCaps(
                K=1,
                C=1,
                B=1,
                H=0,
                M=1,
                R=1,
                max_endogenous_payload_bytes=0,
            ),
            "endogenous payload bytes",
        ),
        (
            RuntimeCaps(
                K=1,
                C=1,
                B=1,
                H=0,
                M=1,
                R=0,
                max_header_referents=0,
            ),
            "dispatch referents",
        ),
        (
            RuntimeCaps(
                K=1,
                C=1,
                B=1,
                H=0,
                M=1,
                R=0,
                max_header_encoded_bytes=1,
            ),
            "encoded header bytes",
        ),
        (RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=0, P=0), "pending update"),
        (
            RuntimeCaps(
                K=1,
                C=1,
                B=1,
                H=0,
                M=1,
                R=0,
                max_pending_update_bytes=1,
            ),
            "pending update authority bytes",
        ),
    ),
)
def test_action_endogenous_header_and_pending_caps_charge_then_fail_closed(caps, match):
    actor = ScriptedActor(
        "actor:alpha",
        emit_action=True,
        endogenous_depth=int(caps.R > 0),
    )
    if caps.R > 0:
        event, event_ledger = make_ledger_event()
    else:
        event, event_ledger = make_event(), None
    runtime, ledger = make_runtime(
        (actor,),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha",)}),
        caps=caps,
        event_ledger=event_ledger,
    )
    before = actor.state_version
    with pytest.raises(RuntimeCapExceeded, match=match):
        runtime.run(event)
    assert actor.state_version == before
    assert ledger.total.total_work > 0


def test_endogenous_proposals_require_a_ledger_and_duplicate_content_is_rejected():
    missing = ScriptedActor("actor:alpha", endogenous_depth=1)
    event = make_event()
    runtime, ledger = make_runtime(
        (missing,),
        ScriptedDispatchPolicy({}, default=("actor:alpha",)),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=1),
    )
    with pytest.raises(RuntimeContractError, match="injected EventLedger"):
        runtime.run(event)
    assert any("endogenous-missing-ledger-rejection" in row.reason for row in ledger.entries)

    class DuplicateProposalActor(ScriptedActor):
        proposal: EndogenousHypothesisProposal | None = None

        def activate(self, context: ActorActivationContext) -> ActorActivationResult:
            base = super().activate(context)
            proposal = base.endogenous_proposals[0]
            self.proposal = proposal
            return ActorActivationResult(
                endogenous_proposals=(proposal, proposal),
                executed_operations=base.executed_operations,
            )

    duplicate = DuplicateProposalActor("actor:alpha", endogenous_depth=1)
    event, event_ledger = make_ledger_event()
    runtime, _ = make_runtime(
        (duplicate,),
        ScriptedDispatchPolicy({}, default=("actor:alpha",)),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=1),
        endogenous_fanout_cap=2,
        queue_cap=2,
        event_ledger=event_ledger,
    )
    with pytest.raises(RuntimeContractError, match="ledger append was rejected") as caught:
        runtime.run(event)
    assert "duplicate event identity" in str(caught.value.__cause__)
    assert duplicate.proposal is not None
    assert not hasattr(duplicate.proposal, "event_id")
    assert len(event_ledger.entries) == 3


def test_endogenous_proposal_rejects_zero_identity_and_noninteger_work():
    event = make_event()
    proposal = EndogenousHypothesisProposal.create(
        producer_actor_id="actor:alpha",
        producer_actor_state_version=digest("actor:alpha:0"),
        source_event_ids=(event.header.event_id,),
        supporting_event_ids=(event.header.event_id,),
        branch_id=event.header.branch_id,
        epistemic_status=event.header.epistemic_status,
        evidence_class=event.header.evidence_class,
        referent_hypotheses=event.header.referent_hypotheses,
        factor_scope=event.header.factor_scope,
        routing_shards=event.header.routing_shards,
        created_tick=4,
        expiry_tick=20,
        calibrated_confidence=0.5,
        predicted_value_of_further_computation=0.1,
        producer_operations=1,
        payload_form="test-state",
        payload_bytes=b"state",
    )

    with pytest.raises(ValueError, match="identity digest mismatch"):
        replace(proposal, proposal_id="0" * 64)
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(proposal, producer_operations=True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        replace(proposal, producer_operations=1.5)  # type: ignore[arg-type]


def test_runtime_exposes_exact_injected_event_ledger_identity():
    event, event_ledger = make_ledger_event()
    runtime, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        ScriptedDispatchPolicy({}, default=()),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
        event_ledger=event_ledger,
    )
    assert runtime.event_ledger is event_ledger

    without_events, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        ScriptedDispatchPolicy({}, default=()),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    assert without_events.event_ledger is None
    assert event.header.event_id in event_ledger.event_ids


def test_trace_sequence_accounting_never_materializes_lifecycle_history():
    class NoHistoryIterationLedger(LifecycleLedger):
        @property
        def entries(self):
            raise AssertionError("runtime materialized lifecycle history")

    ledger = NoHistoryIterationLedger()
    runtime = CoalitionRuntime(
        actors=(ScriptedActor("actor:alpha"),),
        policy=ScriptedDispatchPolicy({}, default=()),
        schemas=SchemaRegistry((CLAIM_SCHEMA,)),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
        ),
        ledger=ledger,
    )

    trace = runtime.run(make_event())

    assert trace.ledger_start_sequence < trace.ledger_end_sequence == ledger.entry_count


def test_length_caps_precede_proportional_beam_hash_and_fanout_work():
    event = make_event()

    class PoisonCoalition:
        def __iter__(self):
            raise AssertionError("over-B beam element was inspected")

    over_b = DispatchDecision(
        selected_actor_ids=(),
        considered_coalitions=(PoisonCoalition(), PoisonCoalition()),  # type: ignore[arg-type]
    )
    runtime, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        StaticDispatchPolicy(over_b),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    with pytest.raises(RuntimeCapExceeded, match="beam exceeded B"):
        runtime.run(event)

    over_c = DispatchDecision(
        selected_actor_ids=([], []),  # type: ignore[arg-type]
        considered_coalitions=(),
    )
    runtime, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        StaticDispatchPolicy(over_c),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    with pytest.raises(RuntimeCapExceeded, match="coalition exceeded C"):
        runtime.run(event)

    class PoisonProposal:
        @property
        def payload_bytes(self):
            raise AssertionError("over-fanout proposal was inspected")

    class OverFanoutActor(ScriptedActor):
        def activate(self, context: ActorActivationContext) -> ActorActivationResult:
            del context
            return ActorActivationResult(
                endogenous_proposals=(PoisonProposal(), PoisonProposal()),  # type: ignore[arg-type]
                executed_operations=1,
            )

    runtime, _ = make_runtime(
        (OverFanoutActor("actor:alpha"),),
        ScriptedDispatchPolicy({}, default=("actor:alpha",)),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=1),
        endogenous_fanout_cap=1,
    )
    with pytest.raises(RuntimeCapExceeded, match="fanout cap"):
        runtime.run(event)


def test_scripted_policy_retains_only_a_saturating_count_and_last_request():
    event = make_event()
    policy = ScriptedDispatchPolicy({}, default=())
    policy.request_count = policy._MAX_REQUEST_COUNT - 1
    runtime, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        policy,
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    runtime.run(event)
    retained_at_limit = policy.retained_state_bytes
    runtime.run(event, now_tick=4)
    assert policy.request_count == policy._MAX_REQUEST_COUNT
    assert policy.retained_state_bytes == retained_at_limit
    assert policy.last_request is not None
    assert not hasattr(policy, "seen_requests")


def test_trace_authority_identity_is_distinct_from_full_trace_integrity():
    event = make_event()
    runtime, _ = make_runtime(
        (ScriptedActor("actor:alpha"),),
        ScriptedDispatchPolicy({}, default=()),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    first = runtime.run(event)
    second = runtime.run(event, now_tick=4)

    assert first.runtime_id == second.runtime_id == runtime.runtime_id
    assert (first.authority_sequence, second.authority_sequence) == (0, 1)
    assert first.trace_id != second.trace_id
    assert first.validate_integrity() and second.validate_integrity()
    assert first.full_trace_sha256 != second.full_trace_sha256

    tampered = replace(first, halt_reason="tampered")
    assert tampered.trace_id == first.trace_id
    assert not tampered.validate_integrity()
    authority_tampered = replace(first, trace_id="a" * 64)
    assert not authority_tampered.validate_integrity()


def test_runtime_authority_binds_policy_and_initial_actor_state() -> None:
    caps = RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0)
    config = RuntimeConfig(mode=CandidateMode.BOUNDED_CENTRAL, caps=caps)

    def build(actor: ScriptedActor, policy: ScriptedDispatchPolicy) -> CoalitionRuntime:
        return CoalitionRuntime(
            actors=(actor,),
            policy=policy,
            schemas=SchemaRegistry((CLAIM_SCHEMA,)),
            config=config,
            ledger=LifecycleLedger(),
            event_ledger=EventLedger(),
        )

    selecting = build(
        ScriptedActor("actor:alpha"),
        ScriptedDispatchPolicy({}, default=("actor:alpha",)),
    )
    abstaining = build(
        ScriptedActor("actor:alpha"),
        ScriptedDispatchPolicy({}, default=()),
    )
    changed_actor = ScriptedActor("actor:alpha")
    changed_actor._private_counter = 1
    changed_state = build(
        changed_actor,
        ScriptedDispatchPolicy({}, default=("actor:alpha",)),
    )

    assert selecting.runtime_id != abstaining.runtime_id
    assert selecting.runtime_id != changed_state.runtime_id


def test_runtime_retention_uses_one_half_open_frontier_and_finalize_reconciles():
    event = make_event()
    runtime, ledger = make_runtime(
        (ScriptedActor("actor:alpha"),),
        ScriptedDispatchPolicy({}, default=()),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=0, R=0),
    )
    initial_bytes = runtime.retained_state_bytes
    runtime.run(event)
    after_first = runtime.retained_state_bytes
    runtime.run(event, now_tick=10)
    after_second = runtime.retained_state_bytes
    runtime.finalize(end_tick=20)

    retention = tuple(row for row in ledger.entries if row.work.retained_byte_time)
    assert tuple((row.start_tick, row.end_tick) for row in retention) == (
        (0, 3),
        (3, 10),
        (10, 20),
    )
    assert tuple(row.work.retained_byte_time for row in retention) == (
        initial_bytes * 3,
        after_first * 7,
        after_second * 10,
    )
    assert sum(row.work.retained_byte_time for row in retention) == (ledger.total.retained_byte_time)
    assert runtime.last_accounted_tick == 20 and runtime.finalized
    with pytest.raises(RuntimeContractError, match="already finalized"):
        runtime.finalize(end_tick=21)
    with pytest.raises(RuntimeContractError, match="finalized runtime"):
        runtime.run(event, now_tick=21)


def test_oracle_taint_cannot_downgrade_in_actions_or_endogenous_events():
    event = make_event(evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE)
    action_actor = ScriptedActor(
        "actor:alpha",
        emit_action=True,
        output_evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )
    runtime, _ = make_runtime(
        (action_actor,),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha",)}),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=0),
    )
    trace = runtime.run(event)
    assert not trace.action_intents
    assert trace.rejected_actions[0].faults == (ActionFault.EVIDENCE_CLASS_DOWNGRADE,)

    endogenous_actor = ScriptedActor(
        "actor:alpha",
        endogenous_depth=1,
        output_evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )
    event, event_ledger = make_ledger_event(evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE)
    runtime, _ = make_runtime(
        (endogenous_actor,),
        ScriptedDispatchPolicy({(event.header.event_id, 0): ("actor:alpha",)}),
        caps=RuntimeCaps(K=1, C=1, B=1, H=0, M=1, R=1),
        event_ledger=event_ledger,
    )
    with pytest.raises(Exception, match="evidence-class taint"):
        runtime.run(event)
