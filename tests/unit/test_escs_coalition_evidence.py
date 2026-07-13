from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from mop.escs.accounting import LifecycleLedger, WorkVector
from mop.escs.actors import (
    ActorActivationContext,
    ActorActivationResult,
    ActorDescriptor,
    ActorUpdateContext,
    ActorUpdatePlan,
    DispatchEvent,
    DispatchEventHeader,
    ReadinessEstimate,
)
from mop.escs.coalition_evidence import (
    FORK_CONTRACT_SCHEMA,
    FORK_INTERVENTION_KEY,
    FORK_PROVENANCE_KEY,
    INTERVENTION_SCHEMA,
    SCRIPTED_OUTCOME_SCHEMA,
    TRACE_CONFIG_FRAME_SCHEMA,
    UTILITY_SCALARIZER_SCHEMA,
    ActorPerspectiveBinding,
    CoalitionEvidenceConfig,
    CoalitionEvidenceError,
    CoalitionFork,
    ExactCoalitionCredit,
    ForkAuthority,
    InteractionCreditSnapshot,
    ShadowCoalitionScore,
    assess_shadow_coalitions,
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
    state_version_for_parents,
)
from mop.escs.ledger import EventLedger
from mop.escs.messages import SchemaRegistry
from mop.escs.perspective_registry import (
    PerspectiveCandidateRegistry,
    load_perspective_candidate_registry,
)
from mop.escs.runtime import (
    CandidateMode,
    CoalitionRuntime,
    DispatchRequest,
    RoundTrace,
    RuntimeCaps,
    RuntimeConfig,
    RuntimeTrace,
    ScriptedDispatchPolicy,
)
from mop.escs.substrate_assembly import SubstrateAssembly, load_substrate_assembly
from mop.substrate.events import BranchRef, EventRef, canonical_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
ACTOR_A = "actor:planner"
ACTOR_B = "actor:checker"
ACTOR_C = "actor:abstractor"
ACTORS = (ACTOR_B, ACTOR_A)
UTILITY_KEY = "utility_milli"
CAPS = RuntimeCaps(K=2, C=2, B=4, H=0, M=0, R=0)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract(**overrides: object) -> dict[str, object]:
    scalarizer_sha256 = canonical_sha256(
        {
            "schema": UTILITY_SCALARIZER_SCHEMA,
            "utility_key": UTILITY_KEY,
            "utility_min_milli": 0,
            "utility_max_milli": 2_000,
            "utility_excludes_resource_cost": True,
        }
    )
    value: dict[str, object] = {
        "schema": FORK_CONTRACT_SCHEMA,
        "fork_group_id": "fork-group:shadow-credit-v1",
        "world_id": "world:fixture-v1",
        "horizon_start_tick": 0,
        "horizon_end_tick": 1,
        "source_state_sha256": _digest("source-state"),
        "environment_state_sha256": _digest("environment-state"),
        "runtime_id": _digest("runtime-authority"),
        "runtime_config_sha256": canonical_sha256(
            {
                "schema": TRACE_CONFIG_FRAME_SCHEMA,
                "mode": CandidateMode.BOUNDED_CENTRAL.value,
                "caps": CAPS.payload(),
            }
        ),
        "policy_state_sha256": _digest("policy-state"),
        "actor_state_versions_sha256": _digest("actor-state-versions"),
        "intervention_schema_sha256": canonical_sha256({"schema": INTERVENTION_SCHEMA}),
        "registered_actor_ids": list(ACTORS),
        "scripted_fixture_id": "fixture:coalition-utility-v1",
        "scripted_fixture_sha256": _digest("scripted-coalition-fixture"),
        "utility_key": UTILITY_KEY,
        "utility_min_milli": 0,
        "utility_max_milli": 2_000,
        "utility_scalarizer_sha256": scalarizer_sha256,
        "utility_excludes_resource_cost": True,
        "utility_source": "scripted-noncausal-fixture",
        "causal_effect_claim_allowed": False,
        "consequence_grounded_credit_claim_allowed": False,
    }
    value.update(overrides)
    return value


def _intervention(
    *,
    contract: dict[str, object],
    branch: BranchRef,
    coalition: tuple[str, ...],
    utility_milli: int,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    registered_value = contract["registered_actor_ids"]
    assert isinstance(registered_value, list) and all(isinstance(actor, str) for actor in registered_value)
    registered_actors = tuple(registered_value)
    value: dict[str, object] = {
        "schema": INTERVENTION_SCHEMA,
        "fork_group_id": contract["fork_group_id"],
        "branch_id": str(branch),
        "registered_actor_ids": list(registered_actors),
        "candidate_actor_ids": list(registered_actors),
        "active_actor_ids": list(coalition),
        "removed_actor_ids": [actor for actor in registered_actors if actor not in coalition],
        "actor_state_versions_sha256": contract["actor_state_versions_sha256"],
        "intervention_kind": "scripted-actor-removal",
        "intervention_schema_sha256": contract["intervention_schema_sha256"],
        "non_actor_inputs_held_fixed": True,
        "action_effect_authority": False,
        "scripted_utility_milli": utility_milli,
        "realized_full_cost": WorkVector.zero().payload(),
        "observation_uncertainty_micros": 0,
        "consequence_start_tick": 1,
        "consequence_end_tick": 1,
        "causal_effect_claim_allowed": False,
        "consequence_grounded_credit_claim_allowed": False,
    }
    value.update(overrides or {})
    value["intervention_sha256"] = canonical_sha256(value)
    return value


def _observation() -> ObservationEvent:
    return ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:coalition-fixture",),
        adapter_version="coalition-fixture/v1",
        sensor_scope={"sensor": "fixture"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"source": "fixture:sensor"},
    )


class _RuntimeActor:
    def __init__(self, actor_id: str) -> None:
        self._descriptor = ActorDescriptor(actor_id=actor_id, subscribed_event_types=("hypothesis",))

    @property
    def descriptor(self) -> ActorDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return _digest(f"runtime-state:{self.descriptor.actor_id}")

    @property
    def retained_state_bytes(self) -> int:
        return 0

    def readiness(self, header: DispatchEventHeader) -> ReadinessEstimate:
        return ReadinessEstimate(
            actor_id=self.descriptor.actor_id,
            state_version=self.state_version,
            compatible=True,
            expected_decision_value=0.0,
            predicted_operations=1,
            predicted_message_bytes=0,
        )

    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        return ActorActivationResult(executed_operations=3)

    def stage_update(self, context: ActorUpdateContext) -> ActorUpdatePlan:
        raise AssertionError("the slice fixture never grants update authority")


def test_native_runtime_produces_a_nonzero_bounded_ledger_slice() -> None:
    lifecycle = LifecycleLedger()
    policy = ScriptedDispatchPolicy({}, default=ACTORS)
    runtime = CoalitionRuntime(
        actors=tuple(_RuntimeActor(actor) for actor in ACTORS),
        policy=policy,
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(mode=CandidateMode.BOUNDED_CENTRAL, caps=CAPS),
        ledger=lifecycle,
        clock_ns=lambda: 1,
    )
    event = DispatchEvent.create(
        event_id="event:native-ledger-slice",
        event_kind="hypothesis",
        branch_id="branch:factual",
        producer_state_version=state_version_for_parents((EventRef("event:native-source"),)),
        epistemic_status=EpistemicStatus.INFERRED,
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        referent_hypotheses=("referent:item/1",),
        factor_scope=("factor:state",),
        routing_shards=("shard:coalition",),
        source_event_ids=("event:native-source",),
        created_tick=1,
        expiry_tick=2,
        payload_bytes=b"native-runtime-slice",
    )

    trace = runtime.run(event)

    assert trace.validate_integrity()
    assert 0 < trace.ledger_start_sequence < trace.ledger_end_sequence <= lifecycle.entry_count
    trace_rows = lifecycle.entries[trace.ledger_start_sequence : trace.ledger_end_sequence]
    assert {row.reason for row in trace_rows if row.reason.startswith("actor-activation:")} == {
        f"actor-activation:{actor}" for actor in ACTORS
    }


def _branch_name(coalition: tuple[str, ...]) -> str:
    labels = {ACTOR_A: "a", ACTOR_B: "b", ACTOR_C: "c"}
    suffix = "-".join(labels[actor] for actor in coalition) or "none"
    return f"branch:coalition-{suffix}"


def _authority_sequence(coalition: tuple[str, ...], registered_actors: tuple[str, ...]) -> int:
    subsets: list[tuple[str, ...]] = [()]
    subsets.extend((actor,) for actor in registered_actors)
    subsets.append(registered_actors)
    return subsets.index(coalition)


def _make_fork(
    observation: ObservationEvent,
    *,
    coalition: tuple[str, ...],
    utility_milli: int,
    contract_overrides: dict[str, object] | None = None,
    intervention_overrides: dict[str, object] | None = None,
    delayed_or_partial: bool = False,
    observed_overrides: dict[str, object] | None = None,
    source_factor_change: dict[str, object] | None = None,
    consequence_cost: WorkVector = WorkVector(),
    consequence_uncertainty: float = 0.0,
    consequence_evidence: EvidenceClass = EvidenceClass.ORACLE_NONPROMOTABLE,
    considered_coalitions: tuple[tuple[str, ...], ...] | None = None,
    staged_message_ids: tuple[str, ...] = (),
    activation_units: dict[str, int] | None = None,
    trace_charge_tick: int = 0,
    registered_actors: tuple[str, ...] = ACTORS,
) -> CoalitionFork:
    assert coalition == tuple(sorted(coalition))
    effective_contract_overrides = dict(contract_overrides or {})
    effective_contract_overrides["registered_actor_ids"] = list(registered_actors)
    contract = _contract(**effective_contract_overrides)
    branch = BranchRef(_branch_name(coalition))
    intervention = _intervention(
        contract=contract,
        branch=branch,
        coalition=coalition,
        utility_milli=utility_milli,
        overrides=intervention_overrides,
    )
    source = HypothesisEvent.create(
        origin=HypothesisOrigin.EVENT_FORMER,
        epistemic_status=EpistemicStatus.SIMULATED,
        referent_hypotheses={"referent:item/1": 1.0},
        factor_change_distribution=source_factor_change or {"factor:state": 1.0},
        decision_relevance_distribution={"relevant": 1.0},
        reducibility_distribution={"reducible": 1.0},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.0,
        causal_parent_ids=(observation.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={
            FORK_PROVENANCE_KEY: contract,
            FORK_INTERVENTION_KEY: intervention,
        },
    )
    header = DispatchEventHeader(
        event_id=str(source.event_id),
        event_kind="hypothesis",
        branch_id=str(branch),
        producer_state_version=source.envelope.producer_state_version,
        epistemic_status=EpistemicStatus.SIMULATED,
        evidence_class=source.evidence_class,
        referent_hypotheses=("referent:item/1",),
        factor_scope=("factor:state",),
        routing_shards=("shard:coalition",),
        source_event_ids=(str(observation.event_id),),
        created_tick=0,
        expiry_tick=1,
        payload_digest=source.envelope.payload_digest,
        representation_payload_digest=_digest("shared-hypothesis-representation"),
    )
    lifecycle = LifecycleLedger()
    lifecycle.charge(
        owner="fixture:unrelated-prefix",
        reason=f"actor-activation:{ACTOR_A}",
        work=WorkVector(actor_execution=50_000),
        start_tick=0,
        end_tick=0,
    )
    trace_start = lifecycle.entry_count
    lifecycle.charge(
        owner="escs.shadow-fork",
        reason="fork-dispatch",
        work=WorkVector(dispatch_and_exploration=2),
        start_tick=trace_charge_tick,
        end_tick=trace_charge_tick,
        branch_id=branch,
        causal_event_ids=(source.event_id,),
    )
    actor_units = activation_units or {ACTOR_A: 10, ACTOR_B: 20, ACTOR_C: 15}
    for actor_id in coalition:
        lifecycle.charge(
            owner=actor_id,
            reason=f"actor-activation:{actor_id}",
            work=WorkVector(actor_execution=actor_units[actor_id]),
            start_tick=trace_charge_tick,
            end_tick=trace_charge_tick,
            branch_id=branch,
            causal_event_ids=(source.event_id,),
        )
    sequence = _authority_sequence(coalition, registered_actors)
    runtime_id = contract["runtime_id"]
    assert isinstance(runtime_id, str)
    trace_id = canonical_sha256(
        {
            "schema": "mop-escs-runtime-trace-authority/v1",
            "runtime_id": runtime_id,
            "authority_sequence": sequence,
        }
    )
    trace = RuntimeTrace(
        trace_id=trace_id,
        runtime_id=runtime_id,
        authority_sequence=sequence,
        mode=CandidateMode.BOUNDED_CENTRAL,
        caps=CAPS,
        initial_event_id=str(source.event_id),
        rounds=(
            RoundTrace(
                event_header=header,
                candidate_actor_ids=registered_actors,
                selected_actor_ids=coalition,
                considered_coalitions=considered_coalitions or (coalition,),
                staged_message_ids=staged_message_ids,
                consumed_message_ids=(),
                accepted_action_ids=(),
                admitted_endogenous_event_ids=(),
            ),
        ),
        message_deliveries=(),
        rejected_claims=(),
        action_intents=(),
        rejected_actions=(),
        active_actor_ids=coalition,
        endogenous_rounds=0,
        quiescent=True,
        halt_reason="quiescent",
        ledger_start_sequence=trace_start,
        ledger_end_sequence=lifecycle.entry_count,
        full_trace_sha256=_digest("placeholder"),
    )
    trace = replace(trace, full_trace_sha256=canonical_sha256(trace.payload(include_digest=False)))
    effect_id = canonical_sha256(
        {
            "schema": "mop-escs-effect-authority/v1",
            "hypothesis_event_id": str(source.event_id),
            "action_id": None,
            "trace_authority_id": trace.trace_id,
        }
    )
    commitment = CommitmentEvent.create(
        coalition_id=f"coalition:{trace.trace_id}",
        commitment_kind=CommitmentKind.ABSTENTION,
        committed_payload={
            "schema": "mop-escs-chassis-commitment/v1",
            "hypothesis_event_id": str(source.event_id),
            "runtime_id": trace.runtime_id,
            "trace_authority_sequence": trace.authority_sequence,
            "trace_authority_id": trace.trace_id,
            "full_trace_sha256": trace.full_trace_sha256,
            "effect_id": effect_id,
            "decision_reason": "simulated-hypothesis-external-effect-refused",
            "action_record": None,
            "blocked_action_id": None,
        },
        decision_distribution={"abstention": 1.0},
        deadline_tick=0,
        predicted_utility_vector={"unscored": 0.0},
        predicted_full_cost=WorkVector.zero(),
        causal_parent_ids=(source.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={
            "producer": "escs.chassis",
            "trace_authority_id": trace.trace_id,
        },
        measured_creation_cost=WorkVector(event_formation=1),
    )
    observed: dict[str, object] = {
        "schema": SCRIPTED_OUTCOME_SCHEMA,
        "scripted_fixture_id": contract["scripted_fixture_id"],
        "scripted_fixture_sha256": contract["scripted_fixture_sha256"],
        "intervention_sha256": intervention["intervention_sha256"],
        "utility_scalarizer_sha256": contract["utility_scalarizer_sha256"],
        "hypothesis_event_id": str(source.event_id),
        "trace_authority_id": trace.trace_id,
        "effect_id": effect_id,
        "scripted_utility_milli": utility_milli,
        "causal_effect_claim_allowed": False,
        "consequence_grounded_credit_claim_allowed": False,
    }
    observed.update(observed_overrides or {})
    consequence = ConsequenceEvent.create(
        commitment_event_id=commitment.event_id,
        observed_outcome=observed,
        realized_utility_vector={UTILITY_KEY: utility_milli},
        delayed_or_partial=delayed_or_partial,
        observation_uncertainty=consequence_uncertainty,
        realized_full_cost=consequence_cost,
        causal_parent_ids=(commitment.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={
            "producer": "escs.scripted-coalition-utility-fixture",
            "scripted_fixture_id": contract["scripted_fixture_id"],
            "scripted_fixture_sha256": contract["scripted_fixture_sha256"],
            "intervention_sha256": intervention["intervention_sha256"],
            "utility_scalarizer_sha256": contract["utility_scalarizer_sha256"],
        },
        measured_creation_cost=WorkVector(event_formation=1),
        evidence_class=consequence_evidence,
    )
    events = EventLedger()
    events.append_batch((observation, source, commitment, consequence))
    lifecycle.charge(
        owner="escs.chassis",
        reason="chassis-commitment-formation-and-indexing",
        work=WorkVector(
            event_formation=1,
            indexing_and_graph_maintenance=1 + len(canonical_bytes(commitment.payload())),
        ),
        start_tick=0,
        end_tick=0,
        branch_id=branch,
        causal_event_ids=(source.event_id,),
    )
    lifecycle.charge(
        owner="escs.chassis",
        reason="scripted-fixture-consequence-formation-and-indexing",
        work=WorkVector(
            event_formation=1,
            indexing_and_graph_maintenance=1 + len(canonical_bytes(consequence.payload())),
        ),
        start_tick=1,
        end_tick=1,
        branch_id=branch,
        causal_event_ids=(commitment.event_id,),
    )
    lifecycle.charge(
        owner="fixture:unrelated-suffix",
        reason=f"actor-activation:{ACTOR_B}",
        work=WorkVector(actor_execution=50_000),
        start_tick=1,
        end_tick=1,
    )
    return CoalitionFork.create(
        source_hypothesis=source,
        trace=trace,
        commitment=commitment,
        consequence=consequence,
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        utility_key=UTILITY_KEY,
    )


@dataclass(frozen=True)
class EvidenceFixture:
    registry: PerspectiveCandidateRegistry
    assembly: SubstrateAssembly
    config: CoalitionEvidenceConfig
    bindings: tuple[ActorPerspectiveBinding, ...]
    forks: tuple[CoalitionFork, ...]
    authority: ForkAuthority
    credit: ExactCoalitionCredit
    snapshot: InteractionCreditSnapshot


@pytest.fixture(scope="module")
def evidence() -> EvidenceFixture:
    registry = load_perspective_candidate_registry(
        ROOT / "configs/experiment/escs_perspective_candidates.json"
    )
    assembly = load_substrate_assembly(ROOT / "configs/experiment/escs_substrate_assembly.json")
    config = CoalitionEvidenceConfig(
        candidate_registry_sha256=registry.sha256,
        assembly_sha256=assembly.assembly_sha256,
        max_actors=2,
        max_forks=4,
        max_beam=4,
        max_work_units=2_000_000,
    )
    bindings = tuple(
        sorted(
            (
                ActorPerspectiveBinding.create(ACTOR_A, "planning", assembly),
                ActorPerspectiveBinding.create(ACTOR_B, "contradiction_detection", assembly),
            ),
            key=lambda row: row.actor_id,
        )
    )
    observation = _observation()
    forks = (
        _make_fork(observation, coalition=(), utility_milli=400),
        _make_fork(observation, coalition=(ACTOR_A,), utility_milli=600),
        _make_fork(observation, coalition=(ACTOR_B,), utility_milli=550),
        _make_fork(observation, coalition=ACTORS, utility_milli=1_000),
    )
    authority = ForkAuthority.create(
        config=config,
        registry=registry,
        assembly=assembly,
        bindings=bindings,
        forks=forks,
        full_coalition_actor_ids=ACTORS,
    )
    credit = ExactCoalitionCredit.derive(authority)
    snapshot = InteractionCreditSnapshot.create(
        (credit,),
        authorities=(authority,),
        config=config,
        fit_tick=2,
    )
    return EvidenceFixture(registry, assembly, config, bindings, forks, authority, credit, snapshot)


def _request(
    *,
    event_id: str = "event:shadow-request",
    source_event_id: str = "event:shadow-source",
    created_tick: int = 3,
    a_expected_value: float = 1_000_000.0,
    b_expected_value: float = -1_000_000.0,
    predicted_operations: int = 5,
    payload_digest: str | None = None,
    actor_ids: tuple[str, ...] = ACTORS,
    stale_state_risk: float = 0.0,
    deadline_risk: float = 0.0,
) -> DispatchRequest:
    parent = EventRef(source_event_id)
    header = DispatchEventHeader(
        event_id=event_id,
        event_kind="hypothesis",
        branch_id="branch:factual",
        producer_state_version=state_version_for_parents((parent,)),
        epistemic_status=EpistemicStatus.INFERRED,
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        referent_hypotheses=("referent:item/2",),
        factor_scope=("factor:state",),
        routing_shards=("shard:coalition",),
        source_event_ids=(str(parent),),
        created_tick=created_tick,
        expiry_tick=10,
        payload_digest=payload_digest or _digest("request-payload"),
        representation_payload_digest=_digest("request-representation"),
    )
    readiness = tuple(
        ReadinessEstimate(
            actor_id=actor_id,
            state_version=_digest(f"request-state:{actor_id}"),
            compatible=True,
            expected_decision_value=(
                a_expected_value if actor_id == ACTOR_A else b_expected_value if actor_id == ACTOR_B else 0.0
            ),
            predicted_operations=predicted_operations,
            predicted_message_bytes=0,
            stale_state_risk=stale_state_risk,
            deadline_risk=deadline_risk,
        )
        for actor_id in actor_ids
    )
    return DispatchRequest(
        event_header=header,
        readiness=readiness,
        mode=CandidateMode.BOUNDED_CENTRAL,
        caps=RuntimeCaps(K=2, C=2, B=4, H=0, M=0, R=0),
        reasoning_round=0,
    )


def test_complete_same_state_forks_derive_exact_difference_and_interaction_credit(
    evidence: EvidenceFixture,
) -> None:
    credit = evidence.credit

    assert dict(credit.resource_debit_milli) == {ACTOR_A: 10, ACTOR_B: 20}
    assert dict(credit.individual_difference_credit_milli) == {ACTOR_A: 440, ACTOR_B: 380}
    assert dict(credit.singleton_main_effect_milli) == {ACTOR_A: 200, ACTOR_B: 150}
    assert credit.pair_interaction_milli == ((ACTOR_B, ACTOR_A, 250),)
    assert credit.full_utility_milli == 1_000
    assert credit.accounting.stage == "derive-exact-coalition-credit"
    assert credit.accounting.work.counterfactual_credit > 0
    assert credit.accounting.charge_applied is False
    full_fork = next(fork for fork in evidence.forks if fork.coalition_actor_ids == ACTORS)
    assert full_fork.trace.ledger_start_sequence == 1
    assert dict(full_fork.actor_work) == {
        ACTOR_A: WorkVector(actor_execution=10),
        ACTOR_B: WorkVector(actor_execution=20),
    }
    assert evidence.snapshot.retained_state_bytes == len(canonical_bytes(evidence.snapshot.payload()))
    assert evidence.snapshot.difference_credit_value(ACTOR_A) == 440
    assert evidence.snapshot.main_effect_value(ACTOR_A) == 200
    assert evidence.snapshot.authority_actor_orders == ((evidence.authority.authority_sha256, ACTORS),)
    assert evidence.snapshot.pairwise_source_exact is True
    same_small_request_from_higher_order_source = ShadowCoalitionScore.create(
        actor_ids=ACTORS,
        candidate_ids=("contradiction_detection", "planning"),
        main_effect=Fraction(350),
        interaction=Fraction(250),
        debit=10,
        pairwise_source_exact=False,
    )
    assert same_small_request_from_higher_order_source.pairwise_reconstruction_exact is False
    assert credit.activation_enabled is False
    assert credit.scientific_promotion_allowed is False


def test_fork_authority_rejects_missing_duplicate_and_state_drift(
    evidence: EvidenceFixture,
) -> None:
    extra_binding = ActorPerspectiveBinding.create(ACTOR_C, "abstraction", evidence.assembly)
    with pytest.raises(CoalitionEvidenceError, match="exactly cover"):
        ForkAuthority.create(
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=(*evidence.bindings, extra_binding),
            forks=evidence.forks,
            full_coalition_actor_ids=ACTORS,
        )
    with pytest.raises(CoalitionEvidenceError, match="trace actors do not match"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            intervention_overrides={
                "active_actor_ids": [ACTOR_A],
                "removed_actor_ids": [ACTOR_B],
            },
        )
    with pytest.raises(CoalitionEvidenceError, match="actor state authority drift"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            intervention_overrides={"actor_state_versions_sha256": _digest("drifted-actor-state")},
        )
    with pytest.raises(CoalitionEvidenceError, match="incomplete"):
        ForkAuthority.create(
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            forks=evidence.forks[:-1],
            full_coalition_actor_ids=ACTORS,
        )
    with pytest.raises(CoalitionEvidenceError, match="duplicate"):
        ForkAuthority.create(
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            forks=(*evidence.forks[:-1], evidence.forks[0]),
            full_coalition_actor_ids=ACTORS,
        )

    drifted = _make_fork(
        _observation(),
        coalition=ACTORS,
        utility_milli=1_000,
        contract_overrides={"environment_state_sha256": _digest("different-environment")},
    )
    with pytest.raises(CoalitionEvidenceError, match="contracts drifted|normalized source state"):
        ForkAuthority.create(
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            forks=(*evidence.forks[:-1], drifted),
            full_coalition_actor_ids=ACTORS,
        )


def test_fork_rejects_nonfinal_future_tainted_or_unjoined_fixture_consequence() -> None:
    with pytest.raises(CoalitionEvidenceError, match="considered coalition escapes"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            considered_coalitions=(("actor:intruder",), ACTORS),
        )
    with pytest.raises(CoalitionEvidenceError, match="round-local"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            staged_message_ids=("message:unjoined",),
        )
    with pytest.raises(CoalitionEvidenceError, match="charge clocks"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            trace_charge_tick=1,
        )
    with pytest.raises(CoalitionEvidenceError, match="max_episode_work"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            activation_units={ACTOR_A: CAPS.max_episode_work, ACTOR_B: 1},
        )
    with pytest.raises(CoalitionEvidenceError, match="partial consequence"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            delayed_or_partial=True,
        )
    with pytest.raises(CoalitionEvidenceError, match="evaluator/future-only"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            source_factor_change={"nested": {"Future-Outcome": "forbidden"}},
        )
    with pytest.raises(CoalitionEvidenceError, match="evaluator/future-only"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            observed_overrides={"futureOutcome": "forbidden"},
        )
    for separatorless in ("futureoutcome", "groundtruth", "oraclelabel"):
        with pytest.raises(CoalitionEvidenceError, match="evaluator/future-only"):
            _make_fork(
                _observation(),
                coalition=ACTORS,
                utility_milli=1_000,
                source_factor_change={"nested": {separatorless: "forbidden"}},
            )
    with pytest.raises(CoalitionEvidenceError, match="uncertainty/cost authority"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            consequence_cost=WorkVector(actor_execution=1),
        )
    with pytest.raises(CoalitionEvidenceError, match="uncertainty/cost authority"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            consequence_uncertainty=0.1,
        )
    with pytest.raises(CoalitionEvidenceError, match="mechanics/evidence authority"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            consequence_evidence=EvidenceClass.SCRIPTED_MECHANICS,
        )
    with pytest.raises(CoalitionEvidenceError, match="scalarizer authority"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            contract_overrides={"utility_scalarizer_sha256": _digest("forged-scalarizer")},
        )
    with pytest.raises(CoalitionEvidenceError, match="does not equal"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            observed_overrides={"scripted_utility_milli": 999},
        )
    with pytest.raises(CoalitionEvidenceError, match="gained a causal claim"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            intervention_overrides={"causal_effect_claim_allowed": True},
        )
    with pytest.raises(CoalitionEvidenceError, match="gained a causal claim"):
        _make_fork(
            _observation(),
            coalition=ACTORS,
            utility_milli=1_000,
            intervention_overrides={"action_effect_authority": True},
        )


def test_delayed_snapshot_rejects_future_and_tampered_credit(evidence: EvidenceFixture) -> None:
    with pytest.raises(CoalitionEvidenceError, match="future credit"):
        InteractionCreditSnapshot.create(
            (evidence.credit,),
            authorities=(evidence.authority,),
            config=evidence.config,
            fit_tick=0,
        )
    with pytest.raises(CoalitionEvidenceError, match="self-hash mismatch"):
        replace(evidence.credit, full_utility_milli=1_001)

    forged_main = ((ACTOR_B, 999_999_999), (ACTOR_A, 999_999_999))
    forged_payload = evidence.credit.payload(include_digest=False)
    forged_payload["singleton_main_effect_milli"] = dict(forged_main)
    forged = ExactCoalitionCredit(
        authority_sha256=evidence.credit.authority_sha256,
        config_sha256=evidence.credit.config_sha256,
        candidate_registry_sha256=evidence.credit.candidate_registry_sha256,
        assembly_sha256=evidence.credit.assembly_sha256,
        fork_group_id=evidence.credit.fork_group_id,
        full_coalition_actor_ids=evidence.credit.full_coalition_actor_ids,
        full_utility_milli=evidence.credit.full_utility_milli,
        singleton_main_effect_milli=forged_main,
        individual_difference_credit_milli=evidence.credit.individual_difference_credit_milli,
        pair_interaction_milli=evidence.credit.pair_interaction_milli,
        resource_debit_milli=evidence.credit.resource_debit_milli,
        available_tick=evidence.credit.available_tick,
        training_event_ids=evidence.credit.training_event_ids,
        training_payload_sha256s=evidence.credit.training_payload_sha256s,
        evidence_class=evidence.credit.evidence_class,
        accounting=evidence.credit.accounting,
        scripted_fixture_only=True,
        causal_effect_claim_allowed=False,
        consequence_grounded_credit_claim_allowed=False,
        activation_enabled=False,
        scientific_promotion_allowed=False,
        credit_sha256=canonical_sha256(forged_payload),
        _validation_token=evidence.credit._validation_token,
    )
    with pytest.raises(CoalitionEvidenceError, match="does not replay exactly"):
        InteractionCreditSnapshot.create(
            (forged,),
            authorities=(evidence.authority,),
            config=evidence.config,
            fit_tick=2,
        )
    with pytest.raises(CoalitionEvidenceError, match="one-to-one snapshot coverage"):
        assess_shadow_coalitions(
            _request(),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(forged,),
            authorities=(evidence.authority,),
        )

    original_binding = evidence.bindings[0]
    forged_binding_payload = original_binding.payload(include_digest=False)
    forged_binding_payload["candidate_sha256"] = _digest("registry-splice")
    forged_binding = ActorPerspectiveBinding(
        actor_id=original_binding.actor_id,
        candidate_id=original_binding.candidate_id,
        candidate_sha256=forged_binding_payload["candidate_sha256"],
        facet=original_binding.facet,
        mode=original_binding.mode,
        trigger_authority=original_binding.trigger_authority,
        binding_sha256=canonical_sha256(forged_binding_payload),
    )
    forged_bindings = tuple(sorted((forged_binding, evidence.bindings[1]), key=lambda row: row.actor_id))
    forged_authority_payload = evidence.authority.payload(include_digest=False)
    forged_authority_payload["binding_sha256s"] = [row.binding_sha256 for row in forged_bindings]
    forged_authority = ForkAuthority(
        config=evidence.authority.config,
        bindings=forged_bindings,
        forks=evidence.authority.forks,
        full_coalition_actor_ids=evidence.authority.full_coalition_actor_ids,
        registered_actor_pairs=evidence.authority.registered_actor_pairs,
        fork_group_id=evidence.authority.fork_group_id,
        normalized_state_sha256=evidence.authority.normalized_state_sha256,
        scripted_fixture_only=True,
        causal_effect_claim_allowed=False,
        consequence_grounded_credit_claim_allowed=False,
        authority_sha256=canonical_sha256(forged_authority_payload),
        _validation_token=evidence.authority._validation_token,
    )
    forged_authority_credit = ExactCoalitionCredit.derive(forged_authority)
    forged_authority_snapshot = InteractionCreditSnapshot.create(
        (forged_authority_credit,),
        authorities=(forged_authority,),
        config=evidence.config,
        fit_tick=2,
    )
    with pytest.raises(CoalitionEvidenceError, match="binding candidate digest drift"):
        assess_shadow_coalitions(
            _request(),
            snapshot=forged_authority_snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(forged_authority_credit,),
            authorities=(forged_authority,),
        )


def test_shadow_arbitration_is_deterministic_header_only_and_never_authority(
    evidence: EvidenceFixture,
) -> None:
    first = assess_shadow_coalitions(
        _request(),
        snapshot=evidence.snapshot,
        config=evidence.config,
        registry=evidence.registry,
        assembly=evidence.assembly,
        bindings=evidence.bindings,
        credits=(evidence.credit,),
        authorities=(evidence.authority,),
    )
    reversed_claims = assess_shadow_coalitions(
        _request(a_expected_value=-9_000_000.0, b_expected_value=9_000_000.0),
        snapshot=evidence.snapshot,
        config=evidence.config,
        registry=evidence.registry,
        assembly=evidence.assembly,
        bindings=evidence.bindings,
        credits=(evidence.credit,),
        authorities=(evidence.authority,),
    )

    assert first.proposed_actor_ids == ACTORS
    assert first.scripted_interaction_term_positive is True
    assert first.pairwise_reconstruction_exact is True
    assert first.scripted_fixture_only is True
    assert first.causal_effect_claim_allowed is False
    assert first.consequence_grounded_credit_claim_allowed is False
    assert first.cooperation_claim_allowed is False
    assert first.consumable_by_runtime is False
    assert first.source_replay_verified is True
    assert [row.actor_ids for row in first.scores] == [row.actor_ids for row in reversed_claims.scores]
    assert [row.net_value for row in first.scores] == [row.net_value for row in reversed_claims.scores]
    assert first.accounting.work.dispatch_and_exploration > 0
    assert first.accounting.charge_applied is False
    assert {
        "accounting-unapplied",
        "cooperation-claim-not-authorized",
        "no-action-effect-authority",
        "scripted-noncausal-fixture-only",
        "shadow-only",
    } <= set(first.blockers)
    pair_score = next(row for row in first.scores if row.actor_ids == ACTORS)
    assert pair_score.main_effect_value_numerator == 350
    assert pair_score.interaction_value_numerator == 250
    assert pair_score.resource_debit_milli == 10
    assert pair_score.net_value == 590
    assert not hasattr(first, "select")
    assert not hasattr(first, "activate")
    assert not hasattr(first, "to_dispatch_decision")
    assert all(
        flag is False
        for flag in (
            first.applied,
            first.activation_enabled,
            first.dispatch_authority,
            first.commitment_authority,
            first.effect_authority,
            first.update_authority,
            first.scientific_promotion_allowed,
        )
    )


def test_pairwise_exactness_requires_a_joint_source_domain(evidence: EvidenceFixture) -> None:
    ac_actors = tuple(sorted((ACTOR_A, ACTOR_C)))
    observation = _observation()
    ac_forks = (
        _make_fork(observation, coalition=(), utility_milli=400, registered_actors=ac_actors),
        _make_fork(
            observation,
            coalition=(ACTOR_A,),
            utility_milli=600,
            registered_actors=ac_actors,
        ),
        _make_fork(
            observation,
            coalition=(ACTOR_C,),
            utility_milli=700,
            registered_actors=ac_actors,
        ),
        _make_fork(
            observation,
            coalition=ac_actors,
            utility_milli=1_100,
            registered_actors=ac_actors,
        ),
    )
    binding_a = next(row for row in evidence.bindings if row.actor_id == ACTOR_A)
    binding_c = ActorPerspectiveBinding.create(ACTOR_C, "abstraction", evidence.assembly)
    ac_bindings = tuple(sorted((binding_a, binding_c), key=lambda row: row.actor_id))
    ac_authority = ForkAuthority.create(
        config=evidence.config,
        registry=evidence.registry,
        assembly=evidence.assembly,
        bindings=ac_bindings,
        forks=ac_forks,
        full_coalition_actor_ids=ac_actors,
    )
    ac_credit = ExactCoalitionCredit.derive(ac_authority)
    mixed_snapshot = InteractionCreditSnapshot.create(
        (evidence.credit, ac_credit),
        authorities=(evidence.authority, ac_authority),
        config=evidence.config,
        fit_tick=2,
    )
    assert mixed_snapshot.pairwise_reconstruction_exact_for(ACTORS) is False
    all_bindings = tuple(sorted((*evidence.bindings, binding_c), key=lambda row: row.actor_id))
    bc_actors = tuple(sorted((ACTOR_B, ACTOR_C)))

    proposal = assess_shadow_coalitions(
        _request(actor_ids=bc_actors),
        snapshot=mixed_snapshot,
        config=evidence.config,
        registry=evidence.registry,
        assembly=evidence.assembly,
        bindings=all_bindings,
        credits=(evidence.credit, ac_credit),
        authorities=(evidence.authority, ac_authority),
    )

    bc_score = next(row for row in proposal.scores if row.actor_ids == bc_actors)
    assert bc_score.interaction_value_numerator == 0
    assert bc_score.pairwise_reconstruction_exact is False
    assert proposal.proposed_actor_ids == bc_actors
    assert proposal.pairwise_reconstruction_exact is False
    assert "higher-order-interactions-unmodeled" in proposal.blockers


def test_shadow_arbitration_abstains_on_cost_and_rejects_temporal_leakage(
    evidence: EvidenceFixture,
) -> None:
    with pytest.raises(CoalitionEvidenceError, match="readiness-risk bound"):
        assess_shadow_coalitions(
            _request(stale_state_risk=1e308),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
    with pytest.raises(CoalitionEvidenceError, match="one-to-one snapshot coverage"):
        assess_shadow_coalitions(
            _request(),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority, evidence.authority),
        )
    costly = assess_shadow_coalitions(
        _request(predicted_operations=10_000),
        snapshot=evidence.snapshot,
        config=evidence.config,
        registry=evidence.registry,
        assembly=evidence.assembly,
        bindings=evidence.bindings,
        credits=(evidence.credit,),
        authorities=(evidence.authority,),
    )
    assert costly.proposed_actor_ids == ()
    assert "no-positive-net-value" in costly.blockers

    first_fork = evidence.forks[0]
    used_event_ids = {
        str(first_fork.source_hypothesis.event_id),
        *(str(value) for value in first_fork.source_hypothesis.envelope.causal_parent_ids),
        str(first_fork.commitment.event_id),
        str(first_fork.consequence.event_id),
    }
    assert used_event_ids <= set(evidence.snapshot.training_event_ids)
    training_event = evidence.snapshot.training_event_ids[0]
    with pytest.raises(CoalitionEvidenceError, match="training-event credit"):
        assess_shadow_coalitions(
            _request(event_id=training_event),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
    with pytest.raises(CoalitionEvidenceError, match="training-event credit"):
        assess_shadow_coalitions(
            _request(event_id=str(first_fork.consequence.event_id)),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
    training_observation = str(evidence.forks[0].source_hypothesis.envelope.causal_parent_ids[0])
    with pytest.raises(CoalitionEvidenceError, match="training ancestry"):
        assess_shadow_coalitions(
            _request(source_event_id=training_observation),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
    with pytest.raises(CoalitionEvidenceError, match="training payload"):
        assess_shadow_coalitions(
            _request(payload_digest=evidence.snapshot.training_payload_sha256s[0]),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
    with pytest.raises(CoalitionEvidenceError, match="not later"):
        assess_shadow_coalitions(
            _request(created_tick=2),
            snapshot=evidence.snapshot,
            config=evidence.config,
            registry=evidence.registry,
            assembly=evidence.assembly,
            bindings=evidence.bindings,
            credits=(evidence.credit,),
            authorities=(evidence.authority,),
        )
