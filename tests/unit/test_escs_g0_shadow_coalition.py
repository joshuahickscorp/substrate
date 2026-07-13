from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from mop.config import REPO_ROOT
from mop.escs.g0_evaluator import G0CounterfactualEvaluation, G0CounterfactualRefusal
from mop.escs.g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    G0StateSlot,
)
from mop.escs.perspective_registry import load_perspective_candidate_registry
from mop.escs.topology_grammar import (
    OperatorPrimitive,
    StatePrimitive,
    load_topology_grammar,
)
from mop.studies import escs_g0_shadow_coalition as shadow_module
from mop.studies.escs_g0_construction import G0ConstructionSnapshot
from mop.studies.escs_g0_shadow_coalition import (
    G0ShadowActorState,
    G0ShadowCaps,
    G0ShadowDeliveryKind,
    G0ShadowEpisode,
    G0ShadowPortBinding,
    G0ShadowSeed,
    G0ShadowTerminalReason,
    execute_g0_shadow_coalition,
    verify_g0_shadow_trace,
)
from mop.substrate.events import canonical_bytes

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _context():
    return load_topology_grammar(GRAMMAR_PATH), load_perspective_candidate_registry(REGISTRY_PATH)


def _recurrent_actor(
    candidate_id: str,
    *,
    recipient: str | None = None,
    message_schema: str = "claim-v1",
) -> G0ActorGenotype:
    state = G0StateSlot(
        slot_id="memory",
        primitive=StatePrimitive.BOUNDED_RECURRENT_STATE,
        schema_id="memory-v1",
        capacity_bytes=64,
    )
    update = G0OperatorNode.create(
        node_id="update",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        state_slot_ids=(state.slot_id,),
        parameters={
            "weights": [[1.0, 1.0]],
            "bias": [0.0],
            "activation": "identity",
            "state_mode": "concatenate",
        },
        declared_operations=4,
        max_output_bytes=64,
    )
    if recipient is None:
        return G0ActorGenotype.create(
            candidate_id=candidate_id,
            state_slots=(state,),
            operator_nodes=(update,),
            output_node_ids=(update.node_id,),
        )
    edge = G0MessageEdge("out", message_schema, 64)
    emit = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=(update.node_id,),
        message_edge_ids=(edge.edge_id,),
        parameters={
            "schema_id": edge.schema_id,
            "recipient": f"actor:{recipient}",
            "payload_form": "numeric-vector",
        },
        declared_operations=1,
        max_output_bytes=64,
    )
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=(state,),
        operator_nodes=(update, emit),
        output_node_ids=(emit.node_id,),
        message_edges=(edge,),
    )


def _stateless_self_actor(candidate_id: str) -> G0ActorGenotype:
    edge = G0MessageEdge("loop", "loop-v1", 64)
    identity = G0OperatorNode.create(
        node_id="identity",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        parameters={
            "weights": [[1.0]],
            "bias": [0.0],
            "activation": "identity",
            "state_mode": "none",
        },
        declared_operations=2,
        max_output_bytes=64,
    )
    emit = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=(identity.node_id,),
        message_edge_ids=(edge.edge_id,),
        parameters={
            "schema_id": edge.schema_id,
            "recipient": f"actor:{candidate_id}",
            "payload_form": "numeric-vector",
        },
        declared_operations=1,
        max_output_bytes=64,
    )
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=(),
        operator_nodes=(identity, emit),
        output_node_ids=(emit.node_id,),
        message_edges=(edge,),
    )


def _snapshot(*actors: G0ActorGenotype) -> G0ConstructionSnapshot:
    return G0ConstructionSnapshot.create(
        actors=actors,
        factor_scopes={actor.candidate_id: (f"scope.{actor.candidate_id}",) for actor in actors},
    )


def _caps(source: G0ConstructionSnapshot, **overrides: int) -> G0ShadowCaps:
    values = {
        "max_rounds": 8,
        "max_activations": 16,
        "max_queue_depth": 16,
        "max_messages": 16,
        "max_actor_operations": 256,
        "max_routed_payload_bytes": 4096,
        "max_retained_state_bytes": max(1, source.retained_state_bytes),
        "max_repeated_state_visits": 1,
    }
    values.update(overrides)
    return G0ShadowCaps(**values)


def _episode(
    source: G0ConstructionSnapshot,
    *,
    seed_payload: object = None,
    receiver_schema: str = "claim-v1",
    caps: G0ShadowCaps | None = None,
) -> G0ShadowEpisode:
    grammar, registry = _context()
    by_id = {actor.candidate_id: actor for actor in source.actors}
    ports = []
    states = []
    for actor_id, actor in sorted(by_id.items()):
        root = next(node.node_id for node in actor.operator_nodes if not node.input_node_ids)
        schema = (
            "stimulus-v1"
            if actor_id == "planning"
            else receiver_schema
            if actor_id == "verification"
            else "loop-v1"
        )
        ports.append(
            G0ShadowPortBinding(
                actor_id=actor_id,
                genotype_sha256=actor.genotype_sha256,
                root_node_id=root,
                schema_id=schema,
                payload_form="numeric-vector",
                max_encoded_bytes=64,
            )
        )
        states.append(
            G0ShadowActorState.create(
                actor_id,
                {slot.slot_id: [0.0] for slot in actor.state_slots},
            )
        )
    seed_actor = "planning" if "planning" in by_id else next(iter(sorted(by_id)))
    seed_schema = next(row.schema_id for row in ports if row.actor_id == seed_actor)
    seed = G0ShadowSeed.create(
        seed_id="root",
        actor_id=seed_actor,
        schema_id=seed_schema,
        payload_form="numeric-vector",
        payload=[1.0] if seed_payload is None else seed_payload,
    )
    return G0ShadowEpisode.create(
        episode_id="test-episode",
        source=source,
        grammar=grammar,
        candidate_registry=registry,
        ports=ports,
        initial_states=states,
        seeds=(seed,),
        caps=caps or _caps(source),
    )


def _chain():
    planning = _recurrent_actor("planning", recipient="verification")
    verification = _recurrent_actor("verification")
    return _snapshot(planning, verification)


def test_two_actor_message_is_next_round_stateful_and_exactly_replayable() -> None:
    grammar, registry = _context()
    source = _chain()
    episode = _episode(source)

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.QUIESCENT
    assert [row.actor_id for row in trace.activations] == ["planning", "verification"]
    assert [row.round_index for row in trace.activations] == [0, 1]
    assert [row.kind for row in trace.deliveries] == [
        G0ShadowDeliveryKind.SEED,
        G0ShadowDeliveryKind.MESSAGE,
    ]
    assert trace.deliveries[1].available_round == 1
    assert trace.routed_message_count == trace.produced_message_count == 1
    final = {row.actor_id: row.state.value() for row in trace.final_states}
    assert final == {"planning": {"memory": [1.0]}, "verification": {"memory": [1.0]}}
    assert trace.work.actor_execution == sum(row.work.actor_execution for row in trace.activations)
    assert trace.work.retained_byte_time == source.retained_state_bytes * 2
    assert (
        trace.source_snapshot_sha256
        == trace.effective_snapshot_sha256
        == trace.rollback_snapshot_sha256
        == source.snapshot_sha256
    )
    assert not any(
        (
            trace.activation_enabled,
            trace.shadow_execution_authorized,
            trace.factual_effects,
            trace.factual_mutation_authorized,
            trace.scientific_promotion_allowed,
        )
    )
    assert (
        verify_g0_shadow_trace(
            trace,
            source,
            episode,
            grammar=grammar,
            candidate_registry=registry,
        )
        == ()
    )


def test_same_actor_next_activation_reads_the_previously_staged_state() -> None:
    grammar, registry = _context()
    source = _snapshot(_recurrent_actor("planning", recipient="planning", message_schema="stimulus-v1"))
    episode = _episode(source, caps=_caps(source, max_rounds=2))

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.CAP_EXHAUSTED
    assert [row.round_index for row in trace.activations] == [0, 1]
    first = trace.activations[0]
    second = trace.activations[1]
    assert isinstance(first.result, G0CounterfactualEvaluation)
    assert isinstance(second.result, G0CounterfactualEvaluation)
    assert second.result.initial_state.value() == {"memory": [1.0]}
    assert second.result.staged_state.value() == {"memory": [2.0]}
    assert trace.final_states[0].state.value() == {"memory": [2.0]}
    assert trace.pending_delivery_count == 1


def test_route_mismatch_refuses_the_whole_output_batch_and_leaves_state_unchanged() -> None:
    grammar, registry = _context()
    source = _chain()
    episode = _episode(source, receiver_schema="other-claim-v1")

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.DELIVERY_REFUSED
    assert trace.problems == ("message-schema-port-mismatch",)
    assert len(trace.activations) == 1
    activation = trace.activations[0]
    assert isinstance(activation.result, G0CounterfactualEvaluation)
    assert not activation.state_applied
    assert activation.emitted_message_sha256s
    assert not activation.enqueued_delivery_sha256s
    assert trace.routed_message_count == 0
    final = {row.actor_id: row.state.value() for row in trace.final_states}
    assert final["planning"] == {"memory": [0.0]}
    assert final["verification"] == {"memory": [0.0]}


def test_routed_byte_cap_is_atomic_at_exact_boundary_and_limit_minus_one() -> None:
    grammar, registry = _context()
    source = _chain()
    encoded = len(canonical_bytes([1.0]))
    exact_episode = _episode(source, caps=_caps(source, max_routed_payload_bytes=encoded))
    exact = execute_g0_shadow_coalition(
        source,
        exact_episode,
        grammar=grammar,
        candidate_registry=registry,
    )
    assert exact.terminal_reason is G0ShadowTerminalReason.QUIESCENT

    refused_episode = _episode(source, caps=_caps(source, max_routed_payload_bytes=encoded - 1))
    refused = execute_g0_shadow_coalition(
        source,
        refused_episode,
        grammar=grammar,
        candidate_registry=registry,
    )
    assert refused.terminal_reason is G0ShadowTerminalReason.DELIVERY_REFUSED
    assert refused.problems == ("routed-payload-byte-cap-exhausted",)
    assert refused.routed_message_count == 0
    assert refused.final_states[0].state.value() == {"memory": [0.0]}


def test_late_evaluator_refusal_charges_full_declared_actor_envelope() -> None:
    grammar, registry = _context()
    source = _chain()
    episode = _episode(source, seed_payload=[1.0, 2.0])

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    actor = next(row for row in source.actors if row.candidate_id == "planning")
    assert trace.terminal_reason is G0ShadowTerminalReason.EVALUATION_REFUSED
    result = trace.activations[0].result
    assert isinstance(result, G0CounterfactualRefusal)
    assert result.charged_operations == actor.declared_operations
    assert trace.work.actor_execution == actor.declared_operations
    assert trace.final_states[0].state.value() == {"memory": [0.0]}


def test_semantic_self_loop_stops_before_duplicate_activation() -> None:
    grammar, registry = _context()
    source = _snapshot(_stateless_self_actor("simulation"))
    episode = _episode(source)

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.REPEATED_STATE
    assert trace.problems == ("repeated-actor-input-state",)
    assert len(trace.activations) == 1
    assert trace.pending_delivery_count == 1


def test_round_cap_stops_before_recipient_activation_without_losing_pending_delivery() -> None:
    grammar, registry = _context()
    source = _chain()
    episode = _episode(source, caps=_caps(source, max_rounds=1))

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.CAP_EXHAUSTED
    assert trace.problems == ("round-cap-exhausted",)
    assert [row.actor_id for row in trace.activations] == ["planning"]
    assert trace.pending_delivery_count == 1
    final = {row.actor_id: row.state.value() for row in trace.final_states}
    assert final["planning"] == {"memory": [1.0]}
    assert final["verification"] == {"memory": [0.0]}


def test_declared_actor_work_cap_stops_before_evaluation() -> None:
    grammar, registry = _context()
    source = _chain()
    planning = next(row for row in source.actors if row.candidate_id == "planning")
    episode = _episode(
        source,
        caps=_caps(source, max_actor_operations=planning.declared_operations - 1),
    )

    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert trace.terminal_reason is G0ShadowTerminalReason.CAP_EXHAUSTED
    assert trace.problems == ("actor-operation-cap-exhausted",)
    assert not trace.activations
    assert trace.pending_delivery_count == 1
    assert trace.work.actor_execution == 0


def test_changed_episode_and_tampered_trace_do_not_verify() -> None:
    grammar, registry = _context()
    source = _chain()
    episode = _episode(source)
    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )
    changed_episode = _episode(source, seed_payload=[2.0])

    assert "deterministic-replay-mismatch" in verify_g0_shadow_trace(
        trace,
        source,
        changed_episode,
        grammar=grammar,
        candidate_registry=registry,
    )
    with pytest.raises(ValueError, match="trace self-hash mismatch"):
        replace(trace, trace_sha256="0" * 64)


def test_source_has_no_live_or_campaign_control_imports() -> None:
    source = inspect.getsource(shadow_module)
    for forbidden in (
        "mop.escs.runtime",
        "mop.escs.chassis",
        "mop.studio",
        "local_throttle",
        "campaign_supervisor",
        "null_safe_campaign_router",
    ):
        assert forbidden not in source
