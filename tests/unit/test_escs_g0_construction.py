from __future__ import annotations

import inspect

import pytest

from mop.config import REPO_ROOT
from mop.escs.g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    G0StateSlot,
)
from mop.escs.perspective_registry import load_perspective_candidate_registry
from mop.escs.topology_grammar import (
    ConstructionMutation,
    OperatorPrimitive,
    StatePrimitive,
    load_topology_grammar,
)
from mop.studies import escs_g0_construction as construction_module
from mop.studies.escs_g0_construction import (
    G0ConstructionAttempt,
    G0ConstructionOperation,
    G0ConstructionRequest,
    G0ConstructionSnapshot,
    G0ConstructionStatus,
    attempt_g0_construction,
    verify_g0_construction_attempt,
)

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _context():
    return load_topology_grammar(GRAMMAR_PATH), load_perspective_candidate_registry(REGISTRY_PATH)


def _emit_node(edges: tuple[G0MessageEdge, ...]) -> G0OperatorNode:
    if len(edges) == 1:
        parameters: dict[str, object] = {
            "schema_id": edges[0].schema_id,
            "recipient": "actor:verification",
            "payload_form": "numeric-scalar",
        }
    else:
        parameters = {
            "recipient_cap": len(edges),
            "emissions": [
                {
                    "edge_id": edge.edge_id,
                    "schema_id": edge.schema_id,
                    "recipient": f"actor:{'verification' if index == 0 else 'reflection'}",
                    "payload_form": "numeric-scalar",
                }
                for index, edge in enumerate(edges)
            ],
        }
    return G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=("temporal",),
        message_edge_ids=tuple(edge.edge_id for edge in edges),
        parameters=parameters,
        declared_operations=16,
        max_output_bytes=512,
    )


def _actor(candidate_id: str, *, edge_count: int = 1) -> G0ActorGenotype:
    state = G0StateSlot(
        "history",
        StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
        "history-v1",
        4096,
    )
    edges = tuple(G0MessageEdge(f"claim-{index}", "claim-v1", 64) for index in range(edge_count))
    temporal = G0OperatorNode.create(
        node_id="temporal",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        state_slot_ids=(state.slot_id,),
        parameters={"window": 4, "reducer": "mean"},
        declared_operations=16,
        max_output_bytes=64,
    )
    auxiliary = G0OperatorNode.create(
        node_id="auxiliary",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        parameters={"allowed_values": ["keep"]},
        declared_operations=4,
        max_output_bytes=32,
    )
    emit = _emit_node(edges)
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=(state,),
        operator_nodes=(temporal, auxiliary, emit),
        output_node_ids=("auxiliary", "emit"),
        message_edges=edges,
    )


def _snapshot(*actors: G0ActorGenotype) -> G0ConstructionSnapshot:
    return G0ConstructionSnapshot.create(
        actors=actors,
        factor_scopes={actor.candidate_id: (f"{actor.candidate_id}.world",) for actor in actors},
    )


def _request(
    source: G0ConstructionSnapshot,
    operation: G0ConstructionOperation,
    parameters: dict[str, object],
    *,
    attempt_id: str = "attempt.one",
    declared_work: int = 101,
) -> G0ConstructionRequest:
    return G0ConstructionRequest.create(
        attempt_id=attempt_id,
        source_snapshot_sha256=source.snapshot_sha256,
        operation=operation,
        parameters=parameters,
        declared_work=declared_work,
    )


def _attempt(
    request: G0ConstructionRequest,
    source: G0ConstructionSnapshot,
) -> G0ConstructionAttempt:
    grammar, registry = _context()
    return attempt_g0_construction(
        request,
        source=source,
        grammar=grammar,
        candidate_registry=registry,
    )


def test_snapshot_request_and_attempt_are_self_hashed_and_exactly_replayable() -> None:
    grammar, registry = _context()
    source = _snapshot(_actor("planning"))
    request = _request(
        source,
        G0ConstructionOperation.ADJUST_STATE_CAPACITY,
        {"actor_id": "planning", "slot_id": "history", "capacity_bytes": 8192},
    )
    attempt = attempt_g0_construction(
        request,
        source=source,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert G0ConstructionSnapshot.from_payload(source.payload()) == source
    assert G0ConstructionRequest.from_payload(request.payload()) == request
    assert G0ConstructionAttempt.from_payload(attempt.payload()) == attempt
    assert verify_g0_construction_attempt(
        attempt,
        request,
        source=source,
        grammar=grammar,
        candidate_registry=registry,
    )
    assert attempt.status is G0ConstructionStatus.APPLIED_SHADOW
    assert attempt.charged_work == request.declared_work
    assert attempt.counterfactual_only is True
    assert attempt.activation_enabled is False
    assert attempt.shadow_execution_authorized is False
    assert attempt.factual_mutation_authorized is False
    assert attempt.scientific_promotion_allowed is False
    assert source.actors[0].retained_state_bytes == 4096

    forged_request = request.payload()
    forged_request["declared_work"] = 1
    with pytest.raises(ValueError, match="self-hash"):
        G0ConstructionRequest.from_payload(forged_request)

    forged_attempt = attempt.payload()
    forged_attempt["charged_work"] = 1
    with pytest.raises(ValueError, match="self-hash"):
        G0ConstructionAttempt.from_payload(forged_attempt)


def test_operator_family_add_replace_remove_and_cycle_or_schema_refusals() -> None:
    source = _snapshot(_actor("planning"))
    added_node = G0OperatorNode.create(
        node_id="final-filter",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        input_node_ids=("auxiliary",),
        parameters={"allowed_values": ["keep"]},
        declared_operations=3,
        max_output_bytes=32,
    )
    add = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADD_OPERATOR,
            {"actor_id": "planning", "node": added_node.payload()},
        ),
        source,
    )
    assert add.status is G0ConstructionStatus.APPLIED_SHADOW
    assert add.candidate_snapshot is not None
    assert {node.node_id for node in add.candidate_snapshot.actors[0].operator_nodes} == {
        "auxiliary",
        "emit",
        "final-filter",
        "temporal",
    }

    remove_source = add.candidate_snapshot
    remove = _attempt(
        _request(
            remove_source,
            G0ConstructionOperation.REMOVE_OPERATOR,
            {"actor_id": "planning", "node_id": "final-filter"},
            attempt_id="attempt.remove",
        ),
        remove_source,
    )
    assert remove.status is G0ConstructionStatus.APPLIED_SHADOW
    assert remove.candidate_snapshot == source

    replacement = G0OperatorNode.create(
        node_id="auxiliary",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        parameters={"allowed_values": ["keep", "defer"]},
        declared_operations=5,
        max_output_bytes=48,
    )
    replace_attempt = _attempt(
        _request(
            source,
            G0ConstructionOperation.REPLACE_OPERATOR,
            {"actor_id": "planning", "node": replacement.payload()},
            attempt_id="attempt.replace",
        ),
        source,
    )
    assert replace_attempt.status is G0ConstructionStatus.APPLIED_SHADOW

    cyclic_temporal = G0OperatorNode.create(
        node_id="temporal",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        input_node_ids=("emit",),
        state_slot_ids=("history",),
        parameters={"window": 4, "reducer": "mean"},
        declared_operations=16,
        max_output_bytes=64,
    )
    cycle = _attempt(
        _request(
            source,
            G0ConstructionOperation.REPLACE_OPERATOR,
            {"actor_id": "planning", "node": cyclic_temporal.payload()},
            attempt_id="attempt.cycle",
            declared_work=777,
        ),
        source,
    )
    assert cycle.status is G0ConstructionStatus.REFUSED
    assert cycle.charged_work == 777
    assert cycle.candidate_snapshot is None
    assert any("cycle" in problem for problem in cycle.problems)

    edge = source.actors[0].message_edges[0]
    wrong_schema = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=("temporal",),
        message_edge_ids=(edge.edge_id,),
        parameters={
            "schema_id": "wrong-schema-v1",
            "recipient": "actor:verification",
            "payload_form": "numeric-scalar",
        },
        declared_operations=16,
        max_output_bytes=512,
    )
    schema_refusal = _attempt(
        _request(
            source,
            G0ConstructionOperation.REPLACE_OPERATOR,
            {"actor_id": "planning", "node": wrong_schema.payload()},
            attempt_id="attempt.schema",
        ),
        source,
    )
    assert schema_refusal.status is G0ConstructionStatus.REFUSED
    assert any("schema" in problem for problem in schema_refusal.problems)


def test_typed_message_edge_add_and_remove_replay_to_original_snapshot() -> None:
    source = _snapshot(_actor("planning"))
    original_actor = source.actors[0]
    second = G0MessageEdge("claim-1", "claim-v1", 64)
    two_edges = (original_actor.message_edges[0], second)
    add = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADD_MESSAGE_EDGE,
            {
                "actor_id": "planning",
                "edge": second.payload(),
                "owner_node": _emit_node(two_edges).payload(),
            },
            attempt_id="attempt.edge-add",
        ),
        source,
    )
    assert add.status is G0ConstructionStatus.APPLIED_SHADOW
    assert add.candidate_snapshot is not None
    assert len(add.candidate_snapshot.actors[0].message_edges) == 2

    remove_source = add.candidate_snapshot
    remove = _attempt(
        _request(
            remove_source,
            G0ConstructionOperation.REMOVE_MESSAGE_EDGE,
            {
                "actor_id": "planning",
                "edge_id": second.edge_id,
                "owner_node": _emit_node((original_actor.message_edges[0],)).payload(),
            },
            attempt_id="attempt.edge-remove",
        ),
        remove_source,
    )
    assert remove.status is G0ConstructionStatus.APPLIED_SHADOW
    assert remove.candidate_snapshot == source


def test_capacity_and_timescale_adjustments_obey_existing_finite_envelopes() -> None:
    grammar, _ = _context()
    source = _snapshot(_actor("planning"))
    capacity = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADJUST_STATE_CAPACITY,
            {"actor_id": "planning", "slot_id": "history", "capacity_bytes": 8192},
            attempt_id="attempt.capacity",
        ),
        source,
    )
    assert capacity.status is G0ConstructionStatus.APPLIED_SHADOW
    assert capacity.retained_state_bytes_delta == 4096

    oversized = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADJUST_STATE_CAPACITY,
            {
                "actor_id": "planning",
                "slot_id": "history",
                "capacity_bytes": (
                    grammar.construction_language.composition_bounds.max_state_bytes_per_actor + 1
                ),
            },
            attempt_id="attempt.capacity-overflow",
            declared_work=909,
        ),
        source,
    )
    assert oversized.status is G0ConstructionStatus.REFUSED
    assert oversized.charged_work == 909
    assert {problem.split(":")[-1] for problem in oversized.problems} & {
        "state-byte-cap-exceeded",
        "actor-total-byte-cap-exceeded",
    }

    timescale = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADJUST_TEMPORAL_WINDOW,
            {"actor_id": "planning", "node_id": "temporal", "window": 8},
            attempt_id="attempt.timescale",
        ),
        source,
    )
    assert timescale.status is G0ConstructionStatus.APPLIED_SHADOW
    assert timescale.candidate_snapshot is not None
    temporal = next(
        node for node in timescale.candidate_snapshot.actors[0].operator_nodes if node.node_id == "temporal"
    )
    assert temporal.parameters.value()["window"] == 8

    unchargeable_window = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADJUST_TEMPORAL_WINDOW,
            {"actor_id": "planning", "node_id": "temporal", "window": 17},
            attempt_id="attempt.timescale-overflow",
            declared_work=515,
        ),
        source,
    )
    assert unchargeable_window.status is G0ConstructionStatus.REFUSED
    assert unchargeable_window.charged_work == 515
    assert any("operation envelope" in problem for problem in unchargeable_window.problems)

    declared_over_cap = grammar.caps.max_structural_work + 1
    overwork = _attempt(
        _request(
            source,
            G0ConstructionOperation.ADJUST_STATE_CAPACITY,
            {"actor_id": "planning", "slot_id": "history", "capacity_bytes": 8192},
            attempt_id="attempt.work-overflow",
            declared_work=declared_over_cap,
        ),
        source,
    )
    assert overwork.status is G0ConstructionStatus.REFUSED
    assert overwork.problems == ("structural-work-cap-exceeded",)
    assert overwork.charged_work == declared_over_cap


def test_clone_and_merge_are_content_addressed_and_identity_safe() -> None:
    source = _snapshot(_actor("planning"))
    clone = _attempt(
        _request(
            source,
            G0ConstructionOperation.CLONE_ACTOR,
            {"source_actor_id": "planning", "new_actor_id": "simulation"},
            attempt_id="attempt.clone",
        ),
        source,
    )
    assert clone.status is G0ConstructionStatus.APPLIED_SHADOW
    assert clone.candidate_snapshot is not None
    assert tuple(actor.candidate_id for actor in clone.candidate_snapshot.actors) == (
        "planning",
        "simulation",
    )
    assert clone.actor_delta == 1
    assert all(actor.activation_enabled is False for actor in clone.candidate_snapshot.actors)

    duplicate = _attempt(
        _request(
            clone.candidate_snapshot,
            G0ConstructionOperation.CLONE_ACTOR,
            {"source_actor_id": "planning", "new_actor_id": "simulation"},
            attempt_id="attempt.clone-duplicate",
            declared_work=333,
        ),
        clone.candidate_snapshot,
    )
    assert duplicate.status is G0ConstructionStatus.REFUSED
    assert duplicate.charged_work == 333

    pair = _snapshot(_actor("planning"), _actor("verification"))
    merge = _attempt(
        _request(
            pair,
            G0ConstructionOperation.MERGE_ACTORS,
            {
                "left_actor_id": "planning",
                "right_actor_id": "verification",
                "target_actor_id": "abstraction",
            },
            attempt_id="attempt.merge",
        ),
        pair,
    )
    assert merge.status is G0ConstructionStatus.APPLIED_SHADOW
    assert merge.candidate_snapshot is not None
    assert tuple(actor.candidate_id for actor in merge.candidate_snapshot.actors) == ("abstraction",)
    assert merge.actor_delta == -1
    merged = merge.candidate_snapshot.actors[0]
    assert all(node.node_id.startswith(("left.", "right.")) for node in merged.operator_nodes)
    assert set(merge.candidate_snapshot.factor_scopes[0].scope_ids) == {
        "planning.world",
        "verification.world",
    }


def test_factor_scope_split_and_merge_preserve_exact_snapshot_identity() -> None:
    actor = _actor("planning")
    source = G0ConstructionSnapshot.create(
        actors=(actor,),
        factor_scopes={"planning": ("world",)},
    )
    split = _attempt(
        _request(
            source,
            G0ConstructionOperation.SPLIT_FACTOR_SCOPE,
            {
                "actor_id": "planning",
                "source_scope_id": "world",
                "left_scope_id": "world.left",
                "right_scope_id": "world.right",
            },
            attempt_id="attempt.scope-split",
        ),
        source,
    )
    assert split.status is G0ConstructionStatus.APPLIED_SHADOW
    assert split.candidate_snapshot is not None
    assert split.factor_scope_delta == 1
    assert split.candidate_snapshot.factor_scopes[0].scope_ids == (
        "world.left",
        "world.right",
    )

    merge_source = split.candidate_snapshot
    merge = _attempt(
        _request(
            merge_source,
            G0ConstructionOperation.MERGE_FACTOR_SCOPES,
            {
                "actor_id": "planning",
                "left_scope_id": "world.left",
                "right_scope_id": "world.right",
                "target_scope_id": "world",
            },
            attempt_id="attempt.scope-merge",
        ),
        merge_source,
    )
    assert merge.status is G0ConstructionStatus.APPLIED_SHADOW
    assert merge.candidate_snapshot == source


def test_all_six_declared_families_are_reachable_without_runtime_authority() -> None:
    source = _snapshot(_actor("planning"))
    representative_operations = (
        G0ConstructionOperation.ADD_OPERATOR,
        G0ConstructionOperation.ADD_MESSAGE_EDGE,
        G0ConstructionOperation.ADJUST_STATE_CAPACITY,
        G0ConstructionOperation.CLONE_ACTOR,
        G0ConstructionOperation.MERGE_ACTORS,
        G0ConstructionOperation.SPLIT_FACTOR_SCOPE,
    )
    families = {
        _request(
            source,
            operation,
            {},
            attempt_id=f"attempt.family-{index}",
        ).family
        for index, operation in enumerate(representative_operations)
    }

    assert families == set(ConstructionMutation)
    module_source = inspect.getsource(construction_module)
    assert "from mop.escs.runtime" not in module_source
    assert "from mop.escs.chassis" not in module_source
    assert "CoalitionRuntime" not in module_source
    assert "EventSourcedCoalitionChassis" not in module_source
