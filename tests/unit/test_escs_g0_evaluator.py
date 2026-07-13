from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from mop.config import REPO_ROOT
from mop.escs.g0_evaluator import (
    G0CounterfactualEvaluation,
    G0CounterfactualRefusal,
    G0EvaluationError,
    attempt_g0_counterfactual,
    evaluate_g0_counterfactual,
    verify_g0_counterfactual,
)
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
from mop.substrate.events import canonical_bytes, canonical_sha256

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _context():
    return load_topology_grammar(GRAMMAR_PATH), load_perspective_candidate_registry(REGISTRY_PATH)


def _full_genotype() -> G0ActorGenotype:
    state_slots = (
        G0StateSlot(
            "graph-state",
            StatePrimitive.BOUNDED_TYPED_FACTOR_SUBGRAPHS,
            "graph-state-v1",
            4096,
        ),
        G0StateSlot(
            "record-state",
            StatePrimitive.BOUNDED_CATEGORICAL_TABLES,
            "record-state-v1",
            4096,
        ),
        G0StateSlot(
            "recurrent-state",
            StatePrimitive.BOUNDED_RECURRENT_STATE,
            "recurrent-state-v1",
            4096,
        ),
        G0StateSlot(
            "table-state",
            StatePrimitive.BOUNDED_CATEGORICAL_TABLES,
            "table-state-v1",
            4096,
        ),
        G0StateSlot(
            "temporal-state",
            StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
            "temporal-state-v1",
            4096,
        ),
    )
    edge = G0MessageEdge("claim-edge", "claim-v1", 256)
    nodes = (
        G0OperatorNode.create(
            node_id="affine",
            operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
            state_slot_ids=("recurrent-state",),
            parameters={
                "weights": [[1, 2, 0, 0], [-1, 1, 0, 0]],
                "bias": [0.5, 0],
                "activation": "relu",
                "state_mode": "concatenate",
            },
            declared_operations=18,
            max_output_bytes=64,
        ),
        G0OperatorNode.create(
            node_id="constraint",
            operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
            parameters={"allowed_values": ["keep"]},
            declared_operations=4,
            max_output_bytes=64,
        ),
        G0OperatorNode.create(
            node_id="emit",
            operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
            input_node_ids=("affine",),
            message_edge_ids=(edge.edge_id,),
            parameters={
                "schema_id": edge.schema_id,
                "recipient": "actor:verifier",
                "payload_form": "numeric-vector",
            },
            declared_operations=64,
            max_output_bytes=1024,
        ),
        G0OperatorNode.create(
            node_id="graph",
            operator=OperatorPrimitive.GRAPH_NEIGHBORHOOD_AGGREGATION,
            state_slot_ids=("graph-state",),
            parameters={"max_neighbors": 2, "reducer": "mean"},
            declared_operations=8,
            max_output_bytes=64,
        ),
        G0OperatorNode.create(
            node_id="retrieval",
            operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
            state_slot_ids=("record-state",),
            parameters={"query_key": "kind", "max_results": 2},
            declared_operations=3,
            max_output_bytes=256,
        ),
        G0OperatorNode.create(
            node_id="rollout",
            operator=OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH,
            parameters={
                "transition_table": {"a": ["b", "c"], "b": ["d"], "c": ["e"]},
                "max_depth": 2,
                "branch_factor": 2,
            },
            declared_operations=4,
            max_output_bytes=64,
        ),
        G0OperatorNode.create(
            node_id="table",
            operator=OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE,
            state_slot_ids=("table-state",),
            parameters={"mode": "stage_update", "max_entries": 3},
            declared_operations=2,
            max_output_bytes=128,
        ),
        G0OperatorNode.create(
            node_id="temporal",
            operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
            state_slot_ids=("temporal-state",),
            parameters={"window": 2, "reducer": "mean"},
            declared_operations=2,
            max_output_bytes=32,
        ),
    )
    return G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=state_slots,
        operator_nodes=nodes,
        output_node_ids=(
            "constraint",
            "emit",
            "graph",
            "retrieval",
            "rollout",
            "table",
            "temporal",
        ),
        message_edges=(edge,),
    )


def _inputs() -> dict[str, object]:
    return {
        "affine": [1, 2],
        "constraint": ["keep", "drop", "keep"],
        "graph": "root",
        "retrieval": "target",
        "rollout": "a",
        "table": ["z", 3],
        "temporal": 3,
    }


def _state() -> dict[str, object]:
    return {
        "graph-state": {
            "root": {"neighbors": ["right", "left"], "value": [0, 0]},
            "left": {"neighbors": [], "value": [1, 3]},
            "right": {"neighbors": [], "value": [3, 5]},
        },
        "record-state": [
            {"kind": "target", "value": 1},
            {"kind": "other", "value": 2},
            {"kind": "target", "value": 3},
        ],
        "recurrent-state": [0, 0],
        "table-state": {"a": 1},
        "temporal-state": [1, 2],
    }


def _evaluate(
    genotype: G0ActorGenotype | None = None,
    *,
    external_inputs: dict[str, object] | None = None,
    initial_state: dict[str, object] | None = None,
) -> G0CounterfactualEvaluation:
    grammar, registry = _context()
    return evaluate_g0_counterfactual(
        genotype or _full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
        external_inputs=_inputs() if external_inputs is None else external_inputs,
        initial_state=_state() if initial_state is None else initial_state,
    )


def _single_node_genotype(
    node: G0OperatorNode,
    *,
    state_slots: tuple[G0StateSlot, ...] = (),
    candidate_id: str = "planning",
) -> G0ActorGenotype:
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=state_slots,
        operator_nodes=(node,),
        output_node_ids=(node.node_id,),
    )


def test_all_eight_primitives_are_deterministic_and_counterfactual_only() -> None:
    first = _evaluate()
    second = _evaluate()
    outputs = first.node_outputs.value()
    staged_state = first.staged_state.value()
    message = first.staged_messages[0].value()

    assert first == second
    assert outputs["affine"] == [5.5, 1.0]
    assert outputs["constraint"] == ["keep", "keep"]
    assert outputs["graph"] == [2.0, 4.0]
    assert [row["value"] for row in outputs["retrieval"]] == [1, 3]
    assert outputs["rollout"] == ["a", "b", "c", "d", "e"]
    assert outputs["table"] == {"a": 1, "z": 3}
    assert outputs["temporal"] == 2.5
    assert staged_state["recurrent-state"] == [5.5, 1.0]
    assert staged_state["table-state"] == {"a": 1, "z": 3}
    assert staged_state["temporal-state"] == [2.0, 3.0]
    assert message["payload"] == [5.5, 1.0]
    assert message["counterfactual_only"] is True
    assert first.counterfactual_only is True
    assert first.activation_enabled is False
    assert first.factual_effects is False
    assert first.scientific_promotion_allowed is False
    assert first.work.actor_execution == first.operations
    assert first.work.messages == first.message_count == 1
    assert first.message_payload_bytes == message["encoded_bytes"]
    assert first.input_bytes > 0
    assert first.initial_state_bytes > 0
    assert first.node_output_bytes > 0
    assert first.staged_state_bytes > 0
    assert first.message_envelope_bytes > first.message_payload_bytes


def test_evaluation_round_trip_and_tamper_refuse_factual_authority() -> None:
    result = _evaluate()
    assert G0CounterfactualEvaluation.from_payload(result.payload()) == result

    payload = result.payload()
    payload["factual_effects"] = True
    with pytest.raises(G0EvaluationError, match="factual effect"):
        G0CounterfactualEvaluation.from_payload(payload)

    payload = result.payload()
    payload["staged_state"]["sha256"] = "0" * 64
    with pytest.raises(G0EvaluationError, match="digest mismatch"):
        G0CounterfactualEvaluation.from_payload(payload)

    payload = result.payload()
    payload["message_count"] = 2
    with pytest.raises(G0EvaluationError, match="work mismatch|message count mismatch|self-hash"):
        G0CounterfactualEvaluation.from_payload(payload)


def test_external_roots_and_strict_json_are_fail_closed() -> None:
    missing = _inputs()
    del missing["rollout"]
    with pytest.raises(G0EvaluationError, match="every and only root"):
        _evaluate(external_inputs=missing)

    extra = {**_inputs(), "emit": [1, 2]}
    with pytest.raises(G0EvaluationError, match="every and only root"):
        _evaluate(external_inputs=extra)

    nonfinite = _inputs()
    nonfinite["temporal"] = float("nan")
    with pytest.raises(G0EvaluationError, match="nonfinite number"):
        _evaluate(external_inputs=nonfinite)


def test_runtime_state_operation_and_output_caps_are_enforced() -> None:
    temporal_state = G0StateSlot(
        "state",
        StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
        "state-v1",
        2,
    )
    temporal = G0OperatorNode.create(
        node_id="temporal",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        state_slot_ids=("state",),
        parameters={"window": 2, "reducer": "mean"},
        declared_operations=2,
        max_output_bytes=32,
    )
    with pytest.raises(G0EvaluationError, match="initial state exceeds capacity"):
        _evaluate(
            _single_node_genotype(temporal, state_slots=(temporal_state,)),
            external_inputs={"temporal": 3},
            initial_state={"state": [1, 2]},
        )

    rollout = G0OperatorNode.create(
        node_id="rollout",
        operator=OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH,
        parameters={
            "transition_table": {"a": ["b", "c"]},
            "max_depth": 1,
            "branch_factor": 2,
        },
        declared_operations=1,
        max_output_bytes=64,
    )
    with pytest.raises(G0EvaluationError, match="exceeded declared operations"):
        _evaluate(
            _single_node_genotype(rollout),
            external_inputs={"rollout": "a"},
            initial_state={},
        )

    affine = G0OperatorNode.create(
        node_id="affine",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        parameters={
            "weights": [[1]],
            "bias": [0],
            "activation": "identity",
            "state_mode": "none",
        },
        declared_operations=2,
        max_output_bytes=2,
    )
    with pytest.raises(G0EvaluationError, match="output cap exceeded"):
        _evaluate(
            _single_node_genotype(affine),
            external_inputs={"affine": [1]},
            initial_state={},
        )


def test_state_primitive_mismatch_and_multiple_writers_are_rejected() -> None:
    wrong_slot = G0StateSlot(
        "state",
        StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
        "state-v1",
        128,
    )
    retrieval = G0OperatorNode.create(
        node_id="retrieval",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        state_slot_ids=("state",),
        parameters={"query_key": "kind", "max_results": 1},
        declared_operations=1,
        max_output_bytes=64,
    )
    with pytest.raises(G0EvaluationError, match="state primitive"):
        _evaluate(
            _single_node_genotype(retrieval, state_slots=(wrong_slot,)),
            external_inputs={"retrieval": "target"},
            initial_state={"state": []},
        )

    shared = G0StateSlot(
        "shared",
        StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
        "shared-v1",
        128,
    )
    left = G0OperatorNode.create(
        node_id="left",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        state_slot_ids=("shared",),
        parameters={"window": 2, "reducer": "last"},
        declared_operations=2,
        max_output_bytes=32,
    )
    right = G0OperatorNode.create(
        node_id="right",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        state_slot_ids=("shared",),
        parameters={"window": 2, "reducer": "last"},
        declared_operations=2,
        max_output_bytes=32,
    )
    genotype = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(shared,),
        operator_nodes=(left, right),
        output_node_ids=("left", "right"),
    )
    with pytest.raises(G0EvaluationError, match="multiple state writers"):
        _evaluate(
            genotype,
            external_inputs={"left": 1, "right": 2},
            initial_state={"shared": [0]},
        )


def test_registry_authority_and_excluded_candidates_are_rejected() -> None:
    grammar, registry = _context()
    changed_candidate = replace(registry.candidates[0], label="authority drift")
    changed_registry = replace(
        registry,
        candidates=(changed_candidate, *registry.candidates[1:]),
    )
    with pytest.raises(G0EvaluationError, match="grammar authority"):
        evaluate_g0_counterfactual(
            _full_genotype(),
            grammar=grammar,
            candidate_registry=changed_registry,
            external_inputs=_inputs(),
            initial_state=_state(),
        )

    node = G0OperatorNode.create(
        node_id="filter",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        parameters={"allowed_values": [1]},
        declared_operations=2,
        max_output_bytes=16,
    )
    with pytest.raises(G0EvaluationError, match="candidate-excluded"):
        evaluate_g0_counterfactual(
            _single_node_genotype(node, candidate_id="curiosity"),
            grammar=grammar,
            candidate_registry=registry,
            external_inputs={"filter": [1]},
            initial_state={},
        )


def test_authorities_state_and_per_node_costs_are_identity_bound() -> None:
    first = _evaluate()
    changed_state = _state()
    changed_state["temporal-state"] = [0, 2]
    second = _evaluate(initial_state=changed_state)
    costs = first.node_costs.value()

    assert first.grammar_sha256 == load_topology_grammar(GRAMMAR_PATH).grammar_sha256
    assert first.candidate_registry_sha256 == load_perspective_candidate_registry(
        REGISTRY_PATH
    ).sha256
    assert first.initial_state_sha256 != second.initial_state_sha256
    assert first.counterfactual_branch_id != second.counterfactual_branch_id
    assert first.external_inputs.sha256 == first.input_sha256
    assert first.initial_state.sha256 == first.initial_state_sha256
    assert len(costs) == len(_full_genotype().operator_nodes) == 8
    assert sum(row["used_operations"] for row in costs) == first.operations
    assert sum(row["output_bytes"] for row in costs) == first.node_output_bytes
    assert all(row["status"] == "completed" for row in costs)
    assert all(row["used_operations"] <= row["declared_operations"] for row in costs)
    grammar, registry = _context()
    assert verify_g0_counterfactual(
        first,
        genotype=_full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
    ) == ()


def test_fully_rehashed_inner_message_forgery_is_still_rejected() -> None:
    payload = _evaluate().payload()
    message_payload = payload["staged_messages"][0]
    message = message_payload["value"]
    message["activation_enabled"] = True
    message["message_sha256"] = canonical_sha256(
        {key: value for key, value in message.items() if key != "message_sha256"}
    )
    message_payload["sha256"] = canonical_sha256(message)
    payload["evaluation_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "evaluation_sha256"}
    )

    with pytest.raises(G0EvaluationError, match="staged message enabled activation"):
        G0CounterfactualEvaluation.from_payload(payload)


def test_independent_state_readers_see_the_activation_start_snapshot() -> None:
    slot = G0StateSlot(
        "table-state",
        StatePrimitive.BOUNDED_CATEGORICAL_TABLES,
        "table-state-v1",
        128,
    )
    update = G0OperatorNode.create(
        node_id="a-update",
        operator=OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE,
        state_slot_ids=(slot.slot_id,),
        parameters={"mode": "stage_update", "max_entries": 2},
        declared_operations=2,
        max_output_bytes=64,
    )
    lookup = G0OperatorNode.create(
        node_id="z-lookup",
        operator=OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE,
        state_slot_ids=(slot.slot_id,),
        parameters={"mode": "lookup"},
        declared_operations=1,
        max_output_bytes=8,
    )
    genotype = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(slot,),
        operator_nodes=(update, lookup),
        output_node_ids=(update.node_id, lookup.node_id),
    )
    result = _evaluate(
        genotype,
        external_inputs={"a-update": ["new", 1], "z-lookup": "new"},
        initial_state={"table-state": {"old": 0}},
    )

    assert result.node_outputs.value()["z-lookup"] is None
    assert result.staged_state.value()["table-state"] == {"new": 1, "old": 0}


def test_ordered_fan_in_and_plural_typed_emit_are_explicit() -> None:
    left = G0OperatorNode.create(
        node_id="left",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        parameters={
            "weights": [[1]],
            "bias": [0],
            "activation": "identity",
            "state_mode": "none",
        },
        declared_operations=2,
        max_output_bytes=16,
    )
    right = G0OperatorNode.create(
        node_id="right",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        parameters={
            "weights": [[2]],
            "bias": [0],
            "activation": "identity",
            "state_mode": "none",
        },
        declared_operations=2,
        max_output_bytes=16,
    )
    first_edge = G0MessageEdge("first-edge", "first-claim-v1", 64)
    second_edge = G0MessageEdge("second-edge", "second-claim-v1", 64)
    emit = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=("right", "left"),
        message_edge_ids=(first_edge.edge_id, second_edge.edge_id),
        parameters={
            "recipient_cap": 2,
            "emissions": [
                {
                    "edge_id": first_edge.edge_id,
                    "schema_id": first_edge.schema_id,
                    "recipient": "actor:first",
                    "payload_form": "json-list",
                },
                {
                    "edge_id": second_edge.edge_id,
                    "schema_id": second_edge.schema_id,
                    "recipient": "actor:second",
                    "payload_form": "json-list",
                },
            ],
        },
        declared_operations=2,
        max_output_bytes=128,
    )
    genotype = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(),
        operator_nodes=(left, right, emit),
        output_node_ids=(emit.node_id,),
        message_edges=(first_edge, second_edge),
    )
    grammar, registry = _context()
    result = evaluate_g0_counterfactual(
        genotype,
        grammar=grammar,
        candidate_registry=registry,
        external_inputs={"left": [3], "right": [5]},
        initial_state={},
    )

    assert result.node_outputs.value()["emit"] == [[10.0], [3.0]]
    assert [message.value()["edge_id"] for message in result.staged_messages] == [
        "first-edge",
        "second-edge",
    ]
    assert result.message_count == result.work.messages == 2


def test_json_identity_and_safe_numeric_domain_are_explicit() -> None:
    slot = G0StateSlot(
        "records",
        StatePrimitive.BOUNDED_CATEGORICAL_TABLES,
        "records-v1",
        128,
    )
    retrieval = G0OperatorNode.create(
        node_id="retrieval",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        state_slot_ids=(slot.slot_id,),
        parameters={"query_key": "kind", "max_results": 1},
        declared_operations=1,
        max_output_bytes=64,
    )
    result = _evaluate(
        _single_node_genotype(retrieval, state_slots=(slot,)),
        external_inputs={"retrieval": 1},
        initial_state={"records": [{"kind": True}]},
    )
    assert result.node_outputs.value()["retrieval"] == []

    temporal_slot = G0StateSlot(
        "history",
        StatePrimitive.BOUNDED_TEMPORAL_DEQUES,
        "history-v1",
        128,
    )
    temporal = G0OperatorNode.create(
        node_id="temporal",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        state_slot_ids=(temporal_slot.slot_id,),
        parameters={"window": 1, "reducer": "last"},
        declared_operations=1,
        max_output_bytes=32,
    )
    with pytest.raises(G0EvaluationError, match="exact integer domain"):
        _evaluate(
            _single_node_genotype(temporal, state_slots=(temporal_slot,)),
            external_inputs={"temporal": 2**53},
            initial_state={"history": [0]},
        )


def test_late_failure_returns_a_conservative_immutable_charge() -> None:
    first = G0OperatorNode.create(
        node_id="first",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        parameters={"allowed_values": [1]},
        declared_operations=2,
        max_output_bytes=16,
    )
    late = G0OperatorNode.create(
        node_id="late",
        operator=OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH,
        parameters={
            "transition_table": {"a": ["b", "c"]},
            "max_depth": 1,
            "branch_factor": 2,
        },
        declared_operations=1,
        max_output_bytes=64,
    )
    genotype = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(),
        operator_nodes=(first, late),
        output_node_ids=(first.node_id, late.node_id),
    )
    grammar, registry = _context()
    result = attempt_g0_counterfactual(
        genotype,
        grammar=grammar,
        candidate_registry=registry,
        external_inputs={"first": [1], "late": "a"},
        initial_state={},
        attempt_id="attempt:g0/late-failure",
    )

    assert isinstance(result, G0CounterfactualRefusal)
    assert result.reason == "late exceeded declared operations"
    assert result.charged_operations == genotype.declared_operations == 3
    assert result.work.actor_execution == 3
    assert result.accounting_mode == "conservative-full-genotype-envelope"
    assert result.attempt_id == "attempt:g0/late-failure"
    assert result.activation_enabled is False
    assert result.factual_effects is False
    assert result.scientific_promotion_allowed is False
    assert G0CounterfactualRefusal.from_payload(result.payload()) == result


def test_parameter_typos_and_ignored_controls_are_rejected() -> None:
    affine = G0OperatorNode.create(
        node_id="affine",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        parameters={
            "weights": [[1]],
            "bias": [0],
            "activtion": "relu",
            "state_mode": "none",
            "ignored_control": "factual",
        },
        declared_operations=2,
        max_output_bytes=16,
    )
    with pytest.raises(G0EvaluationError, match="parameter fields mismatch"):
        _evaluate(
            _single_node_genotype(affine),
            external_inputs={"affine": [-1]},
            initial_state={},
        )


def test_deep_noncanonical_attempt_returns_an_identity_incomplete_receipt() -> None:
    nested: object = "leaf"
    for _ in range(80):
        nested = [nested]
    inputs = _inputs()
    inputs["constraint"] = nested
    grammar, registry = _context()
    result = attempt_g0_counterfactual(
        _full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
        external_inputs=inputs,
        initial_state=_state(),
        attempt_id="attempt:g0/deep-input",
    )

    assert isinstance(result, G0CounterfactualRefusal)
    assert "JSON depth cap" in result.reason
    assert result.input_identity_complete is False
    assert result.external_inputs is None
    assert result.input_bytes == 0
    assert result.initial_state_identity_complete is True
    assert result.initial_state is not None
    assert result.attempt_id == "attempt:g0/deep-input"
    assert G0CounterfactualRefusal.from_payload(result.payload()) == result


def test_malformed_attempt_ids_keep_repeated_invalid_inputs_accounting_distinct() -> None:
    inputs = _inputs()
    inputs["temporal"] = float("nan")
    grammar, registry = _context()
    first = attempt_g0_counterfactual(
        _full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
        external_inputs=inputs,
        initial_state=_state(),
        attempt_id="attempt:g0/nonfinite-1",
    )
    second = attempt_g0_counterfactual(
        _full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
        external_inputs=inputs,
        initial_state=_state(),
        attempt_id="attempt:g0/nonfinite-2",
    )

    assert isinstance(first, G0CounterfactualRefusal)
    assert isinstance(second, G0CounterfactualRefusal)
    assert first.input_identity_complete is False
    assert first.input_sha256 == second.input_sha256
    assert first.refusal_sha256 != second.refusal_sha256


def test_fully_rehashed_duplicate_staged_message_is_rejected() -> None:
    payload = _evaluate().payload()
    duplicate = copy.deepcopy(payload["staged_messages"][0])
    payload["staged_messages"].append(duplicate)
    payload["message_count"] = 2
    payload["message_payload_bytes"] *= 2
    payload["message_envelope_bytes"] *= 2
    payload["work"]["messages"] = 2
    costs = payload["node_costs"]["value"]
    emit_cost = next(row for row in costs if row["node_id"] == "emit")
    emit_cost["message_count"] = 2
    emit_cost["message_payload_bytes"] *= 2
    emit_cost["node_cost_sha256"] = canonical_sha256(
        {key: value for key, value in emit_cost.items() if key != "node_cost_sha256"}
    )
    payload["node_costs"]["sha256"] = canonical_sha256(costs)
    payload["evaluation_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "evaluation_sha256"}
    )

    with pytest.raises(G0EvaluationError, match="message identity is duplicated"):
        G0CounterfactualEvaluation.from_payload(payload)


def test_external_authority_replay_rejects_a_fully_rehashed_output_forgery() -> None:
    original = _evaluate()
    payload = original.payload()
    outputs = payload["node_outputs"]["value"]
    prior_bytes = len(canonical_bytes(outputs["constraint"]))
    outputs["constraint"] = ["drop"]
    next_bytes = len(canonical_bytes(outputs["constraint"]))
    payload["node_outputs"]["sha256"] = canonical_sha256(outputs)
    payload["node_output_bytes"] += next_bytes - prior_bytes
    costs = payload["node_costs"]["value"]
    constraint_cost = next(row for row in costs if row["node_id"] == "constraint")
    constraint_cost["output_bytes"] = next_bytes
    constraint_cost["node_cost_sha256"] = canonical_sha256(
        {key: value for key, value in constraint_cost.items() if key != "node_cost_sha256"}
    )
    payload["node_costs"]["sha256"] = canonical_sha256(costs)
    payload["evaluation_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "evaluation_sha256"}
    )
    forged = G0CounterfactualEvaluation.from_payload(payload)
    grammar, registry = _context()

    assert verify_g0_counterfactual(
        forged,
        genotype=_full_genotype(),
        grammar=grammar,
        candidate_registry=registry,
    ) == ("deterministic-replay-mismatch",)
