from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mop.config import REPO_ROOT
from mop.escs.g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    G0StateSlot,
    assess_g0_genotype,
)
from mop.escs.perspective_registry import load_perspective_candidate_registry
from mop.escs.topology_grammar import (
    OperatorPrimitive,
    StatePrimitive,
    load_topology_grammar,
)
from mop.substrate.events import FrozenJSON

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _genotype(
    *,
    candidate_id: str = "planning",
    state_bytes: int = 4096,
    declared_operations: int = 128,
) -> G0ActorGenotype:
    state = G0StateSlot(
        "plan-state",
        StatePrimitive.BOUNDED_RECURRENT_STATE,
        "plan-state-v1",
        state_bytes,
    )
    edge = G0MessageEdge("action-claim", "action-claim-v1", 256)
    encode = G0OperatorNode.create(
        node_id="encode",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        state_slot_ids=(state.slot_id,),
        parameters={"activation": "tanh", "input_width": 8, "output_width": 8},
        declared_operations=declared_operations,
        max_output_bytes=512,
    )
    emit = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=(encode.node_id,),
        message_edge_ids=(edge.edge_id,),
        parameters={"claim_schema_id": edge.schema_id, "recipient_cap": 1},
        declared_operations=32,
        max_output_bytes=256,
    )
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=(state,),
        operator_nodes=(encode, emit),
        output_node_ids=(emit.node_id,),
        message_edges=(edge,),
    )


def test_valid_finite_genotype_is_structural_but_cannot_execute() -> None:
    genotype = _genotype()
    grammar = load_topology_grammar(GRAMMAR_PATH)
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    assessment = assess_g0_genotype(
        genotype,
        grammar=grammar,
        candidate_registry=registry,
    )

    assert genotype.dag_depth == 2
    assert genotype.retained_state_bytes == 4096
    assert genotype.declared_message_bytes == 256
    assert genotype.activation_enabled is False
    assert assessment.grammar_sha256 == grammar.grammar_sha256
    assert assessment.candidate_registry_sha256 == registry.sha256
    assert assessment.structurally_valid is True
    assert assessment.shadow_authorized is False
    assert assessment.factual_activation_authorized is False
    assert {
        "construction-language-incomplete",
        "candidate-activation-disabled",
        "freeze-authority-absent",
        "genotype-shadow-authorization-disabled",
        "grammar-activation-disabled",
        "grammar-not-frozen",
    } <= set(assessment.blockers)


def test_cycles_and_undeclared_references_are_rejected() -> None:
    left = G0OperatorNode.create(
        node_id="left",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        input_node_ids=("right",),
        parameters={"window": 4},
        declared_operations=1,
        max_output_bytes=8,
    )
    right = G0OperatorNode.create(
        node_id="right",
        operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
        input_node_ids=("left",),
        parameters={"window": 4},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="cycle"):
        G0ActorGenotype.create(
            candidate_id="temporal_reasoning",
            state_slots=(),
            operator_nodes=(left, right),
            output_node_ids=("right",),
        )

    dangling = G0OperatorNode.create(
        node_id="dangling",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        state_slot_ids=("missing",),
        parameters={"max_results": 1},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="undeclared nodes, state, or message"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=(dangling,),
            output_node_ids=("dangling",),
        )


def test_deep_dag_validation_is_iterative_and_does_not_recurse() -> None:
    nodes: list[G0OperatorNode] = []
    for index in range(1_200):
        node_id = f"node-{index:04d}"
        nodes.append(
            G0OperatorNode.create(
                node_id=node_id,
                operator=OperatorPrimitive.TEMPORAL_ACCUMULATION,
                input_node_ids=((nodes[-1].node_id,) if nodes else ()),
                parameters={"window": 1},
                declared_operations=1,
                max_output_bytes=1,
            )
        )
    genotype = G0ActorGenotype.create(
        candidate_id="temporal_reasoning",
        state_slots=(),
        operator_nodes=nodes,
        output_node_ids=(nodes[-1].node_id,),
    )

    assert genotype.dag_depth == 1_200


def test_frozen_records_reject_mutable_nested_collections() -> None:
    node = G0OperatorNode.create(
        node_id="node",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        parameters={"max_results": 1},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="immutable tuples"):
        replace(node, state_slot_ids=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="immutable tuples"):
        G0OperatorNode(
            node_id="node",
            operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
            input_node_ids=(),
            state_slot_ids=(),
            message_edge_ids=[],  # type: ignore[arg-type]
            parameters=FrozenJSON.from_value({"max_results": 1}),
            declared_operations=1,
            max_output_bytes=8,
        )
    with pytest.raises(ValueError, match="collections must be immutable tuples"):
        replace(_genotype(), operator_nodes=list(_genotype().operator_nodes))  # type: ignore[arg-type]


def test_forbidden_generated_code_cannot_hide_in_nested_parameters() -> None:
    with pytest.raises(ValueError, match="forbidden construction"):
        G0OperatorNode.create(
            node_id="unsafe",
            operator=OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH,
            parameters={"transition": {"python": "lambda state: state"}},
            declared_operations=1,
            max_output_bytes=8,
        )

    symbolic = G0OperatorNode.create(
        node_id="symbolic",
        operator=OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE,
        parameters={"code": "category-a", "max_entries": 4},
        declared_operations=1,
        max_output_bytes=8,
    )
    assert symbolic.parameters.value()["code"] == "category-a"

    for capability in (
        "mutation_outside_transactional_governance",
        "silent_schema_creation",
        "unbounded_recursion",
        "undeclared_operators",
    ):
        with pytest.raises(ValueError, match="forbidden construction"):
            G0OperatorNode.create(
                node_id="unsafe",
                operator=OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH,
                parameters={"capability": capability},
                declared_operations=1,
                max_output_bytes=8,
            )

    undeclared_schema = G0OperatorNode.create(
        node_id="schema-leak",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        parameters={"result_schema_id": "silent-v1"},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="schemas not bound"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=(undeclared_schema,),
            output_node_ids=(undeclared_schema.node_id,),
        )


def test_message_edges_require_typed_emit_operators_and_cannot_be_silent() -> None:
    edge = G0MessageEdge("claim", "claim-v1", 16)
    wrong = G0OperatorNode.create(
        node_id="wrong",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        message_edge_ids=(edge.edge_id,),
        parameters={"max_results": 1},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="typed-message operators"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=(wrong,),
            output_node_ids=(wrong.node_id,),
            message_edges=(edge,),
        )

    emitters = tuple(
        G0OperatorNode.create(
            node_id=f"emit-{index}",
            operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
            message_edge_ids=(edge.edge_id,),
            parameters={"claim_schema_id": edge.schema_id, "recipient_cap": 1},
            declared_operations=1,
            max_output_bytes=edge.max_encoded_bytes,
        )
        for index in range(2)
    )
    with pytest.raises(ValueError, match="exactly one emit-operator owner"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=emitters,
            output_node_ids=tuple(node.node_id for node in emitters),
            message_edges=(edge,),
        )

    over_fanout = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        message_edge_ids=(edge.edge_id,),
        parameters={"claim_schema_id": edge.schema_id, "recipient_cap": 2},
        declared_operations=1,
        max_output_bytes=edge.max_encoded_bytes,
    )
    with pytest.raises(ValueError, match="recipient_cap"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=(over_fanout,),
            output_node_ids=(over_fanout.node_id,),
            message_edges=(edge,),
        )

    state = G0StateSlot(
        "state",
        StatePrimitive.BOUNDED_RECURRENT_STATE,
        "state-v1",
        8,
    )
    mismatched_schema = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        state_slot_ids=(state.slot_id,),
        message_edge_ids=(edge.edge_id,),
        parameters={"claim_schema_id": state.schema_id, "recipient_cap": 1},
        declared_operations=1,
        max_output_bytes=edge.max_encoded_bytes,
    )
    with pytest.raises(ValueError, match="schema does not match"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(state,),
            operator_nodes=(mismatched_schema,),
            output_node_ids=(mismatched_schema.node_id,),
            message_edges=(edge,),
        )


def test_outputs_and_state_slots_cannot_be_silent() -> None:
    node = G0OperatorNode.create(
        node_id="node",
        operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
        parameters={"max_results": 1},
        declared_operations=1,
        max_output_bytes=8,
    )
    with pytest.raises(ValueError, match="explicit output"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(),
            operator_nodes=(node,),
            output_node_ids=(),
        )
    unused = G0StateSlot(
        "unused",
        StatePrimitive.BOUNDED_RECURRENT_STATE,
        "unused-v1",
        8,
    )
    with pytest.raises(ValueError, match="every G0 state slot"):
        G0ActorGenotype.create(
            candidate_id="retrieval",
            state_slots=(unused,),
            operator_nodes=(node,),
            output_node_ids=(node.node_id,),
        )


def test_composition_caps_and_excluded_candidates_fail_structural_assessment() -> None:
    grammar = load_topology_grammar(GRAMMAR_PATH)
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    over_state = _genotype(
        state_bytes=grammar.construction_language.composition_bounds.max_state_bytes_per_actor + 1
    )
    state_assessment = assess_g0_genotype(
        over_state,
        grammar=grammar,
        candidate_registry=registry,
    )
    excluded_assessment = assess_g0_genotype(
        _genotype(candidate_id="curiosity"),
        grammar=grammar,
        candidate_registry=registry,
    )
    over_total = assess_g0_genotype(
        _genotype(state_bytes=grammar.construction_language.composition_bounds.max_state_bytes_per_actor - 1),
        grammar=grammar,
        candidate_registry=registry,
    )
    over_operations = assess_g0_genotype(
        _genotype(
            declared_operations=grammar.construction_language.composition_bounds.max_operations_per_activation
        ),
        grammar=grammar,
        candidate_registry=registry,
    )

    assert state_assessment.structurally_valid is False
    assert "state-byte-cap-exceeded" in state_assessment.blockers
    assert excluded_assessment.structurally_valid is False
    assert "candidate-excluded" in excluded_assessment.blockers
    assert "state-byte-cap-exceeded" not in over_total.blockers
    assert "actor-total-byte-cap-exceeded" in over_total.blockers
    assert "operation-cap-exceeded" in over_operations.blockers


def test_assessment_binds_registry_authority_and_never_authorizes_shadow() -> None:
    grammar = load_topology_grammar(GRAMMAR_PATH)
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    mismatched = replace(registry, candidates=tuple(reversed(registry.candidates)))

    assessment = assess_g0_genotype(
        _genotype(),
        grammar=grammar,
        candidate_registry=mismatched,
    )

    assert assessment.structurally_valid is False
    assert "candidate-registry-authority-mismatch" in assessment.blockers
    with pytest.raises(ValueError, match="cannot authorize shadow execution"):
        replace(assessment, shadow_authorized=True)
    with pytest.raises(ValueError, match="must be boolean"):
        assess_g0_genotype(
            _genotype(),
            grammar=grammar,
            candidate_registry=registry,
            freeze_authority_verified=1,  # type: ignore[arg-type]
        )


def test_genotype_identity_and_activation_bits_fail_closed() -> None:
    payload = _genotype().payload()
    payload["activation_enabled"] = True
    with pytest.raises(ValueError, match="activation-disabled"):
        G0ActorGenotype.from_payload(payload)

    payload = _genotype().payload()
    payload["operator_nodes"][0]["declared_operations"] += 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        G0ActorGenotype.from_payload(payload)

    serialized = json.loads(json.dumps(_genotype().payload()))
    assert G0ActorGenotype.from_payload(serialized) == _genotype()

    sources = tuple(
        G0OperatorNode.create(
            node_id=f"source-{suffix}",
            operator=OperatorPrimitive.BOUNDED_RETRIEVAL,
            parameters={"max_results": 1},
            declared_operations=1,
            max_output_bytes=8,
        )
        for suffix in ("a", "b")
    )
    ordered_merge = G0OperatorNode.create(
        node_id="merge",
        operator=OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION,
        input_node_ids=tuple(node.node_id for node in sources),
        parameters={"constraint_count": 1},
        declared_operations=1,
        max_output_bytes=8,
    )
    reversed_merge = replace(ordered_merge, input_node_ids=tuple(reversed(ordered_merge.input_node_ids)))
    ordered = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(),
        operator_nodes=(*sources, ordered_merge),
        output_node_ids=(ordered_merge.node_id,),
    )
    reversed_inputs = G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(),
        operator_nodes=(*sources, reversed_merge),
        output_node_ids=(reversed_merge.node_id,),
    )
    assert ordered.genotype_sha256 != reversed_inputs.genotype_sha256
