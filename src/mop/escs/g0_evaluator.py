
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

from mop.substrate.events import FrozenJSON, canonical_bytes, canonical_sha256

from .accounting import WorkVector
from .g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    assess_g0_genotype,
)
from .perspective_registry import PerspectiveCandidateRegistry
from .topology_grammar import OperatorPrimitive, StatePrimitive, TopologyGrammar

G0_EVALUATION_SCHEMA = "mop-escs-g0-counterfactual-evaluation/v1"
G0_EVALUATION_REFUSAL_SCHEMA = "mop-escs-g0-counterfactual-refusal/v1"
G0_NODE_COST_SCHEMA = "mop-escs-g0-node-cost/v1"
G0_STAGED_MESSAGE_SCHEMA = "mop-escs-g0-staged-message/v1"

_MISSING = object()
_REFERENCE_RE = re.compile(r"^[a-z][a-z0-9+.-]*:[a-z0-9][a-z0-9._:/-]*$")
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 1_000_000
_MAX_JSON_SCALAR_CHARS = 1_048_576
_PAYLOAD_FORMS = frozenset(
    {"json-list", "json-object", "json-value", "numeric-scalar", "numeric-vector"}
)
_G0_EVALUATOR_CONTRACT = {
    "schema": "mop-escs-g0-evaluator-contract/v1",
    "evaluation_schema": G0_EVALUATION_SCHEMA,
    "refusal_schema": G0_EVALUATION_REFUSAL_SCHEMA,
    "node_cost_schema": G0_NODE_COST_SCHEMA,
    "message_schema": G0_STAGED_MESSAGE_SCHEMA,
    "state_read_semantics": "immutable-activation-start-snapshot",
    "state_write_semantics": "single-writer-staged-only",
    "fan_in_semantics": "ordered-parent-values",
    "branch_identity": "full-sha256-over-complete-authority-input-state-tuple",
    "failure_accounting": "conservative-full-genotype-envelope",
    "malformed_attempt_identity": "path-sensitive-sentinel-plus-caller-stable-attempt-id",
    "numeric_domain": "finite-ieee754-with-exact-integers-through-2^53-minus-1",
    "max_json_depth": _MAX_JSON_DEPTH,
    "max_json_nodes": _MAX_JSON_NODES,
    "max_json_scalar_chars": _MAX_JSON_SCALAR_CHARS,
    "payload_forms": sorted(_PAYLOAD_FORMS),
    "activation_enabled": False,
    "factual_effects": False,
    "scientific_promotion_allowed": False,
}
G0_EVALUATOR_CONTRACT_SHA256 = canonical_sha256(_G0_EVALUATOR_CONTRACT)


class G0EvaluationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G0EvaluationError(message)


def _validate_json_shape(value: Any, label: str) -> None:

    stack: list[tuple[str, Any, int, str]] = [("visit", value, 0, "$")]
    active_containers: set[int] = set()
    nodes = 0
    while stack:
        action, nested, depth, path = stack.pop()
        if action == "leave":
            active_containers.remove(cast(int, nested))
            continue
        if action in {"dict-iterator", "list-iterator"}:
            iterator = cast(Any, nested)
            try:
                key, child = next(iterator)
            except StopIteration:
                continue
            stack.append((action, iterator, depth, path))
            if action == "dict-iterator":
                key_text = cast(str, key)
                segment = (
                    key_text
                    if len(key_text) <= 64 and re.fullmatch(r"[A-Za-z0-9_.:-]+", key_text)
                    else f"key-{hashlib.sha256(key_text.encode('utf-8')).hexdigest()[:16]}"
                )
            else:
                segment = str(key)
            stack.append(("visit", child, depth, f"{path}/{segment}"))
            continue
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise G0EvaluationError(f"{label} exceeds the JSON node cap at {path}")
        if depth > _MAX_JSON_DEPTH:
            raise G0EvaluationError(f"{label} exceeds the JSON depth cap at {path}")
        if isinstance(nested, dict):
            container_id = id(nested)
            if container_id in active_containers:
                raise G0EvaluationError(f"{label} contains a cyclic object at {path}")
            if not all(isinstance(key, str) for key in nested):
                raise G0EvaluationError(f"{label} contains a non-string JSON object key at {path}")
            if any(len(key) > _MAX_JSON_SCALAR_CHARS for key in nested):
                raise G0EvaluationError(f"{label} contains an oversized object key at {path}")
            if len(nested) > _MAX_JSON_NODES - nodes:
                raise G0EvaluationError(f"{label} exceeds the JSON node cap at {path}")
            active_containers.add(container_id)
            stack.append(("leave", container_id, depth, path))
            stack.append(("dict-iterator", iter(nested.items()), depth + 1, path))
        elif isinstance(nested, list):
            container_id = id(nested)
            if container_id in active_containers:
                raise G0EvaluationError(f"{label} contains a cyclic array at {path}")
            if len(nested) > _MAX_JSON_NODES - nodes:
                raise G0EvaluationError(f"{label} exceeds the JSON node cap at {path}")
            active_containers.add(container_id)
            stack.append(("leave", container_id, depth, path))
            stack.append(("list-iterator", iter(enumerate(nested)), depth + 1, path))
        elif isinstance(nested, str):
            if len(nested) > _MAX_JSON_SCALAR_CHARS:
                raise G0EvaluationError(f"{label} contains an oversized string at {path}")
        elif isinstance(nested, bool) or nested is None:
            continue
        elif isinstance(nested, int):
            if abs(nested) > 2**53 - 1:
                raise G0EvaluationError(f"{label} exceeds the exact integer domain at {path}")
        elif isinstance(nested, float):
            if not math.isfinite(nested):
                raise G0EvaluationError(f"{label} contains a nonfinite number at {path}")
        else:
            raise G0EvaluationError(f"{label} contains a non-JSON value at {path}")


def _snapshot(value: Any, label: str) -> Any:
    _validate_json_shape(value, label)
    try:
        return json.loads(canonical_bytes(value))
    except (RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise G0EvaluationError(f"{label} is not strict canonical JSON") from exc


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise G0EvaluationError(f"{label} must be a finite number")
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        raise G0EvaluationError(f"{label} exceeds the exact IEEE-754 integer domain")
    try:
        result = float(value)
    except OverflowError as exc:
        raise G0EvaluationError(f"{label} exceeds the finite numeric domain") from exc
    if not math.isfinite(result):
        raise G0EvaluationError(f"{label} must be a finite number")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise G0EvaluationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise G0EvaluationError(f"{label} must be a nonnegative integer")
    return value


def _vector(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise G0EvaluationError(f"{label} must be a nonempty numeric vector")
    return [_finite(item, f"{label} item") for item in value]


def _input_value(values: Sequence[Any], external: Any, label: str) -> Any:
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        return list(values)
    _require(external is not _MISSING, f"{label} requires one external root input")
    return external


def _charge(node: G0OperatorNode, operations: int) -> int:

    _require(operations > 0, f"{node.node_id} reported no work")
    _require(operations <= node.declared_operations, f"{node.node_id} exceeded declared operations")
    return operations


def _require_state_primitive(
    node: G0OperatorNode,
    declared_states: Mapping[str, Any],
    allowed: frozenset[StatePrimitive],
    *,
    optional: bool = False,
) -> None:
    expected_counts = {0, 1} if optional else {1}
    _require(
        len(node.state_slot_ids) in expected_counts,
        f"{node.node_id} has an invalid state-slot arity",
    )
    if node.state_slot_ids:
        primitive = declared_states[node.state_slot_ids[0]].primitive
        _require(
            primitive in allowed,
            f"{node.node_id} state primitive {primitive.value} is incompatible",
        )


def _frozen_from_payload(value: object, label: str) -> FrozenJSON:
    if not isinstance(value, Mapping):
        raise G0EvaluationError(f"{label} must be a frozen JSON payload")
    row = cast(Mapping[str, Any], value)
    _require(set(row) == {"value", "sha256"}, f"{label} fields mismatch")
    snapshot = _snapshot(row["value"], label)
    frozen = FrozenJSON.from_value(snapshot)
    _require(frozen.sha256 == row["sha256"], f"{label} digest mismatch")
    return frozen


def _counterfactual_branch_id(
    *,
    genotype_sha256: str,
    grammar_sha256: str,
    candidate_registry_sha256: str,
    evaluator_contract_sha256: str,
    input_sha256: str,
    initial_state_sha256: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema": "mop-escs-g0-counterfactual-branch/v1",
            "genotype_sha256": genotype_sha256,
            "grammar_sha256": grammar_sha256,
            "candidate_registry_sha256": candidate_registry_sha256,
            "evaluator_contract_sha256": evaluator_contract_sha256,
            "input_sha256": input_sha256,
            "initial_state_sha256": initial_state_sha256,
        }
    )
    return f"branch:g0-shadow/{digest}"


def _require_parameter_keys(
    parameters: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    label: str,
) -> None:
    actual = set(parameters)
    allowed = required | set(optional)
    _require(
        required <= actual <= allowed,
        f"{label} parameter fields mismatch; "
        f"missing={sorted(required - actual)}, extra={sorted(actual - allowed)}",
    )


def _validate_operator_parameters(
    node: G0OperatorNode,
    parameters: Mapping[str, Any],
    declared_states: Mapping[str, Any],
) -> None:
    operator = node.operator
    if operator is OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE:
        _require_parameter_keys(
            parameters,
            required={"weights", "bias", "activation", "state_mode"},
            label="affine",
        )
        state_mode = parameters["state_mode"]
        if not node.state_slot_ids:
            _require(state_mode == "none", "stateless affine requires state_mode=none")
        else:
            primitive = declared_states[node.state_slot_ids[0]].primitive
            expected = (
                "concatenate"
                if primitive is StatePrimitive.BOUNDED_RECURRENT_STATE
                else "replace"
            )
            _require(state_mode == expected, f"affine {primitive.value} requires state_mode={expected}")
    elif operator is OperatorPrimitive.BOUNDED_RETRIEVAL:
        _require_parameter_keys(
            parameters,
            required={"query_key", "max_results"},
            label="retrieval",
        )
    elif operator is OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH:
        _require_parameter_keys(
            parameters,
            required={"transition_table", "max_depth", "branch_factor"},
            label="rollout",
        )
    elif operator is OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION:
        _require_parameter_keys(
            parameters,
            required={"allowed_values"},
            label="constraint",
        )
    elif operator is OperatorPrimitive.GRAPH_NEIGHBORHOOD_AGGREGATION:
        _require_parameter_keys(
            parameters,
            required={"max_neighbors", "reducer"},
            label="graph aggregation",
        )
    elif operator is OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE:
        mode = parameters.get("mode")
        required = {"mode"} if mode == "lookup" else {"mode", "max_entries"}
        _require_parameter_keys(parameters, required=required, label="table")
        _require(mode in {"lookup", "stage_update"}, "table mode is not declared")
    elif operator is OperatorPrimitive.TEMPORAL_ACCUMULATION:
        _require_parameter_keys(
            parameters,
            required={"window", "reducer"},
            label="temporal",
        )
    elif operator is OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT:
        if "emissions" in parameters:
            _require_parameter_keys(
                parameters,
                required={"recipient_cap", "emissions"},
                label="plural message emit",
            )
        else:
            _require_parameter_keys(
                parameters,
                required={"schema_id", "recipient", "payload_form"},
                optional={"recipient_cap"},
                label="message emit",
            )


def _state_value(
    node: G0OperatorNode,
    state: Mapping[str, Any],
    *,
    exactly_one: bool = True,
) -> tuple[str, Any]:
    if exactly_one:
        _require(len(node.state_slot_ids) == 1, f"{node.node_id} requires exactly one state slot")
    else:
        _require(bool(node.state_slot_ids), f"{node.node_id} requires a state slot")
    slot_id = node.state_slot_ids[0]
    _require(slot_id in state, f"{node.node_id} state slot is missing")
    return slot_id, state[slot_id]


def _topological_order(nodes: Sequence[G0OperatorNode]) -> tuple[G0OperatorNode, ...]:
    by_id = {node.node_id: node for node in nodes}
    pending = {node.node_id: set(node.input_node_ids) for node in nodes}
    order: list[G0OperatorNode] = []
    completed: set[str] = set()
    while pending:
        ready = sorted(node_id for node_id, parents in pending.items() if parents <= completed)
        if not ready:
            raise G0EvaluationError("G0 evaluator found a cyclic operator graph")
        for node_id in ready:
            order.append(by_id[node_id])
            completed.add(node_id)
            del pending[node_id]
    return tuple(order)


def _affine(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    source = _input_value(inputs, external, node.node_id)
    if not isinstance(source, list) or not source:
        raise G0EvaluationError(f"{node.node_id} input must be a nonempty numeric vector")
    input_vector = _vector(source, f"{node.node_id} input")
    state_mode = parameters["state_mode"]
    if state_mode == "concatenate":
        _slot_id, prior_state = _state_value(node, state)
        input_vector.extend(_vector(prior_state, f"{node.node_id} recurrent state"))
    weights = parameters.get("weights")
    bias = parameters.get("bias")
    activation = parameters.get("activation", "identity")
    if not isinstance(weights, list) or not weights:
        raise G0EvaluationError("affine weights must be a nonempty matrix")
    if not isinstance(bias, list) or len(bias) != len(weights):
        raise G0EvaluationError("affine bias shape mismatch")
    if not isinstance(activation, str) or activation not in {"identity", "relu", "softsign"}:
        raise G0EvaluationError("affine activation is not declared")
    activation_cost = {"identity": 0, "relu": 1, "softsign": 3}[activation]
    estimated_operations = len(weights) * (2 * len(input_vector) + activation_cost)
    _charge(node, max(1, estimated_operations))
    output: list[float] = []
    operations = 0
    for row, raw_bias in zip(weights, bias, strict=True):
        numeric_row = _vector(row, "affine weight row")
        _require(len(numeric_row) == len(input_vector), "affine input width mismatch")
        value = _finite(raw_bias, "affine bias")
        for weight, item in zip(numeric_row, input_vector, strict=True):
            product = _finite(weight * item, "affine product")
            value = _finite(value + product, "affine accumulation")
            operations += 2
            _charge(node, operations)
        if activation == "softsign":
            value = _finite(value / (1.0 + abs(value)), "affine softsign")
            operations += 3
        elif activation == "relu":
            value = max(0.0, value)
            operations += 1
        elif activation != "identity":  # guarded above; keeps enum-like exhaustiveness local
            raise G0EvaluationError("affine activation is not declared")
        _charge(node, max(1, operations))
        output.append(value)
    updates: dict[str, Any] = {}
    if node.state_slot_ids:
        slot_id, _old_value = _state_value(node, state)
        updates[slot_id] = output
    return output, updates, [], _charge(node, max(1, operations))


def _retrieval(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    _slot_id, records = _state_value(node, state)
    _require(isinstance(records, list), "bounded retrieval state must be a record list")
    query = _input_value(inputs, external, node.node_id)
    key = parameters.get("query_key")
    cap = _positive_int(parameters.get("max_results"), "retrieval max_results")
    _require(isinstance(key, str) and bool(key), "retrieval query_key must be text")
    query_sha256 = canonical_sha256(query)
    selected: list[Any] = []
    inspected = 0
    for record in records:
        inspected += 1
        _charge(node, inspected)
        _require(isinstance(record, dict), "bounded retrieval record must be an object")
        record_value = record.get(key, _MISSING)
        if record_value is not _MISSING and canonical_sha256(record_value) == query_sha256:
            selected.append(_snapshot(record, "retrieval record"))
            if len(selected) == cap:
                break
    return selected, {}, [], _charge(node, max(1, inspected))


def _rollout(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    start = _input_value(inputs, external, node.node_id)
    _require(isinstance(start, str), "bounded rollout start state must be text")
    transitions = parameters.get("transition_table")
    depth_cap = _positive_int(parameters.get("max_depth"), "rollout max_depth")
    branch_cap = _positive_int(parameters.get("branch_factor"), "rollout branch_factor")
    if not isinstance(transitions, dict):
        raise G0EvaluationError("bounded rollout transition table must be an object")
    frontier = [start]
    visited = [start]
    visited_encoded_bytes = 2 + len(canonical_bytes(start))
    _require(visited_encoded_bytes <= node.max_output_bytes, f"{node.node_id} output cap exceeded")
    operations = 0
    for _depth in range(depth_cap):
        next_frontier: list[str] = []
        for state_id in frontier:
            children = transitions.get(state_id, [])
            _require(isinstance(children, list), "rollout transition row must be a list")
            for child in islice(children, branch_cap):
                _require(isinstance(child, str), "rollout child state must be text")
                next_frontier.append(child)
                visited.append(child)
                visited_encoded_bytes += 1 + len(canonical_bytes(child))
                operations += 1
                _charge(node, operations)
                _require(
                    visited_encoded_bytes <= node.max_output_bytes,
                    f"{node.node_id} output cap exceeded during rollout",
                )
        frontier = next_frontier
        if not frontier:
            break
    return visited, {}, [], _charge(node, max(1, operations))


def _constraint_filter(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    candidates = _input_value(inputs, external, node.node_id)
    allowed = parameters.get("allowed_values")
    if not isinstance(candidates, list):
        raise G0EvaluationError("constraint input must be a list")
    if not isinstance(allowed, list):
        raise G0EvaluationError("constraint allowed_values must be a list")
    operations = max(1, len(candidates) + len(allowed))
    _charge(node, operations)
    allowed_digests = {canonical_sha256(value) for value in allowed}
    result = [value for value in candidates if canonical_sha256(value) in allowed_digests]
    return result, {}, [], operations


def _graph_aggregate(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    _slot_id, graph = _state_value(node, state)
    _require(isinstance(graph, dict), "graph aggregation state must be an object")
    target = _input_value(inputs, external, node.node_id)
    _require(isinstance(target, str) and target in graph, "graph target is missing")
    row = graph[target]
    _require(isinstance(row, dict), "graph target row must be an object")
    neighbors = row.get("neighbors")
    cap = _positive_int(parameters.get("max_neighbors"), "graph max_neighbors")
    reducer = parameters.get("reducer", "mean")
    _require(isinstance(neighbors, list), "graph neighbors must be a list")
    neighbor_rows = cast(list[Any], neighbors)
    _charge(node, max(1, len(neighbor_rows)))
    _require(all(isinstance(neighbor, str) for neighbor in neighbor_rows), "graph neighbor is invalid")
    selected_neighbors = sorted(cast(list[str], neighbor_rows))[:cap]
    raw_vectors: list[list[Any]] = []
    for neighbor in selected_neighbors:
        _require(neighbor in graph, "graph neighbor is missing")
        neighbor_row = graph[neighbor]
        _require(isinstance(neighbor_row, dict), "graph neighbor row must be an object")
        raw_vector = neighbor_row.get("value")
        _require(
            isinstance(raw_vector, list) and bool(raw_vector),
            "graph neighbor value must be a nonempty vector",
        )
        raw_vectors.append(cast(list[Any], raw_vector))
    _require(bool(raw_vectors), "graph aggregation requires at least one bounded neighbor")
    width = len(raw_vectors[0])
    _require(all(len(vector) == width for vector in raw_vectors), "graph vector width mismatch")
    operations = len(neighbor_rows) + len(raw_vectors) * width
    if reducer == "mean":
        operations += width
    elif reducer != "sum":
        raise G0EvaluationError("graph reducer is not declared")
    _charge(node, max(1, operations))
    vectors: list[list[float]] = []
    for raw_vector in raw_vectors:
        vectors.append(_vector(raw_vector, "graph neighbor value"))
    output: list[float] = []
    for index in range(width):
        total = 0.0
        for vector in vectors:
            total = _finite(total + vector[index], "graph accumulation")
        output.append(total)
    if reducer == "mean":
        output = [_finite(value / len(vectors), "graph mean") for value in output]
    return output, {}, [], _charge(node, max(1, operations))


def _table(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    slot_id, table = _state_value(node, state)
    _require(isinstance(table, dict), "table state must be an object")
    mode = parameters.get("mode")
    source = _input_value(inputs, external, node.node_id)
    if mode == "lookup":
        _require(isinstance(source, str), "table lookup key must be text")
        return _snapshot(table.get(source), "table result"), {}, [], 1
    if mode == "stage_update":
        _require(isinstance(source, list) and len(source) == 2, "table update requires [key, value]")
        key, value = source
        _require(isinstance(key, str), "table update key must be text")
        cap = _positive_int(parameters.get("max_entries"), "table max_entries")
        resulting_entries = len(table) + int(key not in table)
        _require(resulting_entries <= cap, "table update exceeds its entry cap")
        operations = _charge(node, max(1, len(table) + 1))
        updated = _snapshot(table, "table state")
        updated[key] = _snapshot(value, "table update value")
        return updated, {slot_id: updated}, [], operations
    raise G0EvaluationError("table mode is not declared")


def _temporal(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    slot_id, history = _state_value(node, state)
    _require(isinstance(history, list), "temporal state must be a list")
    value = _finite(_input_value(inputs, external, node.node_id), "temporal input")
    window = _positive_int(parameters.get("window"), "temporal window")
    reducer = parameters.get("reducer", "mean")
    retained_history = history[-max(0, window - 1) :] if window > 1 else []
    operations = _charge(node, max(1, len(retained_history) + 1))
    updated = [*(_finite(item, "temporal state item") for item in retained_history), value]
    if reducer == "mean":
        total = 0.0
        for item in updated:
            total = _finite(total + item, "temporal accumulation")
        result = _finite(total / len(updated), "temporal mean")
    elif reducer == "sum":
        result = 0.0
        for item in updated:
            result = _finite(result + item, "temporal accumulation")
    elif reducer == "last":
        result = updated[-1]
    else:
        raise G0EvaluationError("temporal reducer is not declared")
    return result, {slot_id: updated}, [], operations


def _validate_payload_shape(payload: Any, payload_form: str) -> None:
    if payload_form == "numeric-scalar":
        _finite(payload, "message numeric-scalar payload")
    elif payload_form == "numeric-vector":
        _vector(payload, "message numeric-vector payload")
    elif payload_form == "json-list":
        _require(isinstance(payload, list), "message payload must be a list")
    elif payload_form == "json-object":
        _require(isinstance(payload, dict), "message payload must be an object")
    elif payload_form != "json-value":
        raise G0EvaluationError("message payload form is not declared")


def _message(
    node: G0OperatorNode,
    inputs: Sequence[Any],
    external: Any,
    parameters: Mapping[str, Any],
    edges: Mapping[str, G0MessageEdge],
    *,
    genotype_sha256: str,
    grammar_sha256: str,
    candidate_registry_sha256: str,
    evaluator_contract_sha256: str,
    input_sha256: str,
    initial_state_sha256: str,
    counterfactual_branch_id: str,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], int]:
    payload = _input_value(inputs, external, node.node_id)
    raw_emissions = parameters.get("emissions")
    if raw_emissions is None:
        _require(len(node.message_edge_ids) == 1, "plural emit requires explicit emissions")
        raw_emissions = [
            {
                "edge_id": node.message_edge_ids[0],
                "schema_id": parameters.get("schema_id"),
                "recipient": parameters.get("recipient"),
                "payload_form": parameters.get("payload_form"),
            }
        ]
    _require(isinstance(raw_emissions, list), "message emissions must be a list")
    emissions = cast(list[Any], raw_emissions)
    _require(len(emissions) == len(node.message_edge_ids), "message emission count mismatch")
    expected_fields = {"edge_id", "schema_id", "recipient", "payload_form"}
    normalized: list[dict[str, str]] = []
    for raw_emission in emissions:
        _require(isinstance(raw_emission, dict), "message emission must be an object")
        _require(set(raw_emission) == expected_fields, "message emission fields mismatch")
        emission = cast(dict[str, Any], raw_emission)
        edge_id = emission["edge_id"]
        schema_id = emission["schema_id"]
        recipient = emission["recipient"]
        payload_form = emission["payload_form"]
        _require(isinstance(edge_id, str) and edge_id in edges, "message edge is undeclared")
        edge = edges[edge_id]
        _require(schema_id == edge.schema_id, "message schema does not match its declared edge")
        _require(
            isinstance(recipient, str)
            and recipient.startswith("actor:")
            and _REFERENCE_RE.fullmatch(recipient) is not None,
            "message recipient must be a stable actor reference",
        )
        _require(
            isinstance(payload_form, str) and payload_form in _PAYLOAD_FORMS,
            "message payload form is not declared",
        )
        normalized.append(
            {
                "edge_id": edge_id,
                "schema_id": cast(str, schema_id),
                "recipient": recipient,
                "payload_form": cast(str, payload_form),
            }
        )
    _require(
        tuple(row["edge_id"] for row in normalized) == node.message_edge_ids,
        "message emissions must follow canonical edge order",
    )
    payload_snapshot = _snapshot(payload, "staged message payload")
    for row in normalized:
        _validate_payload_shape(payload_snapshot, row["payload_form"])
    encoded = canonical_bytes(payload)
    messages: list[dict[str, Any]] = []
    for row in normalized:
        edge = edges[row["edge_id"]]
        _require(len(encoded) <= edge.max_encoded_bytes, "staged message exceeds its edge byte cap")
        core = {
            "schema": G0_STAGED_MESSAGE_SCHEMA,
            "genotype_sha256": genotype_sha256,
            "grammar_sha256": grammar_sha256,
            "candidate_registry_sha256": candidate_registry_sha256,
            "evaluator_contract_sha256": evaluator_contract_sha256,
            "input_sha256": input_sha256,
            "initial_state_sha256": initial_state_sha256,
            "counterfactual_branch_id": counterfactual_branch_id,
            "node_id": node.node_id,
            **row,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "payload": payload_snapshot,
            "encoded_bytes": len(encoded),
            "counterfactual_only": True,
            "activation_enabled": False,
            "factual_effects": False,
            "scientific_promotion_allowed": False,
        }
        messages.append({**core, "message_sha256": canonical_sha256(core)})
    return payload_snapshot, {}, messages, _charge(node, max(1, len(messages)))


def _validated_staged_message(message: FrozenJSON) -> dict[str, Any]:
    value = message.value()
    _require(isinstance(value, dict), "staged message value must be an object")
    row = cast(dict[str, Any], value)
    expected = {
        "schema",
        "genotype_sha256",
        "grammar_sha256",
        "candidate_registry_sha256",
        "evaluator_contract_sha256",
        "input_sha256",
        "initial_state_sha256",
        "counterfactual_branch_id",
        "node_id",
        "edge_id",
        "schema_id",
        "recipient",
        "payload_form",
        "payload_sha256",
        "payload",
        "encoded_bytes",
        "counterfactual_only",
        "activation_enabled",
        "factual_effects",
        "scientific_promotion_allowed",
        "message_sha256",
    }
    _require(set(row) == expected, "staged message fields mismatch")
    _require(row["schema"] == G0_STAGED_MESSAGE_SCHEMA, "staged message schema mismatch")
    core = {key: value for key, value in row.items() if key != "message_sha256"}
    _require(row["message_sha256"] == canonical_sha256(core), "staged message self-hash mismatch")
    for key in (
        "genotype_sha256",
        "grammar_sha256",
        "candidate_registry_sha256",
        "evaluator_contract_sha256",
        "input_sha256",
        "initial_state_sha256",
        "payload_sha256",
        "message_sha256",
    ):
        digest = row[key]
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"staged message {key} is invalid",
        )
    for key in ("node_id", "edge_id", "schema_id"):
        identifier = row[key]
        _require(
            isinstance(identifier, str)
            and re.fullmatch(r"[a-z][a-z0-9_.-]*", identifier) is not None,
            f"staged message {key} is invalid",
        )
    _require(
        isinstance(row["recipient"], str)
        and row["recipient"].startswith("actor:")
        and _REFERENCE_RE.fullmatch(row["recipient"]) is not None,
        "staged message recipient is invalid",
    )
    payload_form = row["payload_form"]
    _require(
        isinstance(payload_form, str) and payload_form in _PAYLOAD_FORMS,
        "staged message payload form is invalid",
    )
    payload_bytes = canonical_bytes(row["payload"])
    _require(
        row["encoded_bytes"] == len(payload_bytes),
        "staged message encoded byte count mismatch",
    )
    _require(
        row["payload_sha256"] == hashlib.sha256(payload_bytes).hexdigest(),
        "staged message payload hash mismatch",
    )
    _validate_payload_shape(row["payload"], payload_form)
    _require(row["counterfactual_only"] is True, "staged message escaped counterfactual status")
    _require(row["activation_enabled"] is False, "staged message enabled activation")
    _require(row["factual_effects"] is False, "staged message enabled factual effects")
    _require(
        row["scientific_promotion_allowed"] is False,
        "staged message granted scientific promotion",
    )
    return row


@dataclass(frozen=True, slots=True)
class G0CounterfactualEvaluation:
    genotype_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    evaluator_contract_sha256: str
    input_sha256: str
    initial_state_sha256: str
    external_inputs: FrozenJSON
    initial_state: FrozenJSON
    node_outputs: FrozenJSON
    node_costs: FrozenJSON
    staged_state: FrozenJSON
    staged_messages: tuple[FrozenJSON, ...]
    operations: int
    input_bytes: int
    initial_state_bytes: int
    node_output_bytes: int
    staged_state_bytes: int
    message_count: int
    message_payload_bytes: int
    message_envelope_bytes: int
    work: WorkVector
    counterfactual_branch_id: str
    counterfactual_only: bool
    activation_enabled: bool
    factual_effects: bool
    scientific_promotion_allowed: bool
    evaluation_sha256: str
    schema: str = G0_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_EVALUATION_SCHEMA, "unsupported G0 evaluation schema")
        for digest_value, label in (
            (self.genotype_sha256, "genotype sha256"),
            (self.grammar_sha256, "grammar sha256"),
            (self.candidate_registry_sha256, "candidate registry sha256"),
            (self.evaluator_contract_sha256, "evaluator contract sha256"),
            (self.input_sha256, "input sha256"),
            (self.initial_state_sha256, "initial state sha256"),
            (self.evaluation_sha256, "evaluation sha256"),
        ):
            _require(
                isinstance(digest_value, str)
                and len(digest_value) == 64
                and all(character in "0123456789abcdef" for character in digest_value),
                f"{label} is invalid",
            )
        _require(isinstance(self.external_inputs, FrozenJSON), "external inputs must be frozen")
        _require(isinstance(self.initial_state, FrozenJSON), "initial state must be frozen")
        _require(isinstance(self.node_outputs, FrozenJSON), "node outputs must be frozen")
        _require(
            self.evaluator_contract_sha256 == G0_EVALUATOR_CONTRACT_SHA256,
            "evaluator contract authority mismatch",
        )
        _require(isinstance(self.node_costs, FrozenJSON), "node costs must be frozen")
        _require(isinstance(self.staged_state, FrozenJSON), "staged state must be frozen")
        _require(
            isinstance(self.staged_messages, tuple)
            and all(isinstance(message, FrozenJSON) for message in self.staged_messages),
            "staged messages must be immutable frozen JSON",
        )
        external_value = self.external_inputs.value()
        initial_state_value = self.initial_state.value()
        output_value = self.node_outputs.value()
        cost_value = self.node_costs.value()
        state_value = self.staged_state.value()
        _require(isinstance(external_value, dict), "external inputs must contain an object")
        _require(isinstance(initial_state_value, dict), "initial state must contain an object")
        _require(isinstance(output_value, dict), "node outputs must contain an object")
        _require(isinstance(cost_value, list), "node costs must contain a list")
        _require(isinstance(state_value, dict), "staged state must contain an object")
        _positive_int(self.operations, "operations")
        for count_value, label in (
            (self.input_bytes, "input_bytes"),
            (self.initial_state_bytes, "initial_state_bytes"),
            (self.node_output_bytes, "node_output_bytes"),
            (self.staged_state_bytes, "staged_state_bytes"),
            (self.message_count, "message_count"),
            (self.message_payload_bytes, "message_payload_bytes"),
            (self.message_envelope_bytes, "message_envelope_bytes"),
        ):
            _nonnegative_int(count_value, label)
        for byte_count, label in (
            (self.input_bytes, "input_bytes"),
            (self.initial_state_bytes, "initial_state_bytes"),
            (self.node_output_bytes, "node_output_bytes"),
            (self.staged_state_bytes, "staged_state_bytes"),
        ):
            _positive_int(byte_count, label)
        expected_output_bytes = sum(
            len(canonical_bytes(value)) for value in cast(dict[str, Any], output_value).values()
        )
        _require(self.node_output_bytes == expected_output_bytes, "node-output byte count mismatch")
        _require(
            self.staged_state_bytes == len(canonical_bytes(state_value)),
            "staged-state byte count mismatch",
        )
        _require(self.external_inputs.sha256 == self.input_sha256, "input snapshot hash mismatch")
        _require(
            self.initial_state.sha256 == self.initial_state_sha256,
            "initial-state snapshot hash mismatch",
        )
        _require(
            self.input_bytes == len(canonical_bytes(external_value)),
            "input snapshot byte count mismatch",
        )
        _require(
            self.initial_state_bytes == len(canonical_bytes(initial_state_value)),
            "initial-state snapshot byte count mismatch",
        )
        cost_rows = cast(list[Any], cost_value)
        expected_cost_fields = {
            "schema",
            "node_id",
            "operator",
            "declared_operations",
            "used_operations",
            "output_bytes",
            "message_count",
            "message_payload_bytes",
            "status",
            "node_cost_sha256",
        }
        seen_cost_nodes: set[str] = set()
        validated_cost_rows: list[dict[str, Any]] = []
        for raw_row in cost_rows:
            _require(isinstance(raw_row, dict), "node cost row must be an object")
            row = cast(dict[str, Any], raw_row)
            _require(set(row) == expected_cost_fields, "node cost row fields mismatch")
            _require(row["schema"] == G0_NODE_COST_SCHEMA, "node cost schema mismatch")
            _require(row["status"] == "completed", "successful evaluation has incomplete node cost")
            node_id = row["node_id"]
            _require(
                isinstance(node_id, str)
                and re.fullmatch(r"[a-z][a-z0-9_.-]*", node_id) is not None
                and node_id not in seen_cost_nodes,
                "node cost identity is invalid",
            )
            seen_cost_nodes.add(node_id)
            for key in (
                "declared_operations",
                "used_operations",
                "output_bytes",
                "message_count",
                "message_payload_bytes",
            ):
                _nonnegative_int(row[key], f"node cost {key}")
            _require(row["used_operations"] > 0, "node cost used_operations must be positive")
            _require(
                isinstance(row["operator"], str)
                and row["operator"] in {operator.value for operator in OperatorPrimitive},
                "node cost operator is not declared",
            )
            _require(
                row["used_operations"] <= row["declared_operations"],
                "node cost exceeded its declaration",
            )
            core = {key: value for key, value in row.items() if key != "node_cost_sha256"}
            _require(
                row["node_cost_sha256"] == canonical_sha256(core),
                "node cost self-hash mismatch",
            )
            validated_cost_rows.append(row)
        _require(set(output_value) == seen_cost_nodes, "node cost/output coverage mismatch")
        for row in validated_cost_rows:
            _require(
                row["output_bytes"]
                == len(canonical_bytes(cast(dict[str, Any], output_value)[row["node_id"]])),
                "node cost output bytes do not match node output",
            )
        _require(
            sum(cast(int, row["used_operations"]) for row in validated_cost_rows)
            == self.operations,
            "node cost operation total mismatch",
        )
        _require(
            sum(cast(int, row["output_bytes"]) for row in validated_cost_rows)
            == self.node_output_bytes,
            "node cost output-byte total mismatch",
        )
        message_rows = [_validated_staged_message(message) for message in self.staged_messages]
        message_ids = [cast(str, row["message_sha256"]) for row in message_rows]
        message_keys = [
            (cast(str, row["node_id"]), cast(str, row["edge_id"])) for row in message_rows
        ]
        _require(len(message_ids) == len(set(message_ids)), "staged message identity is duplicated")
        _require(len(message_keys) == len(set(message_keys)), "staged node/edge emission is duplicated")
        cost_node_order = [cast(str, row["node_id"]) for row in validated_cost_rows]
        cost_nodes = set(cost_node_order)
        _require(
            all(node_id in cost_nodes for node_id, _edge_id in message_keys),
            "staged message node lacks a cost row",
        )
        expected_message_order: list[tuple[str, str]] = []
        for node_id in cost_node_order:
            expected_message_order.extend(
                sorted(key for key in message_keys if key[0] == node_id)
            )
        _require(message_keys == expected_message_order, "staged message order is not canonical")
        for cost_row in validated_cost_rows:
            node_messages = [row for row in message_rows if row["node_id"] == cost_row["node_id"]]
            _require(
                cost_row["message_count"] == len(node_messages),
                "node cost message count mismatch",
            )
            _require(
                cost_row["message_payload_bytes"]
                == sum(cast(int, row["encoded_bytes"]) for row in node_messages),
                "node cost grouped message-byte mismatch",
            )
        _require(
            all(row["genotype_sha256"] == self.genotype_sha256 for row in message_rows),
            "staged message genotype authority mismatch",
        )
        _require(
            all(row["grammar_sha256"] == self.grammar_sha256 for row in message_rows),
            "staged message grammar authority mismatch",
        )
        _require(
            all(
                row["candidate_registry_sha256"] == self.candidate_registry_sha256
                for row in message_rows
            ),
            "staged message registry authority mismatch",
        )
        _require(
            all(
                row["evaluator_contract_sha256"] == self.evaluator_contract_sha256
                for row in message_rows
            ),
            "staged message evaluator authority mismatch",
        )
        _require(
            all(row["input_sha256"] == self.input_sha256 for row in message_rows),
            "staged message input authority mismatch",
        )
        _require(
            all(row["initial_state_sha256"] == self.initial_state_sha256 for row in message_rows),
            "staged message state authority mismatch",
        )
        _require(
            self.work == WorkVector(actor_execution=self.operations, messages=self.message_count),
            "work mismatch",
        )
        _require(self.message_count == len(self.staged_messages), "message count mismatch")
        _require(
            sum(cast(int, row["message_count"]) for row in validated_cost_rows)
            == self.message_count,
            "node cost message total mismatch",
        )
        _require(
            self.message_payload_bytes
            == sum(cast(int, row["encoded_bytes"]) for row in message_rows),
            "message payload byte count mismatch",
        )
        _require(
            sum(cast(int, row["message_payload_bytes"]) for row in validated_cost_rows)
            == self.message_payload_bytes,
            "node cost message-byte total mismatch",
        )
        _require(
            self.message_envelope_bytes
            == sum(len(canonical_bytes(row)) for row in message_rows),
            "message envelope byte count mismatch",
        )
        expected_branch = _counterfactual_branch_id(
            genotype_sha256=self.genotype_sha256,
            grammar_sha256=self.grammar_sha256,
            candidate_registry_sha256=self.candidate_registry_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            input_sha256=self.input_sha256,
            initial_state_sha256=self.initial_state_sha256,
        )
        _require(self.counterfactual_branch_id == expected_branch, "branch identity mismatch")
        _require(
            all(row["counterfactual_branch_id"] == expected_branch for row in message_rows),
            "staged message branch mismatch",
        )
        _require(self.counterfactual_only is True, "evaluation escaped counterfactual status")
        _require(self.activation_enabled is False, "evaluation activated a genotype")
        _require(self.factual_effects is False, "evaluation produced a factual effect")
        _require(self.scientific_promotion_allowed is False, "evaluation granted promotion")
        _require(
            self.evaluation_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 evaluation self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "genotype_sha256": self.genotype_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "input_sha256": self.input_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "external_inputs": self.external_inputs.payload(),
            "initial_state": self.initial_state.payload(),
            "node_outputs": self.node_outputs.payload(),
            "node_costs": self.node_costs.payload(),
            "staged_state": self.staged_state.payload(),
            "staged_messages": [message.payload() for message in self.staged_messages],
            "operations": self.operations,
            "input_bytes": self.input_bytes,
            "initial_state_bytes": self.initial_state_bytes,
            "node_output_bytes": self.node_output_bytes,
            "staged_state_bytes": self.staged_state_bytes,
            "message_count": self.message_count,
            "message_payload_bytes": self.message_payload_bytes,
            "message_envelope_bytes": self.message_envelope_bytes,
            "work": self.work.payload(),
            "counterfactual_branch_id": self.counterfactual_branch_id,
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "factual_effects": self.factual_effects,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["evaluation_sha256"] = self.evaluation_sha256
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> G0CounterfactualEvaluation:
        expected = {
            "schema",
            "genotype_sha256",
            "grammar_sha256",
            "candidate_registry_sha256",
            "evaluator_contract_sha256",
            "input_sha256",
            "initial_state_sha256",
            "external_inputs",
            "initial_state",
            "node_outputs",
            "node_costs",
            "staged_state",
            "staged_messages",
            "operations",
            "input_bytes",
            "initial_state_bytes",
            "node_output_bytes",
            "staged_state_bytes",
            "message_count",
            "message_payload_bytes",
            "message_envelope_bytes",
            "work",
            "counterfactual_branch_id",
            "counterfactual_only",
            "activation_enabled",
            "factual_effects",
            "scientific_promotion_allowed",
            "evaluation_sha256",
        }
        _require(set(payload) == expected, "G0 evaluation fields mismatch")
        messages = payload["staged_messages"]
        _require(isinstance(messages, list), "staged_messages must be a list")
        work = payload["work"]
        _require(isinstance(work, Mapping), "work must be a mapping")
        return cls(
            schema=payload["schema"],
            genotype_sha256=payload["genotype_sha256"],
            grammar_sha256=payload["grammar_sha256"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            evaluator_contract_sha256=payload["evaluator_contract_sha256"],
            input_sha256=payload["input_sha256"],
            initial_state_sha256=payload["initial_state_sha256"],
            external_inputs=_frozen_from_payload(payload["external_inputs"], "external_inputs"),
            initial_state=_frozen_from_payload(payload["initial_state"], "initial_state"),
            node_outputs=_frozen_from_payload(payload["node_outputs"], "node_outputs"),
            node_costs=_frozen_from_payload(payload["node_costs"], "node_costs"),
            staged_state=_frozen_from_payload(payload["staged_state"], "staged_state"),
            staged_messages=tuple(
                _frozen_from_payload(message, "staged message") for message in messages
            ),
            operations=payload["operations"],
            input_bytes=payload["input_bytes"],
            initial_state_bytes=payload["initial_state_bytes"],
            node_output_bytes=payload["node_output_bytes"],
            staged_state_bytes=payload["staged_state_bytes"],
            message_count=payload["message_count"],
            message_payload_bytes=payload["message_payload_bytes"],
            message_envelope_bytes=payload["message_envelope_bytes"],
            work=WorkVector.from_payload(work),
            counterfactual_branch_id=payload["counterfactual_branch_id"],
            counterfactual_only=payload["counterfactual_only"],
            activation_enabled=payload["activation_enabled"],
            factual_effects=payload["factual_effects"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            evaluation_sha256=payload["evaluation_sha256"],
        )


@dataclass(frozen=True, slots=True)
class G0CounterfactualRefusal:

    genotype_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    evaluator_contract_sha256: str
    input_sha256: str
    initial_state_sha256: str
    external_inputs: FrozenJSON | None
    initial_state: FrozenJSON | None
    attempt_id: str
    reason: str
    charged_operations: int
    declared_message_bytes_at_risk: int
    input_bytes: int
    initial_state_bytes: int
    input_identity_complete: bool
    initial_state_identity_complete: bool
    work: WorkVector
    counterfactual_branch_id: str
    accounting_mode: str
    status: str
    counterfactual_only: bool
    activation_enabled: bool
    factual_effects: bool
    scientific_promotion_allowed: bool
    refusal_sha256: str
    schema: str = G0_EVALUATION_REFUSAL_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_EVALUATION_REFUSAL_SCHEMA, "unsupported G0 refusal schema")
        for value, label in (
            (self.genotype_sha256, "genotype sha256"),
            (self.grammar_sha256, "grammar sha256"),
            (self.candidate_registry_sha256, "candidate registry sha256"),
            (self.evaluator_contract_sha256, "evaluator contract sha256"),
            (self.input_sha256, "input sha256"),
            (self.initial_state_sha256, "initial state sha256"),
            (self.refusal_sha256, "refusal sha256"),
        ):
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"{label} is invalid",
            )
        _require(
            isinstance(self.attempt_id, str)
            and _REFERENCE_RE.fullmatch(self.attempt_id) is not None,
            "refusal attempt_id must be a stable reference",
        )
        _require(isinstance(self.reason, str) and bool(self.reason), "refusal reason is empty")
        _require(
            self.evaluator_contract_sha256 == G0_EVALUATOR_CONTRACT_SHA256,
            "refusal evaluator contract authority mismatch",
        )
        _positive_int(self.charged_operations, "charged_operations")
        _nonnegative_int(self.declared_message_bytes_at_risk, "declared_message_bytes_at_risk")
        _nonnegative_int(self.input_bytes, "input_bytes")
        _nonnegative_int(self.initial_state_bytes, "initial_state_bytes")
        for complete, byte_count, label in (
            (self.input_identity_complete, self.input_bytes, "input"),
            (self.initial_state_identity_complete, self.initial_state_bytes, "initial state"),
        ):
            _require(type(complete) is bool, f"refusal {label} identity flag must be boolean")
            if complete:
                _positive_int(byte_count, f"{label}_bytes")
            else:
                _require(byte_count == 0, f"incomplete {label} identity cannot claim bytes")
        for complete, frozen, digest, byte_count, label in (
            (
                self.input_identity_complete,
                self.external_inputs,
                self.input_sha256,
                self.input_bytes,
                "input",
            ),
            (
                self.initial_state_identity_complete,
                self.initial_state,
                self.initial_state_sha256,
                self.initial_state_bytes,
                "initial state",
            ),
        ):
            if complete:
                if not isinstance(frozen, FrozenJSON):
                    raise G0EvaluationError(f"complete refusal {label} must be frozen")
                _require(frozen.sha256 == digest, f"refusal {label} snapshot hash mismatch")
                _require(
                    len(frozen.canonical.encode("utf-8")) == byte_count,
                    f"refusal {label} snapshot byte count mismatch",
                )
            else:
                _require(frozen is None, f"incomplete refusal {label} must omit its snapshot")
        _require(
            self.work == WorkVector(actor_execution=self.charged_operations),
            "refusal work mismatch",
        )
        expected_branch = _counterfactual_branch_id(
            genotype_sha256=self.genotype_sha256,
            grammar_sha256=self.grammar_sha256,
            candidate_registry_sha256=self.candidate_registry_sha256,
            evaluator_contract_sha256=self.evaluator_contract_sha256,
            input_sha256=self.input_sha256,
            initial_state_sha256=self.initial_state_sha256,
        )
        _require(self.counterfactual_branch_id == expected_branch, "refusal branch mismatch")
        _require(
            self.accounting_mode == "conservative-full-genotype-envelope",
            "refusal accounting mode drifted",
        )
        _require(self.status == "refused", "G0 failure receipt status drifted")
        _require(self.counterfactual_only is True, "refusal escaped counterfactual status")
        _require(self.activation_enabled is False, "refusal enabled activation")
        _require(self.factual_effects is False, "refusal enabled factual effects")
        _require(
            self.scientific_promotion_allowed is False,
            "refusal granted scientific promotion",
        )
        _require(
            self.refusal_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 refusal self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "genotype_sha256": self.genotype_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "input_sha256": self.input_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "external_inputs": (
                self.external_inputs.payload() if self.external_inputs is not None else None
            ),
            "initial_state": self.initial_state.payload() if self.initial_state is not None else None,
            "attempt_id": self.attempt_id,
            "reason": self.reason,
            "charged_operations": self.charged_operations,
            "declared_message_bytes_at_risk": self.declared_message_bytes_at_risk,
            "input_bytes": self.input_bytes,
            "initial_state_bytes": self.initial_state_bytes,
            "input_identity_complete": self.input_identity_complete,
            "initial_state_identity_complete": self.initial_state_identity_complete,
            "work": self.work.payload(),
            "counterfactual_branch_id": self.counterfactual_branch_id,
            "accounting_mode": self.accounting_mode,
            "status": self.status,
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "factual_effects": self.factual_effects,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["refusal_sha256"] = self.refusal_sha256
        return result

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> G0CounterfactualRefusal:
        expected = {
            "schema",
            "genotype_sha256",
            "grammar_sha256",
            "candidate_registry_sha256",
            "evaluator_contract_sha256",
            "input_sha256",
            "initial_state_sha256",
            "external_inputs",
            "initial_state",
            "attempt_id",
            "reason",
            "charged_operations",
            "declared_message_bytes_at_risk",
            "input_bytes",
            "initial_state_bytes",
            "input_identity_complete",
            "initial_state_identity_complete",
            "work",
            "counterfactual_branch_id",
            "accounting_mode",
            "status",
            "counterfactual_only",
            "activation_enabled",
            "factual_effects",
            "scientific_promotion_allowed",
            "refusal_sha256",
        }
        _require(set(payload) == expected, "G0 refusal fields mismatch")
        work = payload["work"]
        _require(isinstance(work, Mapping), "refusal work must be a mapping")
        external_inputs = payload["external_inputs"]
        initial_state = payload["initial_state"]
        return cls(
            schema=payload["schema"],
            genotype_sha256=payload["genotype_sha256"],
            grammar_sha256=payload["grammar_sha256"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            evaluator_contract_sha256=payload["evaluator_contract_sha256"],
            input_sha256=payload["input_sha256"],
            initial_state_sha256=payload["initial_state_sha256"],
            external_inputs=(
                None
                if external_inputs is None
                else _frozen_from_payload(external_inputs, "refusal external_inputs")
            ),
            initial_state=(
                None
                if initial_state is None
                else _frozen_from_payload(initial_state, "refusal initial_state")
            ),
            attempt_id=payload["attempt_id"],
            reason=payload["reason"],
            charged_operations=payload["charged_operations"],
            declared_message_bytes_at_risk=payload["declared_message_bytes_at_risk"],
            input_bytes=payload["input_bytes"],
            initial_state_bytes=payload["initial_state_bytes"],
            input_identity_complete=payload["input_identity_complete"],
            initial_state_identity_complete=payload["initial_state_identity_complete"],
            work=WorkVector.from_payload(work),
            counterfactual_branch_id=payload["counterfactual_branch_id"],
            accounting_mode=payload["accounting_mode"],
            status=payload["status"],
            counterfactual_only=payload["counterfactual_only"],
            activation_enabled=payload["activation_enabled"],
            factual_effects=payload["factual_effects"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            refusal_sha256=payload["refusal_sha256"],
        )


def evaluate_g0_counterfactual(
    genotype: G0ActorGenotype,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    external_inputs: Mapping[str, Any],
    initial_state: Mapping[str, Any],
) -> G0CounterfactualEvaluation:

    _require(
        candidate_registry.sha256 == grammar.candidate_registry_sha256,
        "candidate registry does not match the grammar authority",
    )
    assessment = assess_g0_genotype(
        genotype,
        grammar=grammar,
        candidate_registry=candidate_registry,
    )
    _require(
        assessment.structurally_valid,
        "genotype is structurally invalid: " + ", ".join(assessment.blockers),
    )
    bounds = grammar.construction_language.composition_bounds
    declared_states = {slot.slot_id: slot for slot in genotype.state_slots}
    state_value = _snapshot(dict(initial_state), "G0 initial state")
    _require(isinstance(state_value, dict), "G0 initial state must be an object")
    initial_state_snapshot = cast(dict[str, Any], state_value)
    _require(
        set(initial_state_snapshot) == set(declared_states),
        "initial state slots do not match genotype",
    )
    staged_state = _snapshot(initial_state_snapshot, "G0 staged state")
    _require(isinstance(staged_state, dict), "G0 staged state must be an object")
    staged_state = cast(dict[str, Any], staged_state)
    for slot_id, slot in declared_states.items():
        _require(
            len(canonical_bytes(initial_state_snapshot[slot_id])) <= slot.capacity_bytes,
            f"initial state exceeds capacity for {slot_id}",
        )
    root_ids = {node.node_id for node in genotype.operator_nodes if not node.input_node_ids}
    external_value = _snapshot(dict(external_inputs), "G0 external input")
    _require(isinstance(external_value, dict), "G0 external input must be an object")
    external = cast(dict[str, Any], external_value)
    _require(set(external) == root_ids, "external inputs must name every and only root node")
    input_bytes = len(canonical_bytes(external))
    initial_state_bytes = len(canonical_bytes(initial_state_snapshot))
    _require(input_bytes <= bounds.max_state_bytes_per_actor, "external input byte cap exceeded")
    input_sha = canonical_sha256(external)
    state_sha = canonical_sha256(initial_state_snapshot)
    grammar_sha = grammar.grammar_sha256
    registry_sha = candidate_registry.sha256
    branch_id = _counterfactual_branch_id(
        genotype_sha256=genotype.genotype_sha256,
        grammar_sha256=grammar_sha,
        candidate_registry_sha256=registry_sha,
        evaluator_contract_sha256=G0_EVALUATOR_CONTRACT_SHA256,
        input_sha256=input_sha,
        initial_state_sha256=state_sha,
    )
    outputs: dict[str, Any] = {}
    node_cost_rows: list[dict[str, Any]] = []
    staged_messages: list[dict[str, Any]] = []
    writers: set[str] = set()
    operations = 0
    node_output_bytes = 0
    edge_by_id = {edge.edge_id: edge for edge in genotype.message_edges}
    for node in _topological_order(genotype.operator_nodes):
        parameters = node.parameters.value()
        _require(isinstance(parameters, dict), "G0 node parameters are not an object")
        _validate_operator_parameters(node, parameters, declared_states)
        values = [outputs[parent] for parent in node.input_node_ids]
        seed = external.get(node.node_id, _MISSING)
        if node.operator is OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE:
            _require_state_primitive(
                node,
                declared_states,
                frozenset(
                    {
                        StatePrimitive.BOUNDED_RECURRENT_STATE,
                        StatePrimitive.BOUNDED_SCALAR_OR_VECTOR_DISTRIBUTIONS,
                    }
                ),
                optional=True,
            )
            value, updates, messages, used = _affine(
                node, values, seed, parameters, initial_state_snapshot
            )
        elif node.operator is OperatorPrimitive.BOUNDED_RETRIEVAL:
            _require_state_primitive(
                node,
                declared_states,
                frozenset({StatePrimitive.BOUNDED_CATEGORICAL_TABLES}),
            )
            value, updates, messages, used = _retrieval(
                node, values, seed, parameters, initial_state_snapshot
            )
        elif node.operator is OperatorPrimitive.BOUNDED_ROLLOUT_OR_SEARCH:
            _require(not node.state_slot_ids, f"{node.node_id} cannot bind state")
            value, updates, messages, used = _rollout(node, values, seed, parameters)
        elif node.operator is OperatorPrimitive.CONSTRAINT_FILTERING_OR_PROPAGATION:
            _require(not node.state_slot_ids, f"{node.node_id} cannot bind state")
            value, updates, messages, used = _constraint_filter(node, values, seed, parameters)
        elif node.operator is OperatorPrimitive.GRAPH_NEIGHBORHOOD_AGGREGATION:
            _require_state_primitive(
                node,
                declared_states,
                frozenset({StatePrimitive.BOUNDED_TYPED_FACTOR_SUBGRAPHS}),
            )
            value, updates, messages, used = _graph_aggregate(
                node, values, seed, parameters, initial_state_snapshot
            )
        elif node.operator is OperatorPrimitive.TABLE_LOOKUP_OR_UPDATE:
            _require_state_primitive(
                node,
                declared_states,
                frozenset({StatePrimitive.BOUNDED_CATEGORICAL_TABLES}),
            )
            value, updates, messages, used = _table(
                node, values, seed, parameters, initial_state_snapshot
            )
        elif node.operator is OperatorPrimitive.TEMPORAL_ACCUMULATION:
            _require_state_primitive(
                node,
                declared_states,
                frozenset({StatePrimitive.BOUNDED_TEMPORAL_DEQUES}),
            )
            value, updates, messages, used = _temporal(
                node, values, seed, parameters, initial_state_snapshot
            )
        elif node.operator is OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT:
            _require(not node.state_slot_ids, f"{node.node_id} cannot bind state")
            value, updates, messages, used = _message(
                node,
                values,
                seed,
                parameters,
                edge_by_id,
                genotype_sha256=genotype.genotype_sha256,
                grammar_sha256=grammar_sha,
                candidate_registry_sha256=registry_sha,
                evaluator_contract_sha256=G0_EVALUATOR_CONTRACT_SHA256,
                input_sha256=input_sha,
                initial_state_sha256=state_sha,
                counterfactual_branch_id=branch_id,
            )
        else:  # defensive against a future enum member without an implementation
            raise G0EvaluationError(f"operator implementation missing: {node.operator.value}")
        value_bytes = len(canonical_bytes(value))
        _require(value_bytes <= node.max_output_bytes, f"{node.node_id} output cap exceeded")
        node_output_bytes += value_bytes
        _require(
            node_output_bytes <= bounds.max_state_bytes_per_actor,
            "aggregate node-output byte cap exceeded",
        )
        for slot_id, updated in updates.items():
            _require(slot_id not in writers, f"multiple state writers target {slot_id}")
            writers.add(slot_id)
            slot = declared_states[slot_id]
            _require(
                len(canonical_bytes(updated)) <= slot.capacity_bytes,
                f"staged state exceeds capacity for {slot_id}",
            )
            staged_state[slot_id] = _snapshot(updated, f"staged state {slot_id}")
        outputs[node.node_id] = _snapshot(value, f"node output {node.node_id}")
        staged_messages.extend(messages)
        operations += used
        node_message_bytes = sum(int(message["encoded_bytes"]) for message in messages)
        cost_core = {
            "schema": G0_NODE_COST_SCHEMA,
            "node_id": node.node_id,
            "operator": node.operator.value,
            "declared_operations": node.declared_operations,
            "used_operations": used,
            "output_bytes": value_bytes,
            "message_count": len(messages),
            "message_payload_bytes": node_message_bytes,
            "status": "completed",
        }
        node_cost_rows.append({**cost_core, "node_cost_sha256": canonical_sha256(cost_core)})
    _require(operations <= genotype.declared_operations, "genotype exceeded declared total work")
    _require(operations <= bounds.max_operations_per_activation, "grammar work cap exceeded")
    message_payload_bytes = sum(int(message["encoded_bytes"]) for message in staged_messages)
    _require(message_payload_bytes <= bounds.max_message_bytes, "grammar message-byte cap exceeded")
    output_payload = {node_id: outputs[node_id] for node_id in sorted(outputs)}
    frozen_external = FrozenJSON.from_value(external)
    frozen_initial_state = FrozenJSON.from_value(initial_state_snapshot)
    frozen_outputs = FrozenJSON.from_value(output_payload)
    frozen_costs = FrozenJSON.from_value(node_cost_rows)
    frozen_state = FrozenJSON.from_value(staged_state)
    frozen_messages = tuple(FrozenJSON.from_value(message) for message in staged_messages)
    staged_state_bytes = len(canonical_bytes(staged_state))
    message_envelope_bytes = sum(len(canonical_bytes(message)) for message in staged_messages)
    message_count = len(staged_messages)
    work = WorkVector(actor_execution=operations, messages=message_count)
    core = {
        "schema": G0_EVALUATION_SCHEMA,
        "genotype_sha256": genotype.genotype_sha256,
        "grammar_sha256": grammar_sha,
        "candidate_registry_sha256": registry_sha,
        "evaluator_contract_sha256": G0_EVALUATOR_CONTRACT_SHA256,
        "input_sha256": input_sha,
        "initial_state_sha256": state_sha,
        "external_inputs": frozen_external.payload(),
        "initial_state": frozen_initial_state.payload(),
        "node_outputs": frozen_outputs.payload(),
        "node_costs": frozen_costs.payload(),
        "staged_state": frozen_state.payload(),
        "staged_messages": [message.payload() for message in frozen_messages],
        "operations": operations,
        "input_bytes": input_bytes,
        "initial_state_bytes": initial_state_bytes,
        "node_output_bytes": node_output_bytes,
        "staged_state_bytes": staged_state_bytes,
        "message_count": message_count,
        "message_payload_bytes": message_payload_bytes,
        "message_envelope_bytes": message_envelope_bytes,
        "work": work.payload(),
        "counterfactual_branch_id": branch_id,
        "counterfactual_only": True,
        "activation_enabled": False,
        "factual_effects": False,
        "scientific_promotion_allowed": False,
    }
    return G0CounterfactualEvaluation(
        genotype_sha256=genotype.genotype_sha256,
        grammar_sha256=grammar_sha,
        candidate_registry_sha256=registry_sha,
        evaluator_contract_sha256=G0_EVALUATOR_CONTRACT_SHA256,
        input_sha256=input_sha,
        initial_state_sha256=state_sha,
        external_inputs=frozen_external,
        initial_state=frozen_initial_state,
        node_outputs=frozen_outputs,
        node_costs=frozen_costs,
        staged_state=frozen_state,
        staged_messages=frozen_messages,
        operations=operations,
        input_bytes=input_bytes,
        initial_state_bytes=initial_state_bytes,
        node_output_bytes=node_output_bytes,
        staged_state_bytes=staged_state_bytes,
        message_count=message_count,
        message_payload_bytes=message_payload_bytes,
        message_envelope_bytes=message_envelope_bytes,
        work=work,
        counterfactual_branch_id=branch_id,
        counterfactual_only=True,
        activation_enabled=False,
        factual_effects=False,
        scientific_promotion_allowed=False,
        evaluation_sha256=canonical_sha256(core),
    )


def attempt_g0_counterfactual(
    genotype: G0ActorGenotype,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    external_inputs: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    attempt_id: str,
) -> G0CounterfactualEvaluation | G0CounterfactualRefusal:
    if type(genotype) is not G0ActorGenotype:
        raise ValueError("genotype must be an exact G0ActorGenotype")
    if type(grammar) is not TopologyGrammar:
        raise ValueError("grammar must be an exact TopologyGrammar")
    if type(candidate_registry) is not PerspectiveCandidateRegistry:
        raise ValueError("candidate_registry must be an exact PerspectiveCandidateRegistry")
    if not isinstance(external_inputs, Mapping) or not isinstance(initial_state, Mapping):
        raise ValueError("attempt inputs must be mappings")
    if not isinstance(attempt_id, str) or _REFERENCE_RE.fullmatch(attempt_id) is None:
        raise ValueError("attempt_id must be a stable reference")

    def capture_identity(
        value: Mapping[str, Any], label: str
    ) -> tuple[dict[str, Any], str, int, bool, str | None]:
        try:
            snapshot = _snapshot(dict(value), label)
        except (G0EvaluationError, RecursionError, TypeError, ValueError) as exc:
            reason = str(exc)
            unavailable = canonical_sha256(
                {
                    "schema": "mop-escs-g0-unavailable-input-identity/v1",
                    "label": label,
                    "error_type": type(exc).__name__,
                    "reason": reason,
                }
            )
            return {}, unavailable, 0, False, reason
        _require(isinstance(snapshot, dict), f"{label} must encode an object")
        row = cast(dict[str, Any], snapshot)
        return row, canonical_sha256(row), len(canonical_bytes(row)), True, None

    external, input_sha, input_bytes, input_complete, input_problem = capture_identity(
        external_inputs, "G0 attempt external input"
    )
    state, state_sha, state_bytes, state_complete, state_problem = capture_identity(
        initial_state, "G0 attempt initial state"
    )
    grammar_sha = grammar.grammar_sha256
    registry_sha = candidate_registry.sha256

    def refusal(reason: str) -> G0CounterfactualRefusal:
        branch_id = _counterfactual_branch_id(
            genotype_sha256=genotype.genotype_sha256,
            grammar_sha256=grammar_sha,
            candidate_registry_sha256=registry_sha,
            evaluator_contract_sha256=G0_EVALUATOR_CONTRACT_SHA256,
            input_sha256=input_sha,
            initial_state_sha256=state_sha,
        )
        charged_operations = genotype.declared_operations
        work = WorkVector(actor_execution=charged_operations)
        frozen_external = FrozenJSON.from_value(external) if input_complete else None
        frozen_state = FrozenJSON.from_value(state) if state_complete else None
        core = {
            "schema": G0_EVALUATION_REFUSAL_SCHEMA,
            "genotype_sha256": genotype.genotype_sha256,
            "grammar_sha256": grammar_sha,
            "candidate_registry_sha256": registry_sha,
            "evaluator_contract_sha256": G0_EVALUATOR_CONTRACT_SHA256,
            "input_sha256": input_sha,
            "initial_state_sha256": state_sha,
            "external_inputs": frozen_external.payload() if frozen_external is not None else None,
            "initial_state": frozen_state.payload() if frozen_state is not None else None,
            "attempt_id": attempt_id,
            "reason": reason,
            "charged_operations": charged_operations,
            "declared_message_bytes_at_risk": genotype.declared_message_bytes,
            "input_bytes": input_bytes,
            "initial_state_bytes": state_bytes,
            "input_identity_complete": input_complete,
            "initial_state_identity_complete": state_complete,
            "work": work.payload(),
            "counterfactual_branch_id": branch_id,
            "accounting_mode": "conservative-full-genotype-envelope",
            "status": "refused",
            "counterfactual_only": True,
            "activation_enabled": False,
            "factual_effects": False,
            "scientific_promotion_allowed": False,
        }
        return G0CounterfactualRefusal(
            genotype_sha256=genotype.genotype_sha256,
            grammar_sha256=grammar_sha,
            candidate_registry_sha256=registry_sha,
            evaluator_contract_sha256=G0_EVALUATOR_CONTRACT_SHA256,
            input_sha256=input_sha,
            initial_state_sha256=state_sha,
            external_inputs=frozen_external,
            initial_state=frozen_state,
            attempt_id=attempt_id,
            reason=reason,
            charged_operations=charged_operations,
            declared_message_bytes_at_risk=genotype.declared_message_bytes,
            input_bytes=input_bytes,
            initial_state_bytes=state_bytes,
            input_identity_complete=input_complete,
            initial_state_identity_complete=state_complete,
            work=work,
            counterfactual_branch_id=branch_id,
            accounting_mode="conservative-full-genotype-envelope",
            status="refused",
            counterfactual_only=True,
            activation_enabled=False,
            factual_effects=False,
            scientific_promotion_allowed=False,
            refusal_sha256=canonical_sha256(core),
        )

    identity_problems = [problem for problem in (input_problem, state_problem) if problem is not None]
    if identity_problems:
        return refusal("; ".join(identity_problems))
    try:
        return evaluate_g0_counterfactual(
            genotype,
            grammar=grammar,
            candidate_registry=candidate_registry,
            external_inputs=external,
            initial_state=state,
        )
    except G0EvaluationError as exc:
        return refusal(str(exc))


def verify_g0_counterfactual(
    evaluation: G0CounterfactualEvaluation,
    *,
    genotype: G0ActorGenotype,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> tuple[str, ...]:

    problems: list[str] = []
    if evaluation.genotype_sha256 != genotype.genotype_sha256:
        problems.append("genotype-authority-mismatch")
    if evaluation.grammar_sha256 != grammar.grammar_sha256:
        problems.append("grammar-authority-mismatch")
    if evaluation.candidate_registry_sha256 != candidate_registry.sha256:
        problems.append("candidate-registry-authority-mismatch")
    if evaluation.evaluator_contract_sha256 != G0_EVALUATOR_CONTRACT_SHA256:
        problems.append("evaluator-contract-authority-mismatch")
    if problems:
        return tuple(sorted(problems))
    try:
        replay = evaluate_g0_counterfactual(
            genotype,
            grammar=grammar,
            candidate_registry=candidate_registry,
            external_inputs=cast(dict[str, Any], evaluation.external_inputs.value()),
            initial_state=cast(dict[str, Any], evaluation.initial_state.value()),
        )
    except G0EvaluationError:
        return ("deterministic-replay-refused",)
    if replay != evaluation:
        return ("deterministic-replay-mismatch",)
    return ()


__all__ = [
    "G0_EVALUATION_SCHEMA",
    "G0_EVALUATION_REFUSAL_SCHEMA",
    "G0_EVALUATOR_CONTRACT_SHA256",
    "G0_NODE_COST_SCHEMA",
    "G0_STAGED_MESSAGE_SCHEMA",
    "G0CounterfactualEvaluation",
    "G0CounterfactualRefusal",
    "G0EvaluationError",
    "attempt_g0_counterfactual",
    "evaluate_g0_counterfactual",
    "verify_g0_counterfactual",
]
