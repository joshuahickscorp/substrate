"""Finite, content-addressed actor genotypes for the disabled G0 scaffold.

This module closes the representation gap between perspective slots and topology
slots: an actor can be described as bounded state plus a finite operator DAG without
using generated code or an implicit neural-module convention.  It is static
mechanics only.  The current G0 grammar remains incomplete, unfrozen, and unable to
authorize shadow or factual execution.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Self

from mop.substrate.events import FrozenJSON, canonical_bytes, canonical_sha256

from .perspective_registry import IntegrationDisposition, PerspectiveCandidateRegistry
from .topology_grammar import (
    GrammarStatus,
    OperatorPrimitive,
    StatePrimitive,
    TopologyGrammar,
)

G0_GENOTYPE_SCHEMA = "mop-escs-g0-actor-genotype/v1"
G0_GENOTYPE_ASSESSMENT_SCHEMA = "mop-escs-g0-genotype-assessment/v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PARAMETER_TERMS = frozenset(
    {
        "arbitrary_generated_code",
        "eval",
        "exec",
        "generated_code",
        "mutation_outside_transactional_governance",
        "new_schema",
        "operator",
        "operator_id",
        "operator_primitive",
        "python",
        "recursion",
        "silent_schema_creation",
        "source_code",
        "state_primitive",
        "suboperator",
        "unbounded_recursion",
        "undeclared_operator",
        "undeclared_operators",
    }
)


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _walk_keys(value: Any) -> set[str]:
    result: set[str] = set()
    stack = [value]
    while stack:
        nested = stack.pop()
        if isinstance(nested, Mapping):
            result.update(str(key) for key in nested)
            stack.extend(nested.values())
        elif isinstance(nested, (list, tuple)):
            stack.extend(nested)
    return result


def _walk_strings(value: Any) -> set[str]:
    result: set[str] = set()
    stack = [value]
    while stack:
        nested = stack.pop()
        if isinstance(nested, str):
            result.add(nested)
        elif isinstance(nested, Mapping):
            result.update(str(key) for key in nested)
            stack.extend(nested.values())
        elif isinstance(nested, (list, tuple)):
            stack.extend(nested)
    return result


def _parameter_schema_bindings(value: Any) -> tuple[tuple[str, str], ...]:
    """Return normalized parameter-key/schema-reference pairs without recursion."""

    result: list[tuple[str, str]] = []
    stack = [value]
    while stack:
        nested = stack.pop()
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                key_text = str(key)
                normalized_key = key_text.lower().replace("-", "_")
                if (
                    normalized_key == "schema"
                    or normalized_key.endswith("_schema")
                    or normalized_key.endswith("_schema_id")
                ):
                    result.append((normalized_key, _require_id(child, f"G0 parameter {key_text}")))
                stack.append(child)
        elif isinstance(nested, (list, tuple)):
            stack.extend(nested)
    return tuple(result)


def _parameter_schema_ids(value: Any) -> frozenset[str]:
    return frozenset(schema_id for _, schema_id in _parameter_schema_bindings(value))


def _canonical_ids(values: Sequence[str], label: str) -> tuple[str, ...]:
    rows = tuple(values)
    if any(not isinstance(value, str) or _ID_RE.fullmatch(value) is None for value in rows):
        raise ValueError(f"{label} contains a noncanonical identifier")
    if rows != tuple(sorted(rows)) or len(rows) != len(set(rows)):
        raise ValueError(f"{label} must be unique and canonically sorted")
    return rows


@dataclass(frozen=True, slots=True)
class G0StateSlot:
    slot_id: str
    primitive: StatePrimitive
    schema_id: str
    capacity_bytes: int

    def __post_init__(self) -> None:
        _require_id(self.slot_id, "G0 state slot_id")
        if not isinstance(self.primitive, StatePrimitive):
            raise ValueError("G0 state primitive must be typed")
        _require_id(self.schema_id, "G0 state schema_id")
        _nonnegative(self.capacity_bytes, "G0 state capacity_bytes", positive=True)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(payload, {"slot_id", "primitive", "schema_id", "capacity_bytes"}, "G0StateSlot")
        return cls(
            slot_id=payload["slot_id"],
            primitive=StatePrimitive(payload["primitive"]),
            schema_id=payload["schema_id"],
            capacity_bytes=payload["capacity_bytes"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "primitive": self.primitive.value,
            "schema_id": self.schema_id,
            "capacity_bytes": self.capacity_bytes,
        }


@dataclass(frozen=True, slots=True)
class G0MessageEdge:
    edge_id: str
    schema_id: str
    max_encoded_bytes: int

    def __post_init__(self) -> None:
        _require_id(self.edge_id, "G0 message edge_id")
        _require_id(self.schema_id, "G0 message schema_id")
        _nonnegative(self.max_encoded_bytes, "G0 max message bytes", positive=True)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(payload, {"edge_id", "schema_id", "max_encoded_bytes"}, "G0MessageEdge")
        return cls(
            edge_id=payload["edge_id"],
            schema_id=payload["schema_id"],
            max_encoded_bytes=payload["max_encoded_bytes"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "schema_id": self.schema_id,
            "max_encoded_bytes": self.max_encoded_bytes,
        }


@dataclass(frozen=True, slots=True)
class G0OperatorNode:
    node_id: str
    operator: OperatorPrimitive
    input_node_ids: tuple[str, ...]
    state_slot_ids: tuple[str, ...]
    message_edge_ids: tuple[str, ...]
    parameters: FrozenJSON
    declared_operations: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        _require_id(self.node_id, "G0 operator node_id")
        if not isinstance(self.operator, OperatorPrimitive):
            raise ValueError("G0 operator primitive must be typed")
        if not all(
            isinstance(values, tuple)
            for values in (self.input_node_ids, self.state_slot_ids, self.message_edge_ids)
        ):
            raise ValueError("G0 operator reference collections must be immutable tuples")
        if len(self.input_node_ids) != len(set(self.input_node_ids)):
            raise ValueError("G0 input_node_ids must be unique")
        for value in self.input_node_ids:
            _require_id(value, "G0 input node id")
        _canonical_ids(self.state_slot_ids, "G0 state_slot_ids")
        _canonical_ids(self.message_edge_ids, "G0 message_edge_ids")
        if not isinstance(self.parameters, FrozenJSON):
            raise ValueError("G0 parameters must be FrozenJSON")
        value = self.parameters.value()
        if not isinstance(value, dict):
            raise ValueError("G0 parameters must contain a JSON object")
        strings = {text.strip().lower() for text in _walk_strings(value)}
        normalized_strings = {re.sub(r"[-\s]+", "_", text) for text in strings}
        forbidden = sorted(
            (_walk_keys(value) | strings | normalized_strings) & _FORBIDDEN_PARAMETER_TERMS
            | {text for text in strings if text.startswith("lambda ")}
        )
        if forbidden:
            raise ValueError(f"G0 parameters request forbidden construction capabilities: {forbidden}")
        _parameter_schema_ids(value)
        _nonnegative(self.declared_operations, "G0 declared_operations", positive=True)
        _nonnegative(self.max_output_bytes, "G0 max_output_bytes", positive=True)

    @classmethod
    def create(
        cls,
        *,
        node_id: str,
        operator: OperatorPrimitive,
        input_node_ids: Sequence[str] = (),
        state_slot_ids: Sequence[str] = (),
        message_edge_ids: Sequence[str] = (),
        parameters: Mapping[str, Any] | FrozenJSON,
        declared_operations: int,
        max_output_bytes: int,
    ) -> Self:
        frozen = parameters if isinstance(parameters, FrozenJSON) else FrozenJSON.from_value(parameters)
        return cls(
            node_id=node_id,
            operator=operator,
            input_node_ids=tuple(input_node_ids),
            state_slot_ids=tuple(sorted(state_slot_ids)),
            message_edge_ids=tuple(sorted(message_edge_ids)),
            parameters=frozen,
            declared_operations=declared_operations,
            max_output_bytes=max_output_bytes,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "node_id",
                "operator",
                "input_node_ids",
                "state_slot_ids",
                "message_edge_ids",
                "parameters",
                "declared_operations",
                "max_output_bytes",
            },
            "G0OperatorNode",
        )
        parameters = payload["parameters"]
        _exact_keys(parameters, {"value", "sha256"}, "G0 operator parameters")
        frozen = FrozenJSON.from_value(parameters["value"])
        if frozen.sha256 != parameters["sha256"]:
            raise ValueError("G0 operator parameter digest mismatch")
        return cls(
            node_id=payload["node_id"],
            operator=OperatorPrimitive(payload["operator"]),
            input_node_ids=tuple(payload["input_node_ids"]),
            state_slot_ids=tuple(payload["state_slot_ids"]),
            message_edge_ids=tuple(payload["message_edge_ids"]),
            parameters=frozen,
            declared_operations=payload["declared_operations"],
            max_output_bytes=payload["max_output_bytes"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operator": self.operator.value,
            "input_node_ids": list(self.input_node_ids),
            "state_slot_ids": list(self.state_slot_ids),
            "message_edge_ids": list(self.message_edge_ids),
            "parameters": self.parameters.payload(),
            "declared_operations": self.declared_operations,
            "max_output_bytes": self.max_output_bytes,
        }


def _dag_depth(nodes: Sequence[G0OperatorNode]) -> int:
    """Return maximum DAG depth without Python recursion or input-order dependence."""

    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("G0 operator node identities must be unique")
    children: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    indegree: dict[str, int] = {}
    depth: dict[str, int] = {}
    for node in nodes:
        indegree[node.node_id] = len(node.input_node_ids)
        depth[node.node_id] = 1
        for parent in node.input_node_ids:
            if parent not in by_id:
                raise ValueError(f"G0 operator graph references undeclared input {parent!r}")
            children[parent].append(node.node_id)
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    heapify(ready)
    visited = 0
    while ready:
        node_id = heappop(ready)
        visited += 1
        for child in sorted(children[node_id]):
            depth[child] = max(depth[child], depth[node_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(ready, child)
    if visited != len(by_id):
        raise ValueError("G0 operator graph contains a cycle")
    return max(depth.values(), default=0)


@dataclass(frozen=True, slots=True)
class G0ActorGenotype:
    candidate_id: str
    state_slots: tuple[G0StateSlot, ...]
    operator_nodes: tuple[G0OperatorNode, ...]
    output_node_ids: tuple[str, ...]
    message_edges: tuple[G0MessageEdge, ...]
    activation_enabled: bool
    scientific_promotion_allowed: bool
    genotype_sha256: str
    schema: str = G0_GENOTYPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G0_GENOTYPE_SCHEMA:
            raise ValueError(f"unsupported G0 genotype schema {self.schema!r}")
        _require_id(self.candidate_id, "G0 candidate_id")
        if not all(
            isinstance(value, tuple)
            for value in (
                self.state_slots,
                self.operator_nodes,
                self.output_node_ids,
                self.message_edges,
            )
        ):
            raise ValueError("G0 genotype collections must be immutable tuples")
        for rows, expected, label in (
            (self.state_slots, G0StateSlot, "state slots"),
            (self.operator_nodes, G0OperatorNode, "operator nodes"),
            (self.message_edges, G0MessageEdge, "message edges"),
        ):
            if not all(type(row) is expected for row in rows):
                raise ValueError(f"G0 {label} must contain exact immutable records")
        for ids, label in (
            (tuple(row.slot_id for row in self.state_slots), "G0 state slots"),
            (tuple(row.node_id for row in self.operator_nodes), "G0 operator nodes"),
            (tuple(row.edge_id for row in self.message_edges), "G0 message edges"),
        ):
            if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
                raise ValueError(f"{label} must be unique and canonically sorted")
        if not self.operator_nodes:
            raise ValueError("G0 genotype requires at least one operator node")
        _canonical_ids(self.output_node_ids, "G0 output_node_ids")
        if not self.output_node_ids:
            raise ValueError("G0 genotype requires at least one explicit output node")
        node_ids = {node.node_id for node in self.operator_nodes}
        state_ids = {slot.slot_id for slot in self.state_slots}
        edge_ids = {edge.edge_id for edge in self.message_edges}
        if not set(self.output_node_ids) <= node_ids:
            raise ValueError("G0 output nodes must exist in the operator graph")
        state_by_id = {slot.slot_id: slot for slot in self.state_slots}
        edge_by_id = {edge.edge_id: edge for edge in self.message_edges}
        for node in self.operator_nodes:
            unknown_inputs = set(node.input_node_ids) - node_ids
            unknown_states = set(node.state_slot_ids) - state_ids
            unknown_edges = set(node.message_edge_ids) - edge_ids
            if unknown_inputs or unknown_states or unknown_edges:
                raise ValueError(
                    "G0 operator references undeclared nodes, state, or message edges: "
                    f"inputs={sorted(unknown_inputs)}, states={sorted(unknown_states)}, "
                    f"edges={sorted(unknown_edges)}"
                )
            emits = node.operator is OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT
            if emits != bool(node.message_edge_ids):
                raise ValueError(
                    "G0 typed-message operators must name an edge and non-message operators cannot"
                )
            allowed_schemas = {state_by_id[slot_id].schema_id for slot_id in node.state_slot_ids} | {
                edge_by_id[edge_id].schema_id for edge_id in node.message_edge_ids
            }
            schema_bindings = _parameter_schema_bindings(node.parameters.value())
            undeclared_schemas = {schema_id for _, schema_id in schema_bindings} - allowed_schemas
            if undeclared_schemas:
                raise ValueError(
                    "G0 operator parameters reference schemas not bound to that node: "
                    f"{sorted(undeclared_schemas)}"
                )
            if emits:
                edge_schemas = {edge_by_id[edge_id].schema_id for edge_id in node.message_edge_ids}
                message_schema_references = {
                    schema_id
                    for key, schema_id in schema_bindings
                    if key in {"schema", "schema_id"}
                    or any(token in key for token in ("claim", "emit", "message", "output"))
                }
                if message_schema_references - edge_schemas:
                    raise ValueError("G0 emit parameter schema does not match its declared message edges")
                recipient_cap = node.parameters.value().get("recipient_cap")
                if recipient_cap is not None and (
                    isinstance(recipient_cap, bool)
                    or not isinstance(recipient_cap, int)
                    or recipient_cap != len(node.message_edge_ids)
                ):
                    raise ValueError("G0 emit recipient_cap must equal the number of declared message edges")
                encoded_bound = sum(
                    edge_by_id[edge_id].max_encoded_bytes for edge_id in node.message_edge_ids
                )
                if encoded_bound > node.max_output_bytes:
                    raise ValueError("G0 emit output bound is smaller than its encoded message bound")
        _dag_depth(self.operator_nodes)
        referenced_states = {state_id for node in self.operator_nodes for state_id in node.state_slot_ids}
        if referenced_states != state_ids:
            raise ValueError("every G0 state slot must be referenced by an operator")
        edge_reference_counts = Counter(
            edge_id for node in self.operator_nodes for edge_id in node.message_edge_ids
        )
        if set(edge_reference_counts) != edge_ids:
            raise ValueError("every G0 message edge must be referenced by an emit operator")
        if any(count != 1 for count in edge_reference_counts.values()):
            raise ValueError("every G0 message edge must have exactly one emit-operator owner")
        consumed_nodes = {
            input_node_id for node in self.operator_nodes for input_node_id in node.input_node_ids
        }
        sink_nodes = node_ids - consumed_nodes
        if set(self.output_node_ids) != sink_nodes:
            raise ValueError("G0 output_node_ids must name every and only operator-graph sink")
        if self.activation_enabled is not False:
            raise ValueError("G0 genotype scaffold must remain activation-disabled")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("G0 genotype cannot grant scientific promotion")
        _require_digest(self.genotype_sha256, "G0 genotype sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.genotype_sha256:
            raise ValueError("G0 genotype self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        state_slots: Sequence[G0StateSlot],
        operator_nodes: Sequence[G0OperatorNode],
        output_node_ids: Sequence[str],
        message_edges: Sequence[G0MessageEdge] = (),
    ) -> Self:
        states = tuple(sorted(state_slots, key=lambda row: row.slot_id))
        nodes = tuple(sorted(operator_nodes, key=lambda row: row.node_id))
        edges = tuple(sorted(message_edges, key=lambda row: row.edge_id))
        core = {
            "schema": G0_GENOTYPE_SCHEMA,
            "candidate_id": candidate_id,
            "state_slots": [row.payload() for row in states],
            "operator_nodes": [row.payload() for row in nodes],
            "output_node_ids": sorted(output_node_ids),
            "message_edges": [row.payload() for row in edges],
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            candidate_id=candidate_id,
            state_slots=states,
            operator_nodes=nodes,
            output_node_ids=tuple(sorted(output_node_ids)),
            message_edges=edges,
            activation_enabled=False,
            scientific_promotion_allowed=False,
            genotype_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "candidate_id",
                "state_slots",
                "operator_nodes",
                "output_node_ids",
                "message_edges",
                "activation_enabled",
                "scientific_promotion_allowed",
                "genotype_sha256",
            },
            "G0ActorGenotype",
        )
        return cls(
            schema=payload["schema"],
            candidate_id=payload["candidate_id"],
            state_slots=tuple(G0StateSlot.from_payload(row) for row in payload["state_slots"]),
            operator_nodes=tuple(G0OperatorNode.from_payload(row) for row in payload["operator_nodes"]),
            output_node_ids=tuple(payload["output_node_ids"]),
            message_edges=tuple(G0MessageEdge.from_payload(row) for row in payload["message_edges"]),
            activation_enabled=payload["activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            genotype_sha256=payload["genotype_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "state_slots": [row.payload() for row in self.state_slots],
            "operator_nodes": [row.payload() for row in self.operator_nodes],
            "output_node_ids": list(self.output_node_ids),
            "message_edges": [row.payload() for row in self.message_edges],
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["genotype_sha256"] = self.genotype_sha256
        return result

    @property
    def retained_state_bytes(self) -> int:
        return sum(slot.capacity_bytes for slot in self.state_slots)

    @property
    def declared_operations(self) -> int:
        return sum(node.declared_operations for node in self.operator_nodes)

    @property
    def declared_message_bytes(self) -> int:
        return sum(edge.max_encoded_bytes for edge in self.message_edges)

    @property
    def declared_output_bytes(self) -> int:
        return sum(node.max_output_bytes for node in self.operator_nodes)

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.payload()))

    @property
    def declared_actor_bytes(self) -> int:
        """Conservative peak bound: retained state, all node outputs, and genotype bytes."""

        return self.retained_state_bytes + self.declared_output_bytes + self.encoded_bytes

    @property
    def dag_depth(self) -> int:
        return _dag_depth(self.operator_nodes)


@dataclass(frozen=True, slots=True)
class G0GenotypeAssessment:
    genotype_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    structurally_valid: bool
    shadow_authorized: bool
    factual_activation_authorized: bool
    blockers: tuple[str, ...]
    schema: str = G0_GENOTYPE_ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        for digest_value, label in (
            (self.genotype_sha256, "G0 assessment genotype sha256"),
            (self.grammar_sha256, "G0 assessment grammar sha256"),
            (self.candidate_registry_sha256, "G0 assessment candidate registry sha256"),
        ):
            _require_digest(digest_value, label)
        if self.schema != G0_GENOTYPE_ASSESSMENT_SCHEMA:
            raise ValueError(f"unsupported G0 genotype assessment schema {self.schema!r}")
        for flag, label in (
            (self.structurally_valid, "structurally_valid"),
            (self.shadow_authorized, "shadow_authorized"),
            (self.factual_activation_authorized, "factual_activation_authorized"),
        ):
            if type(flag) is not bool:
                raise ValueError(f"G0 assessment {label} must be boolean")
        if not isinstance(self.blockers, tuple) or not all(
            isinstance(blocker, str) and blocker for blocker in self.blockers
        ):
            raise ValueError("G0 genotype blockers must be immutable nonempty strings")
        if self.blockers != tuple(sorted(set(self.blockers))):
            raise ValueError("G0 genotype blockers must be unique and sorted")
        if self.shadow_authorized:
            raise ValueError("static G0 genotype assessment cannot authorize shadow execution")
        if self.factual_activation_authorized:
            raise ValueError("static G0 genotype assessment cannot authorize factual activation")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "genotype_sha256": self.genotype_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "structurally_valid": self.structurally_valid,
            "shadow_authorized": self.shadow_authorized,
            "factual_activation_authorized": self.factual_activation_authorized,
            "blockers": list(self.blockers),
        }


def assess_g0_genotype(
    genotype: G0ActorGenotype,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    freeze_authority_verified: bool = False,
) -> G0GenotypeAssessment:
    if type(genotype) is not G0ActorGenotype:
        raise ValueError("genotype must be an exact G0ActorGenotype")
    if type(grammar) is not TopologyGrammar:
        raise ValueError("grammar must be an exact TopologyGrammar")
    if type(candidate_registry) is not PerspectiveCandidateRegistry:
        raise ValueError("candidate_registry must be an exact PerspectiveCandidateRegistry")
    if type(freeze_authority_verified) is not bool:
        raise ValueError("freeze_authority_verified must be boolean")
    blockers: list[str] = []
    registry_matches = candidate_registry.sha256 == grammar.candidate_registry_sha256
    if not registry_matches:
        blockers.append("candidate-registry-authority-mismatch")
    else:
        candidates = {candidate.candidate_id: candidate for candidate in candidate_registry.candidates}
        candidate = candidates.get(genotype.candidate_id)
        if candidate is None:
            blockers.append("candidate-unknown")
        else:
            if candidate.integration_disposition is IntegrationDisposition.EXCLUDED:
                blockers.append("candidate-excluded")
            if not candidate.activation_enabled:
                blockers.append("candidate-activation-disabled")
    bounds = grammar.construction_language.composition_bounds
    if len(genotype.operator_nodes) > bounds.max_operator_nodes_per_actor:
        blockers.append("operator-node-cap-exceeded")
    if genotype.dag_depth > bounds.max_dag_depth:
        blockers.append("dag-depth-cap-exceeded")
    if genotype.retained_state_bytes > bounds.max_state_bytes_per_actor:
        blockers.append("state-byte-cap-exceeded")
    if genotype.declared_operations > bounds.max_operations_per_activation:
        blockers.append("operation-cap-exceeded")
    if len(genotype.message_edges) > bounds.max_message_edges:
        blockers.append("message-edge-cap-exceeded")
    if genotype.declared_message_bytes > bounds.max_message_bytes:
        blockers.append("message-byte-cap-exceeded")
    if genotype.declared_output_bytes > bounds.max_state_bytes_per_actor:
        blockers.append("output-byte-cap-exceeded")
    if genotype.encoded_bytes > bounds.max_state_bytes_per_actor:
        blockers.append("genotype-byte-cap-exceeded")
    if genotype.declared_actor_bytes > bounds.max_state_bytes_per_actor:
        blockers.append("actor-total-byte-cap-exceeded")
    if any(
        slot.primitive not in grammar.construction_language.state_primitives for slot in genotype.state_slots
    ):
        blockers.append("undeclared-state-primitive")
    if any(
        node.operator not in grammar.construction_language.operator_primitives
        for node in genotype.operator_nodes
    ):
        blockers.append("undeclared-operator-primitive")
    if not grammar.construction_language.implementation_complete:
        blockers.append("construction-language-incomplete")
    if grammar.status is not GrammarStatus.FROZEN:
        blockers.append("grammar-not-frozen")
    if not grammar.activation_enabled:
        blockers.append("grammar-activation-disabled")
    if grammar.freeze_authority is None:
        blockers.append("freeze-authority-absent")
    elif not freeze_authority_verified:
        blockers.append("freeze-authority-unverified")
    blockers.append("genotype-shadow-authorization-disabled")
    rows = tuple(sorted(set(blockers)))
    structural_blockers = {
        "candidate-unknown",
        "candidate-excluded",
        "candidate-registry-authority-mismatch",
        "operator-node-cap-exceeded",
        "dag-depth-cap-exceeded",
        "state-byte-cap-exceeded",
        "operation-cap-exceeded",
        "message-edge-cap-exceeded",
        "message-byte-cap-exceeded",
        "output-byte-cap-exceeded",
        "genotype-byte-cap-exceeded",
        "actor-total-byte-cap-exceeded",
        "undeclared-state-primitive",
        "undeclared-operator-primitive",
    }
    structurally_valid = not bool(set(rows) & structural_blockers)
    return G0GenotypeAssessment(
        genotype_sha256=genotype.genotype_sha256,
        grammar_sha256=grammar.grammar_sha256,
        candidate_registry_sha256=candidate_registry.sha256,
        structurally_valid=structurally_valid,
        shadow_authorized=False,
        factual_activation_authorized=False,
        blockers=rows,
    )


__all__ = [
    "G0_GENOTYPE_ASSESSMENT_SCHEMA",
    "G0_GENOTYPE_SCHEMA",
    "G0ActorGenotype",
    "G0GenotypeAssessment",
    "G0MessageEdge",
    "G0OperatorNode",
    "G0StateSlot",
    "assess_g0_genotype",
]
