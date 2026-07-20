
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from mop.escs.g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    G0StateSlot,
    assess_g0_genotype,
)
from mop.escs.perspective_registry import PerspectiveCandidateRegistry
from mop.escs.topology_grammar import (
    ConstructionMutation,
    OperatorPrimitive,
    TopologyGrammar,
)
from mop.substrate.events import FrozenJSON, canonical_bytes, canonical_sha256

G0_CONSTRUCTION_SNAPSHOT_SCHEMA = "mop-escs-g0-construction-snapshot/v1"
G0_CONSTRUCTION_REQUEST_SCHEMA = "mop-escs-g0-construction-request/v1"
G0_CONSTRUCTION_ATTEMPT_SCHEMA = "mop-escs-g0-construction-attempt/v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _canonical_ids(values: Sequence[str], label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    rows = tuple(_require_id(value, label) for value in values)
    if rows != tuple(sorted(rows)) or len(rows) != len(set(rows)):
        raise ValueError(f"{label} must be unique and canonically sorted")
    if nonempty and not rows:
        raise ValueError(f"{label} must be nonempty")
    return rows


class G0ConstructionOperation(StrEnum):
    ADD_OPERATOR = "add_operator"
    REMOVE_OPERATOR = "remove_operator"
    REPLACE_OPERATOR = "replace_operator"
    ADD_MESSAGE_EDGE = "add_message_edge"
    REMOVE_MESSAGE_EDGE = "remove_message_edge"
    ADJUST_STATE_CAPACITY = "adjust_state_capacity"
    ADJUST_TEMPORAL_WINDOW = "adjust_temporal_window"
    CLONE_ACTOR = "clone_actor"
    MERGE_ACTORS = "merge_actors"
    SPLIT_FACTOR_SCOPE = "split_factor_scope"
    MERGE_FACTOR_SCOPES = "merge_factor_scopes"


_FAMILY_BY_OPERATION = {
    G0ConstructionOperation.ADD_OPERATOR: ConstructionMutation.ADD_REMOVE_OR_REPLACE_G0_OPERATOR,
    G0ConstructionOperation.REMOVE_OPERATOR: ConstructionMutation.ADD_REMOVE_OR_REPLACE_G0_OPERATOR,
    G0ConstructionOperation.REPLACE_OPERATOR: ConstructionMutation.ADD_REMOVE_OR_REPLACE_G0_OPERATOR,
    G0ConstructionOperation.ADD_MESSAGE_EDGE: ConstructionMutation.ADD_OR_REMOVE_TYPED_MESSAGE_EDGE,
    G0ConstructionOperation.REMOVE_MESSAGE_EDGE: ConstructionMutation.ADD_OR_REMOVE_TYPED_MESSAGE_EDGE,
    G0ConstructionOperation.ADJUST_STATE_CAPACITY: (
        ConstructionMutation.ADJUST_BOUNDED_TIMESCALE_OR_MEMORY_CAPACITY
    ),
    G0ConstructionOperation.ADJUST_TEMPORAL_WINDOW: (
        ConstructionMutation.ADJUST_BOUNDED_TIMESCALE_OR_MEMORY_CAPACITY
    ),
    G0ConstructionOperation.CLONE_ACTOR: ConstructionMutation.CLONE_ACTOR,
    G0ConstructionOperation.MERGE_ACTORS: ConstructionMutation.MERGE_CAUSALLY_REDUNDANT_ACTORS,
    G0ConstructionOperation.SPLIT_FACTOR_SCOPE: ConstructionMutation.SPLIT_OR_MERGE_FACTOR_SCOPE,
    G0ConstructionOperation.MERGE_FACTOR_SCOPES: ConstructionMutation.SPLIT_OR_MERGE_FACTOR_SCOPE,
}


@dataclass(frozen=True, slots=True)
class G0FactorScopeBinding:
    actor_id: str
    scope_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.actor_id, "G0 factor binding actor_id")
        if not isinstance(self.scope_ids, tuple):
            raise ValueError("G0 factor scopes must be an immutable tuple")
        _canonical_ids(self.scope_ids, "G0 factor scopes", nonempty=True)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(payload, {"actor_id", "scope_ids"}, "G0 factor-scope binding")
        scopes = payload["scope_ids"]
        if not isinstance(scopes, list):
            raise ValueError("G0 factor scope_ids must be a list")
        return cls(actor_id=payload["actor_id"], scope_ids=tuple(scopes))

    def payload(self) -> dict[str, Any]:
        return {"actor_id": self.actor_id, "scope_ids": list(self.scope_ids)}


@dataclass(frozen=True, slots=True)
class G0ConstructionSnapshot:
    actors: tuple[G0ActorGenotype, ...]
    factor_scopes: tuple[G0FactorScopeBinding, ...]
    activation_enabled: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    snapshot_sha256: str
    schema: str = G0_CONSTRUCTION_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G0_CONSTRUCTION_SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported G0 construction snapshot schema {self.schema!r}")
        if not isinstance(self.actors, tuple) or not self.actors:
            raise ValueError("G0 construction snapshot requires an immutable nonempty actor tuple")
        if not all(type(actor) is G0ActorGenotype for actor in self.actors):
            raise ValueError("G0 construction actors must be exact G0ActorGenotype records")
        actor_ids = tuple(actor.candidate_id for actor in self.actors)
        if actor_ids != tuple(sorted(actor_ids)) or len(actor_ids) != len(set(actor_ids)):
            raise ValueError("G0 construction actors must have unique canonical identities")
        if not isinstance(self.factor_scopes, tuple) or not all(
            type(binding) is G0FactorScopeBinding for binding in self.factor_scopes
        ):
            raise ValueError("G0 construction factor scopes must be exact immutable bindings")
        binding_ids = tuple(binding.actor_id for binding in self.factor_scopes)
        if binding_ids != actor_ids:
            raise ValueError("G0 construction requires exactly one ordered factor binding per actor")
        if self.activation_enabled is not False:
            raise ValueError("G0 construction snapshots cannot enable activation")
        if self.factual_mutation_authorized is not False:
            raise ValueError("G0 construction snapshots cannot authorize factual mutation")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("G0 construction snapshots cannot grant scientific promotion")
        _require_digest(self.snapshot_sha256, "G0 construction snapshot digest")
        if canonical_sha256(self.payload(include_digest=False)) != self.snapshot_sha256:
            raise ValueError("G0 construction snapshot self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        actors: Sequence[G0ActorGenotype],
        factor_scopes: Mapping[str, Sequence[str]],
    ) -> Self:
        actor_rows = tuple(sorted(actors, key=lambda actor: actor.candidate_id))
        actor_ids = tuple(actor.candidate_id for actor in actor_rows)
        if set(factor_scopes) != set(actor_ids):
            raise ValueError("G0 factor-scope actors must exactly match genotype actors")
        bindings = tuple(
            G0FactorScopeBinding(
                actor_id=actor_id,
                scope_ids=tuple(sorted(factor_scopes[actor_id])),
            )
            for actor_id in actor_ids
        )
        core = {
            "schema": G0_CONSTRUCTION_SNAPSHOT_SCHEMA,
            "actors": [actor.payload() for actor in actor_rows],
            "factor_scopes": [binding.payload() for binding in bindings],
            "activation_enabled": False,
            "factual_mutation_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            actors=actor_rows,
            factor_scopes=bindings,
            activation_enabled=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            snapshot_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "actors",
                "factor_scopes",
                "activation_enabled",
                "factual_mutation_authorized",
                "scientific_promotion_allowed",
                "snapshot_sha256",
            },
            "G0 construction snapshot",
        )
        actors = payload["actors"]
        scopes = payload["factor_scopes"]
        if not isinstance(actors, list) or not isinstance(scopes, list):
            raise ValueError("G0 construction actors and factor scopes must be lists")
        return cls(
            schema=payload["schema"],
            actors=tuple(G0ActorGenotype.from_payload(actor) for actor in actors),
            factor_scopes=tuple(G0FactorScopeBinding.from_payload(row) for row in scopes),
            activation_enabled=payload["activation_enabled"],
            factual_mutation_authorized=payload["factual_mutation_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            snapshot_sha256=payload["snapshot_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "actors": [actor.payload() for actor in self.actors],
            "factor_scopes": [binding.payload() for binding in self.factor_scopes],
            "activation_enabled": self.activation_enabled,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["snapshot_sha256"] = self.snapshot_sha256
        return result

    @property
    def retained_state_bytes(self) -> int:
        return sum(actor.retained_state_bytes for actor in self.actors)

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.payload()))


@dataclass(frozen=True, slots=True)
class G0ConstructionRequest:
    attempt_id: str
    source_snapshot_sha256: str
    family: ConstructionMutation
    operation: G0ConstructionOperation
    parameters: FrozenJSON
    declared_work: int
    activation_enabled: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    request_sha256: str
    schema: str = G0_CONSTRUCTION_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G0_CONSTRUCTION_REQUEST_SCHEMA:
            raise ValueError(f"unsupported G0 construction request schema {self.schema!r}")
        _require_id(self.attempt_id, "G0 construction attempt_id")
        _require_digest(self.source_snapshot_sha256, "G0 construction source digest")
        if not isinstance(self.family, ConstructionMutation):
            raise ValueError("G0 construction family must be typed")
        if not isinstance(self.operation, G0ConstructionOperation):
            raise ValueError("G0 construction operation must be typed")
        if _FAMILY_BY_OPERATION[self.operation] is not self.family:
            raise ValueError("G0 construction operation does not belong to its declared family")
        if not isinstance(self.parameters, FrozenJSON) or not isinstance(self.parameters.value(), dict):
            raise ValueError("G0 construction parameters must be a frozen JSON object")
        _positive_int(self.declared_work, "G0 construction declared_work")
        if self.activation_enabled is not False:
            raise ValueError("G0 construction requests cannot enable activation")
        if self.factual_mutation_authorized is not False:
            raise ValueError("G0 construction requests cannot authorize factual mutation")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("G0 construction requests cannot grant scientific promotion")
        _require_digest(self.request_sha256, "G0 construction request digest")
        if canonical_sha256(self.payload(include_digest=False)) != self.request_sha256:
            raise ValueError("G0 construction request self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        attempt_id: str,
        source_snapshot_sha256: str,
        operation: G0ConstructionOperation,
        parameters: Mapping[str, Any],
        declared_work: int,
        family: ConstructionMutation | None = None,
    ) -> Self:
        typed_operation = G0ConstructionOperation(operation)
        typed_family = _FAMILY_BY_OPERATION[typed_operation] if family is None else family
        frozen = FrozenJSON.from_value(parameters)
        core = {
            "schema": G0_CONSTRUCTION_REQUEST_SCHEMA,
            "attempt_id": attempt_id,
            "source_snapshot_sha256": source_snapshot_sha256,
            "family": typed_family.value,
            "operation": typed_operation.value,
            "parameters": frozen.payload(),
            "declared_work": declared_work,
            "activation_enabled": False,
            "factual_mutation_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            attempt_id=attempt_id,
            source_snapshot_sha256=source_snapshot_sha256,
            family=typed_family,
            operation=typed_operation,
            parameters=frozen,
            declared_work=declared_work,
            activation_enabled=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            request_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "attempt_id",
                "source_snapshot_sha256",
                "family",
                "operation",
                "parameters",
                "declared_work",
                "activation_enabled",
                "factual_mutation_authorized",
                "scientific_promotion_allowed",
                "request_sha256",
            },
            "G0 construction request",
        )
        parameters = payload["parameters"]
        _exact_keys(parameters, {"value", "sha256"}, "G0 construction parameters")
        frozen = FrozenJSON.from_value(parameters["value"])
        if frozen.sha256 != parameters["sha256"]:
            raise ValueError("G0 construction parameter self-hash mismatch")
        return cls(
            schema=payload["schema"],
            attempt_id=payload["attempt_id"],
            source_snapshot_sha256=payload["source_snapshot_sha256"],
            family=ConstructionMutation(payload["family"]),
            operation=G0ConstructionOperation(payload["operation"]),
            parameters=frozen,
            declared_work=payload["declared_work"],
            activation_enabled=payload["activation_enabled"],
            factual_mutation_authorized=payload["factual_mutation_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            request_sha256=payload["request_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "attempt_id": self.attempt_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "family": self.family.value,
            "operation": self.operation.value,
            "parameters": self.parameters.payload(),
            "declared_work": self.declared_work,
            "activation_enabled": self.activation_enabled,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["request_sha256"] = self.request_sha256
        return result


class G0ConstructionStatus(StrEnum):
    APPLIED_SHADOW = "applied-shadow"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class G0ConstructionAttempt:
    request_sha256: str
    source_snapshot_sha256: str
    grammar_sha256: str
    candidate_registry_sha256: str
    status: G0ConstructionStatus
    candidate_snapshot: G0ConstructionSnapshot | None
    problems: tuple[str, ...]
    charged_work: int
    source_snapshot_bytes: int
    candidate_snapshot_bytes: int
    actor_delta: int
    factor_scope_delta: int
    retained_state_bytes_delta: int
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    attempt_sha256: str
    schema: str = G0_CONSTRUCTION_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != G0_CONSTRUCTION_ATTEMPT_SCHEMA:
            raise ValueError(f"unsupported G0 construction attempt schema {self.schema!r}")
        for digest, label in (
            (self.request_sha256, "request"),
            (self.source_snapshot_sha256, "source snapshot"),
            (self.grammar_sha256, "grammar"),
            (self.candidate_registry_sha256, "candidate registry"),
            (self.attempt_sha256, "attempt"),
        ):
            _require_digest(digest, f"G0 construction {label} digest")
        if not isinstance(self.status, G0ConstructionStatus):
            raise ValueError("G0 construction status must be typed")
        if (
            self.candidate_snapshot is not None
            and type(self.candidate_snapshot) is not G0ConstructionSnapshot
        ):
            raise ValueError("G0 construction candidate must be an exact snapshot")
        if not isinstance(self.problems, tuple) or not all(
            isinstance(problem, str) and problem for problem in self.problems
        ):
            raise ValueError("G0 construction problems must be immutable nonempty strings")
        if self.problems != tuple(sorted(set(self.problems))):
            raise ValueError("G0 construction problems must be unique and sorted")
        if self.status is G0ConstructionStatus.APPLIED_SHADOW:
            if self.candidate_snapshot is None or self.problems:
                raise ValueError("applied G0 construction attempts require one clean candidate")
        elif self.candidate_snapshot is not None or not self.problems:
            raise ValueError("refused G0 construction attempts require problems and no candidate")
        _positive_int(self.charged_work, "G0 construction charged_work")
        for value, label in (
            (self.source_snapshot_bytes, "source_snapshot_bytes"),
            (self.candidate_snapshot_bytes, "candidate_snapshot_bytes"),
        ):
            if _integer(value, f"G0 construction {label}") < 0:
                raise ValueError(f"G0 construction {label} must be nonnegative")
        for value, label in (
            (self.actor_delta, "actor_delta"),
            (self.factor_scope_delta, "factor_scope_delta"),
            (self.retained_state_bytes_delta, "retained_state_bytes_delta"),
        ):
            _integer(value, f"G0 construction {label}")
        if self.counterfactual_only is not True:
            raise ValueError("G0 construction attempts must remain counterfactual-only")
        for value, label in (
            (self.activation_enabled, "activation"),
            (self.shadow_execution_authorized, "shadow execution"),
            (self.factual_mutation_authorized, "factual mutation"),
            (self.scientific_promotion_allowed, "scientific promotion"),
        ):
            if value is not False:
                raise ValueError(f"G0 construction attempts cannot authorize {label}")
        if canonical_sha256(self.payload(include_digest=False)) != self.attempt_sha256:
            raise ValueError("G0 construction attempt self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        request: G0ConstructionRequest,
        source: G0ConstructionSnapshot,
        grammar: TopologyGrammar,
        candidate_registry: PerspectiveCandidateRegistry,
        candidate_snapshot: G0ConstructionSnapshot | None,
        problems: Sequence[str],
    ) -> Self:
        problem_rows = tuple(sorted(set(problems)))
        status = (
            G0ConstructionStatus.APPLIED_SHADOW
            if candidate_snapshot is not None and not problem_rows
            else G0ConstructionStatus.REFUSED
        )
        candidate_bytes = candidate_snapshot.encoded_bytes if candidate_snapshot is not None else 0
        actor_delta = 0 if candidate_snapshot is None else len(candidate_snapshot.actors) - len(source.actors)
        factor_delta = (
            0
            if candidate_snapshot is None
            else sum(len(row.scope_ids) for row in candidate_snapshot.factor_scopes)
            - sum(len(row.scope_ids) for row in source.factor_scopes)
        )
        state_delta = (
            0
            if candidate_snapshot is None
            else candidate_snapshot.retained_state_bytes - source.retained_state_bytes
        )
        core = {
            "schema": G0_CONSTRUCTION_ATTEMPT_SCHEMA,
            "request_sha256": request.request_sha256,
            "source_snapshot_sha256": source.snapshot_sha256,
            "grammar_sha256": grammar.grammar_sha256,
            "candidate_registry_sha256": candidate_registry.sha256,
            "status": status.value,
            "candidate_snapshot": candidate_snapshot.payload() if candidate_snapshot else None,
            "problems": list(problem_rows),
            "charged_work": request.declared_work,
            "source_snapshot_bytes": source.encoded_bytes,
            "candidate_snapshot_bytes": candidate_bytes,
            "actor_delta": actor_delta,
            "factor_scope_delta": factor_delta,
            "retained_state_bytes_delta": state_delta,
            "counterfactual_only": True,
            "activation_enabled": False,
            "shadow_execution_authorized": False,
            "factual_mutation_authorized": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            request_sha256=request.request_sha256,
            source_snapshot_sha256=source.snapshot_sha256,
            grammar_sha256=grammar.grammar_sha256,
            candidate_registry_sha256=candidate_registry.sha256,
            status=status,
            candidate_snapshot=candidate_snapshot,
            problems=problem_rows,
            charged_work=request.declared_work,
            source_snapshot_bytes=source.encoded_bytes,
            candidate_snapshot_bytes=candidate_bytes,
            actor_delta=actor_delta,
            factor_scope_delta=factor_delta,
            retained_state_bytes_delta=state_delta,
            counterfactual_only=True,
            activation_enabled=False,
            shadow_execution_authorized=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            attempt_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact_keys(
            payload,
            {
                "schema",
                "request_sha256",
                "source_snapshot_sha256",
                "grammar_sha256",
                "candidate_registry_sha256",
                "status",
                "candidate_snapshot",
                "problems",
                "charged_work",
                "source_snapshot_bytes",
                "candidate_snapshot_bytes",
                "actor_delta",
                "factor_scope_delta",
                "retained_state_bytes_delta",
                "counterfactual_only",
                "activation_enabled",
                "shadow_execution_authorized",
                "factual_mutation_authorized",
                "scientific_promotion_allowed",
                "attempt_sha256",
            },
            "G0 construction attempt",
        )
        candidate = payload["candidate_snapshot"]
        if candidate is not None and not isinstance(candidate, Mapping):
            raise ValueError("G0 construction candidate snapshot must be a mapping or null")
        problems = payload["problems"]
        if not isinstance(problems, list):
            raise ValueError("G0 construction attempt problems must be a list")
        return cls(
            schema=payload["schema"],
            request_sha256=payload["request_sha256"],
            source_snapshot_sha256=payload["source_snapshot_sha256"],
            grammar_sha256=payload["grammar_sha256"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            status=G0ConstructionStatus(payload["status"]),
            candidate_snapshot=(
                G0ConstructionSnapshot.from_payload(candidate) if candidate is not None else None
            ),
            problems=tuple(problems),
            charged_work=payload["charged_work"],
            source_snapshot_bytes=payload["source_snapshot_bytes"],
            candidate_snapshot_bytes=payload["candidate_snapshot_bytes"],
            actor_delta=payload["actor_delta"],
            factor_scope_delta=payload["factor_scope_delta"],
            retained_state_bytes_delta=payload["retained_state_bytes_delta"],
            counterfactual_only=payload["counterfactual_only"],
            activation_enabled=payload["activation_enabled"],
            shadow_execution_authorized=payload["shadow_execution_authorized"],
            factual_mutation_authorized=payload["factual_mutation_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            attempt_sha256=payload["attempt_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "request_sha256": self.request_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "status": self.status.value,
            "candidate_snapshot": self.candidate_snapshot.payload() if self.candidate_snapshot else None,
            "problems": list(self.problems),
            "charged_work": self.charged_work,
            "source_snapshot_bytes": self.source_snapshot_bytes,
            "candidate_snapshot_bytes": self.candidate_snapshot_bytes,
            "actor_delta": self.actor_delta,
            "factor_scope_delta": self.factor_scope_delta,
            "retained_state_bytes_delta": self.retained_state_bytes_delta,
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["attempt_sha256"] = self.attempt_sha256
        return result


def _parameters(request: G0ConstructionRequest, expected: set[str]) -> dict[str, Any]:
    value = request.parameters.value()
    if not isinstance(value, dict):
        raise ValueError("construction parameters are not an object")
    _exact_keys(value, expected, f"{request.operation.value} parameters")
    return value


def _actor_map(snapshot: G0ConstructionSnapshot) -> dict[str, G0ActorGenotype]:
    return {actor.candidate_id: actor for actor in snapshot.actors}


def _scope_map(snapshot: G0ConstructionSnapshot) -> dict[str, tuple[str, ...]]:
    return {binding.actor_id: binding.scope_ids for binding in snapshot.factor_scopes}


def _required_actor(actors: Mapping[str, G0ActorGenotype], actor_id: object) -> G0ActorGenotype:
    actor_key = _require_id(actor_id, "construction actor_id")
    try:
        return actors[actor_key]
    except KeyError as exc:
        raise ValueError(f"construction actor {actor_key!r} does not exist") from exc


def _sink_ids(nodes: Sequence[G0OperatorNode]) -> tuple[str, ...]:
    consumed = {input_id for node in nodes for input_id in node.input_node_ids}
    return tuple(sorted({node.node_id for node in nodes} - consumed))


def _rebuild_actor(
    actor: G0ActorGenotype,
    *,
    candidate_id: str | None = None,
    state_slots: Sequence[G0StateSlot] | None = None,
    operator_nodes: Sequence[G0OperatorNode] | None = None,
    message_edges: Sequence[G0MessageEdge] | None = None,
) -> G0ActorGenotype:
    nodes = tuple(actor.operator_nodes if operator_nodes is None else operator_nodes)
    edges = tuple(actor.message_edges if message_edges is None else message_edges)
    edge_by_id = {edge.edge_id: edge for edge in edges}
    for node in nodes:
        if node.operator is OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT:
            _validate_message_owner(
                node,
                tuple(edge_by_id[edge_id] for edge_id in node.message_edge_ids),
            )
    return G0ActorGenotype.create(
        candidate_id=actor.candidate_id if candidate_id is None else candidate_id,
        state_slots=actor.state_slots if state_slots is None else state_slots,
        operator_nodes=nodes,
        output_node_ids=_sink_ids(nodes),
        message_edges=edges,
    )


def _replace_actor(
    actors: Mapping[str, G0ActorGenotype], actor: G0ActorGenotype
) -> dict[str, G0ActorGenotype]:
    result = dict(actors)
    if actor.candidate_id not in result:
        raise ValueError("replacement actor identity does not exist")
    result[actor.candidate_id] = actor
    return result


def _node_payload(value: object, label: str) -> G0OperatorNode:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a node payload")
    return G0OperatorNode.from_payload(value)


def _edge_payload(value: object, label: str) -> G0MessageEdge:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an edge payload")
    return G0MessageEdge.from_payload(value)


def _replace_node(actor: G0ActorGenotype, replacement: G0OperatorNode) -> G0ActorGenotype:
    if replacement.node_id not in {node.node_id for node in actor.operator_nodes}:
        raise ValueError("replacement node identity does not exist")
    nodes = tuple(
        replacement if node.node_id == replacement.node_id else node for node in actor.operator_nodes
    )
    return _rebuild_actor(actor, operator_nodes=nodes)


def _replace_node_and_edges(
    actor: G0ActorGenotype,
    replacement: G0OperatorNode,
    edges: Sequence[G0MessageEdge],
) -> G0ActorGenotype:
    if replacement.node_id not in {node.node_id for node in actor.operator_nodes}:
        raise ValueError("replacement node identity does not exist")
    nodes = tuple(
        replacement if node.node_id == replacement.node_id else node for node in actor.operator_nodes
    )
    return _rebuild_actor(actor, operator_nodes=nodes, message_edges=edges)


def _validate_message_owner(
    owner: G0OperatorNode,
    edges: Sequence[G0MessageEdge],
) -> None:
    if owner.operator is not OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT:
        raise ValueError("message-edge owner must be a typed-message operator")
    by_id = {edge.edge_id: edge for edge in edges}
    if tuple(owner.message_edge_ids) != tuple(sorted(by_id)):
        raise ValueError("message-edge owner must name every and only supplied edge")
    parameters = owner.parameters.value()
    if not isinstance(parameters, dict):
        raise ValueError("message-edge owner parameters must be an object")
    emissions = parameters.get("emissions")
    if emissions is None:
        if len(edges) != 1 or parameters.get("schema_id") != edges[0].schema_id:
            raise ValueError("message-edge owner schema does not match its declared edge")
        return
    if not isinstance(emissions, list) or len(emissions) != len(edges):
        raise ValueError("message-edge owner emission count mismatch")
    expected_ids = tuple(sorted(by_id))
    observed_ids: list[str] = []
    for emission in emissions:
        if not isinstance(emission, dict):
            raise ValueError("message-edge owner emission must be an object")
        edge_id = emission.get("edge_id")
        if not isinstance(edge_id, str) or edge_id not in by_id:
            raise ValueError("message-edge owner emission names an undeclared edge")
        if emission.get("schema_id") != by_id[edge_id].schema_id:
            raise ValueError("message-edge owner schema does not match its declared edge")
        observed_ids.append(edge_id)
    if tuple(observed_ids) != expected_ids:
        raise ValueError("message-edge owner emissions must follow canonical edge order")


def _operator_mutation(
    request: G0ConstructionRequest,
    actors: Mapping[str, G0ActorGenotype],
) -> dict[str, G0ActorGenotype]:
    operation = request.operation
    if operation is G0ConstructionOperation.ADD_OPERATOR:
        parameters = _parameters(request, {"actor_id", "node"})
        actor = _required_actor(actors, parameters["actor_id"])
        node = _node_payload(parameters["node"], "added operator")
        if node.node_id in {row.node_id for row in actor.operator_nodes}:
            raise ValueError("added operator identity already exists")
        updated = _rebuild_actor(actor, operator_nodes=(*actor.operator_nodes, node))
    elif operation is G0ConstructionOperation.REMOVE_OPERATOR:
        parameters = _parameters(request, {"actor_id", "node_id"})
        actor = _required_actor(actors, parameters["actor_id"])
        node_id = _require_id(parameters["node_id"], "removed operator node_id")
        by_id = {node.node_id: node for node in actor.operator_nodes}
        if node_id not in by_id:
            raise ValueError("removed operator identity does not exist")
        if any(node_id in node.input_node_ids for node in actor.operator_nodes):
            raise ValueError("removed operator still has dependent nodes")
        removed = by_id[node_id]
        if removed.state_slot_ids or removed.message_edge_ids:
            raise ValueError("remove operator cannot orphan state slots or message edges")
        nodes = tuple(node for node in actor.operator_nodes if node.node_id != node_id)
        if not nodes:
            raise ValueError("remove operator cannot erase the final operator")
        updated = _rebuild_actor(actor, operator_nodes=nodes)
    else:
        parameters = _parameters(request, {"actor_id", "node"})
        actor = _required_actor(actors, parameters["actor_id"])
        updated = _replace_node(actor, _node_payload(parameters["node"], "replacement operator"))
    return _replace_actor(actors, updated)


def _message_edge_mutation(
    request: G0ConstructionRequest,
    actors: Mapping[str, G0ActorGenotype],
) -> dict[str, G0ActorGenotype]:
    if request.operation is G0ConstructionOperation.ADD_MESSAGE_EDGE:
        parameters = _parameters(request, {"actor_id", "edge", "owner_node"})
        actor = _required_actor(actors, parameters["actor_id"])
        edge = _edge_payload(parameters["edge"], "added message edge")
        if edge.edge_id in {row.edge_id for row in actor.message_edges}:
            raise ValueError("added message edge identity already exists")
        owner = _node_payload(parameters["owner_node"], "message-edge owner")
        edges = (*actor.message_edges, edge)
        _validate_message_owner(owner, edges)
        updated = _replace_node_and_edges(actor, owner, edges)
    else:
        parameters = _parameters(request, {"actor_id", "edge_id", "owner_node"})
        actor = _required_actor(actors, parameters["actor_id"])
        edge_id = _require_id(parameters["edge_id"], "removed message edge_id")
        if edge_id not in {edge.edge_id for edge in actor.message_edges}:
            raise ValueError("removed message edge identity does not exist")
        owner = _node_payload(parameters["owner_node"], "message-edge owner")
        edges = tuple(edge for edge in actor.message_edges if edge.edge_id != edge_id)
        _validate_message_owner(owner, edges)
        updated = _replace_node_and_edges(actor, owner, edges)
    return _replace_actor(actors, updated)


def _capacity_mutation(
    request: G0ConstructionRequest,
    actors: Mapping[str, G0ActorGenotype],
) -> dict[str, G0ActorGenotype]:
    if request.operation is G0ConstructionOperation.ADJUST_STATE_CAPACITY:
        parameters = _parameters(request, {"actor_id", "slot_id", "capacity_bytes"})
        actor = _required_actor(actors, parameters["actor_id"])
        slot_id = _require_id(parameters["slot_id"], "adjusted state slot_id")
        capacity = _positive_int(parameters["capacity_bytes"], "adjusted state capacity")
        if slot_id not in {slot.slot_id for slot in actor.state_slots}:
            raise ValueError("adjusted state slot does not exist")
        slots = tuple(
            G0StateSlot(slot.slot_id, slot.primitive, slot.schema_id, capacity)
            if slot.slot_id == slot_id
            else slot
            for slot in actor.state_slots
        )
        updated = _rebuild_actor(actor, state_slots=slots)
    else:
        parameters = _parameters(request, {"actor_id", "node_id", "window"})
        actor = _required_actor(actors, parameters["actor_id"])
        node_id = _require_id(parameters["node_id"], "temporal node_id")
        window = _positive_int(parameters["window"], "temporal window")
        by_id = {node.node_id: node for node in actor.operator_nodes}
        if node_id not in by_id:
            raise ValueError("temporal node does not exist")
        node = by_id[node_id]
        if node.operator is not OperatorPrimitive.TEMPORAL_ACCUMULATION:
            raise ValueError("timescale adjustment requires a temporal-accumulation operator")
        parameter_value = node.parameters.value()
        if not isinstance(parameter_value, dict) or "window" not in parameter_value:
            raise ValueError("temporal node has no explicit bounded window")
        if window > node.declared_operations:
            raise ValueError("temporal window exceeds its declared operation envelope")
        parameter_value["window"] = window
        replacement = G0OperatorNode.create(
            node_id=node.node_id,
            operator=node.operator,
            input_node_ids=node.input_node_ids,
            state_slot_ids=node.state_slot_ids,
            message_edge_ids=node.message_edge_ids,
            parameters=parameter_value,
            declared_operations=node.declared_operations,
            max_output_bytes=node.max_output_bytes,
        )
        updated = _replace_node(actor, replacement)
    return _replace_actor(actors, updated)


def _rewrite_refs(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_rewrite_refs(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_refs(item, replacements) for key, item in value.items()}
    return value


def _prefixed_actor(
    actor: G0ActorGenotype, prefix: str
) -> tuple[tuple[G0StateSlot, ...], tuple[G0OperatorNode, ...], tuple[G0MessageEdge, ...]]:
    state_map = {slot.slot_id: f"{prefix}.{slot.slot_id}" for slot in actor.state_slots}
    node_map = {node.node_id: f"{prefix}.{node.node_id}" for node in actor.operator_nodes}
    edge_map = {edge.edge_id: f"{prefix}.{edge.edge_id}" for edge in actor.message_edges}
    replacements = {**state_map, **node_map, **edge_map}
    states = tuple(
        G0StateSlot(state_map[slot.slot_id], slot.primitive, slot.schema_id, slot.capacity_bytes)
        for slot in actor.state_slots
    )
    edges = tuple(
        G0MessageEdge(edge_map[edge.edge_id], edge.schema_id, edge.max_encoded_bytes)
        for edge in actor.message_edges
    )
    nodes = tuple(
        G0OperatorNode.create(
            node_id=node_map[node.node_id],
            operator=node.operator,
            input_node_ids=tuple(node_map[value] for value in node.input_node_ids),
            state_slot_ids=tuple(state_map[value] for value in node.state_slot_ids),
            message_edge_ids=tuple(edge_map[value] for value in node.message_edge_ids),
            parameters=_rewrite_refs(node.parameters.value(), replacements),
            declared_operations=node.declared_operations,
            max_output_bytes=node.max_output_bytes,
        )
        for node in actor.operator_nodes
    )
    return states, nodes, edges


def _actor_population_mutation(
    request: G0ConstructionRequest,
    actors: Mapping[str, G0ActorGenotype],
    scopes: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, G0ActorGenotype], dict[str, tuple[str, ...]]]:
    result_actors = dict(actors)
    result_scopes = dict(scopes)
    if request.operation is G0ConstructionOperation.CLONE_ACTOR:
        parameters = _parameters(request, {"source_actor_id", "new_actor_id"})
        source = _required_actor(actors, parameters["source_actor_id"])
        new_id = _require_id(parameters["new_actor_id"], "cloned actor identity")
        if new_id in actors:
            raise ValueError("cloned actor identity already exists")
        clone = _rebuild_actor(source, candidate_id=new_id)
        result_actors[new_id] = clone
        result_scopes[new_id] = scopes[source.candidate_id]
    else:
        parameters = _parameters(
            request,
            {"left_actor_id", "right_actor_id", "target_actor_id"},
        )
        left = _required_actor(actors, parameters["left_actor_id"])
        right = _required_actor(actors, parameters["right_actor_id"])
        if left.candidate_id == right.candidate_id:
            raise ValueError("merged actors must be distinct")
        target_id = _require_id(parameters["target_actor_id"], "merged target actor identity")
        if target_id in actors and target_id not in {left.candidate_id, right.candidate_id}:
            raise ValueError("merged target actor identity belongs to an unrelated actor")
        left_states, left_nodes, left_edges = _prefixed_actor(left, "left")
        right_states, right_nodes, right_edges = _prefixed_actor(right, "right")
        merged = G0ActorGenotype.create(
            candidate_id=target_id,
            state_slots=(*left_states, *right_states),
            operator_nodes=(*left_nodes, *right_nodes),
            output_node_ids=_sink_ids((*left_nodes, *right_nodes)),
            message_edges=(*left_edges, *right_edges),
        )
        del result_actors[left.candidate_id]
        del result_actors[right.candidate_id]
        del result_scopes[left.candidate_id]
        del result_scopes[right.candidate_id]
        result_actors[target_id] = merged
        result_scopes[target_id] = tuple(
            sorted(set(scopes[left.candidate_id]) | set(scopes[right.candidate_id]))
        )
    return result_actors, result_scopes


def _factor_scope_mutation(
    request: G0ConstructionRequest,
    actors: Mapping[str, G0ActorGenotype],
    scopes: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    result = dict(scopes)
    if request.operation is G0ConstructionOperation.SPLIT_FACTOR_SCOPE:
        parameters = _parameters(
            request,
            {"actor_id", "source_scope_id", "left_scope_id", "right_scope_id"},
        )
        actor = _required_actor(actors, parameters["actor_id"])
        source = _require_id(parameters["source_scope_id"], "split source scope")
        left = _require_id(parameters["left_scope_id"], "split left scope")
        right = _require_id(parameters["right_scope_id"], "split right scope")
        if left == right:
            raise ValueError("split factor scopes must be distinct")
        current = set(scopes[actor.candidate_id])
        if source not in current:
            raise ValueError("split source factor scope does not exist")
        if ({left, right} - {source}) & current:
            raise ValueError("split factor scope collides with an existing scope")
        current.remove(source)
        current.update((left, right))
        result[actor.candidate_id] = tuple(sorted(current))
    else:
        parameters = _parameters(
            request,
            {"actor_id", "left_scope_id", "right_scope_id", "target_scope_id"},
        )
        actor = _required_actor(actors, parameters["actor_id"])
        left = _require_id(parameters["left_scope_id"], "merged left scope")
        right = _require_id(parameters["right_scope_id"], "merged right scope")
        target = _require_id(parameters["target_scope_id"], "merged target scope")
        if left == right:
            raise ValueError("merged factor scopes must be distinct")
        current = set(scopes[actor.candidate_id])
        if not {left, right} <= current:
            raise ValueError("merged factor scope does not exist")
        if target not in {left, right} and target in current:
            raise ValueError("merged factor scope target already exists")
        current.difference_update((left, right))
        current.add(target)
        result[actor.candidate_id] = tuple(sorted(current))
    return result


def _apply_request(
    request: G0ConstructionRequest,
    source: G0ConstructionSnapshot,
) -> G0ConstructionSnapshot:
    actors = _actor_map(source)
    scopes = _scope_map(source)
    if request.family is ConstructionMutation.ADD_REMOVE_OR_REPLACE_G0_OPERATOR:
        actors = _operator_mutation(request, actors)
    elif request.family is ConstructionMutation.ADD_OR_REMOVE_TYPED_MESSAGE_EDGE:
        actors = _message_edge_mutation(request, actors)
    elif request.family is ConstructionMutation.ADJUST_BOUNDED_TIMESCALE_OR_MEMORY_CAPACITY:
        actors = _capacity_mutation(request, actors)
    elif request.family in {
        ConstructionMutation.CLONE_ACTOR,
        ConstructionMutation.MERGE_CAUSALLY_REDUNDANT_ACTORS,
    }:
        actors, scopes = _actor_population_mutation(request, actors, scopes)
    elif request.family is ConstructionMutation.SPLIT_OR_MERGE_FACTOR_SCOPE:
        scopes = _factor_scope_mutation(request, actors, scopes)
    else:  # pragma: no cover - the typed enum and complete dispatch make this unreachable.
        raise ValueError("unsupported G0 construction family")
    return G0ConstructionSnapshot.create(actors=tuple(actors.values()), factor_scopes=scopes)


def _structural_problems(
    snapshot: G0ConstructionSnapshot,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    prefix: str,
) -> list[str]:
    problems: list[str] = []
    for actor in snapshot.actors:
        assessment = assess_g0_genotype(
            actor,
            grammar=grammar,
            candidate_registry=candidate_registry,
        )
        if not assessment.structurally_valid:
            structural = sorted(
                blocker
                for blocker in assessment.blockers
                if blocker
                in {
                    "actor-total-byte-cap-exceeded",
                    "candidate-excluded",
                    "candidate-registry-authority-mismatch",
                    "candidate-unknown",
                    "dag-depth-cap-exceeded",
                    "genotype-byte-cap-exceeded",
                    "message-byte-cap-exceeded",
                    "message-edge-cap-exceeded",
                    "operation-cap-exceeded",
                    "operator-node-cap-exceeded",
                    "output-byte-cap-exceeded",
                    "state-byte-cap-exceeded",
                    "undeclared-operator-primitive",
                    "undeclared-state-primitive",
                }
            )
            problems.extend(f"{prefix}:{actor.candidate_id}:{blocker}" for blocker in structural)
    return problems


def attempt_g0_construction(
    request: G0ConstructionRequest,
    *,
    source: G0ConstructionSnapshot,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> G0ConstructionAttempt:

    if type(request) is not G0ConstructionRequest:
        raise ValueError("request must be an exact G0ConstructionRequest")
    if type(source) is not G0ConstructionSnapshot:
        raise ValueError("source must be an exact G0ConstructionSnapshot")
    if type(grammar) is not TopologyGrammar:
        raise ValueError("grammar must be an exact TopologyGrammar")
    if type(candidate_registry) is not PerspectiveCandidateRegistry:
        raise ValueError("candidate_registry must be an exact PerspectiveCandidateRegistry")

    problems: list[str] = []
    if request.source_snapshot_sha256 != source.snapshot_sha256:
        problems.append("source-snapshot-authority-mismatch")
    if candidate_registry.sha256 != grammar.candidate_registry_sha256:
        problems.append("candidate-registry-authority-mismatch")
    if request.family not in grammar.construction_language.allowed_mutations:
        problems.append("construction-family-not-declared")
    if request.declared_work > grammar.caps.max_structural_work:
        problems.append("structural-work-cap-exceeded")
    problems.extend(
        _structural_problems(
            source,
            grammar=grammar,
            candidate_registry=candidate_registry,
            prefix="source",
        )
    )
    candidate: G0ConstructionSnapshot | None = None
    if not problems:
        try:
            proposed = _apply_request(request, source)
            actor_delta = len(proposed.actors) - len(source.actors)
            factor_delta = sum(len(row.scope_ids) for row in proposed.factor_scopes) - sum(
                len(row.scope_ids) for row in source.factor_scopes
            )
            state_delta = proposed.retained_state_bytes - source.retained_state_bytes
            if abs(actor_delta) > grammar.caps.max_actor_delta:
                problems.append("actor-delta-cap-exceeded")
            if abs(factor_delta) > grammar.caps.max_factor_scope_delta:
                problems.append("factor-scope-delta-cap-exceeded")
            if abs(state_delta) > grammar.caps.max_retained_state_bytes_delta:
                problems.append("retained-state-delta-cap-exceeded")
            problems.extend(
                _structural_problems(
                    proposed,
                    grammar=grammar,
                    candidate_registry=candidate_registry,
                    prefix="candidate",
                )
            )
            if not problems:
                candidate = proposed
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"construction-refused:{exc}")
    return G0ConstructionAttempt.create(
        request=request,
        source=source,
        grammar=grammar,
        candidate_registry=candidate_registry,
        candidate_snapshot=candidate,
        problems=problems,
    )


def verify_g0_construction_attempt(
    attempt: G0ConstructionAttempt,
    request: G0ConstructionRequest,
    *,
    source: G0ConstructionSnapshot,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> bool:

    if type(attempt) is not G0ConstructionAttempt:
        raise ValueError("attempt must be an exact G0ConstructionAttempt")
    replayed = attempt_g0_construction(
        request,
        source=source,
        grammar=grammar,
        candidate_registry=candidate_registry,
    )
    if replayed != attempt:
        raise ValueError("G0 construction attempt replay mismatch")
    return True


__all__ = [
    "G0_CONSTRUCTION_ATTEMPT_SCHEMA",
    "G0_CONSTRUCTION_REQUEST_SCHEMA",
    "G0_CONSTRUCTION_SNAPSHOT_SCHEMA",
    "G0ConstructionAttempt",
    "G0ConstructionOperation",
    "G0ConstructionRequest",
    "G0ConstructionSnapshot",
    "G0ConstructionStatus",
    "G0FactorScopeBinding",
    "attempt_g0_construction",
    "verify_g0_construction_attempt",
]
