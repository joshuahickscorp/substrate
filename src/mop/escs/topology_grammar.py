"""Finite, disabled-by-default topology grammar for the ESCS research scaffold.

The grammar describes a small set of pure topology rewrites.  It does not mutate a
running :class:`~mop.escs.runtime.CoalitionRuntime`: proposals operate on immutable
topology snapshots, and activation remains impossible until a separately sealed G0
freeze authority enables the exact operators.  This keeps structural adaptation
scaffoldable without turning an untested mechanism into runtime authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Self

from mop.substrate.events import FrozenJSON, canonical_sha256

from .accounting import WorkVector
from .perspective_registry import (
    IntegrationDisposition,
    PerspectiveCandidateRegistry,
    load_perspective_candidate_registry,
)

TOPOLOGY_GRAMMAR_SCHEMA = "mop-escs-topology-grammar/v1"
TOPOLOGY_SNAPSHOT_SCHEMA = "mop-escs-topology-snapshot/v1"
TOPOLOGY_MUTATION_SCHEMA = "mop-escs-topology-mutation/v1"
TOPOLOGY_TRANSACTION_SCHEMA = "mop-escs-topology-transaction/v1"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_REF_RE = re.compile(r"^[a-z][a-z0-9+.-]*:[a-z0-9][a-z0-9._:/-]*$")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _require_ref(value: object, label: str) -> str:
    if not isinstance(value, str) or _REF_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable namespaced reference")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _canonical_rows(values: Sequence[tuple[str, ...]], label: str) -> tuple[tuple[str, ...], ...]:
    rows = tuple(values)
    if len(rows) != len(set(rows)) or rows != tuple(sorted(rows)):
        raise ValueError(f"{label} must be unique and canonically sorted")
    return rows


class GrammarStatus(StrEnum):
    SCAFFOLD = "scaffold"
    FROZEN = "frozen"


class MutationKind(StrEnum):
    ADD_ACTOR_SLOT = "add_actor_slot"
    RETIRE_ACTOR_SLOT = "retire_actor_slot"
    ADD_ROUTING_SUBSCRIPTION = "add_routing_subscription"
    REMOVE_ROUTING_SUBSCRIPTION = "remove_routing_subscription"
    ADD_PEER_EDGE = "add_peer_edge"
    REMOVE_PEER_EDGE = "remove_peer_edge"
    ADD_FACTOR_SCOPE = "add_factor_scope"
    REMOVE_FACTOR_SCOPE = "remove_factor_scope"
    SWAP_TO_SPARE = "swap_to_spare"


class TopologyGuard(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"
    COMMITMENT = "commitment"
    CONSEQUENCE = "consequence"
    ROLLBACK = "rollback"
    LIFECYCLE_ACCOUNTING = "lifecycle-accounting"
    MUTATION_BUDGET = "mutation-budget"
    STATE_DIGEST = "state-digest"


_MANDATORY_GUARDS = frozenset(TopologyGuard)


class StatePrimitive(StrEnum):
    BOUNDED_CATEGORICAL_TABLES = "bounded_categorical_tables"
    BOUNDED_RECURRENT_STATE = "bounded_recurrent_state"
    BOUNDED_SCALAR_OR_VECTOR_DISTRIBUTIONS = "bounded_scalar_or_vector_distributions"
    BOUNDED_TEMPORAL_DEQUES = "bounded_temporal_deques"
    BOUNDED_TYPED_FACTOR_SUBGRAPHS = "bounded_typed_factor_subgraphs"


class OperatorPrimitive(StrEnum):
    AFFINE_OR_NONLINEAR_LOCAL_UPDATE = "affine_or_nonlinear_local_update"
    BOUNDED_RETRIEVAL = "bounded_retrieval"
    BOUNDED_ROLLOUT_OR_SEARCH = "bounded_rollout_or_search"
    CONSTRAINT_FILTERING_OR_PROPAGATION = "constraint_filtering_or_propagation"
    GRAPH_NEIGHBORHOOD_AGGREGATION = "graph_neighborhood_aggregation"
    TABLE_LOOKUP_OR_UPDATE = "table_lookup_or_update"
    TEMPORAL_ACCUMULATION = "temporal_accumulation"
    TYPED_MESSAGE_TRANSFORM_AND_EMIT = "typed_message_transform_and_emit"


class ConstructionMutation(StrEnum):
    ADD_REMOVE_OR_REPLACE_G0_OPERATOR = "add_remove_or_replace_g0_operator"
    ADD_OR_REMOVE_TYPED_MESSAGE_EDGE = "add_or_remove_typed_message_edge"
    ADJUST_BOUNDED_TIMESCALE_OR_MEMORY_CAPACITY = (
        "adjust_bounded_timescale_or_memory_capacity"
    )
    CLONE_ACTOR = "clone_actor"
    MERGE_CAUSALLY_REDUNDANT_ACTORS = "merge_causally_redundant_actors"
    SPLIT_OR_MERGE_FACTOR_SCOPE = "split_or_merge_factor_scope"


_FORBIDDEN_CONSTRUCTION_CAPABILITIES = (
    "arbitrary_generated_code",
    "mutation_outside_transactional_governance",
    "silent_schema_creation",
    "unbounded_recursion",
    "undeclared_operators",
)


@dataclass(frozen=True, slots=True)
class CompositionBounds:
    max_operator_nodes_per_actor: int
    max_dag_depth: int
    max_state_bytes_per_actor: int
    max_operations_per_activation: int
    max_message_bytes: int
    max_message_edges: int

    def __post_init__(self) -> None:
        for name, value in self.payload().items():
            if _require_nonnegative_int(value, f"CompositionBounds.{name}") == 0:
                raise ValueError(f"CompositionBounds.{name} must be positive")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = set(cls.__dataclass_fields__)
        _require_exact_keys(payload, expected, "CompositionBounds")
        return cls(**{name: payload[name] for name in expected})

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ConstructionLanguage:
    state_primitives: tuple[StatePrimitive, ...]
    operator_primitives: tuple[OperatorPrimitive, ...]
    composition_bounds: CompositionBounds
    allowed_mutations: tuple[ConstructionMutation, ...]
    forbidden_capabilities: tuple[str, ...]
    implementation_complete: bool

    def __post_init__(self) -> None:
        finite_sets: tuple[tuple[Sequence[StrEnum], frozenset[StrEnum], str], ...] = (
            (self.state_primitives, frozenset(StatePrimitive), "state_primitives"),
            (self.operator_primitives, frozenset(OperatorPrimitive), "operator_primitives"),
            (self.allowed_mutations, frozenset(ConstructionMutation), "allowed_mutations"),
        )
        for value_sequence, expected, label in finite_sets:
            values = tuple(value_sequence)
            if values != tuple(sorted(values, key=str)) or len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique and canonically sorted")
            if set(values) != expected:
                raise ValueError(f"{label} must declare the complete finite G0 vocabulary")
        if self.forbidden_capabilities != _FORBIDDEN_CONSTRUCTION_CAPABILITIES:
            raise ValueError("forbidden construction capabilities drifted")
        if not isinstance(self.implementation_complete, bool):
            raise ValueError("implementation_complete must be boolean")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {
                "state_primitives",
                "operator_primitives",
                "composition_bounds",
                "allowed_mutations",
                "forbidden_capabilities",
                "implementation_complete",
            },
            "ConstructionLanguage",
        )
        return cls(
            state_primitives=tuple(StatePrimitive(value) for value in payload["state_primitives"]),
            operator_primitives=tuple(
                OperatorPrimitive(value) for value in payload["operator_primitives"]
            ),
            composition_bounds=CompositionBounds.from_payload(payload["composition_bounds"]),
            allowed_mutations=tuple(
                ConstructionMutation(value) for value in payload["allowed_mutations"]
            ),
            forbidden_capabilities=tuple(str(value) for value in payload["forbidden_capabilities"]),
            implementation_complete=payload["implementation_complete"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "state_primitives": [value.value for value in self.state_primitives],
            "operator_primitives": [value.value for value in self.operator_primitives],
            "composition_bounds": self.composition_bounds.payload(),
            "allowed_mutations": [value.value for value in self.allowed_mutations],
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "implementation_complete": self.implementation_complete,
        }


@dataclass(frozen=True, slots=True)
class FreezeAuthority:
    artifact_path: str
    artifact_sha256: str
    artifact_schema: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.artifact_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("freeze authority path must be repository-relative")
        _require_digest(self.artifact_sha256, "freeze authority digest")
        if not self.artifact_schema.strip():
            raise ValueError("freeze authority schema must be nonempty")

    def payload(self) -> dict[str, str]:
        return {
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_schema": self.artifact_schema,
        }


@dataclass(frozen=True, slots=True)
class GrammarCaps:
    max_mutations_per_transaction: int
    max_actor_delta: int
    max_routing_delta: int
    max_peer_edge_delta: int
    max_factor_scope_delta: int
    max_shadow_events: int
    max_canary_events: int
    max_rollback_events: int
    max_structural_work: int
    max_retained_state_bytes_delta: int

    def __post_init__(self) -> None:
        for name, value in self.payload().items():
            if _require_nonnegative_int(value, f"GrammarCaps.{name}") == 0:
                raise ValueError(f"GrammarCaps.{name} must be positive")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        expected = set(cls.__dataclass_fields__)
        _require_exact_keys(payload, expected, "GrammarCaps")
        return cls(**{name: payload[name] for name in expected})

    def payload(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OperatorRule:
    kind: MutationKind
    enabled: bool
    max_per_transaction: int
    required_guards: tuple[TopologyGuard, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MutationKind):
            raise ValueError("operator kind must be a MutationKind")
        if not isinstance(self.enabled, bool):
            raise ValueError("operator enabled must be boolean")
        if _require_nonnegative_int(self.max_per_transaction, "max_per_transaction") == 0:
            raise ValueError("max_per_transaction must be positive")
        if not isinstance(self.required_guards, tuple):
            raise ValueError("required_guards must be an immutable tuple")
        if tuple(sorted(self.required_guards, key=str)) != self.required_guards:
            raise ValueError("required_guards must be canonically sorted")
        if len(set(self.required_guards)) != len(self.required_guards):
            raise ValueError("required_guards must be unique")
        if not set(self.required_guards) >= _MANDATORY_GUARDS:
            raise ValueError("every topology operator must retain all mandatory guards")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {"kind", "enabled", "max_per_transaction", "required_guards"},
            "OperatorRule",
        )
        guards = payload["required_guards"]
        if not isinstance(guards, list):
            raise ValueError("required_guards must be a list")
        return cls(
            kind=MutationKind(payload["kind"]),
            enabled=payload["enabled"],
            max_per_transaction=payload["max_per_transaction"],
            required_guards=tuple(TopologyGuard(value) for value in guards),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "enabled": self.enabled,
            "max_per_transaction": self.max_per_transaction,
            "required_guards": [guard.value for guard in self.required_guards],
        }


@dataclass(frozen=True, slots=True)
class TopologyGrammar:
    grammar_id: str
    revision: int
    status: GrammarStatus
    activation_enabled: bool
    scientific_promotion_allowed: bool
    candidate_registry_path: str
    candidate_registry_sha256: str
    construction_language: ConstructionLanguage
    caps: GrammarCaps
    operators: tuple[OperatorRule, ...]
    freeze_authority: FreezeAuthority | None
    grammar_sha256: str
    schema: str = TOPOLOGY_GRAMMAR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOPOLOGY_GRAMMAR_SCHEMA:
            raise ValueError(f"unsupported topology grammar schema {self.schema!r}")
        _require_id(self.grammar_id, "grammar_id")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("grammar revision must be a positive integer")
        if not isinstance(self.status, GrammarStatus):
            raise ValueError("grammar status must be a GrammarStatus")
        if not isinstance(self.activation_enabled, bool):
            raise ValueError("activation_enabled must be boolean")
        if self.scientific_promotion_allowed is not False:
            raise ValueError("topology grammar cannot grant scientific promotion")
        registry_path = PurePosixPath(self.candidate_registry_path)
        if registry_path.is_absolute() or ".." in registry_path.parts or not registry_path.parts:
            raise ValueError("candidate registry path must be repository-relative")
        _require_digest(self.candidate_registry_sha256, "candidate_registry_sha256")
        if not isinstance(self.construction_language, ConstructionLanguage):
            raise ValueError("construction_language must be a finite ConstructionLanguage")
        if not isinstance(self.operators, tuple) or not self.operators:
            raise ValueError("operators must be a nonempty immutable tuple")
        kinds = tuple(rule.kind for rule in self.operators)
        if kinds != tuple(sorted(kinds, key=str)) or len(set(kinds)) != len(kinds):
            raise ValueError("operators must be complete, unique, and canonically sorted")
        if set(kinds) != set(MutationKind):
            raise ValueError("the finite grammar must declare every known mutation kind")
        if self.status is GrammarStatus.SCAFFOLD:
            if self.activation_enabled or self.freeze_authority is not None:
                raise ValueError("scaffold grammar cannot activate or carry freeze authority")
            if any(rule.enabled for rule in self.operators):
                raise ValueError("all scaffold mutation operators must be disabled")
        else:
            if not self.activation_enabled or self.freeze_authority is None:
                raise ValueError("frozen grammar requires activation and a freeze authority")
            if not self.construction_language.implementation_complete:
                raise ValueError("frozen grammar requires a complete G0 construction implementation")
            if not any(rule.enabled for rule in self.operators):
                raise ValueError("frozen grammar must enable at least one exact operator")
        _require_digest(self.grammar_sha256, "grammar_sha256")
        if canonical_sha256(self.payload(include_digest=False)) != self.grammar_sha256:
            raise ValueError("topology grammar self-hash mismatch")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _require_exact_keys(
            payload,
            {
                "schema",
                "grammar_id",
                "revision",
                "status",
                "activation_enabled",
                "scientific_promotion_allowed",
                "candidate_registry_path",
                "candidate_registry_sha256",
                "construction_language",
                "caps",
                "operators",
                "freeze_authority",
                "grammar_sha256",
            },
            "TopologyGrammar",
        )
        operators = payload["operators"]
        if not isinstance(operators, list):
            raise ValueError("operators must be a list")
        authority_raw = payload["freeze_authority"]
        authority: FreezeAuthority | None = None
        if authority_raw is not None:
            _require_exact_keys(
                authority_raw,
                {"artifact_path", "artifact_sha256", "artifact_schema"},
                "FreezeAuthority",
            )
            authority = FreezeAuthority(**authority_raw)
        return cls(
            schema=payload["schema"],
            grammar_id=payload["grammar_id"],
            revision=payload["revision"],
            status=GrammarStatus(payload["status"]),
            activation_enabled=payload["activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            candidate_registry_path=payload["candidate_registry_path"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            construction_language=ConstructionLanguage.from_payload(payload["construction_language"]),
            caps=GrammarCaps.from_payload(payload["caps"]),
            operators=tuple(OperatorRule.from_payload(row) for row in operators),
            freeze_authority=authority,
            grammar_sha256=payload["grammar_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "grammar_id": self.grammar_id,
            "revision": self.revision,
            "status": self.status.value,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "candidate_registry_path": self.candidate_registry_path,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "construction_language": self.construction_language.payload(),
            "caps": self.caps.payload(),
            "operators": [rule.payload() for rule in self.operators],
            "freeze_authority": self.freeze_authority.payload() if self.freeze_authority else None,
        }
        if include_digest:
            result["grammar_sha256"] = self.grammar_sha256
        return result

    def rule(self, kind: MutationKind) -> OperatorRule:
        return next(rule for rule in self.operators if rule.kind is kind)


@dataclass(frozen=True, slots=True)
class ActorSlot:
    actor_id: str
    candidate_id: str
    state_version: str
    active: bool
    spare_for: str | None = None

    def __post_init__(self) -> None:
        _require_ref(self.actor_id, "actor_id")
        _require_id(self.candidate_id, "candidate_id")
        _require_digest(self.state_version, "actor state_version")
        if not isinstance(self.active, bool):
            raise ValueError("actor active must be boolean")
        if self.spare_for is not None:
            _require_ref(self.spare_for, "spare_for")
            if self.spare_for == self.actor_id or self.active:
                raise ValueError("a spare must be inactive and refer to a different actor")

    def payload(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "candidate_id": self.candidate_id,
            "state_version": self.state_version,
            "active": self.active,
            "spare_for": self.spare_for,
        }


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    actors: tuple[ActorSlot, ...]
    routing_subscriptions: tuple[tuple[str, str], ...] = ()
    peer_edges: tuple[tuple[str, str], ...] = ()
    factor_scopes: tuple[tuple[str, str], ...] = ()
    schema: str = TOPOLOGY_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOPOLOGY_SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported topology snapshot schema {self.schema!r}")
        if not isinstance(self.actors, tuple):
            raise ValueError("actors must be an immutable tuple")
        actor_ids = tuple(actor.actor_id for actor in self.actors)
        if actor_ids != tuple(sorted(actor_ids)) or len(set(actor_ids)) != len(actor_ids):
            raise ValueError("actors must be unique and canonically sorted")
        active_ids = {actor.actor_id for actor in self.actors if actor.active}
        for rows, label in (
            (self.routing_subscriptions, "routing_subscriptions"),
            (self.peer_edges, "peer_edges"),
            (self.factor_scopes, "factor_scopes"),
        ):
            _canonical_rows(rows, label)
            for row in rows:
                if len(row) != 2:
                    raise ValueError(f"{label} rows must have exactly two values")
        for actor_id, shard in self.routing_subscriptions:
            if actor_id not in active_ids or not shard.strip():
                raise ValueError("subscriptions require an active actor and nonempty shard")
        for source, target in self.peer_edges:
            if source not in active_ids or target not in active_ids or source == target:
                raise ValueError("peer edges require two distinct active actors")
        for actor_id, factor in self.factor_scopes:
            if actor_id not in active_ids or not factor.strip():
                raise ValueError("factor scopes require an active actor and nonempty factor")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "actors": [actor.payload() for actor in self.actors],
            "routing_subscriptions": [list(row) for row in self.routing_subscriptions],
            "peer_edges": [list(row) for row in self.peer_edges],
            "factor_scopes": [list(row) for row in self.factor_scopes],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


_PARAMETERS_BY_KIND: Mapping[MutationKind, frozenset[str]] = {
    MutationKind.ADD_ACTOR_SLOT: frozenset(
        {"actor_id", "candidate_id", "state_version", "active", "spare_for"}
    ),
    MutationKind.RETIRE_ACTOR_SLOT: frozenset({"actor_id"}),
    MutationKind.ADD_ROUTING_SUBSCRIPTION: frozenset({"actor_id", "shard"}),
    MutationKind.REMOVE_ROUTING_SUBSCRIPTION: frozenset({"actor_id", "shard"}),
    MutationKind.ADD_PEER_EDGE: frozenset({"source_actor_id", "target_actor_id"}),
    MutationKind.REMOVE_PEER_EDGE: frozenset({"source_actor_id", "target_actor_id"}),
    MutationKind.ADD_FACTOR_SCOPE: frozenset({"actor_id", "factor"}),
    MutationKind.REMOVE_FACTOR_SCOPE: frozenset({"actor_id", "factor"}),
    MutationKind.SWAP_TO_SPARE: frozenset({"failed_actor_id", "spare_actor_id"}),
}


@dataclass(frozen=True, slots=True)
class TopologyMutation:
    mutation_id: str
    kind: MutationKind
    parameters: FrozenJSON
    declared_work: WorkVector
    retained_state_bytes_delta: int
    schema: str = TOPOLOGY_MUTATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOPOLOGY_MUTATION_SCHEMA:
            raise ValueError(f"unsupported topology mutation schema {self.schema!r}")
        _require_digest(self.mutation_id, "mutation_id")
        if not isinstance(self.kind, MutationKind):
            raise ValueError("mutation kind must be a MutationKind")
        if not isinstance(self.parameters, FrozenJSON) or not isinstance(
            self.parameters.value(), dict
        ):
            raise ValueError("mutation parameters must be a frozen JSON object")
        parameter_value = self.parameters.value()
        assert isinstance(parameter_value, dict)
        _require_exact_keys(parameter_value, set(_PARAMETERS_BY_KIND[self.kind]), "mutation parameters")
        if not isinstance(self.declared_work, WorkVector) or self.declared_work.total_work <= 0:
            raise ValueError("topology mutation must declare nonzero lifecycle work")
        _require_nonnegative_int(self.retained_state_bytes_delta, "retained_state_bytes_delta")
        if canonical_sha256(self.payload(include_digest=False)) != self.mutation_id:
            raise ValueError("topology mutation identity mismatch")

    @classmethod
    def create(
        cls,
        *,
        kind: MutationKind,
        parameters: Mapping[str, Any],
        declared_work: WorkVector,
        retained_state_bytes_delta: int = 0,
    ) -> Self:
        frozen = FrozenJSON.from_value(dict(parameters))
        partial = {
            "schema": TOPOLOGY_MUTATION_SCHEMA,
            "kind": kind.value,
            "parameters": {"value": frozen.value(), "sha256": frozen.sha256},
            "declared_work": declared_work.payload(),
            "retained_state_bytes_delta": retained_state_bytes_delta,
        }
        return cls(
            mutation_id=canonical_sha256(partial),
            kind=kind,
            parameters=frozen,
            declared_work=declared_work,
            retained_state_bytes_delta=retained_state_bytes_delta,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "parameters": {"value": self.parameters.value(), "sha256": self.parameters.sha256},
            "declared_work": self.declared_work.payload(),
            "retained_state_bytes_delta": self.retained_state_bytes_delta,
        }
        if include_digest:
            result["mutation_id"] = self.mutation_id
        return result


def _replace_actor(snapshot: TopologySnapshot, actor: ActorSlot) -> tuple[ActorSlot, ...]:
    rows = (actor if row.actor_id == actor.actor_id else row for row in snapshot.actors)
    return tuple(sorted(rows, key=lambda row: row.actor_id))


def apply_topology_mutation(snapshot: TopologySnapshot, mutation: TopologyMutation) -> TopologySnapshot:
    """Apply one mutation to an immutable shadow snapshot, never to a live runtime."""

    values = mutation.parameters.value()
    assert isinstance(values, dict)
    actors = {actor.actor_id: actor for actor in snapshot.actors}
    routing = set(snapshot.routing_subscriptions)
    edges = set(snapshot.peer_edges)
    scopes = set(snapshot.factor_scopes)

    if mutation.kind is MutationKind.ADD_ACTOR_SLOT:
        added_actor = ActorSlot(**values)
        if added_actor.actor_id in actors:
            raise ValueError("actor slot already exists")
        new_actors = tuple(sorted((*snapshot.actors, added_actor), key=lambda row: row.actor_id))
    elif mutation.kind is MutationKind.RETIRE_ACTOR_SLOT:
        actor_id = _require_ref(values["actor_id"], "actor_id")
        retiring_actor = actors.get(actor_id)
        if retiring_actor is None or not retiring_actor.active:
            raise ValueError("retirement requires an active actor")
        new_actors = _replace_actor(snapshot, replace(retiring_actor, active=False))
        routing = {row for row in routing if row[0] != actor_id}
        edges = {row for row in edges if actor_id not in row}
        scopes = {row for row in scopes if row[0] != actor_id}
    elif mutation.kind in {
        MutationKind.ADD_ROUTING_SUBSCRIPTION,
        MutationKind.REMOVE_ROUTING_SUBSCRIPTION,
    }:
        actor_id = _require_ref(values["actor_id"], "actor_id")
        shard = str(values["shard"])
        if actor_id not in actors or not actors[actor_id].active or not shard.strip():
            raise ValueError("routing mutation requires an active actor and nonempty shard")
        row = (actor_id, shard)
        if mutation.kind is MutationKind.ADD_ROUTING_SUBSCRIPTION:
            if row in routing:
                raise ValueError("routing subscription already exists")
            routing.add(row)
        elif row not in routing:
            raise ValueError("routing subscription does not exist")
        else:
            routing.remove(row)
        new_actors = snapshot.actors
    elif mutation.kind in {MutationKind.ADD_PEER_EDGE, MutationKind.REMOVE_PEER_EDGE}:
        row = (
            _require_ref(values["source_actor_id"], "source_actor_id"),
            _require_ref(values["target_actor_id"], "target_actor_id"),
        )
        if row[0] == row[1] or any(actor_id not in actors or not actors[actor_id].active for actor_id in row):
            raise ValueError("peer mutation requires two distinct active actors")
        if mutation.kind is MutationKind.ADD_PEER_EDGE:
            if row in edges:
                raise ValueError("peer edge already exists")
            edges.add(row)
        elif row not in edges:
            raise ValueError("peer edge does not exist")
        else:
            edges.remove(row)
        new_actors = snapshot.actors
    elif mutation.kind in {MutationKind.ADD_FACTOR_SCOPE, MutationKind.REMOVE_FACTOR_SCOPE}:
        actor_id = _require_ref(values["actor_id"], "actor_id")
        factor = str(values["factor"])
        if actor_id not in actors or not actors[actor_id].active or not factor.strip():
            raise ValueError("scope mutation requires an active actor and nonempty factor")
        row = (actor_id, factor)
        if mutation.kind is MutationKind.ADD_FACTOR_SCOPE:
            if row in scopes:
                raise ValueError("factor scope already exists")
            scopes.add(row)
        elif row not in scopes:
            raise ValueError("factor scope does not exist")
        else:
            scopes.remove(row)
        new_actors = snapshot.actors
    else:
        failed_id = _require_ref(values["failed_actor_id"], "failed_actor_id")
        spare_id = _require_ref(values["spare_actor_id"], "spare_actor_id")
        failed = actors.get(failed_id)
        spare = actors.get(spare_id)
        if failed is None or not failed.active:
            raise ValueError("spare swap requires an active failed actor")
        if spare is None or spare.active or spare.spare_for != failed_id:
            raise ValueError("spare swap requires the registered inactive spare")
        rows = [
            replace(failed, active=False),
            replace(spare, active=True, spare_for=None),
            *(actor for actor in snapshot.actors if actor.actor_id not in {failed_id, spare_id}),
        ]
        new_actors = tuple(sorted(rows, key=lambda row: row.actor_id))
        routing = {(spare_id if actor == failed_id else actor, shard) for actor, shard in routing}
        edges = {
            (
                spare_id if source == failed_id else source,
                spare_id if target == failed_id else target,
            )
            for source, target in edges
        }
        scopes = {(spare_id if actor == failed_id else actor, factor) for actor, factor in scopes}

    return TopologySnapshot(
        actors=new_actors,
        routing_subscriptions=tuple(sorted(routing)),
        peer_edges=tuple(sorted(edges)),
        factor_scopes=tuple(sorted(scopes)),
    )


@dataclass(frozen=True, slots=True)
class TransactionAssessment:
    structurally_valid: bool
    shadow_authorized: bool
    factual_commitment_authorized: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopologyTransaction:
    transaction_id: str
    grammar_sha256: str
    base_topology_sha256: str
    proposed_topology_sha256: str
    mutation_ids: tuple[str, ...]
    declared_work: WorkVector
    retained_state_bytes_delta: int
    shadow_trace_sha256: str | None = None
    canary_trace_sha256: str | None = None
    commitment_event_id: str | None = None
    consequence_event_id: str | None = None
    rollback_snapshot_sha256: str | None = None
    scientific_promotion: bool = False
    schema: str = TOPOLOGY_TRANSACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TOPOLOGY_TRANSACTION_SCHEMA:
            raise ValueError(f"unsupported topology transaction schema {self.schema!r}")
        for value, label in (
            (self.transaction_id, "transaction_id"),
            (self.grammar_sha256, "grammar_sha256"),
            (self.base_topology_sha256, "base_topology_sha256"),
            (self.proposed_topology_sha256, "proposed_topology_sha256"),
        ):
            _require_digest(value, label)
        if not self.mutation_ids:
            raise ValueError("mutation_ids must be nonempty")
        if len(set(self.mutation_ids)) != len(self.mutation_ids):
            raise ValueError("mutation_ids must be unique")
        for value in self.mutation_ids:
            _require_digest(value, "mutation_id")
        if self.declared_work.total_work <= 0:
            raise ValueError("transaction must carry nonzero declared work")
        _require_nonnegative_int(self.retained_state_bytes_delta, "retained_state_bytes_delta")
        for optional_digest, label in (
            (self.shadow_trace_sha256, "shadow_trace_sha256"),
            (self.canary_trace_sha256, "canary_trace_sha256"),
            (self.rollback_snapshot_sha256, "rollback_snapshot_sha256"),
        ):
            if optional_digest is not None:
                _require_digest(optional_digest, label)
        for optional_ref, label in (
            (self.commitment_event_id, "commitment_event_id"),
            (self.consequence_event_id, "consequence_event_id"),
        ):
            if optional_ref is not None:
                _require_ref(optional_ref, label)
        if self.scientific_promotion is not False:
            raise ValueError("topology transaction cannot grant scientific promotion")
        if canonical_sha256(self.payload(include_digest=False)) != self.transaction_id:
            raise ValueError("topology transaction identity mismatch")

    @classmethod
    def propose(
        cls,
        *,
        grammar: TopologyGrammar,
        base: TopologySnapshot,
        mutations: Sequence[TopologyMutation],
    ) -> tuple[Self, TopologySnapshot]:
        rows = tuple(mutations)
        if not rows or len({row.mutation_id for row in rows}) != len(rows):
            raise ValueError("transaction mutations must be nonempty and unique")
        proposed = base
        for mutation in rows:
            proposed = apply_topology_mutation(proposed, mutation)
        work = sum((row.declared_work for row in rows), WorkVector.zero())
        retained_delta = sum(row.retained_state_bytes_delta for row in rows)
        mutation_ids = tuple(row.mutation_id for row in rows)
        partial: dict[str, Any] = {
            "schema": TOPOLOGY_TRANSACTION_SCHEMA,
            "grammar_sha256": grammar.grammar_sha256,
            "base_topology_sha256": base.sha256,
            "proposed_topology_sha256": proposed.sha256,
            "mutation_ids": list(mutation_ids),
            "declared_work": work.payload(),
            "retained_state_bytes_delta": retained_delta,
            "shadow_trace_sha256": None,
            "canary_trace_sha256": None,
            "commitment_event_id": None,
            "consequence_event_id": None,
            "rollback_snapshot_sha256": None,
            "scientific_promotion": False,
        }
        transaction = cls(
            transaction_id=canonical_sha256(partial),
            grammar_sha256=grammar.grammar_sha256,
            base_topology_sha256=base.sha256,
            proposed_topology_sha256=proposed.sha256,
            mutation_ids=mutation_ids,
            declared_work=work,
            retained_state_bytes_delta=retained_delta,
        )
        return transaction, proposed

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "grammar_sha256": self.grammar_sha256,
            "base_topology_sha256": self.base_topology_sha256,
            "proposed_topology_sha256": self.proposed_topology_sha256,
            "mutation_ids": list(self.mutation_ids),
            "declared_work": self.declared_work.payload(),
            "retained_state_bytes_delta": self.retained_state_bytes_delta,
            "shadow_trace_sha256": self.shadow_trace_sha256,
            "canary_trace_sha256": self.canary_trace_sha256,
            "commitment_event_id": self.commitment_event_id,
            "consequence_event_id": self.consequence_event_id,
            "rollback_snapshot_sha256": self.rollback_snapshot_sha256,
            "scientific_promotion": self.scientific_promotion,
        }
        if include_digest:
            result["transaction_id"] = self.transaction_id
        return result


def assess_topology_transaction(
    grammar: TopologyGrammar,
    transaction: TopologyTransaction,
    mutations: Sequence[TopologyMutation],
    *,
    base: TopologySnapshot,
    proposed: TopologySnapshot,
    candidate_registry: PerspectiveCandidateRegistry | None = None,
    freeze_authority_verified: bool = False,
) -> TransactionAssessment:
    rows = tuple(mutations)
    blockers: list[str] = []
    structural_blockers: list[str] = []
    if transaction.grammar_sha256 != grammar.grammar_sha256:
        structural_blockers.append("grammar-authority-mismatch")
    if tuple(row.mutation_id for row in rows) != transaction.mutation_ids:
        structural_blockers.append("mutation-authority-mismatch")
    if transaction.base_topology_sha256 != base.sha256:
        structural_blockers.append("base-topology-authority-mismatch")
    if transaction.proposed_topology_sha256 != proposed.sha256:
        structural_blockers.append("proposed-topology-authority-mismatch")
    if candidate_registry is None:
        structural_blockers.append("candidate-registry-unverified")
    elif candidate_registry.sha256 != grammar.candidate_registry_sha256:
        structural_blockers.append("candidate-registry-authority-mismatch")
    else:
        candidates = {row.candidate_id: row for row in candidate_registry.candidates}
        base_active = {row.actor_id for row in base.actors if row.active}
        for actor in proposed.actors:
            candidate = candidates.get(actor.candidate_id)
            if candidate is None:
                structural_blockers.append(f"unknown-candidate:{actor.candidate_id}")
                continue
            if candidate.integration_disposition is IntegrationDisposition.EXCLUDED:
                structural_blockers.append(f"excluded-candidate:{actor.candidate_id}")
            if actor.active and actor.actor_id not in base_active and not candidate.activation_enabled:
                blockers.append(f"candidate-activation-disabled:{actor.candidate_id}")
    try:
        replayed = base
        for mutation in rows:
            replayed = apply_topology_mutation(replayed, mutation)
        if replayed.sha256 != proposed.sha256:
            structural_blockers.append("topology-replay-mismatch")
    except ValueError:
        structural_blockers.append("topology-replay-invalid")
    if len(rows) > grammar.caps.max_mutations_per_transaction:
        structural_blockers.append("transaction-mutation-cap")
    counts: dict[MutationKind, int] = {}
    for mutation in rows:
        counts[mutation.kind] = counts.get(mutation.kind, 0) + 1
        rule = grammar.rule(mutation.kind)
        if not rule.enabled:
            blockers.append(f"operator-disabled:{mutation.kind.value}")
        if counts[mutation.kind] > rule.max_per_transaction:
            structural_blockers.append(f"operator-cap:{mutation.kind.value}")
    if transaction.declared_work.total_work > grammar.caps.max_structural_work:
        structural_blockers.append("structural-work-cap")
    if transaction.retained_state_bytes_delta > grammar.caps.max_retained_state_bytes_delta:
        structural_blockers.append("retained-state-delta-cap")
    base_actors = {row.actor_id: row for row in base.actors}
    proposed_actors = {row.actor_id: row for row in proposed.actors}
    actor_delta = len(set(base_actors) ^ set(proposed_actors)) + sum(
        base_actors[actor_id] != proposed_actors[actor_id]
        for actor_id in set(base_actors) & set(proposed_actors)
    )
    topology_deltas = (
        (actor_delta, grammar.caps.max_actor_delta, "actor-delta-cap"),
        (
            len(set(base.routing_subscriptions) ^ set(proposed.routing_subscriptions)),
            grammar.caps.max_routing_delta,
            "routing-delta-cap",
        ),
        (
            len(set(base.peer_edges) ^ set(proposed.peer_edges)),
            grammar.caps.max_peer_edge_delta,
            "peer-edge-delta-cap",
        ),
        (
            len(set(base.factor_scopes) ^ set(proposed.factor_scopes)),
            grammar.caps.max_factor_scope_delta,
            "factor-scope-delta-cap",
        ),
    )
    for observed, cap, problem in topology_deltas:
        if observed > cap:
            structural_blockers.append(problem)
    if grammar.status is not GrammarStatus.FROZEN:
        blockers.append("grammar-not-frozen")
    elif not freeze_authority_verified:
        blockers.append("freeze-authority-unverified")
    if not grammar.activation_enabled:
        blockers.append("grammar-activation-disabled")
    blockers = sorted(set((*blockers, *structural_blockers)))
    shadow = not blockers
    factual = (
        shadow
        and transaction.shadow_trace_sha256 is not None
        and transaction.canary_trace_sha256 is not None
        and transaction.commitment_event_id is not None
        and transaction.rollback_snapshot_sha256 == transaction.base_topology_sha256
        and transaction.consequence_event_id is None
    )
    return TransactionAssessment(not structural_blockers, shadow, factual, tuple(blockers))


def load_topology_grammar(path: str | Path) -> TopologyGrammar:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("topology grammar artifact must be a JSON object")
    return TopologyGrammar.from_payload(payload)


def verify_freeze_authority(grammar: TopologyGrammar, repository_root: str | Path) -> tuple[str, ...]:
    """Verify the exact external artifact that is allowed to freeze G0.

    A self-hashed grammar cannot authorize itself.  Frozen callers must perform this
    separate join and pass its success explicitly to :func:`assess_topology_transaction`.
    """

    authority = grammar.freeze_authority
    if grammar.status is not GrammarStatus.FROZEN or authority is None:
        return ("grammar-has-no-freeze-authority",)
    root = Path(repository_root).resolve()
    target = (root / authority.artifact_path).resolve()
    if not target.is_relative_to(root):
        return ("freeze-authority-path-escapes-root",)
    if not target.is_file():
        return ("freeze-authority-artifact-missing",)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    problems: list[str] = []
    if digest != authority.artifact_sha256:
        problems.append("freeze-authority-digest-mismatch")
    try:
        payload = json.loads(target.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        problems.append("freeze-authority-artifact-unreadable")
    else:
        if not isinstance(payload, dict) or payload.get("schema") != authority.artifact_schema:
            problems.append("freeze-authority-schema-mismatch")
    return tuple(problems)


def verify_candidate_registry(
    grammar: TopologyGrammar,
    repository_root: str | Path,
) -> tuple[PerspectiveCandidateRegistry | None, tuple[str, ...]]:
    """Load and join the exact permissive-integration registry bound by G0."""

    root = Path(repository_root).resolve()
    target = (root / grammar.candidate_registry_path).resolve()
    if not target.is_relative_to(root):
        return None, ("candidate-registry-path-escapes-root",)
    if not target.is_file():
        return None, ("candidate-registry-artifact-missing",)
    try:
        registry = load_perspective_candidate_registry(target)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ("candidate-registry-artifact-invalid",)
    if registry.sha256 != grammar.candidate_registry_sha256:
        return registry, ("candidate-registry-authority-mismatch",)
    return registry, ()
