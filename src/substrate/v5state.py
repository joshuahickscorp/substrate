"""Permanent, event-sourced cognitive state and model fabric for Substrate v5.

The cognitive entity owns identity, time, world state, memory, goals, body state,
and model contracts.  Models and sensors are replaceable organs: neither owns the
entity.  Every mutation is an immutable event reduced by deterministic code, and
every checkpoint is verified both by replay and by cryptographic self-seals.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from substrate import v5io as io

ACTIVATION = False
CURRENT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
CHECKPOINT_SCHEMA = "substrate-v5-permanent-entity-checkpoint/v1"
EVENT_SCHEMA = "substrate-v5-cognitive-event/v1"
MAX_ACTIVE_EVENTS = 128
MAX_SENSOR_OBSERVATIONS = 64
MAX_QUEUE_DEPTH = 1_024
MAX_ARCHIVAL_INDEX = 128

SERVICE_MODES = frozenset(
    {
        "awake_active",
        "awake_idle",
        "observing",
        "reasoning",
        "acting_in_sandbox",
        "consolidating",
        "learning",
        "sleeping",
        "paused",
        "recovering",
    }
)

MODEL_ROLES = frozenset(
    {
        "primary_performer",
        "independent_performer",
        "draft_generator",
        "verifier",
        "critic",
        "simulator",
        "planner",
        "teacher",
        "student",
        "compressor",
        "retriever",
        "router",
        "specialist",
        "fallback",
        "monitor",
        "arbiter",
        "translator",
        "representation_aligner",
    }
)

RELATIONSHIP_KINDS = frozenset(
    {
        "drafts_for",
        "verifies",
        "critiques",
        "simulates_for",
        "teaches",
        "monitors",
        "replaces",
        "translates_for",
        "aligns_for",
        "fallback_for",
        "routes_to",
    }
)

MEMORY_TIERS = frozenset({"episodic", "semantic", "procedural"})
WORLD_COLLECTIONS = frozenset(
    {
        "tracked_objects",
        "tracked_agents",
        "tracked_places",
        "event_hypotheses",
        "spatial_world",
        "structural_world_models",
    }
)
QUEUE_NAMES = frozenset(
    {
        "background_cognitive_jobs",
        "consolidation_queue",
        "learning_queue",
    }
)

EVENT_KINDS = frozenset(
    {
        "belief_upserted",
        "body_replaced",
        "competence_updated",
        "consolidated",
        "context_updated",
        "entity_created",
        "goal_resolved",
        "goal_upserted",
        "hypothesis_resolved",
        "hypothesis_upserted",
        "idle_gap_recorded",
        "knowledge_upserted",
        "logs_rotated",
        "memory_recorded",
        "mode_changed",
        "model_registered",
        "model_relationship_set",
        "model_replaced",
        "queue_dequeued",
        "queue_enqueued",
        "resource_updated",
        "schema_migrated",
        "sensor_attached",
        "sensor_detached",
        "sensor_interrupted",
        "sensory_observed",
        "service_paused",
        "service_resumed",
        "service_started",
        "service_stopped",
        "task_resolved",
        "task_upserted",
        "tool_updated",
        "world_updated",
    }
)


class Refused(RuntimeError):
    """A state transition failed closed."""


def _json_copy(value: Any) -> Any:
    try:
        # json.loads already returns a detached tree. A second deepcopy added
        # no isolation while doubling the traversal cost of every event
        # payload and semantic-state snapshot.
        return io._normal_json(value)  # noqa: SLF001
    except io.Refused as error:
        raise Refused(str(error)) from error


def _identifier(value: str, label: str = "identity") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise Refused(f"{label} must be a nonempty string of at most 256 characters")
    return value


def _assert_activation(value: Any) -> None:
    try:
        io.assert_activation_false(value)
    except io.Refused as error:
        raise Refused(str(error)) from error


def _assert_normalized_activation(value: Any) -> None:
    """Check an already JSON-normalized tree without normalizing it again."""

    try:
        io._assert_normalized_activation_false(value)  # noqa: SLF001
    except io.Refused as error:
        raise Refused(str(error)) from error


@dataclass(frozen=True)
class CognitiveEvent:
    """One immutable event in the entity's hash chain."""

    sequence: int
    event_time: int
    kind: str
    payload: dict[str, Any]
    previous_sha256: str | None
    source_timestamp: int | float | str | None = None
    temporal_uncertainty: float = 0.0
    schema: str = EVENT_SCHEMA
    event_sha256: str = ""

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        event_time: int,
        kind: str,
        payload: Mapping[str, Any],
        previous_sha256: str | None,
        source_timestamp: int | float | str | None = None,
        temporal_uncertainty: float = 0.0,
    ) -> CognitiveEvent:
        _identifier(kind, "event kind")
        if kind not in EVENT_KINDS:
            raise Refused(f"unsupported cognitive event kind {kind!r}")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(event_time, int)
            or isinstance(event_time, bool)
            or sequence < 1
            or event_time < 1
        ):
            raise Refused("event sequence and internal time must be positive")
        try:
            uncertainty = float(temporal_uncertainty)
        except (TypeError, ValueError) as error:
            raise Refused("temporal uncertainty must be numeric") from error
        if uncertainty < 0:
            raise Refused("temporal uncertainty cannot be negative")
        normalized = _json_copy(dict(payload))
        _assert_normalized_activation(normalized)
        body = {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "event_time": event_time,
            "source_timestamp": source_timestamp,
            "temporal_uncertainty": uncertainty,
            "kind": kind,
            "payload": normalized,
            "previous_sha256": previous_sha256,
        }
        return cls(
            sequence=sequence,
            event_time=event_time,
            kind=kind,
            payload=normalized,
            previous_sha256=previous_sha256,
            source_timestamp=source_timestamp,
            temporal_uncertainty=uncertainty,
            event_sha256=io.sha_obj(body),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sequence": self.sequence,
            "event_time": self.event_time,
            "source_timestamp": self.source_timestamp,
            "temporal_uncertainty": self.temporal_uncertainty,
            "kind": self.kind,
            "payload": _json_copy(self.payload),
            "previous_sha256": self.previous_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "event_sha256": self.event_sha256, "activation": False}

    def validate(self) -> None:
        if self.schema != EVENT_SCHEMA:
            raise Refused("unsupported cognitive event schema")
        if self.event_sha256 != io.sha_obj(self.body()):
            raise Refused(f"corrupt cognitive event at sequence {self.sequence}")
        _assert_activation(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CognitiveEvent:
        required = {
            "schema",
            "sequence",
            "event_time",
            "source_timestamp",
            "temporal_uncertainty",
            "kind",
            "payload",
            "previous_sha256",
            "event_sha256",
            "activation",
        }
        if set(value) != required:
            raise Refused("cognitive event fields are incomplete or unknown")
        if not isinstance(value["payload"], Mapping):
            raise Refused("cognitive event payload must be an object")
        if (
            not isinstance(value["sequence"], int)
            or isinstance(value["sequence"], bool)
            or not isinstance(value["event_time"], int)
            or isinstance(value["event_time"], bool)
        ):
            raise Refused("cognitive event sequence and time must be integers")
        try:
            event = cls(
                schema=str(value["schema"]),
                sequence=value["sequence"],
                event_time=value["event_time"],
                source_timestamp=value["source_timestamp"],
                temporal_uncertainty=float(value["temporal_uncertainty"]),
                kind=str(value["kind"]),
                payload=_json_copy(value["payload"]),
                previous_sha256=value["previous_sha256"],
                event_sha256=str(value["event_sha256"]),
            )
        except (TypeError, ValueError) as error:
            raise Refused("cognitive event fields have invalid types") from error
        if value["activation"] is not False:
            raise Refused("event activation must remain false")
        event.validate()
        return event


@dataclass(frozen=True)
class ModelContract:
    """Serializable contract for one independently callable model organ."""

    identity: str
    checkpoint_identity: str
    version: str = "1"
    license: str = "declared-local"
    runtime: str = "deterministic-python"
    hardware_requirements: tuple[str, ...] = ("cpu",)
    modalities_accepted: tuple[str, ...] = ("structured",)
    modalities_produced: tuple[str, ...] = ("structured",)
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    hidden_state_policy: str = "none"
    cost: float = 0.0
    latency: float = 0.0
    memory: float = 0.0
    confidence_semantics: str = "none"
    training_provenance: tuple[str, ...] = ()
    known_limitations: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ("independent_performer",)
    statefulness: str = "stateless"
    checkpoint_support: bool = False
    batching_support: bool = False
    streaming_support: bool = False

    def validate(self) -> None:
        _identifier(self.identity, "model identity")
        _identifier(self.checkpoint_identity, "checkpoint identity")
        if not self.version or not self.license or not self.runtime:
            raise Refused("model version, license, and runtime are required")
        if self.cost < 0 or self.latency < 0 or self.memory < 0:
            raise Refused("model cost, latency, and memory must be nonnegative")
        roles = set(self.allowed_roles)
        if not roles or not roles <= MODEL_ROLES:
            raise Refused(f"unsupported model roles: {sorted(roles - MODEL_ROLES)}")
        _assert_activation(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _json_copy(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelContract:
        normalized = dict(value)
        for key in (
            "hardware_requirements",
            "modalities_accepted",
            "modalities_produced",
            "training_provenance",
            "known_limitations",
            "allowed_roles",
        ):
            normalized[key] = tuple(normalized.get(key, ()))
        try:
            contract = cls(**normalized)
        except TypeError as error:
            raise Refused("model contract fields are incomplete or unknown") from error
        contract.validate()
        return contract


@dataclass(frozen=True)
class ModelRelationship:
    source: str
    target: str
    kind: str
    measured: bool = False
    evidence: tuple[str, ...] = ()

    def validate(self, model_ids: set[str]) -> None:
        if self.source not in model_ids or self.target not in model_ids:
            raise Refused("model relationship endpoints must be registered")
        if self.source == self.target:
            raise Refused("a model relationship requires two independent endpoints")
        if self.kind not in RELATIONSHIP_KINDS:
            raise Refused(f"unsupported model relationship {self.kind!r}")

    @property
    def identity(self) -> str:
        return io.sha_obj(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(asdict(self))


@dataclass
class DeterministicModel:
    """A model contract plus a deterministic local callable."""

    contract: ModelContract
    operation: Callable[[dict[str, Any]], Any]

    def __call__(self, request: Mapping[str, Any]) -> Any:
        self.contract.validate()
        normalized = _json_copy(dict(request))
        _assert_normalized_activation(normalized)
        result = _json_copy(self.operation(normalized))
        _assert_normalized_activation(result)
        return result


class ModelRegistry:
    """Deterministic model contracts, relationship graph, and local runtimes."""

    def __init__(self) -> None:
        self.contracts: dict[str, ModelContract] = {}
        self.relationships: dict[str, ModelRelationship] = {}
        self.runners: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(
        self,
        model: ModelContract | DeterministicModel,
        runner: Callable[[dict[str, Any]], Any] | None = None,
    ) -> ModelContract:
        contract = model.contract if isinstance(model, DeterministicModel) else model
        operation = model.operation if isinstance(model, DeterministicModel) else runner
        contract.validate()
        existing = self.contracts.get(contract.identity)
        if existing is not None and existing != contract:
            raise Refused(f"model identity {contract.identity!r} already has another contract")
        self.contracts[contract.identity] = contract
        if operation is not None:
            self.runners[contract.identity] = operation
        return contract

    def relate(self, relationship: ModelRelationship) -> ModelRelationship:
        relationship.validate(set(self.contracts))
        self.relationships[relationship.identity] = relationship
        return relationship

    def invoke(self, model_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if model_id not in self.contracts:
            raise Refused(f"unknown model {model_id!r}")
        runner = self.runners.get(model_id)
        if runner is None:
            raise Refused(f"model {model_id!r} has no available runtime")
        normalized_request = _json_copy(dict(request))
        _assert_activation(normalized_request)
        output = _json_copy(runner(copy.deepcopy(normalized_request)))
        _assert_activation(output)
        receipt = {
            "model_identity": model_id,
            "checkpoint_identity": self.contracts[model_id].checkpoint_identity,
            "input_sha256": io.sha_obj(normalized_request),
            "output_sha256": io.sha_obj(output),
            "independent_call": True,
            "activation": False,
        }
        return {"output": output, "receipt": receipt}

    def call(self, model_id: str, request: Mapping[str, Any]) -> Any:
        return self.invoke(model_id, request)["output"]

    def snapshot(self) -> dict[str, Any]:
        return {
            "contracts": {
                identity: contract.to_dict()
                for identity, contract in sorted(self.contracts.items())
            },
            "relationships": {
                identity: relationship.to_dict()
                for identity, relationship in sorted(self.relationships.items())
            },
            "runtime_available": {
                identity: identity in self.runners for identity in sorted(self.contracts)
            },
            "activation": False,
        }


def _blank_state(entity_id: str, schema_version: int) -> dict[str, Any]:
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise Refused(f"unsupported permanent-state schema {schema_version}")
    state = {
        "schema_version": schema_version,
        "identity": {
            "entity_id": entity_id,
            "kind": "permanent_cognitive_entity",
        },
        "continuous_time": {
            "event_time": 0,
            "last_source_timestamp": None,
            "temporal_uncertainty": 0.0,
            "gaps": [],
        },
        "active_context": {},
        "latent_context": {},
        "sensory_buffers": {},
        "sensors": {},
        "tracked_objects": {},
        "tracked_agents": {},
        "tracked_places": {},
        "events": [],
        "event_hypotheses": {},
        "spatial_world": {},
        "body_state": {
            "identity": "body:none",
            "generation": 0,
            "sensors": [],
            "actuators": [],
            "coordinate_frames": [],
            "capabilities": [],
        },
        "tool_state": {},
        "active_goals": {},
        "unfinished_tasks": {},
        "unresolved_hypotheses": {},
        "beliefs": {},
        "knowledge": {},
        "semantic_memory": {},
        "procedural_memory": {},
        "episodic_memory": {},
        "structural_world_models": {},
        "self_model_competence": {},
        "model_registry": {},
        "model_availability": {},
        "model_relationships": {},
        "resource_state": {},
        "background_cognitive_jobs": [],
        "consolidation_queue": [],
        "learning_queue": [],
        "mode": "paused",
        "migration_history": [],
        "event_archive_count": 0,
        "archival_index_count": 0,
        "activation": False,
    }
    if schema_version >= 2:
        state["archival_tiers"] = {"episodic": []}
    return state


def _event_summary(event: CognitiveEvent) -> dict[str, Any]:
    return {
        "event_sha256": event.event_sha256,
        "sequence": event.sequence,
        "event_time": event.event_time,
        "kind": event.kind,
    }


def _reduce(state: dict[str, Any], event: CognitiveEvent) -> None:
    payload = event.payload
    kind = event.kind
    if kind not in EVENT_KINDS:
        raise Refused(f"unsupported cognitive event kind {kind!r}")
    if kind == "entity_created":
        if event.sequence != 1 or payload.get("entity_id") != state["identity"]["entity_id"]:
            raise Refused("the first event must create the checkpointed entity")
        if int(payload.get("schema_version", -1)) != state["schema_version"]:
            raise Refused("entity creation schema does not match its state")
    elif kind == "mode_changed":
        mode = payload.get("mode")
        if mode not in SERVICE_MODES:
            raise Refused(f"unsupported cognitive mode {mode!r}")
        state["mode"] = mode
    elif kind == "context_updated":
        layer = payload.get("layer")
        if layer not in {"active_context", "latent_context"}:
            raise Refused("context layer must be active_context or latent_context")
        state[layer] = _json_copy(payload.get("value", {}))
    elif kind == "goal_upserted":
        goal = _json_copy(payload["goal"])
        goal_id = _identifier(str(goal["identity"]), "goal identity")
        state["active_goals"][goal_id] = goal
    elif kind == "goal_resolved":
        state["active_goals"].pop(str(payload["identity"]), None)
    elif kind == "task_upserted":
        task = _json_copy(payload["task"])
        state["unfinished_tasks"][_identifier(str(task["identity"]), "task identity")] = task
    elif kind == "task_resolved":
        state["unfinished_tasks"].pop(str(payload["identity"]), None)
    elif kind == "hypothesis_upserted":
        hypothesis = _json_copy(payload["hypothesis"])
        identity = _identifier(str(hypothesis["identity"]), "hypothesis identity")
        state["unresolved_hypotheses"][identity] = hypothesis
    elif kind == "hypothesis_resolved":
        state["unresolved_hypotheses"].pop(str(payload["identity"]), None)
    elif kind == "memory_recorded":
        tier = str(payload["tier"])
        if tier not in MEMORY_TIERS:
            raise Refused(f"unsupported memory tier {tier!r}")
        record = _json_copy(payload["record"])
        identity = _identifier(str(record["identity"]), "memory identity")
        state[f"{tier}_memory"][identity] = record
    elif kind == "world_updated":
        collection = str(payload["collection"])
        if collection not in WORLD_COLLECTIONS:
            raise Refused(f"unsupported world-state collection {collection!r}")
        identity = _identifier(str(payload["identity"]), "world-state identity")
        if payload.get("remove") is True:
            state[collection].pop(identity, None)
        else:
            state[collection][identity] = _json_copy(payload["value"])
    elif kind == "belief_upserted":
        record = _json_copy(payload["record"])
        state["beliefs"][_identifier(str(record["identity"]), "belief identity")] = record
    elif kind == "knowledge_upserted":
        record = _json_copy(payload["record"])
        provenance_value = record.get("provenance")
        verifiers_value = record.get("verification")
        evidence_value = record.get("verification_evidence")
        if not all(
            isinstance(value, (list, tuple))
            for value in (provenance_value, verifiers_value, evidence_value)
        ):
            raise Refused("knowledge authority and evidence fields must be identity lists")
        provenance = tuple(provenance_value)
        verifiers = tuple(verifiers_value)
        evidence = tuple(evidence_value)
        if not provenance or not verifiers or not evidence:
            raise Refused(
                "knowledge requires registered provenance, independent verifiers, "
                "and verification evidence"
            )
        provenance_ids = {_identifier(value, "knowledge provenance identity") for value in provenance}
        verifier_ids = {_identifier(value, "knowledge verifier identity") for value in verifiers}
        if provenance_ids & verifier_ids:
            raise Refused("knowledge provenance and verifier identities must be distinct")
        registered_sources = set(state["sensors"]) | set(state["model_registry"])
        unknown_sources = provenance_ids - registered_sources
        if unknown_sources:
            raise Refused(f"knowledge provenance identities are not registered: {sorted(unknown_sources)}")
        unknown_verifiers = verifier_ids - set(state["model_registry"])
        if unknown_verifiers:
            raise Refused(f"knowledge verifier identities are not registered: {sorted(unknown_verifiers)}")
        invalid_verifiers = {
            identity
            for identity in verifier_ids
            if "verifier" not in state["model_registry"][identity]["allowed_roles"]
        }
        if invalid_verifiers:
            raise Refused(
                f"knowledge verifier identities lack the verifier role: {sorted(invalid_verifiers)}"
            )
        for value in evidence:
            _identifier(value, "knowledge verification evidence")
        state["knowledge"][_identifier(str(record["identity"]), "knowledge identity")] = record
    elif kind == "body_replaced":
        body = _json_copy(payload["body"])
        _identifier(str(body["identity"]), "body identity")
        body["generation"] = int(state["body_state"].get("generation", 0)) + 1
        state["body_state"] = body
    elif kind == "tool_updated":
        identity = _identifier(str(payload["identity"]), "tool identity")
        state["tool_state"][identity] = _json_copy(payload["value"])
    elif kind == "sensor_attached":
        identity = _identifier(str(payload["identity"]), "sensor identity")
        state["sensors"][identity] = {
            **_json_copy(payload["contract"]),
            "status": "attached",
            "last_event_time": event.event_time,
        }
        state["sensory_buffers"].setdefault(identity, [])
    elif kind == "sensor_interrupted":
        identity = str(payload["identity"])
        if identity not in state["sensors"]:
            raise Refused(f"unknown sensor {identity!r}")
        state["sensors"][identity]["status"] = "interrupted"
        state["sensors"][identity]["last_event_time"] = event.event_time
    elif kind == "sensor_detached":
        identity = str(payload["identity"])
        if identity not in state["sensors"]:
            raise Refused(f"unknown sensor {identity!r}")
        state["sensors"][identity]["status"] = "detached"
        state["sensors"][identity]["last_event_time"] = event.event_time
    elif kind == "sensory_observed":
        identity = str(payload["sensor_identity"])
        if identity not in state["sensors"]:
            raise Refused(f"unknown sensor {identity!r}")
        if state["sensors"][identity]["status"] == "detached":
            raise Refused("a detached sensor cannot emit observations")
        observation = _json_copy(payload["observation"])
        state["sensory_buffers"][identity].append(observation)
        state["sensory_buffers"][identity] = state["sensory_buffers"][identity][
            -MAX_SENSOR_OBSERVATIONS:
        ]
        state["sensors"][identity]["status"] = "attached"
        state["sensors"][identity]["last_event_time"] = event.event_time
    elif kind == "model_registered":
        contract = ModelContract.from_dict(payload["contract"])
        existing = state["model_registry"].get(contract.identity)
        if existing is not None and existing != contract.to_dict():
            raise Refused("a registered model identity cannot be silently redefined")
        state["model_registry"][contract.identity] = contract.to_dict()
        state["model_availability"][contract.identity] = {
            "available": bool(payload.get("available", True)),
            "checkpoint_identity": contract.checkpoint_identity,
            "replaced_by": None,
        }
    elif kind == "model_relationship_set":
        relationship = ModelRelationship(
            source=str(payload["source"]),
            target=str(payload["target"]),
            kind=str(payload["kind"]),
            measured=bool(payload.get("measured", False)),
            evidence=tuple(payload.get("evidence", ())),
        )
        relationship.validate(set(state["model_registry"]))
        state["model_relationships"][relationship.identity] = relationship.to_dict()
    elif kind == "model_replaced":
        old_identity = str(payload["old_identity"])
        if old_identity not in state["model_registry"]:
            raise Refused(f"cannot replace unknown model {old_identity!r}")
        contract = ModelContract.from_dict(payload["new_contract"])
        existing = state["model_registry"].get(contract.identity)
        if existing is not None and existing != contract.to_dict():
            raise Refused("replacement model identity has a conflicting contract")
        state["model_registry"][contract.identity] = contract.to_dict()
        state["model_availability"][old_identity]["available"] = False
        state["model_availability"][old_identity]["replaced_by"] = contract.identity
        state["model_availability"][contract.identity] = {
            "available": True,
            "checkpoint_identity": contract.checkpoint_identity,
            "replaced_by": None,
        }
        relationship = ModelRelationship(
            source=contract.identity,
            target=old_identity,
            kind="replaces",
            measured=bool(payload.get("measured", False)),
            evidence=tuple(payload.get("evidence", ())),
        )
        relationship.validate(set(state["model_registry"]))
        state["model_relationships"][relationship.identity] = relationship.to_dict()
    elif kind == "queue_enqueued":
        queue = str(payload["queue"])
        if queue not in QUEUE_NAMES:
            raise Refused(f"unsupported cognitive queue {queue!r}")
        if len(state[queue]) >= MAX_QUEUE_DEPTH:
            raise Refused(f"bounded queue {queue!r} is full")
        state[queue].append(_json_copy(payload["item"]))
    elif kind == "queue_dequeued":
        queue = str(payload["queue"])
        if queue not in QUEUE_NAMES:
            raise Refused(f"unsupported cognitive queue {queue!r}")
        identity = str(payload["identity"])
        state[queue] = [
            item for item in state[queue] if str(item.get("identity")) != identity
        ]
    elif kind == "resource_updated":
        state["resource_state"] = _json_copy(payload["resource_state"])
    elif kind == "competence_updated":
        identity = _identifier(str(payload["model_identity"]), "model identity")
        state["self_model_competence"][identity] = _json_copy(payload["estimate"])
    elif kind == "consolidated":
        for identity in payload["source_memory_ids"]:
            state["episodic_memory"].pop(str(identity), None)
        record = _json_copy(payload["semantic_record"])
        state["semantic_memory"][str(record["identity"])] = record
        if state["schema_version"] >= 2:
            archive = state["archival_tiers"]["episodic"]
            archive.append(_json_copy(payload["archive_entry"]))
            state["archival_index_count"] += 1
            if len(archive) > MAX_ARCHIVAL_INDEX:
                del archive[: len(archive) - MAX_ARCHIVAL_INDEX]
    elif kind == "schema_migrated":
        source = int(payload["source_schema"])
        destination = int(payload["destination_schema"])
        if source != state["schema_version"]:
            raise Refused("migration source does not match current state schema")
        if (source, destination) == (1, 2):
            state["archival_tiers"] = {"episodic": []}
        elif (source, destination) == (2, 1):
            if state.get("archival_tiers", {}).get("episodic"):
                raise Refused("schema 2 archival entries must be consolidated before downgrade")
            state.pop("archival_tiers", None)
        else:
            raise Refused(f"unsupported schema migration {source} to {destination}")
        state["schema_version"] = destination
        state["migration_history"].append(
            {
                "source_schema": source,
                "destination_schema": destination,
                "event_sha256": event.event_sha256,
            }
        )
    elif kind in {
        "service_started",
        "service_stopped",
        "service_paused",
        "service_resumed",
        "logs_rotated",
        "idle_gap_recorded",
    }:
        if "mode" in payload:
            mode = payload["mode"]
            if mode not in SERVICE_MODES:
                raise Refused(f"unsupported cognitive mode {mode!r}")
            state["mode"] = mode

    old_time = int(state["continuous_time"]["event_time"])
    if event.event_time <= old_time:
        raise Refused("internal event time must be strictly monotonic")
    if event.event_time > old_time + 1:
        state["continuous_time"]["gaps"].append(
            {
                "after": old_time,
                "before": event.event_time,
                "missing_intervals": event.event_time - old_time - 1,
            }
        )
    state["continuous_time"]["event_time"] = event.event_time
    state["continuous_time"]["last_source_timestamp"] = event.source_timestamp
    state["continuous_time"]["temporal_uncertainty"] = event.temporal_uncertainty
    state["events"].append(_event_summary(event))
    if len(state["events"]) > MAX_ACTIVE_EVENTS:
        overflow = len(state["events"]) - MAX_ACTIVE_EVENTS
        del state["events"][:overflow]
        state["event_archive_count"] += overflow
    state["activation"] = False


@dataclass(frozen=True)
class MigrationReceipt:
    source_schema: int
    destination_schema: int
    entity_id: str
    pre_checkpoint: dict[str, Any]
    pre_checkpoint_sha256: str
    post_checkpoint_sha256: str
    semantic_identity_preserved: bool
    activation: bool = False

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["receipt_sha256"] = io.sha_obj(body)
        return body


class PermanentEntity:
    """One continuing entity reconstructed solely from its immutable events."""

    def __init__(
        self,
        entity_id: str = "substrate-v5",
        *,
        schema_version: int = CURRENT_SCHEMA_VERSION,
        storage_root: Path | None = None,
    ) -> None:
        _identifier(entity_id, "entity identity")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise Refused(f"unsupported permanent-state schema {schema_version}")
        self._lock = threading.RLock()
        self._storage_root = storage_root
        self._events: list[CognitiveEvent] = []
        self._state = _blank_state(entity_id, schema_version)
        self._model_runners: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.append_event(
            "entity_created",
            {"entity_id": entity_id, "schema_version": schema_version},
        )

    @property
    def entity_id(self) -> str:
        return str(self._state["identity"]["entity_id"])

    @property
    def activation(self) -> bool:
        return False

    @property
    def event_time(self) -> int:
        return int(self._state["continuous_time"]["event_time"])

    @property
    def mode(self) -> str:
        return str(self._state["mode"])

    @property
    def events(self) -> tuple[CognitiveEvent, ...]:
        return tuple(self._events)

    @property
    def state(self) -> dict[str, Any]:
        return self.semantic_state()

    def semantic_state(self) -> dict[str, Any]:
        return _json_copy(self._state)

    def state_identity(self) -> str:
        return io.sha_obj(self._state)

    def _persist_event(self, event: CognitiveEvent) -> None:
        if self._storage_root is None:
            return
        relative = (
            Path("entities")
            / self.entity_id
            / "events"
            / f"{event.sequence:020d}-{event.event_sha256}.json"
        )
        io.publish_json(self._storage_root / relative, event.to_dict())

    def append_event(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        event_time: int | None = None,
        source_timestamp: int | float | str | None = None,
        temporal_uncertainty: float = 0.0,
    ) -> dict[str, Any]:
        """Append one event after validating persistence and deterministic reduction."""

        with self._lock:
            next_time = self.event_time + 1 if event_time is None else int(event_time)
            if next_time <= self.event_time:
                raise Refused("internal event time must increase on every event")
            event = CognitiveEvent.create(
                sequence=len(self._events) + 1,
                event_time=next_time,
                kind=kind,
                payload=payload,
                previous_sha256=(
                    self._events[-1].event_sha256 if self._events else None
                ),
                source_timestamp=source_timestamp,
                temporal_uncertainty=temporal_uncertainty,
            )
            # _state is an internally-owned canonical JSON tree: every event
            # payload is normalized before reduction and every committed
            # transition is activation-checked. Copy directly for the
            # speculative reduction; semantic_state() remains the public
            # normalized snapshot boundary.
            candidate = copy.deepcopy(self._state)
            try:
                _reduce(candidate, event)
            except Refused:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise Refused(f"invalid payload for event {kind!r}") from error
            _assert_normalized_activation(candidate)
            self._persist_event(event)
            self._events.append(event)
            self._state = candidate
            return event.to_dict()

    def advance_time(self, event_time: int, *, reason: str = "idle") -> dict[str, Any]:
        return self.append_event(
            "idle_gap_recorded",
            {"reason": reason},
            event_time=event_time,
        )

    def set_mode(self, mode: str) -> dict[str, Any]:
        return self.append_event("mode_changed", {"mode": mode})

    def update_context(self, layer: str, value: Mapping[str, Any]) -> dict[str, Any]:
        return self.append_event(
            "context_updated",
            {"layer": layer, "value": dict(value)},
        )

    def upsert_goal(
        self,
        identity: str,
        description: str,
        *,
        priority: float = 0.5,
        status: str = "active",
        provenance: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not 0 <= priority <= 1:
            raise Refused("goal priority must lie in [0, 1]")
        goal = {
            "identity": _identifier(identity, "goal identity"),
            "description": description,
            "priority": float(priority),
            "status": status,
            "provenance": list(provenance),
            "updated_at_event_time": self.event_time + 1,
        }
        return self.append_event("goal_upserted", {"goal": goal})

    add_goal = upsert_goal

    def resolve_goal(self, identity: str) -> dict[str, Any]:
        return self.append_event("goal_resolved", {"identity": identity})

    def upsert_task(
        self,
        identity: str,
        description: str,
        *,
        goal_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return self.append_event(
            "task_upserted",
            {
                "task": {
                    "identity": identity,
                    "description": description,
                    "goal_ids": list(goal_ids),
                    "updated_at_event_time": self.event_time + 1,
                }
            },
        )

    def resolve_task(self, identity: str) -> dict[str, Any]:
        return self.append_event("task_resolved", {"identity": identity})

    def upsert_hypothesis(
        self,
        identity: str,
        claim: Any,
        *,
        confidence: float,
        evidence: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not 0 <= confidence <= 1:
            raise Refused("hypothesis confidence must lie in [0, 1]")
        return self.append_event(
            "hypothesis_upserted",
            {
                "hypothesis": {
                    "identity": identity,
                    "claim": claim,
                    "confidence": float(confidence),
                    "evidence": list(evidence),
                    "updated_at_event_time": self.event_time + 1,
                }
            },
        )

    def record_memory(
        self,
        tier: str,
        identity: str,
        content: Any,
        *,
        provenance: tuple[str, ...],
        verification: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if tier not in MEMORY_TIERS:
            raise Refused(f"unsupported memory tier {tier!r}")
        if not provenance:
            raise Refused("memory requires provenance")
        return self.append_event(
            "memory_recorded",
            {
                "tier": tier,
                "record": {
                    "identity": identity,
                    "content": content,
                    "provenance": list(provenance),
                    "verification": list(verification),
                    "event_time": self.event_time + 1,
                },
            },
        )

    def update_world(
        self,
        collection: str,
        identity: str,
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collection": collection,
            "identity": identity,
        }
        if value is None:
            payload["remove"] = True
        else:
            payload["value"] = dict(value)
        return self.append_event("world_updated", payload)

    def upsert_belief(
        self,
        identity: str,
        content: Any,
        *,
        confidence: float,
        supporting_evidence: tuple[str, ...] = (),
        contradicting_evidence: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not 0 <= confidence <= 1:
            raise Refused("belief confidence must lie in [0, 1]")
        return self.append_event(
            "belief_upserted",
            {
                "record": {
                    "identity": identity,
                    "content": content,
                    "confidence": float(confidence),
                    "supporting_evidence": list(supporting_evidence),
                    "contradicting_evidence": list(contradicting_evidence),
                }
            },
        )

    def admit_knowledge(
        self,
        identity: str,
        content: Any,
        *,
        provenance: tuple[str, ...],
        verification: tuple[str, ...],
        verification_evidence: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not provenance or not verification or not verification_evidence:
            raise Refused(
                "knowledge requires registered provenance, independent verifiers, "
                "and verification evidence"
            )
        return self.append_event(
            "knowledge_upserted",
            {
                "record": {
                    "identity": identity,
                    "content": content,
                    "provenance": list(provenance),
                    "verification": list(verification),
                    "verification_evidence": list(verification_evidence),
                }
            },
        )

    def replace_body(self, body: Mapping[str, Any]) -> dict[str, Any]:
        before = self._continuity_fields()
        event = self.append_event("body_replaced", {"body": dict(body)})
        after = self._continuity_fields()
        return {
            "event": event,
            "entity_identity_preserved": before["identity"] == after["identity"],
            "goals_preserved": before["goals"] == after["goals"],
            "memory_preserved": before["memory"] == after["memory"],
            "world_preserved": before["world"] == after["world"],
            "activation": False,
        }

    def update_tool(
        self,
        identity: str,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.append_event(
            "tool_updated",
            {"identity": identity, "value": dict(value)},
        )

    def attach_sensor(
        self,
        identity: str,
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.append_event(
            "sensor_attached",
            {"identity": identity, "contract": dict(contract)},
        )

    def interrupt_sensor(self, identity: str) -> dict[str, Any]:
        before = self._continuity_fields()
        event = self.append_event("sensor_interrupted", {"identity": identity})
        after = self._continuity_fields()
        return {
            "event": event,
            "entity_identity_preserved": before["identity"] == after["identity"],
            "goals_preserved": before["goals"] == after["goals"],
            "memory_preserved": before["memory"] == after["memory"],
            "world_preserved": before["world"] == after["world"],
            "activation": False,
        }

    def detach_sensor(self, identity: str) -> dict[str, Any]:
        return self.append_event("sensor_detached", {"identity": identity})

    def observe_sensor(
        self,
        sensor_identity: str,
        observation: Mapping[str, Any],
        *,
        source_timestamp: int | float | str | None,
        temporal_uncertainty: float = 0.0,
    ) -> dict[str, Any]:
        return self.append_event(
            "sensory_observed",
            {
                "sensor_identity": sensor_identity,
                "observation": dict(observation),
            },
            source_timestamp=source_timestamp,
            temporal_uncertainty=temporal_uncertainty,
        )

    def register_model(
        self,
        model: ModelContract | DeterministicModel,
        runner: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        contract = model.contract if isinstance(model, DeterministicModel) else model
        operation = model.operation if isinstance(model, DeterministicModel) else runner
        contract.validate()
        event = self.append_event(
            "model_registered",
            {"contract": contract.to_dict(), "available": True},
        )
        if operation is not None:
            self._model_runners[contract.identity] = operation
        return event

    def relate_models(
        self,
        source: str,
        target: str,
        kind: str,
        *,
        measured: bool = False,
        evidence: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return self.append_event(
            "model_relationship_set",
            {
                "source": source,
                "target": target,
                "kind": kind,
                "measured": measured,
                "evidence": list(evidence),
            },
        )

    def call_model(self, model_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        availability = self._state["model_availability"].get(model_id)
        if not availability or not availability["available"]:
            raise Refused(f"model {model_id!r} is unavailable")
        registry = self.model_registry()
        return registry.invoke(model_id, request)

    def model_registry(self) -> ModelRegistry:
        registry = ModelRegistry()
        for value in self._state["model_registry"].values():
            contract = ModelContract.from_dict(value)
            registry.register(contract, self._model_runners.get(contract.identity))
        for value in self._state["model_relationships"].values():
            registry.relate(
                ModelRelationship(
                    source=value["source"],
                    target=value["target"],
                    kind=value["kind"],
                    measured=bool(value["measured"]),
                    evidence=tuple(value["evidence"]),
                )
            )
        return registry

    def replace_model(
        self,
        old_identity: str,
        new_model: ModelContract | DeterministicModel,
        runner: Callable[[dict[str, Any]], Any] | None = None,
        *,
        measured: bool = False,
        evidence: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        contract = (
            new_model.contract
            if isinstance(new_model, DeterministicModel)
            else new_model
        )
        operation = (
            new_model.operation
            if isinstance(new_model, DeterministicModel)
            else runner
        )
        contract.validate()
        before = self._continuity_fields()
        event = self.append_event(
            "model_replaced",
            {
                "old_identity": old_identity,
                "new_contract": contract.to_dict(),
                "measured": measured,
                "evidence": list(evidence),
            },
        )
        if operation is not None:
            self._model_runners[contract.identity] = operation
        after = self._continuity_fields()
        return {
            "event": event,
            "old_model": old_identity,
            "new_model": contract.identity,
            "entity_identity_preserved": before["identity"] == after["identity"],
            "goals_preserved": before["goals"] == after["goals"],
            "memory_preserved": before["memory"] == after["memory"],
            "world_preserved": before["world"] == after["world"],
            "activation": False,
        }

    def _continuity_fields(self) -> dict[str, Any]:
        return {
            "identity": _json_copy(self._state["identity"]),
            "goals": _json_copy(self._state["active_goals"]),
            "memory": {
                tier: _json_copy(self._state[f"{tier}_memory"])
                for tier in sorted(MEMORY_TIERS)
            },
            "world": {
                collection: _json_copy(self._state[collection])
                for collection in sorted(WORLD_COLLECTIONS)
            },
        }

    def enqueue(self, queue: str, item: Mapping[str, Any]) -> dict[str, Any]:
        if "identity" not in item:
            raise Refused("queued cognitive work requires an identity")
        return self.append_event(
            "queue_enqueued",
            {"queue": queue, "item": dict(item)},
        )

    def dequeue(self, queue: str, identity: str) -> dict[str, Any]:
        return self.append_event(
            "queue_dequeued",
            {"queue": queue, "identity": identity},
        )

    def update_resources(self, resource_state: Mapping[str, Any]) -> dict[str, Any]:
        return self.append_event(
            "resource_updated",
            {"resource_state": dict(resource_state)},
        )

    def update_competence(
        self,
        model_identity: str,
        estimate: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.append_event(
            "competence_updated",
            {"model_identity": model_identity, "estimate": dict(estimate)},
        )

    def consolidate(
        self,
        *,
        max_active_episodic: int = 64,
        batch_size: int = 32,
    ) -> dict[str, Any] | None:
        """Bound active episodic state while retaining exact source-event links."""

        if max_active_episodic < 0 or batch_size < 1:
            raise Refused("consolidation bounds must be nonnegative and finite")
        episodic = self._state["episodic_memory"]
        excess = len(episodic) - max_active_episodic
        if excess <= 0:
            return None
        ordered = sorted(
            episodic.values(),
            key=lambda record: (int(record["event_time"]), str(record["identity"])),
        )
        source_records = ordered[: min(excess, batch_size)]
        source_ids = [str(record["identity"]) for record in source_records]
        content_digest = io.sha_obj(source_records)
        semantic_id = f"consolidation:{content_digest[:24]}"
        semantic_record = {
            "identity": semantic_id,
            "kind": "episodic_consolidation",
            "source_memory_ids": source_ids,
            "source_digest": content_digest,
            "count": len(source_records),
            "event_time_range": [
                min(int(record["event_time"]) for record in source_records),
                max(int(record["event_time"]) for record in source_records),
            ],
            "provenance": [
                source
                for record in source_records
                for source in record.get("provenance", [])
            ],
        }
        archive_entry = {
            "identity": semantic_id,
            "source_memory_ids": source_ids,
            "source_digest": content_digest,
        }
        return self.append_event(
            "consolidated",
            {
                "source_memory_ids": source_ids,
                "semantic_record": semantic_record,
                "archive_entry": archive_entry,
            },
        )

    def workspace_projection(
        self,
        *,
        goal_id: str | None = None,
        model_id: str | None = None,
        max_items: int = 32,
        max_bytes: int = 65_536,
    ) -> dict[str, Any]:
        """Build a deterministic, bounded projection instead of replaying history."""

        if max_items < 0 or max_bytes < 512:
            raise Refused("workspace projection bounds are invalid")
        projection: dict[str, Any] = {
            "identity": _json_copy(self._state["identity"]),
            "event_time": self.event_time,
            "mode": self.mode,
            "body_state": _json_copy(self._state["body_state"]),
            "active_goals": {},
            "unfinished_tasks": {},
            "unresolved_hypotheses": {},
            "tracked_objects": {},
            "recent_events": [],
            "episodic_memory": {},
            "semantic_memory": {},
            "procedural_memory": {},
            "beliefs": {},
            "knowledge": {},
            "model_contracts": {},
            "bounds": {
                "max_items": max_items,
                "max_bytes": max_bytes,
                "included_items": 0,
            },
            "activation": False,
        }
        candidates: list[tuple[str, str, Any]] = []

        def add_mapping(category: str, values: Mapping[str, Any]) -> None:
            for key, value in sorted(values.items()):
                candidates.append((category, str(key), value))

        goals = self._state["active_goals"]
        if goal_id is not None and goal_id in goals:
            candidates.append(("active_goals", goal_id, goals[goal_id]))
        else:
            add_mapping("active_goals", goals)
        add_mapping("unfinished_tasks", self._state["unfinished_tasks"])
        add_mapping("unresolved_hypotheses", self._state["unresolved_hypotheses"])
        add_mapping("tracked_objects", self._state["tracked_objects"])
        for event in reversed(self._state["events"]):
            candidates.append(("recent_events", str(event["sequence"]), event))
        for tier in ("episodic", "semantic", "procedural"):
            add_mapping(f"{tier}_memory", self._state[f"{tier}_memory"])
        add_mapping("beliefs", self._state["beliefs"])
        add_mapping("knowledge", self._state["knowledge"])
        contracts = self._state["model_registry"]
        if model_id is not None and model_id in contracts:
            candidates.append(("model_contracts", model_id, contracts[model_id]))

        accepted: list[tuple[str, str]] = []
        for category, key, value in candidates:
            if len(accepted) >= max_items:
                break
            if category == "recent_events":
                projection[category].append(_json_copy(value))
            else:
                projection[category][key] = _json_copy(value)
            projection["bounds"]["included_items"] = len(accepted) + 1
            if len(io.canonical_json(projection)) > max_bytes:
                if category == "recent_events":
                    projection[category].pop()
                else:
                    projection[category].pop(key, None)
                projection["bounds"]["included_items"] = len(accepted)
                continue
            accepted.append((category, key))
        if len(io.canonical_json(projection)) > max_bytes:
            raise Refused("projection byte bound is too small for required identity state")
        return projection

    project = workspace_projection

    def checkpoint(self) -> dict[str, Any]:
        state = self.semantic_state()
        document = {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": int(state["schema_version"]),
            "owned_identity": self.entity_id,
            "events": [event.to_dict() for event in self._events],
            "event_chain_head": (
                self._events[-1].event_sha256 if self._events else None
            ),
            "state": state,
            "state_sha256": io.sha_obj(state),
            "activation": False,
        }
        return io.sealed_document(document)

    snapshot = checkpoint

    def save_checkpoint(self) -> Path:
        if self._storage_root is None:
            raise Refused("no v5 storage root was configured for this entity")
        return io.content_addressed_json(
            self.checkpoint(),
            root=self._storage_root,
            namespace=f"entities/{self.entity_id}/checkpoints",
        )

    @classmethod
    def restore(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        runners: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
        storage_root: Path | None = None,
    ) -> PermanentEntity:
        try:
            document = io.validate_seal(dict(checkpoint))
        except io.Refused as error:
            raise Refused(f"checkpoint seal is invalid: {error}") from error
        if document.get("schema") != CHECKPOINT_SCHEMA:
            raise Refused("unsupported permanent-entity checkpoint schema")
        try:
            schema_version = int(document.get("schema_version", -1))
        except (TypeError, ValueError) as error:
            raise Refused("checkpoint schema version is invalid") from error
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise Refused(f"unsupported permanent-state schema {schema_version}")
        state = document.get("state")
        events = document.get("events")
        if not isinstance(state, dict) or not isinstance(events, list):
            raise Refused("checkpoint state and events are required")
        if state.get("activation") is not False or document.get("activation") is not False:
            raise Refused("checkpoint activation must remain false")
        if io.sha_obj(state) != document.get("state_sha256"):
            raise Refused("checkpoint state digest does not cover the supplied state")
        entity_id = str(document.get("owned_identity"))
        if state.get("identity", {}).get("entity_id") != entity_id:
            raise Refused("checkpoint owned identity disagrees with state identity")

        if not events:
            raise Refused("a permanent entity checkpoint cannot omit its creation event")
        first_raw = events[0]
        if not isinstance(first_raw, dict):
            raise Refused("checkpoint creation event is not an object")
        first_event = CognitiveEvent.from_dict(first_raw)
        if first_event.kind != "entity_created":
            raise Refused("a permanent entity must begin with entity_created")
        try:
            creation_schema = int(first_event.payload["schema_version"])
        except (KeyError, TypeError, ValueError) as error:
            raise Refused("entity creation schema is invalid") from error
        replayed = _blank_state(entity_id, creation_schema)
        parsed_events: list[CognitiveEvent] = []
        previous: str | None = None
        previous_time = 0
        for expected_sequence, raw_event in enumerate(events, start=1):
            if not isinstance(raw_event, dict):
                raise Refused("checkpoint event is not an object")
            event = CognitiveEvent.from_dict(raw_event)
            if event.sequence != expected_sequence:
                raise Refused("checkpoint event sequence is not contiguous")
            if event.previous_sha256 != previous:
                raise Refused("checkpoint event hash chain is broken")
            if event.event_time <= previous_time:
                raise Refused("checkpoint event time is not strictly monotonic")
            try:
                _reduce(replayed, event)
            except Refused:
                raise
            except (KeyError, TypeError, ValueError) as error:
                raise Refused(
                    f"invalid payload at cognitive event sequence {expected_sequence}"
                ) from error
            parsed_events.append(event)
            previous = event.event_sha256
            previous_time = event.event_time
        if previous != document.get("event_chain_head"):
            raise Refused("checkpoint event-chain head is invalid")
        if replayed != state:
            raise Refused("checkpoint state is not the exact deterministic event projection")
        if replayed["schema_version"] != schema_version:
            raise Refused("checkpoint envelope and projected schema versions disagree")

        entity = cls.__new__(cls)
        entity._lock = threading.RLock()
        entity._storage_root = storage_root
        entity._events = parsed_events
        entity._state = _json_copy(replayed)
        entity._model_runners = dict(runners or {})
        for model_id in entity._model_runners:
            if model_id not in entity._state["model_registry"]:
                raise Refused(f"runtime supplied for unknown model {model_id!r}")
        if entity.checkpoint() != document:
            raise Refused("restored entity does not reproduce the exact checkpoint")
        return entity

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        runners: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
        storage_root: Path | None = None,
    ) -> PermanentEntity:
        try:
            document = io.load_json(path)
        except io.Refused as error:
            raise Refused(str(error)) from error
        return cls.restore(document, runners=runners, storage_root=storage_root)

    def migrate_schema(self, target_version: int) -> MigrationReceipt:
        target_version = int(target_version)
        source_version = int(self._state["schema_version"])
        if target_version == source_version:
            raise Refused("source and destination schema are identical")
        if target_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise Refused(f"unsupported destination schema {target_version}")
        pre = self.checkpoint()
        identity_before = self.entity_id
        self.append_event(
            "schema_migrated",
            {
                "source_schema": source_version,
                "destination_schema": target_version,
            },
        )
        post = self.checkpoint()
        return MigrationReceipt(
            source_schema=source_version,
            destination_schema=target_version,
            entity_id=self.entity_id,
            pre_checkpoint=pre,
            pre_checkpoint_sha256=str(pre["sha256"]),
            post_checkpoint_sha256=str(post["sha256"]),
            semantic_identity_preserved=identity_before == self.entity_id,
        )

    def rollback_migration(self, receipt: MigrationReceipt) -> PermanentEntity:
        if receipt.entity_id != self.entity_id:
            raise Refused("migration receipt belongs to another entity")
        if self.checkpoint()["sha256"] != receipt.post_checkpoint_sha256:
            raise Refused("entity has changed since migration; exact rollback is unsafe")
        restored = self.restore(
            receipt.pre_checkpoint,
            runners=self._model_runners,
            storage_root=self._storage_root,
        )
        if restored.checkpoint()["sha256"] != receipt.pre_checkpoint_sha256:
            raise Refused("migration rollback did not recover the exact prior checkpoint")
        return restored


class EntityService:
    """Operational lifecycle for a permanent entity without external activation."""

    def __init__(self, entity: PermanentEntity):
        self.entity = entity
        self.running = False

    def start(self) -> dict[str, Any]:
        if self.running:
            raise Refused("entity service is already running")
        event = self.entity.append_event(
            "service_started",
            {"mode": "awake_idle"},
        )
        self.running = True
        return {"status": "running", "event": event, "activation": False}

    def health(self) -> dict[str, Any]:
        state = self.entity._state
        return {
            "status": "running" if self.running else "stopped",
            "mode": self.entity.mode,
            "state_integrity": self.entity.checkpoint()["state_sha256"]
            == self.entity.state_identity(),
            "event_chain_head": (
                self.entity.events[-1].event_sha256 if self.entity.events else None
            ),
            "sensor_status": {
                identity: sensor["status"]
                for identity, sensor in sorted(state["sensors"].items())
            },
            "model_health": {
                identity: {
                    "declared_available": availability["available"],
                    "runtime_bound": identity in self.entity._model_runners,
                }
                for identity, availability in sorted(
                    state["model_availability"].items()
                )
            },
            "queue_depth": {
                queue: len(state[queue]) for queue in sorted(QUEUE_NAMES)
            },
            "activation": False,
        }

    def pause(self) -> dict[str, Any]:
        if not self.running:
            raise Refused("a stopped entity service cannot be paused")
        event = self.entity.append_event("service_paused", {"mode": "paused"})
        self.running = False
        return {"status": "paused", "event": event, "activation": False}

    def resume(self) -> dict[str, Any]:
        if self.running:
            raise Refused("entity service is already running")
        event = self.entity.append_event(
            "service_resumed",
            {"mode": "awake_idle"},
        )
        self.running = True
        return {"status": "running", "event": event, "activation": False}

    def checkpoint(self) -> dict[str, Any] | Path:
        if self.entity._storage_root is None:
            return self.entity.checkpoint()
        return self.entity.save_checkpoint()

    def snapshot(self) -> dict[str, Any]:
        return self.entity.checkpoint()

    def restore(
        self,
        checkpoint: Mapping[str, Any],
        *,
        runners: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> dict[str, Any]:
        self.entity = PermanentEntity.restore(
            checkpoint,
            runners=runners,
            storage_root=self.entity._storage_root,
        )
        self.running = False
        return {
            "status": "restored",
            "state_sha256": self.entity.state_identity(),
            "activation": False,
        }

    def rotate_logs(self) -> dict[str, Any]:
        event = self.entity.append_event(
            "logs_rotated",
            {"durable_events_retained": len(self.entity.events)},
        )
        return {"event": event, "activation": False}

    def compact_state(
        self,
        *,
        max_active_episodic: int = 64,
        batch_size: int = 32,
    ) -> dict[str, Any] | None:
        return self.entity.consolidate(
            max_active_episodic=max_active_episodic,
            batch_size=batch_size,
        )

    def replace_model(
        self,
        old_identity: str,
        new_model: ModelContract | DeterministicModel,
        runner: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        return self.entity.replace_model(old_identity, new_model, runner)

    def attach_sensor(
        self,
        identity: str,
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.entity.attach_sensor(identity, contract)

    def detach_sensor(self, identity: str) -> dict[str, Any]:
        return self.entity.detach_sensor(identity)

    def stop(self) -> dict[str, Any]:
        if not self.running:
            raise Refused("entity service is already stopped")
        event = self.entity.append_event(
            "service_stopped",
            {"mode": "paused"},
        )
        self.running = False
        return {"status": "stopped", "event": event, "activation": False}


def migrate_checkpoint(
    checkpoint: Mapping[str, Any],
    target_version: int,
) -> tuple[dict[str, Any], MigrationReceipt]:
    entity = PermanentEntity.restore(checkpoint)
    receipt = entity.migrate_schema(target_version)
    return entity.checkpoint(), receipt


def rollback_checkpoint(receipt: MigrationReceipt) -> dict[str, Any]:
    checkpoint = _json_copy(receipt.pre_checkpoint)
    restored = PermanentEntity.restore(checkpoint)
    if restored.checkpoint()["sha256"] != receipt.pre_checkpoint_sha256:
        raise Refused("rollback receipt does not recover its declared checkpoint")
    return restored.checkpoint()


PersistentEntity = PermanentEntity
PermanentCognitiveState = PermanentEntity
