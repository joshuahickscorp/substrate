
from __future__ import annotations

import base64
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from mop.substrate.events import BranchRef, EventRef, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from .actors import (
    ActionIntent,
    Actor,
    ActorActivationContext,
    ActorActivationResult,
    ActorDescriptor,
    ActorUpdateContext,
    ActorUpdatePlan,
    DispatchEvent,
    DispatchEventHeader,
    EndogenousHypothesisProposal,
    OutboundClaim,
    ReadinessEstimate,
)
from .events import HypothesisEvent, HypothesisOrigin
from .ledger import EventLedger
from .messages import (
    ClaimFault,
    ClaimMessage,
    ClaimValidationContext,
    EpistemicStatus,
    EventClaimEvidence,
    SchemaRegistry,
    epistemic_rank,
    validate_claim,
)


class RuntimeContractError(RuntimeError):
    pass


class RuntimeCapExceeded(RuntimeContractError):
    pass


def _require_sha256_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from exc
    return value


class CandidateMode(StrEnum):
    BOUNDED_CENTRAL = "bounded-central"
    SHARDED_SUBSCRIPTION = "sharded-subscription"
    PEER_NOMINATION = "peer-nomination"


@dataclass(frozen=True, slots=True)
class RuntimeCaps:

    K: int  # candidate actors per dispatch
    C: int  # actors in a selected coalition
    B: int  # candidate coalitions in the policy beam
    H: int  # peer nomination hops
    M: int  # directed message edges in one reasoning episode
    R: int  # endogenous rounds after the initial event
    A: int = 16  # action intents in one reasoning episode
    P: int = 128  # pending branch/actor update authorities across traces
    max_action_encoded_bytes: int = 256 * 1024
    max_message_encoded_bytes: int = 256 * 1024
    max_message_header_items: int = 256
    max_endogenous_payload_bytes: int = 256 * 1024
    max_event_payload_bytes: int = 1024 * 1024
    max_header_referents: int = 64
    max_header_factors: int = 64
    max_header_shards: int = 16
    max_header_source_events: int = 64
    max_header_encoded_bytes: int = 64 * 1024
    max_episode_work: int = 10_000_000
    max_wall_time_ns: int = 5_000_000_000
    max_actor_retained_state_bytes: int = 64 * 1024 * 1024
    max_pending_update_bytes: int = 1024 * 1024
    max_consumed_authorities: int = 4096
    max_consumed_authority_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("K", "C", "B"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"RuntimeCaps.{name} must be a positive integer")
        for name in (
            "H",
            "M",
            "R",
            "A",
            "P",
            "max_action_encoded_bytes",
            "max_message_encoded_bytes",
            "max_message_header_items",
            "max_endogenous_payload_bytes",
            "max_event_payload_bytes",
            "max_header_referents",
            "max_header_factors",
            "max_header_shards",
            "max_header_source_events",
            "max_header_encoded_bytes",
            "max_actor_retained_state_bytes",
            "max_pending_update_bytes",
            "max_consumed_authorities",
            "max_consumed_authority_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"RuntimeCaps.{name} must be a nonnegative integer")
        if self.C > self.K:
            raise ValueError("coalition cap C cannot exceed candidate cap K")
        if self.max_episode_work <= 0 or self.max_wall_time_ns <= 0:
            raise ValueError("episode work and wall-time caps must be positive")

    def payload(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "K",
                "C",
                "B",
                "H",
                "M",
                "R",
                "A",
                "P",
                "max_action_encoded_bytes",
                "max_message_encoded_bytes",
                "max_message_header_items",
                "max_endogenous_payload_bytes",
                "max_event_payload_bytes",
                "max_header_referents",
                "max_header_factors",
                "max_header_shards",
                "max_header_source_events",
                "max_header_encoded_bytes",
                "max_episode_work",
                "max_wall_time_ns",
                "max_actor_retained_state_bytes",
                "max_pending_update_bytes",
                "max_consumed_authorities",
                "max_consumed_authority_bytes",
            )
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: CandidateMode
    caps: RuntimeCaps
    shard_cap: int = 1
    bids_per_shard: int = 1
    peer_entry_cap: int = 1
    peer_nomination_degree: int = 1
    endogenous_fanout_cap: int = 1
    queue_cap: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CandidateMode):
            raise ValueError("mode must be a CandidateMode")
        for name in (
            "shard_cap",
            "bids_per_shard",
            "peer_entry_cap",
            "peer_nomination_degree",
            "endogenous_fanout_cap",
            "queue_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.bids_per_shard > self.caps.K or self.peer_entry_cap > self.caps.K:
            raise ValueError("per-shard bids and peer entries cannot exceed K")
        if self.endogenous_fanout_cap > self.queue_cap:
            raise ValueError("endogenous fanout cannot exceed queue capacity")

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "caps": self.caps.payload(),
            "shard_cap": self.shard_cap,
            "bids_per_shard": self.bids_per_shard,
            "peer_entry_cap": self.peer_entry_cap,
            "peer_nomination_degree": self.peer_nomination_degree,
            "endogenous_fanout_cap": self.endogenous_fanout_cap,
            "queue_cap": self.queue_cap,
        }


@dataclass(frozen=True, slots=True)
class DispatchRequest:

    event_header: DispatchEventHeader
    readiness: tuple[ReadinessEstimate, ...]
    mode: CandidateMode
    caps: RuntimeCaps
    reasoning_round: int


@dataclass(frozen=True, slots=True)
class DispatchDecision:

    selected_actor_ids: tuple[str, ...]
    considered_coalitions: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selected_actor_ids, tuple) or not isinstance(
            self.considered_coalitions, tuple
        ):
            raise ValueError("dispatch decision collections must be immutable tuples")

    @classmethod
    def select(cls, *actor_ids: str) -> DispatchDecision:
        selected = tuple(actor_ids)
        return cls(selected, (selected,) if selected else ())


@runtime_checkable
class DispatchPolicy(Protocol):
    @property
    def authority_id(self) -> str: ...

    @property
    def state_version(self) -> str: ...

    @property
    def retained_state_bytes(self) -> int: ...

    def select(self, request: DispatchRequest) -> DispatchDecision: ...


class ScriptedDispatchPolicy:

    _MAX_REQUEST_COUNT = (1 << 64) - 1

    def __init__(
        self,
        script: Mapping[tuple[str, int], DispatchDecision | Sequence[str]],
        *,
        default: DispatchDecision | Sequence[str] = (),
    ) -> None:
        self._script = MappingProxyType({key: self._coerce(value) for key, value in script.items()})
        self._default = self._coerce(default)
        self.request_count = 0
        self.last_request: DispatchRequest | None = None

    @staticmethod
    def _coerce(value: DispatchDecision | Sequence[str]) -> DispatchDecision:
        if isinstance(value, DispatchDecision):
            return value
        return DispatchDecision.select(*tuple(value))

    def select(self, request: DispatchRequest) -> DispatchDecision:
        self.request_count = min(self.request_count + 1, self._MAX_REQUEST_COUNT)
        self.last_request = request
        value = self._script.get((request.event_header.event_id, request.reasoning_round), self._default)
        return value

    def _static_payload(self) -> dict[str, object]:
        return {
            "script": [
                {
                    "key": [event_id, reasoning_round],
                    "selected": list(decision.selected_actor_ids),
                    "beam": [list(row) for row in decision.considered_coalitions],
                }
                for (event_id, reasoning_round), decision in sorted(self._script.items())
            ],
            "default": {
                "selected": list(self._default.selected_actor_ids),
                "beam": [list(row) for row in self._default.considered_coalitions],
            },
        }

    def _state_payload(self) -> dict[str, object]:
        last_request = None
        if self.last_request is not None:
            last_request = {
                "event_header": self.last_request.event_header.payload(),
                "readiness": [
                    {
                        "actor_id": row.actor_id,
                        "state_version": row.state_version,
                        "compatible": row.compatible,
                        "expected_decision_value": row.expected_decision_value,
                        "predicted_operations": row.predicted_operations,
                        "predicted_message_bytes": row.predicted_message_bytes,
                        "stale_state_risk": row.stale_state_risk,
                        "deadline_risk": row.deadline_risk,
                        "nominated_actor_ids": list(row.nominated_actor_ids),
                        "estimation_operations": row.estimation_operations,
                    }
                    for row in self.last_request.readiness
                ],
                "mode": self.last_request.mode.value,
                "caps": self.last_request.caps.payload(),
                "reasoning_round": self.last_request.reasoning_round,
            }
        return {
            "authority_id": self.authority_id,
            "request_count": self.request_count,
            "last_request": last_request,
        }

    @property
    def authority_id(self) -> str:
        return canonical_sha256(
            {
                "schema": "mop-escs-scripted-dispatch-policy-authority/v1",
                **self._static_payload(),
            }
        )

    @property
    def state_version(self) -> str:
        return canonical_sha256(
            {
                "schema": "mop-escs-dispatch-policy-state/v1",
                **self._state_payload(),
            }
        )

    @property
    def retained_state_bytes(self) -> int:
        return len(canonical_bytes({**self._static_payload(), **self._state_payload()}))


@dataclass(frozen=True, slots=True)
class MessageDelivery:
    message: ClaimMessage
    recipient_actor_id: str
    produced_round: int
    consumed_round: int


@dataclass(frozen=True, slots=True)
class RejectedClaim:
    message_id: str
    producer_actor_id: str
    recipient_actor_id: str
    faults: tuple[ClaimFault | str, ...]
    phase: str


class ActionFault(StrEnum):
    INTEGRITY = "integrity"
    PRODUCER = "producer"
    SOURCE_EVENT = "source_event"
    BRANCH = "branch"
    REFERENT = "referent"
    PRODUCER_STATE = "producer_state"
    PRODUCER_OPERATIONS = "producer_operations"
    EXPIRED = "expired"
    FUTURE_DATED = "future_dated"
    CREATED_BEFORE_SOURCE = "created_before_source"
    EVIDENCE_CLASS_DOWNGRADE = "evidence_class_downgrade"
    EPISTEMIC_LAUNDERING = "epistemic_laundering"
    SIMULATION_ON_FACTUAL_BRANCH = "simulation_on_factual_branch"


@dataclass(frozen=True, slots=True)
class RejectedAction:
    action_id: str
    producer_actor_id: str
    faults: tuple[ActionFault, ...]


@dataclass(frozen=True, slots=True)
class RoundTrace:
    event_header: DispatchEventHeader
    candidate_actor_ids: tuple[str, ...]
    selected_actor_ids: tuple[str, ...]
    considered_coalitions: tuple[tuple[str, ...], ...]
    staged_message_ids: tuple[str, ...]
    consumed_message_ids: tuple[str, ...]
    accepted_action_ids: tuple[str, ...]
    admitted_endogenous_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeTrace:
    trace_id: str
    runtime_id: str
    authority_sequence: int
    mode: CandidateMode
    caps: RuntimeCaps
    initial_event_id: str | None
    rounds: tuple[RoundTrace, ...]
    message_deliveries: tuple[MessageDelivery, ...]
    rejected_claims: tuple[RejectedClaim, ...]
    action_intents: tuple[ActionIntent, ...]
    rejected_actions: tuple[RejectedAction, ...]
    active_actor_ids: tuple[str, ...]
    endogenous_rounds: int
    quiescent: bool
    halt_reason: str
    ledger_start_sequence: int
    ledger_end_sequence: int
    full_trace_sha256: str

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "trace_id": self.trace_id,
            "runtime_id": self.runtime_id,
            "authority_sequence": self.authority_sequence,
            "mode": self.mode.value,
            "caps": self.caps.payload(),
            "initial_event_id": self.initial_event_id,
            "rounds": [
                {
                    "event_header": row.event_header.payload(),
                    "candidate_actor_ids": list(row.candidate_actor_ids),
                    "selected_actor_ids": list(row.selected_actor_ids),
                    "considered_coalitions": [list(value) for value in row.considered_coalitions],
                    "staged_message_ids": list(row.staged_message_ids),
                    "consumed_message_ids": list(row.consumed_message_ids),
                    "accepted_action_ids": list(row.accepted_action_ids),
                    "admitted_endogenous_event_ids": list(row.admitted_endogenous_event_ids),
                }
                for row in self.rounds
            ],
            "message_deliveries": [
                {
                    "message": row.message.wire_payload(),
                    "recipient_actor_id": row.recipient_actor_id,
                    "produced_round": row.produced_round,
                    "consumed_round": row.consumed_round,
                }
                for row in self.message_deliveries
            ],
            "rejected_claims": [
                {
                    "message_id": row.message_id,
                    "producer_actor_id": row.producer_actor_id,
                    "recipient_actor_id": row.recipient_actor_id,
                    "faults": [str(value) for value in row.faults],
                    "phase": row.phase,
                }
                for row in self.rejected_claims
            ],
            "action_intents": [
                {
                    "action_id": row.action_id,
                    "header": row.identity_payload(),
                    "payload_base64": base64.b64encode(row.payload_bytes).decode("ascii"),
                }
                for row in self.action_intents
            ],
            "rejected_actions": [
                {
                    "action_id": row.action_id,
                    "producer_actor_id": row.producer_actor_id,
                    "faults": [value.value for value in row.faults],
                }
                for row in self.rejected_actions
            ],
            "active_actor_ids": list(self.active_actor_ids),
            "endogenous_rounds": self.endogenous_rounds,
            "quiescent": self.quiescent,
            "halt_reason": self.halt_reason,
            "ledger_start_sequence": self.ledger_start_sequence,
            "ledger_end_sequence": self.ledger_end_sequence,
        }
        if include_digest:
            result["full_trace_sha256"] = self.full_trace_sha256
        return result

    def validate_integrity(self) -> bool:
        expected_trace_id = canonical_sha256(
            {
                "schema": "mop-escs-runtime-trace-authority/v1",
                "runtime_id": self.runtime_id,
                "authority_sequence": self.authority_sequence,
            }
        )
        return (
            self.trace_id == expected_trace_id
            and canonical_sha256(self.payload(include_digest=False)) == self.full_trace_sha256
        )


@dataclass(frozen=True, slots=True)
class _BranchUpdate:
    branch_id: str
    active_actor_ids: tuple[str, ...]
    actor_state_versions: tuple[tuple[str, str], ...]
    activated_event_ids: tuple[str, ...]
    emitted_claim_ids: tuple[str, ...]
    emitted_action_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingUpdate:
    branches: tuple[_BranchUpdate, ...]


@dataclass(frozen=True, slots=True)
class _StagedClaim:
    message: ClaimMessage
    recipient_actor_id: str
    producer_actor_id: str
    produced_round: int
    source_header: DispatchEventHeader


class _ActorCatalog:

    def __init__(self, actors: Sequence[Actor]) -> None:
        actor_rows = tuple(actors)
        by_id: dict[str, Actor] = {}
        descriptors: dict[str, ActorDescriptor] = {}
        for actor in actor_rows:
            descriptor = actor.descriptor
            if descriptor.actor_id in by_id:
                raise ValueError(f"duplicate actor id {descriptor.actor_id!r}")
            if actor.state_version != actor.state_version.lower() or len(actor.state_version) != 64:
                raise ValueError(f"actor {descriptor.actor_id!r} has an invalid state version")
            try:
                int(actor.state_version, 16)
            except ValueError as exc:
                raise ValueError(f"actor {descriptor.actor_id!r} has an invalid state version") from exc
            if (
                isinstance(actor.retained_state_bytes, bool)
                or not isinstance(actor.retained_state_bytes, int)
                or actor.retained_state_bytes < 0
            ):
                raise ValueError(f"actor {descriptor.actor_id!r} has invalid retained-state bytes")
            by_id[descriptor.actor_id] = actor
            descriptors[descriptor.actor_id] = descriptor
        for descriptor in descriptors.values():
            unknown = set(descriptor.declared_peer_ids) - set(by_id)
            if unknown:
                raise ValueError(f"actor {descriptor.actor_id!r} declares unknown peers {sorted(unknown)!r}")

        by_event: dict[str, list[str]] = defaultdict(list)
        wildcard: dict[str, list[str]] = defaultdict(list)
        by_event_factor: dict[tuple[str, str], list[str]] = defaultdict(list)
        by_event_shard: dict[tuple[str, str], list[str]] = defaultdict(list)
        peer_entries: dict[str, list[str]] = defaultdict(list)
        for actor_id, descriptor in descriptors.items():
            for event_kind in descriptor.subscribed_event_types:
                by_event[event_kind].append(actor_id)
                if descriptor.factor_scopes:
                    for factor in descriptor.factor_scopes:
                        by_event_factor[(event_kind, factor)].append(actor_id)
                else:
                    wildcard[event_kind].append(actor_id)
                for shard in descriptor.shard_ids:
                    by_event_shard[(event_kind, shard)].append(actor_id)
                if descriptor.peer_entry:
                    peer_entries[event_kind].append(actor_id)
        self._actors = MappingProxyType(by_id)
        self._descriptors = MappingProxyType(descriptors)
        self._by_event = {key: tuple(value) for key, value in by_event.items()}
        self._wildcard = {key: tuple(value) for key, value in wildcard.items()}
        self._by_event_factor = {key: tuple(value) for key, value in by_event_factor.items()}
        self._by_event_shard = {key: tuple(value) for key, value in by_event_shard.items()}
        self._peer_entries = {key: tuple(value) for key, value in peer_entries.items()}
        self.index_construction_units = sum(
            1
            + len(row.subscribed_event_types)
            + len(row.factor_scopes)
            + len(row.shard_ids)
            + len(row.declared_peer_ids)
            for row in descriptors.values()
        )

    def actor(self, actor_id: str) -> Actor:
        return self._actors[actor_id]

    def has_actor(self, actor_id: str) -> bool:
        return actor_id in self._actors

    @property
    def actor_ids(self) -> tuple[str, ...]:
        return tuple(self._actors)

    @property
    def total_retained_state_bytes(self) -> int:
        return sum(actor.retained_state_bytes for actor in self._actors.values())

    def routing_payload(self) -> dict[str, object]:

        descriptors = [
            {
                "actor_id": actor_id,
                "subscribed_event_types": list(descriptor.subscribed_event_types),
                "factor_scopes": list(descriptor.factor_scopes),
                "shard_ids": list(descriptor.shard_ids),
                "declared_peer_ids": list(descriptor.declared_peer_ids),
                "peer_entry": descriptor.peer_entry,
            }
            for actor_id, descriptor in self._descriptors.items()
        ]

        def rows(mapping: Mapping[Any, tuple[str, ...]]) -> list[list[object]]:
            return [
                [list(key) if isinstance(key, tuple) else key, list(value)]
                for key, value in sorted(mapping.items(), key=lambda row: str(row[0]))
            ]

        return {
            "actor_descriptors": descriptors,
            "by_event": rows(self._by_event),
            "wildcard": rows(self._wildcard),
            "by_event_factor": rows(self._by_event_factor),
            "by_event_shard": rows(self._by_event_shard),
            "peer_entries": rows(self._peer_entries),
        }

    def replace_many(self, replacements: Mapping[str, Actor]) -> None:
        next_actors = dict(self._actors)
        next_actors.update(replacements)
        self._actors = MappingProxyType(next_actors)

    def descriptor(self, actor_id: str) -> ActorDescriptor:
        return self._descriptors[actor_id]

    def central_ids(self, header: DispatchEventHeader, cap: int) -> tuple[tuple[str, ...], int]:
        sources = [self._wildcard.get(header.event_kind, ())]
        sources.extend(
            self._by_event_factor.get((header.event_kind, factor), ()) for factor in header.factor_scope
        )
        return self._bounded_merge(sources, cap)

    def sharded_ids(
        self,
        header: DispatchEventHeader,
        *,
        shard_cap: int,
        bids_per_shard: int,
        candidate_cap: int,
    ) -> tuple[tuple[str, ...], int]:
        sources = [
            self._by_event_shard.get((header.event_kind, shard), ())[:bids_per_shard]
            for shard in header.routing_shards[:shard_cap]
        ]
        return self._bounded_merge(sources, candidate_cap)

    def peer_entry_ids(self, event_kind: str, cap: int) -> tuple[str, ...]:
        return self._peer_entries.get(event_kind, ())[:cap]

    @staticmethod
    def _bounded_merge(sources: Sequence[Sequence[str]], cap: int) -> tuple[tuple[str, ...], int]:
        result: list[str] = []
        seen: set[str] = set()
        inspected = 0
        for source in sources:
            for actor_id in source[:cap]:
                inspected += 1
                if actor_id in seen:
                    continue
                seen.add(actor_id)
                result.append(actor_id)
                if len(result) == cap:
                    return tuple(result), inspected
        return tuple(result), inspected


class CoalitionRuntime:

    def __init__(
        self,
        *,
        actors: Sequence[Actor],
        policy: DispatchPolicy,
        schemas: SchemaRegistry,
        config: RuntimeConfig,
        ledger: LifecycleLedger,
        event_ledger: EventLedger | None = None,
        owner: str = "escs.coalition-runtime",
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(policy, DispatchPolicy):
            raise TypeError("policy does not implement DispatchPolicy")
        self._catalog = _ActorCatalog(actors)
        self._policy = policy
        self._schemas = schemas
        self._config = config
        self._ledger = ledger
        self._event_ledger = event_ledger
        self._owner = owner
        self._clock_ns = clock_ns
        self._trace_sequence = 0
        self._pending_updates: dict[str, _PendingUpdate] = {}
        self._consumed_consequence_ids: set[str] = set()
        self._consumed_authorization_ids: set[str] = set()
        self._active_episode_work: int | None = None
        self._active_deadline_ns: int | None = None
        self._retention_last_tick = 0
        self._retention_finalized = False
        self._policy_authority_id = _require_sha256_digest(
            policy.authority_id,
            "dispatch policy authority",
        )
        self._initial_policy_state_version = _require_sha256_digest(
            policy.state_version,
            "dispatch policy state version",
        )
        actor_state_versions = [
            [actor_id, self._catalog.actor(actor_id).state_version]
            for actor_id in sorted(self._catalog.actor_ids)
        ]
        self._runtime_id = canonical_sha256(
            {
                "schema": "mop-escs-runtime-authority/v1",
                "owner": owner,
                "config": config.payload(),
                "routing": self._catalog.routing_payload(),
                "actor_state_versions": actor_state_versions,
                "policy_authority_id": self._policy_authority_id,
                "policy_state_version": self._initial_policy_state_version,
                "initial_lifecycle_head": ledger.head_sha256,
                "initial_event_ledger_head": (event_ledger.head_sha256 if event_ledger is not None else None),
            }
        )
        ledger.charge(
            owner=owner,
            reason="build-bounded-routing-index",
            work=WorkVector(indexing_and_graph_maintenance=max(1, self._catalog.index_construction_units)),
            start_tick=0,
            end_tick=0,
        )
        retained = self._catalog.total_retained_state_bytes
        if retained > config.caps.max_actor_retained_state_bytes:
            ledger.charge(
                owner=owner,
                reason="actor-retained-state-cap-rejection",
                work=WorkVector(indexing_and_graph_maintenance=1 + retained),
                start_tick=0,
                end_tick=0,
            )
            raise RuntimeCapExceeded("actor retained-state bytes exceed the declared cap")

    @property
    def accounting_total(self) -> WorkVector:
        return self._ledger.total

    @property
    def lifecycle_ledger(self) -> LifecycleLedger:

        return self._ledger

    @property
    def event_ledger(self) -> EventLedger | None:

        return self._event_ledger

    @property
    def runtime_id(self) -> str:

        return self._runtime_id

    @property
    def actor_descriptors(self) -> tuple[ActorDescriptor, ...]:

        return tuple(self._catalog.descriptor(actor_id) for actor_id in sorted(self._catalog.actor_ids))

    @property
    def actor_state_versions(self) -> Mapping[str, str]:

        return MappingProxyType(
            {
                actor_id: self._catalog.actor(actor_id).state_version
                for actor_id in sorted(self._catalog.actor_ids)
            }
        )

    @property
    def pending_update_count(self) -> int:
        return sum(
            len(branch.active_actor_ids)
            for pending in self._pending_updates.values()
            for branch in pending.branches
        )

    def _persistent_control_payload(self) -> dict[str, object]:
        policy_state_version = _require_sha256_digest(
            self._policy.state_version,
            "dispatch policy state version",
        )
        if self._policy.authority_id != self._policy_authority_id:
            raise RuntimeContractError("dispatch policy authority changed after runtime construction")
        return {
            "schema": "mop-escs-runtime-retained-control/v1",
            "runtime_id": self._runtime_id,
            "owner": self._owner,
            "config": self._config.payload(),
            "routing": self._catalog.routing_payload(),
            "policy_authority_id": self._policy_authority_id,
            "policy_state_version": policy_state_version,
            "trace_sequence": self._trace_sequence,
            "pending_updates": self._pending_payload(self._pending_updates),
            "consumed_consequence_ids": sorted(self._consumed_consequence_ids),
            "consumed_authorization_ids": sorted(self._consumed_authorization_ids),
            "last_accounted_tick": self._retention_last_tick,
            "finalized": self._retention_finalized,
        }

    @property
    def retained_state_bytes(self) -> int:

        policy_bytes = self._policy.retained_state_bytes
        if isinstance(policy_bytes, bool) or not isinstance(policy_bytes, int) or policy_bytes < 0:
            raise RuntimeContractError("dispatch policy reported invalid retained-state bytes")
        return (
            self._catalog.total_retained_state_bytes
            + policy_bytes
            + len(canonical_bytes(self._persistent_control_payload()))
        )

    @property
    def last_accounted_tick(self) -> int:
        return self._retention_last_tick

    @property
    def finalized(self) -> bool:
        return self._retention_finalized

    def _account_retention_until(self, end_tick: int, *, reason: str) -> None:
        if isinstance(end_tick, bool) or not isinstance(end_tick, int) or end_tick < 0:
            raise ValueError("retention end tick must be a nonnegative integer")
        if end_tick < self._retention_last_tick:
            raise RuntimeContractError("runtime retention accounting cannot move backward")
        if end_tick == self._retention_last_tick:
            return
        retained_bytes = self.retained_state_bytes
        self._ledger.charge_retention(
            owner=self._owner,
            reason=reason,
            retained_bytes=retained_bytes,
            start_tick=self._retention_last_tick,
            end_tick=end_tick,
        )
        self._retention_last_tick = end_tick

    def finalize(self, *, end_tick: int) -> None:

        if self._active_episode_work is not None:
            raise RuntimeContractError("cannot finalize an active runtime episode")
        if self._retention_finalized:
            raise RuntimeContractError("runtime retention ownership is already finalized")
        self._account_retention_until(end_tick, reason="runtime-retained-state-finalization")
        self._retention_finalized = True

    def _charge(
        self,
        *,
        reason: str,
        work: WorkVector,
        tick: int,
        header: DispatchEventHeader | None = None,
    ) -> None:
        branch = BranchRef(header.branch_id) if header is not None else FACTUAL_BRANCH
        event_ids = (EventRef(header.event_id),) if header is not None else ()
        self._ledger.charge(
            owner=self._owner,
            reason=reason,
            work=work,
            start_tick=tick,
            end_tick=tick,
            branch_id=branch,
            causal_event_ids=event_ids,
        )
        if self._active_episode_work is not None:
            self._active_episode_work += work.total_work

    def _check_episode_bounds(self) -> None:
        if self._active_episode_work is None or self._active_deadline_ns is None:
            return
        if self._active_episode_work > self._config.caps.max_episode_work:
            raise RuntimeCapExceeded("episode abstract work exceeded its declared cap")
        if self._clock_ns() > self._active_deadline_ns:
            raise RuntimeCapExceeded("episode wall-time budget was exhausted")

    def _readiness(self, actor_id: str, header: DispatchEventHeader, tick: int) -> ReadinessEstimate:
        actor = self._catalog.actor(actor_id)
        before_state = actor.state_version
        try:
            estimate = actor.readiness(header)
        except Exception:
            self._charge(
                reason=f"header-readiness-error:{actor_id}",
                work=WorkVector(dispatch_and_exploration=1),
                tick=tick,
                header=header,
            )
            raise
        if not isinstance(estimate, ReadinessEstimate):
            self._charge(
                reason=f"header-readiness-invalid-result:{actor_id}",
                work=WorkVector(dispatch_and_exploration=1),
                tick=tick,
                header=header,
            )
            raise RuntimeContractError("actor returned an invalid readiness estimate")
        self._charge(
            reason=f"header-readiness:{actor_id}",
            work=WorkVector(dispatch_and_exploration=estimate.estimation_operations + 1),
            tick=tick,
            header=header,
        )
        self._check_episode_bounds()
        if actor.state_version != before_state:
            raise RuntimeContractError("readiness mutated actor state")
        if estimate.actor_id != actor_id or estimate.state_version != actor.state_version:
            raise RuntimeContractError("readiness actor or state identity mismatch")
        if not self._catalog.descriptor(actor_id).header_compatible(header) and estimate.compatible:
            raise RuntimeContractError("actor claimed readiness outside its declared subscription")
        return estimate

    def _candidate_estimates(self, header: DispatchEventHeader, tick: int) -> tuple[ReadinessEstimate, ...]:
        caps = self._config.caps
        if len(header.factor_scope) > caps.K:
            raise RuntimeCapExceeded("event factor scope exceeds K-bounded routing input")
        mode = self._config.mode
        if mode is CandidateMode.BOUNDED_CENTRAL:
            actor_ids, inspected = self._catalog.central_ids(header, caps.K)
            lookup_units = 1 + len(header.factor_scope) + inspected
            self._charge(
                reason="bounded-central-candidate-lookup",
                work=WorkVector(dispatch_and_exploration=lookup_units),
                tick=tick,
                header=header,
            )
            estimates = tuple(self._readiness(actor_id, header, tick) for actor_id in actor_ids)
        elif mode is CandidateMode.SHARDED_SUBSCRIPTION:
            if len(header.routing_shards) > self._config.shard_cap:
                raise RuntimeCapExceeded("event routing shards exceed the declared shard cap")
            actor_ids, inspected = self._catalog.sharded_ids(
                header,
                shard_cap=self._config.shard_cap,
                bids_per_shard=self._config.bids_per_shard,
                candidate_cap=caps.K,
            )
            lookup_units = 1 + len(header.routing_shards) + inspected
            self._charge(
                reason="sharded-candidate-lookup",
                work=WorkVector(dispatch_and_exploration=lookup_units),
                tick=tick,
                header=header,
            )
            estimates = tuple(self._readiness(actor_id, header, tick) for actor_id in actor_ids)
        else:
            estimates = self._peer_candidates(header, tick)
        compatible = tuple(row for row in estimates if row.compatible)
        if len(estimates) > caps.K or len(compatible) > caps.K:
            raise RuntimeCapExceeded("candidate retrieval exceeded K")
        return compatible

    def _peer_candidates(self, header: DispatchEventHeader, tick: int) -> tuple[ReadinessEstimate, ...]:
        caps = self._config.caps
        entries = self._catalog.peer_entry_ids(header.event_kind, self._config.peer_entry_cap)
        self._charge(
            reason="peer-entry-lookup",
            work=WorkVector(dispatch_and_exploration=1 + len(entries)),
            tick=tick,
            header=header,
        )
        queue = deque((actor_id, 0) for actor_id in entries)
        seen: set[str] = set()
        estimates: list[ReadinessEstimate] = []
        while queue and len(estimates) < caps.K:
            actor_id, depth = queue.popleft()
            if actor_id in seen:
                continue
            seen.add(actor_id)
            descriptor = self._catalog.descriptor(actor_id)
            if not descriptor.header_compatible(header):
                continue
            estimate = self._readiness(actor_id, header, tick)
            estimates.append(estimate)
            if len(estimate.nominated_actor_ids) > self._config.peer_nomination_degree:
                raise RuntimeCapExceeded("peer readiness exceeded the nomination degree")
            if depth >= caps.H:
                continue
            undeclared = set(estimate.nominated_actor_ids) - set(descriptor.declared_peer_ids)
            if undeclared:
                raise RuntimeContractError(
                    f"actor {actor_id!r} nominated undeclared peers {sorted(undeclared)!r}"
                )
            self._charge(
                reason=f"peer-nomination-hop:{actor_id}",
                work=WorkVector(dispatch_and_exploration=1 + len(estimate.nominated_actor_ids)),
                tick=tick,
                header=header,
            )
            for nominee in estimate.nominated_actor_ids:
                if nominee not in seen and len(seen) + len(queue) < caps.K:
                    queue.append((nominee, depth + 1))
        return tuple(estimates)

    def _dispatch(
        self,
        header: DispatchEventHeader,
        readiness: tuple[ReadinessEstimate, ...],
        reasoning_round: int,
        tick: int,
    ) -> DispatchDecision:
        request = DispatchRequest(header, readiness, self._config.mode, self._config.caps, reasoning_round)
        try:
            decision = self._policy.select(request)
        except Exception:
            self._charge(
                reason="dispatch-policy-error",
                work=WorkVector(dispatch_and_exploration=1 + len(readiness)),
                tick=tick,
                header=header,
            )
            raise
        if not isinstance(decision, DispatchDecision):
            self._charge(
                reason="dispatch-policy-invalid-result",
                work=WorkVector(dispatch_and_exploration=1 + len(readiness)),
                tick=tick,
                header=header,
            )
            raise RuntimeContractError("policy returned an invalid dispatch decision")
        try:
            self._validate_decision_shape(decision)
        except RuntimeContractError:
            self._charge(
                reason="dispatch-policy-shape-rejection",
                work=WorkVector(
                    dispatch_and_exploration=(
                        1 + len(decision.selected_actor_ids) + len(decision.considered_coalitions)
                    )
                ),
                tick=tick,
                header=header,
            )
            raise
        beam_nodes = sum(len(coalition) for coalition in decision.considered_coalitions)
        self._charge(
            reason="dispatch-policy-selection",
            work=WorkVector(dispatch_and_exploration=1 + len(readiness) + beam_nodes),
            tick=tick,
            header=header,
        )
        self._check_episode_bounds()
        self._validate_decision(decision, readiness)
        return decision

    def _validate_decision_shape(self, decision: DispatchDecision) -> None:
        caps = self._config.caps
        if len(decision.selected_actor_ids) > caps.C:
            raise RuntimeCapExceeded("selected coalition exceeded C")
        if len(decision.considered_coalitions) > caps.B:
            raise RuntimeCapExceeded("policy beam exceeded B")
        for coalition in decision.considered_coalitions:
            if not isinstance(coalition, tuple):
                raise RuntimeContractError("policy beam coalitions must be immutable tuples")
            if len(coalition) > caps.C:
                raise RuntimeCapExceeded("a policy beam coalition exceeded C")

    def _validate_decision(
        self, decision: DispatchDecision, readiness: tuple[ReadinessEstimate, ...]
    ) -> None:
        self._validate_decision_shape(decision)
        candidate_ids = {row.actor_id for row in readiness}
        selected = decision.selected_actor_ids
        if len(selected) != len(set(selected)):
            raise RuntimeContractError("selected coalition contains duplicate actors")
        if not set(selected) <= candidate_ids:
            raise RuntimeContractError("policy selected an actor outside the candidate set")
        for coalition in decision.considered_coalitions:
            if len(coalition) != len(set(coalition)):
                raise RuntimeContractError("policy beam contains a duplicate actor")
            if not set(coalition) <= candidate_ids:
                raise RuntimeContractError("policy beam contains a non-candidate actor")
        if selected and selected not in decision.considered_coalitions:
            raise RuntimeContractError("selected coalition was not present in the declared beam")

    @staticmethod
    def _action_faults(
        action: ActionIntent,
        *,
        actor_id: str,
        state_version: str,
        header: DispatchEventHeader,
        now_tick: int,
    ) -> tuple[ActionFault, ...]:
        faults: list[ActionFault] = []
        if not action.integrity_valid():
            faults.append(ActionFault.INTEGRITY)
        if action.producer_actor_id != actor_id:
            faults.append(ActionFault.PRODUCER)
        if action.source_event_id != header.event_id:
            faults.append(ActionFault.SOURCE_EVENT)
        if action.branch_id != header.branch_id:
            faults.append(ActionFault.BRANCH)
        if not set(action.referent_hypotheses) <= set(header.referent_hypotheses):
            faults.append(ActionFault.REFERENT)
        if action.producer_state_version != state_version:
            faults.append(ActionFault.PRODUCER_STATE)
        if now_tick > action.expiry_tick:
            faults.append(ActionFault.EXPIRED)
        if now_tick < action.created_tick:
            faults.append(ActionFault.FUTURE_DATED)
        if action.created_tick < header.created_tick:
            faults.append(ActionFault.CREATED_BEFORE_SOURCE)
        if action.evidence_class.taint_rank < header.evidence_class.taint_rank:
            faults.append(ActionFault.EVIDENCE_CLASS_DOWNGRADE)
        if epistemic_rank(action.epistemic_status) < epistemic_rank(header.epistemic_status):
            faults.append(ActionFault.EPISTEMIC_LAUNDERING)
        if action.epistemic_status is EpistemicStatus.SIMULATED and action.branch_id == str(FACTUAL_BRANCH):
            faults.append(ActionFault.SIMULATION_ON_FACTUAL_BRANCH)
        return tuple(dict.fromkeys(faults))

    def _validate_endogenous_proposal(
        self,
        proposal: EndogenousHypothesisProposal,
        *,
        parent: DispatchEventHeader,
        known_evidence: Mapping[str, EventClaimEvidence],
    ) -> None:
        if parent.event_id not in proposal.source_event_ids:
            raise RuntimeContractError("endogenous proposal omits its authorizing parent")
        if not set(proposal.source_event_ids) <= set(known_evidence):
            raise RuntimeContractError("endogenous proposal names an unknown source event")
        if proposal.created_tick < parent.created_tick:
            raise RuntimeContractError("endogenous proposal predates its authorizing parent")
        if epistemic_rank(proposal.epistemic_status) < epistemic_rank(parent.epistemic_status):
            raise RuntimeContractError("endogenous proposal launders epistemic status")
        source_taint = max(
            known_evidence[event_id].evidence_class.taint_rank for event_id in proposal.source_event_ids
        )
        if proposal.evidence_class.taint_rank < source_taint:
            raise RuntimeContractError("endogenous proposal downgrades evidence-class taint")
        if proposal.branch_id != parent.branch_id and not (
            parent.branch_id == str(FACTUAL_BRANCH)
            and proposal.epistemic_status is EpistemicStatus.SIMULATED
            and proposal.branch_id != str(FACTUAL_BRANCH)
        ):
            raise RuntimeContractError("endogenous proposal crossed an unauthorized branch")

    @staticmethod
    def _dispatch_from_hypothesis(
        event: HypothesisEvent,
        *,
        payload_bytes: bytes,
    ) -> DispatchEvent:
        committed = event.envelope.source_and_provenance.value()
        if committed.get("schema") != "mop-escs-runtime-endogenous-provenance/v1":
            raise RuntimeContractError("ledger event lacks runtime endogenous provenance")
        header = DispatchEventHeader(
            event_id=str(event.event_id),
            event_kind=event.kind.value,
            branch_id=str(event.branch_id),
            producer_state_version=event.envelope.producer_state_version,
            epistemic_status=event.epistemic_status,
            evidence_class=event.evidence_class,
            referent_hypotheses=tuple(event.referent_hypotheses.value()["hypotheses"]),
            factor_scope=tuple(committed["factor_scope"]),
            routing_shards=tuple(committed["routing_shards"]),
            source_event_ids=tuple(str(value) for value in event.envelope.causal_parent_ids),
            created_tick=event.envelope.clock_end_tick,
            expiry_tick=int(committed["expiry_tick"]),
            payload_digest=event.envelope.payload_digest,
            representation_payload_digest=str(committed["representation_payload_digest"]),
            endogenous=True,
            reasoning_depth=int(committed["reasoning_depth"]),
        )
        return DispatchEvent(header=header, payload_bytes=payload_bytes)

    def _append_endogenous_proposal(
        self,
        proposal: EndogenousHypothesisProposal,
        *,
        actor_id: str,
        actor_state_version: str,
        parent: DispatchEventHeader,
        reasoning_depth: int,
    ) -> DispatchEvent:
        if self._event_ledger is None:
            raise RuntimeContractError("endogenous proposals require an injected EventLedger")
        if proposal.producer_actor_id != actor_id:
            raise RuntimeContractError("endogenous proposal producer actor mismatch")
        if proposal.producer_actor_state_version != actor_state_version:
            raise RuntimeContractError("endogenous proposal producer state mismatch")
        if parent.event_id not in proposal.source_event_ids:
            raise RuntimeContractError("endogenous proposal omits its authorizing parent")

        provenance = {
            "schema": "mop-escs-runtime-endogenous-provenance/v1",
            "proposal_id": proposal.proposal_id,
            "producer_actor_id": actor_id,
            "producer_actor_state_version": actor_state_version,
            "representation_payload_digest": proposal.representation_payload_digest,
            "payload_form": proposal.payload_form,
            "factor_scope": list(proposal.factor_scope),
            "routing_shards": list(proposal.routing_shards),
            "expiry_tick": proposal.expiry_tick,
            "reasoning_depth": reasoning_depth,
        }
        event = HypothesisEvent.create(
            origin=HypothesisOrigin.ACTOR,
            epistemic_status=proposal.epistemic_status,
            referent_hypotheses={"hypotheses": list(proposal.referent_hypotheses)},
            factor_change_distribution={
                "factor_scope": list(proposal.factor_scope),
                "payload_form": proposal.payload_form,
                "representation_payload_digest": proposal.representation_payload_digest,
            },
            decision_relevance_distribution={"values": []},
            reducibility_distribution={"values": []},
            supporting_event_ids=tuple(EventRef(value) for value in proposal.supporting_event_ids),
            calibrated_confidence=proposal.calibrated_confidence,
            abstention_reason=None,
            predicted_value_of_further_computation=(proposal.predicted_value_of_further_computation),
            causal_parent_ids=tuple(EventRef(value) for value in proposal.source_event_ids),
            counterfactual_branch_id=BranchRef(proposal.branch_id),
            clock_start_tick=proposal.created_tick,
            clock_end_tick=proposal.created_tick,
            source_and_provenance=provenance,
            measured_creation_cost=WorkVector(
                event_formation=1 + proposal.encoded_bytes + proposal.producer_operations
            ),
            evidence_class=proposal.evidence_class,
        )
        preview = self._dispatch_from_hypothesis(event, payload_bytes=proposal.payload_bytes)
        self._admit_dispatch_event(preview, proposal.created_tick)
        try:
            self._event_ledger.append(event)
        except ValueError as exc:
            raise RuntimeContractError("endogenous event ledger append was rejected") from exc
        resident = self._event_ledger.get(event.event_id)
        if not isinstance(resident, HypothesisEvent):
            raise RuntimeContractError("endogenous ledger identity resolved to a non-hypothesis")
        return self._dispatch_from_hypothesis(resident, payload_bytes=proposal.payload_bytes)

    def _admit_dispatch_event(self, event: DispatchEvent, tick: int) -> None:
        header = event.header
        caps = self._config.caps
        count_violations = (
            (len(header.referent_hypotheses), caps.max_header_referents, "referents"),
            (len(header.factor_scope), caps.max_header_factors, "factor scopes"),
            (len(header.routing_shards), caps.max_header_shards, "routing shards"),
            (len(header.source_event_ids), caps.max_header_source_events, "source events"),
            (len(event.payload_bytes), caps.max_event_payload_bytes, "event payload bytes"),
        )
        for actual, maximum, label in count_violations:
            if actual > maximum:
                self._charge(
                    reason=f"dispatch-{label.replace(' ', '-')}-cap-rejection",
                    work=WorkVector(dispatch_and_exploration=1 + actual),
                    tick=tick,
                    header=header,
                )
                raise RuntimeCapExceeded(f"dispatch {label} exceeded its declared cap")
        header_bytes = len(canonical_bytes(header.payload()))
        self._charge(
            reason="dispatch-header-validation",
            work=WorkVector(dispatch_and_exploration=1 + header_bytes),
            tick=tick,
            header=header,
        )
        if header_bytes > caps.max_header_encoded_bytes:
            raise RuntimeCapExceeded("dispatch encoded header bytes exceeded its declared cap")
        self._check_episode_bounds()

    def _consume_staged_claims(
        self,
        *,
        actor_id: str,
        header: DispatchEventHeader,
        tick: int,
        reasoning_round: int,
        staged: Sequence[_StagedClaim],
        known_evidence: Mapping[str, EventClaimEvidence],
        deliveries: list[MessageDelivery],
        rejected: list[RejectedClaim],
    ) -> tuple[tuple[ClaimMessage, ...], tuple[str, ...]]:
        accepted: list[ClaimMessage] = []
        consumed_ids: list[str] = []
        for row in staged:
            message = row.message
            producer_id = message.header.producer_actor_id
            producer_versions: tuple[tuple[str, tuple[str, ...]], ...] = ()
            if self._catalog.has_actor(producer_id):
                producer_versions = ((producer_id, (self._catalog.actor(producer_id).state_version,)),)
            context = ClaimValidationContext(
                now_tick=tick,
                branch_id=header.branch_id,
                factual_branch_id=str(FACTUAL_BRANCH),
                allowed_referents=frozenset(header.referent_hypotheses),
                allowed_factor_scopes=frozenset(header.factor_scope),
                event_evidence=tuple(known_evidence.values()),
                accepted_producer_state_versions=producer_versions,
            )
            self._charge(
                reason=f"message-consumption-validation:{producer_id}->{actor_id}",
                work=WorkVector(messages=1 + message.encoded_bytes),
                tick=tick,
                header=header,
            )
            validation = validate_claim(message, schemas=self._schemas, context=context)
            if validation.faults:
                rejected.append(
                    RejectedClaim(
                        message.header.message_id,
                        row.producer_actor_id,
                        actor_id,
                        validation.faults,
                        "consumption",
                    )
                )
                continue
            accepted.append(message)
            consumed_ids.append(message.header.message_id)
            deliveries.append(MessageDelivery(message, actor_id, row.produced_round, reasoning_round))
        self._check_episode_bounds()
        return tuple(accepted), tuple(consumed_ids)

    @staticmethod
    def _pending_payload(pending: Mapping[str, _PendingUpdate]) -> dict[str, object]:
        return {
            trace_id: [
                {
                    "branch_id": branch.branch_id,
                    "active_actor_ids": list(branch.active_actor_ids),
                    "actor_state_versions": [list(row) for row in branch.actor_state_versions],
                    "activated_event_ids": list(branch.activated_event_ids),
                    "emitted_claim_ids": list(branch.emitted_claim_ids),
                    "emitted_action_ids": list(branch.emitted_action_ids),
                }
                for branch in row.branches
            ]
            for trace_id, row in sorted(pending.items())
        }

    def _invalidate_stale_pending(self, *, tick: int, branch: BranchRef, consequence_ref: EventRef) -> None:
        invalidated = 0
        retained: dict[str, _PendingUpdate] = {}
        for trace_id, pending in self._pending_updates.items():
            live_branches: list[_BranchUpdate] = []
            for partition in pending.branches:
                stale = any(
                    self._catalog.actor(actor_id).state_version != expected
                    for actor_id, expected in partition.actor_state_versions
                )
                if stale:
                    invalidated += len(partition.active_actor_ids)
                else:
                    live_branches.append(partition)
            if live_branches:
                retained[trace_id] = _PendingUpdate(tuple(live_branches))
        self._pending_updates = retained
        if invalidated:
            self._ledger.charge(
                owner=self._owner,
                reason="stale-pending-authority-invalidation",
                work=WorkVector(learning=1 + invalidated),
                start_tick=tick,
                end_tick=tick,
                branch_id=branch,
                causal_event_ids=(consequence_ref,),
            )

    def run(self, event: DispatchEvent | None, *, now_tick: int | None = None) -> RuntimeTrace:

        if self._retention_finalized:
            raise RuntimeContractError("cannot run a finalized runtime")
        if self._active_episode_work is not None:
            raise RuntimeContractError("coalition runtime does not permit reentrant episodes")
        accounting_tick = (
            self._retention_last_tick
            if event is None and now_tick is None
            else event.header.created_tick
            if now_tick is None and event is not None
            else now_tick
        )
        assert accounting_tick is not None
        self._account_retention_until(
            accounting_tick,
            reason="runtime-retained-state-before-episode",
        )
        self._active_episode_work = 0
        self._active_deadline_ns = self._clock_ns() + self._config.caps.max_wall_time_ns
        try:
            return self._run_episode(event, now_tick=now_tick)
        finally:
            self._active_episode_work = None
            self._active_deadline_ns = None

    def _run_episode(self, event: DispatchEvent | None, *, now_tick: int | None = None) -> RuntimeTrace:

        ledger_start = self._ledger.entry_count
        if event is None:
            tick = self._retention_last_tick if now_tick is None else now_tick
            self._charge(
                reason="quiescent-empty-queue-check",
                work=WorkVector(idle_floor=1),
                tick=tick,
            )
            self._check_episode_bounds()
            return self._seal_trace(
                initial_event_id=None,
                rounds=(),
                deliveries=(),
                rejected_claims=(),
                actions=(),
                rejected_actions=(),
                active_ids=(),
                endogenous_rounds=0,
                quiescent=True,
                halt_reason="quiescent-empty-queue",
                ledger_start=ledger_start,
            )
        if event.header.endogenous or event.header.reasoning_depth != 0:
            raise RuntimeContractError("run requires one external depth-zero event")
        EventRef(event.header.event_id)
        BranchRef(event.header.branch_id)
        tick = event.header.created_tick if now_tick is None else now_tick
        if tick < 0:
            raise ValueError("now_tick must be nonnegative")
        if now_tick is not None and event.header.created_tick > now_tick:
            self._admit_dispatch_event(event, now_tick)
            raise RuntimeContractError("initial event is future-dated relative to now_tick")

        queue: deque[DispatchEvent] = deque((event,))
        rounds: list[RoundTrace] = []
        deliveries: list[MessageDelivery] = []
        rejected_claims: list[RejectedClaim] = []
        actions: list[ActionIntent] = []
        rejected_actions: list[RejectedAction] = []
        active_order: list[str] = []
        seen_active: set[str] = set()
        known_evidence: dict[str, EventClaimEvidence] = {}
        inbox: dict[str, list[_StagedClaim]] = defaultdict(list)
        branch_active: dict[str, list[str]] = defaultdict(list)
        branch_events: dict[str, list[str]] = defaultdict(list)
        message_edges = 0
        message_encoded_bytes = 0
        action_count = 0
        action_encoded_bytes = 0
        endogenous_payload_bytes = 0
        halt_reason = "quiescent"

        max_rounds = 1 + self._config.caps.R
        episode_tick = tick
        while queue and len(rounds) < max_rounds:
            current = queue.popleft()
            header = current.header
            current_tick = max(episode_tick, header.created_tick)
            episode_tick = current_tick
            self._admit_dispatch_event(current, current_tick)
            if header.epistemic_status is EpistemicStatus.SIMULATED and header.branch_id == str(
                FACTUAL_BRANCH
            ):
                raise RuntimeContractError("simulated event cannot enter the factual dispatch branch")
            if current_tick > header.expiry_tick:
                self._charge(
                    reason="expired-event-rejection",
                    work=WorkVector(dispatch_and_exploration=1),
                    tick=current_tick,
                    header=header,
                )
                halt_reason = "event-expired"
                continue
            state_cycle = (header.branch_id, header.producer_state_version, header.payload_digest)
            previous_states = {
                (
                    row.event_header.branch_id,
                    row.event_header.producer_state_version,
                    row.event_header.payload_digest,
                )
                for row in rounds
            }
            self._charge(
                reason="event-expiry-and-cycle-control",
                work=WorkVector(dispatch_and_exploration=2 + len(rounds)),
                tick=current_tick,
                header=header,
            )
            if state_cycle in previous_states:
                halt_reason = "repeated-state-cycle"
                queue.clear()
                break
            known_evidence[header.event_id] = EventClaimEvidence(
                header.event_id,
                header.event_kind,
                header.evidence_class,
                header.branch_id,
                header.epistemic_status,
                header.created_tick,
            )
            readiness = self._candidate_estimates(header, current_tick)
            decision = self._dispatch(header, readiness, len(rounds), current_tick)
            selected = decision.selected_actor_ids
            for actor_id in selected:
                if actor_id not in seen_active:
                    seen_active.add(actor_id)
                    active_order.append(actor_id)
                if actor_id not in branch_active[header.branch_id]:
                    branch_active[header.branch_id].append(actor_id)
            branch_events[header.branch_id].append(header.event_id)

            round_staged_message_ids: list[str] = []
            round_consumed_message_ids: list[str] = []
            round_action_ids: list[str] = []
            round_endogenous: list[str] = []
            state_versions: dict[str, str] = {}
            activation_outputs: list[tuple[str, ActorActivationResult]] = []
            for actor_id in selected:
                actor = self._catalog.actor(actor_id)
                incoming, consumed = self._consume_staged_claims(
                    actor_id=actor_id,
                    header=header,
                    tick=current_tick,
                    reasoning_round=len(rounds),
                    staged=tuple(inbox.pop(actor_id, ())),
                    known_evidence=known_evidence,
                    deliveries=deliveries,
                    rejected=rejected_claims,
                )
                round_consumed_message_ids.extend(consumed)
                context = ActorActivationContext(
                    event_header=header,
                    event_payload_bytes=current.payload_bytes,
                    incoming_claims=incoming,
                    reasoning_round=len(rounds),
                )
                before_state_version = actor.state_version
                try:
                    output = actor.activate(context)
                except Exception as exc:
                    self._charge(
                        reason=f"actor-activation-error:{actor_id}",
                        work=WorkVector(actor_execution=1),
                        tick=current_tick,
                        header=header,
                    )
                    if actor.state_version != before_state_version:
                        raise RuntimeContractError(
                            "failed actor activation mutated persistent state"
                        ) from exc
                    raise
                if not isinstance(output, ActorActivationResult):
                    self._charge(
                        reason=f"actor-activation-invalid-result:{actor_id}",
                        work=WorkVector(actor_execution=1),
                        tick=current_tick,
                        header=header,
                    )
                    raise RuntimeContractError("actor returned an invalid activation result")
                self._charge(
                    reason=f"actor-activation:{actor_id}",
                    work=WorkVector(actor_execution=output.executed_operations),
                    tick=current_tick,
                    header=header,
                )
                self._check_episode_bounds()
                if actor.state_version != before_state_version:
                    raise RuntimeContractError("actor activation mutated persistent state")
                state_versions[actor_id] = before_state_version
                if len(actor.state_version) != 64 or actor.state_version != actor.state_version.lower():
                    raise RuntimeContractError("actor activation produced an invalid state version")
                try:
                    int(actor.state_version, 16)
                except ValueError as exc:
                    raise RuntimeContractError("actor activation produced an invalid state version") from exc
                activation_outputs.append((actor_id, output))

            accepted_states = tuple((actor_id, (version,)) for actor_id, version in state_versions.items())
            claim_context = ClaimValidationContext(
                now_tick=current_tick,
                branch_id=header.branch_id,
                factual_branch_id=str(FACTUAL_BRANCH),
                allowed_referents=frozenset(header.referent_hypotheses),
                allowed_factor_scopes=frozenset(header.factor_scope),
                event_evidence=tuple(known_evidence.values()),
                accepted_producer_state_versions=accepted_states,
            )
            pending_endogenous: list[DispatchEvent] = []
            for actor_id, output in activation_outputs:
                if len(output.outbound_claims) > self._config.caps.M - message_edges:
                    self._charge(
                        reason=f"message-claim-count-cap-rejection:{actor_id}",
                        work=WorkVector(messages=1 + len(output.outbound_claims)),
                        tick=current_tick,
                        header=header,
                    )
                    raise RuntimeCapExceeded("outbound claim count exceeded remaining M")
                for outbound in output.outbound_claims:
                    if not isinstance(outbound, OutboundClaim):
                        self._charge(
                            reason=f"message-invalid-outbound:{actor_id}",
                            work=WorkVector(messages=1),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeContractError("actor returned an invalid outbound claim")
                    attempted = len(outbound.recipient_actor_ids)
                    if message_edges + attempted > self._config.caps.M:
                        self._charge(
                            reason=f"message-edge-cap-rejection:{actor_id}",
                            work=WorkVector(messages=1 + attempted),
                            tick=current_tick,
                            header=header,
                        )
                        self._check_episode_bounds()
                        raise RuntimeCapExceeded("directed message edges exceeded M")
                    message_header_items = (
                        len(outbound.message.header.source_hypothesis_event_ids)
                        + len(outbound.message.header.supporting_event_ids)
                        + len(outbound.message.header.referent_hypotheses)
                        + len(outbound.message.header.factor_scope)
                    )
                    if message_header_items > self._config.caps.max_message_header_items:
                        self._charge(
                            reason=f"message-header-cap-rejection:{actor_id}",
                            work=WorkVector(messages=1 + message_header_items),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeCapExceeded("message header items exceeded their cap")
                    raw_message_bytes = len(outbound.message.payload_bytes) * attempted
                    if (
                        message_encoded_bytes + raw_message_bytes
                        > self._config.caps.max_message_encoded_bytes
                    ):
                        self._charge(
                            reason=f"message-byte-cap-rejection:{actor_id}",
                            work=WorkVector(messages=1 + raw_message_bytes),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeCapExceeded("message encoded bytes exceeded their cap")
                    encoded_message_bytes = outbound.message.encoded_bytes
                    prospective_message_bytes = message_encoded_bytes + encoded_message_bytes * attempted
                    if prospective_message_bytes > self._config.caps.max_message_encoded_bytes:
                        self._charge(
                            reason=f"message-byte-cap-rejection:{actor_id}",
                            work=WorkVector(messages=1 + encoded_message_bytes * attempted),
                            tick=current_tick,
                            header=header,
                        )
                        self._check_episode_bounds()
                        raise RuntimeCapExceeded("message encoded bytes exceeded their cap")
                    message_encoded_bytes = prospective_message_bytes
                    message_edges += attempted
                    validation = validate_claim(
                        outbound.message,
                        schemas=self._schemas,
                        context=claim_context,
                    )
                    for recipient in outbound.recipient_actor_ids:
                        self._charge(
                            reason=f"message-edge:{actor_id}->{recipient}",
                            work=WorkVector(messages=1 + encoded_message_bytes),
                            tick=current_tick,
                            header=header,
                        )
                        extra_faults: list[ClaimFault | str] = []
                        if outbound.message.header.producer_actor_id != actor_id:
                            extra_faults.append("producer-actor-mismatch")
                        if outbound.message.header.producer_operations > output.executed_operations:
                            extra_faults.append("producer-operations-exceed-activation")
                        if recipient not in selected:
                            extra_faults.append("recipient-outside-coalition")
                        message_faults = tuple(validation.faults) + tuple(extra_faults)
                        if message_faults:
                            rejected_claims.append(
                                RejectedClaim(
                                    outbound.message.header.message_id,
                                    actor_id,
                                    recipient,
                                    message_faults,
                                    "emission",
                                )
                            )
                            continue
                        inbox[recipient].append(
                            _StagedClaim(
                                outbound.message,
                                recipient,
                                actor_id,
                                len(rounds),
                                header,
                            )
                        )
                        round_staged_message_ids.append(outbound.message.header.message_id)
                    self._check_episode_bounds()

                if action_count + len(output.action_intents) > self._config.caps.A:
                    self._charge(
                        reason=f"action-count-cap-rejection:{actor_id}",
                        work=WorkVector(actor_execution=1 + len(output.action_intents)),
                        tick=current_tick,
                        header=header,
                    )
                    raise RuntimeCapExceeded("action intent count exceeded A")
                for action in output.action_intents:
                    if not isinstance(action, ActionIntent):
                        self._charge(
                            reason=f"action-invalid-intent:{actor_id}",
                            work=WorkVector(actor_execution=1),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeContractError("actor returned an invalid action intent")
                    action_count += 1
                    if (
                        action_encoded_bytes + len(action.payload_bytes)
                        > self._config.caps.max_action_encoded_bytes
                    ):
                        self._charge(
                            reason=f"action-byte-cap-rejection:{actor_id}",
                            work=WorkVector(actor_execution=1 + len(action.payload_bytes)),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeCapExceeded("action encoded bytes exceeded their declared cap")
                    encoded_action_bytes = action.encoded_bytes
                    prospective_action_bytes = action_encoded_bytes + encoded_action_bytes
                    self._charge(
                        reason=f"action-intent:{actor_id}",
                        work=WorkVector(actor_execution=1 + encoded_action_bytes),
                        tick=current_tick,
                        header=header,
                    )
                    self._check_episode_bounds()
                    if prospective_action_bytes > self._config.caps.max_action_encoded_bytes:
                        raise RuntimeCapExceeded("action encoded bytes exceeded their declared cap")
                    action_encoded_bytes = prospective_action_bytes
                    faults = self._action_faults(
                        action,
                        actor_id=actor_id,
                        state_version=state_versions[actor_id],
                        header=header,
                        now_tick=current_tick,
                    )
                    if action.producer_operations > output.executed_operations:
                        faults = (*faults, ActionFault.PRODUCER_OPERATIONS)
                    if faults:
                        rejected_actions.append(RejectedAction(action.action_id, actor_id, faults))
                    else:
                        if any(row.branch_id == action.branch_id for row in actions):
                            raise RuntimeContractError(
                                "at most one accepted action may authorize each branch"
                            )
                        actions.append(action)
                        round_action_ids.append(action.action_id)

                proposals = output.endogenous_proposals
                if len(pending_endogenous) + len(proposals) > self._config.endogenous_fanout_cap:
                    self._charge(
                        reason=f"endogenous-fanout-cap-rejection:{actor_id}",
                        work=WorkVector(event_formation=1 + len(proposals)),
                        tick=current_tick,
                        header=header,
                    )
                    self._check_episode_bounds()
                    raise RuntimeCapExceeded("coalition exceeded the endogenous fanout cap")
                if len(queue) + len(pending_endogenous) + len(proposals) > self._config.queue_cap:
                    self._charge(
                        reason=f"endogenous-queue-cap-rejection:{actor_id}",
                        work=WorkVector(event_formation=1 + len(proposals)),
                        tick=current_tick,
                        header=header,
                    )
                    raise RuntimeCapExceeded("endogenous queue exceeded its declared capacity")
                if proposals and self._event_ledger is None:
                    self._charge(
                        reason=f"endogenous-missing-ledger-rejection:{actor_id}",
                        work=WorkVector(event_formation=1 + len(proposals)),
                        tick=current_tick,
                        header=header,
                    )
                    raise RuntimeContractError("endogenous proposals require an injected EventLedger")
                for proposal in proposals:
                    if not isinstance(proposal, EndogenousHypothesisProposal):
                        self._charge(
                            reason=f"endogenous-invalid-proposal:{actor_id}",
                            work=WorkVector(event_formation=1),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeContractError("actor returned an invalid endogenous proposal")
                    prospective_payload_bytes = endogenous_payload_bytes + len(proposal.payload_bytes)
                    if prospective_payload_bytes > self._config.caps.max_endogenous_payload_bytes:
                        self._charge(
                            reason=f"endogenous-payload-cap-rejection:{actor_id}",
                            work=WorkVector(event_formation=1 + len(proposal.payload_bytes)),
                            tick=current_tick,
                            header=header,
                        )
                        raise RuntimeCapExceeded("endogenous payload bytes exceeded their declared cap")
                    endogenous_payload_bytes = prospective_payload_bytes
                    proposal_counts = (
                        (
                            len(proposal.referent_hypotheses),
                            self._config.caps.max_header_referents,
                            "referents",
                        ),
                        (
                            len(proposal.factor_scope),
                            self._config.caps.max_header_factors,
                            "factor scopes",
                        ),
                        (
                            len(proposal.routing_shards),
                            self._config.caps.max_header_shards,
                            "routing shards",
                        ),
                        (
                            len(proposal.source_event_ids),
                            self._config.caps.max_header_source_events,
                            "source events",
                        ),
                    )
                    for actual, maximum, label in proposal_counts:
                        if actual > maximum:
                            self._charge(
                                reason=(f"endogenous-{label.replace(' ', '-')}-cap-rejection:{actor_id}"),
                                work=WorkVector(event_formation=1 + actual),
                                tick=current_tick,
                                header=header,
                            )
                            raise RuntimeCapExceeded(f"endogenous proposal {label} exceeded its declared cap")
                    self._charge(
                        reason=f"endogenous-proposal-admission:{actor_id}",
                        work=WorkVector(event_formation=1 + proposal.encoded_bytes),
                        tick=current_tick,
                        header=header,
                    )
                    self._check_episode_bounds()
                    if proposal.producer_operations > output.executed_operations:
                        raise RuntimeContractError(
                            "endogenous proposal operations exceed actor activation work"
                        )
                    self._validate_endogenous_proposal(
                        proposal,
                        parent=header,
                        known_evidence=known_evidence,
                    )
                    generated = self._append_endogenous_proposal(
                        proposal,
                        actor_id=actor_id,
                        actor_state_version=state_versions[actor_id],
                        parent=header,
                        reasoning_depth=header.reasoning_depth + 1,
                    )
                    pending_endogenous.append(generated)
                    round_endogenous.append(generated.header.event_id)
            if len(queue) + len(pending_endogenous) > self._config.queue_cap:
                raise RuntimeCapExceeded("endogenous queue exceeded its declared capacity")
            queue.extend(pending_endogenous)
            rounds.append(
                RoundTrace(
                    event_header=header,
                    candidate_actor_ids=tuple(row.actor_id for row in readiness),
                    selected_actor_ids=selected,
                    considered_coalitions=decision.considered_coalitions,
                    staged_message_ids=tuple(round_staged_message_ids),
                    consumed_message_ids=tuple(round_consumed_message_ids),
                    accepted_action_ids=tuple(round_action_ids),
                    admitted_endogenous_event_ids=tuple(round_endogenous),
                )
            )
            self._check_episode_bounds()

        if queue:
            halt_reason = "endogenous-round-cap"
        for recipient, staged_rows in inbox.items():
            for row in staged_rows:
                self._charge(
                    reason=f"message-next-round-drop:{row.producer_actor_id}->{recipient}",
                    work=WorkVector(messages=1),
                    tick=max(tick, row.message.header.created_tick),
                    header=row.source_header,
                )
                rejected_claims.append(
                    RejectedClaim(
                        row.message.header.message_id,
                        row.producer_actor_id,
                        recipient,
                        ("no-authorized-subsequent-consumption",),
                        "unconsumed",
                    )
                )
        self._check_episode_bounds()
        quiescent = not queue
        trace = self._seal_trace(
            initial_event_id=event.header.event_id,
            rounds=tuple(rounds),
            deliveries=tuple(deliveries),
            rejected_claims=tuple(rejected_claims),
            actions=tuple(actions),
            rejected_actions=tuple(rejected_actions),
            active_ids=tuple(active_order),
            endogenous_rounds=max(0, len(rounds) - 1),
            quiescent=quiescent,
            halt_reason=halt_reason,
            ledger_start=ledger_start,
        )
        branch_updates: list[_BranchUpdate] = []
        for branch_id in sorted(branch_active):
            if branch_id != str(FACTUAL_BRANCH):
                continue
            branch_action_ids = tuple(
                sorted(action.action_id for action in trace.action_intents if action.branch_id == branch_id)
            )
            if not branch_action_ids:
                continue
            actor_ids = tuple(branch_active[branch_id])
            branch_updates.append(
                _BranchUpdate(
                    branch_id=branch_id,
                    active_actor_ids=actor_ids,
                    actor_state_versions=tuple(
                        (actor_id, self._catalog.actor(actor_id).state_version) for actor_id in actor_ids
                    ),
                    activated_event_ids=tuple(sorted(branch_events[branch_id])),
                    emitted_claim_ids=tuple(
                        sorted(
                            {
                                row.message.header.message_id
                                for row in trace.message_deliveries
                                if row.message.header.branch_id == branch_id
                            }
                        )
                    ),
                    emitted_action_ids=branch_action_ids,
                )
            )
        if branch_updates:
            attempted_pending = sum(len(row.active_actor_ids) for row in branch_updates)
            current_pending = sum(
                len(branch.active_actor_ids)
                for pending in self._pending_updates.values()
                for branch in pending.branches
            )
            if current_pending + attempted_pending > self._config.caps.P:
                self._charge(
                    reason="pending-update-count-cap-rejection",
                    work=WorkVector(learning=1 + attempted_pending),
                    tick=tick,
                    header=event.header,
                )
                raise RuntimeCapExceeded("pending update authorities exceeded P")
            proposed_pending = dict(self._pending_updates)
            proposed_pending[trace.trace_id] = _PendingUpdate(tuple(branch_updates))
            pending_bytes = len(canonical_bytes(self._pending_payload(proposed_pending)))
            self._charge(
                reason="pending-update-authority-creation",
                work=WorkVector(learning=1 + attempted_pending + pending_bytes),
                tick=tick,
                header=event.header,
            )
            self._check_episode_bounds()
            if pending_bytes > self._config.caps.max_pending_update_bytes:
                raise RuntimeCapExceeded("pending update authority bytes exceeded their cap")
            self._pending_updates = proposed_pending
        return trace

    def _seal_trace(
        self,
        *,
        initial_event_id: str | None,
        rounds: tuple[RoundTrace, ...],
        deliveries: tuple[MessageDelivery, ...],
        rejected_claims: tuple[RejectedClaim, ...],
        actions: tuple[ActionIntent, ...],
        rejected_actions: tuple[RejectedAction, ...],
        active_ids: tuple[str, ...],
        endogenous_rounds: int,
        quiescent: bool,
        halt_reason: str,
        ledger_start: int,
    ) -> RuntimeTrace:
        authority_sequence = self._trace_sequence
        trace_id = canonical_sha256(
            {
                "schema": "mop-escs-runtime-trace-authority/v1",
                "runtime_id": self._runtime_id,
                "authority_sequence": authority_sequence,
            }
        )
        unsealed = RuntimeTrace(
            trace_id=trace_id,
            runtime_id=self._runtime_id,
            authority_sequence=authority_sequence,
            mode=self._config.mode,
            caps=self._config.caps,
            initial_event_id=initial_event_id,
            rounds=rounds,
            message_deliveries=deliveries,
            rejected_claims=rejected_claims,
            action_intents=actions,
            rejected_actions=rejected_actions,
            active_actor_ids=active_ids,
            endogenous_rounds=endogenous_rounds,
            quiescent=quiescent,
            halt_reason=halt_reason,
            ledger_start_sequence=ledger_start,
            ledger_end_sequence=self._ledger.entry_count,
            full_trace_sha256="0" * 64,
        )
        sealed = replace(
            unsealed,
            full_trace_sha256=canonical_sha256(unsealed.payload(include_digest=False)),
        )
        if not sealed.validate_integrity():
            raise RuntimeContractError("runtime trace failed sealing integrity validation")
        self._trace_sequence += 1
        return sealed

    def apply_consequence(
        self,
        *,
        trace_id: str,
        consequence_event_id: str,
        authorization_id: str,
        branch_id: str,
        consequence_payload_bytes: bytes,
        tick: int,
    ) -> tuple[str, ...]:

        consequence_ref = EventRef(consequence_event_id)
        branch = BranchRef(branch_id)
        if not isinstance(consequence_payload_bytes, bytes):
            raise TypeError("consequence payload must be bytes")
        if tick < 0:
            raise ValueError("consequence tick must be nonnegative")
        if self._retention_finalized:
            raise RuntimeContractError("cannot update a finalized runtime")
        self._account_retention_until(
            tick,
            reason="runtime-retained-state-before-consequence",
        )
        if len(authorization_id) != 64 or any(ch not in "0123456789abcdef" for ch in authorization_id):
            raise ValueError("authorization_id must be a lowercase SHA-256 digest")
        self._ledger.charge(
            owner=self._owner,
            reason="consequence-update-dispatch",
            work=WorkVector(learning=1 + len(consequence_payload_bytes)),
            start_tick=tick,
            end_tick=tick,
            branch_id=branch,
            causal_event_ids=(consequence_ref,),
        )
        if len(consequence_payload_bytes) > self._config.caps.max_event_payload_bytes:
            raise RuntimeCapExceeded("consequence payload bytes exceeded their declared cap")
        pending = self._pending_updates.get(trace_id)
        if pending is None:
            raise RuntimeContractError("unknown, idle, or already-consumed trace")
        if consequence_event_id in self._consumed_consequence_ids:
            raise RuntimeContractError("consequence event was already consumed")
        if authorization_id in self._consumed_authorization_ids:
            raise RuntimeContractError("action authorization was already consumed")
        matching = tuple(row for row in pending.branches if row.branch_id == branch_id)
        if len(matching) != 1:
            raise RuntimeContractError("consequence branch has no pending authority in this trace")
        partition = matching[0]
        if authorization_id not in partition.emitted_action_ids:
            raise RuntimeContractError("consequence is not bound to an accepted action authorization")
        expected_versions = dict(partition.actor_state_versions)
        for actor_id in partition.active_actor_ids:
            if self._catalog.actor(actor_id).state_version != expected_versions[actor_id]:
                raise RuntimeContractError("pending actor state version is stale")

        if (
            len(self._consumed_consequence_ids) + 1 > self._config.caps.max_consumed_authorities
            or len(self._consumed_authorization_ids) + 1 > self._config.caps.max_consumed_authorities
        ):
            raise RuntimeCapExceeded("consumed consequence authority count exceeded its cap")
        update_context = ActorUpdateContext(
            trace_id=trace_id,
            activated_event_ids=partition.activated_event_ids,
            emitted_claim_ids=partition.emitted_claim_ids,
            emitted_action_ids=partition.emitted_action_ids,
            consequence_event_id=consequence_event_id,
            authorization_id=authorization_id,
            branch_id=branch_id,
            consequence_payload_bytes=consequence_payload_bytes,
        )
        plans: list[ActorUpdatePlan] = []
        replacements: dict[str, Actor] = {}
        attempted_work = 1 + len(consequence_payload_bytes)
        deadline_ns = self._clock_ns() + self._config.caps.max_wall_time_ns
        for actor_id in partition.active_actor_ids:
            actor = self._catalog.actor(actor_id)
            prior_version = actor.state_version
            try:
                plan = actor.stage_update(update_context)
            except Exception as exc:
                self._ledger.charge(
                    owner=self._owner,
                    reason=f"active-actor-update-plan-error:{actor_id}",
                    work=WorkVector(learning=1),
                    start_tick=tick,
                    end_tick=tick,
                    branch_id=branch,
                    causal_event_ids=(consequence_ref,),
                )
                if actor.state_version != prior_version:
                    raise RuntimeContractError("failed update plan mutated the live actor") from exc
                raise
            if not isinstance(plan, ActorUpdatePlan):
                self._ledger.charge(
                    owner=self._owner,
                    reason=f"active-actor-update-invalid-plan:{actor_id}",
                    work=WorkVector(learning=1),
                    start_tick=tick,
                    end_tick=tick,
                    branch_id=branch,
                    causal_event_ids=(consequence_ref,),
                )
                raise RuntimeContractError("actor returned an invalid update plan")
            self._ledger.charge(
                owner=self._owner,
                reason=f"active-actor-update-plan:{actor_id}",
                work=WorkVector(learning=plan.executed_operations),
                start_tick=tick,
                end_tick=tick,
                branch_id=branch,
                causal_event_ids=(consequence_ref,),
            )
            attempted_work += plan.executed_operations
            if attempted_work > self._config.caps.max_episode_work:
                raise RuntimeCapExceeded("atomic update work exceeded its declared cap")
            if self._clock_ns() > deadline_ns:
                raise RuntimeCapExceeded("atomic update wall-time budget was exhausted")
            if actor.state_version != prior_version:
                raise RuntimeContractError("update planning mutated the live actor")
            replacement = plan.replacement_actor
            if not isinstance(replacement, Actor):
                raise RuntimeContractError("update plan replacement does not implement Actor")
            if (
                plan.actor_id != actor_id
                or plan.prior_state_version != prior_version
                or plan.idempotency_key != update_context.idempotency_key
                or replacement.descriptor != actor.descriptor
                or replacement.state_version != plan.next_state_version
            ):
                raise RuntimeContractError("update plan identity or state binding mismatch")
            plans.append(plan)
            replacements[actor_id] = replacement

        retained_after = sum(
            replacements.get(actor_id, self._catalog.actor(actor_id)).retained_state_bytes
            for actor_id in self._catalog.actor_ids
        )
        if retained_after > self._config.caps.max_actor_retained_state_bytes:
            raise RuntimeCapExceeded("replacement actors exceed the retained-state byte cap")
        proposed_consequences = self._consumed_consequence_ids | {consequence_event_id}
        proposed_authorizations = self._consumed_authorization_ids | {authorization_id}
        consumed_bytes = len(
            canonical_bytes(
                {
                    "consequence_event_ids": sorted(proposed_consequences),
                    "authorization_ids": sorted(proposed_authorizations),
                }
            )
        )
        if consumed_bytes > self._config.caps.max_consumed_authority_bytes:
            raise RuntimeCapExceeded("consumed consequence authority bytes exceeded their cap")

        self._catalog.replace_many(replacements)
        remaining = tuple(row for row in pending.branches if row.branch_id != branch_id)
        if remaining:
            self._pending_updates[trace_id] = _PendingUpdate(remaining)
        else:
            del self._pending_updates[trace_id]
        self._invalidate_stale_pending(
            tick=tick,
            branch=branch,
            consequence_ref=consequence_ref,
        )
        self._consumed_consequence_ids = proposed_consequences
        self._consumed_authorization_ids = proposed_authorizations
        return tuple(plan.actor_id for plan in plans)


__all__ = [
    "ActionFault",
    "CandidateMode",
    "CoalitionRuntime",
    "DispatchDecision",
    "DispatchPolicy",
    "DispatchRequest",
    "MessageDelivery",
    "RejectedAction",
    "RejectedClaim",
    "RoundTrace",
    "RuntimeCapExceeded",
    "RuntimeCaps",
    "RuntimeConfig",
    "RuntimeContractError",
    "RuntimeTrace",
    "ScriptedDispatchPolicy",
]
