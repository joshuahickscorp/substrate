"""Immutable four-stage event records for the Event-Sourced Coalition Substrate.

This is evidence-neutral mechanics.  The records enforce temporal, branch, epistemic, and payload
boundaries; they do not decide which observations are meaningful or whether an actor is useful.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

from mop.substrate.events import (
    BranchRef,
    EventRef,
    FrozenJSON,
    canonical_bytes,
    canonical_sha256,
)

from .accounting import FACTUAL_BRANCH, WorkVector

EVENT_SCHEMA = "mop-escs-event/v1"
EVENT_ENVELOPE_SCHEMA = "mop-escs-event-envelope/v1"
STATE_VERSION_SCHEMA = "mop-escs-causal-state-version/v1"
GENESIS_STATE_VERSION = canonical_sha256({"schema": EVENT_SCHEMA, "state": "genesis"})

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_REF_RE = re.compile(r"^[a-z][a-z0-9+.-]*:[a-z0-9][a-z0-9._:/-]*$")
_HIDDEN_TRUTH_KEYS = frozenset(
    {
        "canonical_referent_truth",
        "evaluator_label",
        "ground_truth",
        "ground_truth_label",
        "hidden_change_point",
        "oracle_label",
    }
)
_OBSERVATION_FUTURE_KEYS = frozenset(
    {
        "commitment_event_id",
        "committed_payload",
        "decision_distribution",
        "future_outcome",
        "observed_outcome",
        "predicted_utility_vector",
        "realized_full_cost",
        "realized_utility_vector",
    }
)
_HYPOTHESIS_FUTURE_KEYS = frozenset(
    {"commitment_event_id", "committed_payload", "observed_outcome", "realized_full_cost"}
)
_COMMITMENT_FUTURE_KEYS = frozenset({"observed_outcome", "realized_full_cost", "realized_utility_vector"})


class EventKind(StrEnum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    COMMITMENT = "commitment"
    CONSEQUENCE = "consequence"


class EvidenceClass(StrEnum):
    """Transitive evidence taint, ordered from least to most promotion-restrictive."""

    LEARNED_UNVERIFIED = "learned-unverified"
    SCRIPTED_MECHANICS = "scripted-mechanics-only"
    ORACLE_NONPROMOTABLE = "oracle-nonpromotable"

    @property
    def taint_rank(self) -> int:
        return {
            EvidenceClass.LEARNED_UNVERIFIED: 0,
            EvidenceClass.SCRIPTED_MECHANICS: 1,
            EvidenceClass.ORACLE_NONPROMOTABLE: 2,
        }[self]


class HypothesisOrigin(StrEnum):
    EVENT_FORMER = "event_former"
    ACTOR = "actor"
    VERIFIER = "verifier"


class EpistemicStatus(StrEnum):
    OBSERVED_CANDIDATE = "observed_candidate"
    INFERRED = "inferred"
    SIMULATED = "simulated"


class CommitmentKind(StrEnum):
    EXTERNAL_ACTION = "external_action"
    ABSTENTION = "abstention"
    MEMORY_WRITE = "memory_write"
    DELETION = "deletion"
    TOPOLOGY_TRANSACTION = "topology_transaction"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_stable_ref(value: object, label: str) -> str:
    if not isinstance(value, str) or _STABLE_REF_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a stable namespaced reference")
    return value


def _freeze(value: Any) -> FrozenJSON:
    return value if isinstance(value, FrozenJSON) else FrozenJSON.from_value(value)


def _frozen_from_payload(payload: Mapping[str, Any], label: str) -> FrozenJSON:
    _require_exact_keys(payload, {"value", "sha256"}, label)
    frozen = FrozenJSON.from_value(payload["value"])
    if payload["sha256"] != frozen.sha256:
        raise ValueError(f"{label} digest mismatch")
    return frozen


def _require_mapping_payload(value: FrozenJSON, label: str) -> None:
    if not isinstance(value.value(), dict):
        raise ValueError(f"{label} must contain a JSON object")


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_walk_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_walk_keys(nested))
        return keys
    return set()


def _reject_keys(value: FrozenJSON, forbidden: frozenset[str], label: str) -> None:
    found = sorted(_walk_keys(value.value()) & forbidden)
    if found:
        raise ValueError(f"{label} contains forbidden future or evaluator fields: {found}")


def _canonical_event_refs(values: Sequence[EventRef], label: str) -> tuple[EventRef, ...]:
    rows = tuple(values)
    if not all(isinstance(value, EventRef) for value in rows):
        raise ValueError(f"{label} must contain EventRef values")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{label} must be unique")
    if rows != tuple(sorted(rows, key=str)):
        raise ValueError(f"{label} must use canonical sorted order")
    return rows


def state_version_for_parents(causal_parent_ids: Sequence[EventRef]) -> str:
    """Bind a derived producer-state version to its complete canonical parent set.

    Parentless roots share the explicit genesis version.  Every derived version is otherwise a
    commitment to the entire sorted set, so omitting, replacing, or duplicating a parent changes or
    invalidates the version before the event can enter a ledger.
    """

    parents = tuple(sorted(causal_parent_ids, key=str))
    _canonical_event_refs(parents, "causal_parent_ids")
    if not parents:
        return GENESIS_STATE_VERSION
    return canonical_sha256(
        {
            "schema": STATE_VERSION_SCHEMA,
            "causal_parent_ids": [str(event_id) for event_id in parents],
        }
    )


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: EventRef
    event_kind: EventKind
    evidence_class: EvidenceClass
    causal_parent_ids: tuple[EventRef, ...]
    counterfactual_branch_id: BranchRef
    clock_start_tick: int
    clock_end_tick: int
    clock_uncertainty: int
    source_and_provenance: FrozenJSON
    payload_digest: str
    producer_state_version: str
    measured_creation_cost: WorkVector
    supersedes_event_ids: tuple[EventRef, ...] = ()
    schema: str = EVENT_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_ENVELOPE_SCHEMA:
            raise ValueError(f"unsupported event envelope schema {self.schema!r}")
        if not isinstance(self.event_kind, EventKind):
            raise ValueError("event_kind must be an EventKind")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("evidence_class must be an EvidenceClass")
        if not isinstance(self.event_id, EventRef):
            raise ValueError("event_id must be an EventRef")
        if not isinstance(self.counterfactual_branch_id, BranchRef):
            raise ValueError("counterfactual_branch_id must be a BranchRef")
        if not isinstance(self.causal_parent_ids, tuple):
            raise ValueError("causal_parent_ids must be an immutable tuple")
        if not isinstance(self.supersedes_event_ids, tuple):
            raise ValueError("supersedes_event_ids must be an immutable tuple")
        _canonical_event_refs(self.causal_parent_ids, "causal_parent_ids")
        _canonical_event_refs(self.supersedes_event_ids, "supersedes_event_ids")
        if not set(self.supersedes_event_ids) <= set(self.causal_parent_ids):
            raise ValueError("superseded events must also be causal parents")
        _require_nonnegative_int(self.clock_start_tick, "clock_start_tick")
        _require_nonnegative_int(self.clock_end_tick, "clock_end_tick")
        _require_nonnegative_int(self.clock_uncertainty, "clock_uncertainty")
        if self.clock_end_tick < self.clock_start_tick:
            raise ValueError("event clock interval is invalid")
        if not isinstance(self.source_and_provenance, FrozenJSON):
            raise ValueError("source_and_provenance must be FrozenJSON")
        _require_mapping_payload(self.source_and_provenance, "source_and_provenance")
        _reject_keys(self.source_and_provenance, _HIDDEN_TRUTH_KEYS, "source_and_provenance")
        _require_digest(self.payload_digest, "payload_digest")
        _require_digest(self.producer_state_version, "producer_state_version")
        expected_state_version = state_version_for_parents(self.causal_parent_ids)
        if self.producer_state_version != expected_state_version:
            raise ValueError("producer_state_version does not bind the complete causal-parent set")
        if not isinstance(self.measured_creation_cost, WorkVector):
            raise ValueError("measured_creation_cost must be a WorkVector")
        expected = EventRef(f"event:{canonical_sha256(self.identity_payload())}")
        if self.event_id != expected:
            raise ValueError("event identity digest mismatch")
        if self.event_id in self.causal_parent_ids:
            raise ValueError("an event cannot be its own causal parent")

    @classmethod
    def create(
        cls,
        *,
        event_kind: EventKind,
        causal_parent_ids: Sequence[EventRef],
        counterfactual_branch_id: BranchRef,
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int,
        source_and_provenance: FrozenJSON | Any,
        payload_digest: str,
        producer_state_version: str | None,
        measured_creation_cost: WorkVector,
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        supersedes_event_ids: Sequence[EventRef] = (),
    ) -> EventEnvelope:
        parents = tuple(sorted(causal_parent_ids, key=str))
        supersedes = tuple(sorted(supersedes_event_ids, key=str))
        source = _freeze(source_and_provenance)
        state_version = (
            state_version_for_parents(parents) if producer_state_version is None else producer_state_version
        )
        partial = {
            "schema": EVENT_ENVELOPE_SCHEMA,
            "event_kind": event_kind.value,
            "evidence_class": evidence_class.value,
            "causal_parent_ids": [str(event_id) for event_id in parents],
            "counterfactual_branch_id": str(counterfactual_branch_id),
            "clock_interval": [clock_start_tick, clock_end_tick],
            "clock_uncertainty": clock_uncertainty,
            "source_and_provenance": source.payload(),
            "payload_digest": payload_digest,
            "producer_state_version": state_version,
            "measured_creation_cost": measured_creation_cost.payload(),
            "supersedes_event_ids": [str(event_id) for event_id in supersedes],
        }
        return cls(
            event_id=EventRef(f"event:{canonical_sha256(partial)}"),
            event_kind=event_kind,
            evidence_class=evidence_class,
            causal_parent_ids=parents,
            counterfactual_branch_id=counterfactual_branch_id,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
            source_and_provenance=source,
            payload_digest=payload_digest,
            producer_state_version=state_version,
            measured_creation_cost=measured_creation_cost,
            supersedes_event_ids=supersedes,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EventEnvelope:
        expected = {
            "schema",
            "event_id",
            "event_kind",
            "evidence_class",
            "causal_parent_ids",
            "counterfactual_branch_id",
            "clock_interval",
            "clock_uncertainty",
            "source_and_provenance",
            "payload_digest",
            "producer_state_version",
            "measured_creation_cost",
            "supersedes_event_ids",
        }
        _require_exact_keys(payload, expected, "EventEnvelope")
        interval = payload["clock_interval"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("clock_interval must contain exactly two ticks")
        parents = payload["causal_parent_ids"]
        supersedes = payload["supersedes_event_ids"]
        if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
            raise ValueError("causal_parent_ids must be a list of strings")
        if not isinstance(supersedes, list) or not all(isinstance(value, str) for value in supersedes):
            raise ValueError("supersedes_event_ids must be a list of strings")
        return cls(
            event_id=EventRef(payload["event_id"]),
            event_kind=EventKind(payload["event_kind"]),
            evidence_class=EvidenceClass(payload["evidence_class"]),
            causal_parent_ids=tuple(EventRef(value) for value in parents),
            counterfactual_branch_id=BranchRef(payload["counterfactual_branch_id"]),
            clock_start_tick=interval[0],
            clock_end_tick=interval[1],
            clock_uncertainty=payload["clock_uncertainty"],
            source_and_provenance=_frozen_from_payload(
                payload["source_and_provenance"], "source_and_provenance"
            ),
            payload_digest=payload["payload_digest"],
            producer_state_version=payload["producer_state_version"],
            measured_creation_cost=WorkVector.from_payload(payload["measured_creation_cost"]),
            supersedes_event_ids=tuple(EventRef(value) for value in supersedes),
            schema=payload["schema"],
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_kind": self.event_kind.value,
            "evidence_class": self.evidence_class.value,
            "causal_parent_ids": [str(event_id) for event_id in self.causal_parent_ids],
            "counterfactual_branch_id": str(self.counterfactual_branch_id),
            "clock_interval": [self.clock_start_tick, self.clock_end_tick],
            "clock_uncertainty": self.clock_uncertainty,
            "source_and_provenance": self.source_and_provenance.payload(),
            "payload_digest": self.payload_digest,
            "producer_state_version": self.producer_state_version,
            "measured_creation_cost": self.measured_creation_cost.payload(),
            "supersedes_event_ids": [str(event_id) for event_id in self.supersedes_event_ids],
        }

    def payload(self) -> dict[str, Any]:
        return {"event_id": str(self.event_id), **self.identity_payload()}


class _EventRecord:
    envelope: EventEnvelope

    @property
    def event_id(self) -> EventRef:
        return self.envelope.event_id

    @property
    def kind(self) -> EventKind:
        return self.envelope.event_kind

    @property
    def branch_id(self) -> BranchRef:
        return self.envelope.counterfactual_branch_id

    @property
    def evidence_class(self) -> EvidenceClass:
        return self.envelope.evidence_class

    @property
    def sha256(self) -> str:
        return str(self.event_id).removeprefix("event:")

    def body_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def payload(self) -> dict[str, Any]:
        return {
            "schema": EVENT_SCHEMA,
            "envelope": self.envelope.payload(),
            "body": self.body_payload(),
        }

    def _validate_envelope(self, kind: EventKind) -> None:
        if self.envelope.event_kind is not kind:
            raise ValueError(f"{type(self).__name__} requires a {kind.value} envelope")
        if canonical_sha256(self.body_payload()) != self.envelope.payload_digest:
            raise ValueError(f"{type(self).__name__} payload digest mismatch")


@dataclass(frozen=True, slots=True)
class ObservationEvent(_EventRecord):
    envelope: EventEnvelope
    raw_packet_or_delta_refs: tuple[str, ...]
    adapter_version: str
    sensor_scope: FrozenJSON
    transport_and_detection_cost: WorkVector

    def __post_init__(self) -> None:
        self._validate_envelope(EventKind.OBSERVATION)
        if self.branch_id != FACTUAL_BRANCH:
            raise ValueError("ObservationEvent must remain on the factual branch")
        if not isinstance(self.raw_packet_or_delta_refs, tuple):
            raise ValueError("raw packet or delta references must be an immutable tuple")
        if not self.raw_packet_or_delta_refs:
            raise ValueError("ObservationEvent requires at least one raw packet or delta reference")
        for value in self.raw_packet_or_delta_refs:
            _require_stable_ref(value, "raw_packet_or_delta_ref")
        if len(set(self.raw_packet_or_delta_refs)) != len(self.raw_packet_or_delta_refs):
            raise ValueError("raw packet or delta references must be unique")
        if tuple(sorted(self.raw_packet_or_delta_refs)) != self.raw_packet_or_delta_refs:
            raise ValueError("raw packet or delta references must use canonical sorted order")
        if not isinstance(self.adapter_version, str) or not self.adapter_version.strip():
            raise ValueError("adapter_version must not be empty")
        _require_mapping_payload(self.sensor_scope, "sensor_scope")
        _reject_keys(
            self.sensor_scope,
            _HIDDEN_TRUTH_KEYS | _OBSERVATION_FUTURE_KEYS,
            "ObservationEvent.sensor_scope",
        )
        if not isinstance(self.transport_and_detection_cost, WorkVector):
            raise ValueError("transport_and_detection_cost must be a WorkVector")

    @classmethod
    def create(
        cls,
        *,
        raw_packet_or_delta_refs: Sequence[str],
        adapter_version: str,
        sensor_scope: FrozenJSON | Any,
        transport_and_detection_cost: WorkVector,
        causal_parent_ids: Sequence[EventRef] = (),
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int = 0,
        source_and_provenance: FrozenJSON | Any,
        producer_state_version: str | None = None,
        measured_creation_cost: WorkVector = WorkVector(),
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        supersedes_event_ids: Sequence[EventRef] = (),
    ) -> ObservationEvent:
        refs = tuple(sorted(raw_packet_or_delta_refs))
        scope = _freeze(sensor_scope)
        body = _observation_body(refs, adapter_version, scope, transport_and_detection_cost)
        envelope = EventEnvelope.create(
            event_kind=EventKind.OBSERVATION,
            causal_parent_ids=causal_parent_ids,
            counterfactual_branch_id=FACTUAL_BRANCH,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
            source_and_provenance=source_and_provenance,
            payload_digest=canonical_sha256(body),
            producer_state_version=producer_state_version,
            measured_creation_cost=measured_creation_cost,
            evidence_class=evidence_class,
            supersedes_event_ids=supersedes_event_ids,
        )
        return cls(envelope, refs, adapter_version, scope, transport_and_detection_cost)

    def body_payload(self) -> dict[str, Any]:
        return _observation_body(
            self.raw_packet_or_delta_refs,
            self.adapter_version,
            self.sensor_scope,
            self.transport_and_detection_cost,
        )


def _observation_body(
    refs: tuple[str, ...], adapter_version: str, scope: FrozenJSON, cost: WorkVector
) -> dict[str, Any]:
    return {
        "raw_packet_or_delta_refs": list(refs),
        "adapter_version": adapter_version,
        "sensor_scope": scope.payload(),
        "transport_and_detection_cost": cost.payload(),
    }


@dataclass(frozen=True, slots=True)
class HypothesisEvent(_EventRecord):
    envelope: EventEnvelope
    origin: HypothesisOrigin
    epistemic_status: EpistemicStatus
    referent_hypotheses: FrozenJSON
    factor_change_distribution: FrozenJSON
    decision_relevance_distribution: FrozenJSON
    reducibility_distribution: FrozenJSON
    supporting_event_ids: tuple[EventRef, ...]
    calibrated_confidence: float
    abstention_reason: str | None
    predicted_value_of_further_computation: float

    def __post_init__(self) -> None:
        self._validate_envelope(EventKind.HYPOTHESIS)
        if not isinstance(self.origin, HypothesisOrigin):
            raise ValueError("origin must be a HypothesisOrigin")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("epistemic_status must be an EpistemicStatus")
        if self.epistemic_status is EpistemicStatus.SIMULATED:
            if self.branch_id == FACTUAL_BRANCH:
                raise ValueError("simulated hypotheses require an explicit counterfactual branch")
        elif self.branch_id != FACTUAL_BRANCH:
            raise ValueError("nonfactual hypotheses must retain simulated epistemic status")
        for label, value in (
            ("referent_hypotheses", self.referent_hypotheses),
            ("factor_change_distribution", self.factor_change_distribution),
            ("decision_relevance_distribution", self.decision_relevance_distribution),
            ("reducibility_distribution", self.reducibility_distribution),
        ):
            if not isinstance(value, FrozenJSON):
                raise ValueError(f"{label} must be FrozenJSON")
            _reject_keys(value, _HIDDEN_TRUTH_KEYS | _HYPOTHESIS_FUTURE_KEYS, label)
        if not isinstance(self.supporting_event_ids, tuple):
            raise ValueError("supporting_event_ids must be an immutable tuple")
        _canonical_event_refs(self.supporting_event_ids, "supporting_event_ids")
        if not set(self.supporting_event_ids) <= set(self.envelope.causal_parent_ids):
            raise ValueError("supporting events must be declared causal parents")
        confidence = _require_nonnegative_number(self.calibrated_confidence, "calibrated_confidence")
        if confidence > 1:
            raise ValueError("calibrated_confidence cannot exceed one")
        if self.abstention_reason is not None and (
            not isinstance(self.abstention_reason, str) or not self.abstention_reason.strip()
        ):
            raise ValueError("abstention_reason must be None or a nonempty string")
        _require_nonnegative_number(
            self.predicted_value_of_further_computation,
            "predicted_value_of_further_computation",
        )

    @classmethod
    def create(
        cls,
        *,
        origin: HypothesisOrigin,
        epistemic_status: EpistemicStatus,
        referent_hypotheses: FrozenJSON | Any,
        factor_change_distribution: FrozenJSON | Any,
        decision_relevance_distribution: FrozenJSON | Any,
        reducibility_distribution: FrozenJSON | Any,
        supporting_event_ids: Sequence[EventRef],
        calibrated_confidence: float,
        abstention_reason: str | None,
        predicted_value_of_further_computation: float,
        causal_parent_ids: Sequence[EventRef],
        counterfactual_branch_id: BranchRef = FACTUAL_BRANCH,
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int = 0,
        source_and_provenance: FrozenJSON | Any,
        producer_state_version: str | None = None,
        measured_creation_cost: WorkVector = WorkVector(),
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        supersedes_event_ids: Sequence[EventRef] = (),
    ) -> HypothesisEvent:
        referents = _freeze(referent_hypotheses)
        changes = _freeze(factor_change_distribution)
        relevance = _freeze(decision_relevance_distribution)
        reducibility = _freeze(reducibility_distribution)
        support = tuple(sorted(supporting_event_ids, key=str))
        body = _hypothesis_body(
            origin,
            epistemic_status,
            referents,
            changes,
            relevance,
            reducibility,
            support,
            calibrated_confidence,
            abstention_reason,
            predicted_value_of_further_computation,
        )
        envelope = EventEnvelope.create(
            event_kind=EventKind.HYPOTHESIS,
            causal_parent_ids=causal_parent_ids,
            counterfactual_branch_id=counterfactual_branch_id,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
            source_and_provenance=source_and_provenance,
            payload_digest=canonical_sha256(body),
            producer_state_version=producer_state_version,
            measured_creation_cost=measured_creation_cost,
            evidence_class=evidence_class,
            supersedes_event_ids=supersedes_event_ids,
        )
        return cls(
            envelope,
            origin,
            epistemic_status,
            referents,
            changes,
            relevance,
            reducibility,
            support,
            calibrated_confidence,
            abstention_reason,
            predicted_value_of_further_computation,
        )

    def body_payload(self) -> dict[str, Any]:
        return _hypothesis_body(
            self.origin,
            self.epistemic_status,
            self.referent_hypotheses,
            self.factor_change_distribution,
            self.decision_relevance_distribution,
            self.reducibility_distribution,
            self.supporting_event_ids,
            self.calibrated_confidence,
            self.abstention_reason,
            self.predicted_value_of_further_computation,
        )


def _hypothesis_body(
    origin: HypothesisOrigin,
    epistemic_status: EpistemicStatus,
    referents: FrozenJSON,
    changes: FrozenJSON,
    relevance: FrozenJSON,
    reducibility: FrozenJSON,
    support: tuple[EventRef, ...],
    confidence: float,
    abstention_reason: str | None,
    value_of_computation: float,
) -> dict[str, Any]:
    return {
        "origin": origin.value,
        "epistemic_status": epistemic_status.value,
        "referent_hypotheses": referents.payload(),
        "factor_change_distribution": changes.payload(),
        "decision_relevance_distribution": relevance.payload(),
        "reducibility_distribution": reducibility.payload(),
        "supporting_event_ids": [str(event_id) for event_id in support],
        "calibrated_confidence": confidence,
        "abstention_reason": abstention_reason,
        "predicted_value_of_further_computation": value_of_computation,
    }


@dataclass(frozen=True, slots=True)
class CommitmentEvent(_EventRecord):
    envelope: EventEnvelope
    coalition_id: str
    commitment_kind: CommitmentKind
    committed_payload: FrozenJSON
    decision_distribution: FrozenJSON
    deadline_tick: int
    predicted_utility_vector: FrozenJSON
    predicted_full_cost: WorkVector

    def __post_init__(self) -> None:
        self._validate_envelope(EventKind.COMMITMENT)
        _require_stable_ref(self.coalition_id, "coalition_id")
        if not isinstance(self.commitment_kind, CommitmentKind):
            raise ValueError("commitment_kind must be a CommitmentKind")
        for label, value in (
            ("committed_payload", self.committed_payload),
            ("decision_distribution", self.decision_distribution),
            ("predicted_utility_vector", self.predicted_utility_vector),
        ):
            if not isinstance(value, FrozenJSON):
                raise ValueError(f"{label} must be FrozenJSON")
            _reject_keys(value, _HIDDEN_TRUTH_KEYS | _COMMITMENT_FUTURE_KEYS, label)
        _require_nonnegative_int(self.deadline_tick, "deadline_tick")
        if self.deadline_tick < self.envelope.clock_end_tick:
            raise ValueError("deadline_tick cannot precede commitment creation")
        if not isinstance(self.predicted_full_cost, WorkVector):
            raise ValueError("predicted_full_cost must be a WorkVector")

    @classmethod
    def create(
        cls,
        *,
        coalition_id: str,
        commitment_kind: CommitmentKind,
        committed_payload: FrozenJSON | Any,
        decision_distribution: FrozenJSON | Any,
        deadline_tick: int,
        predicted_utility_vector: FrozenJSON | Any,
        predicted_full_cost: WorkVector,
        causal_parent_ids: Sequence[EventRef],
        counterfactual_branch_id: BranchRef = FACTUAL_BRANCH,
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int = 0,
        source_and_provenance: FrozenJSON | Any,
        producer_state_version: str | None = None,
        measured_creation_cost: WorkVector = WorkVector(),
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        supersedes_event_ids: Sequence[EventRef] = (),
    ) -> CommitmentEvent:
        committed = _freeze(committed_payload)
        distribution = _freeze(decision_distribution)
        utility = _freeze(predicted_utility_vector)
        body = _commitment_body(
            coalition_id,
            commitment_kind,
            committed,
            distribution,
            deadline_tick,
            utility,
            predicted_full_cost,
        )
        envelope = EventEnvelope.create(
            event_kind=EventKind.COMMITMENT,
            causal_parent_ids=causal_parent_ids,
            counterfactual_branch_id=counterfactual_branch_id,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
            source_and_provenance=source_and_provenance,
            payload_digest=canonical_sha256(body),
            producer_state_version=producer_state_version,
            measured_creation_cost=measured_creation_cost,
            evidence_class=evidence_class,
            supersedes_event_ids=supersedes_event_ids,
        )
        return cls(
            envelope,
            coalition_id,
            commitment_kind,
            committed,
            distribution,
            deadline_tick,
            utility,
            predicted_full_cost,
        )

    def body_payload(self) -> dict[str, Any]:
        return _commitment_body(
            self.coalition_id,
            self.commitment_kind,
            self.committed_payload,
            self.decision_distribution,
            self.deadline_tick,
            self.predicted_utility_vector,
            self.predicted_full_cost,
        )


def _commitment_body(
    coalition_id: str,
    commitment_kind: CommitmentKind,
    committed: FrozenJSON,
    distribution: FrozenJSON,
    deadline_tick: int,
    utility: FrozenJSON,
    predicted_cost: WorkVector,
) -> dict[str, Any]:
    return {
        "coalition_id": coalition_id,
        "commitment_kind": commitment_kind.value,
        "committed_payload": committed.payload(),
        "decision_distribution": distribution.payload(),
        "deadline_tick": deadline_tick,
        "predicted_utility_vector": utility.payload(),
        "predicted_full_cost": predicted_cost.payload(),
    }


@dataclass(frozen=True, slots=True)
class ConsequenceEvent(_EventRecord):
    envelope: EventEnvelope
    commitment_event_id: EventRef
    observed_outcome: FrozenJSON
    realized_utility_vector: FrozenJSON
    delayed_or_partial: bool
    observation_uncertainty: float
    realized_full_cost: WorkVector

    def __post_init__(self) -> None:
        self._validate_envelope(EventKind.CONSEQUENCE)
        if self.commitment_event_id not in self.envelope.causal_parent_ids:
            raise ValueError("commitment_event_id must be a causal parent")
        if not isinstance(self.observed_outcome, FrozenJSON):
            raise ValueError("observed_outcome must be FrozenJSON")
        if not isinstance(self.realized_utility_vector, FrozenJSON):
            raise ValueError("realized_utility_vector must be FrozenJSON")
        if not isinstance(self.delayed_or_partial, bool):
            raise ValueError("delayed_or_partial must be a boolean")
        _require_nonnegative_number(self.observation_uncertainty, "observation_uncertainty")
        if not isinstance(self.realized_full_cost, WorkVector):
            raise ValueError("realized_full_cost must be a WorkVector")

    @classmethod
    def create(
        cls,
        *,
        commitment_event_id: EventRef,
        observed_outcome: FrozenJSON | Any,
        realized_utility_vector: FrozenJSON | Any,
        delayed_or_partial: bool,
        observation_uncertainty: float,
        realized_full_cost: WorkVector,
        causal_parent_ids: Sequence[EventRef],
        counterfactual_branch_id: BranchRef = FACTUAL_BRANCH,
        clock_start_tick: int,
        clock_end_tick: int,
        clock_uncertainty: int = 0,
        source_and_provenance: FrozenJSON | Any,
        producer_state_version: str | None = None,
        measured_creation_cost: WorkVector = WorkVector(),
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
        supersedes_event_ids: Sequence[EventRef] = (),
    ) -> ConsequenceEvent:
        outcome = _freeze(observed_outcome)
        utility = _freeze(realized_utility_vector)
        body = _consequence_body(
            commitment_event_id,
            outcome,
            utility,
            delayed_or_partial,
            observation_uncertainty,
            realized_full_cost,
        )
        envelope = EventEnvelope.create(
            event_kind=EventKind.CONSEQUENCE,
            causal_parent_ids=causal_parent_ids,
            counterfactual_branch_id=counterfactual_branch_id,
            clock_start_tick=clock_start_tick,
            clock_end_tick=clock_end_tick,
            clock_uncertainty=clock_uncertainty,
            source_and_provenance=source_and_provenance,
            payload_digest=canonical_sha256(body),
            producer_state_version=producer_state_version,
            measured_creation_cost=measured_creation_cost,
            evidence_class=evidence_class,
            supersedes_event_ids=supersedes_event_ids,
        )
        return cls(
            envelope,
            commitment_event_id,
            outcome,
            utility,
            delayed_or_partial,
            observation_uncertainty,
            realized_full_cost,
        )

    def body_payload(self) -> dict[str, Any]:
        return _consequence_body(
            self.commitment_event_id,
            self.observed_outcome,
            self.realized_utility_vector,
            self.delayed_or_partial,
            self.observation_uncertainty,
            self.realized_full_cost,
        )


def _consequence_body(
    commitment_event_id: EventRef,
    outcome: FrozenJSON,
    utility: FrozenJSON,
    delayed_or_partial: bool,
    uncertainty: float,
    realized_cost: WorkVector,
) -> dict[str, Any]:
    return {
        "commitment_event_id": str(commitment_event_id),
        "observed_outcome": outcome.payload(),
        "realized_utility_vector": utility.payload(),
        "delayed_or_partial": delayed_or_partial,
        "observation_uncertainty": uncertainty,
        "realized_full_cost": realized_cost.payload(),
    }


ESCSEvent: TypeAlias = ObservationEvent | HypothesisEvent | CommitmentEvent | ConsequenceEvent


def event_from_payload(payload: Mapping[str, Any]) -> ESCSEvent:
    """Parse one exact event payload, rejecting unknown fields and kind/body mismatches."""

    _require_exact_keys(payload, {"schema", "envelope", "body"}, "ESCS event")
    if payload["schema"] != EVENT_SCHEMA:
        raise ValueError(f"unsupported ESCS event schema {payload['schema']!r}")
    envelope = EventEnvelope.from_payload(payload["envelope"])
    body = payload["body"]
    if envelope.event_kind is EventKind.OBSERVATION:
        expected = {
            "raw_packet_or_delta_refs",
            "adapter_version",
            "sensor_scope",
            "transport_and_detection_cost",
        }
        _require_exact_keys(body, expected, "ObservationEvent body")
        refs = body["raw_packet_or_delta_refs"]
        if not isinstance(refs, list) or not all(isinstance(value, str) for value in refs):
            raise ValueError("raw_packet_or_delta_refs must be a list of strings")
        return ObservationEvent(
            envelope=envelope,
            raw_packet_or_delta_refs=tuple(refs),
            adapter_version=body["adapter_version"],
            sensor_scope=_frozen_from_payload(body["sensor_scope"], "sensor_scope"),
            transport_and_detection_cost=WorkVector.from_payload(body["transport_and_detection_cost"]),
        )
    if envelope.event_kind is EventKind.HYPOTHESIS:
        expected = {
            "origin",
            "epistemic_status",
            "referent_hypotheses",
            "factor_change_distribution",
            "decision_relevance_distribution",
            "reducibility_distribution",
            "supporting_event_ids",
            "calibrated_confidence",
            "abstention_reason",
            "predicted_value_of_further_computation",
        }
        _require_exact_keys(body, expected, "HypothesisEvent body")
        support = body["supporting_event_ids"]
        if not isinstance(support, list) or not all(isinstance(value, str) for value in support):
            raise ValueError("supporting_event_ids must be a list of strings")
        return HypothesisEvent(
            envelope=envelope,
            origin=HypothesisOrigin(body["origin"]),
            epistemic_status=EpistemicStatus(body["epistemic_status"]),
            referent_hypotheses=_frozen_from_payload(body["referent_hypotheses"], "referent_hypotheses"),
            factor_change_distribution=_frozen_from_payload(
                body["factor_change_distribution"], "factor_change_distribution"
            ),
            decision_relevance_distribution=_frozen_from_payload(
                body["decision_relevance_distribution"], "decision_relevance_distribution"
            ),
            reducibility_distribution=_frozen_from_payload(
                body["reducibility_distribution"], "reducibility_distribution"
            ),
            supporting_event_ids=tuple(EventRef(value) for value in support),
            calibrated_confidence=body["calibrated_confidence"],
            abstention_reason=body["abstention_reason"],
            predicted_value_of_further_computation=body["predicted_value_of_further_computation"],
        )
    if envelope.event_kind is EventKind.COMMITMENT:
        expected = {
            "coalition_id",
            "commitment_kind",
            "committed_payload",
            "decision_distribution",
            "deadline_tick",
            "predicted_utility_vector",
            "predicted_full_cost",
        }
        _require_exact_keys(body, expected, "CommitmentEvent body")
        return CommitmentEvent(
            envelope=envelope,
            coalition_id=body["coalition_id"],
            commitment_kind=CommitmentKind(body["commitment_kind"]),
            committed_payload=_frozen_from_payload(body["committed_payload"], "committed_payload"),
            decision_distribution=_frozen_from_payload(
                body["decision_distribution"], "decision_distribution"
            ),
            deadline_tick=body["deadline_tick"],
            predicted_utility_vector=_frozen_from_payload(
                body["predicted_utility_vector"], "predicted_utility_vector"
            ),
            predicted_full_cost=WorkVector.from_payload(body["predicted_full_cost"]),
        )
    expected = {
        "commitment_event_id",
        "observed_outcome",
        "realized_utility_vector",
        "delayed_or_partial",
        "observation_uncertainty",
        "realized_full_cost",
    }
    _require_exact_keys(body, expected, "ConsequenceEvent body")
    return ConsequenceEvent(
        envelope=envelope,
        commitment_event_id=EventRef(body["commitment_event_id"]),
        observed_outcome=_frozen_from_payload(body["observed_outcome"], "observed_outcome"),
        realized_utility_vector=_frozen_from_payload(
            body["realized_utility_vector"], "realized_utility_vector"
        ),
        delayed_or_partial=body["delayed_or_partial"],
        observation_uncertainty=body["observation_uncertainty"],
        realized_full_cost=WorkVector.from_payload(body["realized_full_cost"]),
    )


def canonical_event_bytes(event: ESCSEvent) -> bytes:
    return canonical_bytes(event.payload())
