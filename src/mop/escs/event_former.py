"""Finite contracts and charged execution for the ESCS raw-event boundary.

The module freezes immutable raw input, bounded proposal/abstention, injected-policy,
evidence-taint, and resource-cap contracts. ``ChargedEventFormer`` accounts every admitted poll or
packet attempt and atomically publishes typed observation/hypothesis batches. These mechanics do not
identify meaningful events or establish learning, calibration, capability, or efficiency.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mop.substrate.events import EventRef, FrozenJSON, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from .events import (
    EpistemicStatus,
    EvidenceClass,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
)
from .ledger import EventLedger

RAW_PACKET_SCHEMA = "mop-escs-raw-packet/v1"
EVENT_PROPOSAL_SCHEMA = "mop-escs-event-proposal/v1"
EVENT_FORMER_DECISION_SCHEMA = "mop-escs-event-former-decision/v1"
EVENT_FORMER_RESULT_SCHEMA = "mop-escs-event-former-result/v1"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_STABLE_REF_RE = re.compile(r"^[a-z][a-z0-9+.-]*:[a-z0-9][a-z0-9._:/-]*$")
_HIDDEN_KEYS = frozenset(
    {
        "absolute_identity",
        "canonical_referent_truth",
        "causal_label",
        "change_point",
        "evaluator_factor",
        "evaluator_label",
        "event_label",
        "factor_label",
        "ground_truth",
        "ground_truth_label",
        "hidden_change_point",
        "niche_label",
        "optimal_action",
        "oracle_label",
        "target_action",
    }
)


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_event_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("event:"):
        raise ValueError(f"{label} must use the event namespace")
    _require_digest(value.removeprefix("event:"), f"{label} digest")
    return value


def _freeze(value: Any) -> FrozenJSON:
    return value if isinstance(value, FrozenJSON) else FrozenJSON.from_value(value)


def _hidden_paths(value: Any, *, path: str = "$") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            normalized = key_text.strip().lower().replace("-", "_")
            nested_path = f"{path}.{key_text}"
            if normalized in _HIDDEN_KEYS:
                found.append(nested_path)
            found.extend(_hidden_paths(nested, path=nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_hidden_paths(nested, path=f"{path}[{index}]"))
    return tuple(found)


def _reject_hidden(value: Any, label: str) -> None:
    if found := _hidden_paths(value):
        raise ValueError(f"{label} contains evaluator-only fields: {sorted(found)}")


def _validate_opaque_json(raw: bytes) -> None:
    """Inspect JSON bytes without pretending arbitrary binary semantics can be certified."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    _reject_hidden(value, "RawPacket.payload_bytes")


@dataclass(frozen=True, slots=True)
class RawPacket:
    """Content-addressed raw input with clocks and explicitly declared intake work.

    The exact schema has no evaluator label, hidden change point, target action, or semantic event
    field.  Opaque non-JSON bytes still require an independent source audit.
    """

    packet_id: str
    sensor_id: str
    capture_start_tick: int
    capture_end_tick: int
    arrival_tick: int
    clock_uncertainty: int
    payload_bytes: bytes
    sensor_scope: FrozenJSON
    source_and_provenance: FrozenJSON
    transport_operations: int
    adaptation_operations: int
    detection_operations: int
    evidence_class: EvidenceClass
    schema: str = RAW_PACKET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RAW_PACKET_SCHEMA:
            raise ValueError("unsupported raw-packet schema")
        if not isinstance(self.packet_id, str) or not self.packet_id.startswith("packet:"):
            raise ValueError("packet_id must use the packet namespace")
        _require_digest(self.packet_id.removeprefix("packet:"), "packet_id digest")
        if not isinstance(self.sensor_id, str) or _STABLE_REF_RE.fullmatch(self.sensor_id) is None:
            raise ValueError("sensor_id must be a stable namespaced reference")
        for name in (
            "capture_start_tick",
            "capture_end_tick",
            "arrival_tick",
            "clock_uncertainty",
            "transport_operations",
            "adaptation_operations",
            "detection_operations",
        ):
            _require_nonnegative_int(getattr(self, name), f"RawPacket.{name}")
        if self.capture_end_tick < self.capture_start_tick:
            raise ValueError("raw-packet capture interval is invalid")
        if self.arrival_tick < self.capture_end_tick:
            raise ValueError("raw-packet arrival cannot precede capture completion")
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("RawPacket.payload_bytes must be immutable bytes")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("RawPacket.evidence_class must be typed")
        if not isinstance(self.sensor_scope, FrozenJSON) or not isinstance(
            self.source_and_provenance, FrozenJSON
        ):
            raise ValueError("raw-packet scope and provenance must be FrozenJSON")
        if not isinstance(self.sensor_scope.value(), dict):
            raise ValueError("RawPacket.sensor_scope must contain a JSON object")
        if not isinstance(self.source_and_provenance.value(), dict):
            raise ValueError("RawPacket.source_and_provenance must contain a JSON object")
        _reject_hidden(self.sensor_scope.value(), "RawPacket.sensor_scope")
        _reject_hidden(self.source_and_provenance.value(), "RawPacket.source_and_provenance")
        _validate_opaque_json(self.payload_bytes)
        if self.transport_operations < len(self.payload_bytes):
            raise ValueError("raw transport work cannot underreport payload bytes")
        if self.adaptation_operations <= 0 or self.detection_operations <= 0:
            raise ValueError("raw adaptation and detection must each report positive work")
        expected = f"packet:{canonical_sha256(self.identity_payload())}"
        if self.packet_id != expected:
            raise ValueError("raw-packet identity digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        sensor_id: str,
        capture_start_tick: int,
        capture_end_tick: int,
        arrival_tick: int,
        payload_bytes: bytes,
        sensor_scope: FrozenJSON | Any,
        source_and_provenance: FrozenJSON | Any,
        transport_operations: int,
        adaptation_operations: int,
        detection_operations: int,
        clock_uncertainty: int = 0,
        evidence_class: EvidenceClass = EvidenceClass.LEARNED_UNVERIFIED,
    ) -> RawPacket:
        scope = _freeze(sensor_scope)
        provenance = _freeze(source_and_provenance)
        raw = bytes(payload_bytes)
        partial = {
            "schema": RAW_PACKET_SCHEMA,
            "sensor_id": sensor_id,
            "capture_interval": [capture_start_tick, capture_end_tick],
            "arrival_tick": arrival_tick,
            "clock_uncertainty": clock_uncertainty,
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_bytes": len(raw),
            "sensor_scope": scope.payload(),
            "source_and_provenance": provenance.payload(),
            "transport_operations": transport_operations,
            "adaptation_operations": adaptation_operations,
            "detection_operations": detection_operations,
            "evidence_class": evidence_class.value,
        }
        return cls(
            packet_id=f"packet:{canonical_sha256(partial)}",
            sensor_id=sensor_id,
            capture_start_tick=capture_start_tick,
            capture_end_tick=capture_end_tick,
            arrival_tick=arrival_tick,
            clock_uncertainty=clock_uncertainty,
            payload_bytes=raw,
            sensor_scope=scope,
            source_and_provenance=provenance,
            transport_operations=transport_operations,
            adaptation_operations=adaptation_operations,
            detection_operations=detection_operations,
            evidence_class=evidence_class,
        )

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload_bytes).hexdigest()

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.identity_payload())) + len(self.payload_bytes)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sensor_id": self.sensor_id,
            "capture_interval": [self.capture_start_tick, self.capture_end_tick],
            "arrival_tick": self.arrival_tick,
            "clock_uncertainty": self.clock_uncertainty,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": len(self.payload_bytes),
            "sensor_scope": self.sensor_scope.payload(),
            "source_and_provenance": self.source_and_provenance.payload(),
            "transport_operations": self.transport_operations,
            "adaptation_operations": self.adaptation_operations,
            "detection_operations": self.detection_operations,
            "evidence_class": self.evidence_class.value,
        }


@dataclass(frozen=True, slots=True)
class EventProposal:
    """One bounded candidate hypothesis; its distributions remain representation-neutral JSON."""

    proposal_id: str
    epistemic_status: EpistemicStatus
    referent_hypotheses: FrozenJSON
    factor_change_distribution: FrozenJSON
    decision_relevance_distribution: FrozenJSON
    reducibility_distribution: FrozenJSON
    calibrated_confidence: float
    predicted_value_of_further_computation: float
    formation_operations: int
    schema: str = EVENT_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_PROPOSAL_SCHEMA:
            raise ValueError("unsupported event-proposal schema")
        _require_digest(self.proposal_id, "proposal_id")
        if self.epistemic_status not in {
            EpistemicStatus.OBSERVED_CANDIDATE,
            EpistemicStatus.INFERRED,
        }:
            raise ValueError("a raw event former cannot emit simulated or factualized proposals")
        for label, value in (
            ("referent_hypotheses", self.referent_hypotheses),
            ("factor_change_distribution", self.factor_change_distribution),
            ("decision_relevance_distribution", self.decision_relevance_distribution),
            ("reducibility_distribution", self.reducibility_distribution),
        ):
            if not isinstance(value, FrozenJSON):
                raise ValueError(f"{label} must be FrozenJSON")
            _reject_hidden(value.value(), f"EventProposal.{label}")
        if not math.isfinite(self.calibrated_confidence) or not 0 <= self.calibrated_confidence <= 1:
            raise ValueError("calibrated_confidence must be finite inside [0, 1]")
        if (
            not math.isfinite(self.predicted_value_of_further_computation)
            or self.predicted_value_of_further_computation < 0
        ):
            raise ValueError("predicted value of further computation must be finite and nonnegative")
        if (
            isinstance(self.formation_operations, bool)
            or not isinstance(self.formation_operations, int)
            or self.formation_operations <= 0
        ):
            raise ValueError("every event proposal must report positive integer formation work")
        if self.proposal_id != canonical_sha256(self.identity_payload()):
            raise ValueError("event-proposal identity digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        epistemic_status: EpistemicStatus,
        referent_hypotheses: FrozenJSON | Any,
        factor_change_distribution: FrozenJSON | Any,
        decision_relevance_distribution: FrozenJSON | Any,
        reducibility_distribution: FrozenJSON | Any,
        calibrated_confidence: float,
        predicted_value_of_further_computation: float,
        formation_operations: int,
    ) -> EventProposal:
        referents = _freeze(referent_hypotheses)
        changes = _freeze(factor_change_distribution)
        relevance = _freeze(decision_relevance_distribution)
        reducibility = _freeze(reducibility_distribution)
        partial = {
            "schema": EVENT_PROPOSAL_SCHEMA,
            "epistemic_status": epistemic_status.value,
            "referent_hypotheses": referents.payload(),
            "factor_change_distribution": changes.payload(),
            "decision_relevance_distribution": relevance.payload(),
            "reducibility_distribution": reducibility.payload(),
            "calibrated_confidence": calibrated_confidence,
            "predicted_value_of_further_computation": predicted_value_of_further_computation,
            "formation_operations": formation_operations,
        }
        return cls(
            proposal_id=canonical_sha256(partial),
            epistemic_status=epistemic_status,
            referent_hypotheses=referents,
            factor_change_distribution=changes,
            decision_relevance_distribution=relevance,
            reducibility_distribution=reducibility,
            calibrated_confidence=calibrated_confidence,
            predicted_value_of_further_computation=predicted_value_of_further_computation,
            formation_operations=formation_operations,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "epistemic_status": self.epistemic_status.value,
            "referent_hypotheses": self.referent_hypotheses.payload(),
            "factor_change_distribution": self.factor_change_distribution.payload(),
            "decision_relevance_distribution": self.decision_relevance_distribution.payload(),
            "reducibility_distribution": self.reducibility_distribution.payload(),
            "calibrated_confidence": self.calibrated_confidence,
            "predicted_value_of_further_computation": self.predicted_value_of_further_computation,
            "formation_operations": self.formation_operations,
        }

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.identity_payload()))


@dataclass(frozen=True, slots=True)
class EventFormerDescriptor:
    """Frozen evidence declaration for one injected policy.

    Oracle access is legal only under ``ORACLE_NONPROMOTABLE`` and is refused by a promotable-mode
    configuration.  Non-oracle classes remain mechanics labels, not positive evidence.
    """

    policy_id: str
    evidence_class: EvidenceClass
    oracle_access: bool = False

    def __post_init__(self) -> None:
        if _STABLE_REF_RE.fullmatch(self.policy_id) is None:
            raise ValueError("event-former policy_id must be a stable namespaced reference")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("event-former evidence_class must be typed")
        if not isinstance(self.oracle_access, bool):
            raise ValueError("oracle_access must be a boolean")
        if self.oracle_access != (self.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE):
            raise ValueError("oracle access must use the explicit nonpromotable evidence class")

    @property
    def promotable_input(self) -> bool:
        return self.evidence_class is EvidenceClass.LEARNED_UNVERIFIED


@dataclass(frozen=True, slots=True)
class EventFormerDecision:
    """Bounded policy output: admitted proposals or an explicit charged abstention."""

    proposals: tuple[EventProposal, ...]
    discarded_candidates: int
    policy_operations: int
    retained_state_bytes: int
    idle_operations: int
    abstention_reason: str | None
    evidence_class: EvidenceClass
    schema: str = EVENT_FORMER_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMER_DECISION_SCHEMA:
            raise ValueError("unsupported event-former decision schema")
        if not isinstance(self.proposals, tuple):
            raise ValueError("event-former proposals must be an immutable tuple")
        if any(type(proposal) is not EventProposal for proposal in self.proposals):
            raise ValueError("event-former proposals must be exact EventProposal records")
        if len({proposal.proposal_id for proposal in self.proposals}) != len(self.proposals):
            raise ValueError("event-former proposal identities must be unique")
        if self.proposals != tuple(sorted(self.proposals, key=lambda row: row.proposal_id)):
            raise ValueError("event-former proposals must use canonical proposal-id order")
        for name in (
            "discarded_candidates",
            "policy_operations",
            "retained_state_bytes",
            "idle_operations",
        ):
            _require_nonnegative_int(getattr(self, name), f"EventFormerDecision.{name}")
        minimum_reported = sum(row.formation_operations for row in self.proposals)
        if self.policy_operations < minimum_reported:
            raise ValueError("event former underreported proposal-formation work")
        if self.proposals:
            if self.abstention_reason is not None or self.idle_operations != 0:
                raise ValueError("an admitted decision cannot also claim idle abstention")
        else:
            _require_text(self.abstention_reason, "abstention_reason")
            if self.idle_operations <= 0:
                raise ValueError("abstention must report positive idle-check work")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("decision evidence_class must be typed")


@runtime_checkable
class EventFormerPolicy(Protocol):
    """Injected policy contract; implementations are outside this evidence-neutral stage."""

    @property
    def descriptor(self) -> EventFormerDescriptor: ...

    @property
    def state_version(self) -> str: ...

    @property
    def retained_state_bytes(self) -> int: ...

    def evaluate(self, packet: RawPacket) -> EventFormerDecision: ...


@dataclass(frozen=True, slots=True)
class EventFormerConfig:
    max_packet_bytes: int = 1024 * 1024
    max_packet_encoded_bytes: int = 2 * 1024 * 1024
    max_candidates_per_packet: int = 1024
    max_hypothesis_fanout: int = 16
    max_proposal_encoded_bytes: int = 256 * 1024
    max_policy_operations: int = 10_000_000
    max_poll_operations: int = 1_000_000
    max_idle_operations: int = 1_000_000
    max_abstention_reason_bytes: int = 4096
    max_observation_parents: int = 64
    max_retained_state_bytes: int = 64 * 1024 * 1024
    minimum_policy_operations: int = 1
    promotable_mode: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_packet_bytes",
            "max_packet_encoded_bytes",
            "max_candidates_per_packet",
            "max_proposal_encoded_bytes",
            "max_policy_operations",
            "max_poll_operations",
            "max_idle_operations",
            "max_abstention_reason_bytes",
            "max_observation_parents",
            "max_retained_state_bytes",
            "minimum_policy_operations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"EventFormerConfig.{name} must be a positive integer")
        _require_nonnegative_int(self.max_hypothesis_fanout, "max_hypothesis_fanout")
        if self.max_hypothesis_fanout > self.max_candidates_per_packet:
            raise ValueError("hypothesis fanout cannot exceed the total candidate cap")
        if not isinstance(self.promotable_mode, bool):
            raise ValueError("promotable_mode must be a boolean")
        if self.minimum_policy_operations > self.max_policy_operations:
            raise ValueError("minimum policy work cannot exceed its cap")

    def validate_descriptor(self, descriptor: EventFormerDescriptor) -> None:
        if not isinstance(descriptor, EventFormerDescriptor):
            raise TypeError("policy descriptor must be EventFormerDescriptor")
        if self.promotable_mode and not descriptor.promotable_input:
            raise ValueError("only learned-unverified event formation is allowed in promotable mode")

    def validate_policy(self, policy: EventFormerPolicy) -> None:
        if not isinstance(policy, EventFormerPolicy):
            raise TypeError("policy does not implement EventFormerPolicy")
        self.validate_descriptor(policy.descriptor)
        _require_digest(policy.state_version, "event-former policy state_version")
        retained = _require_nonnegative_int(policy.retained_state_bytes, "event-former retained_state_bytes")
        if retained > self.max_retained_state_bytes:
            raise ValueError("event-former retained state exceeds its cap")

    def validate_packet(self, packet: RawPacket) -> None:
        if not isinstance(packet, RawPacket):
            raise TypeError("event former requires a RawPacket")
        if len(packet.payload_bytes) > self.max_packet_bytes:
            raise ValueError("raw packet exceeds the payload-byte cap")
        if packet.encoded_bytes > self.max_packet_encoded_bytes:
            raise ValueError("raw packet exceeds the encoded-byte cap")
        if self.promotable_mode and packet.evidence_class is not EvidenceClass.LEARNED_UNVERIFIED:
            raise ValueError("non-learned packet evidence cannot cross a promotable boundary")

    def validate_decision(
        self,
        decision: EventFormerDecision,
        *,
        descriptor: EventFormerDescriptor,
        retained_state_bytes: int,
    ) -> None:
        if not isinstance(decision, EventFormerDecision):
            raise TypeError("policy returned an invalid EventFormerDecision")
        self.validate_descriptor(descriptor)
        candidate_count = len(decision.proposals) + decision.discarded_candidates
        if candidate_count > self.max_candidates_per_packet:
            raise ValueError("event former exceeded the total candidate cap")
        if len(decision.proposals) > self.max_hypothesis_fanout:
            raise ValueError("event former exceeded the hypothesis-fanout cap")
        if not self.minimum_policy_operations <= decision.policy_operations <= self.max_policy_operations:
            raise ValueError("event-former policy work lies outside its declared bounds")
        if decision.retained_state_bytes != retained_state_bytes:
            raise ValueError("decision retained bytes do not match policy state")
        if retained_state_bytes > self.max_retained_state_bytes:
            raise ValueError("event-former retained state exceeds its cap")
        if decision.evidence_class is not descriptor.evidence_class:
            raise ValueError("decision evidence class differs from its frozen descriptor")
        if decision.idle_operations > self.max_idle_operations:
            raise ValueError("event former exceeded its idle-operation cap")
        if (
            decision.abstention_reason is not None
            and len(decision.abstention_reason.encode("utf-8")) > self.max_abstention_reason_bytes
        ):
            raise ValueError("event former exceeded its abstention-reason byte cap")
        if any(proposal.encoded_bytes > self.max_proposal_encoded_bytes for proposal in decision.proposals):
            raise ValueError("event-former proposal exceeds its encoded-byte cap")

    def effective_evidence_class(
        self,
        *,
        packet: RawPacket,
        descriptor: EventFormerDescriptor,
        decision: EventFormerDecision,
        parent_evidence_classes: tuple[EvidenceClass, ...] = (),
    ) -> EvidenceClass:
        """Join all ingress taints; a restrictive source can never be relabeled downward."""

        if not isinstance(parent_evidence_classes, tuple) or not all(
            isinstance(row, EvidenceClass) for row in parent_evidence_classes
        ):
            raise ValueError("parent evidence classes must be an immutable typed tuple")
        self.validate_packet(packet)
        self.validate_descriptor(descriptor)
        classes = (
            packet.evidence_class,
            *parent_evidence_classes,
            descriptor.evidence_class,
            decision.evidence_class,
        )
        effective = max(classes, key=lambda row: row.taint_rank)
        if decision.evidence_class.taint_rank < descriptor.evidence_class.taint_rank:
            raise ValueError("decision evidence class downgrades its policy descriptor")
        if self.promotable_mode and effective is not EvidenceClass.LEARNED_UNVERIFIED:
            raise ValueError("non-learned evidence cannot cross a promotable boundary")
        return effective

    def validate_poll(
        self,
        *,
        polling_operations: int,
        deadline_operations: int,
    ) -> None:
        _require_nonnegative_int(polling_operations, "polling_operations")
        _require_nonnegative_int(deadline_operations, "deadline_operations")
        if not 1 <= polling_operations <= self.max_poll_operations:
            raise ValueError("event-former poll work lies outside its declared bounds")
        if deadline_operations > self.max_idle_operations:
            raise ValueError("event-former deadline work exceeds its declared bound")

    def validate_observation_parents(self, parent_ids: tuple[EventRef, ...]) -> None:
        if not isinstance(parent_ids, tuple):
            raise ValueError("observation parents must be an immutable tuple")
        if len(parent_ids) > self.max_observation_parents:
            raise ValueError("observation parent count exceeds its cap")
        if not all(isinstance(event_id, EventRef) for event_id in parent_ids):
            raise ValueError("observation parents must contain EventRef values")
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("observation parents must be unique")
        if parent_ids != tuple(sorted(parent_ids, key=str)):
            raise ValueError("observation parents must use canonical order")


class EventFormerContractError(RuntimeError):
    """An injected policy violated the finite charged event-former contract."""


@dataclass(frozen=True, slots=True)
class EventFormerResult:
    packet_id: str
    observation_event_id: str | None
    hypothesis_event_ids: tuple[str, ...]
    evidence_class: EvidenceClass
    admitted: bool
    lifecycle_start_sequence: int
    lifecycle_end_sequence: int
    retained_state_bytes: int
    schema: str = EVENT_FORMER_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVENT_FORMER_RESULT_SCHEMA:
            raise ValueError("unsupported event-former result schema")
        if not isinstance(self.admitted, bool):
            raise ValueError("event-former admitted flag must be boolean")
        if self.admitted != bool(self.hypothesis_event_ids):
            raise ValueError("event-former admitted flag must match hypothesis output")
        if self.admitted != (self.observation_event_id is not None):
            raise ValueError("an admitted result requires exactly one observation event")
        if not isinstance(self.hypothesis_event_ids, tuple):
            raise ValueError("hypothesis event ids must be an immutable tuple")
        if not isinstance(self.packet_id, str) or not self.packet_id.startswith("packet:"):
            raise ValueError("result packet_id must use the packet namespace")
        _require_digest(self.packet_id.removeprefix("packet:"), "result packet_id digest")
        if self.observation_event_id is not None:
            _require_event_id(self.observation_event_id, "observation_event_id")
        if not all(isinstance(row, str) for row in self.hypothesis_event_ids):
            raise ValueError("hypothesis event ids must contain strings")
        for event_id in self.hypothesis_event_ids:
            _require_event_id(event_id, "hypothesis_event_id")
        if len(set(self.hypothesis_event_ids)) != len(self.hypothesis_event_ids):
            raise ValueError("hypothesis event ids must be unique")
        if self.observation_event_id in self.hypothesis_event_ids:
            raise ValueError("observation and hypothesis event identities must be distinct")
        if self.hypothesis_event_ids != tuple(sorted(self.hypothesis_event_ids)):
            raise ValueError("hypothesis event ids must use canonical order")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("result evidence_class must be typed")
        _require_nonnegative_int(self.lifecycle_start_sequence, "lifecycle_start_sequence")
        _require_nonnegative_int(self.lifecycle_end_sequence, "lifecycle_end_sequence")
        _require_nonnegative_int(self.retained_state_bytes, "retained_state_bytes")
        if self.lifecycle_end_sequence < self.lifecycle_start_sequence:
            raise ValueError("event-former lifecycle sequence interval is invalid")


class ChargedEventFormer:
    """Execute one finite injected event policy and publish its events as one atomic batch.

    Policy evaluation is required to be observationally pure at this boundary. Learned policy
    updates need an explicit future transaction contract; silent mutation poisons this instance.
    Rejected reports above a declared cap receive an explicitly named saturated rejection charge;
    such runs are invalid for exact-work or promotion claims.
    """

    def __init__(
        self,
        *,
        policy: EventFormerPolicy,
        config: EventFormerConfig,
        event_ledger: EventLedger,
        lifecycle_ledger: LifecycleLedger,
        owner: str = "escs.event-former",
        deployment_start_tick: int | None = None,
    ) -> None:
        config.validate_policy(policy)
        if not isinstance(event_ledger, EventLedger):
            raise TypeError("event_ledger must be an EventLedger")
        if not isinstance(lifecycle_ledger, LifecycleLedger):
            raise TypeError("lifecycle_ledger must be a LifecycleLedger")
        self._policy: EventFormerPolicy | None = policy
        self._descriptor = policy.descriptor
        self._policy_state_version = policy.state_version
        self._policy_retained_state_bytes = policy.retained_state_bytes
        self._config = config
        self._events = event_ledger
        self._lifecycle = lifecycle_ledger
        self._owner = _require_text(owner, "event-former owner")
        if deployment_start_tick is not None:
            _require_nonnegative_int(deployment_start_tick, "deployment_start_tick")
        self._last_tick = 0 if deployment_start_tick is None else deployment_start_tick
        self._started = deployment_start_tick is not None
        self._retained_state_bytes = policy.retained_state_bytes
        self._closed = False

    @property
    def retained_state_bytes(self) -> int:
        return self._retained_state_bytes

    @property
    def finalized(self) -> bool:
        return self._closed

    def _charge_at(self, *, reason: str, work: WorkVector, tick: int) -> None:
        self._lifecycle.charge(
            owner=self._owner,
            reason=reason,
            work=work,
            start_tick=tick,
            end_tick=tick,
            branch_id=FACTUAL_BRANCH,
        )

    def _advance_clock(self, tick: int) -> None:
        _require_nonnegative_int(tick, "event-former tick")
        if tick < self._last_tick:
            raise EventFormerContractError("event-former time cannot move backwards")
        if self._started and tick > self._last_tick and self._retained_state_bytes:
            self._lifecycle.charge_retention(
                owner=self._owner,
                reason="event-former-retained-state",
                retained_bytes=self._retained_state_bytes,
                start_tick=self._last_tick,
                end_tick=tick,
            )
        self._last_tick = tick
        self._started = True

    def _poison(self) -> None:
        self._policy = None
        self._retained_state_bytes = 0
        self._closed = True

    def _assert_policy_unchanged(self, *, tick: int) -> EventFormerPolicy:
        policy = self._policy
        if policy is None:
            raise EventFormerContractError("event former policy is unavailable")
        if (
            policy.descriptor != self._descriptor
            or policy.state_version != self._policy_state_version
            or policy.retained_state_bytes != self._policy_retained_state_bytes
        ):
            self._charge_at(
                reason="event-former-out-of-band-policy-mutation",
                work=WorkVector(event_formation=1),
                tick=tick,
            )
            self._poison()
            raise EventFormerContractError("event-former policy changed outside its transaction boundary")
        return policy

    def poll(
        self,
        *,
        tick: int,
        polling_operations: int,
        deadline_operations: int = 0,
    ) -> tuple[int, int]:
        """Charge a bounded no-packet poll without inventing an ObservationEvent."""

        if self._closed:
            raise EventFormerContractError("event former is finalized")
        _require_nonnegative_int(tick, "event-former poll tick")
        if tick < self._last_tick:
            self._charge_at(
                reason="event-former-poll-time-order-rejection",
                work=WorkVector(raw_transport_and_adapters=1, idle_floor=1),
                tick=self._last_tick,
            )
            raise EventFormerContractError("event-former poll time cannot move backwards")
        start_sequence = self._lifecycle.entry_count
        self._advance_clock(tick)
        self._assert_policy_unchanged(tick=tick)
        try:
            self._config.validate_poll(
                polling_operations=polling_operations,
                deadline_operations=deadline_operations,
            )
        except (TypeError, ValueError) as exc:
            attempted_poll = (
                min(max(1, polling_operations), self._config.max_poll_operations)
                if isinstance(polling_operations, int) and not isinstance(polling_operations, bool)
                else 1
            )
            attempted_idle = (
                min(max(1, deadline_operations), self._config.max_idle_operations)
                if isinstance(deadline_operations, int) and not isinstance(deadline_operations, bool)
                else 1
            )
            self._charge_at(
                reason="event-former-no-packet-poll-saturated-cap-rejection",
                work=WorkVector(
                    raw_transport_and_adapters=attempted_poll,
                    idle_floor=attempted_idle,
                ),
                tick=tick,
            )
            raise EventFormerContractError(str(exc)) from exc
        self._charge_at(
            reason="event-former-no-packet-poll",
            work=WorkVector(
                raw_transport_and_adapters=polling_operations,
                idle_floor=max(1, deadline_operations),
            ),
            tick=tick,
        )
        return start_sequence, self._lifecycle.entry_count

    def process(
        self,
        packet: RawPacket,
        *,
        observation_parent_ids: tuple[EventRef, ...] = (),
    ) -> EventFormerResult:
        if not isinstance(packet, RawPacket):
            raise TypeError("ChargedEventFormer requires a RawPacket")
        if self._closed:
            raise EventFormerContractError("event former is finalized")
        if packet.arrival_tick < self._last_tick:
            self._charge_at(
                reason="event-former-packet-time-order-rejection",
                work=WorkVector(
                    raw_transport_and_adapters=(
                        packet.transport_operations
                        + packet.adaptation_operations
                        + packet.detection_operations
                    ),
                    event_formation=1,
                ),
                tick=self._last_tick,
            )
            raise EventFormerContractError("event-former packet time cannot move backwards")

        start_sequence = self._lifecycle.entry_count
        self._advance_clock(packet.arrival_tick)
        raw_work = packet.transport_operations + packet.adaptation_operations + packet.detection_operations
        self._charge_at(
            reason="raw-transport-adaptation-detection",
            work=WorkVector(raw_transport_and_adapters=raw_work),
            tick=packet.arrival_tick,
        )
        policy = self._assert_policy_unchanged(tick=packet.arrival_tick)
        try:
            self._config.validate_packet(packet)
        except (TypeError, ValueError) as exc:
            self._charge_at(
                reason="event-former-intake-cap-rejection",
                work=WorkVector(event_formation=1),
                tick=packet.arrival_tick,
            )
            raise EventFormerContractError(str(exc)) from exc

        if not isinstance(observation_parent_ids, tuple):
            self._charge_at(
                reason="event-former-parent-validation-rejection",
                work=WorkVector(indexing_and_graph_maintenance=1),
                tick=packet.arrival_tick,
            )
            raise EventFormerContractError("observation parents must be an immutable tuple")
        parents = observation_parent_ids
        self._charge_at(
            reason="event-former-parent-validation-and-lookup-attempt",
            work=WorkVector(
                indexing_and_graph_maintenance=(1 + min(len(parents), self._config.max_observation_parents))
            ),
            tick=packet.arrival_tick,
        )
        try:
            self._config.validate_observation_parents(parents)
        except ValueError as exc:
            raise EventFormerContractError(str(exc)) from exc
        parent_events = []
        for event_id in parents:
            try:
                parent_events.append(self._events.get(event_id))
            except ValueError as exc:
                raise ValueError(f"missing causal parent {event_id}") from exc

        prior_policy_state = policy.state_version
        prior_policy_bytes = policy.retained_state_bytes
        prior_policy_descriptor = policy.descriptor
        try:
            decision = policy.evaluate(packet)
        except Exception as exc:
            self._charge_at(
                reason="event-former-policy-error",
                work=WorkVector(event_formation=1 + self._config.minimum_policy_operations),
                tick=packet.arrival_tick,
            )
            if (
                policy.state_version != prior_policy_state
                or policy.retained_state_bytes != prior_policy_bytes
                or policy.descriptor != prior_policy_descriptor
            ):
                self._poison()
                raise EventFormerContractError("failed policy evaluation mutated event-former state") from exc
            raise
        if (
            policy.state_version != prior_policy_state
            or policy.retained_state_bytes != prior_policy_bytes
            or policy.descriptor != prior_policy_descriptor
        ):
            self._charge_at(
                reason="event-former-policy-state-mutation",
                work=WorkVector(event_formation=1 + self._config.minimum_policy_operations),
                tick=packet.arrival_tick,
            )
            self._poison()
            raise EventFormerContractError("policy evaluation mutated event-former state")

        if not isinstance(decision, EventFormerDecision):
            self._charge_at(
                reason="event-former-invalid-policy-result",
                work=WorkVector(event_formation=1 + self._config.minimum_policy_operations),
                tick=packet.arrival_tick,
            )
            raise EventFormerContractError("policy returned an invalid EventFormerDecision")
        attempted_work = max(
            self._config.minimum_policy_operations,
            min(decision.policy_operations, self._config.max_policy_operations)
            + min(decision.discarded_candidates, self._config.max_candidates_per_packet),
        )
        saturated_rejection = (
            decision.policy_operations > self._config.max_policy_operations
            or decision.discarded_candidates > self._config.max_candidates_per_packet
        )
        self._charge_at(
            reason=(
                "event-former-policy-and-discarded-candidates-saturated-cap-rejection"
                if saturated_rejection
                else "event-former-policy-and-discarded-candidates"
            ),
            work=WorkVector(event_formation=1 + attempted_work),
            tick=packet.arrival_tick,
        )
        try:
            self._config.validate_decision(
                decision,
                descriptor=self._descriptor,
                retained_state_bytes=policy.retained_state_bytes,
            )
            effective_evidence_class = self._config.effective_evidence_class(
                packet=packet,
                descriptor=self._descriptor,
                decision=decision,
                parent_evidence_classes=tuple(row.evidence_class for row in parent_events),
            )
        except (TypeError, ValueError) as exc:
            raise EventFormerContractError(str(exc)) from exc

        self._retained_state_bytes = decision.retained_state_bytes
        if not decision.proposals:
            self._charge_at(
                reason=f"event-former-abstention:{decision.abstention_reason}",
                work=WorkVector(idle_floor=decision.idle_operations),
                tick=packet.arrival_tick,
            )
            return EventFormerResult(
                packet_id=packet.packet_id,
                observation_event_id=None,
                hypothesis_event_ids=(),
                evidence_class=effective_evidence_class,
                admitted=False,
                lifecycle_start_sequence=start_sequence,
                lifecycle_end_sequence=self._lifecycle.entry_count,
                retained_state_bytes=decision.retained_state_bytes,
            )

        raw_cost = WorkVector(raw_transport_and_adapters=raw_work)
        observation = ObservationEvent.create(
            raw_packet_or_delta_refs=(packet.packet_id,),
            adapter_version=self._descriptor.policy_id,
            sensor_scope=packet.sensor_scope,
            transport_and_detection_cost=raw_cost,
            causal_parent_ids=parents,
            clock_start_tick=packet.capture_start_tick,
            clock_end_tick=packet.arrival_tick,
            clock_uncertainty=packet.clock_uncertainty,
            source_and_provenance=packet.source_and_provenance,
            measured_creation_cost=WorkVector(event_formation=1),
            evidence_class=effective_evidence_class,
        )
        hypotheses = tuple(
            HypothesisEvent.create(
                origin=HypothesisOrigin.EVENT_FORMER,
                epistemic_status=proposal.epistemic_status,
                referent_hypotheses=proposal.referent_hypotheses,
                factor_change_distribution=proposal.factor_change_distribution,
                decision_relevance_distribution=proposal.decision_relevance_distribution,
                reducibility_distribution=proposal.reducibility_distribution,
                supporting_event_ids=(observation.event_id,),
                calibrated_confidence=proposal.calibrated_confidence,
                abstention_reason=None,
                predicted_value_of_further_computation=(proposal.predicted_value_of_further_computation),
                causal_parent_ids=(observation.event_id,),
                counterfactual_branch_id=FACTUAL_BRANCH,
                clock_start_tick=packet.arrival_tick,
                clock_end_tick=packet.arrival_tick,
                source_and_provenance={
                    "packet_id": packet.packet_id,
                    "policy_id": self._descriptor.policy_id,
                    "proposal_id": proposal.proposal_id,
                    "evidence_class": effective_evidence_class.value,
                },
                measured_creation_cost=WorkVector(event_formation=1),
                evidence_class=effective_evidence_class,
            )
            for proposal in decision.proposals
        )
        hypotheses = tuple(sorted(hypotheses, key=lambda row: str(row.event_id)))
        event_count = 1 + len(hypotheses)
        self._charge_at(
            reason="event-former-event-publication-attempt",
            work=WorkVector(
                event_formation=event_count,
                indexing_and_graph_maintenance=event_count,
            ),
            tick=packet.arrival_tick,
        )
        self._events.append_batch((observation, *hypotheses))
        return EventFormerResult(
            packet_id=packet.packet_id,
            observation_event_id=str(observation.event_id),
            hypothesis_event_ids=tuple(str(row.event_id) for row in hypotheses),
            evidence_class=effective_evidence_class,
            admitted=True,
            lifecycle_start_sequence=start_sequence,
            lifecycle_end_sequence=self._lifecycle.entry_count,
            retained_state_bytes=decision.retained_state_bytes,
        )

    def finalize(self, *, end_tick: int) -> None:
        """Close the deployment horizon, charge final retention, and release policy state."""

        if self._closed:
            raise EventFormerContractError("event former was already finalized")
        _require_nonnegative_int(end_tick, "event-former final end_tick")
        if end_tick < self._last_tick:
            self._charge_at(
                reason="event-former-finalize-time-order-rejection",
                work=WorkVector(event_formation=1),
                tick=self._last_tick,
            )
            raise EventFormerContractError("finalization cannot move event-former time backwards")
        self._assert_policy_unchanged(tick=self._last_tick)
        if self._started and end_tick > self._last_tick and self._retained_state_bytes:
            self._lifecycle.charge_retention(
                owner=self._owner,
                reason="event-former-final-retained-state",
                retained_bytes=self._retained_state_bytes,
                start_tick=self._last_tick,
                end_tick=end_tick,
            )
        self._last_tick = end_tick
        self._policy = None
        self._retained_state_bytes = 0
        self._closed = True


__all__ = [
    "ChargedEventFormer",
    "EVENT_FORMER_DECISION_SCHEMA",
    "EVENT_FORMER_RESULT_SCHEMA",
    "EVENT_PROPOSAL_SCHEMA",
    "RAW_PACKET_SCHEMA",
    "EvidenceClass",
    "EventFormerConfig",
    "EventFormerContractError",
    "EventFormerDecision",
    "EventFormerDescriptor",
    "EventFormerPolicy",
    "EventFormerResult",
    "EventProposal",
    "RawPacket",
]
