
from __future__ import annotations

import base64
import hashlib
import math
import re
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from mop.substrate.events import BranchRef, EventRef, canonical_bytes, canonical_sha256

from .events import EvidenceClass, state_version_for_parents
from .messages import ClaimMessage, EpistemicStatus

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _require_digest(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be an immutable tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique values")
    for value in values:
        _require_text(value, label)


def _require_canonical(values: tuple[str, ...], label: str) -> None:
    _require_unique(values, label)
    if values != tuple(sorted(values)):
        raise ValueError(f"{label} must use canonical sorted order")


@dataclass(frozen=True, slots=True)
class DispatchEventHeader:

    event_id: str
    event_kind: str
    branch_id: str
    producer_state_version: str
    epistemic_status: EpistemicStatus
    evidence_class: EvidenceClass
    referent_hypotheses: tuple[str, ...]
    factor_scope: tuple[str, ...]
    routing_shards: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    created_tick: int
    expiry_tick: int
    payload_digest: str
    representation_payload_digest: str
    endogenous: bool = False
    reasoning_depth: int = 0

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.event_kind, "event_kind")
        if self.event_kind != "hypothesis":
            raise ValueError("dispatch events must be hypothesis events")
        _require_text(self.branch_id, "branch_id")
        EventRef(self.event_id)
        BranchRef(self.branch_id)
        _require_digest(self.producer_state_version, "producer_state_version")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("epistemic_status must be an EpistemicStatus")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("evidence_class must be an EvidenceClass")
        _require_canonical(self.referent_hypotheses, "referent_hypotheses")
        _require_canonical(self.factor_scope, "factor_scope")
        _require_unique(self.routing_shards, "routing_shards")
        _require_canonical(self.source_event_ids, "source_event_ids")
        parent_ids = tuple(EventRef(event_id) for event_id in self.source_event_ids)
        if self.producer_state_version != state_version_for_parents(parent_ids):
            raise ValueError("event state version does not bind its complete source-event set")
        if self.created_tick < 0 or self.expiry_tick < self.created_tick:
            raise ValueError("event tick interval is invalid")
        _require_digest(self.payload_digest, "payload_digest")
        _require_digest(self.representation_payload_digest, "representation_payload_digest")
        if self.reasoning_depth < 0:
            raise ValueError("reasoning_depth must be nonnegative")
        if self.endogenous != (self.reasoning_depth > 0):
            raise ValueError("endogenous events must have positive reasoning depth and vice versa")
        if self.epistemic_status is EpistemicStatus.SIMULATED and self.branch_id == "branch:factual":
            raise ValueError("simulated dispatch events require a counterfactual branch")
        if self.epistemic_status is not EpistemicStatus.SIMULATED and self.branch_id != "branch:factual":
            raise ValueError("counterfactual dispatch events must retain simulated status")

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "branch_id": self.branch_id,
            "producer_state_version": self.producer_state_version,
            "epistemic_status": self.epistemic_status.value,
            "evidence_class": self.evidence_class.value,
            "referent_hypotheses": list(self.referent_hypotheses),
            "factor_scope": list(self.factor_scope),
            "routing_shards": list(self.routing_shards),
            "source_event_ids": list(self.source_event_ids),
            "created_tick": self.created_tick,
            "expiry_tick": self.expiry_tick,
            "payload_digest": self.payload_digest,
            "representation_payload_digest": self.representation_payload_digest,
            "endogenous": self.endogenous,
            "reasoning_depth": self.reasoning_depth,
        }


@dataclass(frozen=True, slots=True)
class DispatchEvent:

    header: DispatchEventHeader
    payload_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("event payload must be bytes")
        if hashlib.sha256(self.payload_bytes).hexdigest() != self.header.representation_payload_digest:
            raise ValueError("event payload digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        event_kind: str,
        branch_id: str,
        producer_state_version: str,
        epistemic_status: EpistemicStatus,
        referent_hypotheses: tuple[str, ...],
        factor_scope: tuple[str, ...],
        routing_shards: tuple[str, ...],
        source_event_ids: tuple[str, ...],
        created_tick: int,
        expiry_tick: int,
        payload_bytes: bytes,
        endogenous: bool = False,
        reasoning_depth: int = 0,
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
    ) -> DispatchEvent:
        header = DispatchEventHeader(
            event_id=event_id,
            event_kind=event_kind,
            branch_id=branch_id,
            producer_state_version=producer_state_version,
            epistemic_status=epistemic_status,
            evidence_class=evidence_class,
            referent_hypotheses=tuple(sorted(referent_hypotheses)),
            factor_scope=tuple(sorted(factor_scope)),
            routing_shards=routing_shards,
            source_event_ids=tuple(sorted(source_event_ids)),
            created_tick=created_tick,
            expiry_tick=expiry_tick,
            payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
            representation_payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
            endogenous=endogenous,
            reasoning_depth=reasoning_depth,
        )
        return cls(header=header, payload_bytes=payload_bytes)


@dataclass(frozen=True, slots=True)
class ActorDescriptor:

    actor_id: str
    subscribed_event_types: tuple[str, ...]
    factor_scopes: tuple[str, ...] = ()
    shard_ids: tuple[str, ...] = ()
    declared_peer_ids: tuple[str, ...] = ()
    peer_entry: bool = False

    def __post_init__(self) -> None:
        _require_text(self.actor_id, "actor_id")
        if not self.subscribed_event_types:
            raise ValueError("actor must subscribe to at least one event type")
        _require_unique(self.subscribed_event_types, "subscribed_event_types")
        _require_unique(self.factor_scopes, "factor_scopes")
        _require_unique(self.shard_ids, "shard_ids")
        _require_unique(self.declared_peer_ids, "declared_peer_ids")
        if self.actor_id in self.declared_peer_ids:
            raise ValueError("actor cannot nominate itself as a declared peer")

    def header_compatible(self, header: DispatchEventHeader) -> bool:
        if header.event_kind not in self.subscribed_event_types:
            return False
        return not self.factor_scopes or bool(set(self.factor_scopes) & set(header.factor_scope))


@dataclass(frozen=True, slots=True)
class ReadinessEstimate:

    actor_id: str
    state_version: str
    compatible: bool
    expected_decision_value: float
    predicted_operations: int
    predicted_message_bytes: int
    stale_state_risk: float = 0.0
    deadline_risk: float = 0.0
    nominated_actor_ids: tuple[str, ...] = ()
    estimation_operations: int = 1

    def __post_init__(self) -> None:
        _require_text(self.actor_id, "actor_id")
        _require_digest(self.state_version, "state_version")
        finite = (
            self.expected_decision_value,
            self.stale_state_risk,
            self.deadline_risk,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("readiness values must be finite")
        if self.predicted_operations < 0 or self.predicted_message_bytes < 0:
            raise ValueError("predicted costs must be nonnegative")
        if self.stale_state_risk < 0.0 or self.deadline_risk < 0.0:
            raise ValueError("readiness risks must be nonnegative")
        _require_unique(self.nominated_actor_ids, "nominated_actor_ids")
        if self.actor_id in self.nominated_actor_ids:
            raise ValueError("readiness cannot nominate its own actor")
        if self.estimation_operations <= 0:
            raise ValueError("every readiness estimate must declare positive work")


@dataclass(frozen=True, slots=True)
class OutboundClaim:

    message: ClaimMessage
    recipient_actor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.recipient_actor_ids:
            raise ValueError("outbound claim must name at least one recipient")
        _require_unique(self.recipient_actor_ids, "recipient_actor_ids")
        if self.message.header.producer_actor_id in self.recipient_actor_ids:
            raise ValueError("an actor cannot send a coalition claim to itself")


@dataclass(frozen=True, slots=True)
class ActionIntent:

    action_id: str
    source_event_id: str
    branch_id: str
    referent_hypotheses: tuple[str, ...]
    epistemic_status: EpistemicStatus
    evidence_class: EvidenceClass
    producer_actor_id: str
    producer_state_version: str
    created_tick: int
    expiry_tick: int
    producer_operations: int
    payload_form: str
    payload_digest: str
    payload_bytes: bytes

    def __post_init__(self) -> None:
        _require_digest(self.action_id, "action_id")
        _require_text(self.source_event_id, "source_event_id")
        _require_text(self.branch_id, "branch_id")
        EventRef(self.source_event_id)
        BranchRef(self.branch_id)
        _require_canonical(self.referent_hypotheses, "referent_hypotheses")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("action epistemic status must be typed")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("action evidence class must be typed")
        _require_text(self.producer_actor_id, "producer_actor_id")
        _require_digest(self.producer_state_version, "producer_state_version")
        if self.created_tick < 0 or self.expiry_tick < self.created_tick:
            raise ValueError("action tick interval is invalid")
        if self.producer_operations < 0:
            raise ValueError("producer_operations must be nonnegative")
        _require_text(self.payload_form, "payload_form")
        _require_digest(self.payload_digest, "payload_digest")
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("action payload must be bytes")
        if self.epistemic_status is EpistemicStatus.SIMULATED and self.branch_id == "branch:factual":
            raise ValueError("simulated action intents require a counterfactual branch")
        if self.epistemic_status is not EpistemicStatus.SIMULATED and self.branch_id != "branch:factual":
            raise ValueError("counterfactual action intents must retain simulated status")

    def identity_payload(self) -> dict[str, object]:
        return {
            "source_event_id": self.source_event_id,
            "branch_id": self.branch_id,
            "referent_hypotheses": list(self.referent_hypotheses),
            "epistemic_status": self.epistemic_status.value,
            "evidence_class": self.evidence_class.value,
            "producer_actor_id": self.producer_actor_id,
            "producer_state_version": self.producer_state_version,
            "created_tick": self.created_tick,
            "expiry_tick": self.expiry_tick,
            "producer_operations": self.producer_operations,
            "payload_form": self.payload_form,
            "payload_digest": self.payload_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        source_event_id: str,
        branch_id: str,
        referent_hypotheses: tuple[str, ...],
        epistemic_status: EpistemicStatus,
        producer_actor_id: str,
        producer_state_version: str,
        created_tick: int,
        expiry_tick: int,
        producer_operations: int,
        payload_form: str,
        payload_bytes: bytes,
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
    ) -> ActionIntent:
        placeholder = "0" * 64
        row = cls(
            action_id=placeholder,
            source_event_id=source_event_id,
            branch_id=branch_id,
            referent_hypotheses=tuple(sorted(referent_hypotheses)),
            epistemic_status=epistemic_status,
            evidence_class=evidence_class,
            producer_actor_id=producer_actor_id,
            producer_state_version=producer_state_version,
            created_tick=created_tick,
            expiry_tick=expiry_tick,
            producer_operations=producer_operations,
            payload_form=payload_form,
            payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
            payload_bytes=payload_bytes,
        )
        identity = {
            "header": row.identity_payload(),
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        }
        return replace(row, action_id=canonical_sha256(identity))

    @property
    def encoded_bytes(self) -> int:
        return len(
            canonical_bytes(
                {
                    "action_id": self.action_id,
                    "header": self.identity_payload(),
                    "payload_base64": base64.b64encode(self.payload_bytes).decode("ascii"),
                }
            )
        )

    def integrity_valid(self) -> bool:
        expected = canonical_sha256(
            {
                "header": self.identity_payload(),
                "payload_base64": base64.b64encode(self.payload_bytes).decode("ascii"),
            }
        )
        return (
            hashlib.sha256(self.payload_bytes).hexdigest() == self.payload_digest
            and expected == self.action_id
        )


@dataclass(frozen=True, slots=True)
class ActorActivationContext:
    event_header: DispatchEventHeader
    event_payload_bytes: bytes
    incoming_claims: tuple[ClaimMessage, ...]
    reasoning_round: int


@dataclass(frozen=True, slots=True)
class EndogenousHypothesisProposal:

    proposal_id: str
    producer_actor_id: str
    producer_actor_state_version: str
    source_event_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    branch_id: str
    epistemic_status: EpistemicStatus
    evidence_class: EvidenceClass
    referent_hypotheses: tuple[str, ...]
    factor_scope: tuple[str, ...]
    routing_shards: tuple[str, ...]
    created_tick: int
    expiry_tick: int
    calibrated_confidence: float
    predicted_value_of_further_computation: float
    producer_operations: int
    payload_form: str
    representation_payload_digest: str
    payload_bytes: bytes

    def __post_init__(self) -> None:
        _require_digest(self.proposal_id, "proposal_id")
        _require_text(self.producer_actor_id, "producer_actor_id")
        _require_digest(self.producer_actor_state_version, "producer_actor_state_version")
        if not self.source_event_ids:
            raise ValueError("endogenous proposal requires at least one source event")
        _require_canonical(self.source_event_ids, "source_event_ids")
        _require_canonical(self.supporting_event_ids, "supporting_event_ids")
        if not set(self.supporting_event_ids) <= set(self.source_event_ids):
            raise ValueError("supporting events must be declared source events")
        BranchRef(self.branch_id)
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("proposal epistemic status must be typed")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("proposal evidence class must be typed")
        _require_canonical(self.referent_hypotheses, "referent_hypotheses")
        _require_canonical(self.factor_scope, "factor_scope")
        _require_unique(self.routing_shards, "routing_shards")
        if (
            isinstance(self.created_tick, bool)
            or not isinstance(self.created_tick, int)
            or isinstance(self.expiry_tick, bool)
            or not isinstance(self.expiry_tick, int)
            or self.created_tick < 0
            or self.expiry_tick < self.created_tick
        ):
            raise ValueError("proposal tick interval is invalid")
        if not math.isfinite(self.calibrated_confidence) or not (0.0 <= self.calibrated_confidence <= 1.0):
            raise ValueError("proposal confidence must be finite and inside [0, 1]")
        if not math.isfinite(self.predicted_value_of_further_computation) or (
            self.predicted_value_of_further_computation < 0.0
        ):
            raise ValueError("proposal value of computation must be finite and nonnegative")
        if (
            isinstance(self.producer_operations, bool)
            or not isinstance(self.producer_operations, int)
            or self.producer_operations < 0
        ):
            raise ValueError("proposal producer operations must be a nonnegative integer")
        _require_text(self.payload_form, "payload_form")
        _require_digest(self.representation_payload_digest, "representation_payload_digest")
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("proposal payload must be bytes")
        if hashlib.sha256(self.payload_bytes).hexdigest() != self.representation_payload_digest:
            raise ValueError("proposal representation payload digest mismatch")
        if self.epistemic_status is EpistemicStatus.SIMULATED and self.branch_id == "branch:factual":
            raise ValueError("simulated proposal requires a counterfactual branch")
        if self.epistemic_status is not EpistemicStatus.SIMULATED and self.branch_id != "branch:factual":
            raise ValueError("counterfactual proposal must retain simulated status")
        if canonical_sha256(self.identity_payload()) != self.proposal_id:
            raise ValueError("endogenous proposal identity digest mismatch")

    def identity_payload(self) -> dict[str, object]:
        return {
            "producer_actor_id": self.producer_actor_id,
            "producer_actor_state_version": self.producer_actor_state_version,
            "source_event_ids": list(self.source_event_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "branch_id": self.branch_id,
            "epistemic_status": self.epistemic_status.value,
            "evidence_class": self.evidence_class.value,
            "referent_hypotheses": list(self.referent_hypotheses),
            "factor_scope": list(self.factor_scope),
            "routing_shards": list(self.routing_shards),
            "created_tick": self.created_tick,
            "expiry_tick": self.expiry_tick,
            "calibrated_confidence": self.calibrated_confidence,
            "predicted_value_of_further_computation": self.predicted_value_of_further_computation,
            "producer_operations": self.producer_operations,
            "payload_form": self.payload_form,
            "representation_payload_digest": self.representation_payload_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        producer_actor_id: str,
        producer_actor_state_version: str,
        source_event_ids: tuple[str, ...],
        supporting_event_ids: tuple[str, ...],
        branch_id: str,
        epistemic_status: EpistemicStatus,
        evidence_class: EvidenceClass,
        referent_hypotheses: tuple[str, ...],
        factor_scope: tuple[str, ...],
        routing_shards: tuple[str, ...],
        created_tick: int,
        expiry_tick: int,
        calibrated_confidence: float,
        predicted_value_of_further_computation: float,
        producer_operations: int,
        payload_form: str,
        payload_bytes: bytes,
    ) -> EndogenousHypothesisProposal:
        canonical_sources = tuple(sorted(source_event_ids))
        canonical_support = tuple(sorted(supporting_event_ids))
        canonical_referents = tuple(sorted(referent_hypotheses))
        canonical_factors = tuple(sorted(factor_scope))
        representation_digest = hashlib.sha256(payload_bytes).hexdigest()
        identity = {
            "producer_actor_id": producer_actor_id,
            "producer_actor_state_version": producer_actor_state_version,
            "source_event_ids": list(canonical_sources),
            "supporting_event_ids": list(canonical_support),
            "branch_id": branch_id,
            "epistemic_status": epistemic_status.value,
            "evidence_class": evidence_class.value,
            "referent_hypotheses": list(canonical_referents),
            "factor_scope": list(canonical_factors),
            "routing_shards": list(routing_shards),
            "created_tick": created_tick,
            "expiry_tick": expiry_tick,
            "calibrated_confidence": calibrated_confidence,
            "predicted_value_of_further_computation": predicted_value_of_further_computation,
            "producer_operations": producer_operations,
            "payload_form": payload_form,
            "representation_payload_digest": representation_digest,
        }
        return cls(
            proposal_id=canonical_sha256(identity),
            producer_actor_id=producer_actor_id,
            producer_actor_state_version=producer_actor_state_version,
            source_event_ids=canonical_sources,
            supporting_event_ids=canonical_support,
            branch_id=branch_id,
            epistemic_status=epistemic_status,
            evidence_class=evidence_class,
            referent_hypotheses=canonical_referents,
            factor_scope=canonical_factors,
            routing_shards=routing_shards,
            created_tick=created_tick,
            expiry_tick=expiry_tick,
            calibrated_confidence=calibrated_confidence,
            predicted_value_of_further_computation=predicted_value_of_further_computation,
            producer_operations=producer_operations,
            payload_form=payload_form,
            representation_payload_digest=representation_digest,
            payload_bytes=payload_bytes,
        )

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.identity_payload())) + len(self.payload_bytes)


@dataclass(frozen=True, slots=True)
class ActorActivationResult:
    outbound_claims: tuple[OutboundClaim, ...] = ()
    action_intents: tuple[ActionIntent, ...] = ()
    endogenous_proposals: tuple[EndogenousHypothesisProposal, ...] = ()
    executed_operations: int = 0

    def __post_init__(self) -> None:
        if not all(
            isinstance(values, tuple)
            for values in (self.outbound_claims, self.action_intents, self.endogenous_proposals)
        ):
            raise ValueError("actor activation outputs must be immutable tuples")
        if self.executed_operations <= 0:
            raise ValueError("every actor activation must declare positive executed operations")


@dataclass(frozen=True, slots=True)
class ActorUpdateContext:
    trace_id: str
    activated_event_ids: tuple[str, ...]
    emitted_claim_ids: tuple[str, ...]
    emitted_action_ids: tuple[str, ...]
    consequence_event_id: str
    authorization_id: str
    branch_id: str
    consequence_payload_bytes: bytes

    def __post_init__(self) -> None:
        _require_digest(self.trace_id, "trace_id")
        _require_canonical(self.activated_event_ids, "activated_event_ids")
        _require_canonical(self.emitted_claim_ids, "emitted_claim_ids")
        _require_canonical(self.emitted_action_ids, "emitted_action_ids")
        _require_text(self.consequence_event_id, "consequence_event_id")
        _require_digest(self.authorization_id, "authorization_id")
        _require_text(self.branch_id, "branch_id")
        EventRef(self.consequence_event_id)
        BranchRef(self.branch_id)
        if not isinstance(self.consequence_payload_bytes, bytes):
            raise TypeError("consequence payload must be bytes")

    @property
    def idempotency_key(self) -> str:
        return canonical_sha256(
            {
                "trace_id": self.trace_id,
                "consequence_event_id": self.consequence_event_id,
                "authorization_id": self.authorization_id,
                "branch_id": self.branch_id,
            }
        )


@dataclass(frozen=True, slots=True)
class ActorUpdatePlan:

    actor_id: str
    prior_state_version: str
    next_state_version: str
    idempotency_key: str
    executed_operations: int
    replacement_actor: object

    def __post_init__(self) -> None:
        _require_text(self.actor_id, "actor_id")
        _require_digest(self.prior_state_version, "prior_state_version")
        _require_digest(self.next_state_version, "next_state_version")
        _require_digest(self.idempotency_key, "idempotency_key")
        if self.executed_operations <= 0:
            raise ValueError("every actor update plan must declare positive executed operations")
        if self.replacement_actor is None:
            raise ValueError("actor update plan requires a replacement actor")


@runtime_checkable
class Actor(Protocol):

    @property
    def descriptor(self) -> ActorDescriptor: ...

    @property
    def state_version(self) -> str: ...

    @property
    def retained_state_bytes(self) -> int: ...

    def readiness(self, header: DispatchEventHeader) -> ReadinessEstimate: ...

    def activate(self, context: ActorActivationContext) -> ActorActivationResult: ...

    def stage_update(self, context: ActorUpdateContext) -> ActorUpdatePlan: ...


__all__ = [
    "ActionIntent",
    "Actor",
    "ActorActivationContext",
    "ActorActivationResult",
    "ActorDescriptor",
    "ActorUpdateContext",
    "ActorUpdatePlan",
    "DispatchEvent",
    "DispatchEventHeader",
    "EndogenousHypothesisProposal",
    "OutboundClaim",
    "ReadinessEstimate",
]
