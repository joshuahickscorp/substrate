
from __future__ import annotations

import base64
import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from mop.substrate.events import BranchRef, EventRef, canonical_bytes, canonical_sha256

from .events import EpistemicStatus, EvidenceClass

CLAIM_MESSAGE_SCHEMA = "mop-escs-claim-message/v1"
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
        raise ValueError(f"{label} must be unique")
    for value in values:
        _require_text(value, label)


def _require_canonical(values: tuple[str, ...], label: str) -> None:
    _require_unique(values, label)
    if values != tuple(sorted(values)):
        raise ValueError(f"{label} must use canonical sorted order")


def epistemic_rank(status: EpistemicStatus) -> int:

    return {
        EpistemicStatus.OBSERVED_CANDIDATE: 0,
        EpistemicStatus.INFERRED: 1,
        EpistemicStatus.SIMULATED: 2,
    }[status]


@dataclass(frozen=True, slots=True)
class ClaimSchema:

    schema_id: str
    version: int
    claim_types: frozenset[str]
    payload_forms: frozenset[str]
    epistemic_statuses: frozenset[EpistemicStatus] = field(default_factory=lambda: frozenset(EpistemicStatus))
    max_payload_bytes: int = 64 * 1024
    allow_empty_referents: bool = False

    def __post_init__(self) -> None:
        _require_text(self.schema_id, "schema_id")
        if self.version <= 0:
            raise ValueError("schema version must be positive")
        if not all(
            isinstance(values, frozenset)
            for values in (self.claim_types, self.payload_forms, self.epistemic_statuses)
        ):
            raise ValueError("schema sets must be immutable frozensets")
        if not self.claim_types or not self.payload_forms or not self.epistemic_statuses:
            raise ValueError("schema claim types, payload forms, and epistemic statuses must be nonempty")
        for value in self.claim_types:
            _require_text(value, "claim_type")
        for value in self.payload_forms:
            _require_text(value, "payload_form")
        if any(not isinstance(status, EpistemicStatus) for status in self.epistemic_statuses):
            raise ValueError("schema contains an unsupported epistemic status")
        if self.max_payload_bytes < 0:
            raise ValueError("max_payload_bytes must be nonnegative")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "claim_types": sorted(self.claim_types),
            "payload_forms": sorted(self.payload_forms),
            "epistemic_statuses": sorted(status.value for status in self.epistemic_statuses),
            "max_payload_bytes": self.max_payload_bytes,
            "allow_empty_referents": self.allow_empty_referents,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload())


class SchemaRegistry:

    def __init__(self, schemas: Iterable[ClaimSchema]):
        rows = tuple(schemas)
        by_key: dict[tuple[str, int], ClaimSchema] = {}
        for schema in rows:
            key = (schema.schema_id, schema.version)
            if key in by_key:
                raise ValueError(f"duplicate claim schema {key!r}")
            by_key[key] = schema
        self._schemas = rows
        self._by_key: Mapping[tuple[str, int], ClaimSchema] = MappingProxyType(by_key)

    @property
    def schemas(self) -> tuple[ClaimSchema, ...]:
        return self._schemas

    def get(self, schema_id: str, version: int) -> ClaimSchema | None:
        return self._by_key.get((schema_id, version))


@dataclass(frozen=True, slots=True)
class ClaimHeader:

    message_id: str
    wire_schema: str
    claim_schema_id: str
    claim_schema_version: int
    claim_schema_digest: str
    evidence_class: EvidenceClass
    source_hypothesis_event_ids: tuple[str, ...]
    referent_hypotheses: tuple[str, ...]
    branch_id: str
    factor_scope: tuple[str, ...]
    claim_type: str
    epistemic_status: EpistemicStatus
    supporting_event_ids: tuple[str, ...]
    producer_actor_id: str
    producer_state_version: str
    calibrated_confidence: float
    created_tick: int
    expiry_tick: int
    predicted_utility: tuple[float, ...]
    producer_operations: int
    payload_form: str
    payload_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.message_id, "message_id")
        if self.wire_schema != CLAIM_MESSAGE_SCHEMA:
            raise ValueError(f"unsupported claim wire schema {self.wire_schema!r}")
        _require_text(self.claim_schema_id, "claim_schema_id")
        if self.claim_schema_version <= 0:
            raise ValueError("claim_schema_version must be positive")
        _require_digest(self.claim_schema_digest, "claim_schema_digest")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("evidence_class must be an EvidenceClass")
        if not self.source_hypothesis_event_ids:
            raise ValueError("a claim must name at least one source hypothesis event")
        _require_canonical(self.source_hypothesis_event_ids, "source_hypothesis_event_ids")
        for event_id in self.source_hypothesis_event_ids:
            EventRef(event_id)
        _require_canonical(self.referent_hypotheses, "referent_hypotheses")
        _require_text(self.branch_id, "branch_id")
        BranchRef(self.branch_id)
        _require_canonical(self.factor_scope, "factor_scope")
        _require_text(self.claim_type, "claim_type")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("epistemic_status must be an EpistemicStatus")
        _require_canonical(self.supporting_event_ids, "supporting_event_ids")
        for event_id in self.supporting_event_ids:
            EventRef(event_id)
        _require_text(self.producer_actor_id, "producer_actor_id")
        _require_digest(self.producer_state_version, "producer_state_version")
        if not math.isfinite(self.calibrated_confidence) or not 0.0 <= self.calibrated_confidence <= 1.0:
            raise ValueError("calibrated_confidence must be finite and inside [0, 1]")
        if self.created_tick < 0 or self.expiry_tick < self.created_tick:
            raise ValueError("claim tick interval is invalid")
        if any(not math.isfinite(value) for value in self.predicted_utility):
            raise ValueError("predicted_utility must contain finite values")
        if self.producer_operations < 0:
            raise ValueError("producer_operations must be nonnegative")
        _require_text(self.payload_form, "payload_form")
        _require_digest(self.payload_digest, "payload_digest")

    def identity_payload(self) -> dict[str, Any]:

        return {
            "wire_schema": self.wire_schema,
            "claim_schema_id": self.claim_schema_id,
            "claim_schema_version": self.claim_schema_version,
            "claim_schema_digest": self.claim_schema_digest,
            "evidence_class": self.evidence_class.value,
            "source_hypothesis_event_ids": list(self.source_hypothesis_event_ids),
            "referent_hypotheses": list(self.referent_hypotheses),
            "branch_id": self.branch_id,
            "factor_scope": list(self.factor_scope),
            "claim_type": self.claim_type,
            "epistemic_status": self.epistemic_status.value,
            "supporting_event_ids": list(self.supporting_event_ids),
            "producer_actor_id": self.producer_actor_id,
            "producer_state_version": self.producer_state_version,
            "calibrated_confidence": self.calibrated_confidence,
            "created_tick": self.created_tick,
            "expiry_tick": self.expiry_tick,
            "predicted_utility": list(self.predicted_utility),
            "producer_operations": self.producer_operations,
            "payload_form": self.payload_form,
            "payload_digest": self.payload_digest,
        }

    def payload(self) -> dict[str, Any]:
        return {"message_id": self.message_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class ClaimMessage:

    header: ClaimHeader
    payload_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("claim payload must be bytes")

    @staticmethod
    def _identity_digest(identity_payload: Mapping[str, Any], payload_bytes: bytes) -> str:
        body = {
            "header": dict(identity_payload),
            "payload_base64": base64.b64encode(payload_bytes).decode("ascii"),
        }
        return canonical_sha256(body)

    @classmethod
    def create(
        cls,
        *,
        schema: ClaimSchema,
        source_hypothesis_event_ids: Iterable[str],
        referent_hypotheses: Iterable[str],
        branch_id: str,
        factor_scope: Iterable[str],
        claim_type: str,
        epistemic_status: EpistemicStatus,
        supporting_event_ids: Iterable[str],
        producer_actor_id: str,
        producer_state_version: str,
        calibrated_confidence: float,
        created_tick: int,
        expiry_tick: int,
        predicted_utility: Iterable[float],
        producer_operations: int,
        payload_form: str,
        payload_bytes: bytes,
        evidence_class: EvidenceClass = EvidenceClass.SCRIPTED_MECHANICS,
    ) -> ClaimMessage:
        payload_digest = hashlib.sha256(payload_bytes).hexdigest()
        placeholder = "0" * 64
        header = ClaimHeader(
            message_id=placeholder,
            wire_schema=CLAIM_MESSAGE_SCHEMA,
            claim_schema_id=schema.schema_id,
            claim_schema_version=schema.version,
            claim_schema_digest=schema.digest,
            evidence_class=evidence_class,
            source_hypothesis_event_ids=tuple(sorted(source_hypothesis_event_ids)),
            referent_hypotheses=tuple(sorted(referent_hypotheses)),
            branch_id=branch_id,
            factor_scope=tuple(sorted(factor_scope)),
            claim_type=claim_type,
            epistemic_status=epistemic_status,
            supporting_event_ids=tuple(sorted(supporting_event_ids)),
            producer_actor_id=producer_actor_id,
            producer_state_version=producer_state_version,
            calibrated_confidence=float(calibrated_confidence),
            created_tick=created_tick,
            expiry_tick=expiry_tick,
            predicted_utility=tuple(float(value) for value in predicted_utility),
            producer_operations=producer_operations,
            payload_form=payload_form,
            payload_digest=payload_digest,
        )
        message_id = cls._identity_digest(header.identity_payload(), payload_bytes)
        sealed = replace(header, message_id=message_id)
        return cls(header=sealed, payload_bytes=payload_bytes)

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.wire_payload()))

    def wire_payload(self) -> dict[str, Any]:
        return {
            "header": self.header.payload(),
            "payload_base64": base64.b64encode(self.payload_bytes).decode("ascii"),
        }

    def integrity_valid(self) -> bool:
        return (
            hashlib.sha256(self.payload_bytes).hexdigest() == self.header.payload_digest
            and self._identity_digest(self.header.identity_payload(), self.payload_bytes)
            == self.header.message_id
        )


@dataclass(frozen=True, slots=True)
class EventClaimEvidence:

    event_id: str
    event_kind: str
    evidence_class: EvidenceClass
    branch_id: str
    epistemic_status: EpistemicStatus
    created_tick: int

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.event_kind, "event_kind")
        if self.event_kind not in {"observation", "hypothesis", "commitment", "consequence"}:
            raise ValueError("event evidence kind is unsupported")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("event evidence class must be typed")
        _require_text(self.branch_id, "branch_id")
        EventRef(self.event_id)
        BranchRef(self.branch_id)
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("event evidence requires a typed epistemic status")
        if self.created_tick < 0:
            raise ValueError("event evidence created_tick must be nonnegative")


@dataclass(frozen=True, slots=True)
class ClaimValidationContext:

    now_tick: int
    branch_id: str
    factual_branch_id: str
    allowed_referents: frozenset[str]
    allowed_factor_scopes: frozenset[str]
    event_evidence: tuple[EventClaimEvidence, ...]
    accepted_producer_state_versions: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if self.now_tick < 0:
            raise ValueError("now_tick must be nonnegative")
        _require_text(self.branch_id, "branch_id")
        _require_text(self.factual_branch_id, "factual_branch_id")
        BranchRef(self.branch_id)
        BranchRef(self.factual_branch_id)
        if not isinstance(self.allowed_referents, frozenset) or not isinstance(
            self.allowed_factor_scopes, frozenset
        ):
            raise ValueError("allowed referents and factor scopes must be frozensets")
        if not isinstance(self.event_evidence, tuple) or not isinstance(
            self.accepted_producer_state_versions, tuple
        ):
            raise ValueError("validation evidence and producer states must be immutable tuples")
        for referent in self.allowed_referents:
            _require_text(referent, "allowed referent")
        for factor in self.allowed_factor_scopes:
            _require_text(factor, "allowed factor scope")
        if len({row.event_id for row in self.event_evidence}) != len(self.event_evidence):
            raise ValueError("event evidence identifiers must be unique")
        actor_ids = [actor_id for actor_id, _ in self.accepted_producer_state_versions]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("producer state rows must have unique actor identifiers")
        for actor_id, versions in self.accepted_producer_state_versions:
            _require_text(actor_id, "producer actor id")
            if not versions:
                raise ValueError("each producer must have at least one accepted state version")
            for version in versions:
                _require_digest(version, "accepted producer state version")

    @property
    def evidence_by_id(self) -> Mapping[str, EventClaimEvidence]:
        return MappingProxyType({row.event_id: row for row in self.event_evidence})

    @property
    def state_versions_by_actor(self) -> Mapping[str, frozenset[str]]:
        return MappingProxyType(
            {actor_id: frozenset(versions) for actor_id, versions in self.accepted_producer_state_versions}
        )


class ClaimFault(StrEnum):
    INTEGRITY = "integrity"
    UNKNOWN_SCHEMA = "unknown_schema"
    SCHEMA_DIGEST = "schema_digest"
    CLAIM_TYPE = "claim_type"
    PAYLOAD_FORM = "payload_form"
    PAYLOAD_SIZE = "payload_size"
    SCHEMA_EPISTEMIC_STATUS = "schema_epistemic_status"
    BRANCH = "branch"
    UNKNOWN_EVENT = "unknown_event"
    EVENT_BRANCH = "event_branch"
    SOURCE_NOT_HYPOTHESIS = "source_not_hypothesis"
    REFERENT = "referent"
    FACTOR_SCOPE = "factor_scope"
    PRODUCER_STATE = "producer_state"
    EXPIRED = "expired"
    FUTURE_DATED = "future_dated"
    CREATED_BEFORE_SOURCE = "created_before_source"
    EPISTEMIC_LAUNDERING = "epistemic_laundering"
    SIMULATION_ON_FACTUAL_BRANCH = "simulation_on_factual_branch"
    NON_SIMULATED_COUNTERFACTUAL = "non_simulated_counterfactual"
    EVIDENCE_CLASS_DOWNGRADE = "evidence_class_downgrade"


@dataclass(frozen=True, slots=True)
class ClaimValidation:
    faults: tuple[ClaimFault, ...]

    @property
    def accepted(self) -> bool:
        return not self.faults


def validate_claim(
    message: ClaimMessage,
    *,
    schemas: SchemaRegistry,
    context: ClaimValidationContext,
) -> ClaimValidation:

    faults: list[ClaimFault] = []
    header = message.header
    if not message.integrity_valid():
        faults.append(ClaimFault.INTEGRITY)

    schema = schemas.get(header.claim_schema_id, header.claim_schema_version)
    if schema is None:
        faults.append(ClaimFault.UNKNOWN_SCHEMA)
    else:
        if header.claim_schema_digest != schema.digest:
            faults.append(ClaimFault.SCHEMA_DIGEST)
        if header.claim_type not in schema.claim_types:
            faults.append(ClaimFault.CLAIM_TYPE)
        if header.payload_form not in schema.payload_forms:
            faults.append(ClaimFault.PAYLOAD_FORM)
        if len(message.payload_bytes) > schema.max_payload_bytes:
            faults.append(ClaimFault.PAYLOAD_SIZE)
        if header.epistemic_status not in schema.epistemic_statuses:
            faults.append(ClaimFault.SCHEMA_EPISTEMIC_STATUS)
        if not header.referent_hypotheses and not schema.allow_empty_referents:
            faults.append(ClaimFault.REFERENT)

    if header.branch_id != context.branch_id:
        faults.append(ClaimFault.BRANCH)
    if not set(header.referent_hypotheses) <= context.allowed_referents:
        faults.append(ClaimFault.REFERENT)
    if not set(header.factor_scope) <= context.allowed_factor_scopes:
        faults.append(ClaimFault.FACTOR_SCOPE)

    accepted_versions = context.state_versions_by_actor.get(header.producer_actor_id)
    if accepted_versions is None or header.producer_state_version not in accepted_versions:
        faults.append(ClaimFault.PRODUCER_STATE)

    if context.now_tick > header.expiry_tick:
        faults.append(ClaimFault.EXPIRED)
    if context.now_tick < header.created_tick:
        faults.append(ClaimFault.FUTURE_DATED)

    evidence_by_id = context.evidence_by_id
    evidence_rows: list[EventClaimEvidence] = []
    for event_id in header.source_hypothesis_event_ids:
        evidence = evidence_by_id.get(event_id)
        if evidence is None:
            faults.append(ClaimFault.UNKNOWN_EVENT)
            continue
        if evidence.event_kind != "hypothesis":
            faults.append(ClaimFault.SOURCE_NOT_HYPOTHESIS)
        evidence_rows.append(evidence)
        allowed_source_branches = {header.branch_id}
        if header.branch_id != context.factual_branch_id:
            allowed_source_branches.add(context.factual_branch_id)
        if evidence.branch_id not in allowed_source_branches:
            faults.append(ClaimFault.EVENT_BRANCH)
    for event_id in header.supporting_event_ids:
        evidence = evidence_by_id.get(event_id)
        if evidence is None:
            faults.append(ClaimFault.UNKNOWN_EVENT)
            continue
        evidence_rows.append(evidence)
        allowed_source_branches = {header.branch_id}
        if header.branch_id != context.factual_branch_id:
            allowed_source_branches.add(context.factual_branch_id)
        if evidence.branch_id not in allowed_source_branches:
            faults.append(ClaimFault.EVENT_BRANCH)

    if evidence_rows:
        most_derived_source = max(epistemic_rank(row.epistemic_status) for row in evidence_rows)
        if epistemic_rank(header.epistemic_status) < most_derived_source:
            faults.append(ClaimFault.EPISTEMIC_LAUNDERING)
        most_restrictive_evidence = max(row.evidence_class.taint_rank for row in evidence_rows)
        if header.evidence_class.taint_rank < most_restrictive_evidence:
            faults.append(ClaimFault.EVIDENCE_CLASS_DOWNGRADE)
    source_rows = [
        evidence_by_id[event_id]
        for event_id in header.source_hypothesis_event_ids
        if event_id in evidence_by_id
    ]
    if source_rows and header.created_tick < max(row.created_tick for row in source_rows):
        faults.append(ClaimFault.CREATED_BEFORE_SOURCE)
    if header.epistemic_status is EpistemicStatus.SIMULATED and header.branch_id == context.factual_branch_id:
        faults.append(ClaimFault.SIMULATION_ON_FACTUAL_BRANCH)
    if header.epistemic_status is not EpistemicStatus.SIMULATED and (
        header.branch_id != context.factual_branch_id
    ):
        faults.append(ClaimFault.NON_SIMULATED_COUNTERFACTUAL)

    return ClaimValidation(tuple(dict.fromkeys(faults)))


__all__ = [
    "CLAIM_MESSAGE_SCHEMA",
    "ClaimFault",
    "ClaimHeader",
    "ClaimMessage",
    "ClaimSchema",
    "ClaimValidation",
    "ClaimValidationContext",
    "EpistemicStatus",
    "EvidenceClass",
    "EventClaimEvidence",
    "SchemaRegistry",
    "epistemic_rank",
    "validate_claim",
]
