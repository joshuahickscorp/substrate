"""Fail-closed joins for one ledger hypothesis, runtime trace, action, and consequence.

The chassis makes ordering and identity explicit; it is not a distributed transaction manager. A
commitment appended to a persisted/replayed EventLedger before an external callback acts as a
restart-stable, at-most-once invocation fence. A crash after that append but before the callback can
therefore omit the effect. Exactly-once effects require an external idempotent or transactional adapter
and are not claimed here. Archive publication is intentionally outside this boundary.
"""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from typing import Any, Protocol, runtime_checkable

from mop.substrate.events import BranchRef, EventRef, FrozenJSON, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from .actors import ActionIntent, DispatchEvent
from .events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    EvidenceClass,
    HypothesisEvent,
)
from .ledger import EventLedger
from .messages import epistemic_rank
from .runtime import CoalitionRuntime, RuntimeContractError, RuntimeTrace

CHASSIS_COMMITMENT_SCHEMA = "mop-escs-chassis-commitment/v1"
CHASSIS_CONSEQUENCE_SCHEMA = "mop-escs-chassis-consequence/v1"
CHASSIS_RESULT_SCHEMA = "mop-escs-chassis-result/v1"
EFFECT_AUTHORITY_SCHEMA = "mop-escs-effect-authority/v1"


class ChassisStatus(StrEnum):
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    EFFECT_FAILED = "effect-failed-after-commitment"
    COMMITTED_EFFECT_NOT_REPLAYED = "committed-effect-not-replayed"
    ALREADY_COMPLETED = "already-completed"
    CONSEQUENCE_RECORDED_UPDATE_UNAVAILABLE = "consequence-recorded-update-unavailable"


class ChassisFailpoint(StrEnum):
    AFTER_DISPATCH = "after-dispatch"
    AFTER_COMMITMENT = "after-commitment"
    AFTER_EFFECT = "after-effect"
    AFTER_CONSEQUENCE = "after-consequence"


class ChassisContractError(RuntimeError):
    """A ledger, trace, action, effect, or authority join failed closed."""


class InjectedChassisFailure(ChassisContractError):
    def __init__(self, failpoint: ChassisFailpoint):
        self.failpoint = failpoint
        super().__init__(f"injected chassis failure at {failpoint.value}")


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _freeze(value: Any) -> FrozenJSON:
    return value if isinstance(value, FrozenJSON) else FrozenJSON.from_value(value)


def _mapping_keys(value: FrozenJSON) -> tuple[str, ...]:
    decoded = value.value()
    if not isinstance(decoded, dict):
        return ()
    return tuple(sorted(str(key) for key in decoded))


def _canonical_value(value: Any) -> Any:
    """Convert a complete dataclass trace into strict canonical-JSON values."""

    if hasattr(value, "__dataclass_fields__"):
        return {name: _canonical_value(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple | list):
        return [_canonical_value(row) for row in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(row) for key, row in sorted(value.items())}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ChassisContractError(f"trace contains unsupported value type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class EffectRequest:
    effect_id: str
    commitment_event_id: str
    hypothesis_event_id: str
    trace_authority_id: str
    full_trace_sha256: str
    action_id: str
    branch_id: str
    evidence_class: EvidenceClass
    payload_form: str
    payload_digest: str
    payload_bytes: bytes

    def __post_init__(self) -> None:
        for label, value in (
            ("effect_id", self.effect_id),
            ("trace_authority_id", self.trace_authority_id),
            ("full_trace_sha256", self.full_trace_sha256),
            ("action_id", self.action_id),
            ("payload_digest", self.payload_digest),
        ):
            _require_digest(value, label)
        EventRef(self.commitment_event_id)
        EventRef(self.hypothesis_event_id)
        BranchRef(self.branch_id)
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("effect request evidence class must be typed")
        if not isinstance(self.payload_form, str) or not self.payload_form.strip():
            raise ValueError("effect payload form must not be empty")
        if not isinstance(self.payload_bytes, bytes):
            raise TypeError("effect payload must be immutable bytes")
        if hashlib.sha256(self.payload_bytes).hexdigest() != self.payload_digest:
            raise ValueError("effect payload digest mismatch")


@dataclass(frozen=True, slots=True)
class EffectOutcome:
    observed_outcome: FrozenJSON
    realized_utility_vector: FrozenJSON
    realized_full_cost: WorkVector
    delayed_or_partial: bool = False
    observation_uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.observed_outcome, FrozenJSON):
            raise ValueError("observed_outcome must be FrozenJSON")
        if not isinstance(self.realized_utility_vector, FrozenJSON):
            raise ValueError("realized_utility_vector must be FrozenJSON")
        if not isinstance(self.realized_full_cost, WorkVector):
            raise ValueError("realized_full_cost must be WorkVector")
        if not isinstance(self.delayed_or_partial, bool):
            raise ValueError("delayed_or_partial must be a boolean")
        if (
            isinstance(self.observation_uncertainty, bool)
            or not isinstance(self.observation_uncertainty, int | float)
            or not math.isfinite(float(self.observation_uncertainty))
            or self.observation_uncertainty < 0
        ):
            raise ValueError("observation_uncertainty must be finite and nonnegative")

    @classmethod
    def create(
        cls,
        *,
        observed_outcome: FrozenJSON | Any,
        realized_utility_vector: FrozenJSON | Any,
        realized_full_cost: WorkVector,
        delayed_or_partial: bool = False,
        observation_uncertainty: float = 0.0,
    ) -> EffectOutcome:
        return cls(
            observed_outcome=_freeze(observed_outcome),
            realized_utility_vector=_freeze(realized_utility_vector),
            realized_full_cost=realized_full_cost,
            delayed_or_partial=delayed_or_partial,
            observation_uncertainty=observation_uncertainty,
        )


@runtime_checkable
class ExternalEffect(Protocol):
    """Single-attempt adapter.  Retry/idempotency semantics belong to the implementation."""

    def execute(self, request: EffectRequest) -> EffectOutcome: ...


@dataclass(frozen=True, slots=True)
class ChassisResult:
    status: ChassisStatus
    hypothesis_event_id: str
    trace_authority_id: str | None
    full_trace_sha256: str | None
    action_id: str | None
    effect_id: str
    commitment_event_id: str
    consequence_event_id: str | None
    updated_actor_ids: tuple[str, ...]
    effect_invoked: bool
    resumed_from_ledger: bool
    lifecycle_start_sequence: int
    lifecycle_end_sequence: int
    failure_reason: str | None = None
    schema: str = CHASSIS_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CHASSIS_RESULT_SCHEMA:
            raise ValueError("unsupported chassis-result schema")
        if not isinstance(self.status, ChassisStatus):
            raise ValueError("chassis status must be typed")
        EventRef(self.hypothesis_event_id)
        EventRef(self.commitment_event_id)
        if self.consequence_event_id is not None:
            EventRef(self.consequence_event_id)
        _require_digest(self.effect_id, "effect_id")
        if self.trace_authority_id is not None:
            _require_digest(self.trace_authority_id, "trace_authority_id")
        if self.full_trace_sha256 is not None:
            _require_digest(self.full_trace_sha256, "full_trace_sha256")
        if self.action_id is not None:
            _require_digest(self.action_id, "action_id")
        if not isinstance(self.updated_actor_ids, tuple):
            raise ValueError("updated_actor_ids must be an immutable tuple")
        _require_nonnegative_int(self.lifecycle_start_sequence, "lifecycle_start_sequence")
        _require_nonnegative_int(self.lifecycle_end_sequence, "lifecycle_end_sequence")
        if self.lifecycle_end_sequence < self.lifecycle_start_sequence:
            raise ValueError("chassis lifecycle sequence interval is invalid")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status.value,
            "hypothesis_event_id": self.hypothesis_event_id,
            "trace_authority_id": self.trace_authority_id,
            "full_trace_sha256": self.full_trace_sha256,
            "action_id": self.action_id,
            "effect_id": self.effect_id,
            "commitment_event_id": self.commitment_event_id,
            "consequence_event_id": self.consequence_event_id,
            "updated_actor_ids": list(self.updated_actor_ids),
            "effect_invoked": self.effect_invoked,
            "resumed_from_ledger": self.resumed_from_ledger,
            "lifecycle_start_sequence": self.lifecycle_start_sequence,
            "lifecycle_end_sequence": self.lifecycle_end_sequence,
            "failure_reason": self.failure_reason,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


class EventSourcedCoalitionChassis:
    """One single-writer mechanics join over authoritative event and lifecycle ledgers."""

    _MAX_TRUSTED_REPLAY_RUNTIMES = 16

    def __init__(
        self,
        *,
        event_ledger: EventLedger,
        lifecycle_ledger: LifecycleLedger,
        runtime: CoalitionRuntime,
        owner: str = "escs.chassis",
        trusted_replay_runtime_ids: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(event_ledger, EventLedger):
            raise TypeError("chassis requires EventLedger")
        if not isinstance(lifecycle_ledger, LifecycleLedger):
            raise TypeError("chassis requires LifecycleLedger")
        if not isinstance(runtime, CoalitionRuntime):
            raise TypeError("chassis requires CoalitionRuntime")
        if runtime.lifecycle_ledger is not lifecycle_ledger:
            raise ValueError("runtime and chassis must share one authoritative LifecycleLedger")
        if runtime.event_ledger is not event_ledger:
            raise ValueError("runtime and chassis must share one authoritative EventLedger")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("chassis owner must not be empty")
        if not isinstance(trusted_replay_runtime_ids, tuple):
            raise TypeError("trusted replay runtime authorities must be an immutable tuple")
        if len(trusted_replay_runtime_ids) > self._MAX_TRUSTED_REPLAY_RUNTIMES:
            raise ValueError("too many trusted replay runtime authorities")
        accepted_runtime_ids = {runtime.runtime_id}
        for runtime_id in trusted_replay_runtime_ids:
            accepted_runtime_ids.add(_require_digest(runtime_id, "trusted replay runtime authority"))
        self._events = event_ledger
        self._lifecycle = lifecycle_ledger
        self._runtime = runtime
        self._owner = owner
        self._accepted_runtime_ids = frozenset(accepted_runtime_ids)
        self._updated_consequence_ids: set[str] = set()

    def _charge(
        self,
        *,
        reason: str,
        work: WorkVector,
        tick: int,
        branch: BranchRef,
        causal_event_ids: tuple[EventRef, ...] = (),
    ) -> None:
        self._lifecycle.charge(
            owner=self._owner,
            reason=reason,
            work=work,
            start_tick=tick,
            end_tick=tick,
            branch_id=branch,
            causal_event_ids=causal_event_ids,
        )

    def dispatch_from_hypothesis(self, event_id: EventRef) -> DispatchEvent:
        """Derive every dispatch field from one authoritative ledger-resident hypothesis."""

        event = self._events.get(event_id)
        if type(event) is not HypothesisEvent:
            raise ChassisContractError("dispatch authority must be a ledger HypothesisEvent")
        payload_bytes = canonical_bytes(event.body_payload())
        if hashlib.sha256(payload_bytes).hexdigest() != event.envelope.payload_digest:
            raise ChassisContractError("hypothesis body drifted from its envelope payload commitment")
        dispatch = DispatchEvent.create(
            event_id=str(event.event_id),
            event_kind=event.kind.value,
            branch_id=str(event.branch_id),
            producer_state_version=event.envelope.producer_state_version,
            epistemic_status=event.epistemic_status,
            evidence_class=event.evidence_class,
            referent_hypotheses=_mapping_keys(event.referent_hypotheses),
            factor_scope=_mapping_keys(event.factor_change_distribution),
            routing_shards=(),
            source_event_ids=tuple(str(row) for row in event.envelope.causal_parent_ids),
            created_tick=event.envelope.clock_end_tick,
            expiry_tick=event.envelope.clock_end_tick + event.envelope.clock_uncertainty,
            payload_bytes=payload_bytes,
        )
        if (
            dispatch.header.event_id != str(event.event_id)
            or dispatch.header.branch_id != str(event.branch_id)
            or dispatch.header.evidence_class is not event.evidence_class
            or dispatch.header.producer_state_version != event.envelope.producer_state_version
            or dispatch.header.payload_digest != event.envelope.payload_digest
        ):
            raise ChassisContractError("derived dispatch header diverged from its ledger authority")
        return dispatch

    @staticmethod
    def _trace_digest(trace: RuntimeTrace) -> str:
        validator = getattr(trace, "validate_integrity", None)
        if not callable(validator):
            raise ChassisContractError("runtime trace lacks an integrity validator")
        validation = validator()
        if validation is False:
            raise ChassisContractError("runtime trace integrity validation failed")
        full_digest = getattr(trace, "full_trace_sha256", None)
        return _require_digest(full_digest, "runtime full_trace_sha256")

    def _scan_commitment(
        self,
        hypothesis: HypothesisEvent,
    ) -> tuple[CommitmentEvent | None, int]:
        events = self._events.commitments_for(hypothesis.event_id)
        matches: list[CommitmentEvent] = []
        for event in events:
            if type(event) is not CommitmentEvent:
                continue
            payload = event.committed_payload.value()
            if not isinstance(payload, dict) or payload.get("schema") != CHASSIS_COMMITMENT_SCHEMA:
                continue
            authority = self._validate_commitment_authority(event, hypothesis)
            if authority["runtime_id"] not in self._accepted_runtime_ids:
                raise ChassisContractError("replayed commitment was issued by a different runtime authority")
            matches.append(event)
        if len(matches) > 1:
            raise ChassisContractError("multiple chassis commitments exist for one hypothesis")
        return (matches[0] if matches else None), len(events)

    @staticmethod
    def _action_record(action: ActionIntent) -> dict[str, Any]:
        return {
            "action_id": action.action_id,
            "identity": action.identity_payload(),
            "payload_base64": base64.b64encode(action.payload_bytes).decode("ascii"),
        }

    @staticmethod
    def _effect_id(*, hypothesis_event_id: str, action_id: str | None, trace_authority_id: str) -> str:
        return canonical_sha256(
            {
                "schema": EFFECT_AUTHORITY_SCHEMA,
                "hypothesis_event_id": hypothesis_event_id,
                "action_id": action_id,
                "trace_authority_id": trace_authority_id,
            }
        )

    def _select_external_action(
        self, hypothesis: HypothesisEvent, trace: RuntimeTrace
    ) -> tuple[ActionIntent | None, str, str | None]:
        factual = tuple(action for action in trace.action_intents if action.branch_id == str(FACTUAL_BRANCH))
        if len(factual) > 1:
            raise ChassisContractError("runtime returned more than one factual action authority")
        candidate = factual[0] if factual else None
        if hypothesis.branch_id != FACTUAL_BRANCH or (
            hypothesis.epistemic_status is EpistemicStatus.SIMULATED
        ):
            return (
                None,
                "simulated-hypothesis-external-effect-refused",
                (candidate.action_id if candidate else None),
            )
        if hypothesis.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE:
            return (
                None,
                "oracle-hypothesis-external-effect-refused",
                (candidate.action_id if candidate else None),
            )
        if candidate is None:
            return None, "runtime-produced-no-factual-action", None
        if (
            candidate.source_event_id != str(hypothesis.event_id)
            or candidate.branch_id != str(hypothesis.branch_id)
            or candidate.epistemic_status is EpistemicStatus.SIMULATED
            or not candidate.integrity_valid()
        ):
            raise ChassisContractError("accepted action does not bind the authoritative hypothesis")
        if candidate.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE:
            return None, "oracle-action-external-effect-refused", candidate.action_id
        if candidate.evidence_class.taint_rank < hypothesis.evidence_class.taint_rank:
            raise ChassisContractError("accepted action downgraded hypothesis evidence taint")
        return candidate, "accepted-factual-action", None

    def _append_commitment(
        self,
        *,
        hypothesis: HypothesisEvent,
        trace: RuntimeTrace,
        trace_digest: str,
        action: ActionIntent | None,
        decision_reason: str,
        blocked_action_id: str | None,
        tick: int,
    ) -> CommitmentEvent:
        effect_id = self._effect_id(
            hypothesis_event_id=str(hypothesis.event_id),
            action_id=action.action_id if action else None,
            trace_authority_id=trace.trace_id,
        )
        payload = {
            "schema": CHASSIS_COMMITMENT_SCHEMA,
            "hypothesis_event_id": str(hypothesis.event_id),
            "runtime_id": trace.runtime_id,
            "trace_authority_sequence": trace.authority_sequence,
            "trace_authority_id": trace.trace_id,
            "full_trace_sha256": trace_digest,
            "effect_id": effect_id,
            "decision_reason": decision_reason,
            "action_record": self._action_record(action) if action else None,
            "blocked_action_id": blocked_action_id,
        }
        evidence = max(
            (hypothesis.evidence_class, action.evidence_class if action else hypothesis.evidence_class),
            key=lambda row: row.taint_rank,
        )
        commitment = CommitmentEvent.create(
            coalition_id=f"coalition:{trace.trace_id}",
            commitment_kind=(CommitmentKind.EXTERNAL_ACTION if action else CommitmentKind.ABSTENTION),
            committed_payload=payload,
            decision_distribution={
                ("external_action" if action else "abstention"): 1.0,
            },
            deadline_tick=max(tick, action.expiry_tick if action else tick),
            predicted_utility_vector={"unscored": 0.0},
            predicted_full_cost=WorkVector.zero(),
            causal_parent_ids=(hypothesis.event_id,),
            counterfactual_branch_id=hypothesis.branch_id,
            clock_start_tick=tick,
            clock_end_tick=tick,
            source_and_provenance={
                "producer": "escs.chassis",
                "trace_authority_id": trace.trace_id,
            },
            measured_creation_cost=WorkVector(event_formation=1),
            evidence_class=evidence,
        )
        serialized = canonical_bytes(commitment.payload())
        self._charge(
            reason="chassis-commitment-formation-and-indexing",
            work=WorkVector(
                event_formation=1,
                indexing_and_graph_maintenance=1 + len(serialized),
            ),
            tick=tick,
            branch=hypothesis.branch_id,
            causal_event_ids=(hypothesis.event_id,),
        )
        self._events.append(commitment)
        return commitment

    @staticmethod
    def _commitment_data(commitment: CommitmentEvent) -> dict[str, Any]:
        payload = commitment.committed_payload.value()
        expected = {
            "schema",
            "hypothesis_event_id",
            "runtime_id",
            "trace_authority_sequence",
            "trace_authority_id",
            "full_trace_sha256",
            "effect_id",
            "decision_reason",
            "action_record",
            "blocked_action_id",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ChassisContractError("chassis commitment payload schema drift")
        if payload["schema"] != CHASSIS_COMMITMENT_SCHEMA:
            raise ChassisContractError("unsupported chassis commitment payload")
        for label in ("runtime_id", "trace_authority_id", "full_trace_sha256", "effect_id"):
            _require_digest(payload[label], label)
        _require_nonnegative_int(payload["trace_authority_sequence"], "trace_authority_sequence")
        expected_trace_id = canonical_sha256(
            {
                "schema": "mop-escs-runtime-trace-authority/v1",
                "runtime_id": payload["runtime_id"],
                "authority_sequence": payload["trace_authority_sequence"],
            }
        )
        if payload["trace_authority_id"] != expected_trace_id:
            raise ChassisContractError("runtime trace authority digest mismatch")
        return payload

    @staticmethod
    def _action_from_record(record: object) -> ActionIntent | None:
        if record is None:
            return None
        if not isinstance(record, dict) or set(record) != {
            "action_id",
            "identity",
            "payload_base64",
        }:
            raise ChassisContractError("committed action record schema drift")
        identity = record["identity"]
        expected_identity = {
            "source_event_id",
            "branch_id",
            "referent_hypotheses",
            "epistemic_status",
            "evidence_class",
            "producer_actor_id",
            "producer_state_version",
            "created_tick",
            "expiry_tick",
            "producer_operations",
            "payload_form",
            "payload_digest",
        }
        if not isinstance(identity, dict) or set(identity) != expected_identity:
            raise ChassisContractError("committed action identity schema drift")
        referents = identity["referent_hypotheses"]
        if not isinstance(referents, list) or not all(isinstance(row, str) for row in referents):
            raise ChassisContractError("committed action referents are invalid")
        try:
            action_bytes = base64.b64decode(record["payload_base64"], validate=True)
            action = ActionIntent(
                action_id=record["action_id"],
                source_event_id=identity["source_event_id"],
                branch_id=identity["branch_id"],
                referent_hypotheses=tuple(referents),
                epistemic_status=EpistemicStatus(identity["epistemic_status"]),
                evidence_class=EvidenceClass(identity["evidence_class"]),
                producer_actor_id=identity["producer_actor_id"],
                producer_state_version=identity["producer_state_version"],
                created_tick=identity["created_tick"],
                expiry_tick=identity["expiry_tick"],
                producer_operations=identity["producer_operations"],
                payload_form=identity["payload_form"],
                payload_digest=identity["payload_digest"],
                payload_bytes=action_bytes,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ChassisContractError("committed action record is invalid") from exc
        if not action.integrity_valid():
            raise ChassisContractError("committed action identity digest mismatch")
        return action

    @classmethod
    def _validate_commitment_authority(
        cls,
        commitment: CommitmentEvent,
        hypothesis: HypothesisEvent,
    ) -> dict[str, Any]:
        payload = cls._commitment_data(commitment)
        if commitment.envelope.causal_parent_ids != (hypothesis.event_id,):
            raise ChassisContractError("chassis commitment has the wrong causal hypothesis")
        if commitment.branch_id != hypothesis.branch_id:
            raise ChassisContractError("chassis commitment crossed its hypothesis branch")
        if payload["hypothesis_event_id"] != str(hypothesis.event_id):
            raise ChassisContractError("chassis commitment payload names the wrong hypothesis")
        if commitment.coalition_id != f"coalition:{payload['trace_authority_id']}":
            raise ChassisContractError("chassis commitment coalition/trace authority mismatch")
        if commitment.envelope.source_and_provenance.value() != {
            "producer": "escs.chassis",
            "trace_authority_id": payload["trace_authority_id"],
        }:
            raise ChassisContractError("chassis commitment provenance authority drift")
        if not isinstance(payload["decision_reason"], str) or not payload["decision_reason"].strip():
            raise ChassisContractError("chassis commitment decision reason is invalid")
        if payload["blocked_action_id"] is not None:
            _require_digest(payload["blocked_action_id"], "blocked_action_id")
        if commitment.predicted_utility_vector.value() != {"unscored": 0.0}:
            raise ChassisContractError("chassis commitment predicted utility drift")
        if commitment.predicted_full_cost != WorkVector.zero():
            raise ChassisContractError("chassis commitment predicted cost drift")

        action = cls._action_from_record(payload["action_record"])
        action_id = action.action_id if action is not None else None
        expected_effect_id = cls._effect_id(
            hypothesis_event_id=str(hypothesis.event_id),
            action_id=action_id,
            trace_authority_id=payload["trace_authority_id"],
        )
        if payload["effect_id"] != expected_effect_id:
            raise ChassisContractError("chassis commitment effect authority digest mismatch")

        if action is None:
            if commitment.commitment_kind is not CommitmentKind.ABSTENTION:
                raise ChassisContractError("actionless chassis commitment must be an abstention")
            if commitment.decision_distribution.value() != {"abstention": 1.0}:
                raise ChassisContractError("abstention commitment decision distribution drift")
            if commitment.deadline_tick != commitment.envelope.clock_end_tick:
                raise ChassisContractError("abstention commitment deadline drift")
            simulated = (
                hypothesis.branch_id != FACTUAL_BRANCH
                or hypothesis.epistemic_status is EpistemicStatus.SIMULATED
            )
            oracle_hypothesis = hypothesis.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE
            reason = payload["decision_reason"]
            blocked = payload["blocked_action_id"]
            if simulated:
                if reason != "simulated-hypothesis-external-effect-refused" or blocked is not None:
                    raise ChassisContractError("simulated abstention authority drift")
            elif oracle_hypothesis:
                if reason != "oracle-hypothesis-external-effect-refused":
                    raise ChassisContractError("oracle-hypothesis abstention authority drift")
            elif reason == "runtime-produced-no-factual-action":
                if blocked is not None:
                    raise ChassisContractError("no-action abstention cannot name a blocked action")
            elif reason == "oracle-action-external-effect-refused":
                if blocked is None:
                    raise ChassisContractError("oracle-action abstention lacks its blocked action identity")
            else:
                raise ChassisContractError("actionless commitment has an impossible decision reason")
            expected_evidence = hypothesis.evidence_class
        else:
            if commitment.commitment_kind is not CommitmentKind.EXTERNAL_ACTION:
                raise ChassisContractError("action-bearing chassis commitment has the wrong kind")
            if (
                hypothesis.branch_id != FACTUAL_BRANCH
                or hypothesis.epistemic_status is EpistemicStatus.SIMULATED
                or hypothesis.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE
            ):
                raise ChassisContractError("action-bearing commitment binds a non-executable hypothesis")
            if payload["blocked_action_id"] is not None:
                raise ChassisContractError("accepted action commitment also claims a blocked action")
            if payload["decision_reason"] != "accepted-factual-action":
                raise ChassisContractError("accepted action commitment has the wrong decision reason")
            if commitment.decision_distribution.value() != {"external_action": 1.0}:
                raise ChassisContractError("external-action decision distribution drift")
            if action.source_event_id != str(hypothesis.event_id):
                raise ChassisContractError("committed action source does not match its hypothesis")
            if action.branch_id != str(hypothesis.branch_id):
                raise ChassisContractError("committed action crossed its hypothesis branch")
            if not set(action.referent_hypotheses) <= set(_mapping_keys(hypothesis.referent_hypotheses)):
                raise ChassisContractError("committed action names an unauthorized referent")
            if action.epistemic_status is EpistemicStatus.SIMULATED:
                raise ChassisContractError("simulated action entered an external commitment")
            if epistemic_rank(action.epistemic_status) < epistemic_rank(hypothesis.epistemic_status):
                raise ChassisContractError("committed action launders hypothesis epistemic status")
            if action.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE:
                raise ChassisContractError("oracle action entered an external commitment")
            if action.evidence_class.taint_rank < hypothesis.evidence_class.taint_rank:
                raise ChassisContractError("committed action downgrades hypothesis evidence")
            if not (
                hypothesis.envelope.clock_end_tick
                <= action.created_tick
                <= commitment.envelope.clock_end_tick
                <= action.expiry_tick
            ):
                raise ChassisContractError("committed action timing is outside its authority window")
            if commitment.deadline_tick != max(commitment.envelope.clock_end_tick, action.expiry_tick):
                raise ChassisContractError("external-action commitment deadline drift")
            expected_evidence = max(
                (hypothesis.evidence_class, action.evidence_class),
                key=lambda row: row.taint_rank,
            )
        if commitment.evidence_class is not expected_evidence:
            raise ChassisContractError("chassis commitment evidence authority drift")
        return payload

    @staticmethod
    def _effect_request(commitment: CommitmentEvent, hypothesis: HypothesisEvent) -> EffectRequest | None:
        payload = EventSourcedCoalitionChassis._validate_commitment_authority(commitment, hypothesis)
        action = EventSourcedCoalitionChassis._action_from_record(payload["action_record"])
        if action is None:
            return None
        return EffectRequest(
            effect_id=payload["effect_id"],
            commitment_event_id=str(commitment.event_id),
            hypothesis_event_id=str(hypothesis.event_id),
            trace_authority_id=payload["trace_authority_id"],
            full_trace_sha256=payload["full_trace_sha256"],
            action_id=action.action_id,
            branch_id=action.branch_id,
            evidence_class=action.evidence_class,
            payload_form=action.payload_form,
            payload_digest=action.payload_digest,
            payload_bytes=action.payload_bytes,
        )

    def _scan_consequence(
        self,
        commitment: CommitmentEvent,
    ) -> tuple[ConsequenceEvent | None, int]:
        rows = self._events.consequences_for(commitment.event_id)
        matches: list[ConsequenceEvent] = []
        for row in rows:
            observed = row.observed_outcome.value()
            if not isinstance(observed, dict) or observed.get("schema") != CHASSIS_CONSEQUENCE_SCHEMA:
                continue
            self._validate_consequence_authority(row, commitment)
            matches.append(row)
        if len(matches) > 1:
            raise ChassisContractError("multiple chassis consequences exist for one commitment")
        return (matches[0] if matches else None), len(rows)

    @classmethod
    def _validate_consequence_authority(
        cls,
        consequence: ConsequenceEvent,
        commitment: CommitmentEvent,
    ) -> dict[str, Any]:
        commitment_payload = cls._commitment_data(commitment)
        observed = consequence.observed_outcome.value()
        expected_fields = {
            "schema",
            "effect_id",
            "hypothesis_event_id",
            "trace_authority_id",
            "full_trace_sha256",
            "action_id",
            "outcome",
        }
        if not isinstance(observed, dict) or set(observed) != expected_fields:
            raise ChassisContractError("chassis consequence payload schema drift")
        if observed["schema"] != CHASSIS_CONSEQUENCE_SCHEMA:
            raise ChassisContractError("unsupported chassis consequence payload")
        expected_action_id = (
            commitment_payload["action_record"]["action_id"]
            if commitment_payload["action_record"] is not None
            else None
        )
        expected_authority = {
            "effect_id": commitment_payload["effect_id"],
            "hypothesis_event_id": commitment_payload["hypothesis_event_id"],
            "trace_authority_id": commitment_payload["trace_authority_id"],
            "full_trace_sha256": commitment_payload["full_trace_sha256"],
            "action_id": expected_action_id,
        }
        if any(observed[key] != value for key, value in expected_authority.items()):
            raise ChassisContractError("chassis consequence authority does not match its commitment")
        if consequence.commitment_event_id != commitment.event_id:
            raise ChassisContractError("chassis consequence names the wrong commitment")
        if consequence.envelope.causal_parent_ids != (commitment.event_id,):
            raise ChassisContractError("chassis consequence has unauthorized causal parents")
        if consequence.branch_id != commitment.branch_id:
            raise ChassisContractError("chassis consequence crossed its commitment branch")
        if consequence.evidence_class is not commitment.evidence_class:
            raise ChassisContractError("chassis consequence evidence authority drift")
        if consequence.envelope.source_and_provenance.value() != {
            "producer": "escs.chassis",
            "effect_id": commitment_payload["effect_id"],
        }:
            raise ChassisContractError("chassis consequence provenance authority drift")
        if consequence.envelope.clock_start_tick < commitment.envelope.clock_end_tick:
            raise ChassisContractError("chassis consequence predates its commitment")
        return observed

    def _append_consequence(
        self,
        *,
        commitment: CommitmentEvent,
        outcome: EffectOutcome,
        tick: int,
    ) -> ConsequenceEvent:
        payload = self._commitment_data(commitment)
        observed = {
            "schema": CHASSIS_CONSEQUENCE_SCHEMA,
            "effect_id": payload["effect_id"],
            "hypothesis_event_id": payload["hypothesis_event_id"],
            "trace_authority_id": payload["trace_authority_id"],
            "full_trace_sha256": payload["full_trace_sha256"],
            "action_id": (
                payload["action_record"]["action_id"] if payload["action_record"] is not None else None
            ),
            "outcome": outcome.observed_outcome.value(),
        }
        consequence = ConsequenceEvent.create(
            commitment_event_id=commitment.event_id,
            observed_outcome=observed,
            realized_utility_vector=outcome.realized_utility_vector,
            delayed_or_partial=outcome.delayed_or_partial,
            observation_uncertainty=outcome.observation_uncertainty,
            realized_full_cost=outcome.realized_full_cost,
            causal_parent_ids=(commitment.event_id,),
            counterfactual_branch_id=commitment.branch_id,
            clock_start_tick=tick,
            clock_end_tick=tick,
            source_and_provenance={
                "producer": "escs.chassis",
                "effect_id": payload["effect_id"],
            },
            measured_creation_cost=WorkVector(event_formation=1),
            evidence_class=commitment.evidence_class,
        )
        serialized = canonical_bytes(consequence.payload())
        self._charge(
            reason="chassis-consequence-formation-and-indexing",
            work=WorkVector(
                event_formation=1,
                indexing_and_graph_maintenance=1 + len(serialized),
            ),
            tick=tick,
            branch=commitment.branch_id,
            causal_event_ids=(commitment.event_id,),
        )
        self._events.append(consequence)
        return consequence

    def _result(
        self,
        *,
        status: ChassisStatus,
        hypothesis: HypothesisEvent,
        commitment: CommitmentEvent,
        consequence: ConsequenceEvent | None,
        updated: tuple[str, ...],
        effect_invoked: bool,
        resumed: bool,
        start_sequence: int,
        failure_reason: str | None = None,
    ) -> ChassisResult:
        payload = self._commitment_data(commitment)
        action_record = payload["action_record"]
        return ChassisResult(
            status=status,
            hypothesis_event_id=str(hypothesis.event_id),
            trace_authority_id=payload["trace_authority_id"],
            full_trace_sha256=payload["full_trace_sha256"],
            action_id=action_record["action_id"] if action_record else None,
            effect_id=payload["effect_id"],
            commitment_event_id=str(commitment.event_id),
            consequence_event_id=(str(consequence.event_id) if consequence else None),
            updated_actor_ids=updated,
            effect_invoked=effect_invoked,
            resumed_from_ledger=resumed,
            lifecycle_start_sequence=start_sequence,
            lifecycle_end_sequence=self._lifecycle.entry_count,
            failure_reason=failure_reason,
        )

    def _apply_recorded_consequence(
        self,
        *,
        hypothesis: HypothesisEvent,
        commitment: CommitmentEvent,
        consequence: ConsequenceEvent,
        tick: int,
        start_sequence: int,
        resumed: bool,
    ) -> ChassisResult:
        self._validate_commitment_authority(commitment, hypothesis)
        self._validate_consequence_authority(consequence, commitment)
        request = self._effect_request(commitment, hypothesis)
        if request is None:
            return self._result(
                status=(ChassisStatus.ALREADY_COMPLETED if resumed else ChassisStatus.ABSTAINED),
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=consequence,
                updated=(),
                effect_invoked=False,
                resumed=resumed,
                start_sequence=start_sequence,
            )
        if str(consequence.event_id) in self._updated_consequence_ids:
            return self._result(
                status=ChassisStatus.ALREADY_COMPLETED,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=consequence,
                updated=(),
                effect_invoked=False,
                resumed=True,
                start_sequence=start_sequence,
            )
        consequence_bytes = canonical_bytes(consequence.body_payload())
        try:
            updated = self._runtime.apply_consequence(
                trace_id=request.trace_authority_id,
                consequence_event_id=str(consequence.event_id),
                authorization_id=request.action_id,
                branch_id=request.branch_id,
                consequence_payload_bytes=consequence_bytes,
                tick=tick,
            )
        except RuntimeContractError as exc:
            self._charge(
                reason="chassis-recorded-consequence-update-unavailable",
                work=WorkVector(learning=1),
                tick=tick,
                branch=commitment.branch_id,
                causal_event_ids=(consequence.event_id,),
            )
            return self._result(
                status=ChassisStatus.CONSEQUENCE_RECORDED_UPDATE_UNAVAILABLE,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=consequence,
                updated=(),
                effect_invoked=False,
                resumed=True,
                start_sequence=start_sequence,
                failure_reason=type(exc).__name__,
            )
        self._updated_consequence_ids.add(str(consequence.event_id))
        return self._result(
            status=ChassisStatus.COMPLETED,
            hypothesis=hypothesis,
            commitment=commitment,
            consequence=consequence,
            updated=updated,
            effect_invoked=False,
            resumed=resumed,
            start_sequence=start_sequence,
        )

    def execute_hypothesis(
        self,
        hypothesis_event_id: EventRef,
        *,
        effect: ExternalEffect | None,
        now_tick: int | None = None,
        failpoint: ChassisFailpoint | None = None,
    ) -> ChassisResult:
        """Run or resume one single-writer hypothesis transaction.

        Existing action commitments are never externally invoked again.  A commitment without a
        consequence is therefore reported as an unresolved omission/outcome window.
        """

        hypothesis = self._events.get(hypothesis_event_id)
        if type(hypothesis) is not HypothesisEvent:
            raise ChassisContractError("chassis authority must be a HypothesisEvent")
        existing, commitment_rows_inspected = self._scan_commitment(hypothesis)
        if existing is None:
            recorded_consequence = None
            consequence_rows_inspected = 0
        else:
            recorded_consequence, consequence_rows_inspected = self._scan_consequence(existing)
        authority_end_tick = max(
            hypothesis.envelope.clock_end_tick,
            self._runtime.last_accounted_tick,
            existing.envelope.clock_end_tick if existing is not None else 0,
            (recorded_consequence.envelope.clock_end_tick if recorded_consequence is not None else 0),
        )
        tick = authority_end_tick if now_tick is None else now_tick
        _require_nonnegative_int(tick, "now_tick")
        if tick < authority_end_tick:
            raise ChassisContractError("chassis tick predates its recorded authority frontier")
        if self._runtime.finalized:
            raise ChassisContractError("chassis runtime retention ownership is finalized")
        start_sequence = self._lifecycle.entry_count
        self._charge(
            reason="chassis-stage-control",
            work=WorkVector(indexing_and_graph_maintenance=1),
            tick=tick,
            branch=hypothesis.branch_id,
            causal_event_ids=(hypothesis.event_id,),
        )
        self._charge(
            reason="chassis-commitment-idempotency-scan",
            work=WorkVector(
                indexing_and_graph_maintenance=1 + commitment_rows_inspected,
            ),
            tick=tick,
            branch=hypothesis.branch_id,
            causal_event_ids=(hypothesis.event_id,),
        )
        if existing is not None:
            self._charge(
                reason="chassis-consequence-idempotency-scan",
                work=WorkVector(
                    indexing_and_graph_maintenance=1 + consequence_rows_inspected,
                ),
                tick=tick,
                branch=existing.branch_id,
                causal_event_ids=(existing.event_id,),
            )
            consequence = recorded_consequence
            if consequence is None:
                if existing.commitment_kind is CommitmentKind.ABSTENTION:
                    abstention = EffectOutcome.create(
                        observed_outcome={"abstained": True},
                        realized_utility_vector={"unscored": 0.0},
                        realized_full_cost=WorkVector.zero(),
                    )
                    consequence = self._append_consequence(commitment=existing, outcome=abstention, tick=tick)
                    return self._result(
                        status=ChassisStatus.ABSTAINED,
                        hypothesis=hypothesis,
                        commitment=existing,
                        consequence=consequence,
                        updated=(),
                        effect_invoked=False,
                        resumed=True,
                        start_sequence=start_sequence,
                    )
                return self._result(
                    status=ChassisStatus.COMMITTED_EFFECT_NOT_REPLAYED,
                    hypothesis=hypothesis,
                    commitment=existing,
                    consequence=None,
                    updated=(),
                    effect_invoked=False,
                    resumed=True,
                    start_sequence=start_sequence,
                    failure_reason="at-most-once invocation fence already persisted",
                )
            return self._apply_recorded_consequence(
                hypothesis=hypothesis,
                commitment=existing,
                consequence=consequence,
                tick=tick,
                start_sequence=start_sequence,
                resumed=True,
            )

        dispatch = self.dispatch_from_hypothesis(hypothesis.event_id)
        self._charge(
            reason="chassis-ledger-to-dispatch-adaptation",
            work=WorkVector(indexing_and_graph_maintenance=1 + len(dispatch.payload_bytes)),
            tick=tick,
            branch=hypothesis.branch_id,
            causal_event_ids=(hypothesis.event_id,),
        )
        trace = self._runtime.run(dispatch, now_tick=tick)
        if trace.initial_event_id != str(hypothesis.event_id):
            raise ChassisContractError("runtime trace does not bind the dispatched hypothesis")
        if trace.runtime_id != self._runtime.runtime_id:
            raise ChassisContractError("runtime trace was issued by a different runtime authority")
        trace_digest = self._trace_digest(trace)
        trace_bytes = canonical_bytes(_canonical_value(trace))
        self._charge(
            reason="chassis-full-trace-binding",
            work=WorkVector(indexing_and_graph_maintenance=1 + len(trace_bytes)),
            tick=tick,
            branch=hypothesis.branch_id,
            causal_event_ids=(hypothesis.event_id,),
        )
        if failpoint is ChassisFailpoint.AFTER_DISPATCH:
            raise InjectedChassisFailure(failpoint)
        action, reason, blocked_action_id = self._select_external_action(hypothesis, trace)
        commitment = self._append_commitment(
            hypothesis=hypothesis,
            trace=trace,
            trace_digest=trace_digest,
            action=action,
            decision_reason=reason,
            blocked_action_id=blocked_action_id,
            tick=tick,
        )
        if failpoint is ChassisFailpoint.AFTER_COMMITMENT:
            raise InjectedChassisFailure(failpoint)
        if action is None:
            abstention = EffectOutcome.create(
                observed_outcome={"abstained": True, "reason": reason},
                realized_utility_vector={"unscored": 0.0},
                realized_full_cost=WorkVector.zero(),
            )
            consequence = self._append_consequence(commitment=commitment, outcome=abstention, tick=tick)
            return self._result(
                status=ChassisStatus.ABSTAINED,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=consequence,
                updated=(),
                effect_invoked=False,
                resumed=False,
                start_sequence=start_sequence,
            )

        request = self._effect_request(commitment, hypothesis)
        if request is None:
            raise ChassisContractError("external-action commitment lost its action record")
        self._charge(
            reason="chassis-effect-attempt",
            work=WorkVector(actor_execution=1 + len(request.payload_bytes)),
            tick=tick,
            branch=commitment.branch_id,
            causal_event_ids=(commitment.event_id,),
        )
        if effect is None or not isinstance(effect, ExternalEffect):
            self._charge(
                reason="chassis-effect-adapter-failure",
                work=WorkVector(actor_execution=1),
                tick=tick,
                branch=commitment.branch_id,
                causal_event_ids=(commitment.event_id,),
            )
            return self._result(
                status=ChassisStatus.EFFECT_FAILED,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=None,
                updated=(),
                effect_invoked=False,
                resumed=False,
                start_sequence=start_sequence,
                failure_reason="missing or invalid external effect adapter",
            )
        try:
            outcome = effect.execute(request)
        except Exception as exc:
            self._charge(
                reason="chassis-effect-exception",
                work=WorkVector(actor_execution=1),
                tick=tick,
                branch=commitment.branch_id,
                causal_event_ids=(commitment.event_id,),
            )
            return self._result(
                status=ChassisStatus.EFFECT_FAILED,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=None,
                updated=(),
                effect_invoked=True,
                resumed=False,
                start_sequence=start_sequence,
                failure_reason=type(exc).__name__,
            )
        if type(outcome) is not EffectOutcome:
            self._charge(
                reason="chassis-effect-invalid-outcome",
                work=WorkVector(actor_execution=1),
                tick=tick,
                branch=commitment.branch_id,
                causal_event_ids=(commitment.event_id,),
            )
            return self._result(
                status=ChassisStatus.EFFECT_FAILED,
                hypothesis=hypothesis,
                commitment=commitment,
                consequence=None,
                updated=(),
                effect_invoked=True,
                resumed=False,
                start_sequence=start_sequence,
                failure_reason="invalid EffectOutcome",
            )
        self._charge(
            reason="chassis-effect-realized-cost",
            work=outcome.realized_full_cost,
            tick=tick,
            branch=commitment.branch_id,
            causal_event_ids=(commitment.event_id,),
        )
        if failpoint is ChassisFailpoint.AFTER_EFFECT:
            raise InjectedChassisFailure(failpoint)
        consequence = self._append_consequence(commitment=commitment, outcome=outcome, tick=tick)
        if failpoint is ChassisFailpoint.AFTER_CONSEQUENCE:
            raise InjectedChassisFailure(failpoint)
        result = self._apply_recorded_consequence(
            hypothesis=hypothesis,
            commitment=commitment,
            consequence=consequence,
            tick=tick,
            start_sequence=start_sequence,
            resumed=False,
        )
        return replace(result, effect_invoked=True)


__all__ = [
    "CHASSIS_COMMITMENT_SCHEMA",
    "CHASSIS_CONSEQUENCE_SCHEMA",
    "CHASSIS_RESULT_SCHEMA",
    "ChassisContractError",
    "ChassisFailpoint",
    "ChassisResult",
    "ChassisStatus",
    "EffectOutcome",
    "EffectRequest",
    "EventSourcedCoalitionChassis",
    "ExternalEffect",
    "InjectedChassisFailure",
]
