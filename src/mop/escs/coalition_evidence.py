"""Scripted, noncausal, activation-disabled coalition fixture evidence for ESCS.

This module is a shadow evidence plane, not a dispatch policy.  It validates complete same-state
counterfactual fixture families made from native ESCS events, runtime traces, and lifecycle ledgers;
derives arithmetic difference and interaction terms; freezes delayed fixture terms into a small
immutable table; and ranks a bounded ``DispatchRequest`` without applying the result.  An
abstention cannot cause arbitrary utility, so every utility in v1 is explicitly scripted,
oracle-tainted, noncausal, and nonpromotable.

There is deliberately no ``select``, ``activate``, ``stage_update``, ``DispatchDecision`` conversion,
commitment, effect, or promotion surface in this module.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from fractions import Fraction
from itertools import combinations
from typing import Any, Self

from mop.substrate.events import EventRef, FrozenJSON, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from .actors import DispatchEventHeader
from .chassis import CHASSIS_COMMITMENT_SCHEMA, EFFECT_AUTHORITY_SCHEMA
from .events import (
    CommitmentEvent,
    CommitmentKind,
    ConsequenceEvent,
    EpistemicStatus,
    EvidenceClass,
    HypothesisEvent,
)
from .ledger import EventLedger
from .perspective_registry import PerspectiveCandidateRegistry, TriggerAuthority
from .runtime import (
    CandidateMode,
    DispatchRequest,
    ReadinessEstimate,
    RoundTrace,
    RuntimeCaps,
    RuntimeTrace,
)
from .substrate_assembly import SlotMode, SubstrateAssembly

CONFIG_SCHEMA = "mop-escs-coalition-evidence-config/v1"
BINDING_SCHEMA = "mop-escs-coalition-actor-binding/v1"
FORK_CONTRACT_SCHEMA = "mop-escs-coalition-fork-contract/v1"
INTERVENTION_SCHEMA = "mop-escs-scripted-actor-removal-intervention/v1"
SCRIPTED_OUTCOME_SCHEMA = "mop-escs-scripted-coalition-utility-outcome/v1"
UTILITY_SCALARIZER_SCHEMA = "mop-escs-scripted-utility-scalarizer/v1"
TRACE_CONFIG_FRAME_SCHEMA = "mop-escs-trace-config-frame/v1"
FORK_SCHEMA = "mop-escs-coalition-fork/v1"
AUTHORITY_SCHEMA = "mop-escs-coalition-fork-authority/v1"
ACCOUNTING_SCHEMA = "mop-escs-coalition-evidence-accounting/v1"
CREDIT_SCHEMA = "mop-escs-exact-coalition-credit/v1"
SNAPSHOT_SCHEMA = "mop-escs-interaction-credit-snapshot/v1"
SCORE_SCHEMA = "mop-escs-shadow-coalition-score/v1"
PROPOSAL_SCHEMA = "mop-escs-shadow-arbitration-proposal/v1"

ACTIVATION_ENABLED = False
SCIENTIFIC_PROMOTION_ALLOWED = False
FORK_PROVENANCE_KEY = "coalition_evidence_fork"
FORK_INTERVENTION_KEY = "coalition_evidence_intervention"

MAX_HARD_ACTORS = 8
MAX_HARD_FORKS = 46
MAX_HARD_CREDIT_RECORDS = 128
MAX_HARD_BEAM = 256
MAX_HARD_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_HARD_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_HARD_WORK_UNITS = 16 * 1024 * 1024
MAX_HARD_READINESS_RISK_MICROS = 1_000_000

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_REF_RE = re.compile(r"^[a-z][a-z0-9+.-]*:[a-z0-9][a-z0-9._:/-]*$")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")
_FORBIDDEN_KEYS = frozenset(
    {
        "canonical_referent_truth",
        "evaluator",
        "evaluator_label",
        "evaluator_truth",
        "future_consequence",
        "future_outcome",
        "ground_truth",
        "ground_truth_label",
        "hidden_change_point",
        "hidden_state",
        "hidden_state_digest",
        "hidden_shock",
        "irreducible_noise",
        "niche_label",
        "oracle_coalition",
        "oracle_label",
    }
)
_FORBIDDEN_KEY_FINGERPRINTS = frozenset(value.replace("_", "") for value in _FORBIDDEN_KEYS)
_FORK_AUTHORITY_TOKEN = object()
_CREDIT_TOKEN = object()
_SNAPSHOT_TOKEN = object()


class CoalitionEvidenceError(ValueError):
    """A fork, credit record, or shadow proposal violated the finite evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoalitionEvidenceError(message)


def _nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise CoalitionEvidenceError(f"{label} must be a {qualifier} integer")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CoalitionEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise CoalitionEvidenceError(f"{label} must be a canonical identifier")
    return value


def _stable_ref(value: object, label: str) -> str:
    if not isinstance(value, str) or _REF_RE.fullmatch(value) is None:
        raise CoalitionEvidenceError(f"{label} must be a stable namespaced reference")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise CoalitionEvidenceError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise CoalitionEvidenceError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _normalize_key(value: object) -> str:
    separated = _CAMEL_BOUNDARY_RE.sub("_", str(value)).casefold()
    return _KEY_SEPARATOR_RE.sub("_", separated).strip("_")


def _walk_keys(value: Any) -> set[str]:
    result: set[str] = set()
    stack = [value]
    while stack:
        nested = stack.pop()
        if isinstance(nested, Mapping):
            result.update(_normalize_key(key) for key in nested)
            stack.extend(nested.values())
        elif isinstance(nested, (list, tuple)):
            stack.extend(nested)
    return result


def _reject_forbidden(value: Any, label: str) -> None:
    found = sorted(
        key
        for key in _walk_keys(value)
        if key in _FORBIDDEN_KEYS or key.replace("_", "") in _FORBIDDEN_KEY_FINGERPRINTS
    )
    if found:
        raise CoalitionEvidenceError(f"{label} contains evaluator/future-only fields: {found}")


def _canonical_actor_ids(values: Sequence[str], label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    rows = tuple(values)
    if not allow_empty and not rows:
        raise CoalitionEvidenceError(f"{label} must not be empty")
    for value in rows:
        _stable_ref(value, label)
    if rows != tuple(sorted(rows)) or len(rows) != len(set(rows)):
        raise CoalitionEvidenceError(f"{label} must be unique and canonically sorted")
    return rows


def _canonical_pairs(values: Sequence[tuple[str, str]], label: str) -> tuple[tuple[str, str], ...]:
    rows = tuple(values)
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2 or row[0] >= row[1]:
            raise CoalitionEvidenceError(f"{label} must contain canonical two-actor tuples")
        _stable_ref(row[0], label)
        _stable_ref(row[1], label)
    if rows != tuple(sorted(rows)) or len(rows) != len(set(rows)):
        raise CoalitionEvidenceError(f"{label} must be unique and canonically sorted")
    return rows


def _strict_bytes(value: Any, *, limit: int, label: str) -> bytes:
    try:
        result = canonical_bytes(value)
    except (RecursionError, TypeError, ValueError) as exc:
        raise CoalitionEvidenceError(f"{label} is not strict canonical JSON") from exc
    if len(result) > limit:
        raise CoalitionEvidenceError(f"{label} exceeds its byte cap")
    return result


def _work_without(left: WorkVector, right: WorkVector) -> WorkVector:
    values: dict[str, int] = {}
    for descriptor in fields(WorkVector):
        difference = getattr(left, descriptor.name) - getattr(right, descriptor.name)
        _require(difference >= 0, "actor-attributed work exceeds full fork work")
        values[descriptor.name] = difference
    return WorkVector(**values)


def _work_in_bucket(bucket: str, units: int) -> WorkVector:
    values = {descriptor.name: 0 for descriptor in fields(WorkVector)}
    _require(bucket in values and bucket != "retained_byte_time", "unknown operation work bucket")
    values[bucket] = _nonnegative_int(units, "accounting work units")
    return WorkVector(**values)


@dataclass(frozen=True, slots=True)
class CoalitionEvidenceConfig:
    candidate_registry_sha256: str
    assembly_sha256: str
    max_actors: int = 4
    max_forks: int = 16
    max_credit_records: int = 32
    max_beam: int = 64
    max_snapshot_bytes: int = 1024 * 1024
    max_payload_bytes: int = 2 * 1024 * 1024
    max_work_units: int = 1024 * 1024
    compute_price_milli: int = 1
    bandwidth_price_milli: int = 1
    retained_byte_time_price_milli: int = 0
    stale_risk_price_milli: int = 1000
    deadline_risk_price_milli: int = 1000
    max_readiness_risk_micros: int = 1_000_000
    minimum_net_value_milli: int = 1
    activation_enabled: bool = ACTIVATION_ENABLED
    scientific_promotion_allowed: bool = SCIENTIFIC_PROMOTION_ALLOWED

    def __post_init__(self) -> None:
        _digest(self.candidate_registry_sha256, "candidate registry authority")
        _digest(self.assembly_sha256, "assembly authority")
        for name in (
            "max_actors",
            "max_forks",
            "max_credit_records",
            "max_beam",
            "max_snapshot_bytes",
            "max_payload_bytes",
            "max_work_units",
        ):
            _nonnegative_int(getattr(self, name), f"CoalitionEvidenceConfig.{name}", positive=True)
        _require(self.max_actors <= MAX_HARD_ACTORS, "actor cap exceeds hard bound")
        _require(self.max_forks <= MAX_HARD_FORKS, "fork cap exceeds hard bound")
        _require(self.max_credit_records <= MAX_HARD_CREDIT_RECORDS, "credit cap exceeds hard bound")
        _require(self.max_beam <= MAX_HARD_BEAM, "beam cap exceeds hard bound")
        _require(self.max_snapshot_bytes <= MAX_HARD_SNAPSHOT_BYTES, "snapshot cap exceeds hard bound")
        _require(self.max_payload_bytes <= MAX_HARD_PAYLOAD_BYTES, "payload cap exceeds hard bound")
        _require(self.max_work_units <= MAX_HARD_WORK_UNITS, "work cap exceeds hard bound")
        for name in (
            "compute_price_milli",
            "bandwidth_price_milli",
            "retained_byte_time_price_milli",
            "stale_risk_price_milli",
            "deadline_risk_price_milli",
            "minimum_net_value_milli",
            "max_readiness_risk_micros",
        ):
            _nonnegative_int(getattr(self, name), f"CoalitionEvidenceConfig.{name}")
        _require(
            self.max_readiness_risk_micros <= MAX_HARD_READINESS_RISK_MICROS,
            "readiness risk bound exceeds the hard probability ceiling",
        )
        _require(self.activation_enabled is False, "coalition evidence activation must remain disabled")
        _require(
            self.scientific_promotion_allowed is False,
            "coalition evidence cannot grant scientific promotion",
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "assembly_sha256": self.assembly_sha256,
            "max_actors": self.max_actors,
            "max_forks": self.max_forks,
            "max_credit_records": self.max_credit_records,
            "max_beam": self.max_beam,
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "max_payload_bytes": self.max_payload_bytes,
            "max_work_units": self.max_work_units,
            "compute_price_milli": self.compute_price_milli,
            "bandwidth_price_milli": self.bandwidth_price_milli,
            "retained_byte_time_price_milli": self.retained_byte_time_price_milli,
            "stale_risk_price_milli": self.stale_risk_price_milli,
            "deadline_risk_price_milli": self.deadline_risk_price_milli,
            "max_readiness_risk_micros": self.max_readiness_risk_micros,
            "minimum_net_value_milli": self.minimum_net_value_milli,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ActorPerspectiveBinding:
    actor_id: str
    candidate_id: str
    candidate_sha256: str
    facet: str
    mode: SlotMode
    trigger_authority: TriggerAuthority
    binding_sha256: str

    def __post_init__(self) -> None:
        _stable_ref(self.actor_id, "actor_id")
        _identifier(self.candidate_id, "candidate_id")
        _digest(self.candidate_sha256, "candidate_sha256")
        _identifier(self.facet, "facet")
        _require(isinstance(self.mode, SlotMode), "binding mode must be typed")
        _require(isinstance(self.trigger_authority, TriggerAuthority), "trigger authority must be typed")
        _digest(self.binding_sha256, "binding_sha256")
        _require(
            self.binding_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "actor-perspective binding self-hash mismatch",
        )

    @classmethod
    def create(cls, actor_id: str, candidate_id: str, assembly: SubstrateAssembly) -> Self:
        slot = next((row for row in assembly.slots if row.candidate_id == candidate_id), None)
        if slot is None:
            raise CoalitionEvidenceError(f"candidate {candidate_id!r} is absent from the assembly")
        core = {
            "schema": BINDING_SCHEMA,
            "actor_id": actor_id,
            "candidate_id": candidate_id,
            "candidate_sha256": slot.candidate_sha256,
            "facet": slot.facet,
            "mode": slot.mode.value,
            "trigger_authority": slot.trigger_authority.value,
        }
        return cls(
            actor_id=actor_id,
            candidate_id=candidate_id,
            candidate_sha256=slot.candidate_sha256,
            facet=slot.facet,
            mode=slot.mode,
            trigger_authority=slot.trigger_authority,
            binding_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": BINDING_SCHEMA,
            "actor_id": self.actor_id,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "facet": self.facet,
            "mode": self.mode.value,
            "trigger_authority": self.trigger_authority.value,
        }
        if include_digest:
            result["binding_sha256"] = self.binding_sha256
        return result


@dataclass(frozen=True, slots=True)
class CoalitionEvidenceAccounting:
    stage: str
    source_payload_bytes: int
    target_payload_bytes: int
    validation_operations: int
    bytes_serialized: int
    bytes_hashed: int
    work_bucket: str
    work: WorkVector
    charge_applied: bool
    accounting_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.stage, str) and bool(self.stage), "accounting stage is empty")
        for name in (
            "source_payload_bytes",
            "target_payload_bytes",
            "validation_operations",
            "bytes_serialized",
            "bytes_hashed",
        ):
            _nonnegative_int(getattr(self, name), name)
        expected_bytes = self.source_payload_bytes + self.target_payload_bytes
        _require(self.bytes_serialized == expected_bytes, "accounting serialization bytes are incomplete")
        _require(self.bytes_hashed == expected_bytes, "accounting hash bytes are incomplete")
        expected_units = self.validation_operations + self.bytes_serialized + self.bytes_hashed
        _require(self.work == _work_in_bucket(self.work_bucket, expected_units), "accounting work mismatch")
        _require(self.charge_applied is False, "shadow evidence cannot apply its own lifecycle charge")
        _digest(self.accounting_sha256, "accounting_sha256")
        _require(
            self.accounting_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "coalition evidence accounting self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        stage: str,
        source_payload: Any,
        target_payload: Any,
        validation_operations: int,
        work_bucket: str,
        payload_limit: int,
    ) -> Self:
        source = _strict_bytes(source_payload, limit=payload_limit, label=f"{stage} source")
        target = _strict_bytes(target_payload, limit=payload_limit, label=f"{stage} target")
        total_bytes = len(source) + len(target)
        operations = _nonnegative_int(validation_operations, "validation_operations")
        work = _work_in_bucket(work_bucket, operations + 2 * total_bytes)
        core = {
            "schema": ACCOUNTING_SCHEMA,
            "stage": stage,
            "source_payload_bytes": len(source),
            "target_payload_bytes": len(target),
            "validation_operations": operations,
            "bytes_serialized": total_bytes,
            "bytes_hashed": total_bytes,
            "work_bucket": work_bucket,
            "work": work.payload(),
            "charge_applied": False,
        }
        return cls(
            stage=stage,
            source_payload_bytes=len(source),
            target_payload_bytes=len(target),
            validation_operations=operations,
            bytes_serialized=total_bytes,
            bytes_hashed=total_bytes,
            work_bucket=work_bucket,
            work=work,
            charge_applied=False,
            accounting_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": ACCOUNTING_SCHEMA,
            "stage": self.stage,
            "source_payload_bytes": self.source_payload_bytes,
            "target_payload_bytes": self.target_payload_bytes,
            "validation_operations": self.validation_operations,
            "bytes_serialized": self.bytes_serialized,
            "bytes_hashed": self.bytes_hashed,
            "work_bucket": self.work_bucket,
            "work": self.work.payload(),
            "charge_applied": self.charge_applied,
        }
        if include_digest:
            result["accounting_sha256"] = self.accounting_sha256
        return result


def _fork_contract(hypothesis: HypothesisEvent) -> dict[str, Any]:
    provenance = hypothesis.envelope.source_and_provenance.value()
    _exact_keys(
        provenance,
        {FORK_PROVENANCE_KEY, FORK_INTERVENTION_KEY},
        "coalition fork provenance",
    )
    contract = provenance[FORK_PROVENANCE_KEY]
    _exact_keys(
        contract,
        {
            "schema",
            "fork_group_id",
            "world_id",
            "horizon_start_tick",
            "horizon_end_tick",
            "source_state_sha256",
            "environment_state_sha256",
            "runtime_id",
            "runtime_config_sha256",
            "policy_state_sha256",
            "actor_state_versions_sha256",
            "intervention_schema_sha256",
            "registered_actor_ids",
            "scripted_fixture_id",
            "scripted_fixture_sha256",
            "utility_key",
            "utility_min_milli",
            "utility_max_milli",
            "utility_scalarizer_sha256",
            "utility_excludes_resource_cost",
            "utility_source",
            "causal_effect_claim_allowed",
            "consequence_grounded_credit_claim_allowed",
        },
        "coalition fork contract",
    )
    _require(contract["schema"] == FORK_CONTRACT_SCHEMA, "fork contract schema drift")
    _stable_ref(contract["fork_group_id"], "fork_group_id")
    _stable_ref(contract["world_id"], "world_id")
    start = _nonnegative_int(contract["horizon_start_tick"], "horizon_start_tick")
    end = _nonnegative_int(contract["horizon_end_tick"], "horizon_end_tick")
    _require(end >= start, "fork horizon is inverted")
    for key in (
        "source_state_sha256",
        "environment_state_sha256",
        "runtime_id",
        "runtime_config_sha256",
        "policy_state_sha256",
        "actor_state_versions_sha256",
        "intervention_schema_sha256",
        "scripted_fixture_sha256",
        "utility_scalarizer_sha256",
    ):
        _digest(contract[key], key)
    registered = contract["registered_actor_ids"]
    _require(isinstance(registered, list), "registered_actor_ids must be a list")
    _canonical_actor_ids(tuple(registered), "registered actor ids", allow_empty=False)
    _stable_ref(contract["scripted_fixture_id"], "scripted_fixture_id")
    _identifier(contract["utility_key"], "utility_key")
    minimum = _nonnegative_int(contract["utility_min_milli"], "utility_min_milli")
    maximum = _nonnegative_int(contract["utility_max_milli"], "utility_max_milli")
    _require(maximum >= minimum, "utility scalarizer bounds are inverted")
    _require(
        contract["utility_excludes_resource_cost"] is True,
        "fork utility must exclude resource cost before actor debits",
    )
    expected_scalarizer = canonical_sha256(
        {
            "schema": UTILITY_SCALARIZER_SCHEMA,
            "utility_key": contract["utility_key"],
            "utility_min_milli": minimum,
            "utility_max_milli": maximum,
            "utility_excludes_resource_cost": True,
        }
    )
    _require(
        contract["utility_scalarizer_sha256"] == expected_scalarizer,
        "utility scalarizer authority drift",
    )
    _require(
        contract["intervention_schema_sha256"] == canonical_sha256({"schema": INTERVENTION_SCHEMA}),
        "intervention schema authority drift",
    )
    _require(
        contract["utility_source"] == "scripted-noncausal-fixture"
        and contract["causal_effect_claim_allowed"] is False
        and contract["consequence_grounded_credit_claim_allowed"] is False,
        "scripted fixture evidence ceiling drift",
    )
    _reject_forbidden(contract, "coalition fork contract")
    return dict(contract)


def _fork_intervention(hypothesis: HypothesisEvent) -> dict[str, Any]:
    provenance = hypothesis.envelope.source_and_provenance.value()
    contract = _fork_contract(hypothesis)
    intervention = provenance[FORK_INTERVENTION_KEY]
    _exact_keys(
        intervention,
        {
            "schema",
            "fork_group_id",
            "branch_id",
            "registered_actor_ids",
            "candidate_actor_ids",
            "active_actor_ids",
            "removed_actor_ids",
            "actor_state_versions_sha256",
            "intervention_kind",
            "intervention_schema_sha256",
            "non_actor_inputs_held_fixed",
            "action_effect_authority",
            "scripted_utility_milli",
            "realized_full_cost",
            "observation_uncertainty_micros",
            "consequence_start_tick",
            "consequence_end_tick",
            "causal_effect_claim_allowed",
            "consequence_grounded_credit_claim_allowed",
            "intervention_sha256",
        },
        "coalition fork intervention",
    )
    _require(intervention["schema"] == INTERVENTION_SCHEMA, "intervention schema drift")
    _require(intervention["fork_group_id"] == contract["fork_group_id"], "intervention group drift")
    _require(intervention["branch_id"] == str(hypothesis.branch_id), "intervention branch drift")
    for name in (
        "registered_actor_ids",
        "candidate_actor_ids",
        "active_actor_ids",
        "removed_actor_ids",
    ):
        values = intervention[name]
        _require(isinstance(values, list), f"{name} must be a list")
        _canonical_actor_ids(tuple(values), name)
    registered = tuple(intervention["registered_actor_ids"])
    candidates = tuple(intervention["candidate_actor_ids"])
    active = tuple(intervention["active_actor_ids"])
    removed = tuple(intervention["removed_actor_ids"])
    _require(registered == tuple(contract["registered_actor_ids"]), "intervention actor registry drift")
    _require(candidates == registered, "intervention candidate actor set drift")
    _require(
        not (set(active) & set(removed)) and tuple(sorted((*active, *removed))) == registered,
        "intervention active/removed actor partition drift",
    )
    _require(
        intervention["intervention_kind"] == "scripted-actor-removal",
        "intervention kind drift",
    )
    _digest(intervention["actor_state_versions_sha256"], "intervention actor state authority")
    _require(
        intervention["actor_state_versions_sha256"] == contract["actor_state_versions_sha256"],
        "intervention actor state authority drift",
    )
    _require(
        intervention["intervention_schema_sha256"] == contract["intervention_schema_sha256"],
        "intervention schema digest drift",
    )
    utility = _nonnegative_int(intervention["scripted_utility_milli"], "scripted utility")
    _require(
        contract["utility_min_milli"] <= utility <= contract["utility_max_milli"],
        "scripted utility escapes scalarizer bounds",
    )
    try:
        WorkVector.from_payload(intervention["realized_full_cost"])
    except (TypeError, ValueError) as exc:
        raise CoalitionEvidenceError("intervention realized cost is invalid") from exc
    uncertainty = _nonnegative_int(
        intervention["observation_uncertainty_micros"],
        "observation uncertainty micros",
    )
    _require(uncertainty <= 1_000_000, "observation uncertainty exceeds one")
    start = _nonnegative_int(intervention["consequence_start_tick"], "consequence start tick")
    end = _nonnegative_int(intervention["consequence_end_tick"], "consequence end tick")
    _require(
        start == end == contract["horizon_end_tick"],
        "scripted consequence clocks do not equal the registered horizon",
    )
    _require(
        intervention["non_actor_inputs_held_fixed"] is True
        and intervention["action_effect_authority"] is False
        and intervention["causal_effect_claim_allowed"] is False
        and intervention["consequence_grounded_credit_claim_allowed"] is False,
        "intervention gained a causal claim",
    )
    digest = intervention["intervention_sha256"]
    _digest(digest, "intervention_sha256")
    core = dict(intervention)
    core.pop("intervention_sha256")
    _require(digest == canonical_sha256(core), "intervention self-hash mismatch")
    return dict(intervention)


def _normalized_hypothesis_state(hypothesis: HypothesisEvent) -> str:
    return canonical_sha256(
        {
            "causal_parent_ids": [str(value) for value in hypothesis.envelope.causal_parent_ids],
            "clock_interval": [
                hypothesis.envelope.clock_start_tick,
                hypothesis.envelope.clock_end_tick,
            ],
            "clock_uncertainty": hypothesis.envelope.clock_uncertainty,
            "evidence_class": hypothesis.evidence_class.value,
            "body": hypothesis.body_payload(),
            "fork_contract": _fork_contract(hypothesis),
        }
    )


def _commitment_trace_payload(commitment: CommitmentEvent) -> dict[str, Any]:
    value = commitment.committed_payload.value()
    _exact_keys(
        value,
        {
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
        },
        "chassis commitment payload",
    )
    _require(value["schema"] == CHASSIS_COMMITMENT_SCHEMA, "chassis commitment schema drift")
    for key in ("runtime_id", "trace_authority_id", "full_trace_sha256", "effect_id"):
        _digest(value[key], f"commitment {key}")
    _nonnegative_int(value["trace_authority_sequence"], "trace_authority_sequence")
    _require(
        isinstance(value["decision_reason"], str) and bool(value["decision_reason"].strip()),
        "commitment decision reason is invalid",
    )
    _reject_forbidden(value, "chassis commitment payload")
    return dict(value)


def _actor_for_reason(reason: str, actors: tuple[str, ...]) -> str | None:
    direct_prefixes = (
        "header-readiness:",
        "actor-activation:",
        "action-intent:",
        "endogenous-proposal-admission:",
        "active-actor-update-plan:",
    )
    for prefix in direct_prefixes:
        if reason.startswith(prefix):
            candidate = reason[len(prefix) :]
            return candidate if candidate in actors else None
    if reason.startswith("message-edge:"):
        candidate = reason[len("message-edge:") :].split("->", 1)[0]
        return candidate if candidate in actors else None
    return None


@dataclass(frozen=True, slots=True)
class CoalitionFork:
    source_hypothesis: HypothesisEvent
    trace: RuntimeTrace
    commitment: CommitmentEvent
    consequence: ConsequenceEvent
    event_snapshot: FrozenJSON
    lifecycle_snapshot: FrozenJSON
    utility_key: str
    fork_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.source_hypothesis) is HypothesisEvent, "fork source must be a HypothesisEvent")
        _require(type(self.trace) is RuntimeTrace, "fork trace must be an exact RuntimeTrace")
        _require(type(self.commitment) is CommitmentEvent, "fork commitment must be exact")
        _require(type(self.consequence) is ConsequenceEvent, "fork consequence must be exact")
        _require(isinstance(self.event_snapshot, FrozenJSON), "event snapshot must be FrozenJSON")
        _require(isinstance(self.lifecycle_snapshot, FrozenJSON), "lifecycle snapshot must be FrozenJSON")
        _identifier(self.utility_key, "utility_key")
        event_bytes = _strict_bytes(
            self.event_snapshot.value(), limit=MAX_HARD_SNAPSHOT_BYTES, label="fork event snapshot"
        )
        lifecycle_bytes = _strict_bytes(
            self.lifecycle_snapshot.value(),
            limit=MAX_HARD_SNAPSHOT_BYTES,
            label="fork lifecycle snapshot",
        )
        _require(bool(event_bytes) and bool(lifecycle_bytes), "fork snapshots must not be empty")
        try:
            events = EventLedger.from_payload(self.event_snapshot.value())
            lifecycle = LifecycleLedger.from_payload(self.lifecycle_snapshot.value())
        except (TypeError, ValueError) as exc:
            raise CoalitionEvidenceError(f"fork snapshot replay failed: {exc}") from exc
        _require(events.verify() == [], "fork event snapshot does not replay exactly")
        _require(
            lifecycle.verify(event_ids=set(events.event_ids)) == [],
            "fork lifecycle/event provenance does not replay exactly",
        )
        for event in (self.source_hypothesis, self.commitment, self.consequence):
            try:
                replayed = events.get(event.event_id)
            except ValueError as exc:
                raise CoalitionEvidenceError("fork event is absent from its event snapshot") from exc
            _require(replayed.payload() == event.payload(), "fork event bytes drifted from snapshot")
        hypothesis = self.source_hypothesis
        _require(
            hypothesis.epistemic_status is EpistemicStatus.SIMULATED
            and hypothesis.branch_id != FACTUAL_BRANCH
            and hypothesis.evidence_class is EvidenceClass.SCRIPTED_MECHANICS,
            "coalition credit forks must remain simulated and counterfactual",
        )
        contract = _fork_contract(hypothesis)
        intervention = _fork_intervention(hypothesis)
        _reject_forbidden(hypothesis.body_payload(), "fork source hypothesis")
        _require(contract["utility_key"] == self.utility_key, "fork utility authority drift")
        _require(
            hypothesis.envelope.clock_start_tick
            == hypothesis.envelope.clock_end_tick
            == contract["horizon_start_tick"]
            and hypothesis.envelope.clock_uncertainty == 0,
            "fork source clock authority drift",
        )
        _require(self.trace.validate_integrity(), "fork runtime trace integrity failed")
        _require(self.trace.runtime_id == contract["runtime_id"], "fork runtime authority drift")
        expected_trace_config = canonical_sha256(
            {
                "schema": TRACE_CONFIG_FRAME_SCHEMA,
                "mode": self.trace.mode.value,
                "caps": self.trace.caps.payload(),
            }
        )
        _require(
            contract["runtime_config_sha256"] == expected_trace_config,
            "trace runtime configuration authority drift",
        )
        _require(self.trace.initial_event_id == str(hypothesis.event_id), "trace source event mismatch")
        _require(
            len(self.trace.rounds) == 1 and self.trace.endogenous_rounds == 0,
            "v1 scripted fixture requires one bounded dispatch round",
        )
        round_trace = self.trace.rounds[0]
        _require(type(round_trace) is RoundTrace, "fork round trace must be exact")
        header = round_trace.event_header
        _require(type(header) is DispatchEventHeader, "fork round header must be exact")
        _require(header.event_id == str(hypothesis.event_id), "trace does not begin from the fork hypothesis")
        _require(header.event_kind == hypothesis.kind.value, "trace header event kind drift")
        _require(
            header.producer_state_version == hypothesis.envelope.producer_state_version,
            "trace header producer state drift",
        )
        _require(
            header.epistemic_status is hypothesis.epistemic_status
            and header.evidence_class is hypothesis.evidence_class,
            "trace header epistemic/evidence authority drift",
        )
        _require(
            header.source_event_ids == tuple(str(value) for value in hypothesis.envelope.causal_parent_ids),
            "trace header source-event authority drift",
        )
        _require(
            header.payload_digest == hypothesis.envelope.payload_digest,
            "trace header payload authority drift",
        )
        _require(
            header.created_tick == hypothesis.envelope.clock_end_tick
            and not header.endogenous
            and header.reasoning_depth == 0,
            "trace header time/depth authority drift",
        )
        _require(
            all(row.event_header.branch_id == str(hypothesis.branch_id) for row in self.trace.rounds),
            "runtime trace crossed the fork branch",
        )
        candidates = _canonical_actor_ids(round_trace.candidate_actor_ids, "round candidate actors")
        selected_actors = _canonical_actor_ids(round_trace.selected_actor_ids, "round selected actors")
        _require(len(candidates) <= self.trace.caps.K, "round candidate actors exceed K")
        _require(
            len(selected_actors) <= self.trace.caps.C and set(selected_actors) <= set(candidates),
            "round selected actors escape the candidate/C authority",
        )
        _require(
            isinstance(round_trace.considered_coalitions, tuple)
            and len(round_trace.considered_coalitions) <= self.trace.caps.B,
            "round considered beam escapes B or immutability",
        )
        considered: list[tuple[str, ...]] = []
        for coalition in round_trace.considered_coalitions:
            _require(isinstance(coalition, tuple), "considered coalition must be immutable")
            canonical = _canonical_actor_ids(coalition, "considered coalition")
            _require(
                len(canonical) <= self.trace.caps.C and set(canonical) <= set(candidates),
                "considered coalition escapes candidate/C authority",
            )
            considered.append(canonical)
        _require(
            len(considered) == len(set(considered)),
            "round considered beam contains duplicate coalitions",
        )
        _require(
            not selected_actors or selected_actors in considered,
            "selected coalition is absent from the considered beam",
        )
        _require(
            round_trace.staged_message_ids == ()
            and round_trace.consumed_message_ids == ()
            and round_trace.admitted_endogenous_event_ids == ()
            and round_trace.accepted_action_ids
            == tuple(action.action_id for action in self.trace.action_intents),
            "round-local message/action/endogenous IDs drift from the trace payload",
        )
        _require(
            self.trace.rejected_claims == () and self.trace.rejected_actions == (),
            "v1 scripted fixture rejects unresolved message/action payloads",
        )
        active = _canonical_actor_ids(tuple(sorted(self.trace.active_actor_ids)), "active actor ids")
        _require(len(active) == len(self.trace.active_actor_ids), "runtime active actor identities duplicate")
        _require(self.trace.active_actor_ids == active, "runtime active actor identities are noncanonical")
        _require(
            active == tuple(intervention["active_actor_ids"]),
            "trace actors do not match the branch intervention",
        )
        selected = {actor for row in self.trace.rounds for actor in row.selected_actor_ids}
        _require(selected == set(active), "trace active actors do not equal selected actors")
        _require(selected_actors == active, "selected actors must be exact and canonical")
        _require(self.trace.action_intents == (), "v1 coalition evidence rejects opaque action payloads")
        _require(self.trace.message_deliveries == (), "v1 coalition evidence rejects opaque message payloads")
        _require(
            self.commitment.envelope.causal_parent_ids == (hypothesis.event_id,),
            "fork commitment has the wrong source hypothesis",
        )
        _require(self.commitment.branch_id == hypothesis.branch_id, "fork commitment crossed branches")
        _require(
            self.commitment.envelope.clock_start_tick
            == self.commitment.envelope.clock_end_tick
            == contract["horizon_start_tick"]
            and self.commitment.envelope.clock_uncertainty == 0,
            "fork commitment clock authority drift",
        )
        committed = _commitment_trace_payload(self.commitment)
        _require(
            committed["hypothesis_event_id"] == str(hypothesis.event_id),
            "commitment hypothesis identity mismatch",
        )
        _require(committed["runtime_id"] == self.trace.runtime_id, "commitment runtime identity mismatch")
        _require(
            committed["trace_authority_id"] == self.trace.trace_id, "commitment trace authority mismatch"
        )
        _require(
            committed["trace_authority_sequence"] == self.trace.authority_sequence,
            "commitment trace sequence mismatch",
        )
        _require(
            committed["full_trace_sha256"] == self.trace.full_trace_sha256,
            "commitment full trace digest mismatch",
        )
        _require(
            self.commitment.coalition_id == f"coalition:{self.trace.trace_id}",
            "commitment coalition identity mismatch",
        )
        _require(
            self.commitment.commitment_kind is CommitmentKind.ABSTENTION,
            "v1 counterfactual fork commitments must be explicit abstentions",
        )
        _require(
            committed["action_record"] is None and committed["blocked_action_id"] is None,
            "v1 fork commitment cannot retain an action authority",
        )
        _require(
            committed["decision_reason"] == "simulated-hypothesis-external-effect-refused",
            "fork commitment is not the native simulated-hypothesis abstention",
        )
        _require(
            self.commitment.decision_distribution.value() == {"abstention": 1.0}
            and self.commitment.predicted_utility_vector.value() == {"unscored": 0.0}
            and self.commitment.predicted_full_cost == WorkVector.zero(),
            "fork abstention commitment payload drift",
        )
        _require(
            self.commitment.deadline_tick == self.commitment.envelope.clock_end_tick,
            "fork abstention deadline drift",
        )
        _require(
            self.commitment.evidence_class is hypothesis.evidence_class
            and self.commitment.envelope.measured_creation_cost == WorkVector(event_formation=1)
            and self.commitment.envelope.source_and_provenance.value()
            == {"producer": "escs.chassis", "trace_authority_id": self.trace.trace_id},
            "fork commitment evidence/provenance drift",
        )
        expected_effect_id = canonical_sha256(
            {
                "schema": EFFECT_AUTHORITY_SCHEMA,
                "hypothesis_event_id": str(hypothesis.event_id),
                "action_id": None,
                "trace_authority_id": self.trace.trace_id,
            }
        )
        _require(committed["effect_id"] == expected_effect_id, "fork effect authority drift")
        _reject_forbidden(
            {
                "decision_distribution": self.commitment.decision_distribution.value(),
                "predicted_utility_vector": self.commitment.predicted_utility_vector.value(),
            },
            "fork commitment",
        )
        _require(
            self.consequence.commitment_event_id == self.commitment.event_id,
            "fork consequence binds the wrong commitment",
        )
        _require(
            self.consequence.envelope.causal_parent_ids == (self.commitment.event_id,),
            "v1 fork consequence must have exactly its commitment parent",
        )
        _require(self.consequence.branch_id == hypothesis.branch_id, "fork consequence crossed branches")
        _require(
            self.consequence.delayed_or_partial is False,
            "partial consequence cannot enter a scripted fixture term",
        )
        _require(
            events.consequences_for(self.commitment.event_id) == (self.consequence,),
            "fork commitment has duplicate or ambiguous consequences",
        )
        utility = self.consequence.realized_utility_vector.value()
        _exact_keys(utility, {self.utility_key}, "fork realized utility")
        observed = self.consequence.observed_outcome.value()
        _reject_forbidden(observed, "fork observed outcome")
        _exact_keys(
            observed,
            {
                "schema",
                "scripted_fixture_id",
                "scripted_fixture_sha256",
                "intervention_sha256",
                "utility_scalarizer_sha256",
                "hypothesis_event_id",
                "trace_authority_id",
                "effect_id",
                "scripted_utility_milli",
                "causal_effect_claim_allowed",
                "consequence_grounded_credit_claim_allowed",
            },
            "scripted fixture outcome",
        )
        _require(observed["schema"] == SCRIPTED_OUTCOME_SCHEMA, "scripted outcome schema drift")
        _require(
            observed["scripted_fixture_id"] == contract["scripted_fixture_id"]
            and observed["scripted_fixture_sha256"] == contract["scripted_fixture_sha256"]
            and observed["intervention_sha256"] == intervention["intervention_sha256"]
            and observed["utility_scalarizer_sha256"] == contract["utility_scalarizer_sha256"]
            and observed["hypothesis_event_id"] == str(hypothesis.event_id)
            and observed["trace_authority_id"] == self.trace.trace_id
            and observed["effect_id"] == committed["effect_id"],
            "scripted outcome authority drift",
        )
        _require(
            observed["causal_effect_claim_allowed"] is False
            and observed["consequence_grounded_credit_claim_allowed"] is False,
            "scripted outcome gained a causal claim",
        )
        _reject_forbidden(utility, "fork realized utility")
        utility_value = _nonnegative_int(utility[self.utility_key], "fork utility")
        _require(
            utility_value == observed["scripted_utility_milli"] == intervention["scripted_utility_milli"],
            "realized utility does not equal its scripted fixture authority",
        )
        _require(
            0 <= self.trace.ledger_start_sequence < self.trace.ledger_end_sequence <= lifecycle.entry_count,
            "trace lifecycle interval escapes its snapshot",
        )
        _require(
            self.consequence.envelope.clock_start_tick == intervention["consequence_start_tick"]
            and self.consequence.envelope.clock_end_tick == intervention["consequence_end_tick"]
            and self.consequence.envelope.clock_uncertainty == 0,
            "scripted consequence clock authority drift",
        )
        _require(
            self.consequence.observation_uncertainty
            == intervention["observation_uncertainty_micros"] / 1_000_000
            and self.consequence.realized_full_cost
            == WorkVector.from_payload(intervention["realized_full_cost"]),
            "scripted consequence uncertainty/cost authority drift",
        )
        _require(
            self.consequence.envelope.measured_creation_cost == WorkVector(event_formation=1)
            and self.consequence.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE,
            "scripted consequence mechanics/evidence authority drift",
        )
        expected_consequence_provenance = {
            "producer": "escs.scripted-coalition-utility-fixture",
            "scripted_fixture_id": contract["scripted_fixture_id"],
            "scripted_fixture_sha256": contract["scripted_fixture_sha256"],
            "intervention_sha256": intervention["intervention_sha256"],
            "utility_scalarizer_sha256": contract["utility_scalarizer_sha256"],
        }
        _require(
            self.consequence.envelope.source_and_provenance.value() == expected_consequence_provenance,
            "scripted consequence provenance drift",
        )
        trace_charges = lifecycle.entries[self.trace.ledger_start_sequence : self.trace.ledger_end_sequence]
        _require(
            all(entry.branch_id == hypothesis.branch_id for entry in trace_charges),
            "runtime trace slice crossed the fork branch",
        )
        _require(
            all(
                entry.start_tick == entry.end_tick == contract["horizon_start_tick"]
                for entry in trace_charges
            ),
            "runtime trace charge clocks drift from the source horizon",
        )
        _require(
            all(
                not entry.causal_event_ids or set(entry.causal_event_ids) == {hypothesis.event_id}
                for entry in trace_charges
            ),
            "runtime trace slice contains work unrelated to the source hypothesis",
        )
        trace_work = sum((entry.work for entry in trace_charges), WorkVector.zero())
        _require(
            trace_work.total_work <= self.trace.caps.max_episode_work,
            "runtime trace work exceeds max_episode_work",
        )
        for actor_id in active:
            _require(
                sum(entry.reason == f"actor-activation:{actor_id}" for entry in trace_charges) == 1,
                f"fork actor {actor_id!r} does not have exactly one activation charge",
            )
        suffix = lifecycle.entries[self.trace.ledger_end_sequence :]
        commitment_work = WorkVector(
            event_formation=1,
            indexing_and_graph_maintenance=1 + len(canonical_bytes(self.commitment.payload())),
        )
        commitment_charges = tuple(
            entry
            for entry in suffix
            if entry.reason == "chassis-commitment-formation-and-indexing"
            and entry.branch_id == hypothesis.branch_id
            and entry.causal_event_ids == (hypothesis.event_id,)
        )
        _require(
            len(commitment_charges) == 1
            and commitment_charges[0].work == commitment_work
            and commitment_charges[0].start_tick
            == commitment_charges[0].end_tick
            == self.commitment.envelope.clock_end_tick,
            "native commitment formation charge is absent or inexact",
        )
        consequence_work = WorkVector(
            event_formation=1,
            indexing_and_graph_maintenance=1 + len(canonical_bytes(self.consequence.payload())),
        )
        consequence_charges = tuple(
            entry
            for entry in suffix
            if entry.reason == "scripted-fixture-consequence-formation-and-indexing"
            and entry.branch_id == hypothesis.branch_id
            and entry.causal_event_ids == (self.commitment.event_id,)
        )
        _require(
            len(consequence_charges) == 1
            and consequence_charges[0].work == consequence_work
            and consequence_charges[0].start_tick
            == consequence_charges[0].end_tick
            == self.consequence.envelope.clock_end_tick,
            "scripted consequence formation charge is absent or inexact",
        )
        _digest(self.fork_sha256, "fork_sha256")
        _require(
            self.fork_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "coalition fork self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        source_hypothesis: HypothesisEvent,
        trace: RuntimeTrace,
        commitment: CommitmentEvent,
        consequence: ConsequenceEvent,
        event_ledger: EventLedger,
        lifecycle_ledger: LifecycleLedger,
        utility_key: str,
    ) -> Self:
        event_snapshot = FrozenJSON.from_value(event_ledger.payload())
        lifecycle_snapshot = FrozenJSON.from_value(lifecycle_ledger.payload())
        core = {
            "schema": FORK_SCHEMA,
            "source_hypothesis_event_id": str(source_hypothesis.event_id),
            "trace_id": trace.trace_id,
            "full_trace_sha256": trace.full_trace_sha256,
            "commitment_event_id": str(commitment.event_id),
            "consequence_event_id": str(consequence.event_id),
            "event_snapshot_sha256": event_snapshot.sha256,
            "lifecycle_snapshot_sha256": lifecycle_snapshot.sha256,
            "utility_key": utility_key,
        }
        return cls(
            source_hypothesis=source_hypothesis,
            trace=trace,
            commitment=commitment,
            consequence=consequence,
            event_snapshot=event_snapshot,
            lifecycle_snapshot=lifecycle_snapshot,
            utility_key=utility_key,
            fork_sha256=canonical_sha256(core),
        )

    @property
    def coalition_actor_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.trace.active_actor_ids))

    @property
    def utility_milli(self) -> int:
        value = self.consequence.realized_utility_vector.value()[self.utility_key]
        return _nonnegative_int(value, "fork utility")

    @property
    def available_tick(self) -> int:
        return self.consequence.envelope.clock_end_tick

    @property
    def contract(self) -> dict[str, Any]:
        return _fork_contract(self.source_hypothesis)

    @property
    def intervention(self) -> dict[str, Any]:
        return _fork_intervention(self.source_hypothesis)

    @property
    def normalized_state_sha256(self) -> str:
        return _normalized_hypothesis_state(self.source_hypothesis)

    @property
    def training_event_ids(self) -> tuple[str, ...]:
        ledger = EventLedger.from_payload(self.event_snapshot.value())
        pending = [
            self.source_hypothesis.event_id,
            self.commitment.event_id,
            self.consequence.event_id,
        ]
        seen: set[str] = set()
        while pending:
            event_id = pending.pop()
            if str(event_id) in seen:
                continue
            event = ledger.get(event_id)
            seen.add(str(event_id))
            pending.extend(event.envelope.causal_parent_ids)
        return tuple(sorted(seen))

    @property
    def source_ancestry_event_ids(self) -> tuple[str, ...]:
        ledger = EventLedger.from_payload(self.event_snapshot.value())
        pending = list(self.source_hypothesis.envelope.causal_parent_ids)
        seen: set[str] = set()
        while pending:
            event_id = pending.pop()
            if str(event_id) in seen:
                continue
            event = ledger.get(event_id)
            seen.add(str(event_id))
            pending.extend(event.envelope.causal_parent_ids)
        return tuple(sorted(seen))

    @property
    def training_payload_sha256s(self) -> tuple[str, ...]:
        ledger = EventLedger.from_payload(self.event_snapshot.value())
        values = {
            ledger.get(EventRef(event_id)).envelope.payload_digest for event_id in self.training_event_ids
        }
        values.add(self.trace.rounds[0].event_header.representation_payload_digest)
        return tuple(sorted(values))

    @property
    def trace_work(self) -> WorkVector:
        ledger = LifecycleLedger.from_payload(self.lifecycle_snapshot.value())
        return sum(
            (
                entry.work
                for entry in ledger.entries[self.trace.ledger_start_sequence : self.trace.ledger_end_sequence]
            ),
            WorkVector.zero(),
        )

    @property
    def total_work(self) -> WorkVector:
        return self.trace_work

    @property
    def actor_work(self) -> tuple[tuple[str, WorkVector], ...]:
        actors = self.coalition_actor_ids
        totals = {actor_id: WorkVector.zero() for actor_id in actors}
        ledger = LifecycleLedger.from_payload(self.lifecycle_snapshot.value())
        for entry in ledger.entries[self.trace.ledger_start_sequence : self.trace.ledger_end_sequence]:
            actor_id = _actor_for_reason(entry.reason, actors)
            if actor_id is not None:
                totals[actor_id] = totals[actor_id] + entry.work
        return tuple(sorted(totals.items()))

    @property
    def shared_work(self) -> WorkVector:
        attributed = sum((value for _, value in self.actor_work), WorkVector.zero())
        return _work_without(self.total_work, attributed)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "fork_sha256": self.fork_sha256,
            "source_hypothesis_event_id": str(self.source_hypothesis.event_id),
            "branch_id": str(self.source_hypothesis.branch_id),
            "coalition_actor_ids": list(self.coalition_actor_ids),
            "utility_milli": self.utility_milli,
            "available_tick": self.available_tick,
            "total_work": self.total_work.payload(),
            "actor_work": {actor: work.payload() for actor, work in self.actor_work},
            "shared_work": self.shared_work.payload(),
        }

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": FORK_SCHEMA,
            "source_hypothesis_event_id": str(self.source_hypothesis.event_id),
            "trace_id": self.trace.trace_id,
            "full_trace_sha256": self.trace.full_trace_sha256,
            "commitment_event_id": str(self.commitment.event_id),
            "consequence_event_id": str(self.consequence.event_id),
            "event_snapshot_sha256": self.event_snapshot.sha256,
            "lifecycle_snapshot_sha256": self.lifecycle_snapshot.sha256,
            "utility_key": self.utility_key,
        }
        if include_digest:
            result["fork_sha256"] = self.fork_sha256
        return result


def _validate_binding_authority(
    bindings: tuple[ActorPerspectiveBinding, ...],
    registry: PerspectiveCandidateRegistry,
    assembly: SubstrateAssembly,
    config: CoalitionEvidenceConfig,
) -> None:
    _require(type(registry) is PerspectiveCandidateRegistry, "registry must be exact")
    _require(type(assembly) is SubstrateAssembly, "assembly must be exact")
    _require(assembly.validate_registry(registry) == (), "assembly/registry authority mismatch")
    _require(
        all(type(binding) is ActorPerspectiveBinding for binding in bindings),
        "bindings must be exact actor-perspective records",
    )
    _require(registry.sha256 == config.candidate_registry_sha256, "config registry authority mismatch")
    _require(assembly.assembly_sha256 == config.assembly_sha256, "config assembly authority mismatch")
    actor_ids = tuple(row.actor_id for row in bindings)
    _canonical_actor_ids(actor_ids, "binding actor ids")
    slots = {slot.candidate_id: slot for slot in assembly.slots}
    for binding in bindings:
        slot = slots.get(binding.candidate_id)
        if slot is None:
            raise CoalitionEvidenceError("binding candidate is absent from assembly")
        _require(slot.candidate_sha256 == binding.candidate_sha256, "binding candidate digest drift")
        _require(slot.facet == binding.facet, "binding facet drift")
        _require(slot.mode is binding.mode, "binding slot mode drift")
        _require(slot.trigger_authority is binding.trigger_authority, "binding trigger authority drift")


@dataclass(frozen=True, slots=True)
class ForkAuthority:
    config: CoalitionEvidenceConfig
    bindings: tuple[ActorPerspectiveBinding, ...]
    forks: tuple[CoalitionFork, ...]
    full_coalition_actor_ids: tuple[str, ...]
    registered_actor_pairs: tuple[tuple[str, str], ...]
    fork_group_id: str
    normalized_state_sha256: str
    scripted_fixture_only: bool
    causal_effect_claim_allowed: bool
    consequence_grounded_credit_claim_allowed: bool
    authority_sha256: str
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(
            self._validation_token is _FORK_AUTHORITY_TOKEN,
            "fork authority must come from create()",
        )
        _require(type(self.config) is CoalitionEvidenceConfig, "fork authority config must be exact")
        _canonical_actor_ids(self.full_coalition_actor_ids, "full coalition", allow_empty=False)
        _canonical_pairs(self.registered_actor_pairs, "registered actor pairs")
        _stable_ref(self.fork_group_id, "fork_group_id")
        _digest(self.normalized_state_sha256, "normalized_state_sha256")
        _require(
            self.scripted_fixture_only is True
            and self.causal_effect_claim_allowed is False
            and self.consequence_grounded_credit_claim_allowed is False,
            "fork authority escaped its scripted noncausal ceiling",
        )
        _require(
            isinstance(self.bindings, tuple) and isinstance(self.forks, tuple),
            "fork authority collections must be immutable",
        )
        _require(
            all(type(fork) is CoalitionFork for fork in self.forks),
            "fork authority requires exact coalition fork records",
        )
        _require(
            len(self.full_coalition_actor_ids) <= self.config.max_actors, "full coalition exceeds actor cap"
        )
        _require(len(self.forks) <= self.config.max_forks, "fork family exceeds fork cap")
        for fork in self.forks:
            _require(
                len(fork.event_snapshot.canonical.encode("utf-8")) <= self.config.max_snapshot_bytes
                and len(fork.lifecycle_snapshot.canonical.encode("utf-8")) <= self.config.max_snapshot_bytes,
                "fork snapshot exceeds the configured byte cap",
            )
        expected_pairs = tuple(combinations(self.full_coalition_actor_ids, 2))
        _require(self.registered_actor_pairs == expected_pairs, "v1 requires every pair-removal fork")
        expected_coalitions = {self.full_coalition_actor_ids}
        expected_coalitions.update(
            tuple(actor for actor in self.full_coalition_actor_ids if actor != removed)
            for removed in self.full_coalition_actor_ids
        )
        expected_coalitions.update(
            tuple(actor for actor in self.full_coalition_actor_ids if actor not in pair)
            for pair in self.registered_actor_pairs
        )
        expected_coalitions.update((actor,) for actor in self.full_coalition_actor_ids)
        expected_coalitions.add(())
        observed = [fork.coalition_actor_ids for fork in self.forks]
        _require(len(observed) == len(set(observed)), "fork family contains duplicate coalitions")
        _require(
            set(observed) == expected_coalitions, "fork family is incomplete or contains extra coalitions"
        )
        _require(len(self.forks) == len(expected_coalitions), "fork family completeness count mismatch")
        contracts = [fork.contract for fork in self.forks]
        _require(all(contract == contracts[0] for contract in contracts), "fork contracts drifted")
        _require(contracts[0]["fork_group_id"] == self.fork_group_id, "fork group identity drift")
        _require(
            tuple(contracts[0]["registered_actor_ids"]) == self.full_coalition_actor_ids,
            "fork contract actor registry does not match the authority",
        )
        _require(
            all(fork.normalized_state_sha256 == self.normalized_state_sha256 for fork in self.forks),
            "forks do not share an exact normalized source state",
        )
        ancestries = [fork.source_ancestry_event_ids for fork in self.forks]
        _require(
            all(ancestry == ancestries[0] for ancestry in ancestries),
            "fork source ancestries drifted",
        )
        runtime_frames: list[dict[str, Any]] = []
        intervention_frames: list[dict[str, Any]] = []
        for fork in self.forks:
            round_trace = fork.trace.rounds[0]
            candidates = _canonical_actor_ids(round_trace.candidate_actor_ids, "fork candidates")
            _require(
                candidates == self.full_coalition_actor_ids,
                "fork candidate set is not the full registered actor set",
            )
            intervention = fork.intervention
            _require(
                tuple(intervention["registered_actor_ids"]) == self.full_coalition_actor_ids
                and tuple(intervention["active_actor_ids"]) == fork.coalition_actor_ids
                and tuple(intervention["removed_actor_ids"])
                == tuple(
                    actor for actor in self.full_coalition_actor_ids if actor not in fork.coalition_actor_ids
                ),
                "fork actor-removal assignment drift",
            )
            intervention_frames.append(
                {
                    key: intervention[key]
                    for key in (
                        "schema",
                        "fork_group_id",
                        "registered_actor_ids",
                        "candidate_actor_ids",
                        "actor_state_versions_sha256",
                        "intervention_kind",
                        "intervention_schema_sha256",
                        "non_actor_inputs_held_fixed",
                        "action_effect_authority",
                        "observation_uncertainty_micros",
                        "consequence_start_tick",
                        "consequence_end_tick",
                        "causal_effect_claim_allowed",
                        "consequence_grounded_credit_claim_allowed",
                    )
                }
            )
            header = round_trace.event_header.payload()
            header.pop("event_id")
            header.pop("branch_id")
            runtime_frames.append(
                {
                    "mode": fork.trace.mode.value,
                    "caps": fork.trace.caps.payload(),
                    "header": header,
                    "candidate_actor_ids": list(candidates),
                }
            )
        _require(
            all(frame == runtime_frames[0] for frame in runtime_frames),
            "fork runtime/header input frames drifted",
        )
        _require(
            all(frame == intervention_frames[0] for frame in intervention_frames),
            "fork non-assignment intervention inputs drifted",
        )
        branch_ids = [str(fork.source_hypothesis.branch_id) for fork in self.forks]
        _require(len(branch_ids) == len(set(branch_ids)), "fork branches must be unique")
        trace_ids = [fork.trace.trace_id for fork in self.forks]
        trace_sequences = [fork.trace.authority_sequence for fork in self.forks]
        intervention_ids = [fork.intervention["intervention_sha256"] for fork in self.forks]
        _require(len(trace_ids) == len(set(trace_ids)), "fork trace identities must be unique")
        _require(
            len(trace_sequences) == len(set(trace_sequences)),
            "fork trace authority sequences must be unique",
        )
        _require(
            len(intervention_ids) == len(set(intervention_ids)),
            "fork intervention identities must be unique",
        )
        bound_ids = {row.actor_id for row in self.bindings}
        _require(
            set(self.full_coalition_actor_ids) == bound_ids,
            "fork authority bindings must exactly cover the registered actor set",
        )
        _digest(self.authority_sha256, "authority_sha256")
        _require(
            self.authority_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "fork authority self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        config: CoalitionEvidenceConfig,
        registry: PerspectiveCandidateRegistry,
        assembly: SubstrateAssembly,
        bindings: Sequence[ActorPerspectiveBinding],
        forks: Sequence[CoalitionFork],
        full_coalition_actor_ids: Sequence[str],
    ) -> Self:
        normalized_bindings = tuple(sorted(bindings, key=lambda row: row.actor_id))
        normalized_forks = tuple(sorted(forks, key=lambda row: row.coalition_actor_ids))
        _validate_binding_authority(normalized_bindings, registry, assembly, config)
        full = tuple(sorted(full_coalition_actor_ids))
        pairs = tuple(combinations(full, 2))
        _require(bool(normalized_forks), "fork authority requires at least one fork")
        contract = normalized_forks[0].contract
        state_sha = normalized_forks[0].normalized_state_sha256
        core = {
            "schema": AUTHORITY_SCHEMA,
            "config_sha256": config.sha256,
            "candidate_registry_sha256": config.candidate_registry_sha256,
            "assembly_sha256": config.assembly_sha256,
            "binding_sha256s": [row.binding_sha256 for row in normalized_bindings],
            "fork_sha256s": [row.fork_sha256 for row in normalized_forks],
            "full_coalition_actor_ids": list(full),
            "registered_actor_pairs": [list(row) for row in pairs],
            "fork_group_id": contract["fork_group_id"],
            "normalized_state_sha256": state_sha,
            "scripted_fixture_only": True,
            "causal_effect_claim_allowed": False,
            "consequence_grounded_credit_claim_allowed": False,
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            config=config,
            bindings=normalized_bindings,
            forks=normalized_forks,
            full_coalition_actor_ids=full,
            registered_actor_pairs=pairs,
            fork_group_id=contract["fork_group_id"],
            normalized_state_sha256=state_sha,
            scripted_fixture_only=True,
            causal_effect_claim_allowed=False,
            consequence_grounded_credit_claim_allowed=False,
            authority_sha256=canonical_sha256(core),
            _validation_token=_FORK_AUTHORITY_TOKEN,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": AUTHORITY_SCHEMA,
            "config_sha256": self.config.sha256,
            "candidate_registry_sha256": self.config.candidate_registry_sha256,
            "assembly_sha256": self.config.assembly_sha256,
            "binding_sha256s": [row.binding_sha256 for row in self.bindings],
            "fork_sha256s": [row.fork_sha256 for row in self.forks],
            "full_coalition_actor_ids": list(self.full_coalition_actor_ids),
            "registered_actor_pairs": [list(row) for row in self.registered_actor_pairs],
            "fork_group_id": self.fork_group_id,
            "normalized_state_sha256": self.normalized_state_sha256,
            "scripted_fixture_only": self.scripted_fixture_only,
            "causal_effect_claim_allowed": self.causal_effect_claim_allowed,
            "consequence_grounded_credit_claim_allowed": self.consequence_grounded_credit_claim_allowed,
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }
        if include_digest:
            result["authority_sha256"] = self.authority_sha256
        return result


def _resource_debit(work: WorkVector, config: CoalitionEvidenceConfig) -> int:
    compute_units = work.total_work - work.messages
    return (
        compute_units * config.compute_price_milli
        + work.messages * config.bandwidth_price_milli
        + work.retained_byte_time * config.retained_byte_time_price_milli
    )


def _bounded_risk_debit(
    risk: float,
    *,
    price_milli: int,
    max_risk_micros: int,
    label: str,
) -> int:
    _require(
        isinstance(risk, int | float) and not isinstance(risk, bool) and math.isfinite(risk) and risk >= 0,
        f"{label} must be a finite nonnegative number",
    )
    exact_risk = Fraction(risk)
    _require(
        exact_risk <= Fraction(max_risk_micros, 1_000_000),
        f"{label} exceeds the declared readiness-risk bound",
    )
    exact_debit = exact_risk * price_milli
    return (exact_debit.numerator + exact_debit.denominator - 1) // exact_debit.denominator


@dataclass(frozen=True, slots=True)
class ExactCoalitionCredit:
    authority_sha256: str
    config_sha256: str
    candidate_registry_sha256: str
    assembly_sha256: str
    fork_group_id: str
    full_coalition_actor_ids: tuple[str, ...]
    full_utility_milli: int
    singleton_main_effect_milli: tuple[tuple[str, int], ...]
    individual_difference_credit_milli: tuple[tuple[str, int], ...]
    pair_interaction_milli: tuple[tuple[str, str, int], ...]
    resource_debit_milli: tuple[tuple[str, int], ...]
    available_tick: int
    training_event_ids: tuple[str, ...]
    training_payload_sha256s: tuple[str, ...]
    evidence_class: EvidenceClass
    accounting: CoalitionEvidenceAccounting
    scripted_fixture_only: bool
    causal_effect_claim_allowed: bool
    consequence_grounded_credit_claim_allowed: bool
    activation_enabled: bool
    scientific_promotion_allowed: bool
    credit_sha256: str
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(self._validation_token is _CREDIT_TOKEN, "credit must come from derive()")
        for digest_value, label in (
            (self.authority_sha256, "credit authority"),
            (self.config_sha256, "credit config"),
            (self.candidate_registry_sha256, "credit registry"),
            (self.assembly_sha256, "credit assembly"),
        ):
            _digest(digest_value, label)
        _stable_ref(self.fork_group_id, "fork_group_id")
        _canonical_actor_ids(self.full_coalition_actor_ids, "credit full coalition", allow_empty=False)
        _nonnegative_int(self.full_utility_milli, "full utility")
        _nonnegative_int(self.available_tick, "available_tick")
        actor_rows = (
            self.singleton_main_effect_milli,
            self.individual_difference_credit_milli,
            self.resource_debit_milli,
        )
        _require(all(tuple(sorted(rows)) == rows for rows in actor_rows), "actor rows must be canonical")
        _require(
            tuple(sorted(self.pair_interaction_milli)) == self.pair_interaction_milli,
            "pair credit rows must be canonical",
        )
        expected_actor_domain = set(self.full_coalition_actor_ids)
        _require(
            all({actor for actor, _ in rows} == expected_actor_domain for rows in actor_rows),
            "credit actor-row domain drift",
        )
        expected_pairs = set(combinations(self.full_coalition_actor_ids, 2))
        _require(
            {(left, right) for left, right, _ in self.pair_interaction_milli} == expected_pairs,
            "credit pair-row domain drift",
        )
        for rows in actor_rows:
            _require(
                all(isinstance(value, int) and not isinstance(value, bool) for _, value in rows),
                "credit actor values must be integers",
            )
        for _, value in self.resource_debit_milli:
            _nonnegative_int(value, "historical resource debit")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for _, _, value in self.pair_interaction_milli
            ),
            "credit interaction values must be integers",
        )
        _require(
            self.training_event_ids == tuple(sorted(set(self.training_event_ids)))
            and bool(self.training_event_ids),
            "credit training event ancestry must be nonempty, unique, and sorted",
        )
        for event_id in self.training_event_ids:
            _stable_ref(event_id, "credit training event")
        _require(
            self.training_payload_sha256s == tuple(sorted(set(self.training_payload_sha256s)))
            and bool(self.training_payload_sha256s),
            "credit training payload authorities must be nonempty, unique, and sorted",
        )
        for payload_sha256 in self.training_payload_sha256s:
            _digest(payload_sha256, "training payload sha256")
        _require(
            self.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE,
            "scripted credit must retain oracle-nonpromotable taint",
        )
        _require(type(self.accounting) is CoalitionEvidenceAccounting, "credit accounting must be exact")
        _require(self.accounting.charge_applied is False, "credit accounting was self-applied")
        _require(
            self.scripted_fixture_only is True
            and self.causal_effect_claim_allowed is False
            and self.consequence_grounded_credit_claim_allowed is False,
            "credit escaped its scripted noncausal ceiling",
        )
        _require(self.activation_enabled is False, "scripted fixture term cannot activate")
        _require(self.scientific_promotion_allowed is False, "scripted fixture term cannot promote")
        _digest(self.credit_sha256, "credit_sha256")
        _require(
            self.credit_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "exact coalition credit self-hash mismatch",
        )

    @classmethod
    def derive(cls, authority: ForkAuthority) -> Self:
        _require(type(authority) is ForkAuthority, "credit authority must be exact")
        by_coalition = {fork.coalition_actor_ids: fork for fork in authority.forks}
        full = authority.full_coalition_actor_ids
        full_fork = by_coalition[full]
        debits = {
            actor: _resource_debit(dict(full_fork.actor_work)[actor], authority.config) for actor in full
        }
        difference = {
            actor: full_fork.utility_milli
            - by_coalition[tuple(value for value in full if value != actor)].utility_milli
            - debits[actor]
            for actor in full
        }
        empty_utility = by_coalition[()].utility_milli
        main_effects = {actor: by_coalition[(actor,)].utility_milli - empty_utility for actor in full}
        interactions = {
            (left, right): full_fork.utility_milli
            - by_coalition[tuple(value for value in full if value != left)].utility_milli
            - by_coalition[tuple(value for value in full if value != right)].utility_milli
            + by_coalition[tuple(value for value in full if value not in {left, right})].utility_milli
            for left, right in authority.registered_actor_pairs
        }
        sources = [fork.summary_payload() for fork in authority.forks]
        target = {
            "authority_sha256": authority.authority_sha256,
            "full_utility_milli": full_fork.utility_milli,
            "singleton_main_effect_milli": main_effects,
            "individual_difference_credit_milli": difference,
            "pair_interaction_milli": {
                f"{left}|{right}": value for (left, right), value in interactions.items()
            },
            "resource_debit_milli": debits,
        }
        accounting = CoalitionEvidenceAccounting.create(
            stage="derive-exact-coalition-credit",
            source_payload=sources,
            target_payload=target,
            validation_operations=(len(authority.forks) * 8 + len(full) * 6 + len(interactions) * 5),
            work_bucket="counterfactual_credit",
            payload_limit=authority.config.max_payload_bytes,
        )
        _require(
            accounting.work.total_work <= authority.config.max_work_units,
            "exact-credit derivation work cap exceeded",
        )
        evidence_class = EvidenceClass.ORACLE_NONPROMOTABLE
        training_event_ids = tuple(
            sorted({event_id for fork in authority.forks for event_id in fork.training_event_ids})
        )
        training_payload_sha256s = tuple(
            sorted({digest for fork in authority.forks for digest in fork.training_payload_sha256s})
        )
        core = {
            "schema": CREDIT_SCHEMA,
            "authority_sha256": authority.authority_sha256,
            "config_sha256": authority.config.sha256,
            "candidate_registry_sha256": authority.config.candidate_registry_sha256,
            "assembly_sha256": authority.config.assembly_sha256,
            "fork_group_id": authority.fork_group_id,
            "full_coalition_actor_ids": list(full),
            "full_utility_milli": full_fork.utility_milli,
            "singleton_main_effect_milli": dict(sorted(main_effects.items())),
            "individual_difference_credit_milli": dict(sorted(difference.items())),
            "pair_interaction_milli": [
                [left, right, value] for (left, right), value in sorted(interactions.items())
            ],
            "resource_debit_milli": dict(sorted(debits.items())),
            "available_tick": max(fork.available_tick for fork in authority.forks),
            "training_event_ids": list(training_event_ids),
            "training_payload_sha256s": list(training_payload_sha256s),
            "evidence_class": evidence_class.value,
            "accounting_sha256": accounting.accounting_sha256,
            "scripted_fixture_only": True,
            "causal_effect_claim_allowed": False,
            "consequence_grounded_credit_claim_allowed": False,
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }
        return cls(
            authority_sha256=authority.authority_sha256,
            config_sha256=authority.config.sha256,
            candidate_registry_sha256=authority.config.candidate_registry_sha256,
            assembly_sha256=authority.config.assembly_sha256,
            fork_group_id=authority.fork_group_id,
            full_coalition_actor_ids=full,
            full_utility_milli=full_fork.utility_milli,
            singleton_main_effect_milli=tuple(sorted(main_effects.items())),
            individual_difference_credit_milli=tuple(sorted(difference.items())),
            pair_interaction_milli=tuple(
                (left, right, value) for (left, right), value in sorted(interactions.items())
            ),
            resource_debit_milli=tuple(sorted(debits.items())),
            available_tick=max(fork.available_tick for fork in authority.forks),
            training_event_ids=training_event_ids,
            training_payload_sha256s=training_payload_sha256s,
            evidence_class=evidence_class,
            accounting=accounting,
            scripted_fixture_only=True,
            causal_effect_claim_allowed=False,
            consequence_grounded_credit_claim_allowed=False,
            activation_enabled=False,
            scientific_promotion_allowed=False,
            credit_sha256=canonical_sha256(core),
            _validation_token=_CREDIT_TOKEN,
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": CREDIT_SCHEMA,
            "authority_sha256": self.authority_sha256,
            "config_sha256": self.config_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "assembly_sha256": self.assembly_sha256,
            "fork_group_id": self.fork_group_id,
            "full_coalition_actor_ids": list(self.full_coalition_actor_ids),
            "full_utility_milli": self.full_utility_milli,
            "singleton_main_effect_milli": dict(self.singleton_main_effect_milli),
            "individual_difference_credit_milli": dict(self.individual_difference_credit_milli),
            "pair_interaction_milli": [list(row) for row in self.pair_interaction_milli],
            "resource_debit_milli": dict(self.resource_debit_milli),
            "available_tick": self.available_tick,
            "training_event_ids": list(self.training_event_ids),
            "training_payload_sha256s": list(self.training_payload_sha256s),
            "evidence_class": self.evidence_class.value,
            "accounting_sha256": self.accounting.accounting_sha256,
            "scripted_fixture_only": self.scripted_fixture_only,
            "causal_effect_claim_allowed": self.causal_effect_claim_allowed,
            "consequence_grounded_credit_claim_allowed": self.consequence_grounded_credit_claim_allowed,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["credit_sha256"] = self.credit_sha256
        return result


@dataclass(frozen=True, slots=True)
class InteractionCreditSnapshot:
    config_sha256: str
    candidate_registry_sha256: str
    assembly_sha256: str
    fit_tick: int
    authority_sha256s: tuple[str, ...]
    authority_actor_orders: tuple[tuple[str, tuple[str, ...]], ...]
    credit_sha256s: tuple[str, ...]
    training_event_ids: tuple[str, ...]
    training_payload_sha256s: tuple[str, ...]
    main_effect_terms: tuple[tuple[str, int, int], ...]
    difference_credit_terms: tuple[tuple[str, int, int], ...]
    pair_terms: tuple[tuple[str, str, int, int], ...]
    evidence_class: EvidenceClass
    retained_state_bytes: int
    accounting: CoalitionEvidenceAccounting
    scripted_fixture_only: bool
    causal_effect_claim_allowed: bool
    consequence_grounded_credit_claim_allowed: bool
    activation_enabled: bool
    scientific_promotion_allowed: bool
    snapshot_sha256: str
    _validation_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(self._validation_token is _SNAPSHOT_TOKEN, "snapshot must come from create()")
        for value, label in (
            (self.config_sha256, "snapshot config"),
            (self.candidate_registry_sha256, "snapshot registry"),
            (self.assembly_sha256, "snapshot assembly"),
            (self.snapshot_sha256, "snapshot sha256"),
        ):
            _digest(value, label)
        _nonnegative_int(self.fit_tick, "fit_tick")
        _nonnegative_int(self.retained_state_bytes, "retained_state_bytes")
        _require(
            self.authority_sha256s == tuple(sorted(set(self.authority_sha256s))),
            "snapshot authority ids must be unique and sorted",
        )
        _require(
            self.authority_actor_orders == tuple(sorted(self.authority_actor_orders, key=lambda row: row[0]))
            and tuple(row[0] for row in self.authority_actor_orders) == self.authority_sha256s,
            "snapshot authority actor-order rows drifted",
        )
        for authority_sha256, actor_order in self.authority_actor_orders:
            _digest(authority_sha256, "snapshot actor-order authority")
            _canonical_actor_ids(actor_order, "snapshot authority actor order", allow_empty=False)
        _require(
            self.credit_sha256s == tuple(sorted(set(self.credit_sha256s))),
            "snapshot credit ids must be unique and sorted",
        )
        _require(
            bool(self.credit_sha256s) and len(self.authority_sha256s) == len(self.credit_sha256s),
            "snapshot authority/credit arity drift",
        )
        _require(
            self.training_event_ids == tuple(sorted(set(self.training_event_ids)))
            and bool(self.training_event_ids),
            "training event ids must be unique and sorted",
        )
        for event_id in self.training_event_ids:
            _stable_ref(event_id, "snapshot training event")
        _require(
            self.training_payload_sha256s == tuple(sorted(set(self.training_payload_sha256s)))
            and bool(self.training_payload_sha256s),
            "training payload digests must be unique and sorted",
        )
        for value in (*self.authority_sha256s, *self.credit_sha256s, *self.training_payload_sha256s):
            _digest(value, "snapshot referenced digest")
        for rows, label in (
            (self.main_effect_terms, "main-effect terms"),
            (self.difference_credit_terms, "difference-credit terms"),
        ):
            _require(rows == tuple(sorted(rows)), f"{label} must be canonical")
            for actor, total, count in rows:
                _stable_ref(actor, "snapshot actor")
                _require(
                    isinstance(total, int) and not isinstance(total, bool),
                    "term total must be integer",
                )
                _nonnegative_int(count, "term count", positive=True)
        _require(
            {actor for actor, _, _ in self.main_effect_terms}
            == {actor for actor, _, _ in self.difference_credit_terms},
            "snapshot main/difference actor domains drifted",
        )
        _require(
            {actor: count for actor, _, count in self.main_effect_terms}
            == {actor: count for actor, _, count in self.difference_credit_terms},
            "snapshot main/difference sample counts drifted",
        )
        _require(self.pair_terms == tuple(sorted(self.pair_terms)), "pair terms must be canonical")
        for left, right, total, count in self.pair_terms:
            _require(left < right, "snapshot pair order is noncanonical")
            _require(isinstance(total, int) and not isinstance(total, bool), "pair total must be integer")
            _nonnegative_int(count, "pair count", positive=True)
        _require(
            self.evidence_class is EvidenceClass.ORACLE_NONPROMOTABLE,
            "scripted snapshot lost oracle-nonpromotable taint",
        )
        _require(type(self.accounting) is CoalitionEvidenceAccounting, "snapshot accounting must be exact")
        _require(self.accounting.charge_applied is False, "snapshot accounting was self-applied")
        _require(
            self.scripted_fixture_only is True
            and self.causal_effect_claim_allowed is False
            and self.consequence_grounded_credit_claim_allowed is False,
            "snapshot escaped its scripted noncausal ceiling",
        )
        _require(self.activation_enabled is False, "credit snapshot cannot activate")
        _require(self.scientific_promotion_allowed is False, "credit snapshot cannot promote")
        _require(
            self.snapshot_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "interaction credit snapshot self-hash mismatch",
        )
        _require(
            self.retained_state_bytes == len(canonical_bytes(self.payload())),
            "snapshot retained-state bytes do not equal its actual canonical payload",
        )

    @classmethod
    def create(
        cls,
        credits: Sequence[ExactCoalitionCredit],
        *,
        authorities: Sequence[ForkAuthority],
        config: CoalitionEvidenceConfig,
        fit_tick: int,
    ) -> Self:
        _require(type(config) is CoalitionEvidenceConfig, "snapshot config must be exact")
        credit_input = tuple(credits)
        authority_input = tuple(authorities)
        _require(
            all(type(row) is ExactCoalitionCredit for row in credit_input),
            "snapshot credits must be exact",
        )
        _require(
            all(type(row) is ForkAuthority for row in authority_input),
            "snapshot authorities must be exact",
        )
        rows = tuple(sorted(credit_input, key=lambda row: row.credit_sha256))
        authority_rows = tuple(sorted(authority_input, key=lambda row: row.authority_sha256))
        _require(bool(rows), "interaction snapshot requires credit")
        _require(len(rows) <= config.max_credit_records, "credit record cap exceeded")
        _nonnegative_int(fit_tick, "fit_tick")
        _require(len({row.credit_sha256 for row in rows}) == len(rows), "duplicate credit record")
        _require(
            len(authority_rows) == len(rows)
            and len({row.authority_sha256 for row in authority_rows}) == len(authority_rows)
            and {row.authority_sha256 for row in authority_rows} == {row.authority_sha256 for row in rows},
            "snapshot credit/authority replay set mismatch",
        )
        _require(
            all(
                row.config_sha256 == config.sha256
                and row.candidate_registry_sha256 == config.candidate_registry_sha256
                and row.assembly_sha256 == config.assembly_sha256
                for row in rows
            ),
            "credit records do not share the requested config/registry/assembly authority",
        )
        authority_by_id = {row.authority_sha256: row for row in authority_rows}
        for row in rows:
            authority = authority_by_id[row.authority_sha256]
            _require(authority.config.sha256 == config.sha256, "snapshot authority/config drift")
            _require(
                ExactCoalitionCredit.derive(authority).payload() == row.payload(),
                "credit record does not replay exactly from its fork authority",
            )
        _require(all(row.available_tick <= fit_tick for row in rows), "future credit crossed the fit cutoff")
        main_effects: dict[str, list[int]] = defaultdict(list)
        differences: dict[str, list[int]] = defaultdict(list)
        pairs: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows:
            for actor, value in row.singleton_main_effect_milli:
                main_effects[actor].append(value)
            for actor, value in row.individual_difference_credit_milli:
                differences[actor].append(value)
            for left, right, value in row.pair_interaction_milli:
                pairs[(left, right)].append(value)
        main_effect_terms = tuple(
            (actor, sum(values), len(values)) for actor, values in sorted(main_effects.items())
        )
        difference_terms = tuple(
            (actor, sum(values), len(values)) for actor, values in sorted(differences.items())
        )
        pair_terms = tuple(
            (left, right, sum(values), len(values)) for (left, right), values in sorted(pairs.items())
        )
        target_core = {
            "config_sha256": config.sha256,
            "fit_tick": fit_tick,
            "authority_sha256s": [row.authority_sha256 for row in authority_rows],
            "authority_actor_orders": [
                [row.authority_sha256, list(row.full_coalition_actor_ids)] for row in authority_rows
            ],
            "credit_sha256s": [row.credit_sha256 for row in rows],
            "main_effect_terms": [list(row) for row in main_effect_terms],
            "difference_credit_terms": [list(row) for row in difference_terms],
            "pair_terms": [list(row) for row in pair_terms],
        }
        accounting = CoalitionEvidenceAccounting.create(
            stage="fit-interaction-credit-snapshot",
            source_payload={
                "credits": [row.payload() for row in rows],
                "authority_sha256s": [row.authority_sha256 for row in authority_rows],
            },
            target_payload=target_core,
            validation_operations=sum(
                len(row.singleton_main_effect_milli)
                + len(row.individual_difference_credit_milli)
                + len(row.pair_interaction_milli)
                + len(authority_by_id[row.authority_sha256].forks)
                for row in rows
            )
            + len(rows),
            work_bucket="learning",
            payload_limit=config.max_payload_bytes,
        )
        _require(
            accounting.work.total_work <= config.max_work_units,
            "credit-snapshot fit work cap exceeded",
        )
        evidence_class = EvidenceClass.ORACLE_NONPROMOTABLE
        training_ids = tuple(sorted({event_id for row in rows for event_id in row.training_event_ids}))
        training_payloads = tuple(sorted({value for row in rows for value in row.training_payload_sha256s}))
        base = {
            "schema": SNAPSHOT_SCHEMA,
            "config_sha256": config.sha256,
            "candidate_registry_sha256": config.candidate_registry_sha256,
            "assembly_sha256": config.assembly_sha256,
            "fit_tick": fit_tick,
            "authority_sha256s": [row.authority_sha256 for row in authority_rows],
            "authority_actor_orders": [
                [row.authority_sha256, list(row.full_coalition_actor_ids)] for row in authority_rows
            ],
            "credit_sha256s": [row.credit_sha256 for row in rows],
            "training_event_ids": list(training_ids),
            "training_payload_sha256s": list(training_payloads),
            "main_effect_terms": [list(row) for row in main_effect_terms],
            "difference_credit_terms": [list(row) for row in difference_terms],
            "pair_terms": [list(row) for row in pair_terms],
            "evidence_class": evidence_class.value,
            "accounting": accounting.payload(),
            "scripted_fixture_only": True,
            "causal_effect_claim_allowed": False,
            "consequence_grounded_credit_claim_allowed": False,
            "activation_enabled": False,
            "scientific_promotion_allowed": False,
        }
        retained = 0
        for _ in range(16):
            candidate = {
                **base,
                "retained_state_bytes": retained,
                "snapshot_sha256": "0" * 64,
            }
            measured = len(_strict_bytes(candidate, limit=config.max_snapshot_bytes, label="credit snapshot"))
            if measured == retained:
                break
            retained = measured
        _require(
            retained
            == len(
                _strict_bytes(
                    {**base, "retained_state_bytes": retained, "snapshot_sha256": "0" * 64},
                    limit=config.max_snapshot_bytes,
                    label="credit snapshot",
                )
            ),
            "snapshot retained-byte fixed point did not converge",
        )
        core = {**base, "retained_state_bytes": retained}
        return cls(
            config_sha256=config.sha256,
            candidate_registry_sha256=config.candidate_registry_sha256,
            assembly_sha256=config.assembly_sha256,
            fit_tick=fit_tick,
            authority_sha256s=tuple(row.authority_sha256 for row in authority_rows),
            authority_actor_orders=tuple(
                (row.authority_sha256, row.full_coalition_actor_ids) for row in authority_rows
            ),
            credit_sha256s=tuple(row.credit_sha256 for row in rows),
            training_event_ids=training_ids,
            training_payload_sha256s=training_payloads,
            main_effect_terms=main_effect_terms,
            difference_credit_terms=difference_terms,
            pair_terms=pair_terms,
            evidence_class=evidence_class,
            retained_state_bytes=retained,
            accounting=accounting,
            scripted_fixture_only=True,
            causal_effect_claim_allowed=False,
            consequence_grounded_credit_claim_allowed=False,
            activation_enabled=False,
            scientific_promotion_allowed=False,
            snapshot_sha256=canonical_sha256(core),
            _validation_token=_SNAPSHOT_TOKEN,
        )

    def main_effect_value(self, actor_id: str) -> Fraction:
        for actor, total, count in self.main_effect_terms:
            if actor == actor_id:
                return Fraction(total, count)
        return Fraction(0)

    def difference_credit_value(self, actor_id: str) -> Fraction:
        for actor, total, count in self.difference_credit_terms:
            if actor == actor_id:
                return Fraction(total, count)
        return Fraction(0)

    def pair_value(self, left: str, right: str) -> Fraction:
        key = tuple(sorted((left, right)))
        for row_left, row_right, total, count in self.pair_terms:
            if (row_left, row_right) == key:
                return Fraction(total, count)
        return Fraction(0)

    @property
    def pairwise_source_exact(self) -> bool:
        return all(len(actor_order) == 2 for _, actor_order in self.authority_actor_orders)

    def pairwise_reconstruction_exact_for(self, actor_ids: Sequence[str]) -> bool:
        coalition = _canonical_actor_ids(tuple(actor_ids), "reconstruction coalition")
        if not self.pairwise_source_exact or len(coalition) > 2:
            return False
        main_actors = {actor for actor, _, _ in self.main_effect_terms}
        if not set(coalition) <= main_actors:
            return False
        if len(coalition) < 2:
            return True
        return all(actor_order == coalition for _, actor_order in self.authority_actor_orders)

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA,
            "config_sha256": self.config_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "assembly_sha256": self.assembly_sha256,
            "fit_tick": self.fit_tick,
            "authority_sha256s": list(self.authority_sha256s),
            "authority_actor_orders": [
                [authority_sha256, list(actor_order)]
                for authority_sha256, actor_order in self.authority_actor_orders
            ],
            "credit_sha256s": list(self.credit_sha256s),
            "training_event_ids": list(self.training_event_ids),
            "training_payload_sha256s": list(self.training_payload_sha256s),
            "main_effect_terms": [list(row) for row in self.main_effect_terms],
            "difference_credit_terms": [list(row) for row in self.difference_credit_terms],
            "pair_terms": [list(row) for row in self.pair_terms],
            "evidence_class": self.evidence_class.value,
            "retained_state_bytes": self.retained_state_bytes,
            "accounting": self.accounting.payload(),
            "scripted_fixture_only": self.scripted_fixture_only,
            "causal_effect_claim_allowed": self.causal_effect_claim_allowed,
            "consequence_grounded_credit_claim_allowed": self.consequence_grounded_credit_claim_allowed,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["snapshot_sha256"] = self.snapshot_sha256
        return result


@dataclass(frozen=True, slots=True)
class ShadowCoalitionScore:
    actor_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    main_effect_value_numerator: int
    main_effect_value_denominator: int
    interaction_value_numerator: int
    interaction_value_denominator: int
    resource_debit_milli: int
    net_value_numerator: int
    net_value_denominator: int
    heterogeneous: bool
    interaction_term_positive: bool
    pairwise_reconstruction_exact: bool
    scripted_fixture_only: bool
    score_sha256: str

    def __post_init__(self) -> None:
        _canonical_actor_ids(self.actor_ids, "score actors")
        _require(len(self.actor_ids) == len(self.candidate_ids), "score actor/candidate arity mismatch")
        for candidate_id in self.candidate_ids:
            _identifier(candidate_id, "score candidate")
        for name in (
            "main_effect_value_denominator",
            "interaction_value_denominator",
            "net_value_denominator",
        ):
            _nonnegative_int(getattr(self, name), name, positive=True)
        _nonnegative_int(self.resource_debit_milli, "resource debit")
        _require(
            type(self.heterogeneous) is bool
            and type(self.interaction_term_positive) is bool
            and type(self.pairwise_reconstruction_exact) is bool
            and self.scripted_fixture_only is True,
            "score flags must be boolean",
        )
        _digest(self.score_sha256, "score_sha256")
        _require(
            self.score_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "shadow coalition score self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        actor_ids: tuple[str, ...],
        candidate_ids: tuple[str, ...],
        main_effect: Fraction,
        interaction: Fraction,
        debit: int,
        pairwise_source_exact: bool,
    ) -> Self:
        _require(type(pairwise_source_exact) is bool, "pairwise source exactness must be boolean")
        net = main_effect + interaction - debit
        reconstruction_exact = pairwise_source_exact and len(actor_ids) <= 2
        core = {
            "schema": SCORE_SCHEMA,
            "actor_ids": list(actor_ids),
            "candidate_ids": list(candidate_ids),
            "main_effect_value": [main_effect.numerator, main_effect.denominator],
            "interaction_value": [interaction.numerator, interaction.denominator],
            "resource_debit_milli": debit,
            "net_value": [net.numerator, net.denominator],
            "heterogeneous": len(set(candidate_ids)) > 1,
            "interaction_term_positive": interaction > 0,
            "pairwise_reconstruction_exact": reconstruction_exact,
            "scripted_fixture_only": True,
        }
        return cls(
            actor_ids=actor_ids,
            candidate_ids=candidate_ids,
            main_effect_value_numerator=main_effect.numerator,
            main_effect_value_denominator=main_effect.denominator,
            interaction_value_numerator=interaction.numerator,
            interaction_value_denominator=interaction.denominator,
            resource_debit_milli=debit,
            net_value_numerator=net.numerator,
            net_value_denominator=net.denominator,
            heterogeneous=len(set(candidate_ids)) > 1,
            interaction_term_positive=interaction > 0,
            pairwise_reconstruction_exact=reconstruction_exact,
            scripted_fixture_only=True,
            score_sha256=canonical_sha256(core),
        )

    @property
    def net_value(self) -> Fraction:
        return Fraction(self.net_value_numerator, self.net_value_denominator)

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": SCORE_SCHEMA,
            "actor_ids": list(self.actor_ids),
            "candidate_ids": list(self.candidate_ids),
            "main_effect_value": [
                self.main_effect_value_numerator,
                self.main_effect_value_denominator,
            ],
            "interaction_value": [
                self.interaction_value_numerator,
                self.interaction_value_denominator,
            ],
            "resource_debit_milli": self.resource_debit_milli,
            "net_value": [self.net_value_numerator, self.net_value_denominator],
            "heterogeneous": self.heterogeneous,
            "interaction_term_positive": self.interaction_term_positive,
            "pairwise_reconstruction_exact": self.pairwise_reconstruction_exact,
            "scripted_fixture_only": self.scripted_fixture_only,
        }
        if include_digest:
            result["score_sha256"] = self.score_sha256
        return result


def _readiness_payload(row: ReadinessEstimate) -> dict[str, Any]:
    return {
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


@dataclass(frozen=True, slots=True)
class ShadowArbitrationProposal:
    request_event_id: str
    request_sha256: str
    config_sha256: str
    credit_snapshot_sha256: str
    candidate_registry_sha256: str
    assembly_sha256: str
    scores: tuple[ShadowCoalitionScore, ...]
    proposed_actor_ids: tuple[str, ...]
    scripted_interaction_term_positive: bool
    pairwise_reconstruction_exact: bool
    scripted_fixture_only: bool
    causal_effect_claim_allowed: bool
    consequence_grounded_credit_claim_allowed: bool
    cooperation_claim_allowed: bool
    consumable_by_runtime: bool
    source_replay_verified: bool
    blockers: tuple[str, ...]
    accounting: CoalitionEvidenceAccounting
    applied: bool
    activation_enabled: bool
    dispatch_authority: bool
    commitment_authority: bool
    effect_authority: bool
    update_authority: bool
    scientific_promotion_allowed: bool
    proposal_sha256: str

    def __post_init__(self) -> None:
        _stable_ref(self.request_event_id, "request_event_id")
        for value, label in (
            (self.request_sha256, "request_sha256"),
            (self.config_sha256, "config_sha256"),
            (self.credit_snapshot_sha256, "credit_snapshot_sha256"),
            (self.candidate_registry_sha256, "candidate_registry_sha256"),
            (self.assembly_sha256, "assembly_sha256"),
            (self.proposal_sha256, "proposal_sha256"),
        ):
            _digest(value, label)
        _require(isinstance(self.scores, tuple), "proposal scores must be immutable")
        _canonical_actor_ids(self.proposed_actor_ids, "proposed actor ids")
        _require(self.blockers == tuple(sorted(set(self.blockers))), "proposal blockers must be canonical")
        _require(
            type(self.scripted_interaction_term_positive) is bool
            and type(self.pairwise_reconstruction_exact) is bool
            and self.scripted_fixture_only is True
            and self.causal_effect_claim_allowed is False
            and self.consequence_grounded_credit_claim_allowed is False
            and self.cooperation_claim_allowed is False
            and self.consumable_by_runtime is False,
            "shadow interaction/cooperation flags violated the Gate-A evidence ceiling",
        )
        _require(self.source_replay_verified is True, "shadow source replay was not verified")
        for authority_flag in (
            self.applied,
            self.activation_enabled,
            self.dispatch_authority,
            self.commitment_authority,
            self.effect_authority,
            self.update_authority,
            self.scientific_promotion_allowed,
        ):
            _require(authority_flag is False, "shadow arbitration result gained authority")
        _require(
            {
                "shadow-only",
                "activation-disabled",
                "scripted-noncausal-fixture-only",
                "no-action-effect-authority",
                "accounting-unapplied",
            }
            <= set(self.blockers),
            "shadow arbitration blockers lost the authority fence",
        )
        _require(type(self.accounting) is CoalitionEvidenceAccounting, "proposal accounting must be exact")
        _require(
            self.proposal_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "shadow arbitration proposal self-hash mismatch",
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": PROPOSAL_SCHEMA,
            "request_event_id": self.request_event_id,
            "request_sha256": self.request_sha256,
            "config_sha256": self.config_sha256,
            "credit_snapshot_sha256": self.credit_snapshot_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "assembly_sha256": self.assembly_sha256,
            "scores": [row.payload() for row in self.scores],
            "proposed_actor_ids": list(self.proposed_actor_ids),
            "scripted_interaction_term_positive": self.scripted_interaction_term_positive,
            "pairwise_reconstruction_exact": self.pairwise_reconstruction_exact,
            "scripted_fixture_only": self.scripted_fixture_only,
            "causal_effect_claim_allowed": self.causal_effect_claim_allowed,
            "consequence_grounded_credit_claim_allowed": self.consequence_grounded_credit_claim_allowed,
            "cooperation_claim_allowed": self.cooperation_claim_allowed,
            "consumable_by_runtime": self.consumable_by_runtime,
            "source_replay_verified": self.source_replay_verified,
            "blockers": list(self.blockers),
            "accounting_sha256": self.accounting.accounting_sha256,
            "applied": self.applied,
            "activation_enabled": self.activation_enabled,
            "dispatch_authority": self.dispatch_authority,
            "commitment_authority": self.commitment_authority,
            "effect_authority": self.effect_authority,
            "update_authority": self.update_authority,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["proposal_sha256"] = self.proposal_sha256
        return result


def assess_shadow_coalitions(
    request: DispatchRequest,
    *,
    snapshot: InteractionCreditSnapshot,
    config: CoalitionEvidenceConfig,
    registry: PerspectiveCandidateRegistry,
    assembly: SubstrateAssembly,
    bindings: Sequence[ActorPerspectiveBinding],
    credits: Sequence[ExactCoalitionCredit],
    authorities: Sequence[ForkAuthority],
) -> ShadowArbitrationProposal:
    """Rank a finite request without implementing or returning runtime dispatch authority."""

    _require(type(request) is DispatchRequest, "shadow request must be an exact DispatchRequest")
    _require(type(snapshot) is InteractionCreditSnapshot, "credit snapshot must be exact")
    _require(type(request.event_header) is DispatchEventHeader, "request header must be exact")
    _require(type(request.mode) is CandidateMode, "request mode must be typed")
    _require(type(request.caps) is RuntimeCaps, "request caps must be exact")
    _nonnegative_int(request.reasoning_round, "reasoning_round")
    normalized_bindings = tuple(sorted(bindings, key=lambda row: row.actor_id))
    _validate_binding_authority(normalized_bindings, registry, assembly, config)
    credit_rows = tuple(credits)
    authority_rows = tuple(authorities)
    _require(
        all(type(row) is ExactCoalitionCredit for row in credit_rows),
        "shadow source credits must be exact",
    )
    _require(
        all(type(row) is ForkAuthority for row in authority_rows),
        "shadow source authorities must be exact",
    )
    supplied_authority_ids = tuple(sorted(row.authority_sha256 for row in authority_rows))
    supplied_credit_ids = tuple(sorted(row.credit_sha256 for row in credit_rows))
    _require(
        supplied_authority_ids == snapshot.authority_sha256s
        and len(set(supplied_authority_ids)) == len(supplied_authority_ids),
        "shadow authorities do not provide exact one-to-one snapshot coverage",
    )
    _require(
        supplied_credit_ids == snapshot.credit_sha256s
        and len(set(supplied_credit_ids)) == len(supplied_credit_ids),
        "shadow credits do not provide exact one-to-one snapshot coverage",
    )
    _require(
        tuple(sorted(row.authority_sha256 for row in credit_rows)) == supplied_authority_ids,
        "shadow credit-to-authority coverage is not one-to-one",
    )
    current_binding_by_actor = {row.actor_id: row for row in normalized_bindings}
    for authority in authority_rows:
        _require(
            authority.config.payload() == config.payload(),
            "shadow fork authority retained a different configuration",
        )
        replayed_authority = ForkAuthority.create(
            config=authority.config,
            registry=registry,
            assembly=assembly,
            bindings=authority.bindings,
            forks=authority.forks,
            full_coalition_actor_ids=authority.full_coalition_actor_ids,
        )
        _require(
            replayed_authority.payload() == authority.payload(),
            "fork authority does not replay against the exact registry/assembly",
        )
        _require(
            all(
                current_binding_by_actor.get(binding.actor_id) == binding
                for binding in authority.bindings
                if binding.actor_id in authority.full_coalition_actor_ids
            ),
            "historical actor-perspective bindings drifted at shadow assessment",
        )
    replayed_snapshot = InteractionCreditSnapshot.create(
        credit_rows,
        authorities=authority_rows,
        config=config,
        fit_tick=snapshot.fit_tick,
    )
    _require(
        replayed_snapshot.payload() == snapshot.payload(),
        "credit snapshot does not replay from its exact source authorities",
    )
    _require(snapshot.config_sha256 == config.sha256, "credit/config authority mismatch")
    _require(snapshot.candidate_registry_sha256 == registry.sha256, "credit/registry authority mismatch")
    _require(snapshot.assembly_sha256 == assembly.assembly_sha256, "credit/assembly authority mismatch")
    _require(
        request.event_header.event_id not in snapshot.training_event_ids,
        "training-event credit leaked into its own dispatch request",
    )
    _require(
        set(request.event_header.source_event_ids).isdisjoint(snapshot.training_event_ids),
        "training ancestry leaked through the dispatch source events",
    )
    _require(
        request.event_header.payload_digest not in snapshot.training_payload_sha256s
        and request.event_header.representation_payload_digest not in snapshot.training_payload_sha256s,
        "training payload authority leaked into the dispatch request",
    )
    _require(
        request.event_header.created_tick > snapshot.fit_tick,
        "shadow request is not later than its delayed credit snapshot",
    )
    _require(request.reasoning_round <= request.caps.R, "request reasoning round exceeds R")
    _require(
        config.max_actors <= request.caps.K and config.max_actors <= request.caps.C,
        "shadow config actor cap exceeds runtime K/C authority",
    )
    _require(config.max_beam <= MAX_HARD_BEAM, "shadow beam exceeds hard bound")
    _require(isinstance(request.readiness, tuple), "request readiness must be immutable")
    _require(
        len(request.readiness) <= min(config.max_actors, request.caps.K),
        "request readiness exceeds the bounded candidate cap",
    )
    _require(
        all(type(row) is ReadinessEstimate for row in request.readiness),
        "request readiness rows must be exact",
    )
    readiness_ids = tuple(row.actor_id for row in request.readiness)
    _require(len(readiness_ids) == len(set(readiness_ids)), "request readiness actor duplicates")
    _require(all(row.compatible for row in request.readiness), "incompatible readiness crossed shadow input")
    binding_by_actor = {row.actor_id: row for row in normalized_bindings}
    _require(set(readiness_ids) <= set(binding_by_actor), "request actor lacks a perspective binding")
    eligible_modes = {SlotMode.INFRASTRUCTURE_INERT, SlotMode.FEATURE_CANDIDATE_INERT}
    candidates = tuple(
        sorted(
            row.actor_id for row in request.readiness if binding_by_actor[row.actor_id].mode in eligible_modes
        )
    )
    readiness_by_actor = {row.actor_id: row for row in request.readiness}
    coalitions: list[tuple[str, ...]] = [()]
    truncated = False
    for size in range(1, min(config.max_actors, len(candidates)) + 1):
        for coalition in combinations(candidates, size):
            if len(coalitions) >= min(config.max_beam, request.caps.B):
                truncated = True
                break
            coalitions.append(coalition)
        if truncated:
            break
    scores: list[ShadowCoalitionScore] = []
    for coalition in coalitions:
        main_effect = sum((snapshot.main_effect_value(actor) for actor in coalition), Fraction(0))
        interaction = sum(
            (snapshot.pair_value(left, right) for left, right in combinations(coalition, 2)),
            Fraction(0),
        )
        debit = 0
        for actor in coalition:
            readiness = readiness_by_actor[actor]
            debit += (
                readiness.predicted_operations * config.compute_price_milli
                + readiness.predicted_message_bytes * config.bandwidth_price_milli
                + _bounded_risk_debit(
                    readiness.stale_state_risk,
                    price_milli=config.stale_risk_price_milli,
                    max_risk_micros=config.max_readiness_risk_micros,
                    label="stale_state_risk",
                )
                + _bounded_risk_debit(
                    readiness.deadline_risk,
                    price_milli=config.deadline_risk_price_milli,
                    max_risk_micros=config.max_readiness_risk_micros,
                    label="deadline_risk",
                )
            )
        candidate_ids = tuple(binding_by_actor[actor].candidate_id for actor in coalition)
        scores.append(
            ShadowCoalitionScore.create(
                actor_ids=coalition,
                candidate_ids=candidate_ids,
                main_effect=main_effect,
                interaction=interaction,
                debit=debit,
                pairwise_source_exact=snapshot.pairwise_reconstruction_exact_for(coalition),
            )
        )
    ranked = tuple(sorted(scores, key=lambda row: (-row.net_value, len(row.actor_ids), row.actor_ids)))
    best = ranked[0]
    proposed = best.actor_ids if best.net_value >= config.minimum_net_value_milli else ()
    blockers = {
        "accounting-unapplied",
        "activation-disabled",
        "no-action-effect-authority",
        "scripted-noncausal-fixture-only",
        "scientific-promotion-blocked",
        "shadow-only",
    }
    if truncated:
        blockers.add("beam-truncated")
    if not candidates:
        blockers.add("no-eligible-candidates")
    if not proposed:
        blockers.add("no-positive-net-value")
    elif all(binding_by_actor[actor].trigger_authority is TriggerAuthority.NONE for actor in proposed):
        blockers.add("no-trigger-authorized-member")
        proposed = ()
    winning = next(row for row in ranked if row.actor_ids == proposed) if proposed else ranked[0]
    distinct_facets = len({binding_by_actor[actor].facet for actor in proposed}) > 1
    scripted_interaction_positive = bool(
        proposed and winning.heterogeneous and distinct_facets and winning.interaction_term_positive
    )
    pairwise_exact = winning.pairwise_reconstruction_exact
    if not pairwise_exact:
        blockers.add("higher-order-interactions-unmodeled")
    blockers.update({"arbitration-standing-control-only", "cooperation-claim-not-authorized"})
    request_payload = {
        "event_header": request.event_header.payload(),
        "readiness": [_readiness_payload(row) for row in request.readiness],
        "mode": request.mode.value,
        "caps": request.caps.payload(),
        "reasoning_round": request.reasoning_round,
    }
    request_sha = canonical_sha256(request_payload)
    target_payload = {
        "scores": [row.payload() for row in ranked],
        "proposed_actor_ids": list(proposed),
        "scripted_interaction_term_positive": scripted_interaction_positive,
        "pairwise_reconstruction_exact": pairwise_exact,
        "scripted_fixture_only": True,
        "causal_effect_claim_allowed": False,
        "consequence_grounded_credit_claim_allowed": False,
        "cooperation_claim_allowed": False,
        "consumable_by_runtime": False,
        "source_replay_verified": True,
        "blockers": sorted(blockers),
    }
    pair_lookups = sum(len(row.actor_ids) * (len(row.actor_ids) - 1) // 2 for row in ranked)
    accounting = CoalitionEvidenceAccounting.create(
        stage="shadow-coalition-ranking",
        source_payload={
            "request": request_payload,
            "credit_snapshot_sha256": snapshot.snapshot_sha256,
            "source_credit_sha256s": sorted(row.credit_sha256 for row in credit_rows),
            "source_authority_sha256s": sorted(row.authority_sha256 for row in authority_rows),
            "binding_sha256s": [row.binding_sha256 for row in normalized_bindings],
        },
        target_payload=target_payload,
        validation_operations=(
            len(request.readiness) * 5
            + len(ranked) * 3
            + pair_lookups
            + len(credit_rows)
            + sum(len(row.forks) for row in authority_rows)
        ),
        work_bucket="dispatch_and_exploration",
        payload_limit=config.max_payload_bytes,
    )
    _require(
        accounting.work.total_work <= config.max_work_units,
        "shadow ranking work cap exceeded",
    )
    core = {
        "schema": PROPOSAL_SCHEMA,
        "request_event_id": request.event_header.event_id,
        "request_sha256": request_sha,
        "config_sha256": config.sha256,
        "credit_snapshot_sha256": snapshot.snapshot_sha256,
        "candidate_registry_sha256": registry.sha256,
        "assembly_sha256": assembly.assembly_sha256,
        "scores": [row.payload() for row in ranked],
        "proposed_actor_ids": list(proposed),
        "scripted_interaction_term_positive": scripted_interaction_positive,
        "pairwise_reconstruction_exact": pairwise_exact,
        "scripted_fixture_only": True,
        "causal_effect_claim_allowed": False,
        "consequence_grounded_credit_claim_allowed": False,
        "cooperation_claim_allowed": False,
        "consumable_by_runtime": False,
        "source_replay_verified": True,
        "blockers": sorted(blockers),
        "accounting_sha256": accounting.accounting_sha256,
        "applied": False,
        "activation_enabled": False,
        "dispatch_authority": False,
        "commitment_authority": False,
        "effect_authority": False,
        "update_authority": False,
        "scientific_promotion_allowed": False,
    }
    return ShadowArbitrationProposal(
        request_event_id=request.event_header.event_id,
        request_sha256=request_sha,
        config_sha256=config.sha256,
        credit_snapshot_sha256=snapshot.snapshot_sha256,
        candidate_registry_sha256=registry.sha256,
        assembly_sha256=assembly.assembly_sha256,
        scores=ranked,
        proposed_actor_ids=proposed,
        scripted_interaction_term_positive=scripted_interaction_positive,
        pairwise_reconstruction_exact=pairwise_exact,
        scripted_fixture_only=True,
        causal_effect_claim_allowed=False,
        consequence_grounded_credit_claim_allowed=False,
        cooperation_claim_allowed=False,
        consumable_by_runtime=False,
        source_replay_verified=True,
        blockers=tuple(sorted(blockers)),
        accounting=accounting,
        applied=False,
        activation_enabled=False,
        dispatch_authority=False,
        commitment_authority=False,
        effect_authority=False,
        update_authority=False,
        scientific_promotion_allowed=False,
        proposal_sha256=canonical_sha256(core),
    )


__all__ = [
    "ACCOUNTING_SCHEMA",
    "ACTIVATION_ENABLED",
    "AUTHORITY_SCHEMA",
    "BINDING_SCHEMA",
    "CONFIG_SCHEMA",
    "CREDIT_SCHEMA",
    "FORK_CONTRACT_SCHEMA",
    "FORK_INTERVENTION_KEY",
    "FORK_PROVENANCE_KEY",
    "FORK_SCHEMA",
    "INTERVENTION_SCHEMA",
    "PROPOSAL_SCHEMA",
    "SCRIPTED_OUTCOME_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SCORE_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "TRACE_CONFIG_FRAME_SCHEMA",
    "UTILITY_SCALARIZER_SCHEMA",
    "ActorPerspectiveBinding",
    "CoalitionEvidenceAccounting",
    "CoalitionEvidenceConfig",
    "CoalitionEvidenceError",
    "CoalitionFork",
    "ExactCoalitionCredit",
    "ForkAuthority",
    "InteractionCreditSnapshot",
    "ShadowArbitrationProposal",
    "ShadowCoalitionScore",
    "assess_shadow_coalitions",
]
