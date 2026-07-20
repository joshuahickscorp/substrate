from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

MEMORY_ORGANIZATION_SCHEMA = "mop-memory-organization/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


REPLAY_NULL = "replay-parity-null"
EPISODIC_HARM_NULL = "episodic-harm-null"
PRIOR_NULLS: tuple[str, ...] = (REPLAY_NULL, EPISODIC_HARM_NULL)

REQUIRED_CAPABILITIES: tuple[str, ...] = ("revision", "provenance", "deletion", "action")

FUTURE_DECISION_CONTROLS: tuple[str, ...] = (
    "no-memory",
    "flat-memory",
    "replay-only",
    "stale-memory",
)

FUTURE_DECISION_TARGET = "future-decisions"
VALUE_TARGETS: tuple[str, ...] = (FUTURE_DECISION_TARGET, "prediction")

DECISION_ARMS: tuple[str, ...] = ("organized-memory",) + FUTURE_DECISION_CONTROLS


class MemoryOrganizationRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise MemoryOrganizationRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise MemoryOrganizationRefusal(f"{label} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, label: str) -> None:
    if not value.strip():
        raise MemoryOrganizationRefusal(f"{label} must not be empty")


def _require_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MemoryOrganizationRefusal(f"{label} must be a finite number")


@dataclass(frozen=True, slots=True)
class CapabilityExercise:
    capability: str
    exercised: bool
    op_log: tuple[str, ...]
    witness_sha256: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.capability not in REQUIRED_CAPABILITIES:
            raise MemoryOrganizationRefusal(f"unsupported capability {self.capability!r}")
        if not self.exercised:
            raise MemoryOrganizationRefusal(
                f"capability {self.capability!r} must be exercised, not only declared"
            )
        if not self.op_log:
            raise MemoryOrganizationRefusal(f"capability {self.capability!r} exercise needs at least one op")
        if any(not str(op).strip() for op in self.op_log):
            raise MemoryOrganizationRefusal("capability op log entries must be non-empty")
        _require_sha256(self.witness_sha256, "CapabilityExercise.witness_sha256")
        expected = canonical_sha256(self._witness_body())
        if expected != self.witness_sha256:
            raise MemoryOrganizationRefusal("capability witness digest does not match its op log")
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("capability claim scope cannot be widened")

    def _witness_body(self) -> dict[str, Any]:
        return {"capability": self.capability, "op_log": list(self.op_log)}

    @classmethod
    def build(cls, capability: str, op_log: Sequence[str]) -> CapabilityExercise:

        ops = tuple(op_log)
        digest = canonical_sha256({"capability": capability, "op_log": list(ops)})
        return cls(capability=capability, exercised=True, op_log=ops, witness_sha256=digest)

    def payload(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "exercised": self.exercised,
            "op_log": list(self.op_log),
            "witness_sha256": self.witness_sha256,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class MemoryOrganizationContract:
    schema: str
    organization_id: str
    exercises: tuple[CapabilityExercise, ...]
    prediction_only_sufficient: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_ORGANIZATION_SCHEMA:
            raise MemoryOrganizationRefusal(f"unsupported memory organization schema {self.schema!r}")
        _require_id(self.organization_id, "MemoryOrganizationContract.organization_id")
        declared = tuple(row.capability for row in self.exercises)
        if declared != REQUIRED_CAPABILITIES:
            raise MemoryOrganizationRefusal("capability coverage or order drift")
        if any(not row.exercised for row in self.exercises):
            raise MemoryOrganizationRefusal("every declared capability must be exercised")
        if self.prediction_only_sufficient:
            raise MemoryOrganizationRefusal(
                "prediction-only improvement is not sufficient to retain a memory organization"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("memory organization claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "organization_id": self.organization_id,
            "exercises": [row.payload() for row in self.exercises],
            "prediction_only_sufficient": self.prediction_only_sufficient,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MatchedRetrievalBudget:
    params: int
    retrieval_ops: int
    memory_bytes: int
    write_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("retrieval_ops", self.retrieval_ops),
            ("memory_bytes", self.memory_bytes),
            ("write_steps", self.write_steps),
        ):
            if value <= 0:
                raise MemoryOrganizationRefusal(f"matched budget {name} must be positive (non-vacuous)")

    def payload(self) -> dict[str, int]:
        return {
            "params": self.params,
            "retrieval_ops": self.retrieval_ops,
            "memory_bytes": self.memory_bytes,
            "write_steps": self.write_steps,
        }


@dataclass(frozen=True, slots=True)
class FutureDecisionValueContract:
    schema: str
    value_target: str
    controls: tuple[str, ...]
    matched: MatchedRetrievalBudget
    matched_cost_required: bool
    replication_min: int
    prediction_only_sufficient: bool
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_ORGANIZATION_SCHEMA:
            raise MemoryOrganizationRefusal(f"unsupported future-decision schema {self.schema!r}")
        if self.value_target != FUTURE_DECISION_TARGET:
            raise MemoryOrganizationRefusal(
                "future-decision value must be measured on future decisions, not prediction"
            )
        if tuple(self.controls) != FUTURE_DECISION_CONTROLS:
            raise MemoryOrganizationRefusal("future-decision controls or order drift")
        if not self.matched_cost_required:
            raise MemoryOrganizationRefusal("future-decision value must require matched retrieval cost")
        if self.replication_min < 2:
            raise MemoryOrganizationRefusal(
                "future-decision value requires at least two independent replications"
            )
        if self.prediction_only_sufficient:
            raise MemoryOrganizationRefusal(
                "prediction-only improvement is not sufficient future-decision value"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("future-decision claim scope cannot be widened")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "value_target": self.value_target,
            "controls": list(self.controls),
            "matched": self.matched.payload(),
            "matched_cost_required": self.matched_cost_required,
            "replication_min": self.replication_min,
            "prediction_only_sufficient": self.prediction_only_sufficient,
            "claim_scope": self.claim_scope,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class EpisodicHarmGuard:
    schema: str
    harm_tolerance: float
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_ORGANIZATION_SCHEMA:
            raise MemoryOrganizationRefusal(f"unsupported episodic-harm schema {self.schema!r}")
        _require_finite(self.harm_tolerance, "EpisodicHarmGuard.harm_tolerance")
        if self.harm_tolerance < 0.0:
            raise MemoryOrganizationRefusal("episodic-harm tolerance must be nonnegative")
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("episodic-harm claim scope cannot be widened")

    def assert_no_episodic_harm(self, *, no_memory_value: float, episodic_value: float) -> float:

        _require_finite(no_memory_value, "no_memory_value")
        _require_finite(episodic_value, "episodic_value")
        harm = float(no_memory_value) - float(episodic_value)
        if harm > self.harm_tolerance:
            raise MemoryOrganizationRefusal(
                "episodic memory measurably harms future decisions; organization refused"
            )
        return harm

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "harm_tolerance": self.harm_tolerance,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class ActivationReceipt:
    schema: str
    issuer: str
    cleared_nulls: tuple[str, ...]
    future_decision_win_replications: int
    matched_cost_confirmed: bool
    episodic_harm_absent: bool
    receipt_sha256: str
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_ORGANIZATION_SCHEMA:
            raise MemoryOrganizationRefusal(f"unsupported activation receipt schema {self.schema!r}")
        _require_nonempty(self.issuer, "ActivationReceipt.issuer")
        if tuple(self.cleared_nulls) != PRIOR_NULLS:
            raise MemoryOrganizationRefusal(
                "activation receipt must clear both named prior nulls in canonical order"
            )
        if self.future_decision_win_replications < 2:
            raise MemoryOrganizationRefusal(
                "activation receipt requires at least two replicated future-decision wins"
            )
        if not self.matched_cost_confirmed:
            raise MemoryOrganizationRefusal("activation receipt requires confirmed matched cost")
        if not self.episodic_harm_absent:
            raise MemoryOrganizationRefusal("activation receipt requires confirmed absence of episodic harm")
        _require_sha256(self.receipt_sha256, "ActivationReceipt.receipt_sha256")
        expected = canonical_sha256(self._receipt_body())
        if expected != self.receipt_sha256:
            raise MemoryOrganizationRefusal("activation receipt digest does not match its body")
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("activation receipt claim scope cannot be widened")

    def _receipt_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "issuer": self.issuer,
            "cleared_nulls": list(self.cleared_nulls),
            "future_decision_win_replications": self.future_decision_win_replications,
            "matched_cost_confirmed": self.matched_cost_confirmed,
            "episodic_harm_absent": self.episodic_harm_absent,
        }

    @classmethod
    def build(cls, *, issuer: str, replications: int) -> ActivationReceipt:

        body = {
            "schema": MEMORY_ORGANIZATION_SCHEMA,
            "issuer": issuer,
            "cleared_nulls": list(PRIOR_NULLS),
            "future_decision_win_replications": replications,
            "matched_cost_confirmed": True,
            "episodic_harm_absent": True,
        }
        return cls(
            schema=MEMORY_ORGANIZATION_SCHEMA,
            issuer=issuer,
            cleared_nulls=PRIOR_NULLS,
            future_decision_win_replications=replications,
            matched_cost_confirmed=True,
            episodic_harm_absent=True,
            receipt_sha256=canonical_sha256(body),
        )

    def payload(self) -> dict[str, Any]:
        body = self._receipt_body()
        body["receipt_sha256"] = self.receipt_sha256
        body["claim_scope"] = self.claim_scope
        return body


@dataclass(frozen=True, slots=True)
class MemoryActivationGate:
    schema: str = MEMORY_ORGANIZATION_SCHEMA
    activation_permitted: bool = False
    receipt: ActivationReceipt | None = None
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != MEMORY_ORGANIZATION_SCHEMA:
            raise MemoryOrganizationRefusal(f"unsupported activation gate schema {self.schema!r}")
        if self.activation_permitted:
            raise MemoryOrganizationRefusal(
                "memory-organization activation cannot be permitted by default construction"
            )
        if self.claim_scope != CLAIM_SCOPE:
            raise MemoryOrganizationRefusal("activation gate claim scope cannot be widened")

    def authorize(self, receipt: ActivationReceipt | None = None) -> None:

        supplied = receipt if receipt is not None else self.receipt
        if supplied is None:
            raise MemoryOrganizationRefusal(
                "memory-organization activation is not earned; no confirmation receipt supplied. "
                "Clear the replay-parity and episodic-harm nulls under matched cost first."
            )
        if tuple(supplied.cleared_nulls) != PRIOR_NULLS:
            raise MemoryOrganizationRefusal("supplied receipt does not clear both named prior nulls")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "activation_permitted": self.activation_permitted,
            "receipt": None if self.receipt is None else self.receipt.payload(),
            "claim_scope": self.claim_scope,
        }


def simulate_capability_ops(seed: int, capability: str) -> tuple[str, ...]:

    if seed < 0:
        raise MemoryOrganizationRefusal("simulation seed must be nonnegative")
    if capability not in REQUIRED_CAPABILITIES:
        raise MemoryOrganizationRefusal(f"unsupported capability {capability!r}")
    digest = canonical_sha256({"seed": seed, "capability": capability})
    count = 2 + (int(digest[:2], 16) % 3)
    return tuple(f"{capability}.op:{digest[4 * i : 4 * i + 4]}" for i in range(count))


def build_memory_organization(seed: int) -> MemoryOrganizationContract:

    exercises = tuple(
        CapabilityExercise.build(capability, simulate_capability_ops(seed, capability))
        for capability in REQUIRED_CAPABILITIES
    )
    return MemoryOrganizationContract(
        schema=MEMORY_ORGANIZATION_SCHEMA,
        organization_id=f"memorg.seed_{seed}",
        exercises=exercises,
        prediction_only_sufficient=False,
    )


def toy_decision_values(seed: int) -> dict[str, float]:

    if seed < 0:
        raise MemoryOrganizationRefusal("decision-value seed must be nonnegative")
    values: dict[str, float] = {}
    for arm in DECISION_ARMS:
        digest = canonical_sha256({"seed": seed, "arm": arm})
        values[arm] = int(digest[:8], 16) / float(0xFFFFFFFF)
    return values


def default_future_decision_contract() -> FutureDecisionValueContract:

    return FutureDecisionValueContract(
        schema=MEMORY_ORGANIZATION_SCHEMA,
        value_target=FUTURE_DECISION_TARGET,
        controls=FUTURE_DECISION_CONTROLS,
        matched=MatchedRetrievalBudget(params=256, retrieval_ops=64, memory_bytes=4096, write_steps=32),
        matched_cost_required=True,
        replication_min=3,
        prediction_only_sufficient=False,
    )


def default_episodic_harm_guard() -> EpisodicHarmGuard:
    return EpisodicHarmGuard(schema=MEMORY_ORGANIZATION_SCHEMA, harm_tolerance=0.0)


def default_activation_gate() -> MemoryActivationGate:
    return MemoryActivationGate()
