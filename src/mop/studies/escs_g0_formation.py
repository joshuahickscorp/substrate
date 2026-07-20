
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from mop.escs.accounting import WorkVector
from mop.escs.g0_genotype import assess_g0_genotype
from mop.escs.perspective_registry import PerspectiveCandidateRegistry
from mop.escs.topology_grammar import TopologyGrammar
from mop.substrate.events import canonical_sha256

from .escs_g0_construction import (
    G0ConstructionAttempt,
    G0ConstructionRequest,
    G0ConstructionSnapshot,
    G0ConstructionStatus,
    verify_g0_construction_attempt,
)
from .escs_g0_shadow_coalition import (
    G0ShadowEpisode,
    G0ShadowTrace,
    verify_g0_shadow_trace,
)

G0_CANDIDATE_BUNDLE_SCHEMA = "mop-escs-g0-candidate-bundle/v1"
G0_TRACE_ASSESSMENT_SCHEMA = "mop-escs-g0-trace-assessment/v1"
G0_OBJECTIVE_VECTOR_SCHEMA = "mop-escs-g0-objective-vector/v1"
G0_FORMATION_COSTS_SCHEMA = "mop-escs-g0-formation-costs/v1"
G0_FORMATION_ATTEMPT_SCHEMA = "mop-escs-g0-formation-attempt/v1"
G0_FORMATION_LEDGER_ENTRY_SCHEMA = "mop-escs-g0-formation-ledger-entry/v1"
G0_FORMATION_LEDGER_SCHEMA = "mop-escs-g0-formation-ledger/v1"
G0_PARETO_DECISION_SCHEMA = "mop-escs-g0-pareto-decision/v1"
G0_PARETO_ARCHIVE_SCHEMA = "mop-escs-g0-pareto-archive/v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _exact(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _problems(rows: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(rows)
    if not all(isinstance(row, str) and row for row in result):
        raise ValueError(f"{label} must contain nonempty strings")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be unique and canonically sorted")
    return result


def _inert_flags(payload: Mapping[str, Any], label: str) -> None:
    _require(payload["counterfactual_only"] is True, f"{label} must remain counterfactual-only")
    for key in (
        "activation_enabled",
        "shadow_execution_authorized",
        "factual_effects",
        "factual_mutation_authorized",
        "scientific_promotion_allowed",
    ):
        _require(payload[key] is False, f"{label} cannot authorize {key}")


_INERT = {
    "counterfactual_only": True,
    "activation_enabled": False,
    "shadow_execution_authorized": False,
    "factual_effects": False,
    "factual_mutation_authorized": False,
    "scientific_promotion_allowed": False,
}


class G0CandidateOrigin(StrEnum):
    BASE = "base"
    MUTATION = "mutation"


@dataclass(frozen=True, slots=True)
class G0CandidateBundle:
    candidate_id: str
    origin: G0CandidateOrigin
    snapshot: G0ConstructionSnapshot
    grammar_sha256: str
    candidate_registry_sha256: str
    parent_candidate_sha256: str | None
    mutation_request_sha256: str | None
    construction_attempt_sha256: str | None
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_effects: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    bundle_sha256: str
    schema: str = G0_CANDIDATE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_CANDIDATE_BUNDLE_SCHEMA, "unsupported G0 candidate schema")
        _id(self.candidate_id, "G0 candidate_id")
        _require(isinstance(self.origin, G0CandidateOrigin), "G0 candidate origin must be typed")
        _require(type(self.snapshot) is G0ConstructionSnapshot, "G0 candidate snapshot must be exact")
        _digest(self.grammar_sha256, "G0 candidate grammar digest")
        _digest(self.candidate_registry_sha256, "G0 candidate registry digest")
        if self.origin is G0CandidateOrigin.BASE:
            _require(
                self.parent_candidate_sha256 is None
                and self.mutation_request_sha256 is None
                and self.construction_attempt_sha256 is None,
                "base G0 candidates cannot claim mutation lineage",
            )
        else:
            _digest(self.parent_candidate_sha256, "G0 candidate parent digest")
            _digest(self.mutation_request_sha256, "G0 candidate mutation request digest")
            _digest(self.construction_attempt_sha256, "G0 candidate construction attempt digest")
        _inert_flags(self.payload(include_digest=False), "G0 candidate")
        _digest(self.bundle_sha256, "G0 candidate bundle digest")
        _require(
            self.bundle_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 candidate bundle self-hash mismatch",
        )

    @classmethod
    def create_base(
        cls,
        *,
        candidate_id: str,
        snapshot: G0ConstructionSnapshot,
        grammar: TopologyGrammar,
        candidate_registry: PerspectiveCandidateRegistry,
    ) -> Self:
        _validate_snapshot(snapshot, grammar=grammar, candidate_registry=candidate_registry)
        return cls._create(
            candidate_id=candidate_id,
            origin=G0CandidateOrigin.BASE,
            snapshot=snapshot,
            grammar_sha256=grammar.grammar_sha256,
            candidate_registry_sha256=candidate_registry.sha256,
            parent_candidate_sha256=None,
            mutation_request_sha256=None,
            construction_attempt_sha256=None,
        )

    @classmethod
    def create_derived(
        cls,
        *,
        candidate_id: str,
        parent: G0CandidateBundle,
        request: G0ConstructionRequest,
        construction_attempt: G0ConstructionAttempt,
    ) -> Self:
        _require(type(parent) is cls, "G0 parent candidate must be an exact bundle")
        _require(type(request) is G0ConstructionRequest, "G0 mutation request must be exact")
        _require(
            type(construction_attempt) is G0ConstructionAttempt,
            "G0 construction attempt must be exact",
        )
        _require(
            construction_attempt.status is G0ConstructionStatus.APPLIED_SHADOW
            and construction_attempt.candidate_snapshot is not None,
            "derived G0 candidate requires an applied construction attempt",
        )
        _require(
            request.source_snapshot_sha256
            == parent.snapshot.snapshot_sha256
            == construction_attempt.source_snapshot_sha256,
            "derived G0 candidate parent snapshot authority mismatch",
        )
        _require(
            request.request_sha256 == construction_attempt.request_sha256,
            "derived G0 candidate mutation authority mismatch",
        )
        _require(
            parent.grammar_sha256 == construction_attempt.grammar_sha256
            and parent.candidate_registry_sha256 == construction_attempt.candidate_registry_sha256,
            "derived G0 candidate implementation authorities changed",
        )
        assert construction_attempt.candidate_snapshot is not None
        return cls._create(
            candidate_id=candidate_id,
            origin=G0CandidateOrigin.MUTATION,
            snapshot=construction_attempt.candidate_snapshot,
            grammar_sha256=parent.grammar_sha256,
            candidate_registry_sha256=parent.candidate_registry_sha256,
            parent_candidate_sha256=parent.bundle_sha256,
            mutation_request_sha256=request.request_sha256,
            construction_attempt_sha256=construction_attempt.attempt_sha256,
        )

    @classmethod
    def _create(
        cls,
        *,
        candidate_id: str,
        origin: G0CandidateOrigin,
        snapshot: G0ConstructionSnapshot,
        grammar_sha256: str,
        candidate_registry_sha256: str,
        parent_candidate_sha256: str | None,
        mutation_request_sha256: str | None,
        construction_attempt_sha256: str | None,
    ) -> Self:
        core = {
            "schema": G0_CANDIDATE_BUNDLE_SCHEMA,
            "candidate_id": candidate_id,
            "origin": origin.value,
            "snapshot": snapshot.payload(),
            "grammar_sha256": grammar_sha256,
            "candidate_registry_sha256": candidate_registry_sha256,
            "parent_candidate_sha256": parent_candidate_sha256,
            "mutation_request_sha256": mutation_request_sha256,
            "construction_attempt_sha256": construction_attempt_sha256,
            **_INERT,
        }
        return cls(
            candidate_id=candidate_id,
            origin=origin,
            snapshot=snapshot,
            grammar_sha256=grammar_sha256,
            candidate_registry_sha256=candidate_registry_sha256,
            parent_candidate_sha256=parent_candidate_sha256,
            mutation_request_sha256=mutation_request_sha256,
            construction_attempt_sha256=construction_attempt_sha256,
            counterfactual_only=True,
            activation_enabled=False,
            shadow_execution_authorized=False,
            factual_effects=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            bundle_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {
                "schema",
                "candidate_id",
                "origin",
                "snapshot",
                "grammar_sha256",
                "candidate_registry_sha256",
                "parent_candidate_sha256",
                "mutation_request_sha256",
                "construction_attempt_sha256",
                *_INERT,
                "bundle_sha256",
            },
            "G0 candidate bundle",
        )
        snapshot = payload["snapshot"]
        _require(isinstance(snapshot, Mapping), "G0 candidate snapshot payload must be a mapping")
        return cls(
            schema=payload["schema"],
            candidate_id=payload["candidate_id"],
            origin=G0CandidateOrigin(payload["origin"]),
            snapshot=G0ConstructionSnapshot.from_payload(snapshot),
            grammar_sha256=payload["grammar_sha256"],
            candidate_registry_sha256=payload["candidate_registry_sha256"],
            parent_candidate_sha256=payload["parent_candidate_sha256"],
            mutation_request_sha256=payload["mutation_request_sha256"],
            construction_attempt_sha256=payload["construction_attempt_sha256"],
            counterfactual_only=payload["counterfactual_only"],
            activation_enabled=payload["activation_enabled"],
            shadow_execution_authorized=payload["shadow_execution_authorized"],
            factual_effects=payload["factual_effects"],
            factual_mutation_authorized=payload["factual_mutation_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            bundle_sha256=payload["bundle_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "snapshot": self.snapshot.payload(),
            "grammar_sha256": self.grammar_sha256,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "parent_candidate_sha256": self.parent_candidate_sha256,
            "mutation_request_sha256": self.mutation_request_sha256,
            "construction_attempt_sha256": self.construction_attempt_sha256,
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_effects": self.factual_effects,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["bundle_sha256"] = self.bundle_sha256
        return result


def _validate_snapshot(
    snapshot: G0ConstructionSnapshot,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
) -> None:
    _require(type(snapshot) is G0ConstructionSnapshot, "G0 snapshot must be exact")
    _require(type(grammar) is TopologyGrammar, "G0 grammar must be exact")
    _require(
        type(candidate_registry) is PerspectiveCandidateRegistry,
        "G0 candidate registry must be exact",
    )
    _require(
        grammar.candidate_registry_sha256 == candidate_registry.sha256,
        "G0 grammar and candidate registry authorities differ",
    )
    invalid = [
        actor.candidate_id
        for actor in snapshot.actors
        if not assess_g0_genotype(
            actor, grammar=grammar, candidate_registry=candidate_registry
        ).structurally_valid
    ]
    _require(not invalid, f"G0 snapshot contains structurally invalid actors: {sorted(invalid)}")


@dataclass(frozen=True, slots=True)
class G0TraceAssessment:
    task_id: str
    task_authority_sha256: str
    episode_sha256: str
    trace_sha256: str
    scorer_sha256: str
    quality_microunits: int
    robustness_microunits: int
    diversity_microunits: int
    work: WorkVector
    routed_payload_bytes: int
    message_envelope_bytes: int
    retained_state_bytes: int
    retained_state_byte_rounds: int
    assessment_sha256: str
    schema: str = G0_TRACE_ASSESSMENT_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_TRACE_ASSESSMENT_SCHEMA, "unsupported G0 assessment schema")
        _id(self.task_id, "G0 assessment task_id")
        for digest_value, label in (
            (self.task_authority_sha256, "task authority"),
            (self.episode_sha256, "episode"),
            (self.trace_sha256, "trace"),
            (self.scorer_sha256, "scorer"),
            (self.assessment_sha256, "assessment"),
        ):
            _digest(digest_value, f"G0 assessment {label} digest")
        _integer(self.quality_microunits, "G0 assessment quality_microunits")
        _integer(self.robustness_microunits, "G0 assessment robustness_microunits")
        _integer(self.diversity_microunits, "G0 assessment diversity_microunits")
        _require(type(self.work) is WorkVector, "G0 assessment work must be exact")
        for count_value, label in (
            (self.routed_payload_bytes, "routed_payload_bytes"),
            (self.message_envelope_bytes, "message_envelope_bytes"),
            (self.retained_state_bytes, "retained_state_bytes"),
            (self.retained_state_byte_rounds, "retained_state_byte_rounds"),
        ):
            _integer(count_value, f"G0 assessment {label}", nonnegative=True)
        _require(
            self.work.retained_byte_time == self.retained_state_byte_rounds,
            "G0 assessment retained-state work mismatch",
        )
        _require(
            self.assessment_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 trace assessment self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        task_authority_sha256: str,
        episode: G0ShadowEpisode,
        trace: G0ShadowTrace,
        scorer_sha256: str,
        quality_microunits: int,
        robustness_microunits: int,
        diversity_microunits: int,
    ) -> Self:
        _require(type(episode) is G0ShadowEpisode, "G0 assessment episode must be exact")
        _require(type(trace) is G0ShadowTrace, "G0 assessment trace must be exact")
        _require(trace.episode_sha256 == episode.episode_sha256, "G0 assessment episode mismatch")
        _require(
            trace.source_snapshot_sha256 == episode.source_snapshot_sha256,
            "G0 assessment source snapshot authority mismatch",
        )
        _require(
            trace.grammar_sha256 == episode.grammar_sha256
            and trace.candidate_registry_sha256 == episode.candidate_registry_sha256
            and trace.executor_contract_sha256 == episode.executor_contract_sha256,
            "G0 assessment execution authority mismatch",
        )
        core = {
            "schema": G0_TRACE_ASSESSMENT_SCHEMA,
            "task_id": task_id,
            "task_authority_sha256": task_authority_sha256,
            "episode_sha256": episode.episode_sha256,
            "trace_sha256": trace.trace_sha256,
            "scorer_sha256": scorer_sha256,
            "quality_microunits": quality_microunits,
            "robustness_microunits": robustness_microunits,
            "diversity_microunits": diversity_microunits,
            "work": trace.work.payload(),
            "routed_payload_bytes": trace.routed_payload_bytes,
            "message_envelope_bytes": trace.message_envelope_bytes,
            "retained_state_bytes": trace.declared_retained_state_bytes,
            "retained_state_byte_rounds": trace.retained_state_byte_rounds,
        }
        return cls(
            task_id=task_id,
            task_authority_sha256=task_authority_sha256,
            episode_sha256=episode.episode_sha256,
            trace_sha256=trace.trace_sha256,
            scorer_sha256=scorer_sha256,
            quality_microunits=quality_microunits,
            robustness_microunits=robustness_microunits,
            diversity_microunits=diversity_microunits,
            work=trace.work,
            routed_payload_bytes=trace.routed_payload_bytes,
            message_envelope_bytes=trace.message_envelope_bytes,
            retained_state_bytes=trace.declared_retained_state_bytes,
            retained_state_byte_rounds=trace.retained_state_byte_rounds,
            assessment_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {
                "schema",
                "task_id",
                "task_authority_sha256",
                "episode_sha256",
                "trace_sha256",
                "scorer_sha256",
                "quality_microunits",
                "robustness_microunits",
                "diversity_microunits",
                "work",
                "routed_payload_bytes",
                "message_envelope_bytes",
                "retained_state_bytes",
                "retained_state_byte_rounds",
                "assessment_sha256",
            },
            "G0 trace assessment",
        )
        work = payload["work"]
        _require(isinstance(work, Mapping), "G0 assessment work must be a mapping")
        return cls(
            schema=payload["schema"],
            task_id=payload["task_id"],
            task_authority_sha256=payload["task_authority_sha256"],
            episode_sha256=payload["episode_sha256"],
            trace_sha256=payload["trace_sha256"],
            scorer_sha256=payload["scorer_sha256"],
            quality_microunits=payload["quality_microunits"],
            robustness_microunits=payload["robustness_microunits"],
            diversity_microunits=payload["diversity_microunits"],
            work=WorkVector.from_payload(work),
            routed_payload_bytes=payload["routed_payload_bytes"],
            message_envelope_bytes=payload["message_envelope_bytes"],
            retained_state_bytes=payload["retained_state_bytes"],
            retained_state_byte_rounds=payload["retained_state_byte_rounds"],
            assessment_sha256=payload["assessment_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "task_id": self.task_id,
            "task_authority_sha256": self.task_authority_sha256,
            "episode_sha256": self.episode_sha256,
            "trace_sha256": self.trace_sha256,
            "scorer_sha256": self.scorer_sha256,
            "quality_microunits": self.quality_microunits,
            "robustness_microunits": self.robustness_microunits,
            "diversity_microunits": self.diversity_microunits,
            "work": self.work.payload(),
            "routed_payload_bytes": self.routed_payload_bytes,
            "message_envelope_bytes": self.message_envelope_bytes,
            "retained_state_bytes": self.retained_state_bytes,
            "retained_state_byte_rounds": self.retained_state_byte_rounds,
        }
        if include_digest:
            result["assessment_sha256"] = self.assessment_sha256
        return result


def _assessment_sort_key(row: G0TraceAssessment) -> tuple[str, ...]:
    return (
        row.task_id,
        row.task_authority_sha256,
        row.scorer_sha256,
        row.episode_sha256,
        row.assessment_sha256,
    )


def _validated_assessments(
    assessments: Sequence[G0TraceAssessment],
    *,
    require_nonempty: bool,
    require_canonical: bool,
) -> tuple[G0TraceAssessment, ...]:
    rows = tuple(assessments)
    _require(not require_nonempty or bool(rows), "G0 objective requires at least one assessment")
    _require(
        all(type(row) is G0TraceAssessment for row in rows),
        "G0 assessments must be exact",
    )
    if require_canonical:
        _require(
            tuple(_assessment_sort_key(row) for row in rows)
            == tuple(sorted(_assessment_sort_key(row) for row in rows)),
            "G0 assessments must be canonically ordered",
        )
    task_ids = tuple(row.task_id for row in rows)
    task_authorities = tuple(row.task_authority_sha256 for row in rows)
    _require(len(task_ids) == len(set(task_ids)), "G0 assessment task_id duplicated")
    _require(
        len(task_authorities) == len(set(task_authorities)),
        "G0 assessment task authority duplicated",
    )
    return rows


@dataclass(frozen=True, slots=True)
class G0ObjectiveVector:
    evaluation_cohort_sha256: str
    quality_microunits: int
    total_work: int
    communicated_bytes: int
    retained_state_bytes: int
    retained_state_byte_rounds: int
    robustness_microunits: int
    diversity_microunits: int
    objective_sha256: str
    schema: str = G0_OBJECTIVE_VECTOR_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_OBJECTIVE_VECTOR_SCHEMA, "unsupported G0 objective schema")
        _digest(self.evaluation_cohort_sha256, "G0 objective cohort digest")
        _integer(self.quality_microunits, "G0 objective quality")
        _integer(self.robustness_microunits, "G0 objective robustness")
        _integer(self.diversity_microunits, "G0 objective diversity")
        _integer(self.total_work, "G0 objective total work", nonnegative=True)
        _integer(self.communicated_bytes, "G0 objective communicated bytes", nonnegative=True)
        _integer(self.retained_state_bytes, "G0 objective retained state", nonnegative=True)
        _integer(
            self.retained_state_byte_rounds,
            "G0 objective retained state byte-rounds",
            nonnegative=True,
        )
        _digest(self.objective_sha256, "G0 objective digest")
        _require(
            self.objective_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 objective self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        construction_work: int,
        retained_state_bytes: int,
        assessments: Sequence[G0TraceAssessment],
    ) -> Self:
        _integer(construction_work, "G0 objective construction_work", nonnegative=True)
        _integer(retained_state_bytes, "G0 objective retained_state_bytes", nonnegative=True)
        rows = _validated_assessments(
            assessments,
            require_nonempty=True,
            require_canonical=True,
        )
        cohort = canonical_sha256(
            {
                "schema": "mop-escs-g0-evaluation-cohort/v1",
                "tasks": [
                    {
                        "task_id": row.task_id,
                        "task_authority_sha256": row.task_authority_sha256,
                        "scorer_sha256": row.scorer_sha256,
                    }
                    for row in rows
                ],
            }
        )
        quality = sum(row.quality_microunits for row in rows)
        total_work = construction_work + sum(row.work.total_work for row in rows)
        communicated_bytes = sum(row.routed_payload_bytes + row.message_envelope_bytes for row in rows)
        robustness = sum(row.robustness_microunits for row in rows)
        diversity = sum(row.diversity_microunits for row in rows)
        retained_state_byte_rounds = sum(row.retained_state_byte_rounds for row in rows)
        core = {
            "schema": G0_OBJECTIVE_VECTOR_SCHEMA,
            "evaluation_cohort_sha256": cohort,
            "quality_microunits": quality,
            "total_work": total_work,
            "communicated_bytes": communicated_bytes,
            "retained_state_bytes": retained_state_bytes,
            "retained_state_byte_rounds": retained_state_byte_rounds,
            "robustness_microunits": robustness,
            "diversity_microunits": diversity,
        }
        return cls(
            evaluation_cohort_sha256=cohort,
            quality_microunits=quality,
            total_work=total_work,
            communicated_bytes=communicated_bytes,
            retained_state_bytes=retained_state_bytes,
            retained_state_byte_rounds=retained_state_byte_rounds,
            robustness_microunits=robustness,
            diversity_microunits=diversity,
            objective_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {
                "schema",
                "evaluation_cohort_sha256",
                "quality_microunits",
                "total_work",
                "communicated_bytes",
                "retained_state_bytes",
                "retained_state_byte_rounds",
                "robustness_microunits",
                "diversity_microunits",
                "objective_sha256",
            },
            "G0 objective vector",
        )
        return cls(**dict(payload))

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "evaluation_cohort_sha256": self.evaluation_cohort_sha256,
            "quality_microunits": self.quality_microunits,
            "total_work": self.total_work,
            "communicated_bytes": self.communicated_bytes,
            "retained_state_bytes": self.retained_state_bytes,
            "retained_state_byte_rounds": self.retained_state_byte_rounds,
            "robustness_microunits": self.robustness_microunits,
            "diversity_microunits": self.diversity_microunits,
        }
        if include_digest:
            result["objective_sha256"] = self.objective_sha256
        return result


def g0_objective_dominates(left: G0ObjectiveVector, right: G0ObjectiveVector) -> bool:

    if left.evaluation_cohort_sha256 != right.evaluation_cohort_sha256:
        return False
    weak = (
        left.quality_microunits >= right.quality_microunits
        and left.total_work <= right.total_work
        and left.communicated_bytes <= right.communicated_bytes
        and left.retained_state_bytes <= right.retained_state_bytes
        and left.retained_state_byte_rounds <= right.retained_state_byte_rounds
        and left.robustness_microunits >= right.robustness_microunits
        and left.diversity_microunits >= right.diversity_microunits
    )
    strict = (
        left.quality_microunits > right.quality_microunits
        or left.total_work < right.total_work
        or left.communicated_bytes < right.communicated_bytes
        or left.retained_state_bytes < right.retained_state_bytes
        or left.retained_state_byte_rounds < right.retained_state_byte_rounds
        or left.robustness_microunits > right.robustness_microunits
        or left.diversity_microunits > right.diversity_microunits
    )
    return weak and strict


@dataclass(frozen=True, slots=True)
class G0FormationCosts:
    construction_work: int
    trace_work: WorkVector
    routed_payload_bytes: int
    message_envelope_bytes: int
    retained_state_bytes: int
    retained_state_byte_rounds: int
    evaluated_episode_count: int
    costs_sha256: str
    schema: str = G0_FORMATION_COSTS_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_FORMATION_COSTS_SCHEMA, "unsupported G0 costs schema")
        for value, label in (
            (self.construction_work, "construction_work"),
            (self.routed_payload_bytes, "routed_payload_bytes"),
            (self.message_envelope_bytes, "message_envelope_bytes"),
            (self.retained_state_bytes, "retained_state_bytes"),
            (self.retained_state_byte_rounds, "retained_state_byte_rounds"),
            (self.evaluated_episode_count, "evaluated_episode_count"),
        ):
            _integer(value, f"G0 formation costs {label}", nonnegative=True)
        _require(type(self.trace_work) is WorkVector, "G0 formation trace work must be exact")
        _require(
            self.trace_work.retained_byte_time == self.retained_state_byte_rounds,
            "G0 formation retained-state work mismatch",
        )
        _digest(self.costs_sha256, "G0 formation costs digest")
        _require(
            self.costs_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 formation costs self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        construction_work: int,
        retained_state_bytes: int,
        assessments: Sequence[G0TraceAssessment],
    ) -> Self:
        _integer(construction_work, "G0 formation costs construction_work", nonnegative=True)
        _integer(retained_state_bytes, "G0 formation costs retained_state_bytes", nonnegative=True)
        rows = _validated_assessments(
            assessments,
            require_nonempty=False,
            require_canonical=False,
        )
        trace_work = sum((row.work for row in rows), WorkVector.zero())
        routed_payload_bytes = sum(row.routed_payload_bytes for row in rows)
        message_envelope_bytes = sum(row.message_envelope_bytes for row in rows)
        retained_state_byte_rounds = sum(row.retained_state_byte_rounds for row in rows)
        core = {
            "schema": G0_FORMATION_COSTS_SCHEMA,
            "construction_work": construction_work,
            "trace_work": trace_work.payload(),
            "routed_payload_bytes": routed_payload_bytes,
            "message_envelope_bytes": message_envelope_bytes,
            "retained_state_bytes": retained_state_bytes,
            "retained_state_byte_rounds": retained_state_byte_rounds,
            "evaluated_episode_count": len(rows),
        }
        return cls(
            construction_work=construction_work,
            trace_work=trace_work,
            routed_payload_bytes=routed_payload_bytes,
            message_envelope_bytes=message_envelope_bytes,
            retained_state_bytes=retained_state_bytes,
            retained_state_byte_rounds=retained_state_byte_rounds,
            evaluated_episode_count=len(rows),
            costs_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {
                "schema",
                "construction_work",
                "trace_work",
                "routed_payload_bytes",
                "message_envelope_bytes",
                "retained_state_bytes",
                "retained_state_byte_rounds",
                "evaluated_episode_count",
                "costs_sha256",
            },
            "G0 formation costs",
        )
        work = payload["trace_work"]
        _require(isinstance(work, Mapping), "G0 formation trace work must be a mapping")
        return cls(
            schema=payload["schema"],
            construction_work=payload["construction_work"],
            trace_work=WorkVector.from_payload(work),
            routed_payload_bytes=payload["routed_payload_bytes"],
            message_envelope_bytes=payload["message_envelope_bytes"],
            retained_state_bytes=payload["retained_state_bytes"],
            retained_state_byte_rounds=payload["retained_state_byte_rounds"],
            evaluated_episode_count=payload["evaluated_episode_count"],
            costs_sha256=payload["costs_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "construction_work": self.construction_work,
            "trace_work": self.trace_work.payload(),
            "routed_payload_bytes": self.routed_payload_bytes,
            "message_envelope_bytes": self.message_envelope_bytes,
            "retained_state_bytes": self.retained_state_bytes,
            "retained_state_byte_rounds": self.retained_state_byte_rounds,
            "evaluated_episode_count": self.evaluated_episode_count,
        }
        if include_digest:
            result["costs_sha256"] = self.costs_sha256
        return result


class G0FormationStatus(StrEnum):
    ADMITTED_INERT = "admitted-inert"
    EVALUATED = "evaluated"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class G0FormationAttempt:
    formation_attempt_id: str
    parent_candidate_sha256: str | None
    construction_request: G0ConstructionRequest | None
    construction_attempt: G0ConstructionAttempt | None
    candidate: G0CandidateBundle | None
    assessments: tuple[G0TraceAssessment, ...]
    status: G0FormationStatus
    validation_problems: tuple[str, ...]
    costs: G0FormationCosts
    objective: G0ObjectiveVector | None
    counterfactual_only: bool
    activation_enabled: bool
    shadow_execution_authorized: bool
    factual_effects: bool
    factual_mutation_authorized: bool
    scientific_promotion_allowed: bool
    attempt_sha256: str
    schema: str = G0_FORMATION_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_FORMATION_ATTEMPT_SCHEMA, "unsupported G0 formation schema")
        _id(self.formation_attempt_id, "G0 formation attempt_id")
        _problems(self.validation_problems, "G0 formation validation problems")
        if self.parent_candidate_sha256 is not None:
            _digest(self.parent_candidate_sha256, "G0 formation parent digest")
        if self.construction_request is not None:
            _require(type(self.construction_request) is G0ConstructionRequest, "bad G0 request")
        if self.construction_attempt is not None:
            _require(type(self.construction_attempt) is G0ConstructionAttempt, "bad G0 construction")
        if self.candidate is not None:
            _require(type(self.candidate) is G0CandidateBundle, "bad G0 candidate")
        request_present = self.construction_request is not None
        construction_present = self.construction_attempt is not None
        _require(
            request_present == construction_present,
            "G0 formation construction authorities must be present as a pair",
        )
        if request_present:
            assert self.construction_request is not None
            assert self.construction_attempt is not None
            request = self.construction_request
            construction = self.construction_attempt
            _require(
                request.request_sha256 == construction.request_sha256,
                "G0 formation construction request authority mismatch",
            )
            _require(
                request.source_snapshot_sha256 == construction.source_snapshot_sha256,
                "G0 formation construction source authority mismatch",
            )
            if self.candidate is None:
                _require(
                    construction.status is G0ConstructionStatus.REFUSED,
                    "G0 formation omitted an applied construction candidate",
                )
                _require(
                    set(construction.problems).issubset(self.validation_problems),
                    "G0 formation lost construction refusal problems",
                )
            else:
                _require(
                    self.candidate.origin is G0CandidateOrigin.MUTATION,
                    "base G0 formation cannot claim construction authorities",
                )
                _require(
                    construction.status is G0ConstructionStatus.APPLIED_SHADOW
                    and construction.candidate_snapshot == self.candidate.snapshot,
                    "derived G0 formation construction candidate mismatch",
                )
                _require(
                    self.candidate.mutation_request_sha256 == request.request_sha256
                    and self.candidate.construction_attempt_sha256 == construction.attempt_sha256,
                    "derived G0 formation lineage authority mismatch",
                )
                _require(
                    self.candidate.grammar_sha256 == construction.grammar_sha256
                    and self.candidate.candidate_registry_sha256 == construction.candidate_registry_sha256,
                    "derived G0 formation implementation authority mismatch",
                )
        elif self.candidate is not None:
            _require(
                self.candidate.origin is G0CandidateOrigin.BASE,
                "derived G0 formation requires construction authorities",
            )
        _require(
            isinstance(self.assessments, tuple)
            and all(type(row) is G0TraceAssessment for row in self.assessments),
            "G0 formation assessments must be exact immutable records",
        )
        _validated_assessments(
            self.assessments,
            require_nonempty=False,
            require_canonical=True,
        )
        _require(
            self.candidate is not None or not self.assessments,
            "G0 formation cannot assess an absent candidate",
        )
        if self.candidate is not None:
            _require(
                all(
                    row.retained_state_bytes == self.candidate.snapshot.retained_state_bytes
                    for row in self.assessments
                ),
                "G0 formation assessment retained-state authority mismatch",
            )
        _require(isinstance(self.status, G0FormationStatus), "G0 formation status must be typed")
        _require(type(self.costs) is G0FormationCosts, "G0 formation costs must be exact")
        construction_work = (
            self.construction_attempt.charged_work if self.construction_attempt is not None else 0
        )
        retained_state_bytes = (
            self.candidate.snapshot.retained_state_bytes if self.candidate is not None else 0
        )
        expected_costs = G0FormationCosts.create(
            construction_work=construction_work,
            retained_state_bytes=retained_state_bytes,
            assessments=self.assessments,
        )
        _require(self.costs == expected_costs, "G0 formation aggregate costs mismatch")
        if self.objective is not None:
            _require(type(self.objective) is G0ObjectiveVector, "G0 objective must be exact")
        if self.status is G0FormationStatus.REFUSED:
            _require(bool(self.validation_problems), "refused G0 formation must explain refusal")
            _require(self.objective is None, "refused G0 formation cannot enter the archive")
        else:
            _require(self.candidate is not None, "admitted G0 formation requires a candidate")
            _require(not self.validation_problems, "admitted G0 formation cannot have problems")
            if self.status is G0FormationStatus.ADMITTED_INERT:
                _require(not self.assessments and self.objective is None, "inert admission was evaluated")
            else:
                _require(bool(self.assessments) and self.objective is not None, "bad evaluated formation")
                expected_objective = G0ObjectiveVector.create(
                    construction_work=construction_work,
                    retained_state_bytes=retained_state_bytes,
                    assessments=self.assessments,
                )
                _require(
                    self.objective == expected_objective,
                    "G0 formation aggregate objective mismatch",
                )
        if self.candidate is not None:
            _require(
                self.parent_candidate_sha256 == self.candidate.parent_candidate_sha256,
                "G0 formation candidate parent mismatch",
            )
        _inert_flags(self.payload(include_digest=False), "G0 formation attempt")
        _digest(self.attempt_sha256, "G0 formation attempt digest")
        _require(
            self.attempt_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 formation attempt self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        formation_attempt_id: str,
        candidate: G0CandidateBundle | None,
        assessments: Sequence[G0TraceAssessment] = (),
        construction_request: G0ConstructionRequest | None = None,
        construction_attempt: G0ConstructionAttempt | None = None,
        validation_problems: Sequence[str] = (),
    ) -> Self:
        assessment_rows = _validated_assessments(
            assessments,
            require_nonempty=False,
            require_canonical=False,
        )
        rows = tuple(sorted(assessment_rows, key=_assessment_sort_key))
        raw_problems = tuple(validation_problems)
        _require(
            all(isinstance(problem, str) and problem for problem in raw_problems),
            "G0 formation validation problems must contain nonempty strings",
        )
        problem_rows = tuple(sorted(set(raw_problems)))
        parent_sha = candidate.parent_candidate_sha256 if candidate is not None else None
        if candidate is None and construction_attempt is not None:
            _require(
                construction_attempt.status is G0ConstructionStatus.REFUSED,
                "missing G0 candidate requires a refused construction attempt",
            )
            problem_rows = tuple(sorted(set(problem_rows) | set(construction_attempt.problems)))
        construction_work = construction_attempt.charged_work if construction_attempt is not None else 0
        state_bytes = candidate.snapshot.retained_state_bytes if candidate is not None else 0
        costs = G0FormationCosts.create(
            construction_work=construction_work,
            retained_state_bytes=state_bytes,
            assessments=rows,
        )
        if problem_rows:
            status = G0FormationStatus.REFUSED
            objective = None
        elif rows:
            status = G0FormationStatus.EVALUATED
            objective = G0ObjectiveVector.create(
                construction_work=construction_work,
                retained_state_bytes=state_bytes,
                assessments=rows,
            )
        else:
            _require(candidate is not None, "clean inert admission requires a G0 candidate")
            status = G0FormationStatus.ADMITTED_INERT
            objective = None
        core = {
            "schema": G0_FORMATION_ATTEMPT_SCHEMA,
            "formation_attempt_id": formation_attempt_id,
            "parent_candidate_sha256": parent_sha,
            "construction_request": construction_request.payload() if construction_request else None,
            "construction_attempt": construction_attempt.payload() if construction_attempt else None,
            "candidate": candidate.payload() if candidate else None,
            "assessments": [row.payload() for row in rows],
            "status": status.value,
            "validation_problems": list(problem_rows),
            "costs": costs.payload(),
            "objective": objective.payload() if objective else None,
            **_INERT,
        }
        return cls(
            formation_attempt_id=formation_attempt_id,
            parent_candidate_sha256=parent_sha,
            construction_request=construction_request,
            construction_attempt=construction_attempt,
            candidate=candidate,
            assessments=rows,
            status=status,
            validation_problems=problem_rows,
            costs=costs,
            objective=objective,
            counterfactual_only=True,
            activation_enabled=False,
            shadow_execution_authorized=False,
            factual_effects=False,
            factual_mutation_authorized=False,
            scientific_promotion_allowed=False,
            attempt_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {
                "schema",
                "formation_attempt_id",
                "parent_candidate_sha256",
                "construction_request",
                "construction_attempt",
                "candidate",
                "assessments",
                "status",
                "validation_problems",
                "costs",
                "objective",
                *_INERT,
                "attempt_sha256",
            },
            "G0 formation attempt",
        )
        request, construction, candidate = (
            payload["construction_request"],
            payload["construction_attempt"],
            payload["candidate"],
        )
        for value, label in ((request, "request"), (construction, "construction"), (candidate, "candidate")):
            _require(
                value is None or isinstance(value, Mapping), f"G0 formation {label} must be mapping/null"
            )
        assessments = payload["assessments"]
        problems = payload["validation_problems"]
        costs = payload["costs"]
        objective = payload["objective"]
        _require(isinstance(assessments, list), "G0 formation assessments must be a list")
        _require(isinstance(problems, list), "G0 formation problems must be a list")
        _require(isinstance(costs, Mapping), "G0 formation costs must be a mapping")
        _require(objective is None or isinstance(objective, Mapping), "G0 objective must be mapping/null")
        return cls(
            schema=payload["schema"],
            formation_attempt_id=payload["formation_attempt_id"],
            parent_candidate_sha256=payload["parent_candidate_sha256"],
            construction_request=G0ConstructionRequest.from_payload(request) if request else None,
            construction_attempt=G0ConstructionAttempt.from_payload(construction) if construction else None,
            candidate=G0CandidateBundle.from_payload(candidate) if candidate else None,
            assessments=tuple(G0TraceAssessment.from_payload(row) for row in assessments),
            status=G0FormationStatus(payload["status"]),
            validation_problems=tuple(problems),
            costs=G0FormationCosts.from_payload(costs),
            objective=G0ObjectiveVector.from_payload(objective) if objective else None,
            counterfactual_only=payload["counterfactual_only"],
            activation_enabled=payload["activation_enabled"],
            shadow_execution_authorized=payload["shadow_execution_authorized"],
            factual_effects=payload["factual_effects"],
            factual_mutation_authorized=payload["factual_mutation_authorized"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            attempt_sha256=payload["attempt_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "formation_attempt_id": self.formation_attempt_id,
            "parent_candidate_sha256": self.parent_candidate_sha256,
            "construction_request": self.construction_request.payload()
            if self.construction_request
            else None,
            "construction_attempt": self.construction_attempt.payload()
            if self.construction_attempt
            else None,
            "candidate": self.candidate.payload() if self.candidate else None,
            "assessments": [row.payload() for row in self.assessments],
            "status": self.status.value,
            "validation_problems": list(self.validation_problems),
            "costs": self.costs.payload(),
            "objective": self.objective.payload() if self.objective else None,
            "counterfactual_only": self.counterfactual_only,
            "activation_enabled": self.activation_enabled,
            "shadow_execution_authorized": self.shadow_execution_authorized,
            "factual_effects": self.factual_effects,
            "factual_mutation_authorized": self.factual_mutation_authorized,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }
        if include_digest:
            result["attempt_sha256"] = self.attempt_sha256
        return result


def verify_g0_formation_attempt(
    attempt: G0FormationAttempt,
    *,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    parents: Mapping[str, G0CandidateBundle],
    episodes: Mapping[str, G0ShadowEpisode],
    traces: Mapping[str, G0ShadowTrace],
) -> tuple[str, ...]:

    if type(attempt) is not G0FormationAttempt:
        raise ValueError("G0 formation verifier requires an exact attempt")
    problems: list[str] = []
    candidate = attempt.candidate
    if candidate is not None:
        if candidate.grammar_sha256 != grammar.grammar_sha256:
            problems.append("candidate-grammar-authority-mismatch")
        if candidate.candidate_registry_sha256 != candidate_registry.sha256:
            problems.append("candidate-registry-authority-mismatch")
        try:
            _validate_snapshot(candidate.snapshot, grammar=grammar, candidate_registry=candidate_registry)
        except ValueError as exc:
            problems.append(f"candidate-validation-refused:{exc}")
    request = attempt.construction_request
    construction = attempt.construction_attempt
    if request is None or construction is None:
        if candidate is not None and candidate.origin is not G0CandidateOrigin.BASE:
            problems.append("mutation-authorities-absent")
    else:
        parent = parents.get(construction.source_snapshot_sha256)
        if parent is None:
            problems.append("parent-snapshot-authority-absent")
        else:
            try:
                verify_g0_construction_attempt(
                    construction,
                    request,
                    source=parent.snapshot,
                    grammar=grammar,
                    candidate_registry=candidate_registry,
                )
            except ValueError as exc:
                problems.append(f"construction-replay-refused:{exc}")
            if candidate is not None:
                try:
                    replayed_bundle = G0CandidateBundle.create_derived(
                        candidate_id=candidate.candidate_id,
                        parent=parent,
                        request=request,
                        construction_attempt=construction,
                    )
                except ValueError as exc:
                    problems.append(f"candidate-lineage-refused:{exc}")
                else:
                    if replayed_bundle != candidate:
                        problems.append("candidate-bundle-replay-mismatch")
    for assessment in attempt.assessments:
        episode = episodes.get(assessment.episode_sha256)
        trace = traces.get(assessment.trace_sha256)
        if episode is None:
            problems.append(f"assessment-{assessment.task_id}:episode-authority-absent")
            continue
        if trace is None:
            problems.append(f"assessment-{assessment.task_id}:trace-authority-absent")
            continue
        if candidate is None:
            problems.append(f"assessment-{assessment.task_id}:candidate-absent")
            continue
        try:
            replayed_assessment = G0TraceAssessment.create(
                task_id=assessment.task_id,
                task_authority_sha256=assessment.task_authority_sha256,
                episode=episode,
                trace=trace,
                scorer_sha256=assessment.scorer_sha256,
                quality_microunits=assessment.quality_microunits,
                robustness_microunits=assessment.robustness_microunits,
                diversity_microunits=assessment.diversity_microunits,
            )
            trace_problems = verify_g0_shadow_trace(
                trace,
                candidate.snapshot,
                episode,
                grammar=grammar,
                candidate_registry=candidate_registry,
            )
        except ValueError as exc:
            problems.append(f"assessment-{assessment.task_id}:replay-refused:{exc}")
        else:
            if replayed_assessment != assessment:
                problems.append(f"assessment-{assessment.task_id}:receipt-mismatch")
            problems.extend(f"assessment-{assessment.task_id}:{problem}" for problem in trace_problems)
    return tuple(sorted(set(problems)))


@dataclass(frozen=True, slots=True)
class G0FormationLedgerEntry:
    sequence: int
    attempt: G0FormationAttempt
    previous_entry_sha256: str | None
    entry_sha256: str
    schema: str = G0_FORMATION_LEDGER_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_FORMATION_LEDGER_ENTRY_SCHEMA, "unsupported G0 ledger entry schema")
        _integer(self.sequence, "G0 ledger sequence", nonnegative=True)
        _require(type(self.attempt) is G0FormationAttempt, "G0 ledger attempt must be exact")
        if self.previous_entry_sha256 is not None:
            _digest(self.previous_entry_sha256, "G0 ledger previous digest")
        _digest(self.entry_sha256, "G0 ledger entry digest")
        _require(
            self.entry_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 ledger entry self-hash mismatch",
        )

    @classmethod
    def create(cls, *, sequence: int, attempt: G0FormationAttempt, previous_entry_sha256: str | None) -> Self:
        core = {
            "schema": G0_FORMATION_LEDGER_ENTRY_SCHEMA,
            "sequence": sequence,
            "attempt": attempt.payload(),
            "previous_entry_sha256": previous_entry_sha256,
        }
        return cls(
            sequence=sequence,
            attempt=attempt,
            previous_entry_sha256=previous_entry_sha256,
            entry_sha256=canonical_sha256(core),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {"schema", "sequence", "attempt", "previous_entry_sha256", "entry_sha256"},
            "G0 formation ledger entry",
        )
        attempt = payload["attempt"]
        _require(isinstance(attempt, Mapping), "G0 ledger attempt payload must be a mapping")
        return cls(
            schema=payload["schema"],
            sequence=payload["sequence"],
            attempt=G0FormationAttempt.from_payload(attempt),
            previous_entry_sha256=payload["previous_entry_sha256"],
            entry_sha256=payload["entry_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "sequence": self.sequence,
            "attempt": self.attempt.payload(),
            "previous_entry_sha256": self.previous_entry_sha256,
        }
        if include_digest:
            result["entry_sha256"] = self.entry_sha256
        return result


@dataclass(frozen=True, slots=True)
class G0FormationLedger:
    entries: tuple[G0FormationLedgerEntry, ...] = ()

    def __post_init__(self) -> None:
        _require(
            isinstance(self.entries, tuple)
            and all(type(row) is G0FormationLedgerEntry for row in self.entries),
            "G0 formation ledger entries must be exact immutable records",
        )

    @classmethod
    def empty(cls) -> Self:
        return cls()

    @property
    def head_sha256(self) -> str | None:
        return self.entries[-1].entry_sha256 if self.entries else None

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())

    def append(self, attempt: G0FormationAttempt) -> Self:
        _require(type(attempt) is G0FormationAttempt, "G0 ledger append requires an exact attempt")
        known_attempt_ids = {row.attempt.formation_attempt_id for row in self.entries}
        known_attempt_hashes = {row.attempt.attempt_sha256 for row in self.entries}
        _require(attempt.formation_attempt_id not in known_attempt_ids, "duplicate G0 formation attempt_id")
        _require(attempt.attempt_sha256 not in known_attempt_hashes, "duplicate G0 formation attempt")
        if attempt.parent_candidate_sha256 is not None:
            known_candidates = {
                row.attempt.candidate.bundle_sha256
                for row in self.entries
                if row.attempt.candidate is not None and row.attempt.status is not G0FormationStatus.REFUSED
            }
            _require(
                attempt.parent_candidate_sha256 in known_candidates,
                "G0 formation parent candidate is absent from the admitted prefix",
            )
        entry = G0FormationLedgerEntry.create(
            sequence=len(self.entries),
            attempt=attempt,
            previous_entry_sha256=self.head_sha256,
        )
        return type(self)(entries=(*self.entries, entry))

    def verify(self) -> tuple[str, ...]:
        problems: list[str] = []
        replay = type(self).empty()
        for index, entry in enumerate(self.entries):
            if entry.sequence != index:
                problems.append(f"entry-{index}:sequence-mismatch")
            if entry.previous_entry_sha256 != replay.head_sha256:
                problems.append(f"entry-{index}:previous-digest-mismatch")
            try:
                replay = replay.append(entry.attempt)
            except ValueError as exc:
                problems.append(f"entry-{index}:append-refused:{exc}")
            else:
                if replay.entries[-1] != entry:
                    problems.append(f"entry-{index}:replay-mismatch")
        return tuple(sorted(set(problems)))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": G0_FORMATION_LEDGER_SCHEMA,
            "entries": [row.payload() for row in self.entries],
            "head_sha256": self.head_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(payload, {"schema", "entries", "head_sha256"}, "G0 formation ledger")
        _require(payload["schema"] == G0_FORMATION_LEDGER_SCHEMA, "unsupported G0 ledger schema")
        rows = payload["entries"]
        _require(isinstance(rows, list), "G0 formation ledger entries must be a list")
        parsed = tuple(G0FormationLedgerEntry.from_payload(row) for row in rows)
        ledger = cls(entries=parsed)
        _require(payload["head_sha256"] == ledger.head_sha256, "G0 ledger head digest mismatch")
        if problems := ledger.verify():
            raise ValueError("invalid G0 formation ledger: " + "; ".join(problems))
        return ledger


class G0ParetoDecisionStatus(StrEnum):
    RETAINED = "retained"
    DOMINATED = "dominated"
    OBJECTIVE_TIE = "objective-tie"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class G0ParetoDecision:
    attempt_sha256: str
    candidate_sha256: str | None
    status: G0ParetoDecisionStatus
    dominating_attempt_sha256s: tuple[str, ...]
    schema: str = G0_PARETO_DECISION_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_PARETO_DECISION_SCHEMA, "unsupported G0 Pareto decision schema")
        _digest(self.attempt_sha256, "G0 Pareto attempt digest")
        if self.candidate_sha256 is not None:
            _digest(self.candidate_sha256, "G0 Pareto candidate digest")
        _require(isinstance(self.status, G0ParetoDecisionStatus), "G0 Pareto status must be typed")
        _require(
            self.dominating_attempt_sha256s == tuple(sorted(set(self.dominating_attempt_sha256s))),
            "G0 Pareto dominators must be unique and sorted",
        )
        for row in self.dominating_attempt_sha256s:
            _digest(row, "G0 Pareto dominator digest")
        if self.status is G0ParetoDecisionStatus.RETAINED:
            _require(not self.dominating_attempt_sha256s, "retained G0 candidate has a dominator")
        elif self.status in {G0ParetoDecisionStatus.DOMINATED, G0ParetoDecisionStatus.OBJECTIVE_TIE}:
            _require(bool(self.dominating_attempt_sha256s), "excluded G0 candidate lacks comparator")
        else:
            _require(
                not self.dominating_attempt_sha256s,
                "ineligible G0 candidate cannot claim a Pareto comparator",
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {"schema", "attempt_sha256", "candidate_sha256", "status", "dominating_attempt_sha256s"},
            "G0 Pareto decision",
        )
        rows = payload["dominating_attempt_sha256s"]
        _require(isinstance(rows, list), "G0 Pareto dominators must be a list")
        return cls(
            schema=payload["schema"],
            attempt_sha256=payload["attempt_sha256"],
            candidate_sha256=payload["candidate_sha256"],
            status=G0ParetoDecisionStatus(payload["status"]),
            dominating_attempt_sha256s=tuple(rows),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attempt_sha256": self.attempt_sha256,
            "candidate_sha256": self.candidate_sha256,
            "status": self.status.value,
            "dominating_attempt_sha256s": list(self.dominating_attempt_sha256s),
        }


@dataclass(frozen=True, slots=True)
class G0ParetoArchive:
    source_ledger_sha256: str
    decisions: tuple[G0ParetoDecision, ...]
    retained_attempt_sha256s: tuple[str, ...]
    archive_sha256: str
    schema: str = G0_PARETO_ARCHIVE_SCHEMA

    def __post_init__(self) -> None:
        _require(self.schema == G0_PARETO_ARCHIVE_SCHEMA, "unsupported G0 Pareto archive schema")
        _digest(self.source_ledger_sha256, "G0 Pareto source ledger digest")
        _require(
            isinstance(self.decisions, tuple)
            and all(type(row) is G0ParetoDecision for row in self.decisions),
            "G0 Pareto decisions must be exact immutable records",
        )
        attempts = tuple(row.attempt_sha256 for row in self.decisions)
        _require(len(attempts) == len(set(attempts)), "G0 Pareto decision attempt duplicated")
        expected = tuple(
            sorted(
                row.attempt_sha256 for row in self.decisions if row.status is G0ParetoDecisionStatus.RETAINED
            )
        )
        _require(self.retained_attempt_sha256s == expected, "G0 Pareto retained set mismatch")
        _digest(self.archive_sha256, "G0 Pareto archive digest")
        _require(
            self.archive_sha256 == canonical_sha256(self.payload(include_digest=False)),
            "G0 Pareto archive self-hash mismatch",
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        _exact(
            payload,
            {"schema", "source_ledger_sha256", "decisions", "retained_attempt_sha256s", "archive_sha256"},
            "G0 Pareto archive",
        )
        decisions, retained = payload["decisions"], payload["retained_attempt_sha256s"]
        _require(isinstance(decisions, list) and isinstance(retained, list), "G0 Pareto rows must be lists")
        return cls(
            schema=payload["schema"],
            source_ledger_sha256=payload["source_ledger_sha256"],
            decisions=tuple(G0ParetoDecision.from_payload(row) for row in decisions),
            retained_attempt_sha256s=tuple(retained),
            archive_sha256=payload["archive_sha256"],
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema": self.schema,
            "source_ledger_sha256": self.source_ledger_sha256,
            "decisions": [row.payload() for row in self.decisions],
            "retained_attempt_sha256s": list(self.retained_attempt_sha256s),
        }
        if include_digest:
            result["archive_sha256"] = self.archive_sha256
        return result


def build_g0_pareto_archive(ledger: G0FormationLedger) -> G0ParetoArchive:

    _require(type(ledger) is G0FormationLedger, "G0 Pareto source must be an exact ledger")
    _require(not ledger.verify(), "G0 Pareto source ledger does not replay")
    eligible = [
        row.attempt
        for row in ledger.entries
        if row.attempt.status is G0FormationStatus.EVALUATED
        and row.attempt.objective is not None
        and row.attempt.candidate is not None
    ]
    by_objective: dict[str, list[G0FormationAttempt]] = {}
    for attempt in eligible:
        assert attempt.objective is not None
        by_objective.setdefault(attempt.objective.objective_sha256, []).append(attempt)
    tie_winner: dict[str, str] = {
        objective_sha: min(
            rows,
            key=lambda row: (row.candidate.bundle_sha256 if row.candidate else "", row.attempt_sha256),
        ).attempt_sha256
        for objective_sha, rows in by_objective.items()
    }
    representatives = [
        row
        for row in eligible
        if row.objective is not None and tie_winner[row.objective.objective_sha256] == row.attempt_sha256
    ]
    decisions: list[G0ParetoDecision] = []
    for entry in ledger.entries:
        attempt = entry.attempt
        candidate_sha = attempt.candidate.bundle_sha256 if attempt.candidate else None
        if attempt not in eligible:
            status = G0ParetoDecisionStatus.INELIGIBLE
            dominators: tuple[str, ...] = ()
        else:
            assert attempt.objective is not None
            canonical = tie_winner[attempt.objective.objective_sha256]
            if canonical != attempt.attempt_sha256:
                status = G0ParetoDecisionStatus.OBJECTIVE_TIE
                dominators = (canonical,)
            else:
                dominating = tuple(
                    sorted(
                        other.attempt_sha256
                        for other in representatives
                        if other.attempt_sha256 != attempt.attempt_sha256
                        and other.objective is not None
                        and g0_objective_dominates(other.objective, attempt.objective)
                    )
                )
                status = G0ParetoDecisionStatus.DOMINATED if dominating else G0ParetoDecisionStatus.RETAINED
                dominators = dominating
        decisions.append(
            G0ParetoDecision(
                attempt_sha256=attempt.attempt_sha256,
                candidate_sha256=candidate_sha,
                status=status,
                dominating_attempt_sha256s=dominators,
            )
        )
    retained = tuple(
        sorted(row.attempt_sha256 for row in decisions if row.status is G0ParetoDecisionStatus.RETAINED)
    )
    core = {
        "schema": G0_PARETO_ARCHIVE_SCHEMA,
        "source_ledger_sha256": ledger.sha256,
        "decisions": [row.payload() for row in decisions],
        "retained_attempt_sha256s": list(retained),
    }
    return G0ParetoArchive(
        source_ledger_sha256=ledger.sha256,
        decisions=tuple(decisions),
        retained_attempt_sha256s=retained,
        archive_sha256=canonical_sha256(core),
    )


def verify_g0_pareto_archive(
    archive: G0ParetoArchive,
    ledger: G0FormationLedger,
) -> bool:

    _require(type(archive) is G0ParetoArchive, "G0 Pareto verifier requires an exact archive")
    _require(type(ledger) is G0FormationLedger, "G0 Pareto verifier requires an exact ledger")
    replayed = build_g0_pareto_archive(ledger)
    _require(replayed == archive, "G0 Pareto archive replay mismatch")
    return True


__all__ = [
    "G0_CANDIDATE_BUNDLE_SCHEMA",
    "G0_FORMATION_ATTEMPT_SCHEMA",
    "G0_FORMATION_COSTS_SCHEMA",
    "G0_FORMATION_LEDGER_ENTRY_SCHEMA",
    "G0_FORMATION_LEDGER_SCHEMA",
    "G0_OBJECTIVE_VECTOR_SCHEMA",
    "G0_PARETO_ARCHIVE_SCHEMA",
    "G0_PARETO_DECISION_SCHEMA",
    "G0_TRACE_ASSESSMENT_SCHEMA",
    "G0CandidateBundle",
    "G0CandidateOrigin",
    "G0FormationAttempt",
    "G0FormationCosts",
    "G0FormationLedger",
    "G0FormationLedgerEntry",
    "G0FormationStatus",
    "G0ObjectiveVector",
    "G0ParetoArchive",
    "G0ParetoDecision",
    "G0ParetoDecisionStatus",
    "G0TraceAssessment",
    "build_g0_pareto_archive",
    "g0_objective_dominates",
    "verify_g0_formation_attempt",
    "verify_g0_pareto_archive",
]
