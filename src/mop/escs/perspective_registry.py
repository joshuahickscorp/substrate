
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Self

from mop.substrate.events import canonical_sha256

PERSPECTIVE_CANDIDATE_REGISTRY_SCHEMA = "mop-escs-perspective-candidates/v1"

_CANDIDATE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_EVIDENCE_ROOTS = frozenset({"docs", "proof", "runs"})


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_false(value: object, label: str) -> None:
    if not isinstance(value, bool) or value:
        raise ValueError(f"{label} must be the boolean false")


class PerspectiveFacet(StrEnum):

    ABSTRACTION = "abstraction"
    ADAPTIVE_ACTIVATION = "adaptive_activation"
    CAUSAL_REASONING = "causal_reasoning"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    CONTRADICTION_DETECTION = "contradiction_detection"
    CURIOSITY = "curiosity"
    DECOMPOSITION = "decomposition"
    EPISODIC_MEMORY = "episodic_memory"
    EVENT_DRIVEN_COGNITION = "event_driven_cognition"
    IMAGINATION = "imagination"
    INTUITION = "intuition"
    MOTOR_REASONING = "motor_reasoning"
    NOVELTY_DETECTION = "novelty_detection"
    PERSPECTIVE_ARBITRATION = "perspective_arbitration"
    PERSPECTIVE_COMMUNICATION = "perspective_communication"
    PERSPECTIVE_COMPETITION = "perspective_competition"
    PERSPECTIVE_COOPERATION = "perspective_cooperation"
    PLANNING = "planning"
    PROBABILISTIC_REASONING = "probabilistic_reasoning"
    REFLECTION = "reflection"
    RETRIEVAL = "retrieval"
    SELECTIVE_COMPUTATION = "selective_computation"
    SELF_CRITIQUE = "self_critique"
    SEMANTIC_MEMORY = "semantic_memory"
    SIMULATION = "simulation"
    SPATIAL_REASONING = "spatial_reasoning"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"
    UNCERTAINTY_ESTIMATION = "uncertainty_estimation"
    VERIFICATION = "verification"
    VISUAL_REASONING = "visual_reasoning"


class EvidenceStanding(StrEnum):
    MECHANICS = "mechanics"
    TOY_POSITIVE = "toy-positive"
    NULL = "null"
    FAILED = "failed"
    PENDING = "pending"
    BLOCKED = "blocked"
    UNTESTED = "untested"


class IntegrationDisposition(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    FEATURE_FLAGGED = "feature-flagged"
    CONTROL_ONLY = "control-only"
    SANDBOX_STUB = "sandbox-stub"
    EXCLUDED = "excluded"


class CandidateInterface(StrEnum):
    RAW_EVENT_FORMER = "raw-event-former"
    ACTOR_CLAIMS = "actor-claims"
    ENDOGENOUS_HYPOTHESIS = "endogenous-hypothesis"
    EVENT_MEMORY = "event-memory"
    CHASSIS_ACTION = "chassis-action"
    CONTROL_ARM = "control-arm"
    SANDBOX = "sandbox"


class EffectBoundary(StrEnum):
    NONE = "none"
    CLAIMS_ONLY = "claims-only"
    COUNTERFACTUAL_ONLY = "counterfactual-only"
    CHASSIS_COMMITMENT = "chassis-commitment"


class TriggerAuthority(StrEnum):

    NONE = "none"
    EXTERNAL_EVENT = "external-event"
    EVENT_AUTHORIZED = "event-authorized"
    DECISION_VALUE_GATED = "decision-value-gated"


class ResourceTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very-high"


class WorkDimension(StrEnum):
    RAW_TRANSPORT_AND_ADAPTERS = "raw_transport_and_adapters"
    EVENT_FORMATION = "event_formation"
    INDEXING_AND_GRAPH_MAINTENANCE = "indexing_and_graph_maintenance"
    DISPATCH_AND_EXPLORATION = "dispatch_and_exploration"
    ACTOR_EXECUTION = "actor_execution"
    MESSAGES = "messages"
    COUNTERFACTUAL_CREDIT = "counterfactual_credit"
    LEARNING = "learning"
    ARCHIVAL_AND_ERASURE = "archival_and_erasure"
    RETAINED_BYTE_TIME = "retained_byte_time"
    IDLE_FLOOR = "idle_floor"


class PerspectiveGuard(StrEnum):
    ACTION_SHUFFLE = "action-shuffle"
    ATOMIC_ROLLBACK = "atomic-rollback"
    BOUNDED_ACTIVATION = "bounded-activation"
    CHASSIS_COMMITMENT = "chassis-commitment"
    CONTENT_DIGEST = "content-digest"
    COUNTERFACTUAL_BRANCH = "counterfactual-branch"
    DECISION_VALUE = "decision-value"
    EVIDENCE_TAINT = "evidence-taint"
    EXTERNAL_CONSEQUENCE = "external-consequence"
    LIFECYCLE_ACCOUNTING = "lifecycle-accounting"
    MATCHED_COMPUTE = "matched-compute"
    NOISY_TV = "noisy-tv"
    NO_FACTUAL_EFFECT = "no-factual-effect"
    PARENT_AUTHORITY = "parent-authority"
    QUIESCENCE = "quiescence"
    RANDOM_INIT_CONTROL = "random-init-control"
    REDUCIBLE_ERROR = "reducible-error"
    RETENTION_CAP = "retention-cap"
    SCHEMA_VERSION = "schema-version"
    SHUFFLED_CONTROL = "shuffled-control"
    SNAPSHOT_ISOLATION = "snapshot-isolation"
    SOURCE_PROVENANCE = "source-provenance"


@dataclass(frozen=True, slots=True)
class EvidenceReference:

    artifact_path: str
    locator: str
    scope: str

    def __post_init__(self) -> None:
        path_text = _require_text(self.artifact_path, "evidence artifact_path")
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("evidence artifact_path must be a repository-relative path")
        if path.parts[0] not in _EVIDENCE_ROOTS:
            raise ValueError("evidence artifact_path must live under docs/, proof/, or runs/")
        _require_text(self.locator, "evidence locator")
        _require_text(self.scope, "evidence scope")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("evidence reference must be a mapping")
        _require_exact_keys(payload, {"artifact_path", "locator", "scope"}, "evidence reference")
        return cls(
            artifact_path=payload["artifact_path"],
            locator=payload["locator"],
            scope=payload["scope"],
        )

    def payload(self) -> dict[str, str]:
        return {
            "artifact_path": self.artifact_path,
            "locator": self.locator,
            "scope": self.scope,
        }


_DISPOSITION_BY_STANDING = {
    EvidenceStanding.MECHANICS: IntegrationDisposition.INFRASTRUCTURE,
    EvidenceStanding.TOY_POSITIVE: IntegrationDisposition.FEATURE_FLAGGED,
    EvidenceStanding.NULL: IntegrationDisposition.CONTROL_ONLY,
    EvidenceStanding.FAILED: IntegrationDisposition.EXCLUDED,
    EvidenceStanding.PENDING: IntegrationDisposition.SANDBOX_STUB,
    EvidenceStanding.BLOCKED: IntegrationDisposition.SANDBOX_STUB,
    EvidenceStanding.UNTESTED: IntegrationDisposition.SANDBOX_STUB,
}

_INDEPENDENT_TRIGGER_FORBIDDEN = frozenset(
    {
        PerspectiveFacet.UNCERTAINTY_ESTIMATION,
        PerspectiveFacet.NOVELTY_DETECTION,
        PerspectiveFacet.CURIOSITY,
    }
)

_COUNTERFACTUAL_ONLY = frozenset(
    {
        PerspectiveFacet.IMAGINATION,
        PerspectiveFacet.SIMULATION,
    }
)


@dataclass(frozen=True, slots=True)
class PerspectiveCandidate:

    candidate_id: str
    facet: PerspectiveFacet
    label: str
    evidence_standing: EvidenceStanding
    integration_disposition: IntegrationDisposition
    interface: CandidateInterface
    effect_boundary: EffectBoundary
    trigger_authority: TriggerAuthority
    resource_tier: ResourceTier
    work_dimensions: tuple[WorkDimension, ...]
    required_guards: tuple[PerspectiveGuard, ...]
    evidence_refs: tuple[EvidenceReference, ...]
    claim_scope: str
    activation_enabled: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or _CANDIDATE_ID_RE.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must be canonical snake_case")
        _require_text(self.label, "perspective label")
        _require_text(self.claim_scope, "perspective claim_scope")
        for value, expected, label in (
            (self.facet, PerspectiveFacet, "facet"),
            (self.evidence_standing, EvidenceStanding, "evidence_standing"),
            (self.integration_disposition, IntegrationDisposition, "integration_disposition"),
            (self.interface, CandidateInterface, "interface"),
            (self.effect_boundary, EffectBoundary, "effect_boundary"),
            (self.trigger_authority, TriggerAuthority, "trigger_authority"),
            (self.resource_tier, ResourceTier, "resource_tier"),
        ):
            if not isinstance(value, expected):
                raise ValueError(f"perspective {label} must be typed")
        self._validate_typed_tuple(self.work_dimensions, WorkDimension, "work_dimensions")
        self._validate_typed_tuple(self.required_guards, PerspectiveGuard, "required_guards")
        self._validate_typed_tuple(self.evidence_refs, EvidenceReference, "evidence_refs")
        if not self.work_dimensions:
            raise ValueError("perspective candidate requires at least one work dimension")
        if not self.required_guards:
            raise ValueError("perspective candidate requires at least one guard")
        if not self.evidence_refs:
            raise ValueError("perspective candidate requires at least one exact evidence reference")
        _require_false(self.activation_enabled, "perspective activation_enabled")
        _require_false(self.scientific_promotion_allowed, "perspective scientific_promotion_allowed")

        expected_disposition = _DISPOSITION_BY_STANDING[self.evidence_standing]
        if self.integration_disposition is not expected_disposition:
            raise ValueError(
                f"{self.evidence_standing.value} evidence requires {expected_disposition.value} integration"
            )
        if self.integration_disposition is IntegrationDisposition.EXCLUDED and (
            self.effect_boundary is not EffectBoundary.NONE
            or self.trigger_authority is not TriggerAuthority.NONE
        ):
            raise ValueError("excluded candidates cannot trigger work or cross an effect boundary")
        if (
            self.integration_disposition
            in {
                IntegrationDisposition.CONTROL_ONLY,
                IntegrationDisposition.SANDBOX_STUB,
            }
            and self.effect_boundary is EffectBoundary.CHASSIS_COMMITMENT
        ):
            raise ValueError("controls and inert stubs cannot authorize factual effects")

        guard_set = set(self.required_guards)
        if self.facet in _INDEPENDENT_TRIGGER_FORBIDDEN:
            if self.trigger_authority is not TriggerAuthority.NONE:
                raise ValueError("uncertainty, novelty, and curiosity cannot independently trigger ESCS work")
            required = {
                PerspectiveGuard.NOISY_TV,
                PerspectiveGuard.REDUCIBLE_ERROR,
                PerspectiveGuard.DECISION_VALUE,
            }
            if not required <= guard_set:
                raise ValueError(
                    "uncertainty, novelty, and curiosity require noisy-TV, reducible-error, "
                    "and decision-value guards"
                )
        if self.facet in _COUNTERFACTUAL_ONLY:
            required = {
                PerspectiveGuard.BOUNDED_ACTIVATION,
                PerspectiveGuard.COUNTERFACTUAL_BRANCH,
                PerspectiveGuard.NO_FACTUAL_EFFECT,
                PerspectiveGuard.QUIESCENCE,
            }
            if self.effect_boundary is not EffectBoundary.COUNTERFACTUAL_ONLY:
                raise ValueError("imagination and simulation must remain counterfactual-only")
            if self.interface is not CandidateInterface.ENDOGENOUS_HYPOTHESIS:
                raise ValueError("imagination and simulation require the endogenous-hypothesis interface")
            if not required <= guard_set:
                raise ValueError("counterfactual candidates lack required isolation guards")
        if self.effect_boundary is EffectBoundary.CHASSIS_COMMITMENT:
            required = {
                PerspectiveGuard.ATOMIC_ROLLBACK,
                PerspectiveGuard.CHASSIS_COMMITMENT,
                PerspectiveGuard.EXTERNAL_CONSEQUENCE,
            }
            if self.interface is not CandidateInterface.CHASSIS_ACTION:
                raise ValueError("factual effects require the chassis-action interface")
            if not required <= guard_set:
                raise ValueError("factual effects lack chassis commitment and rollback guards")

    @staticmethod
    def _validate_typed_tuple(values: object, item_type: type[object], label: str) -> None:
        if not isinstance(values, tuple):
            raise ValueError(f"{label} must be an immutable tuple")
        if not all(isinstance(value, item_type) for value in values):
            raise ValueError(f"{label} contains an untyped value")
        if len(values) != len(set(values)):
            raise ValueError(f"{label} must contain unique values")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("perspective candidate must be a mapping")
        expected = {
            "candidate_id",
            "facet",
            "label",
            "evidence_standing",
            "integration_disposition",
            "interface",
            "effect_boundary",
            "trigger_authority",
            "resource_tier",
            "work_dimensions",
            "required_guards",
            "evidence_refs",
            "claim_scope",
            "activation_enabled",
            "scientific_promotion_allowed",
        }
        _require_exact_keys(payload, expected, "perspective candidate")
        for name in ("work_dimensions", "required_guards", "evidence_refs"):
            if not isinstance(payload[name], list):
                raise ValueError(f"perspective candidate {name} must be a list")
        return cls(
            candidate_id=payload["candidate_id"],
            facet=PerspectiveFacet(payload["facet"]),
            label=payload["label"],
            evidence_standing=EvidenceStanding(payload["evidence_standing"]),
            integration_disposition=IntegrationDisposition(payload["integration_disposition"]),
            interface=CandidateInterface(payload["interface"]),
            effect_boundary=EffectBoundary(payload["effect_boundary"]),
            trigger_authority=TriggerAuthority(payload["trigger_authority"]),
            resource_tier=ResourceTier(payload["resource_tier"]),
            work_dimensions=tuple(WorkDimension(value) for value in payload["work_dimensions"]),
            required_guards=tuple(PerspectiveGuard(value) for value in payload["required_guards"]),
            evidence_refs=tuple(EvidenceReference.from_payload(value) for value in payload["evidence_refs"]),
            claim_scope=payload["claim_scope"],
            activation_enabled=payload["activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "facet": self.facet.value,
            "label": self.label,
            "evidence_standing": self.evidence_standing.value,
            "integration_disposition": self.integration_disposition.value,
            "interface": self.interface.value,
            "effect_boundary": self.effect_boundary.value,
            "trigger_authority": self.trigger_authority.value,
            "resource_tier": self.resource_tier.value,
            "work_dimensions": [value.value for value in self.work_dimensions],
            "required_guards": [value.value for value in self.required_guards],
            "evidence_refs": [value.payload() for value in self.evidence_refs],
            "claim_scope": self.claim_scope,
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }


@dataclass(frozen=True, slots=True)
class PerspectiveCandidateRegistry:

    candidates: tuple[PerspectiveCandidate, ...]
    default_activation_enabled: bool = False
    scientific_promotion_allowed: bool = False
    schema: str = PERSPECTIVE_CANDIDATE_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERSPECTIVE_CANDIDATE_REGISTRY_SCHEMA:
            raise ValueError("unsupported perspective candidate registry schema")
        _require_false(self.default_activation_enabled, "registry default_activation_enabled")
        _require_false(self.scientific_promotion_allowed, "registry scientific_promotion_allowed")
        if not isinstance(self.candidates, tuple):
            raise ValueError("registry candidates must be an immutable tuple")
        if not all(type(candidate) is PerspectiveCandidate for candidate in self.candidates):
            raise ValueError("registry candidates must be exact PerspectiveCandidate records")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("registry candidate IDs must be unique")
        facets = tuple(candidate.facet for candidate in self.candidates)
        if len(facets) != len(set(facets)):
            raise ValueError("registry facets must be unique")
        missing_facets = set(PerspectiveFacet) - set(facets)
        if missing_facets:
            raise ValueError(
                "registry omits requested facets: "
                + ", ".join(sorted(facet.value for facet in missing_facets))
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("perspective registry payload must be a mapping")
        expected = {
            "schema",
            "default_activation_enabled",
            "scientific_promotion_allowed",
            "candidates",
        }
        _require_exact_keys(payload, expected, "perspective registry")
        rows = payload["candidates"]
        if not isinstance(rows, list):
            raise ValueError("perspective registry candidates must be a list")
        return cls(
            schema=payload["schema"],
            default_activation_enabled=payload["default_activation_enabled"],
            scientific_promotion_allowed=payload["scientific_promotion_allowed"],
            candidates=tuple(PerspectiveCandidate.from_payload(row) for row in rows),
        )

    def candidate_for(self, facet: PerspectiveFacet) -> PerspectiveCandidate:
        if not isinstance(facet, PerspectiveFacet):
            raise ValueError("candidate lookup facet must be typed")
        for candidate in self.candidates:
            if candidate.facet is facet:
                return candidate
        raise ValueError(f"registry has no candidate for {facet.value}")

    def missing_evidence_paths(self, repository_root: str | Path) -> tuple[str, ...]:
        root = Path(repository_root)
        missing = {
            evidence.artifact_path
            for candidate in self.candidates
            for evidence in candidate.evidence_refs
            if not (root / evidence.artifact_path).is_file()
        }
        return tuple(sorted(missing))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "default_activation_enabled": self.default_activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
            "candidates": [candidate.payload() for candidate in self.candidates],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


def load_perspective_candidate_registry(path: str | Path) -> PerspectiveCandidateRegistry:

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("perspective registry document must contain a JSON object")
    return PerspectiveCandidateRegistry.from_payload(payload)


__all__ = [
    "PERSPECTIVE_CANDIDATE_REGISTRY_SCHEMA",
    "CandidateInterface",
    "EffectBoundary",
    "EvidenceReference",
    "EvidenceStanding",
    "IntegrationDisposition",
    "PerspectiveCandidate",
    "PerspectiveCandidateRegistry",
    "PerspectiveFacet",
    "PerspectiveGuard",
    "ResourceTier",
    "TriggerAuthority",
    "WorkDimension",
    "load_perspective_candidate_registry",
]
