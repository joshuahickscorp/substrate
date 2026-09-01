"""Deterministic, model-neutral computation fabric for Substrate v5.

The objects in this module are deliberately small model-equivalent modules.  They
are useful for contract, routing, support, replacement, and instrumentation tests
without downloading a checkpoint or calling a service.  A real model adapter can
implement the same ``invoke`` contract.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, Protocol, cast

from substrate import v5io as io


class ModelContractError(ValueError):
    """A model request or declaration violates the public fabric contract."""


class ModelRole(StrEnum):
    PRIMARY_PERFORMER = "primary_performer"
    INDEPENDENT_PERFORMER = "independent_performer"
    DRAFT_GENERATOR = "draft_generator"
    VERIFIER = "verifier"
    CRITIC = "critic"
    SIMULATOR = "simulator"
    PLANNER = "planner"
    TEACHER = "teacher"
    STUDENT = "student"
    COMPRESSOR = "compressor"
    RETRIEVER = "retriever"
    ROUTER = "router"
    SPECIALIST = "specialist"
    FALLBACK = "fallback"
    MONITOR = "monitor"
    ARBITER = "arbiter"
    TRANSLATOR = "translator"
    REPRESENTATION_ALIGNER = "representation_aligner"


ALL_MODALITIES = (
    "text",
    "image",
    "video",
    "motion",
    "audio",
    "speech",
    "depth_3d",
    "body_tool",
)


@dataclass(frozen=True)
class ModelContract:
    """Auditable capabilities and operating envelope of one callable module."""

    identity: str
    checkpoint_identity: str
    version: str
    license: str
    runtime: str
    hardware_requirements: tuple[str, ...]
    modalities_accepted: tuple[str, ...]
    modalities_produced: tuple[str, ...]
    input_schema: str
    output_schema: str
    hidden_state_policy: str
    cost: float
    latency_ms: float
    memory_mb: float
    confidence_semantics: str
    calibrated_confidence: float
    training_provenance: str
    known_limitations: tuple[str, ...]
    allowed_roles: tuple[ModelRole, ...]
    stateful: bool = False
    checkpoint_support: bool = False
    batching_support: bool = True
    streaming_support: bool = False

    def __post_init__(self) -> None:
        if not self.identity or not self.checkpoint_identity or not self.version:
            raise ModelContractError("model, checkpoint, and version identities are required")
        if not self.modalities_accepted or not self.modalities_produced:
            raise ModelContractError("accepted and produced modalities must be declared")
        if ModelRole.INDEPENDENT_PERFORMER not in self.allowed_roles:
            raise ModelContractError("every v5 module must remain independently callable")
        if self.cost < 0.0 or self.latency_ms < 0.0 or self.memory_mb < 0.0:
            raise ModelContractError("cost, latency, and memory must be non-negative")
        if not 0.0 <= self.calibrated_confidence <= 1.0:
            raise ModelContractError("calibrated confidence must be in [0, 1]")


_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "answer",
        "answer_id",
        "oracle",
        "oracle_id",
        "outcome",
        "physical_id",
        "private_target",
        "target",
        "target_id",
        "truth",
        "truth_id",
    }
)
_REQUEST_SCALAR_TYPES = (bool, int, float, str, bytes, bytearray)


def _forbidden_request_keys(value: object) -> set[str]:
    """Return only outcome-authority keys found in a JSON-shaped payload."""

    # ModelRequest construction is on the episode hot path.  The normal
    # requests are exact flat dicts containing only scalar public cues; those
    # cannot contain a nested authority key, so avoid the recursive walker.
    # Any non-scalar, non-string-key, or forbidden-key shape still takes the
    # complete defensive path below.
    if type(value) is dict:
        for key, child in cast(dict[object, object], value).items():
            if (
                type(key) is not str
                or key.lower() in _FORBIDDEN_REQUEST_KEYS
                or (child is not None and type(child) not in _REQUEST_SCALAR_TYPES)
            ):
                break
        else:
            return set()
    leaked: set[str] = set()
    _collect_forbidden_request_keys(value, leaked)
    return leaked


def _collect_forbidden_request_keys(value: object, leaked: set[str]) -> None:
    value_type = type(value)
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        for key, child in mapping.items():
            normalized = key.lower() if type(key) is str else str(key).lower()
            if normalized in _FORBIDDEN_REQUEST_KEYS:
                leaked.add(normalized)
            child_type = type(child)
            if child_type is dict or child_type is list or child_type is tuple or (
                child is not None
                and child_type not in (bool, int, float, str, bytes, bytearray)
                and isinstance(child, (Mapping, list, tuple))
            ):
                _collect_forbidden_request_keys(child, leaked)
        return
    if value_type is list or value_type is tuple:
        sequence = cast(list[object] | tuple[object, ...], value)
        for child in sequence:
            child_type = type(child)
            if child_type is dict or child_type is list or child_type is tuple or (
                child is not None
                and child_type not in (bool, int, float, str, bytes, bytearray)
                and isinstance(child, (Mapping, list, tuple))
            ):
                _collect_forbidden_request_keys(child, leaked)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_REQUEST_KEYS:
                leaked.add(normalized)
            if isinstance(child, (Mapping, list, tuple)):
                _collect_forbidden_request_keys(child, leaked)
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                _collect_forbidden_request_keys(child, leaked)


@dataclass(frozen=True)
class ModelRequest:
    task_id: str
    operation: str
    modality: str
    payload: Mapping[str, Any]
    role: ModelRole = ModelRole.INDEPENDENT_PERFORMER
    minimum_confidence: float = 0.0
    maximum_cost: float = math.inf
    maximum_latency_ms: float = math.inf
    privacy: str = "local_non_sensitive"

    def __post_init__(self) -> None:
        if not self.task_id or not self.operation:
            raise ModelContractError("task and operation identities are required")
        leaked = _forbidden_request_keys(self.payload)
        if leaked:
            raise ModelContractError(f"outcome authority leaked into model request: {sorted(leaked)}")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ModelContractError("minimum confidence must be in [0, 1]")
        if self.maximum_cost < 0.0 or self.maximum_latency_ms < 0.0:
            raise ModelContractError("request budgets must be non-negative")


@dataclass(frozen=True)
class ModelOutput:
    model_identity: str
    checkpoint_identity: str
    task_id: str
    operation: str
    value: Any
    confidence: float
    uncertainty: float
    cost: float
    latency_ms: float
    evidence: tuple[str, ...]
    independently_callable: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.uncertainty <= 1.0:
            raise ModelContractError("output confidence and uncertainty must be in [0, 1]")


class CallableModel(Protocol):
    contract: ModelContract

    def invoke(self, request: ModelRequest) -> ModelOutput:
        """Execute one independently measurable request."""


Evaluator = Callable[[ModelRequest, ModelContract], tuple[Any, float, tuple[str, ...]]]


@dataclass
class DeterministicModelModule:
    """A deterministic local implementation of the callable-model protocol."""

    contract: ModelContract
    evaluator: Evaluator
    call_count: int = 0

    def invoke(self, request: ModelRequest) -> ModelOutput:
        if request.modality not in self.contract.modalities_accepted:
            raise ModelContractError(
                f"{self.contract.identity} does not accept modality {request.modality!r}",
            )
        if request.role not in self.contract.allowed_roles and request.role != ModelRole.INDEPENDENT_PERFORMER:
            raise ModelContractError(f"{self.contract.identity} does not allow role {request.role.value!r}")
        if self.contract.cost > request.maximum_cost or self.contract.latency_ms > request.maximum_latency_ms:
            raise ModelContractError("request budget cannot admit this model")
        value, confidence, evidence = self.evaluator(request, self.contract)
        confidence = max(0.0, min(1.0, float(confidence)))
        self.call_count += 1
        return ModelOutput(
            model_identity=self.contract.identity,
            checkpoint_identity=self.contract.checkpoint_identity,
            task_id=request.task_id,
            operation=request.operation,
            value=value,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            cost=self.contract.cost,
            latency_ms=self.contract.latency_ms,
            evidence=evidence,
        )

    __call__ = invoke


@dataclass(frozen=True)
class SupportRelationship:
    source_model: str
    target_model: str
    relationship: str
    trigger: str
    measured_quantity: str
    source_remains_independent: bool = True
    target_remains_independent: bool = True


@dataclass(frozen=True)
class RoutingDecision:
    task_id: str
    selected_model: str
    eligible_models: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    inputs_used: tuple[str, ...]
    outcome_information_used: bool = False


_ROUTE_CACHE_MAX_ENTRIES = 128


class ModelRegistry:
    """Registry, relationship graph, and outcome-blind deterministic router."""

    def __init__(self) -> None:
        self._models: dict[str, CallableModel] = {}
        self._relationships: list[SupportRelationship] = []
        self._route_cache: dict[
            tuple[str, ModelRole, float, float, float],
            tuple[str, tuple[str, ...], tuple[tuple[str, float], ...]],
        ] = {}

    @property
    def contracts(self) -> tuple[ModelContract, ...]:
        return tuple(self._models[name].contract for name in sorted(self._models))

    @property
    def relationships(self) -> tuple[SupportRelationship, ...]:
        return tuple(self._relationships)

    def register(self, model: CallableModel) -> None:
        identity = model.contract.identity
        if identity in self._models:
            raise ModelContractError(f"duplicate model identity {identity!r}")
        self._models[identity] = model
        self._route_cache.clear()

    def connect(self, relationship: SupportRelationship) -> None:
        if relationship.source_model not in self._models or relationship.target_model not in self._models:
            raise ModelContractError("support relationship endpoints must be registered")
        if not relationship.source_remains_independent or not relationship.target_remains_independent:
            raise ModelContractError("support may not remove independent callable status")
        self._relationships.append(relationship)

    def invoke(self, identity: str, request: ModelRequest) -> ModelOutput:
        try:
            model = self._models[identity]
        except KeyError as exc:
            raise ModelContractError(f"unknown model {identity!r}") from exc
        return model.invoke(request)

    def route(self, request: ModelRequest) -> RoutingDecision:
        cache_key = (
            request.modality,
            request.role,
            request.minimum_confidence,
            request.maximum_cost,
            request.maximum_latency_ms,
        )
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            selected_model, eligible_models, scores = cached
            return RoutingDecision(
                task_id=request.task_id,
                selected_model=selected_model,
                eligible_models=eligible_models,
                scores=scores,
                inputs_used=("modality", "role", "minimum_confidence", "maximum_cost", "maximum_latency_ms"),
            )

        eligible = []
        for model in self._models.values():
            contract = model.contract
            if request.modality not in contract.modalities_accepted:
                continue
            if request.role not in contract.allowed_roles and request.role != ModelRole.INDEPENDENT_PERFORMER:
                continue
            if contract.cost > request.maximum_cost or contract.latency_ms > request.maximum_latency_ms:
                continue
            if contract.calibrated_confidence < request.minimum_confidence:
                continue
            # Specific competence dominates broad nominal support; cost and latency
            # then select among sufficiently capable organs.
            modality_specificity = 0.50 / len(contract.modalities_accepted)
            score = (
                0.60 * contract.calibrated_confidence
                + modality_specificity
                - 0.30 * contract.cost
                - 0.0002 * contract.latency_ms
            )
            eligible.append((contract.identity, score))
        if not eligible:
            raise ModelContractError("no model satisfies modality, role, confidence, and budget")
        ranked = sorted(eligible, key=lambda row: (-row[1], row[0]))
        eligible_models = tuple(sorted(identity for identity, _ in eligible))
        scores = tuple(ranked)
        if len(self._route_cache) >= _ROUTE_CACHE_MAX_ENTRIES:
            self._route_cache.clear()
        self._route_cache[cache_key] = (ranked[0][0], eligible_models, scores)
        return RoutingDecision(
            task_id=request.task_id,
            selected_model=ranked[0][0],
            eligible_models=eligible_models,
            scores=scores,
            inputs_used=("modality", "role", "minimum_confidence", "maximum_cost", "maximum_latency_ms"),
        )

    def execute_routed(self, request: ModelRequest) -> tuple[RoutingDecision, ModelOutput]:
        decision = self.route(request)
        return decision, self.invoke(decision.selected_model, request)


def _stable_digest(value: object) -> str:
    return hashlib.sha256(io.stable_json(value)).hexdigest()


def _default_evaluator(request: ModelRequest, contract: ModelContract) -> tuple[Any, float, tuple[str, ...]]:
    payload = request.payload
    if request.operation == "binary_draft":
        signal = float(payload["coarse_signal"])
        value = "positive" if signal >= 0.0 else "negative"
        return value, min(0.95, 0.45 + abs(signal) / 2.0), ("coarse_signal",)
    if request.operation == "binary_verify":
        signal = float(payload["fine_signal"])
        value = "positive" if signal >= 0.0 else "negative"
        return value, min(0.99, 0.80 + abs(signal) / 5.0), ("fine_signal",)
    if request.operation == "modality_classify":
        signal = float(payload["observable_cue"])
        value = "present" if signal >= 0.0 else "absent"
        return value, contract.calibrated_confidence, ("observable_cue", request.modality)
    if request.operation == "verify_candidate":
        candidate = payload.get("candidate")
        evidence_sign = float(payload.get("evidence_signal", 0.0))
        expected = "positive" if evidence_sign >= 0.0 else "negative"
        return candidate == expected, contract.calibrated_confidence, ("candidate", "evidence_signal")
    if request.operation in {"represent", "extract", "independent"}:
        digest = _stable_digest({"operation": request.operation, "modality": request.modality, "payload": payload})
        return {"representation": digest[:20], "modality": request.modality}, contract.calibrated_confidence, ("public_payload",)
    digest = _stable_digest({"operation": request.operation, "payload": payload, "model": contract.identity})
    return {"result": digest[:20]}, contract.calibrated_confidence, ("public_payload",)


def _contract(
    identity: str,
    modalities: tuple[str, ...],
    roles: tuple[ModelRole, ...],
    *,
    cost: float,
    latency_ms: float,
    confidence: float,
    output: str,
) -> ModelContract:
    allowed = tuple(dict.fromkeys((ModelRole.INDEPENDENT_PERFORMER, *roles)))
    return ModelContract(
        identity=identity,
        checkpoint_identity=f"builtin-sha256:{_stable_digest(identity)[:24]}",
        version="v5.0.0",
        license="project-local-deterministic-fixture",
        runtime="python-deterministic",
        hardware_requirements=("cpu",),
        modalities_accepted=modalities,
        modalities_produced=(output,),
        input_schema="substrate.v5.model-request/1",
        output_schema="substrate.v5.model-output/1",
        hidden_state_policy="none",
        cost=cost,
        latency_ms=latency_ms,
        memory_mb=1.0,
        confidence_semantics="calibrated fixture probability; uncertainty = 1-confidence",
        calibrated_confidence=confidence,
        training_provenance="hand-specified deterministic scientific fixture; no training data",
        known_limitations=("bounded synthetic operations only",),
        allowed_roles=allowed,
    )


_DEFAULT_MODULE_SPECS = (
    ("language_interpreter", ("text", "speech"), (ModelRole.SPECIALIST, ModelRole.TRANSLATOR), 0.12, 2.0, 0.88, "text"),
    ("image_object_detector", ("image", "video"), (ModelRole.SPECIALIST, ModelRole.DRAFT_GENERATOR), 0.14, 2.4, 0.89, "proposal"),
    ("video_event_segmenter", ("video",), (ModelRole.SPECIALIST,), 0.16, 2.7, 0.90, "event"),
    ("motion_estimator", ("motion", "video"), (ModelRole.SPECIALIST, ModelRole.MONITOR), 0.13, 2.1, 0.88, "motion"),
    ("audio_event_encoder", ("audio",), (ModelRole.SPECIALIST,), 0.11, 1.8, 0.87, "audio"),
    ("speech_grounder", ("speech", "audio"), (ModelRole.SPECIALIST, ModelRole.REPRESENTATION_ALIGNER), 0.15, 2.2, 0.90, "text"),
    ("depth_estimator", ("depth_3d", "image"), (ModelRole.SPECIALIST, ModelRole.DRAFT_GENERATOR), 0.18, 2.8, 0.91, "depth_3d"),
    ("spatial_scene_mapper", ("depth_3d", "video"), (ModelRole.SPECIALIST, ModelRole.SIMULATOR), 0.20, 3.0, 0.92, "scene"),
    ("body_dynamics_predictor", ("body_tool",), (ModelRole.SPECIALIST, ModelRole.SIMULATOR), 0.12, 1.9, 0.90, "body_tool"),
    ("cross_modal_binder", ALL_MODALITIES, (ModelRole.ARBITER, ModelRole.REPRESENTATION_ALIGNER), 0.46, 4.0, 0.91, "binding"),
    ("evidence_verifier", ALL_MODALITIES, (ModelRole.VERIFIER, ModelRole.CRITIC, ModelRole.ARBITER), 0.75, 6.0, 0.98, "verification"),
    ("contextual_router", ALL_MODALITIES, (ModelRole.ROUTER, ModelRole.MONITOR), 0.25, 1.2, 0.92, "routing"),
    ("plan_simulator", ("text", "body_tool", "depth_3d"), (ModelRole.PLANNER, ModelRole.SIMULATOR), 0.28, 3.8, 0.93, "plan"),
)


@lru_cache(maxsize=1)
def _default_model_contracts() -> tuple[ModelContract, ...]:
    """Build immutable default contracts once; modules remain per-registry."""

    return tuple(
        _contract(
            identity,
            modalities,
            roles,
            cost=cost,
            latency_ms=latency_ms,
            confidence=confidence,
            output=output,
        )
        for identity, modalities, roles, cost, latency_ms, confidence, output in _DEFAULT_MODULE_SPECS
    )


_DEFAULT_RELATIONSHIPS = (
    SupportRelationship(
        "image_object_detector",
        "evidence_verifier",
        "drafts_for",
        "uncertainty_above_threshold",
        "cost_adjusted_accuracy",
    ),
    SupportRelationship(
        "evidence_verifier",
        "image_object_detector",
        "verifies",
        "draft_is_uncertain",
        "verified_accuracy",
    ),
    SupportRelationship(
        "plan_simulator",
        "body_dynamics_predictor",
        "simulates_for",
        "plan_has_body_action",
        "physical_validity",
    ),
    SupportRelationship(
        "speech_grounder",
        "cross_modal_binder",
        "translates_for",
        "spoken_visible_reference",
        "binding_accuracy",
    ),
    SupportRelationship(
        "contextual_router",
        "depth_estimator",
        "routes_to",
        "depth_modality_present",
        "routing_utility",
    ),
)


def default_model_registry() -> ModelRegistry:
    """Return thirteen independently callable, role-plural local modules."""

    registry = ModelRegistry()
    for contract in _default_model_contracts():
        registry.register(DeterministicModelModule(contract, _default_evaluator))
    for relationship in _DEFAULT_RELATIONSHIPS:
        registry.connect(relationship)
    return registry


def model_support_positive_fixture() -> dict[str, Any]:
    """Measure selective draft-and-verify support against both independent organs."""

    registry = default_model_registry()
    cases = (
        ({"coarse_signal": 0.90, "fine_signal": 0.80}, "positive"),
        ({"coarse_signal": -0.80, "fine_signal": -0.90}, "negative"),
        ({"coarse_signal": -0.04, "fine_signal": 0.75}, "positive"),
        ({"coarse_signal": 0.03, "fine_signal": -0.70}, "negative"),
    )
    draft_correct = 0
    supported_correct = 0
    verifier_correct = 0
    draft_cost = supported_cost = verifier_cost = 0.0
    verification_calls = 0
    rows = []
    for index, (public, private_label) in enumerate(cases):
        draft_request = ModelRequest(f"support-{index}", "binary_draft", "image", public, ModelRole.DRAFT_GENERATOR)
        draft = registry.invoke("image_object_detector", draft_request)
        draft_cost += draft.cost
        supported = draft
        if draft.confidence < 0.60:
            verify_request = ModelRequest(f"support-{index}", "binary_verify", "image", public, ModelRole.VERIFIER)
            supported = registry.invoke("evidence_verifier", verify_request)
            supported_cost += supported.cost
            verification_calls += 1
        supported_cost += draft.cost
        independent_verify = registry.invoke(
            "evidence_verifier",
            ModelRequest(f"verify-{index}", "binary_verify", "image", public, ModelRole.INDEPENDENT_PERFORMER),
        )
        verifier_cost += independent_verify.cost
        draft_correct += int(draft.value == private_label)
        supported_correct += int(supported.value == private_label)
        verifier_correct += int(independent_verify.value == private_label)
        rows.append(
            {
                "task": f"support-{index}",
                "draft": draft.value,
                "draft_confidence": draft.confidence,
                "support_output": supported.value,
                "verification_used": supported is not draft,
            },
        )
    count = len(cases)
    result = {
        "independent_units": count,
        "draft_accuracy": draft_correct / count,
        "supported_accuracy": supported_correct / count,
        "verifier_accuracy": verifier_correct / count,
        "draft_cost": draft_cost,
        "supported_cost": supported_cost,
        "verifier_always_cost": verifier_cost,
        "verification_calls": verification_calls,
        "rows": rows,
    }
    result["positive"] = (
        result["supported_accuracy"] > result["draft_accuracy"]
        and result["supported_accuracy"] == result["verifier_accuracy"]
        and result["supported_cost"] < result["verifier_always_cost"]
    )
    return result


def model_routing_positive_fixture() -> dict[str, Any]:
    """Measure modality routing without making outcome or target available to the router."""

    registry = default_model_registry()
    specialist = {
        "text": "language_interpreter",
        "image": "image_object_detector",
        "video": "video_event_segmenter",
        "motion": "motion_estimator",
        "audio": "audio_event_encoder",
        "speech": "speech_grounder",
        "depth_3d": "depth_estimator",
        "body_tool": "body_dynamics_predictor",
    }
    cases = tuple(
        (modality, {"observable_cue": 1.0 if index % 2 == 0 else -1.0}, "present" if index % 2 == 0 else "absent")
        for index, modality in enumerate(ALL_MODALITIES)
    )
    routed_correct = general_correct = 0
    routed_cost = general_cost = 0.0
    routes = []
    for index, (modality, public, private_label) in enumerate(cases):
        request = ModelRequest(
            f"route-{index}",
            "modality_classify",
            modality,
            public,
            minimum_confidence=0.85,
        )
        decision, output = registry.execute_routed(request)
        expected_specialist = specialist[modality]
        routed_correct += int(output.value == private_label)
        routed_cost += output.cost
        general = registry.invoke("evidence_verifier", request)
        general_correct += int(general.value == private_label)
        general_cost += general.cost
        routes.append(
            {
                "task": request.task_id,
                "modality": modality,
                "selected": decision.selected_model,
                "expected_specialist": expected_specialist,
                "outcome_information_used": decision.outcome_information_used,
            },
        )
    count = len(cases)
    result = {
        "independent_units": count,
        "modalities": tuple(modality for modality, _, _ in cases),
        "routed_accuracy": routed_correct / count,
        "generalist_accuracy": general_correct / count,
        "routed_cost": routed_cost,
        "generalist_cost": general_cost,
        "routes": routes,
    }
    result["positive"] = (
        result["routed_accuracy"] >= result["generalist_accuracy"]
        and result["routed_cost"] < result["generalist_cost"]
        and all(not row["outcome_information_used"] for row in routes)
    )
    return result


__all__ = [
    "ALL_MODALITIES",
    "CallableModel",
    "DeterministicModelModule",
    "ModelContract",
    "ModelContractError",
    "ModelOutput",
    "ModelRegistry",
    "ModelRequest",
    "ModelRole",
    "RoutingDecision",
    "SupportRelationship",
    "default_model_registry",
    "model_routing_positive_fixture",
    "model_support_positive_fixture",
]
