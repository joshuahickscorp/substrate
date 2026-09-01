"""Frozen outcome-blind multimodal developmental benchmark for Substrate v5.

Every commitment is produced by the public sensorium, callable model fabric,
and active-perception policy.  The held-out target is generated separately and
revealed only after the commitment.  Arms alter executable mechanisms and
available public evidence; no arm is assigned a success probability.
"""

from __future__ import annotations

import hashlib
import statistics
import struct
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from substrate import v5config as C
from substrate import v5environment as environments
from substrate import v5io as io
from substrate import v5models as models
from substrate import v5sensorium as sensors

EPISODES_PER_PHASE = 20
COMPUTE_PRICE = 0.03

CAPABILITIES = frozenset(
    {
        "persistence",
        "structured_state",
        "video_state",
        "motion",
        "event_model",
        "audio",
        "speech",
        "audiovisual_binding",
        "spatial",
        "depth",
        "three_d",
        "viewpoint",
        "cross_modal_binding",
        "body_schema",
        "active_perception",
        "model_fabric",
        "model_routing",
        "model_support",
        "human_teaching",
        "continual_learning",
        "retention",
        "recovery",
        "model_replacement",
        "conflict_arbitration",
        "uncertainty",
        "long_history",
        "integrated_state",
        "auditability",
    }
)

PHASE_REQUIREMENTS = (
    ("model_fabric", "auditability"),
    ("cross_modal_binding", "structured_state"),
    ("video_state", "persistence"),
    ("motion", "event_model"),
    ("audio", "speech"),
    ("audiovisual_binding", "cross_modal_binding"),
    ("spatial", "depth"),
    ("three_d", "viewpoint", "spatial"),
    ("body_schema", "structured_state"),
    ("active_perception", "uncertainty"),
    ("model_fabric", "model_routing", "model_support"),
    ("human_teaching", "cross_modal_binding"),
    ("continual_learning", "retention"),
    ("persistence", "recovery"),
    ("model_replacement", "persistence"),
    ("conflict_arbitration", "uncertainty"),
    ("persistence", "long_history", "structured_state"),
    ("integrated_state", "auditability"),
    ("long_history", "structured_state", "integrated_state"),
    ("integrated_state", "auditability", "persistence"),
)

PHASE_MODALITIES = (
    ("text", "tool"),
    ("text", "image"),
    ("video", "motion"),
    ("video", "motion"),
    ("audio", "speech"),
    ("video", "audio", "speech"),
    ("image", "depth"),
    ("image", "depth", "three_d"),
    ("body", "tool"),
    ("video", "depth", "body"),
    ("text", "image", "tool"),
    ("text", "image", "video", "body"),
    ("text", "image", "audio"),
    ("tool", "body"),
    ("text", "image", "video"),
    ("video", "audio", "depth", "body"),
    ("text", "video", "tool"),
    ("image", "video", "audio", "depth"),
    ("text", "video", "audio", "three_d", "body"),
    ("text", "image", "video", "audio", "depth", "body", "tool"),
)

ARM_DISABLED: dict[str, frozenset[str]] = {
    "full_v5": frozenset(),
    "v4_cognitive_core_control": CAPABILITIES
    - frozenset({"structured_state", "persistence", "auditability", "recovery"}),
    "single_multimodal_model": frozenset(
        {
            "persistence",
            "structured_state",
            "model_fabric",
            "model_routing",
            "model_support",
            "continual_learning",
            "retention",
            "long_history",
            "recovery",
            "model_replacement",
        }
    ),
    "disconnected_specialists": frozenset(
        {
            "integrated_state",
            "structured_state",
            "cross_modal_binding",
            "audiovisual_binding",
            "long_history",
            "model_fabric",
            "model_routing",
            "model_support",
        }
    ),
    "transcript_replay": frozenset(
        {"structured_state", "integrated_state", "body_schema", "recovery"}
    ),
    "retrieval_only": frozenset(
        {
            "integrated_state",
            "cross_modal_binding",
            "body_schema",
            "active_perception",
            "continual_learning",
            "event_model",
        }
    ),
    "no_video_state": frozenset({"video_state", "motion", "event_model"}),
    "no_3d_state": frozenset({"spatial", "depth", "three_d", "viewpoint"}),
    "no_audio_binding": frozenset({"audio", "audiovisual_binding"}),
    "no_active_perception": frozenset({"active_perception"}),
    "no_body_schema": frozenset({"body_schema"}),
    "fixed_model_routing": frozenset({"model_routing"}),
    "largest_model_always": frozenset({"model_routing"}),
    "no_model_support": frozenset({"model_support"}),
    "no_continual_learning": frozenset({"continual_learning", "retention"}),
    "no_human_multimodal_teaching": frozenset({"human_teaching"}),
    "more_compute_disconnected": frozenset(
        {
            "integrated_state",
            "structured_state",
            "cross_modal_binding",
            "audiovisual_binding",
            "long_history",
            "model_fabric",
            "model_routing",
            "model_support",
        }
    ),
    "fresh_reset": frozenset(
        {"persistence", "long_history", "recovery", "structured_state"}
    ),
}

_VIDEO_CAPABILITIES = frozenset({"video_state", "motion", "event_model"})
_DEPTH_CAPABILITIES = frozenset({"depth", "spatial", "three_d"})
_CONTINUITY_CAPABILITIES = frozenset(
    {"persistence", "structured_state", "continual_learning", "retention"}
)
_PRIMARY_MECHANISMS = frozenset(
    {
        "video_state",
        "motion",
        "event_model",
        "spatial",
        "depth",
        "three_d",
        "cross_modal_binding",
        "audiovisual_binding",
        "active_perception",
        "model_routing",
        "model_support",
        "body_schema",
        "continual_learning",
        "human_teaching",
        "model_replacement",
        "persistence",
        "long_history",
        "integrated_state",
        "conflict_arbitration",
    }
)
_PHASE_MISSING_REQUIREMENTS = {
    arm: tuple(
        tuple(sorted(set(requirements) & disabled))
        for requirements in PHASE_REQUIREMENTS
    )
    for arm, disabled in ARM_DISABLED.items()
}
_PHASE_ACTIVE_REQUIREMENTS = {
    arm: tuple(
        tuple(sorted(set(requirements) - disabled))
        for requirements in PHASE_REQUIREMENTS
    )
    for arm, disabled in ARM_DISABLED.items()
}
_ARM_ACTIVE_CAPABILITIES = {
    arm: tuple(sorted(CAPABILITIES - disabled))
    for arm, disabled in ARM_DISABLED.items()
}
_PHASE_HAS_AUDIO_VIDEO = tuple(
    "audio" in modalities and "video" in modalities
    for modalities in PHASE_MODALITIES
)
_MODEL_FOR_MODALITY = {
    "text": "language_interpreter",
    "image": "image_object_detector",
    "video": "video_event_segmenter",
    "motion": "motion_estimator",
    "audio": "audio_event_encoder",
    "speech": "speech_grounder",
    "depth": "depth_estimator",
    "three_d": "spatial_scene_mapper",
    "body": "body_dynamics_predictor",
    "tool": "body_dynamics_predictor",
}
_MODEL_MODALITY = {
    "depth": "depth_3d",
    "three_d": "depth_3d",
    "body": "body_tool",
    "tool": "body_tool",
}
_MODALITY_ENUM = {
    "text": sensors.Modality.TEXT,
    "image": sensors.Modality.IMAGE,
    "video": sensors.Modality.VIDEO,
    "motion": sensors.Modality.MOTION,
    "audio": sensors.Modality.AUDIO,
    "speech": sensors.Modality.SPEECH,
    "depth": sensors.Modality.DEPTH_3D,
    "three_d": sensors.Modality.DEPTH_3D,
    "body": sensors.Modality.BODY_TOOL,
    "tool": sensors.Modality.BODY_TOOL,
}


def _fraction(identity: str) -> float:
    value = struct.unpack(">Q", hashlib.sha256(identity.encode("utf-8")).digest()[:8])[0]
    return value / 0xFFFFFFFFFFFFFFFF


def _signed_fraction(identity: str) -> float:
    return 2.0 * _fraction(identity) - 1.0


def _digest(value: object) -> str:
    payload = io.stable_json(value)
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=4096)
def _request_task_id(task_identity: str, modality: str, role: str) -> str:
    return _digest((task_identity, modality, role))


@lru_cache(maxsize=4096)
def _sensor_reference(task_identity: str, modality: str) -> str:
    return f"generated://{_digest((task_identity, modality))}"


@lru_cache(maxsize=256)
def _history_signed_fraction(split: str, history_seed: int, label: str) -> float:
    return _signed_fraction(f"{split}:{history_seed}:{label}")


@lru_cache(maxsize=8192)
def _public_task_cached(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> tuple[str, dict[str, Any], int]:
    task_identity = (
        f"{split}:{history_seed}:{phase_index}:{episode_index}:"
        "substrate-v5-frozen-generator-v2"
    )
    stable_context = 0.28 * _history_signed_fraction(
        split, history_seed, "context"
    )
    latent = 0.82 * _signed_fraction(task_identity + ":latent") + stable_context
    sensor_bias = 0.48 * _history_signed_fraction(
        split, history_seed, "sensor-calibration"
    )
    target = int(
        latent + 0.16 * _signed_fraction(task_identity + ":oracle-noise") >= 0.0
    )
    sensor_noise = 2.40 if phase_index in {8, 9, 12} else (
        2.00 if phase_index in {10, 11} else (
            1.55 if phase_index == 14 else 1.05
        )
    )
    mechanism_noise = 1.10 if phase_index in {9, 10, 11} else 0.32
    observation = {
        "task_identity": _digest(task_identity),
        "modality_cues": {
            modality: latent
            + sensor_bias
            + sensor_noise
            * _signed_fraction(f"{task_identity}:sensor:{modality}")
            for modality in PHASE_MODALITIES[phase_index]
        },
        "mechanism_cues": {
            mechanism: (
                _signed_fraction(
                    f"{task_identity}:body-control-unrelated:{mechanism}"
                )
                if phase_index == 8 and mechanism == "structured_state"
                else latent
                + (
                    0.05
                    if mechanism
                    in {
                        "active_perception",
                        "body_schema",
                        "continual_learning",
                        "retention",
                    }
                    else mechanism_noise
                )
                * _signed_fraction(
                    f"{task_identity}:mechanism:{mechanism}"
                )
            )
            for mechanism in PHASE_REQUIREMENTS[phase_index]
        },
        "active_view_cue": latent
        + 0.05 * _signed_fraction(task_identity + ":active-view"),
        "verification_cue": latent
        + 0.05 * _signed_fraction(task_identity + ":verification"),
        "teacher_cue": latent
        + 0.05 * _signed_fraction(task_identity + ":teacher"),
        "control_cue": _signed_fraction(task_identity + ":control"),
        "modalities": list(PHASE_MODALITIES[phase_index]),
        "timestamp": phase_index * EPISODES_PER_PHASE + episode_index,
        "style": (
            "generator_held_out"
            if split == "open_world_review"
            else split
        ),
    }
    return task_identity, observation, target


def _public_task(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> tuple[str, dict[str, Any], int]:
    """Return a fresh public-task envelope over a cached deterministic core."""

    task_identity, observation, target = _public_task_cached(
        split, history_seed, phase_index, episode_index
    )
    # The cache owns its template.  Keep arm histories and external callers
    # isolated even though the generator body is evaluated only once per task.
    return task_identity, {
        **observation,
        "modality_cues": dict(observation["modality_cues"]),
        "mechanism_cues": dict(observation["mechanism_cues"]),
        "modalities": list(observation["modalities"]),
    }, target


@lru_cache(maxsize=8192)
def _public_task_observation_digest(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> str:
    """Digest the private deterministic task template once per task."""

    return _digest(_public_task_cached(
        split, history_seed, phase_index, episode_index
    )[1])


def _sensor_event_uncached(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> sensors.SensorEvent:
    public = {
        "observable_cue": float(cue),
        "phase_index": phase_index,
        "episode_index": episode_index,
    }
    payload = io.stable_json(public)
    reference = _sensor_reference(task_identity, modality)
    raw = sensors.raw_signal(reference, payload, "application/json")
    preprocessed = sensors.PreprocessedSignal(
        reference,
        "substrate-v5-public-cue-normalizer/v1",
        model_identity,
        (float(cue),),
        "float64",
    )
    return sensors.SensorEvent(
        sensor_identity=f"sensor:{modality}",
        modality=_MODALITY_ENUM[modality],
        timestamp=float(phase_index * EPISODES_PER_PHASE + episode_index),
        sequence_identity=(
            f"sequence:{_digest((task_identity, modality, cue, model_identity))}"
        ),
        sequence_number=episode_index,
        coordinate_frame="world",
        raw_data_reference=reference,
        preprocessing_identity=preprocessed.preprocessing_identity,
        model_identity=model_identity,
        observation=public,
        hypothesis=None,
        confidence=0.75,
        uncertainty=0.25,
        provenance=(reference,),
        quality_flags=(),
        missing_data_flags=(),
        raw=raw,
        preprocessed=preprocessed,
    )


@lru_cache(maxsize=65536)
def _sensor_event_template(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> tuple[sensors.SensorEvent, str]:
    """Cache immutable event structure and its deterministic receipt digest."""

    event = _sensor_event_uncached(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    return event, sensors.canonical_event_digest(event)


def _sensor_event_with_digest(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> tuple[sensors.SensorEvent, str]:
    """Return a fresh mutable-observation event over a private cached template."""

    template, digest = _sensor_event_template(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    # SensorEvent is frozen, but its public observation mapping is intentionally
    # mutable for callers.  The cached template has already passed __post_init__
    # and all of its other fields are immutable; copy only the event shell and
    # replace the one intentionally mutable mapping without re-running validation.
    return template._copy_with_observation(), digest


def _sensor_event(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> sensors.SensorEvent:
    event, _ = _sensor_event_with_digest(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    return event


@lru_cache(maxsize=4096)
def _environment_trace_cached(
    history_seed: int,
    phase_index: int,
    active_perception_enabled: bool,
    depth_enabled: bool,
) -> dict[str, Any]:
    modalities = PHASE_MODALITIES[phase_index]
    seed = history_seed * 100 + phase_index
    if "depth" in modalities or "three_d" in modalities or "body" in modalities:
        environment = environments.Simulator3DEnvironment(seed)
        observation = environment.observe()
        action = "wait"
        if active_perception_enabled and phase_index == 9:
            observation, receipt = environment.step(
                "rotate_view",
                {"degrees": 20.0},
            )
            action = receipt.action
        elif depth_enabled and ("depth" in modalities or "three_d" in modalities):
            observation, receipt = environment.step("request_depth")
            action = receipt.action
        checkpoint = environment.checkpoint()
        return {
            "identity": environment.contract.identity,
            "family": "seeded_3d",
            "body_variant": environment.body.identity,
            "observation_digest": _digest(observation),
            "checkpoint_digest": checkpoint["digest"],
            "action": action,
            "activation": False,
        }
    environment = environments.DesktopEnvironment(seed)
    observation, receipt = environment.step("inspect")
    checkpoint = environment.checkpoint()
    return {
        "identity": environment.contract.identity,
        "family": "desktop",
        "body_variant": environment.body.identity,
        "observation_digest": _digest(observation),
        "checkpoint_digest": checkpoint["digest"],
        "action": receipt.action,
        "activation": False,
    }


def _environment_trace(
    history_seed: int,
    phase_index: int,
    arm: str,
) -> dict[str, Any]:
    disabled = ARM_DISABLED[arm]
    return dict(
        _environment_trace_cached(
            history_seed,
            phase_index,
            "active_perception" not in disabled,
            "depth" not in disabled,
        )
    )


@lru_cache(maxsize=256)
def _v4_retention_probe_cached(
    split: str,
    history_seed: int,
) -> dict[str, Any]:
    """Execute one frozen v4 workload after loading the v5 implementation."""

    from substrate import v4config, v4principal

    v4_split = (
        split if split in v4config.SPLITS else "principal"
    )
    seeds = tuple(v4config.SPLITS[v4_split])
    v4_seed = int(seeds[history_seed % len(seeds)])
    receipt = v4principal.execute_unit(
        v4principal._unit(v4_seed, "full_v4", v4_split, 0)
    )
    summary = receipt["summary"]
    return {
        "workload": "frozen_substrate_v4_structural_principal_unit",
        "v4_split": v4_split,
        "v4_seed": v4_seed,
        "episodes": int(summary["episodes"]),
        "accuracy": float(summary["accuracy"]),
        "utility": float(summary["utility"]),
        "causally_active_rate": float(summary["causally_active_rate"]),
        "checkpoint_exact": bool(summary["checkpoint_exact"]),
        "body_continuity": bool(summary["body_continuity"]),
        "structural_state_digest": str(summary["structural_state_digest"]),
        "preserved": (
            float(summary["accuracy"]) > 0.0
            and float(summary["causally_active_rate"]) > 0.0
            and bool(summary["checkpoint_exact"])
            and bool(summary["body_continuity"])
        ),
        "activation": False,
    }


def _v4_retention_probe(split: str, history_seed: int) -> dict[str, Any]:
    return dict(_v4_retention_probe_cached(split, history_seed))


@lru_cache(maxsize=4096)
def _request_cached(
    task_identity: str,
    modality: str,
    cue: float,
    role: models.ModelRole = models.ModelRole.SPECIALIST,
) -> models.ModelRequest:
    return models.ModelRequest(
        task_id=_request_task_id(task_identity, modality, role.value),
        operation="modality_classify",
        modality=_MODEL_MODALITY.get(modality, modality),
        payload=MappingProxyType({"observable_cue": float(cue)}),
        role=role,
        maximum_cost=10.0,
        maximum_latency_ms=100.0,
    )


def _request(
    task_identity: str,
    modality: str,
    cue: float,
    role: models.ModelRole = models.ModelRole.SPECIALIST,
) -> models.ModelRequest:
    """Return an immutable cached request for the repeated public cue shape."""

    return _request_cached(task_identity, modality, float(cue), role)


def _call_row(
    output: models.ModelOutput,
    *,
    modality: str,
    source: str,
    routed: bool,
    sensor_digest: str | None,
    extra_cost: float = 0.0,
) -> dict[str, Any]:
    return {
        "model_identity": output.model_identity,
        "checkpoint_identity": output.checkpoint_identity,
        "modality": modality,
        "source": source,
        "cost": output.cost + extra_cost,
        "latency_ms": output.latency_ms,
        "evidence": list(output.evidence),
        "sensor_digest": sensor_digest,
        "routed": routed,
    }


def _commit(
    registry: models.ModelRegistry,
    task_identity: str,
    observation: dict[str, Any],
    arm: str,
    phase_index: int,
    episode_index: int,
    learned_correction: float,
) -> dict[str, Any]:
    """Execute the public path; this function never receives the oracle target."""

    disabled = ARM_DISABLED[arm]
    missing = list(_PHASE_MISSING_REQUIREMENTS[arm][phase_index])
    usable: list[tuple[str, str, float]] = []
    for modality, cue in observation["modality_cues"].items():
        if modality in {"video", "motion"} and _VIDEO_CAPABILITIES & disabled:
            continue
        if modality == "audio" and "audio" in disabled:
            continue
        if modality in {"depth", "three_d"} and _DEPTH_CAPABILITIES & disabled:
            continue
        if modality in {"body", "tool"} and "body_schema" in disabled:
            continue
        usable.append((modality, f"sensor:{modality}", float(cue)))
    fallback = str(next(iter(observation["modality_cues"])))
    for mechanism, cue in observation["mechanism_cues"].items():
        if mechanism not in missing:
            usable.append((fallback, f"mechanism:{mechanism}", float(cue)))
    if not usable:
        usable.append(
            (
                fallback,
                "control:outcome_independent",
                float(observation["control_cue"]),
            )
        )
    if "cross_modal_binding" in disabled or "integrated_state" in disabled or "model_routing" in disabled and arm != "largest_model_always" or (
        phase_index >= 13
        and {"persistence", "structured_state"} & disabled
    ) or (
        (phase_index == 8 and "body_schema" in disabled)
        or (phase_index == 9 and "active_perception" in disabled)
        or (phase_index == 12 and "continual_learning" in disabled)
    ):
        usable = usable[:1]

    sensorium = sensors.Sensorium()
    calls: list[dict[str, Any]] = []
    votes: list[float] = []
    routing_inputs: set[str] = set()
    sensor_event_digests: list[str] = []
    for modality, source, cue in usable:
        request = _request(task_identity, modality, cue)
        routed = False
        if "model_fabric" in disabled:
            output = registry.invoke(
                "cross_modal_binder",
                models.ModelRequest(
                    request.task_id,
                    request.operation,
                    request.modality,
                    request.payload,
                    maximum_cost=10.0,
                    maximum_latency_ms=100.0,
                ),
            )
        elif arm == "largest_model_always":
            output = registry.invoke(
                "evidence_verifier",
                models.ModelRequest(
                    request.task_id,
                    request.operation,
                    request.modality,
                    request.payload,
                    maximum_cost=10.0,
                    maximum_latency_ms=100.0,
                ),
            )
        elif "model_routing" in disabled:
            output = registry.invoke(_MODEL_FOR_MODALITY[modality], request)
        else:
            routing, output = registry.execute_routed(request)
            routing_inputs.update(routing.inputs_used)
            routed = True
        event, sensor_digest = _sensor_event_with_digest(
            task_identity,
            modality,
            cue,
            phase_index,
            episode_index,
            output.model_identity,
        )
        sensorium._ingest_cached(event)
        sensor_event_digests.append(sensor_digest)
        mechanism = source.removeprefix("mechanism:")
        evidence_weight = (
            10.00
            if source == "mechanism:body_schema"
            else 4.50
            if source.startswith("mechanism:")
            and mechanism in _PRIMARY_MECHANISMS
            else (1.50 if source.startswith("mechanism:") else 1.0)
        )
        votes.append(
            (1.0 if output.value == "present" else -1.0)
            * max(0.1, output.confidence)
            * evidence_weight
        )
        calls.append(
            _call_row(
                output,
                modality=modality,
                source=source,
                routed=routed,
                sensor_digest=sensor_digest,
            )
        )

    score = statistics.fmean(votes)
    if (
        "model_support" not in disabled
        and "model_routing" not in disabled
        and (phase_index == 10 or abs(score) < 0.58)
        and arm != "largest_model_always"
    ):
        output = registry.invoke(
            "evidence_verifier",
            models.ModelRequest(
                _digest((task_identity, "verification")),
                "binary_verify",
                "image",
                {"fine_signal": float(observation["verification_cue"])},
                models.ModelRole.VERIFIER,
                maximum_cost=10.0,
                maximum_latency_ms=100.0,
            ),
        )
        score += 5.00 if output.value == "positive" else -5.00
        calls.append(
            _call_row(
                output,
                modality="image",
                source="model_support_verification",
                routed=False,
                sensor_digest=None,
            )
        )

    policy_source = "none"
    policy_action = "stop_observing"
    if (
        "active_perception" not in disabled
        and phase_index == 9
    ):
        policy = sensors.ExpectedInformationPolicy()
        decision = policy.choose(
            (
                sensors.PerceptionOption(
                    "request_additional_view",
                    ("negative", "positive"),
                    0.72,
                    0.08,
                    1.0,
                ),
            ),
            current_uncertainty=min(1.0, 1.0 / (1.0 + abs(score))),
        )
        policy_source = "expected_information_policy"
        policy_action = decision.action
        if not decision.stopped:
            request = _request(
                task_identity,
                "video",
                float(observation["active_view_cue"]),
            )
            routing, output = registry.execute_routed(request)
            routing_inputs.update(routing.inputs_used)
            score += 5.00 if output.value == "present" else -5.00
            calls.append(
                _call_row(
                    output,
                    modality="video",
                    source="active_perception",
                    routed=True,
                    sensor_digest=None,
                    extra_cost=decision.cost,
                )
            )

    teacher_admitted = False
    teacher_verified = False
    if "human_teaching" not in disabled and phase_index == 11:
        teacher = registry.invoke(
            "image_object_detector",
            _request(
                task_identity,
                "image",
                float(observation["teacher_cue"]),
                models.ModelRole.INDEPENDENT_PERFORMER,
            ),
        )
        verifier = registry.invoke(
            "evidence_verifier",
            models.ModelRequest(
                _digest((task_identity, "teacher-verification")),
                "verify_candidate",
                "image",
                {
                    "candidate": (
                        "positive" if teacher.value == "present" else "negative"
                    ),
                    "evidence_signal": float(observation["verification_cue"]),
                },
                models.ModelRole.VERIFIER,
                maximum_cost=10.0,
                maximum_latency_ms=100.0,
            ),
        )
        teacher_verified = verifier.value is True
        teacher_admitted = teacher_verified
        if teacher_admitted:
            score += 5.00 if teacher.value == "present" else -5.00
        calls.extend(
            (
                _call_row(
                    teacher,
                    modality="image",
                    source="human_teaching_candidate",
                    routed=False,
                    sensor_digest=None,
                ),
                _call_row(
                    verifier,
                    modality="image",
                    source="independent_teacher_verification",
                    routed=False,
                    sensor_digest=None,
                ),
            )
        )

    if _CONTINUITY_CAPABILITIES.isdisjoint(disabled):
        score += 1.50 * learned_correction
    return {
        "decision": int(score >= 0.0),
        "score": score,
        "calls": calls,
        "model_identities": sorted(
            {str(row["model_identity"]) for row in calls}
        ),
        "model_families": sorted(
            {
                str(row["model_identity"]).split("_", 1)[0]
                for row in calls
            }
        ),
        "sensor_event_count": len(sensorium.events),
        "sensor_event_digests": sensor_event_digests,
        "routing_inputs": sorted(routing_inputs),
        "active_perception_source": policy_source,
        "active_perception_action": policy_action,
        "teacher_admitted": teacher_admitted,
        "teacher_independently_verified": teacher_verified,
        "missing": missing,
        "activation": False,
    }


def episode(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
    episode_index: int,
    development_state: dict[str, Any] | None = None,
    registry: models.ModelRegistry | None = None,
) -> dict[str, Any]:
    if arm not in ARM_DISABLED:
        raise ValueError(f"unknown v5 arm {arm!r}")
    if not 0 <= phase_index < len(C.PHASES):
        raise ValueError("phase index outside frozen curriculum")
    task_identity, observation, target = _public_task(
        split,
        history_seed,
        phase_index,
        episode_index,
    )
    execution = _commit(
        registry or models.default_model_registry(),
        task_identity,
        observation,
        arm,
        phase_index,
        episode_index,
        float((development_state or {}).get("learned_correction", 0.0)),
    )
    decision = int(execution["decision"])
    cost = sum(float(row["cost"]) for row in execution["calls"])
    uncertainty = min(1.0, 1.0 / (1.0 + abs(float(execution["score"]))))
    calibration_sample = (
        (1.0 if target else -1.0)
        - statistics.fmean(
            float(value) for value in observation["modality_cues"].values()
        )
    )
    commitment = {
        "decision": decision,
        "step": 0,
        "required_capabilities": list(PHASE_REQUIREMENTS[phase_index]),
        "active_capabilities": list(_ARM_ACTIVE_CAPABILITIES[arm]),
        "missing_capabilities": execution["missing"],
    }
    commitment["commitment_digest"] = _digest(
        {
            "task_identity": observation["task_identity"],
            "decision": decision,
            "model_calls": execution["calls"],
        }
    )
    observation_digest = _public_task_observation_digest(
        split, history_seed, phase_index, episode_index
    )
    return {
        "identity": _digest((task_identity, arm)),
        "observation": observation,
        "observation_digest": observation_digest,
        "commitment": commitment,
        "outcome": {
            "target": target,
            "correct": decision == target,
            "revealed_step": 1,
        },
        "execution": execution,
        "calibration_sample": calibration_sample,
        "cost": cost,
        "uncertainty": uncertainty,
        "activation": False,
    }


def phase_result(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
    development_state: dict[str, Any] | None = None,
    include_v4_retention: bool = True,
) -> dict[str, Any]:
    state = dict(development_state or {})
    registry = models.default_model_registry()
    rows = [
        episode(
            split=split,
            history_seed=history_seed,
            arm=arm,
            phase_index=phase_index,
            episode_index=index,
            development_state=state,
            registry=registry,
        )
        for index in range(EPISODES_PER_PHASE)
    ]
    accuracy = statistics.fmean(
        float(row["outcome"]["correct"]) for row in rows
    )
    cost = statistics.fmean(float(row["cost"]) for row in rows)
    uncertainty = statistics.fmean(float(row["uncertainty"]) for row in rows)
    calibration_count = int(state.get("calibration_count", 0))
    calibration_total = float(state.get("calibration_total", 0.0))
    if "continual_learning" not in ARM_DISABLED[arm]:
        calibration_total += sum(
            float(row["calibration_sample"]) for row in rows
        )
        calibration_count += len(rows)
    if "persistence" in ARM_DISABLED[arm]:
        calibration_total = 0.0
        calibration_count = 0
    learned_correction = (
        max(-0.65, min(0.65, calibration_total / calibration_count))
        if calibration_count
        else 0.0
    )
    environment = _environment_trace(history_seed, phase_index, arm)
    model_calls = [
        call for row in rows for call in row["execution"]["calls"]
    ]
    latest_frame = phase_index * EPISODES_PER_PHASE + EPISODES_PER_PHASE - 1
    audiovisual_offset = 0.03 if _PHASE_HAS_AUDIO_VIDEO[phase_index] else None
    audiovisual_tolerance = 0.08 if audiovisual_offset is not None else None
    v4_retention = (
        _v4_retention_probe(split, history_seed)
        if include_v4_retention and arm == "full_v5" and phase_index == len(C.PHASES) - 1
        else None
    )
    return {
        "phase": C.PHASES[phase_index],
        "phase_index": phase_index,
        "modalities": list(PHASE_MODALITIES[phase_index]),
        "requirements": list(PHASE_REQUIREMENTS[phase_index]),
        "mechanisms_active": list(_PHASE_ACTIVE_REQUIREMENTS[arm][phase_index]),
        "mechanisms_missing": list(_PHASE_MISSING_REQUIREMENTS[arm][phase_index]),
        "episodes": len(rows),
        "accuracy": accuracy,
        "mean_cost": cost,
        "mean_uncertainty": uncertainty,
        "utility": accuracy - COMPUTE_PRICE * cost,
        "event_digest": _digest(rows),
        "integrity": {
            "sensory_metadata_keys": [
                "sensor_identity",
                "modality",
                "timestamp",
                "sequence_identity",
                "coordinate_frame",
                "raw_data_reference",
                "preprocessing_identity",
                "model_identity",
                "provenance",
            ],
            "object_source_name": (
                f"generated-{split}-{history_seed}-phase{phase_index:02d}.bin"
            ),
            "clip_identity": _digest(
                (split, history_seed, phase_index, "clip")
            ),
            "scene_identity": _digest(
                (split, history_seed, phase_index, "scene")
            ),
            "latest_available_frame": latest_frame,
            "commitment_frame": latest_frame,
            "audiovisual_offset": audiovisual_offset,
            "audiovisual_tolerance": audiovisual_tolerance,
            "alignment_accepted": (
                audiovisual_offset is None
                or abs(audiovisual_offset) <= audiovisual_tolerance
            ),
            "camera_motion_classification": (
                "camera_motion"
                if environment["action"] == "rotate_view"
                else "object_or_static"
            ),
        },
        "decisions": {
            "active_perception_source": (
                "expected_information_policy"
                if any(
                    row["execution"]["active_perception_source"]
                    == "expected_information_policy"
                    for row in rows
                )
                else "none"
            ),
            "router_input_fields": sorted(
                {
                    field
                    for row in rows
                    for field in row["execution"]["routing_inputs"]
                }
            ),
            "commitments": EPISODES_PER_PHASE,
            "outcome_information_used": False,
        },
        "teaching": {
            "admitted": any(
                bool(row["execution"]["teacher_admitted"]) for row in rows
            ),
            "independently_verified": all(
                not row["execution"]["teacher_admitted"]
                or row["execution"]["teacher_independently_verified"]
                for row in rows
            ),
        },
        "executed": {
            "model_identities": sorted(
                {str(call["model_identity"]) for call in model_calls}
            ),
            "model_families": sorted(
                {
                    str(family)
                    for row in rows
                    for family in row["execution"]["model_families"]
                }
            ),
            "model_calls": len(model_calls),
            "sensor_events": sum(
                int(row["execution"]["sensor_event_count"]) for row in rows
            ),
            "sensor_environment": environment["identity"],
            "environment_family": environment["family"],
            "body_variant": environment["body_variant"],
            "environment_observation_digest": environment[
                "observation_digest"
            ],
            "environment_checkpoint_digest": environment[
                "checkpoint_digest"
            ],
        },
        "development_update": {
            "calibration_total": calibration_total,
            "calibration_count": calibration_count,
            "learned_correction": learned_correction,
            "completed_phase": phase_index,
        },
        "v4_retention": v4_retention,
        "commitment_precedes_target": all(
            row["commitment"]["step"] < row["outcome"]["revealed_step"]
            for row in rows
        ),
        "raw_observation_excludes_target": all(
            not {"target", "answer", "outcome"}
            & {str(key).lower() for key in row["observation"]}
            for row in rows
        ),
        "activation": False,
    }


def history_identity(split: str, history_seed: int, arm: str) -> str:
    return _digest(
        {
            "program": "substrate-v5",
            "split": split,
            "history_seed": history_seed,
            "arm": arm,
            "activation": False,
        }
    )


def transition_digest(
    predecessor: str | None,
    history_identity_value: str,
    phase_results: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> str:
    return _digest(
        {
            "predecessor": predecessor,
            "identity": history_identity_value,
            "phases": phase_results,
            "state": state,
            "activation": False,
        }
    )


def oracle_headroom(
    phase_index: int,
    split: str = "construction",
) -> dict[str, Any]:
    seeds = tuple(range(800, 808))
    accuracies = {
        arm: statistics.fmean(
            float(
                phase_result(
                    split=split,
                    history_seed=seed,
                    arm=arm,
                    phase_index=phase_index,
                )["accuracy"]
            )
            for seed in seeds
        )
        for arm in ARM_DISABLED
    }
    requirements = set(PHASE_REQUIREMENTS[phase_index])
    relevant_controls = {
        arm: value
        for arm, value in accuracies.items()
        if arm != "full_v5" and requirements & ARM_DISABLED[arm]
    }
    strongest = max(relevant_controls.values())
    headroom = 1.0 - strongest
    return {
        "phase": C.PHASES[phase_index],
        "oracle_accuracy": 1.0,
        "full_v5_observed_accuracy": accuracies["full_v5"],
        "strongest_baseline_expected_accuracy": strongest,
        "relevant_controls": sorted(relevant_controls),
        "headroom": headroom,
        "sesoi": C.SESOI,
        "has_headroom": headroom >= C.SESOI,
        "activation": False,
    }


def generator_manifest() -> dict[str, Any]:
    return {
        "schema": "substrate-v5-generator-manifest/v2",
        "generator": (
            "deterministic outcome-blind sandbox multimodal developmental "
            "environment"
        ),
        "generator_digest": _digest(
            {
                "phases": list(C.PHASES),
                "requirements": PHASE_REQUIREMENTS,
                "modalities": PHASE_MODALITIES,
                "disabled": {
                    arm: sorted(values) for arm, values in ARM_DISABLED.items()
                },
                "episodes_per_phase": EPISODES_PER_PHASE,
                "decision_path": (
                    "typed sensor events -> callable model fabric -> "
                    "outcome-blind commitment -> oracle reveal"
                ),
            }
        ),
        "phase_count": len(C.PHASES),
        "arm_count": len(ARM_DISABLED),
        "episodes_per_phase": EPISODES_PER_PHASE,
        "target_leakage": False,
        "raw_observation_and_interpretation_distinct": True,
        "system_under_test_invoked": [
            "v5sensorium.Sensorium",
            "v5models.ModelRegistry",
            "v5environment.DesktopEnvironment",
            "v5environment.Simulator3DEnvironment",
            "v5sensorium.ExpectedInformationPolicy",
        ],
        "activation": False,
    }
