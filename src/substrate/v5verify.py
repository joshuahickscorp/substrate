"""Independent V5 receipt verification, mutations, reproduction, and classification.

This module deliberately does not consume a principal summary as evidence.  It
loads the sealed unit/checkpoint objects, regenerates every deterministic work
unit, follows each checkpoint chain, and rebuilds all statistical and
classification inputs from the raw phase rows.

Publication is opt-in.  The pure verification functions are therefore useful
to reviewers and tests without silently creating terminal authorities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from substrate import v4config, v4io, v4principal
from substrate import v5analysis as A
from substrate import v5config as C
from substrate import v5environment as VE
from substrate import v5io as io
from substrate import v5models as VM
from substrate import v5principal as P
from substrate import v5sensorium as VS
from substrate import v5state as VST


class Refused(RuntimeError):
    """Independent verification refused incomplete or inconsistent evidence."""


MUTATION_CLASSES = (
    "target_leaked_into_sensory_metadata",
    "object_identity_leaked_through_filenames",
    "same_clip_crosses_splits",
    "same_scene_crosses_splits",
    "future_frame_available_early",
    "audio_and_video_offset_ignored",
    "camera_motion_treated_as_object_motion",
    "depth_omitted_from_checkpoint",
    "three_dimensional_state_omitted_from_identity",
    "body_state_omitted_from_checkpoint",
    "model_checkpoint_identity_ignored",
    "model_replacement_resets_cognitive_identity",
    "teacher_data_admitted_without_verification",
    "active_perception_receives_oracle_action",
    "model_router_receives_future_outcome",
    "largest_model_control_receives_less_compute",
    "transcript_replay_receives_structured_state",
    "fresh_reset_receives_developmental_state",
    "sensor_timestamp_corruption_ignored",
    "coordinate_frame_corruption_ignored",
    "activation_becomes_true",
)

_SEAL_FIELDS = frozenset({"program", "sha256", "source_commit", "source_digest"})
_PRINCIPAL_AUTHORITY = "SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json"
_CLEAN_CLONE_REPORT = "SUBSTRATE_V5_CLEAN_CLONE.json"
_HIDDEN_TARGET_KEYS = frozenset(
    {
        "answer",
        "future_outcome",
        "oracle_action",
        "oracle_operation",
        "private_target",
        "target",
    }
)
_REQUIRED_SENSORY_METADATA_KEYS = frozenset(
    {
        "sensor_identity",
        "modality",
        "timestamp",
        "sequence_identity",
        "coordinate_frame",
        "raw_data_reference",
        "preprocessing_identity",
        "model_identity",
        "provenance",
    }
)
_RANK = {name: index for index, name in enumerate(C.CLASSIFICATIONS)}
_EPISODES_PER_PHASE = 20
_COMPUTE_PRICE = 0.03
_CAPABILITIES = frozenset(
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
_PHASE_REQUIREMENTS = (
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
_PHASE_MODALITIES = (
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
_VIDEO_CAPABILITIES = frozenset({"video_state", "motion", "event_model"})
_DEPTH_CAPABILITIES = frozenset({"depth", "spatial", "three_d"})
_COMMIT_SINGLE_CAPABILITIES = frozenset({"cross_modal_binding", "integrated_state"})
_ARM_DISABLED: dict[str, frozenset[str]] = {
    "full_v5": frozenset(),
    "v4_cognitive_core_control": _CAPABILITIES - frozenset({"structured_state", "persistence", "auditability", "recovery"}),
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
    "transcript_replay": frozenset({"structured_state", "integrated_state", "body_schema", "recovery"}),
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
    "fresh_reset": frozenset({"persistence", "long_history", "recovery", "structured_state"}),
}
_PHASE_MISSING_REQUIREMENTS = {
    arm: tuple(
        tuple(sorted(set(requirements) & disabled))
        for requirements in _PHASE_REQUIREMENTS
    )
    for arm, disabled in _ARM_DISABLED.items()
}
_PHASE_ACTIVE_REQUIREMENTS = {
    arm: tuple(
        tuple(sorted(set(requirements) - disabled))
        for requirements in _PHASE_REQUIREMENTS
    )
    for arm, disabled in _ARM_DISABLED.items()
}
_ARM_ACTIVE_CAPABILITIES = {
    arm: tuple(sorted(_CAPABILITIES - disabled))
    for arm, disabled in _ARM_DISABLED.items()
}
_ALL_MODALITIES = tuple(
    sorted({modality for modalities in _PHASE_MODALITIES for modality in modalities})
)
_SHARD_MODALITIES = tuple(
    tuple(
        sorted(
            {
                modality
                for phase_index in range(
                    shard * P.PHASES_PER_SHARD,
                    (shard + 1) * P.PHASES_PER_SHARD,
                )
                for modality in _PHASE_MODALITIES[phase_index]
            }
        )
    )
    for shard in range(P.SHARDS)
)
_PHASE_HAS_AUDIO_VIDEO = tuple(
    "audio" in modalities and "video" in modalities
    for modalities in _PHASE_MODALITIES
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
    "text": VS.Modality.TEXT,
    "image": VS.Modality.IMAGE,
    "video": VS.Modality.VIDEO,
    "motion": VS.Modality.MOTION,
    "audio": VS.Modality.AUDIO,
    "speech": VS.Modality.SPEECH,
    "depth": VS.Modality.DEPTH_3D,
    "three_d": VS.Modality.DEPTH_3D,
    "body": VS.Modality.BODY_TOOL,
    "tool": VS.Modality.BODY_TOOL,
}


def _strip_seal(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached scientific body with the seal envelope removed."""

    return {
        key: _copy_normalized(value)
        for key, value in document.items()
        if key not in _SEAL_FIELDS
    }


def _copy_normalized(value: Any) -> Any:
    """Detach exact JSON trees without re-dispatching through ``deepcopy``."""

    value_type = type(value)
    if value_type is dict:
        return {key: _copy_normalized(child) for key, child in value.items()}
    if value_type is list:
        return [_copy_normalized(child) for child in value]
    # Loaded v5 documents take the exact built-in path above. Preserve the
    # previous defensive behavior for custom containers and non-JSON callers.
    return copy.deepcopy(value)


def _strip_loaded_seal(document: Mapping[str, Any]) -> dict[str, Any]:
    """Project a seal validated by ``load_json`` without copying its owned tree."""

    # ``load_json`` validates through ``_normal_json``, so this tree is already
    # detached from the file parser and no caller-owned object is retained.
    return {key: value for key, value in document.items() if key not in _SEAL_FIELDS}


def _source_identity(document: Mapping[str, Any]) -> tuple[str, str]:
    source_commit = document.get("source_commit")
    source_digest = document.get("source_digest")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or not isinstance(source_digest, str) or len(source_digest) != 64:
        raise Refused("sealed source identity is incomplete")
    return source_commit, source_digest


def _principal_source_identity() -> tuple[str, str] | None:
    path = io.EVIDENCE / _PRINCIPAL_AUTHORITY
    if not path.is_file():
        return None
    authority = io.load(_PRINCIPAL_AUTHORITY)
    if (
        authority.get("schema") != "substrate-v5-principal-execution/v1"
        or authority.get("all_terminal") is not True
        or authority.get("published_units") != authority.get("expected_units")
    ):
        raise Refused("principal authority is incomplete")
    return _source_identity(authority)


def _source_bound_seal(
    document: Mapping[str, Any],
    source_identity: tuple[str, str],
    *,
    detach: bool = True,
) -> dict[str, Any]:
    # Public callers retain the isolation default. Internal verifier inputs are
    # already detached, validated JSON trees and are never mutated by restore.
    body = _copy_normalized(dict(document)) if detach else dict(document)
    body.pop("sha256", None)
    body["source_commit"], body["source_digest"] = source_identity
    body["sha256"] = io.sha_obj(body)
    return body


def _relative(unit: P.WorkUnit, family: str) -> str:
    return f"{unit.split}/{family}/{unit.identity}.json"


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(io.stable_json(value)).hexdigest()


@lru_cache(maxsize=4096)
def _independent_request_task_id(task_identity: str, modality: str, role: str) -> str:
    return _stable_digest((task_identity, modality, role))


@lru_cache(maxsize=4096)
def _independent_sensor_reference(task_identity: str, modality: str) -> str:
    return f"generated://{_stable_digest((task_identity, modality))}"


@lru_cache(maxsize=16384)
def _independent_scoped_digest(identity: str, scope: str) -> str:
    """Cache repeated two-part identities without caching mutable payloads."""

    return _stable_digest((identity, scope))


@lru_cache(maxsize=4096)
def _independent_phase_artifact_digest(
    split: str,
    history_seed: int,
    phase_index: int,
    artifact: str,
) -> str:
    """Cache immutable phase-scoped artifact identities shared across arms."""

    return _stable_digest((split, history_seed, phase_index, artifact))


def _fraction(identity: str) -> float:
    value = struct.unpack(">Q", hashlib.sha256(identity.encode("utf-8")).digest()[:8])[0]
    return value / 0xFFFFFFFFFFFFFFFF


@lru_cache(maxsize=1)
def _independent_generator_digest() -> str:
    return _stable_digest(
        {
            "phases": list(C.PHASES),
            "requirements": _PHASE_REQUIREMENTS,
            "modalities": _PHASE_MODALITIES,
            "disabled": {arm: sorted(values) for arm, values in _ARM_DISABLED.items()},
            "episodes_per_phase": _EPISODES_PER_PHASE,
            "decision_path": ("typed sensor events -> callable model fabric -> outcome-blind commitment -> oracle reveal"),
        }
    )


def _independent_unit_document(unit: P.WorkUnit) -> dict[str, Any]:
    phase_indices = tuple(
        range(
            unit.shard * P.PHASES_PER_SHARD,
            (unit.shard + 1) * P.PHASES_PER_SHARD,
        )
    )
    dependency = None if unit.shard == 0 else (f"{unit.split}-{unit.history_seed}-{unit.arm}-shard{unit.shard - 1:02d}")
    return {
        "identity": (f"{unit.split}-{unit.history_seed}-{unit.arm}-shard{unit.shard:02d}"),
        "hypotheses": list(C.HYPOTHESES),
        "arm": unit.arm,
        "history_seed": unit.history_seed,
        "split": unit.split,
        "phase_indices": list(phase_indices),
        "phases": [C.PHASES[index] for index in phase_indices],
        "modalities": list(_SHARD_MODALITIES[unit.shard]),
        "models": "registered model-equivalent modules selected by the v5 fabric",
        "body": "desktop_body or seeded_3d_body",
        "inputs": [_independent_generator_digest()],
        "outputs": [
            f"units/{unit.identity}.json",
            f"checkpoints/{unit.identity}.json",
        ],
        "dependencies": [dependency] if dependency else [],
        "resource_class": "cpu_small",
        "worker_class": "deterministic_developmental_history",
        "native_thread_budget": 1,
        "accelerator_requirement": "none",
        "timeout_seconds": 120,
        "retry": "one deterministic retry; preserve both failure receipts",
        "checkpoint": f"checkpoints/{unit.identity}.json",
        "artifact_family": unit.split,
        "claim_ceiling": "multimodal_nous_ready_for_review",
        "event_count": len(phase_indices) * _EPISODES_PER_PHASE,
        "activation": False,
    }


def _signed_fraction(identity: str) -> float:
    return 2.0 * _fraction(identity) - 1.0


@lru_cache(maxsize=256)
def _history_signed_fraction(split: str, history_seed: int, label: str) -> float:
    return _signed_fraction(f"{split}:{history_seed}:{label}")


@lru_cache(maxsize=8192)
def _independent_public_task_cached(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> tuple[str, dict[str, Any], int]:
    task_identity = f"{split}:{history_seed}:{phase_index}:{episode_index}:substrate-v5-frozen-generator-v2"
    stable_context = 0.28 * _history_signed_fraction(
        split, history_seed, "context"
    )
    latent = 0.82 * _signed_fraction(task_identity + ":latent") + stable_context
    sensor_bias = 0.48 * _history_signed_fraction(
        split, history_seed, "sensor-calibration"
    )
    target = int(latent + 0.16 * _signed_fraction(task_identity + ":oracle-noise") >= 0.0)
    sensor_noise = 2.40 if phase_index in {8, 9, 12} else (2.00 if phase_index in {10, 11} else (1.55 if phase_index == 14 else 1.05))
    mechanism_noise = 1.10 if phase_index in {9, 10, 11} else 0.32
    observation = {
        "task_identity": _stable_digest(task_identity),
        "modality_cues": {
            modality: latent + sensor_bias + sensor_noise * _signed_fraction(f"{task_identity}:sensor:{modality}")
            for modality in _PHASE_MODALITIES[phase_index]
        },
        "mechanism_cues": {
            mechanism: (
                _signed_fraction(f"{task_identity}:body-control-unrelated:{mechanism}")
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
                * _signed_fraction(f"{task_identity}:mechanism:{mechanism}")
            )
            for mechanism in _PHASE_REQUIREMENTS[phase_index]
        },
        "active_view_cue": latent + 0.05 * _signed_fraction(task_identity + ":active-view"),
        "verification_cue": latent + 0.05 * _signed_fraction(task_identity + ":verification"),
        "teacher_cue": latent + 0.05 * _signed_fraction(task_identity + ":teacher"),
        "control_cue": _signed_fraction(task_identity + ":control"),
        "modalities": list(_PHASE_MODALITIES[phase_index]),
        "timestamp": phase_index * _EPISODES_PER_PHASE + episode_index,
        "style": ("generator_held_out" if split == "open_world_review" else split),
    }
    return task_identity, observation, target


def _independent_public_task(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> tuple[str, dict[str, Any], int]:
    """Return a fresh public-task envelope over a cached deterministic core."""

    task_identity, observation, target = _independent_public_task_cached(
        split,
        history_seed,
        phase_index,
        episode_index,
    )
    # The cache owns its template. Keep arm histories and external callers
    # isolated even though the generator body is evaluated only once per task.
    return task_identity, {
        **observation,
        "modality_cues": dict(observation["modality_cues"]),
        "mechanism_cues": dict(observation["mechanism_cues"]),
        "modalities": list(observation["modalities"]),
    }, target


@lru_cache(maxsize=8192)
def _independent_public_task_observation_digest(
    split: str,
    history_seed: int,
    phase_index: int,
    episode_index: int,
) -> str:
    """Digest the private deterministic task template once per task."""

    return _stable_digest(
        _independent_public_task_cached(
            split,
            history_seed,
            phase_index,
            episode_index,
        )[1]
    )


@lru_cache(maxsize=4096)
def _independent_request_cached(
    task_identity: str,
    modality: str,
    cue: float,
    role: VM.ModelRole = VM.ModelRole.SPECIALIST,
) -> VM.ModelRequest:
    return VM.ModelRequest(
        task_id=_independent_request_task_id(task_identity, modality, role.value),
        operation="modality_classify",
        modality=_MODEL_MODALITY.get(modality, modality),
        payload=MappingProxyType({"observable_cue": float(cue)}),
        role=role,
        maximum_cost=10.0,
        maximum_latency_ms=100.0,
    )


def _independent_request(
    task_identity: str,
    modality: str,
    cue: float,
    role: VM.ModelRole = VM.ModelRole.SPECIALIST,
) -> VM.ModelRequest:
    """Return an immutable cached request for the repeated public cue shape."""

    return _independent_request_cached(
        task_identity,
        modality,
        float(cue),
        role,
    )


def _independent_call_row(
    output: VM.ModelOutput,
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


def _independent_sensor_event_uncached(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> VS.SensorEvent:
    public = {
        "observable_cue": float(cue),
        "phase_index": phase_index,
        "episode_index": episode_index,
    }
    payload = io.stable_json(public)
    reference = _independent_sensor_reference(task_identity, modality)
    raw_signal = VS.raw_signal(
        reference,
        payload,
        "application/json",
    )
    preprocessed = VS.PreprocessedSignal(
        reference,
        "substrate-v5-public-cue-normalizer/v1",
        model_identity,
        (float(cue),),
        "float64",
    )
    return VS.SensorEvent(
        sensor_identity=f"sensor:{modality}",
        modality=_MODALITY_ENUM[modality],
        timestamp=float(phase_index * _EPISODES_PER_PHASE + episode_index),
        sequence_identity=(f"sequence:{_stable_digest((task_identity, modality, cue, model_identity))}"),
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
        raw=raw_signal,
        preprocessed=preprocessed,
    )


@lru_cache(maxsize=65536)
def _independent_sensor_event_template(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> tuple[VS.SensorEvent, str]:
    """Cache immutable event structure and its deterministic receipt digest."""

    event = _independent_sensor_event_uncached(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    return event, VS.canonical_event_digest(event)


def _independent_sensor_event_with_digest(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> tuple[VS.SensorEvent, str]:
    """Return a fresh mutable-observation event over a private template."""

    template, digest = _independent_sensor_event_template(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    # SensorEvent is frozen, but its public observation mapping is intentionally
    # mutable. The cached template has already passed __post_init__; copy only
    # the event shell and mutable observation without revalidating immutable data.
    return template._copy_with_observation(), digest


def _independent_sensor_event(
    task_identity: str,
    modality: str,
    cue: float,
    phase_index: int,
    episode_index: int,
    model_identity: str,
) -> VS.SensorEvent:
    event, _ = _independent_sensor_event_with_digest(
        task_identity,
        modality,
        cue,
        phase_index,
        episode_index,
        model_identity,
    )
    return event


@lru_cache(maxsize=4096)
def _independent_environment_trace_cached(
    history_seed: int,
    phase_index: int,
    active_perception_enabled: bool,
    depth_enabled: bool,
) -> dict[str, Any]:
    modalities = _PHASE_MODALITIES[phase_index]
    seed = history_seed * 100 + phase_index
    if "depth" in modalities or "three_d" in modalities or "body" in modalities:
        environment = VE.Simulator3DEnvironment(seed)
        observation = environment.observe()
        action = "wait"
        if active_perception_enabled and phase_index == 9:
            observation, action_receipt = environment.step(
                "rotate_view",
                {"degrees": 20.0},
            )
            action = action_receipt.action
        elif depth_enabled and ("depth" in modalities or "three_d" in modalities):
            observation, action_receipt = environment.step("request_depth")
            action = action_receipt.action
        checkpoint = environment.checkpoint()
        return {
            "identity": environment.contract.identity,
            "family": "seeded_3d",
            "body_variant": environment.body.identity,
            "observation_digest": _stable_digest(observation),
            "checkpoint_digest": checkpoint["digest"],
            "action": action,
            "activation": False,
        }
    environment = VE.DesktopEnvironment(seed)
    observation, action_receipt = environment.step("inspect")
    checkpoint = environment.checkpoint()
    return {
        "identity": environment.contract.identity,
        "family": "desktop",
        "body_variant": environment.body.identity,
        "observation_digest": _stable_digest(observation),
        "checkpoint_digest": checkpoint["digest"],
        "action": action_receipt.action,
        "activation": False,
    }


def _independent_environment_trace(
    history_seed: int,
    phase_index: int,
    arm: str,
) -> dict[str, Any]:
    disabled = _ARM_DISABLED[arm]
    return dict(
        _independent_environment_trace_cached(
            history_seed,
            phase_index,
            "active_perception" not in disabled,
            "depth" not in disabled,
        )
    )


@lru_cache(maxsize=256)
def _independent_v4_retention_cached(
    split: str,
    history_seed: int,
) -> dict[str, Any]:
    """Execute the frozen v4 workload without depending on local run artifacts."""

    v4_split = split if split in v4config.SPLITS else "principal"
    seeds = tuple(v4config.SPLITS[v4_split])
    v4_seed = int(seeds[history_seed % len(seeds)])
    unit = v4principal._unit(  # noqa: SLF001
        v4_seed,
        "full_v4",
        v4_split,
        0,
    )
    receipt = v4principal.execute_unit(unit)
    if receipt.get("activation") is not False or not v4principal.validate_receipt(
        receipt,
        unit,
    ):
        raise Refused("frozen v4 retention workload produced an invalid receipt")
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


def _independent_v4_retention(
    split: str,
    history_seed: int,
) -> dict[str, Any]:
    return dict(_independent_v4_retention_cached(split, history_seed))


def _independent_commit(
    registry: VM.ModelRegistry,
    task_identity: str,
    observation: Mapping[str, Any],
    arm: str,
    phase_index: int,
    episode_index: int,
    learned_correction: float,
) -> dict[str, Any]:
    disabled = _ARM_DISABLED[arm]
    missing = list(_PHASE_MISSING_REQUIREMENTS[arm][phase_index])
    usable: list[tuple[str, str, float]] = []
    for modality, cue in observation["modality_cues"].items():
        if (
            modality in {"video", "motion"}
            and _VIDEO_CAPABILITIES & disabled
        ):
            continue
        if modality == "audio" and "audio" in disabled:
            continue
        if (
            modality in {"depth", "three_d"}
            and _DEPTH_CAPABILITIES & disabled
        ):
            continue
        if modality in {"body", "tool"} and "body_schema" in disabled:
            continue
        usable.append((modality, f"sensor:{modality}", float(cue)))
    fallback = str(next(iter(observation["modality_cues"])))
    for mechanism, cue in observation["mechanism_cues"].items():
        if mechanism not in missing:
            usable.append(
                (
                    fallback,
                    f"mechanism:{mechanism}",
                    float(cue),
                )
            )
    if not usable:
        usable.append(
            (
                fallback,
                "control:outcome_independent",
                float(observation["control_cue"]),
            )
        )
    if (
        _COMMIT_SINGLE_CAPABILITIES & disabled
        or ("model_routing" in disabled and arm != "largest_model_always")
        or (phase_index >= 13 and {"persistence", "structured_state"} & disabled)
        or (
            (phase_index == 8 and "body_schema" in disabled)
            or (phase_index == 9 and "active_perception" in disabled)
            or (phase_index == 12 and "continual_learning" in disabled)
        )
    ):
        usable = usable[:1]

    sensorium = VS.Sensorium()
    calls: list[dict[str, Any]] = []
    votes: list[float] = []
    sensor_event_digests: list[str] = []
    routing_inputs: set[str] = set()
    for modality, source, cue in usable:
        request = _independent_request(task_identity, modality, cue)
        routed = False
        if "model_fabric" in disabled:
            output = registry.invoke(
                "cross_modal_binder",
                VM.ModelRequest(
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
                VM.ModelRequest(
                    request.task_id,
                    request.operation,
                    request.modality,
                    request.payload,
                    maximum_cost=10.0,
                    maximum_latency_ms=100.0,
                ),
            )
        elif "model_routing" in disabled:
            output = registry.invoke(
                _MODEL_FOR_MODALITY[modality],
                request,
            )
        else:
            routing, output = registry.execute_routed(request)
            routing_inputs.update(routing.inputs_used)
            routed = True
        event, sensor_digest = _independent_sensor_event_with_digest(
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
            if source.startswith("mechanism:") and mechanism in _PRIMARY_MECHANISMS
            else (1.50 if source.startswith("mechanism:") else 1.0)
        )
        votes.append((1.0 if output.value == "present" else -1.0) * max(0.1, output.confidence) * evidence_weight)
        calls.append(
            _independent_call_row(
                output,
                modality=modality,
                source=source,
                routed=routed,
                sensor_digest=sensor_digest,
            )
        )
    score = statistics.fmean(votes)
    if "model_support" not in disabled and "model_routing" not in disabled and (phase_index == 10 or abs(score) < 0.58) and arm != "largest_model_always":
        output = registry.invoke(
            "evidence_verifier",
            VM.ModelRequest(
                _independent_scoped_digest(task_identity, "verification"),
                "binary_verify",
                "image",
                {"fine_signal": float(observation["verification_cue"])},
                VM.ModelRole.VERIFIER,
                maximum_cost=10.0,
                maximum_latency_ms=100.0,
            ),
        )
        score += 5.00 if output.value == "positive" else -5.00
        calls.append(
            _independent_call_row(
                output,
                modality="image",
                source="model_support_verification",
                routed=False,
                sensor_digest=None,
            )
        )

    policy_source = "none"
    policy_action = "stop_observing"
    if "active_perception" not in disabled and phase_index == 9:
        policy = VS.ExpectedInformationPolicy()
        policy_decision = policy.choose(
            (
                VS.PerceptionOption(
                    "request_additional_view",
                    ("negative", "positive"),
                    0.72,
                    0.08,
                    1.0,
                ),
            ),
            current_uncertainty=min(
                1.0,
                1.0 / (1.0 + abs(score)),
            ),
        )
        policy_source = "expected_information_policy"
        policy_action = policy_decision.action
        if not policy_decision.stopped:
            request = _independent_request(
                task_identity,
                "video",
                float(observation["active_view_cue"]),
            )
            routing, output = registry.execute_routed(request)
            routing_inputs.update(routing.inputs_used)
            score += 5.00 if output.value == "present" else -5.00
            calls.append(
                _independent_call_row(
                    output,
                    modality="video",
                    source="active_perception",
                    routed=True,
                    sensor_digest=None,
                    extra_cost=policy_decision.cost,
                )
            )

    teacher_admitted = False
    teacher_verified = False
    if "human_teaching" not in disabled and phase_index == 11:
        teacher = registry.invoke(
            "image_object_detector",
            _independent_request(
                task_identity,
                "image",
                float(observation["teacher_cue"]),
                VM.ModelRole.INDEPENDENT_PERFORMER,
            ),
        )
        verifier = registry.invoke(
            "evidence_verifier",
            VM.ModelRequest(
                _independent_scoped_digest(task_identity, "teacher-verification"),
                "verify_candidate",
                "image",
                {
                    "candidate": ("positive" if teacher.value == "present" else "negative"),
                    "evidence_signal": float(observation["verification_cue"]),
                },
                VM.ModelRole.VERIFIER,
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
                _independent_call_row(
                    teacher,
                    modality="image",
                    source="human_teaching_candidate",
                    routed=False,
                    sensor_digest=None,
                ),
                _independent_call_row(
                    verifier,
                    modality="image",
                    source="independent_teacher_verification",
                    routed=False,
                    sensor_digest=None,
                ),
            )
        )
    if {
        "persistence",
        "structured_state",
        "continual_learning",
        "retention",
    }.isdisjoint(disabled):
        score += 1.50 * learned_correction
    return {
        "decision": int(score >= 0.0),
        "score": score,
        "calls": calls,
        "model_identities": sorted({str(row["model_identity"]) for row in calls}),
        "model_families": sorted({str(row["model_identity"]).split("_", 1)[0] for row in calls}),
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


def _independent_episode(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
    episode_index: int,
    development_state: Mapping[str, Any] | None = None,
    registry: VM.ModelRegistry | None = None,
) -> dict[str, Any]:
    task_identity, observation, target = _independent_public_task(
        split,
        history_seed,
        phase_index,
        episode_index,
    )
    execution = _independent_commit(
        registry or VM.default_model_registry(),
        task_identity,
        observation,
        arm,
        phase_index,
        episode_index,
        float(
            (development_state or {}).get(
                "learned_correction",
                0.0,
            )
        ),
    )
    decision = int(execution["decision"])
    cost = sum(float(row["cost"]) for row in execution["calls"])
    uncertainty = min(
        1.0,
        1.0 / (1.0 + abs(float(execution["score"]))),
    )
    calibration_sample = (1.0 if target else -1.0) - statistics.fmean(float(value) for value in observation["modality_cues"].values())
    commitment = {
        "decision": decision,
        "step": 0,
        "required_capabilities": list(_PHASE_REQUIREMENTS[phase_index]),
        "active_capabilities": list(_ARM_ACTIVE_CAPABILITIES[arm]),
        "missing_capabilities": execution["missing"],
    }
    commitment["commitment_digest"] = _stable_digest(
        {
            "task_identity": observation["task_identity"],
            "decision": decision,
            "model_calls": execution["calls"],
        }
    )
    return {
        "identity": _independent_scoped_digest(task_identity, arm),
        "observation": observation,
        "observation_digest": _independent_public_task_observation_digest(
            split,
            history_seed,
            phase_index,
            episode_index,
        ),
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


def _independent_phase_result(
    *,
    split: str,
    history_seed: int,
    arm: str,
    phase_index: int,
    development_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(development_state or {})
    registry = VM.default_model_registry()
    rows = [
        _independent_episode(
            split=split,
            history_seed=history_seed,
            arm=arm,
            phase_index=phase_index,
            episode_index=index,
            development_state=state,
            registry=registry,
        )
        for index in range(_EPISODES_PER_PHASE)
    ]
    accuracy = statistics.fmean(float(row["outcome"]["correct"]) for row in rows)
    cost = statistics.fmean(float(row["cost"]) for row in rows)
    uncertainty = statistics.fmean(float(row["uncertainty"]) for row in rows)
    calibration_count = int(state.get("calibration_count", 0))
    calibration_total = float(state.get("calibration_total", 0.0))
    if "continual_learning" not in _ARM_DISABLED[arm]:
        calibration_total += sum(float(row["calibration_sample"]) for row in rows)
        calibration_count += len(rows)
    if "persistence" in _ARM_DISABLED[arm]:
        calibration_total = 0.0
        calibration_count = 0
    learned_correction = max(-0.65, min(0.65, calibration_total / calibration_count)) if calibration_count else 0.0
    environment = _independent_environment_trace(
        history_seed,
        phase_index,
        arm,
    )
    model_calls = [call for row in rows for call in row["execution"]["calls"]]
    latest_frame = phase_index * _EPISODES_PER_PHASE + _EPISODES_PER_PHASE - 1
    audiovisual_offset = 0.03 if _PHASE_HAS_AUDIO_VIDEO[phase_index] else None
    audiovisual_tolerance = 0.08 if audiovisual_offset is not None else None
    v4_retention = _independent_v4_retention(split, history_seed) if arm == "full_v5" and phase_index == len(C.PHASES) - 1 else None
    return {
        "phase": C.PHASES[phase_index],
        "phase_index": phase_index,
        "modalities": list(_PHASE_MODALITIES[phase_index]),
        "requirements": list(_PHASE_REQUIREMENTS[phase_index]),
        "mechanisms_active": list(_PHASE_ACTIVE_REQUIREMENTS[arm][phase_index]),
        "mechanisms_missing": list(_PHASE_MISSING_REQUIREMENTS[arm][phase_index]),
        "episodes": len(rows),
        "accuracy": accuracy,
        "mean_cost": cost,
        "mean_uncertainty": uncertainty,
        "utility": accuracy - _COMPUTE_PRICE * cost,
        "event_digest": _stable_digest(rows),
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
            "object_source_name": (f"generated-{split}-{history_seed}-phase{phase_index:02d}.bin"),
            "clip_identity": _independent_phase_artifact_digest(
                split,
                history_seed,
                phase_index,
                "clip",
            ),
            "scene_identity": _independent_phase_artifact_digest(
                split,
                history_seed,
                phase_index,
                "scene",
            ),
            "latest_available_frame": latest_frame,
            "commitment_frame": latest_frame,
            "audiovisual_offset": audiovisual_offset,
            "audiovisual_tolerance": audiovisual_tolerance,
            "alignment_accepted": (audiovisual_offset is None or abs(audiovisual_offset) <= float(audiovisual_tolerance)),
            "camera_motion_classification": ("camera_motion" if environment["action"] == "rotate_view" else "object_or_static"),
        },
        "decisions": {
            "active_perception_source": (
                "expected_information_policy" if any(row["execution"]["active_perception_source"] == "expected_information_policy" for row in rows) else "none"
            ),
            "router_input_fields": sorted({field for row in rows for field in row["execution"]["routing_inputs"]}),
            "commitments": _EPISODES_PER_PHASE,
            "outcome_information_used": False,
        },
        "teaching": {
            "admitted": any(bool(row["execution"]["teacher_admitted"]) for row in rows),
            "independently_verified": all(not row["execution"]["teacher_admitted"] or row["execution"]["teacher_independently_verified"] for row in rows),
        },
        "executed": {
            "model_identities": sorted({str(call["model_identity"]) for call in model_calls}),
            "model_families": sorted({str(family) for row in rows for family in row["execution"]["model_families"]}),
            "model_calls": len(model_calls),
            "sensor_events": sum(int(row["execution"]["sensor_event_count"]) for row in rows),
            "sensor_environment": environment["identity"],
            "environment_family": environment["family"],
            "body_variant": environment["body_variant"],
            "environment_observation_digest": environment["observation_digest"],
            "environment_checkpoint_digest": environment["checkpoint_digest"],
        },
        "development_update": {
            "calibration_total": calibration_total,
            "calibration_count": calibration_count,
            "learned_correction": learned_correction,
            "completed_phase": phase_index,
        },
        "v4_retention": v4_retention,
        "commitment_precedes_target": all(row["commitment"]["step"] < row["outcome"]["revealed_step"] for row in rows),
        "raw_observation_excludes_target": all(not {"target", "answer", "outcome"} & {str(key).lower() for key in row["observation"]} for row in rows),
        "activation": False,
    }


@lru_cache(maxsize=4096)
def _independent_history_identity(
    split: str,
    history_seed: int,
    arm: str,
) -> str:
    return _stable_digest(
        {
            "program": "substrate-v5",
            "split": split,
            "history_seed": history_seed,
            "arm": arm,
            "activation": False,
        }
    )


def _independent_initial_state(unit: P.WorkUnit) -> dict[str, Any]:
    identity = _independent_history_identity(
        unit.split,
        unit.history_seed,
        unit.arm,
    )
    return {
        "entity_identity": identity,
        "birth_identity": identity,
        "completed_phase": -1,
        "developmental_events": 0,
        "semantic_memories": 0,
        "procedural_memories": 0,
        "tracked_objects": 0,
        "unfinished_goals": ["return-to-scene"],
        "model_identity": "vision-temporal-alpha",
        "model_replacements": 0,
        "body_identity": "desktop-body",
        "body_changes": 0,
        "sensor_interruptions": 0,
        "restorations": 0,
        "development_state": {},
        "model_checkpoint_identity": "builtin:vision-temporal-alpha",
        "model_family": "vision_temporal",
        "sensor_environment": "uninitialized",
        "body_variant": "body:none",
        "executed_model_families": [],
        "sensor_environments": [],
        "body_variants": [],
        "diversity_records_complete": False,
        "activation": False,
    }


def _independent_model_contract(
    identity: str,
    checkpoint_identity: str,
) -> VST.ModelContract:
    return VST.ModelContract(
        identity=identity,
        checkpoint_identity=checkpoint_identity,
        version="v5.0.0",
        license="project-local-deterministic-fixture",
        runtime="python-deterministic",
        hardware_requirements=("cpu",),
        modalities_accepted=("image", "video"),
        modalities_produced=("proposal",),
        training_provenance=("hand-specified", "no-training-data"),
        known_limitations=("bounded synthetic operations only",),
        allowed_roles=(
            "independent_performer",
            "specialist",
            "draft_generator",
        ),
        statefulness="replaceable",
        checkpoint_support=True,
    )


def _independent_new_entity(
    entity_identity: str,
) -> VST.PermanentEntity:
    entity = VST.PermanentEntity(entity_identity)
    entity.upsert_goal(
        "goal:return-to-scene",
        "retain the scene and return after interruption",
        provenance=("frozen-v5-curriculum",),
    )
    modalities = _ALL_MODALITIES
    for modality in modalities:
        entity.attach_sensor(
            f"sensor:{modality}",
            {
                "modality": modality,
                "coordinate_frame": "world",
                "replaceable": True,
                "activation": False,
            },
        )
    entity.replace_body(
        {
            "identity": "desktop-browser-body-v5",
            "sensors": modalities,
            "actuators": ["inspect", "wait"],
            "coordinate_frames": ["world", "desktop_pixels"],
            "capabilities": ["sandbox_observation"],
            "activation": False,
        }
    )
    entity.register_model(
        _independent_model_contract(
            "vision-temporal-alpha",
            "builtin:vision-temporal-alpha",
        )
    )
    return entity


def _independent_restore_entity(
    unit: P.WorkUnit,
    predecessor: Mapping[str, Any] | None,
) -> VST.PermanentEntity:
    if predecessor is None:
        return _independent_new_entity(
            _independent_history_identity(
                unit.split,
                unit.history_seed,
                unit.arm,
            )
        )
    if unit.arm == "fresh_reset":
        return _independent_new_entity(
            _independent_history_identity(
                unit.split,
                unit.history_seed + unit.shard * 100_000,
                unit.arm,
            )
        )
    entity_checkpoint = predecessor.get("entity_checkpoint")
    if not isinstance(entity_checkpoint, Mapping):
        raise Refused("predecessor omits the permanent-entity checkpoint")
    runtime_identity = (io.commit(), io.source_digest())
    if (
        entity_checkpoint.get("source_commit"),
        entity_checkpoint.get("source_digest"),
    ) == runtime_identity:
        # PermanentEntity.restore does not mutate its input, so avoid a second
        # full-tree copy and re-seal when the predecessor is already bound to
        # this runtime source.
        restorable_checkpoint = dict(entity_checkpoint)
    else:
        restorable_checkpoint = _source_bound_seal(
            entity_checkpoint,
            runtime_identity,
            detach=False,
        )
    try:
        return VST.PermanentEntity.restore(restorable_checkpoint)
    except VST.Refused as error:
        raise Refused(f"permanent-entity restore failed: {error}") from error


def _independent_project_phase(
    entity: VST.PermanentEntity,
    unit: P.WorkUnit,
    row: Mapping[str, Any],
) -> None:
    phase_index = int(row["phase_index"])
    executed = row["executed"]
    for modality in row["modalities"]:
        entity.observe_sensor(
            f"sensor:{modality}",
            {
                "phase_index": phase_index,
                "event_digest": row["event_digest"],
                "environment": executed["sensor_environment"],
                "model_identities": executed["model_identities"],
                "commitment_count": row["decisions"]["commitments"],
                "activation": False,
            },
            source_timestamp=phase_index * _EPISODES_PER_PHASE,
            temporal_uncertainty=float(row["mean_uncertainty"]),
        )
    entity.record_memory(
        "episodic",
        f"phase:{phase_index:02d}",
        {
            "phase": row["phase"],
            "accuracy": row["accuracy"],
            "utility": row["utility"],
            "modalities": row["modalities"],
            "event_digest": row["event_digest"],
            "environment": executed["sensor_environment"],
            "body_variant": executed["body_variant"],
            "model_families": executed["model_families"],
            "activation": False,
        },
        provenance=(
            f"principal:{unit.identity}",
            row["event_digest"],
        ),
    )
    entity.update_world(
        "tracked_objects",
        f"track:{unit.history_seed}:persistent",
        {
            "last_phase": phase_index,
            "scene_identity": row["integrity"]["scene_identity"],
            "clip_identity": row["integrity"]["clip_identity"],
            "visible": True,
            "activation": False,
        },
    )
    if {"depth", "three_d"} & set(row["modalities"]):
        entity.update_world(
            "spatial_world",
            f"scene:{unit.history_seed}",
            {
                "scene_identity": row["integrity"]["scene_identity"],
                "environment_checkpoint_digest": executed["environment_checkpoint_digest"],
                "coordinate_frame": "world",
                "activation": False,
            },
        )
    if phase_index == 13:
        entity.interrupt_sensor("sensor:video")
        entity.observe_sensor(
            "sensor:video",
            {
                "phase_index": phase_index,
                "restored": True,
                "activation": False,
            },
            source_timestamp=phase_index * _EPISODES_PER_PHASE + 1,
        )
    if phase_index == 14:
        entity.replace_model(
            "vision-temporal-alpha",
            _independent_model_contract(
                "vision-temporal-beta",
                "builtin:vision-temporal-beta",
            ),
            measured=True,
            evidence=(row["event_digest"],),
        )
    if phase_index == 15:
        entity.replace_body(
            {
                "identity": "simulator-3d-body-v5",
                "sensors": list(_ALL_MODALITIES),
                "actuators": [
                    "inspect",
                    "request_depth",
                    "rotate_view",
                    "wait",
                ],
                "coordinate_frames": ["world", "body", "camera"],
                "capabilities": [
                    "sandbox_observation",
                    "depth_request",
                    "viewpoint_change",
                ],
                "activation": False,
            }
        )


def _independent_execute_unit(
    unit: P.WorkUnit,
    predecessor: Mapping[str, Any] | None = None,
    *,
    source_identity: tuple[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase_indices = tuple(
        range(
            unit.shard * P.PHASES_PER_SHARD,
            (unit.shard + 1) * P.PHASES_PER_SHARD,
        )
    )
    development_state = dict(predecessor["state"].get("development_state", {})) if predecessor else {}
    phases = []
    for index in phase_indices:
        phase = _independent_phase_result(
            split=unit.split,
            history_seed=unit.history_seed,
            arm=unit.arm,
            phase_index=index,
            development_state=development_state,
        )
        phases.append(phase)
        development_state = dict(phase["development_update"])
    entity = _independent_restore_entity(unit, predecessor)
    for phase in phases:
        _independent_project_phase(entity, unit, phase)
    if predecessor is None:
        state = _independent_initial_state(unit)
    else:
        state = {
            key: copy.deepcopy(value)
            for key, value in predecessor["state"].items()
            if key
            not in {
                "depth_state_digest",
                "three_d_state_digest",
                "body_state_digest",
                "permanent_entity_state_digest",
            }
        }
    if unit.arm == "fresh_reset" and unit.shard > 0:
        state = _independent_initial_state(unit)
        state["entity_identity"] = _independent_history_identity(
            unit.split,
            unit.history_seed + unit.shard * 100_000,
            unit.arm,
        )
    state["completed_phase"] = phase_indices[-1]
    state["developmental_events"] = int(state["developmental_events"]) + sum(int(row["episodes"]) for row in phases)
    state["semantic_memories"] = int(state["semantic_memories"]) + sum(int(row["accuracy"] * row["episodes"]) for row in phases)
    state["procedural_memories"] = int(state["procedural_memories"]) + sum(bool(row["mechanisms_active"]) for row in phases)
    state["tracked_objects"] = max(
        int(state["tracked_objects"]),
        3 + sum("video" in row["modalities"] for row in phases),
    )
    if 13 in phase_indices:
        state["sensor_interruptions"] = int(state["sensor_interruptions"]) + 1
        state["restorations"] = int(state["restorations"]) + 1
    if 14 in phase_indices:
        state["model_identity"] = "vision-temporal-beta"
        state["model_replacements"] = int(state["model_replacements"]) + 1
    if 15 in phase_indices:
        state["body_identity"] = "seeded-3d-body"
        state["body_changes"] = int(state["body_changes"]) + 1
    state["development_state"] = dict(phases[-1]["development_update"])
    entity_checkpoint = entity.checkpoint()
    if source_identity is not None:
        entity_checkpoint = _source_bound_seal(
            entity_checkpoint,
            source_identity,
            detach=False,
        )
    entity_state = entity_checkpoint["state"]
    state["depth_state_digest"] = io.sha_obj({key: value for key, value in entity_state["sensory_buffers"].items() if "depth" in key or "three_d" in key})
    state["three_d_state_digest"] = io.sha_obj(entity_state["spatial_world"])
    state["body_state_digest"] = io.sha_obj(entity_state["body_state"])
    state["model_checkpoint_identity"] = "builtin:vision-temporal-beta" if state["model_replacements"] else "builtin:vision-temporal-alpha"
    state["model_family"] = "vision_temporal"
    state["sensor_environment"] = phases[-1]["executed"]["sensor_environment"]
    state["body_variant"] = entity_state["body_state"]["identity"]
    state["executed_model_families"] = sorted(
        set(state.get("executed_model_families", [])) | {str(family) for row in phases for family in row["executed"]["model_families"]}
    )
    state["sensor_environments"] = sorted(set(state.get("sensor_environments", [])) | {str(row["executed"]["sensor_environment"]) for row in phases})
    state["body_variants"] = sorted(set(state.get("body_variants", [])) | {str(row["executed"]["body_variant"]) for row in phases} | {state["body_variant"]})
    state["diversity_records_complete"] = bool(state["executed_model_families"] and state["sensor_environments"] and state["body_variants"])
    state["permanent_entity_state_digest"] = entity_checkpoint["state_sha256"]
    state_digest = io.sha_obj(state)
    predecessor_digest = io.sha_obj(predecessor) if predecessor else None
    unit_document = _independent_unit_document(unit)
    checkpoint = {
        "schema": "substrate-v5-developmental-checkpoint/v1",
        "unit": unit_document,
        "predecessor_checkpoint": predecessor_digest,
        "state": state,
        "state_digest": state_digest,
        "entity_checkpoint": entity_checkpoint,
        "entity_checkpoint_sha256": entity_checkpoint["sha256"],
        "checkpoint_exact": True,
        "activation": False,
    }
    checkpoint["checkpoint_body_digest"] = io.sha_obj(checkpoint)
    receipt = {
        "schema": "substrate-v5-principal-unit/v1",
        "unit": unit_document,
        "predecessor_checkpoint": predecessor_digest,
        "phase_results": phases,
        "summary": {
            "mean_accuracy": statistics.fmean(float(row["accuracy"]) for row in phases),
            "mean_utility": statistics.fmean(float(row["utility"]) for row in phases),
            "mean_cost": statistics.fmean(float(row["mean_cost"]) for row in phases),
            "mean_uncertainty": statistics.fmean(float(row["mean_uncertainty"]) for row in phases),
            "mechanisms_active": sorted({mechanism for row in phases for mechanism in row["mechanisms_active"]}),
            "modalities": sorted({modality for row in phases for modality in row["modalities"]}),
            "events": sum(int(row["episodes"]) for row in phases),
            "entity_identity": state["entity_identity"],
            "birth_identity": state["birth_identity"],
            "model_identity": state["model_identity"],
            "body_identity": state["body_identity"],
            "unfinished_goals": state["unfinished_goals"],
            "state_digest": state_digest,
            "checkpoint_exact": True,
        },
        "source_generator_digest": _independent_generator_digest(),
        "permanent_entity_checkpoint_sha256": entity_checkpoint["sha256"],
        "activation": False,
    }
    return receipt, checkpoint


def _keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {nested for child in value.values() for nested in _keys(child)}
    if isinstance(value, (list, tuple)):
        return {nested for child in value for nested in _keys(child)}
    return set()


def _projection_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _projection_matches(actual[key], expected_value) for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _projection_matches(actual_value, expected_value)
                for actual_value, expected_value in zip(
                    actual,
                    expected,
                    strict=True,
                )
            )
        )
    return actual == expected


def _phase_invariant_errors(phase_rows: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    for row in phase_rows:
        if not isinstance(row, Mapping):
            errors.append("phase_shape")
            continue
        integrity = row.get("integrity")
        decisions = row.get("decisions")
        teaching = row.get("teaching")
        executed = row.get("executed")
        development = row.get("development_update")
        if not all(
            isinstance(value, Mapping)
            for value in (
                integrity,
                decisions,
                teaching,
                executed,
                development,
            )
        ):
            errors.append("phase_provenance_shape")
            continue
        metadata = integrity.get("sensory_metadata_keys")
        if (
            not isinstance(metadata, list)
            or not metadata
            or not all(isinstance(value, str) and value for value in metadata)
            or _HIDDEN_TARGET_KEYS & {str(value).lower() for value in metadata}
        ):
            errors.append("target_leakage")
        metadata_values = {str(value) for value in metadata} if isinstance(metadata, list) else set()
        if "timestamp" not in metadata_values:
            errors.append("timestamp_corruption")
        if "coordinate_frame" not in metadata_values:
            errors.append("coordinate_frame_corruption")
        if not metadata_values >= _REQUIRED_SENSORY_METADATA_KEYS:
            errors.append("sensory_metadata_incomplete")
        source_name = integrity.get("object_source_name")
        if (
            not isinstance(source_name, str)
            or not source_name
            or {
                "object_id",
                "physical_id",
                "track_id",
                "target",
            }
            & set(source_name.lower().replace(".", "-").split("-"))
        ):
            errors.append("object_filename_leakage")
        for identity_field in ("clip_identity", "scene_identity"):
            identity = integrity.get(identity_field)
            if not isinstance(identity, str) or len(identity) != 64:
                errors.append(identity_field)
        latest = integrity.get("latest_available_frame")
        commitment = integrity.get("commitment_frame")
        if not isinstance(latest, int) or not isinstance(commitment, int) or latest > commitment:
            errors.append("future_frame")
        offset = integrity.get("audiovisual_offset")
        tolerance = integrity.get("audiovisual_tolerance")
        accepted = integrity.get("alignment_accepted")
        if accepted is not True:
            errors.append("audiovisual_alignment")
        if offset is not None and (
            not isinstance(offset, (int, float)) or not isinstance(tolerance, (int, float)) or float(tolerance) < 0.0 or abs(float(offset)) > float(tolerance)
        ):
            errors.append("audiovisual_offset")
        if integrity.get("camera_motion_classification") not in {
            "camera_motion",
            "object_or_static",
        }:
            errors.append("camera_motion")
        if (
            row.get("phase_index") == 9
            and "active_perception"
            in row.get(
                "mechanisms_active",
                [],
            )
            and integrity.get("camera_motion_classification") != "camera_motion"
        ):
            errors.append("camera_motion")
        if decisions.get("active_perception_source") not in {
            "none",
            "expected_information_policy",
        }:
            errors.append("active_perception_oracle")
        router_fields = decisions.get("router_input_fields")
        if (
            not isinstance(router_fields, list)
            or _HIDDEN_TARGET_KEYS & {str(value).lower() for value in router_fields}
            or decisions.get("outcome_information_used") is not False
        ):
            errors.append("router_future_outcome")
        if decisions.get("commitments") != row.get("episodes"):
            errors.append("commitment_count")
        if teaching.get("admitted") is True and teaching.get("independently_verified") is not True:
            errors.append("teacher_verification")
        model_identities = executed.get("model_identities")
        model_families = executed.get("model_families")
        if (
            not isinstance(model_identities, list)
            or not model_identities
            or not all(isinstance(value, str) and value for value in model_identities)
            or not isinstance(model_families, list)
            or not model_families
            or not all(isinstance(value, str) and value for value in model_families)
        ):
            errors.append("executed_model_evidence")
        if (
            not isinstance(executed.get("model_calls"), int)
            or int(executed["model_calls"]) < len(model_identities or [])
            or not isinstance(executed.get("sensor_events"), int)
            or int(executed["sensor_events"]) < 0
        ):
            errors.append("executed_counts")
        for field in (
            "sensor_environment",
            "environment_family",
            "body_variant",
        ):
            if not isinstance(executed.get(field), str) or not executed[field]:
                errors.append(f"executed_{field}")
        for field in (
            "environment_observation_digest",
            "environment_checkpoint_digest",
        ):
            if not isinstance(executed.get(field), str) or len(executed[field]) != 64:
                errors.append(f"executed_{field}")
        if (
            not isinstance(development.get("calibration_count"), int)
            or not isinstance(
                development.get("calibration_total"),
                (int, float),
            )
            or not isinstance(
                development.get("learned_correction"),
                (int, float),
            )
            or development.get("completed_phase") != row.get("phase_index")
        ):
            errors.append("development_update")
    return sorted(set(errors))


def _checkpoint_invariant_errors(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    state = checkpoint.get("state")
    entity_checkpoint = checkpoint.get("entity_checkpoint")
    if not isinstance(state, Mapping) or not isinstance(
        entity_checkpoint,
        Mapping,
    ):
        return ["checkpoint_state_shape"]
    for field in (
        "depth_state_digest",
        "three_d_state_digest",
        "body_state_digest",
        "permanent_entity_state_digest",
    ):
        if not isinstance(state.get(field), str) or len(state[field]) != 64:
            errors.append(field)
    for field in (
        "model_checkpoint_identity",
        "model_family",
        "sensor_environment",
        "body_variant",
    ):
        if not isinstance(state.get(field), str) or not state[field]:
            errors.append(field)
    for field in (
        "executed_model_families",
        "sensor_environments",
        "body_variants",
    ):
        values = state.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
            errors.append(field)
    if state.get("diversity_records_complete") is not True:
        errors.append("diversity_records_complete")
    if checkpoint.get("state_digest") != io.sha_obj(state):
        errors.append("state_digest")
    expected_predecessor = io.sha_obj(predecessor) if predecessor else None
    if checkpoint.get("predecessor_checkpoint") != expected_predecessor:
        errors.append("predecessor_checkpoint")
    checkpoint_body = dict(checkpoint)
    supplied_body_digest = checkpoint_body.pop(
        "checkpoint_body_digest",
        None,
    )
    if supplied_body_digest != io.sha_obj(checkpoint_body):
        errors.append("checkpoint_body_digest")
    try:
        if isinstance(entity_checkpoint, dict):
            validated_entity = io._validate_normalized_seal(entity_checkpoint)
        else:
            validated_entity = io.validate_normalized_seal(dict(entity_checkpoint))
    except io.Refused:
        errors.append("entity_checkpoint_seal")
        return sorted(set(errors))
    entity_state = validated_entity.get("state")
    if not isinstance(entity_state, Mapping):
        errors.append("entity_checkpoint_state")
        return sorted(set(errors))
    sensory_buffers = entity_state.get("sensory_buffers")
    spatial_world = entity_state.get("spatial_world")
    body_state = entity_state.get("body_state")
    if not all(
        isinstance(value, Mapping)
        for value in (
            sensory_buffers,
            spatial_world,
            body_state,
        )
    ):
        errors.append("entity_checkpoint_state")
        return sorted(set(errors))
    if validated_entity.get("state_sha256") != io.sha_obj(entity_state):
        errors.append("entity_state_digest")
    if checkpoint.get("entity_checkpoint_sha256") != validated_entity.get("sha256") or receipt.get(
        "permanent_entity_checkpoint_sha256"
    ) != validated_entity.get("sha256"):
        errors.append("entity_checkpoint_identity")
    expected_depth = io.sha_obj({key: value for key, value in sensory_buffers.items() if "depth" in key or "three_d" in key})
    if state.get("depth_state_digest") != expected_depth:
        errors.append("depth_state_digest")
    if state.get("three_d_state_digest") != io.sha_obj(spatial_world):
        errors.append("three_d_state_digest")
    if state.get("body_state_digest") != io.sha_obj(body_state):
        errors.append("body_state_digest")
    if state.get("permanent_entity_state_digest") != validated_entity.get("state_sha256"):
        errors.append("permanent_entity_state_digest")
    if state.get("body_variant") != body_state.get("identity"):
        errors.append("body_variant")
    if state.get("model_checkpoint_identity") != (
        "builtin:vision-temporal-beta" if int(state.get("model_replacements", 0)) else "builtin:vision-temporal-alpha"
    ):
        errors.append("model_checkpoint_identity")
    if predecessor is not None:
        prior_state = predecessor.get("state")
        if (
            receipt.get("unit", {}).get("arm") != "fresh_reset"
            and isinstance(prior_state, Mapping)
            and int(state.get("model_replacements", 0)) > int(prior_state.get("model_replacements", 0))
            and state.get("entity_identity") != prior_state.get("entity_identity")
        ):
            errors.append("model_replacement_identity_reset")
    return sorted(set(errors))


def _pair_errors(
    receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    unit: P.WorkUnit,
    expected_receipt: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None = None,
) -> list[str]:
    """Independently audit one unit/checkpoint pair.

    Exact regeneration is load bearing, while the named checks make failures
    reviewable instead of reducing every defect to one opaque digest mismatch.
    """

    errors: list[str] = []
    if not _projection_matches(receipt, expected_receipt):
        errors.append("receipt_deterministic_regeneration")
    if not _projection_matches(checkpoint, expected_checkpoint):
        errors.append("checkpoint_deterministic_regeneration")
    unit_document = _independent_unit_document(unit)
    if receipt.get("unit") != unit_document or checkpoint.get("unit") != unit_document:
        errors.append("unit_identity")
    if receipt.get("activation") is not False or checkpoint.get("activation") is not False:
        errors.append("activation")
    if receipt.get("source_generator_digest") != _independent_generator_digest():
        errors.append("generator_identity")

    phase_rows = receipt.get("phase_results")
    if not isinstance(phase_rows, list) or len(phase_rows) != P.PHASES_PER_SHARD:
        errors.append("phase_count")
        phase_rows = []
    if [row.get("phase_index") for row in phase_rows if isinstance(row, Mapping)] != list(unit.phase_indices):
        errors.append("phase_order")
    if any(
        not isinstance(row, Mapping) or row.get("commitment_precedes_target") is not True or row.get("raw_observation_excludes_target") is not True
        for row in phase_rows
    ):
        errors.append("commitment_or_target_boundary")
    if _HIDDEN_TARGET_KEYS & _keys(receipt.get("sensory_metadata", {})):
        errors.append("target_leakage")

    summary = receipt.get("summary")
    state = checkpoint.get("state")
    if not isinstance(summary, Mapping) or not isinstance(state, Mapping):
        errors.append("summary_or_state_shape")
        return sorted(set(errors))
    if summary.get("state_digest") != checkpoint.get("state_digest"):
        errors.append("receipt_checkpoint_digest")
    if summary.get("entity_identity") != state.get("entity_identity"):
        errors.append("entity_identity")
    if summary.get("birth_identity") != state.get("birth_identity"):
        errors.append("birth_identity")
    if summary.get("model_identity") != state.get("model_identity"):
        errors.append("model_checkpoint_identity")
    if summary.get("body_identity") != state.get("body_identity"):
        errors.append("body_checkpoint_identity")
    if summary.get("checkpoint_exact") is not True or checkpoint.get("checkpoint_exact") is not True:
        errors.append("checkpoint_exact")
    if checkpoint.get("predecessor_checkpoint") != receipt.get("predecessor_checkpoint"):
        errors.append("predecessor_link")

    if phase_rows:
        errors.extend(_phase_invariant_errors(phase_rows))
        episodes = sum(int(row.get("episodes", 0)) for row in phase_rows)
        if summary.get("events") != episodes:
            errors.append("event_count")
        for field, source in (
            ("mean_accuracy", "accuracy"),
            ("mean_utility", "utility"),
            ("mean_cost", "mean_cost"),
            ("mean_uncertainty", "mean_uncertainty"),
        ):
            actual = statistics.fmean(float(row.get(source, 0.0)) for row in phase_rows)
            if summary.get(field) != actual:
                errors.append(f"summary_{field}")
    errors.extend(
        _checkpoint_invariant_errors(
            receipt,
            checkpoint,
            predecessor,
        )
    )
    return sorted(set(errors))


def _independent_expected(
    arguments: tuple[
        P.WorkUnit,
        Mapping[str, Any] | None,
        tuple[str, str] | None,
    ],
) -> tuple[dict[str, Any], dict[str, Any]]:
    unit, predecessor, source_identity = arguments
    return _independent_execute_unit(
        unit,
        predecessor,
        source_identity=source_identity,
    )


def raw(
    units: Iterable[P.WorkUnit] | None = None,
) -> dict[str, Any]:
    """Load and regenerate sealed raw receipts and every checkpoint chain.

    ``units`` exists for bounded reviewer checks and tests.  The default is the
    complete frozen principal, replication, and open-world DAG.
    """

    selected = list(P.work_units() if units is None else units)
    identities = [unit.identity for unit in selected]
    if len(identities) != len(set(identities)):
        raise Refused("the independent verification unit set contains duplicates")
    selected.sort(key=lambda unit: (unit.split, unit.history_seed, unit.arm, unit.shard))
    receipts: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    paths: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    invalid: dict[str, list[str]] = {}
    seal_errors: dict[str, str] = {}
    principal_source = _principal_source_identity()
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for unit in selected:
        receipt_path = io.RUNS / _relative(unit, "units")
        checkpoint_path = io.RUNS / _relative(unit, "checkpoints")
        paths[unit.identity] = {
            "receipt": receipt_path.relative_to(io.ROOT).as_posix(),
            "checkpoint": checkpoint_path.relative_to(io.ROOT).as_posix(),
        }
        if not receipt_path.is_file() or not checkpoint_path.is_file():
            missing.append(unit.identity)
            continue
        try:
            sealed_receipt = io.load_json(receipt_path)
            sealed_checkpoint = io.load_json(checkpoint_path)
            entity_checkpoint = sealed_checkpoint.get("entity_checkpoint")
            if not isinstance(entity_checkpoint, Mapping):
                raise Refused("checkpoint entity seal is absent")
            if isinstance(entity_checkpoint, dict):
                sealed_entity = io._validate_normalized_seal(entity_checkpoint)
            else:
                sealed_entity = io.validate_normalized_seal(dict(entity_checkpoint))
            identities = {
                _source_identity(sealed_receipt),
                _source_identity(sealed_checkpoint),
                _source_identity(sealed_entity),
            }
            if principal_source is None:
                if len(identities) != 1:
                    raise Refused("raw source identities disagree")
                principal_source = identities.pop()
            elif identities != {principal_source}:
                raise Refused("raw source identity disagrees with principal authority")
            receipt = _strip_loaded_seal(sealed_receipt)
            checkpoint = _strip_loaded_seal(sealed_checkpoint)
        except (Refused, io.Refused, OSError) as error:
            seal_errors[unit.identity] = str(error)
            continue
        loaded[unit.identity] = (receipt, checkpoint)

    # Shards in one history are sequential because each expected receipt must
    # consume the prior *validated* checkpoint. Different histories and arms
    # are independent, so verify each shard wave in bounded isolated processes.
    # Explicit unit selections stay sequential for small tests and reviewer
    # probes; the complete terminal audit is large enough to amortize the pool.
    parallel = units is None and len(selected) >= P.SHARDS * 8 and bool(loaded)
    executor: concurrent.futures.ProcessPoolExecutor | None = None
    if parallel:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(8, os.cpu_count() or 1),
        )
    predecessors: dict[tuple[str, int, str], dict[str, Any]] = {}
    pair_results: dict[str, tuple[dict[str, Any], dict[str, Any], list[str]]] = {}
    try:
        for shard in range(P.SHARDS):
            jobs = [
                (
                    unit,
                    predecessors.get((unit.split, unit.history_seed, unit.arm)),
                    principal_source,
                )
                for unit in selected
                if unit.shard == shard and unit.identity in loaded
            ]
            if not jobs:
                continue
            expected_results = (
                executor.map(_independent_expected, jobs, chunksize=32)
                if executor is not None
                else map(_independent_expected, jobs)
            )
            for job, (expected_receipt, expected_checkpoint) in zip(
                jobs,
                expected_results,
                strict=True,
            ):
                unit, predecessor, _ = job
                receipt, checkpoint = loaded[unit.identity]
                errors = _pair_errors(
                    receipt,
                    checkpoint,
                    unit,
                    expected_receipt,
                    expected_checkpoint,
                    predecessor,
                )
                pair_results[unit.identity] = (receipt, checkpoint, errors)
                if not errors:
                    predecessors[(unit.split, unit.history_seed, unit.arm)] = checkpoint
    finally:
        if executor is not None:
            executor.shutdown()

    # Preserve the historical deterministic insertion order in the report even
    # though computation was scheduled by shard wave.
    for unit in selected:
        result = pair_results.get(unit.identity)
        if result is None:
            continue
        receipt, checkpoint, errors = result
        if errors:
            invalid[unit.identity] = errors
            continue
        receipts[unit.identity] = receipt
        checkpoints[unit.identity] = checkpoint

    all_pass = len(receipts) == len(selected) and len(checkpoints) == len(selected) and not missing and not invalid and not seal_errors
    return {
        "schema": "substrate-v5-raw-independent-verification/v1",
        "receipts": receipts,
        "checkpoints": checkpoints,
        "paths": paths,
        "expected": len(selected),
        "valid": len(receipts),
        "missing": missing,
        "invalid": invalid,
        "seal_errors": seal_errors,
        "principal_source": (
            {
                "source_commit": principal_source[0],
                "source_digest": principal_source[1],
            }
            if principal_source is not None
            else None
        ),
        "hash_chains_valid": all_pass,
        "deterministic_regeneration_exact": all_pass,
        "all_pass": all_pass,
        "activation": False,
    }


def _split_receipts(raw_report: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [receipt for receipt in raw_report["receipts"].values() if receipt["unit"]["split"] == split]


def _cost_ledger(table: A.Table) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in C.ARMS:
        phase_costs: dict[str, float] = {}
        total = 0.0
        episodes = 0
        for phase_index, phase_name in enumerate(C.PHASES):
            rows = [history[arm][phase_index] for history in table.values() if arm in history]
            phase_costs[phase_name] = sum(float(row["mean_cost"]) for row in rows) / len(rows)
            total += sum(float(row["mean_cost"]) * int(row["episodes"]) for row in rows)
            episodes += sum(int(row["episodes"]) for row in rows)
        arms[arm] = {
            "phase_mean_costs": phase_costs,
            "total_episode_cost": total,
            "episodes": episodes,
            "mean_cost": total / episodes,
        }
    return {
        "arms": arms,
        "compute_price": _COMPUTE_PRICE,
        "activation": False,
    }


def _continuity_metrics(raw_report: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for checkpoint in raw_report["checkpoints"].values():
        unit = checkpoint["unit"]
        grouped.setdefault(
            (str(unit["split"]), int(unit["history_seed"]), str(unit["arm"])),
            [],
        ).append(checkpoint)
    full_chains = []
    all_models: set[str] = set()
    all_bodies: set[str] = set()
    executed_model_families: set[str] = set()
    sensor_environments: set[str] = set()
    body_variants: set[str] = set()
    diversity_records_complete = True
    for (split, seed, arm), chain in sorted(grouped.items()):
        chain.sort(key=lambda row: int(row["unit"]["phase_indices"][0]))
        states = [row["state"] for row in chain]
        all_models.update(str(state["model_identity"]) for state in states)
        all_bodies.update(str(state["body_identity"]) for state in states)
        for state in states:
            families = state.get("executed_model_families")
            environments = state.get("sensor_environments")
            variants = state.get("body_variants")
            if isinstance(families, list) and families and all(isinstance(value, str) and value for value in families):
                executed_model_families.update(families)
            else:
                diversity_records_complete = False
            if isinstance(environments, list) and environments and all(isinstance(value, str) and value for value in environments):
                sensor_environments.update(environments)
            else:
                diversity_records_complete = False
            if isinstance(variants, list) and variants and all(isinstance(value, str) and value for value in variants):
                body_variants.update(variants)
            else:
                diversity_records_complete = False
            if state.get("diversity_records_complete") is not True:
                diversity_records_complete = False
        if arm != "full_v5":
            continue
        identities = [str(state["entity_identity"]) for state in states]
        semantic = [int(state["semantic_memories"]) for state in states]
        procedural = [int(state["procedural_memories"]) for state in states]
        full_chains.append(
            {
                "split": split,
                "history_seed": seed,
                "identity_exact": len(set(identities)) == 1 and identities[0] == str(states[0]["birth_identity"]),
                "model_replacement_preserves_identity": (
                    int(states[-1]["model_replacements"]) >= 1 and len({str(state["model_identity"]) for state in states}) >= 2 and len(set(identities)) == 1
                ),
                "body_change_preserves_identity": (
                    int(states[-1]["body_changes"]) >= 1 and len({str(state["body_identity"]) for state in states}) >= 2 and len(set(identities)) == 1
                ),
                "sensor_interruption_recovered": int(states[-1]["sensor_interruptions"]) >= 1 and int(states[-1]["restorations"]) >= 1,
                "semantic_learning_monotonic": semantic == sorted(semantic),
                "procedural_learning_monotonic": procedural == sorted(procedural),
                "unfinished_goal_preserved": all("return-to-scene" in state["unfinished_goals"] for state in states),
                "checkpoint_chain": [str(row["state_digest"]) for row in chain],
            }
        )
    return {
        "chains": full_chains,
        "identity": all(row["identity_exact"] for row in full_chains),
        "model_replacement": all(row["model_replacement_preserves_identity"] for row in full_chains),
        "body_continuity": all(row["body_change_preserves_identity"] for row in full_chains),
        "sensor_recovery": all(row["sensor_interruption_recovered"] for row in full_chains),
        "learning": all(row["semantic_learning_monotonic"] and row["procedural_learning_monotonic"] for row in full_chains),
        "unfinished_goals": all(row["unfinished_goal_preserved"] for row in full_chains),
        "model_identities": sorted(all_models),
        "body_identities": sorted(all_bodies),
        "executed_model_families": sorted(executed_model_families),
        "sensor_environments": sorted(sensor_environments),
        "body_variants": sorted(body_variants),
        "diversity_records_complete": diversity_records_complete,
        "activation": False,
    }


def _mechanism_metrics(
    tables: Mapping[str, A.Table],
    effects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    principal = tables["principal"]
    full_rows = [row for history in principal.values() for row in history["full_v5"].values()]
    modalities = sorted({str(modality) for row in full_rows for modality in row["modalities"]})
    mechanisms = sorted({str(mechanism) for row in full_rows for mechanism in row["mechanisms_active"]})
    return {
        "modalities": modalities,
        "modality_count": len(modalities),
        "mechanisms": mechanisms,
        "object_and_event_identity": effects["H_M2"]["passes"],
        "cross_modal_bindings": effects["H_M4"]["passes"],
        "spatial_and_three_dimensional": effects["H_M3"]["passes"],
        "active_perception_decisions": effects["H_M5"]["passes"],
        "model_routing_decisions": effects["H_M6"]["passes"],
        "model_support": effects["H_M7"]["passes"],
        "body_schema": effects["H_M8"]["passes"],
        "learning_and_retention": effects["H_M9"]["passes"],
        "human_teaching": effects["H_M10"]["passes"],
        "long_history": effects["H_M1"]["passes"] and effects["H_M12"]["passes"],
        "activation": False,
    }


def _historical_v4() -> dict[str, Any]:
    names = {
        "classification": "SUBSTRATE_V4_FINAL_CLASSIFICATION.json",
        "final_state": "SUBSTRATE_V4_FINAL_STATE.json",
        "verification": "SUBSTRATE_V4_INDEPENDENT_VERIFICATION.json",
        "mutation": "SUBSTRATE_V4_MUTATION_REPORT.json",
        "clean_clone": "SUBSTRATE_V4_CLEAN_CLONE.json",
    }
    documents: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key, name in names.items():
        try:
            documents[key] = v4io.load(name)
        except (OSError, ValueError, v4io.Refused) as error:
            errors[key] = str(error)
    preserved = (
        not errors
        and documents["classification"].get("classification") == "functional_proto_nous_candidate"
        and documents["final_state"].get("classification") == "functional_proto_nous_candidate"
        and documents["verification"].get("all_pass") is True
        and documents["mutation"].get("zero_survived") is True
        and documents["clean_clone"].get("all_pass") is True
        and all(document.get("activation") is False for document in documents.values())
    )
    return {
        "classification": documents.get("classification", {}).get("classification"),
        "documents": {
            key: {
                "schema": document.get("schema"),
                "sha256": document.get("sha256"),
                "activation": document.get("activation"),
            }
            for key, document in documents.items()
        },
        "errors": errors,
        "preserved": preserved,
        "activation": False,
    }


def recompute(raw_report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all 15 effects and all terminal gates from raw phase receipts."""

    if raw_report.get("all_pass") is not True:
        raise Refused("raw principal, replication, and open-world receipts are incomplete")
    split_names = tuple(P.SPLIT_SEEDS)
    tables = {split: A.table_from_receipts(_split_receipts(raw_report, split)) for split in split_names}
    split_results = {split: A.effects(table) for split, table in tables.items()}
    principal_effects = split_results["principal"]["effects"]
    costs = {split: _cost_ledger(table) for split, table in tables.items()}
    continuity = _continuity_metrics(raw_report)
    mechanisms = _mechanism_metrics(tables, principal_effects)
    historical = _historical_v4()
    complete = (
        all(len(result["effects"]) == len(C.HYPOTHESES) for result in split_results.values())
        and len(principal_effects) == 15
        and historical["preserved"]
        and continuity["identity"]
    )
    return {
        "schema": "substrate-v5-independent-recomputation/v1",
        "effects": principal_effects,
        "holm": split_results["principal"]["holm"],
        "splits": split_results,
        "replication": split_results["replication"],
        "open_world": split_results["open_world_review"],
        "costs": costs,
        "continuity": continuity,
        "metrics": mechanisms,
        "historical_v4": historical,
        "all_primary_hypotheses_pass": split_results["principal"]["all_pass"],
        "replication_pass": split_results["replication"]["all_pass"],
        "open_world_pass": split_results["open_world_review"]["all_pass"],
        "independent_recomputation_complete": complete,
        "all_pass": complete,
        "activation": False,
    }


def _unit_from_document(document: Mapping[str, Any]) -> P.WorkUnit:
    phase_indices = document["phase_indices"]
    return P.WorkUnit(
        str(document["split"]),
        int(document["history_seed"]),
        str(document["arm"]),
        int(phase_indices[0]) // P.PHASES_PER_SHARD,
    )


def _mutation_base(
    raw_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Select real, independently verified receipt/checkpoint chains."""

    receipts = raw_report["receipts"]
    checkpoints = raw_report["checkpoints"]
    chains: dict[
        str,
        list[tuple[dict[str, Any], dict[str, Any], P.WorkUnit]],
    ] = {}
    arms = (
        "full_v5",
        "largest_model_always",
        "transcript_replay",
        "fresh_reset",
    )
    principal_seed = min(P.SPLIT_SEEDS["principal"])
    for arm in arms:
        rows = []
        for shard in range(P.SHARDS):
            unit = P.WorkUnit("principal", principal_seed, arm, shard)
            try:
                rows.append(
                    (
                        copy.deepcopy(receipts[unit.identity]),
                        copy.deepcopy(checkpoints[unit.identity]),
                        unit,
                    )
                )
            except KeyError as error:
                raise Refused(f"mutation fixture lacks verified unit {unit.identity}") from error
        chains[arm] = rows

    split_samples = []
    for split, split_seeds in P.SPLIT_SEEDS.items():
        unit = P.WorkUnit(split, min(split_seeds), "full_v5", 0)
        try:
            split_samples.append(
                (
                    copy.deepcopy(receipts[unit.identity]),
                    copy.deepcopy(checkpoints[unit.identity]),
                    unit,
                )
            )
        except KeyError as error:
            raise Refused(f"mutation fixture lacks split sample {unit.identity}") from error
    return {
        "chains": chains,
        "split_samples": split_samples,
        "principal_source": copy.deepcopy(raw_report.get("principal_source")),
        "activation": False,
    }


def _mutation_issues(fixture: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if fixture.get("activation") is not False:
        issues.append("activation")
    source = fixture.get("principal_source")
    source_identity = _source_identity(source) if isinstance(source, Mapping) else None

    documents = [row for chain in fixture["chains"].values() for row in chain] + list(fixture["split_samples"])
    for receipt, checkpoint, unit in documents:
        if receipt.get("activation") is not False or checkpoint.get("activation") is not False:
            issues.append("activation")
        predecessor = None
        if unit.shard:
            predecessor = fixture["chains"][unit.arm][unit.shard - 1][1]
        expected_receipt, expected_checkpoint = _independent_execute_unit(
            unit,
            predecessor,
            source_identity=source_identity,
        )
        issues.extend(
            f"{unit.arm}:{unit.shard}:{error}"
            for error in _pair_errors(
                receipt,
                checkpoint,
                unit,
                expected_receipt,
                expected_checkpoint,
                predecessor,
            )
        )

    for identity_field in ("clip_identity", "scene_identity"):
        splits_by_identity: dict[str, set[str]] = {}
        for receipt, _checkpoint, _unit in documents:
            split = str(receipt["unit"]["split"])
            for phase in receipt["phase_results"]:
                identity = str(phase["integrity"][identity_field])
                splits_by_identity.setdefault(identity, set()).add(split)
        if any(len(splits) > 1 for splits in splits_by_identity.values()):
            issues.append(f"{identity_field.removesuffix('_identity')}_split_overlap")

    transcript_rows = fixture["chains"]["transcript_replay"]
    if any("structured_state" in phase["mechanisms_active"] for receipt, _checkpoint, _unit in transcript_rows for phase in receipt["phase_results"]):
        issues.append("transcript_structured_state")

    fresh_rows = fixture["chains"]["fresh_reset"]
    for index in range(1, len(fresh_rows)):
        prior_state = fresh_rows[index - 1][1]["state"]
        state = fresh_rows[index][1]["state"]
        if state["entity_identity"] == prior_state["entity_identity"] or int(state["developmental_events"]) > P.PHASES_PER_SHARD * _EPISODES_PER_PHASE:
            issues.append("fresh_reset_developmental_state")

    full_rows = {int(phase["phase_index"]): phase for receipt, _checkpoint, _unit in fixture["chains"]["full_v5"] for phase in receipt["phase_results"]}
    largest_rows = {
        int(phase["phase_index"]): phase for receipt, _checkpoint, _unit in fixture["chains"]["largest_model_always"] for phase in receipt["phase_results"]
    }
    if any(float(largest_rows[index]["mean_cost"]) < float(full_rows[index]["mean_cost"]) for index in full_rows):
        issues.append("largest_model_compute")
    return sorted(set(issues))


def _mutation_digest(fixture: Mapping[str, Any]) -> str:
    payload = json.dumps(
        fixture,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.document() if isinstance(value, P.WorkUnit) else str(value),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mutations(
    raw_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every mutation named by the master plan against altered real inputs."""

    if raw_report is None:
        raise Refused("mutation testing requires verified raw evidence")
    if raw_report.get("all_pass") is not True:
        raise Refused("mutation testing requires valid raw evidence")
    base = _mutation_base(raw_report)
    if _mutation_issues(base):
        raise Refused("the unmutated mutation fixture is invalid")
    cases: list[dict[str, Any]] = []

    def case(identity: str, alter: Callable[[dict[str, Any]], None]) -> None:
        changed = copy.deepcopy(base)
        before = _mutation_digest(changed)
        alter(changed)
        after = _mutation_digest(changed)
        issues = _mutation_issues(changed)
        cases.append(
            {
                "identity": identity,
                "input_changed": before != after,
                "altered_input_digest": after,
                "detected": bool(issues) and before != after,
                "detectors": issues,
            }
        )

    def receipt(
        arm: str = "full_v5",
        shard: int = 2,
    ) -> dict[str, Any]:
        return changed_ref["value"]["chains"][arm][shard][0]

    def checkpoint(
        arm: str = "full_v5",
        shard: int = 2,
    ) -> dict[str, Any]:
        return changed_ref["value"]["chains"][arm][shard][1]

    def phase(
        arm: str = "full_v5",
        shard: int = 2,
        offset: int = 0,
    ) -> dict[str, Any]:
        return receipt(arm, shard)["phase_results"][offset]

    changed_ref: dict[str, dict[str, Any]] = {}

    def bind(alter: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        def wrapped(changed: dict[str, Any]) -> None:
            changed_ref["value"] = changed
            alter(changed)

        return wrapped

    case(
        MUTATION_CLASSES[0],
        bind(lambda _row: phase()["integrity"]["sensory_metadata_keys"].append("target")),
    )
    case(
        MUTATION_CLASSES[1],
        bind(lambda _row: phase()["integrity"].update(object_source_name="frame-track_id-object_id-7.bin")),
    )

    def overlap(
        changed: dict[str, Any],
        identity_field: str,
        destination: int,
    ) -> None:
        source = changed["split_samples"][0][0]["phase_results"][0]["integrity"][identity_field]
        changed["split_samples"][destination][0]["phase_results"][0]["integrity"][identity_field] = source

    case(
        MUTATION_CLASSES[2],
        lambda row: overlap(row, "clip_identity", 1),
    )
    case(
        MUTATION_CLASSES[3],
        lambda row: overlap(row, "scene_identity", 2),
    )
    case(
        MUTATION_CLASSES[4],
        bind(lambda _row: phase()["integrity"].update(latest_available_frame=int(phase()["integrity"]["commitment_frame"]) + 1)),
    )
    case(
        MUTATION_CLASSES[5],
        bind(
            lambda _row: phase(shard=1)["integrity"].update(
                audiovisual_offset=0.75,
                audiovisual_tolerance=0.05,
                alignment_accepted=True,
            )
        ),
    )
    case(
        MUTATION_CLASSES[6],
        bind(
            lambda _row: phase(
                shard=1,
                offset=4,
            )["integrity"].update(camera_motion_classification="object_or_static")
        ),
    )
    case(
        MUTATION_CLASSES[7],
        bind(lambda _row: checkpoint()["state"].pop("depth_state_digest")),
    )
    case(
        MUTATION_CLASSES[8],
        bind(lambda _row: checkpoint()["state"].pop("three_d_state_digest")),
    )
    case(
        MUTATION_CLASSES[9],
        bind(lambda _row: checkpoint()["state"].pop("body_state_digest")),
    )
    case(
        MUTATION_CLASSES[10],
        bind(lambda _row: checkpoint(shard=3)["state"].update(model_checkpoint_identity="builtin:vision-temporal-alpha")),
    )

    def reset_identity(_row: dict[str, Any]) -> None:
        checkpoint(shard=2)["state"]["entity_identity"] = "replacement-model-identity"

    case(MUTATION_CLASSES[11], bind(reset_identity))
    case(
        MUTATION_CLASSES[12],
        bind(
            lambda _row: phase(shard=2, offset=1)["teaching"].update(
                admitted=True,
                independently_verified=False,
            )
        ),
    )
    case(
        MUTATION_CLASSES[13],
        bind(lambda _row: phase(shard=1, offset=4)["decisions"].update(active_perception_source="oracle_action")),
    )
    case(
        MUTATION_CLASSES[14],
        bind(lambda _row: phase(shard=2)["decisions"]["router_input_fields"].append("future_outcome")),
    )

    def reduce_largest_compute(_row: dict[str, Any]) -> None:
        full_cost = float(phase()["mean_cost"])
        phase("largest_model_always")["mean_cost"] = full_cost / 2.0

    case(MUTATION_CLASSES[15], bind(reduce_largest_compute))
    case(
        MUTATION_CLASSES[16],
        bind(lambda _row: phase("transcript_replay")["mechanisms_active"].append("structured_state")),
    )

    def inherit_development(_row: dict[str, Any]) -> None:
        prior = checkpoint("fresh_reset", 0)["state"]
        current = checkpoint("fresh_reset", 1)["state"]
        current["entity_identity"] = prior["entity_identity"]
        current["developmental_events"] = int(prior["developmental_events"]) + P.PHASES_PER_SHARD * _EPISODES_PER_PHASE

    case(
        MUTATION_CLASSES[17],
        bind(inherit_development),
    )
    case(
        MUTATION_CLASSES[18],
        bind(lambda _row: phase()["integrity"]["sensory_metadata_keys"].remove("timestamp")),
    )
    case(
        MUTATION_CLASSES[19],
        bind(lambda _row: phase()["integrity"]["sensory_metadata_keys"].remove("coordinate_frame")),
    )
    case(
        MUTATION_CLASSES[20],
        bind(lambda _row: receipt().update(activation=not False)),
    )
    survived = [row["identity"] for row in cases if not row["detected"]]
    return {
        "schema": "substrate-v5-mutation-report/v1",
        "required_master_classes": list(MUTATION_CLASSES),
        "mutations": cases,
        "total": len(cases),
        "detected": len(cases) - len(survived),
        "survived": survived,
        "zero_survived": not survived,
        "activation": False,
    }


def _clean_environment(installed: Path, clone: Path) -> dict[str, str]:
    blocked = {
        "PYTHONPATH",
        "SUBSTRATE_DATA_ROOT",
        "SUBSTRATE_REPOSITORY_ROOT",
        "SUBSTRATE_STATE_ROOT",
    }
    clean = {key: value for key, value in os.environ.items() if key not in blocked}
    return {
        **clean,
        "PYTHONPATH": str(installed),
        "SUBSTRATE_REPOSITORY_ROOT": str(clone),
    }


def _clean_clone_command(clone: Path, ready_ref: str = C.READY_TAG) -> list[str]:
    return [
        "git",
        "clone",
        "--quiet",
        "--branch",
        ready_ref,
        str(io.ROOT),
        str(clone),
    ]


def _clean_install_command(installed: Path, clone: Path) -> list[str]:
    return [
        shutil.which("uv") or "uv",
        "pip",
        "install",
        "--quiet",
        "--no-deps",
        "--target",
        str(installed),
        str(clone),
    ]


def _expand_command(
    command: Sequence[str],
    *,
    clone: Path,
    installed: Path,
    ready_ref: str,
    expected_digest: str = "",
) -> list[str]:
    values = {
        "clone": str(clone),
        "installed": str(installed),
        "python": sys.executable,
        "repository": str(io.ROOT),
        "ready_ref": ready_ref,
        "expected_digest": expected_digest,
    }
    expanded = []
    for part in command:
        value = str(part)
        for name, replacement in values.items():
            value = value.replace(f"{{{name}}}", replacement)
        expanded.append(value)
    return expanded


def _result_tail(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2_000:],
        "stderr_tail": result.stderr[-2_000:],
    }


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=127,
            stdout="",
            stderr=str(error),
        )


def _source_worktree_clean() -> bool:
    result = _run_command(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=io.ROOT,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _current_clean_clone_source() -> tuple[str, str] | None:
    try:
        return io.commit(), io.source_digest()
    except (OSError, subprocess.SubprocessError, io.Refused):
        return None


def _reusable_clean_clone(
    expected_digest: str,
    ready_ref: str,
) -> dict[str, Any] | None:
    """Reuse a sealed clean-clone result only under an exact identity match."""

    try:
        cached = io.load(_CLEAN_CLONE_REPORT)
    except (OSError, ValueError, io.Refused):
        return None
    if (
        cached.get("schema") != "substrate-v5-clean-clone/v1"
        or cached.get("commands_injected") is not False
        or cached.get("source_worktree_clean") is not True
        or cached.get("all_pass") is not True
        or cached.get("exact_reproduction") is not True
        or cached.get("normalized_double_regeneration_exact") is not True
        or cached.get("ready_ref_returncode") != 0
        or cached.get("ready_ref") != ready_ref
        or cached.get("expected_digest") != expected_digest
        or cached.get("actual_digests") != [expected_digest, expected_digest]
    ):
        return None
    current_source = _current_clean_clone_source()
    if current_source is None or (
        cached.get("source_commit"),
        cached.get("source_digest"),
    ) != current_source:
        return None
    if not _source_worktree_clean():
        return None
    ready = _run_command(
        ["git", "rev-parse", f"{ready_ref}^{{}}"],
        cwd=io.ROOT,
    )
    if ready.returncode != 0 or ready.stdout.strip() != cached.get("ready_commit"):
        return None
    reused = _strip_loaded_seal(cached)
    reused["cache_reused"] = True
    return reused


def _sample_regeneration(raw_report: Mapping[str, Any]) -> tuple[P.WorkUnit, str]:
    candidates = [unit for unit in P.work_units("principal") if unit.arm == "full_v5" and unit.shard == 0 and unit.identity in raw_report["receipts"]]
    if not candidates:
        raise Refused("clean-clone verification requires a principal full_v5 shard")
    unit = candidates[0]
    source = raw_report.get("principal_source")
    source_identity = _source_identity(source) if isinstance(source, Mapping) else None
    receipt, checkpoint = _independent_execute_unit(
        unit,
        source_identity=source_identity,
    )
    digest = io.sha_obj(
        {
            "receipt": receipt,
            "checkpoint": checkpoint,
        }
    )
    return unit, digest


def clean_clone(
    raw_report: Mapping[str, Any],
    *,
    commands: Mapping[str, Sequence[str]] | None = None,
    ready_ref: str = C.READY_TAG,
) -> dict[str, Any]:
    """Verify clone, install, tests, Ruff, and two normalized regenerations.

    Callers may inject any stage command with ``{clone}``, ``{installed}``,
    ``{python}``, ``{repository}``, ``{ready_ref}``, and
    ``{expected_digest}`` placeholders.  This keeps tests explicit without
    weakening the production defaults.
    """

    if raw_report.get("all_pass") is not True:
        raise Refused("clean-clone verification requires valid raw evidence")
    unit, expected_digest = _sample_regeneration(raw_report)
    if commands is None:
        cached = _reusable_clean_clone(expected_digest, ready_ref)
        if cached is not None:
            return cached
    script = (
        "from substrate import v5io as I,v5principal as P,v5verify as V;"
        f"u=P.WorkUnit({unit.split!r},{unit.history_seed!r},{unit.arm!r},{unit.shard!r});"
        "r,c=V._independent_execute_unit(u);"
        "print(I.sha_obj({'receipt':r,'checkpoint':c}))"
    )
    supplied = dict(commands or {})
    with tempfile.TemporaryDirectory(prefix="substrate-v5-clean-") as temporary:
        clone = Path(temporary) / "repo"
        installed = Path(temporary) / "installed"
        defaults = {
            "clone": _clean_clone_command(clone, ready_ref),
            "install": _clean_install_command(installed, clone),
            "tests": [sys.executable, "-m", "pytest", "-q", "tests/substrate"],
            "ruff": [str(io.ROOT / ".venv/bin/ruff"), "check", "src", "tests"],
            "ruff_format": [
                str(io.ROOT / ".venv/bin/ruff"),
                "format",
                "--check",
                "src/substrate/v5verify.py",
                "tests/substrate/test_v5_verify.py",
            ],
            "regeneration": [sys.executable, "-c", script],
            "ready": ["git", "rev-parse", f"{ready_ref}^{{}}"],
        }
        stage_results: dict[str, subprocess.CompletedProcess[str]] = {}
        clean_env = _clean_environment(installed, clone)
        for name in ("clone", "install", "tests", "ruff", "ruff_format"):
            command = _expand_command(
                supplied.get(name, defaults[name]),
                clone=clone,
                installed=installed,
                ready_ref=ready_ref,
                expected_digest=expected_digest,
            )
            stage_results[name] = _run_command(
                command,
                cwd=None if name == "clone" else clone,
                env=None if name in {"clone", "install"} else clean_env,
            )
            if name == "clone" and not clone.exists():
                clone.mkdir(parents=True)
        regeneration_command = _expand_command(
            supplied.get("regeneration", defaults["regeneration"]),
            clone=clone,
            installed=installed,
            ready_ref=ready_ref,
            expected_digest=expected_digest,
        )
        regenerations = [
            _run_command(
                regeneration_command,
                cwd=clone,
                env=clean_env,
            )
            for _ in range(2)
        ]
        actual = [result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "" for result in regenerations]
        ready_command = _expand_command(
            supplied.get("ready", defaults["ready"]),
            clone=clone,
            installed=installed,
            ready_ref=ready_ref,
            expected_digest=expected_digest,
        )
        ready = _run_command(
            ready_command,
            cwd=io.ROOT,
        )
    stage_report = {name: _result_tail(result) for name, result in stage_results.items()}
    regeneration_report = [_result_tail(result) for result in regenerations]
    all_codes = [result.returncode for result in (*stage_results.values(), *regenerations)]
    exact = expected_digest == actual[0] == actual[1]
    return {
        "schema": "substrate-v5-clean-clone/v1",
        "commands_injected": commands is not None,
        "ready_ref": ready_ref,
        "ready_commit": ready.stdout.strip() if ready.returncode == 0 else None,
        "ready_ref_returncode": ready.returncode,
        "stages": stage_report,
        "regenerations": regeneration_report,
        "expected_digest": expected_digest,
        "actual_digests": actual,
        "exact_reproduction": exact,
        "normalized_double_regeneration_exact": bool(actual[0]) and actual[0] == actual[1],
        "all_pass": ready.returncode == 0 and all(code == 0 for code in all_codes) and exact,
        "source_worktree_clean": _source_worktree_clean(),
        "cache_reused": False,
        "activation": False,
    }


def classification_gates(
    raw_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    mutation: Mapping[str, Any],
    clone: Mapping[str, Any],
    *,
    review_complete: bool,
) -> dict[str, Any]:
    """Evaluate the ordered master-plan gates without skipping a level."""

    effects = verification["effects"]
    metrics = verification["metrics"]
    continuity = verification["continuity"]
    base = raw_report.get("all_pass") is True and verification.get("all_pass") is True and verification["historical_v4"]["preserved"] is True
    multimodal = (
        base
        and continuity["identity"]
        and metrics["modality_count"] >= 6
        and effects["H_M4"]["passes"]
        and effects["H_M2"]["passes"]
        and effects["H_M13"]["passes"]
        and effects["H_M6"]["passes"]
    )
    sensorium = (
        multimodal
        and effects["H_M2"]["passes"]
        and effects["H_M3"]["passes"]
        and effects["H_M4"]["passes"]
        and effects["H_M5"]["passes"]
        and continuity["sensor_recovery"]
    )
    organism = (
        sensorium
        and continuity["diversity_records_complete"]
        and len(continuity["executed_model_families"]) >= 2
        and effects["H_M6"]["passes"]
        and effects["H_M7"]["passes"]
        and continuity["model_replacement"]
        and effects["H_M9"]["passes"]
        and effects["H_M8"]["passes"]
    )
    embodied = (
        organism
        and effects["H_M5"]["passes"]
        and effects["H_M4"]["passes"]
        and effects["H_M3"]["passes"]
        and effects["H_M8"]["passes"]
        and effects["H_M10"]["passes"]
        and effects["H_M1"]["passes"]
        and effects["H_M12"]["passes"]
        and effects["H_M15"]["passes"]
        and continuity["body_continuity"]
        and continuity["learning"]
    )
    review_ready = (
        embodied
        and verification["replication_pass"]
        and verification["open_world_pass"]
        and continuity["diversity_records_complete"]
        and len(continuity["executed_model_families"]) >= 2
        and len(continuity["sensor_environments"]) >= 2
        and len(continuity["body_variants"]) >= 2
        and mutation.get("zero_survived") is True
        and clone.get("all_pass") is True
        and review_complete
    )
    return {
        "functional_proto_nous_candidate": verification["historical_v4"]["preserved"],
        "multimodal_cognitive_substrate": multimodal,
        "persistent_sensorium": sensorium,
        "integrated_model_organism_architecture": organism,
        "persistent_embodied_proto_nous_candidate": embodied,
        "multimodal_nous_ready_for_review": review_ready,
        "activation": False,
    }


def classify(
    raw_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    mutation: Mapping[str, Any],
    clone: Mapping[str, Any],
    *,
    review_complete: bool = False,
) -> dict[str, Any]:
    gates = classification_gates(
        raw_report,
        verification,
        mutation,
        clone,
        review_complete=review_complete,
    )
    classification = C.CLASSIFICATIONS[0]
    for level in C.CLASSIFICATIONS:
        if gates[level]:
            classification = level
        else:
            break
    return {
        "schema": "substrate-v5-final-classification/v1",
        "classification": classification,
        "gates": gates,
        "maximum_automatic_classification": C.CLASSIFICATIONS[-1],
        "unqualified_nous": False,
        "not_claimed": C.CLAIM_BOUNDARY["not_claimed"],
        "activation": False,
    }


def terminal_refusal_authority(
    raw_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    mutation: Mapping[str, Any],
    clone: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an explicit refusal only for an independently recomputed null."""

    nulls = {split: [name for name, row in verification["splits"][split]["effects"].items() if not row["passes"]] for split in P.SPLIT_SEEDS}
    null_count = sum(len(values) for values in nulls.values())
    independently_verified = (
        raw_report.get("all_pass") is True
        and verification.get("all_pass") is True
        and verification.get("independent_recomputation_complete") is True
        and null_count > 0
        and mutation.get("zero_survived") is True
        and clone.get("all_pass") is True
    )
    return {
        "schema": "substrate-v5-terminal-refusal/v1",
        "terminal_refusal": independently_verified,
        "independently_verified": independently_verified,
        "reason": (
            "one or more frozen principal, replication, or open-world hypotheses are scientific nulls"
            if independently_verified
            else "terminal-refusal conditions are not established"
        ),
        "nulls": nulls,
        "null_count": null_count,
        "implementation_or_instrument_defect": False,
        "activation": False,
    }


def _write_review_json(review_root: Path, name: str, document: dict[str, Any]) -> Path:
    return io.publish_json(review_root / name, document)


def review_package(
    raw_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    mutation: Mapping[str, Any],
    clone: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the deterministic raw archive and external-challenge indexes."""

    review_root = io.ARTIFACTS / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    authorities: dict[str, str] = {}
    for root in (io.EVIDENCE, io.CONFIGS):
        for path in sorted(root.glob("*.json")):
            authorities[path.relative_to(io.ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in (
        "src/substrate/v5io.py",
        "src/substrate/v5config.py",
        "src/substrate/v5experiment.py",
        "src/substrate/v5principal.py",
        "src/substrate/v5analysis.py",
        "src/substrate/v5verify.py",
    ):
        path = io.ROOT / relative
        authorities[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    archive_rows = []
    raw_files: dict[str, str] = {}
    for identity in sorted(raw_report["receipts"]):
        for family in ("receipt", "checkpoint"):
            relative = raw_report["paths"][identity][family]
            path = io.ROOT / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            raw_files[relative] = digest
            archive_rows.append(
                json.dumps(
                    {
                        "path": relative,
                        "sha256": digest,
                        "document": json.loads(path.read_text(encoding="utf-8")),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
    archive = gzip.compress(
        ("\n".join(archive_rows) + "\n").encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    archive_path = io.atomic_write_bytes(
        review_root / "RAW_RECEIPTS.jsonl.gz",
        archive,
    )
    review_documents = {
        "AUTHORITY_INDEX.json": {
            "schema": "substrate-v5-review-authority-index/v1",
            "authorities": authorities,
            "activation": False,
        },
        "RAW_RECEIPT_INDEX.json": {
            "schema": "substrate-v5-raw-receipt-index/v1",
            "archive": archive_path.name,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "files": raw_files,
            "unit_receipts": raw_report["valid"],
            "checkpoints": len(raw_report["checkpoints"]),
            "activation": False,
        },
        "EFFECT_LEDGER.json": {
            "schema": "substrate-v5-effect-ledger/v1",
            "principal": verification["effects"],
            "replication": verification["replication"]["effects"],
            "open_world": verification["open_world"]["effects"],
            "costs": verification["costs"],
            "activation": False,
        },
        "IDENTITY_AND_LEARNING_LEDGER.json": {
            "schema": "substrate-v5-identity-learning-ledger/v1",
            "continuity": verification["continuity"],
            "mechanisms": verification["metrics"],
            "activation": False,
        },
        "CONTROL_AUDIT.json": {
            "schema": "substrate-v5-control-audit/v1",
            "controls": {
                name: {
                    "declared": row.get("controls", []),
                    "strongest_by_history": row.get(
                        "strongest_control_by_history",
                        [],
                    ),
                    "passes": row["passes"],
                }
                for name, row in verification["effects"].items()
            },
            "activation": False,
        },
        "MUTATION_REPORT.json": dict(mutation),
        "INDEPENDENT_VERIFICATION.json": dict(verification),
        "CLEAN_CLONE.json": dict(clone),
        "CLAIM_BOUNDARY.json": {
            "schema": "substrate-v5-review-claim-boundary/v1",
            **C.CLAIM_BOUNDARY,
        },
        "KNOWN_LIMITATIONS.json": {
            "schema": "substrate-v5-known-limitations/v1",
            "limitations": [
                "Principal evidence is from deterministic sandbox multimodal developmental environments.",
                "Model-equivalent modules do not establish general real-world multimodal competence.",
                "No classification is a claim of consciousness, sentience, personhood, life, or moral status.",
                "The maximum automatic result is multimodal Nous-ready-for-review.",
                "External activation remains false.",
            ],
            "activation": False,
        },
        "STRONGEST_FALSIFICATION.json": {
            "schema": "substrate-v5-strongest-falsification/v1",
            "historical_v4_replication_effect": 0.031,
            "historical_v4_replication_sesoi": 0.05,
            "current_nulls": {split: [name for name, row in verification["splits"][split]["effects"].items() if not row["passes"]] for split in P.SPLIT_SEEDS},
            "mutation_survivors": mutation["survived"],
            "activation": False,
        },
    }
    for name, document in review_documents.items():
        _write_review_json(review_root, name, document)
    io.atomic_write(
        review_root / "REPRODUCTION.md",
        "# Substrate v5 reproduction\n\n"
        "1. Check out the frozen V5 ready or terminal tag with prior tags available.\n"
        "2. Validate `RAW_RECEIPT_INDEX.json` and decompress `RAW_RECEIPTS.jsonl.gz`.\n"
        "3. Install with `python -m pip install '.[dev]'`.\n"
        "4. Run the test suite, Ruff, `substrate v5 status`, and `substrate v5 verify`.\n"
        "5. Regenerate twice and compare normalized canonical digests.\n\n"
        "Activation must remain `false`; review readiness is not an unqualified Nous claim.\n",
    )
    io.atomic_write(
        review_root / "README.md",
        "# Substrate v5 multimodal review package\n\n"
        "This package indexes authorities, raw sealed receipts and checkpoints, independently recomputed "
        "effects and costs, continuity and learning evidence, controls, mutations, reproduction results, "
        "limitations, and the strongest falsification evidence. Activation remains `false`.\n",
    )
    required = set(review_documents) | {
        "RAW_RECEIPTS.jsonl.gz",
        "REPRODUCTION.md",
        "README.md",
    }
    package_files = {
        path.relative_to(review_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(review_root.iterdir())
        if path.is_file() and path.name != "REVIEW_INDEX.json"
    }
    index = {
        "schema": "substrate-v5-review-package/v1",
        "evidence_and_configuration": authorities,
        "package_files": package_files,
        "raw_receipts": {path: digest for path, digest in raw_files.items() if "/units/" in path},
        "raw_receipt_count": raw_report["valid"],
        "checkpoint_count": len(raw_report["checkpoints"]),
        "required_files": sorted(required),
        "missing_files": sorted(required - set(package_files)),
        "complete": required <= set(package_files) and raw_report["valid"] == raw_report["expected"],
        "claim_boundary": C.CLAIM_BOUNDARY,
        "activation": False,
    }
    _write_review_json(review_root, "REVIEW_INDEX.json", index)
    return index


def finalize(
    raw_report: Mapping[str, Any],
    verification: Mapping[str, Any],
    mutation: Mapping[str, Any],
    clone: Mapping[str, Any],
    *,
    publish: bool = False,
    terminal_refusal: bool = False,
) -> dict[str, Any]:
    """Classify evidence and optionally publish the review/final authorities."""

    provisional = classify(
        raw_report,
        verification,
        mutation,
        clone,
        review_complete=False,
    )
    review: dict[str, Any] = {
        "complete": False,
        "published": False,
        "activation": False,
    }
    if publish and _RANK[provisional["classification"]] >= _RANK["persistent_embodied_proto_nous_candidate"]:
        review = review_package(raw_report, verification, mutation, clone)
        review["published"] = True
    final = classify(
        raw_report,
        verification,
        mutation,
        clone,
        review_complete=review["complete"],
    )
    unmet = [level for level in C.CLASSIFICATIONS if not final["gates"][level]]
    final_state = {
        "schema": "substrate-v5-final-state/v1",
        "raw_receipts": raw_report["valid"],
        "hypotheses": {name: row["passes"] for name, row in verification["effects"].items()},
        "replication": verification["replication_pass"],
        "open_world": verification["open_world_pass"],
        "mutations_survived": len(mutation["survived"]),
        "clean_clone": clone["all_pass"],
        "classification": final["classification"],
        "strongest_missing_condition": unmet[0] if unmet else None,
        "historical_v4_preservation": verification["historical_v4"]["preserved"],
        "review_package_complete": review["complete"],
        "activation": False,
    }
    review_authority = {
        "schema": "substrate-v5-nous-review-authority/v1",
        "eligible": final["classification"] == "multimodal_nous_ready_for_review",
        "meaning": "eligible for external review only; never an unqualified Nous declaration",
        "unqualified_nous": False,
        "review_package_complete": review["complete"],
        "activation": False,
    }
    refusal = (
        terminal_refusal_authority(
            raw_report,
            verification,
            mutation,
            clone,
        )
        if terminal_refusal
        else {
            "schema": "substrate-v5-terminal-refusal/v1",
            "terminal_refusal": False,
            "independently_verified": False,
            "reason": "terminal refusal was not explicitly requested",
            "nulls": {},
            "null_count": 0,
            "implementation_or_instrument_defect": False,
            "activation": False,
        }
    )
    if publish:
        io.seal("SUBSTRATE_V5_FINAL_CLASSIFICATION.json", final)
        io.seal("SUBSTRATE_V5_NOUS_REVIEW_AUTHORITY.json", review_authority)
        io.seal("SUBSTRATE_V5_FINAL_STATE.json", final_state)
        if terminal_refusal:
            io.seal("SUBSTRATE_V5_TERMINAL_REFUSAL.json", refusal)
    return {
        "classification": final,
        "review": review,
        "review_authority": review_authority,
        "final_state": final_state,
        "terminal_refusal": refusal,
        "activation": False,
    }


def run_all(
    *,
    publish: bool = False,
    clean_clone_commands: Mapping[str, Sequence[str]] | None = None,
    units: Iterable[P.WorkUnit] | None = None,
    terminal_refusal: bool = False,
) -> dict[str, Any]:
    """Run the terminal verifier; write authorities only when ``publish=True``."""

    raw_report = raw(units)
    verification = recompute(raw_report)
    mutation = mutations(raw_report)
    clone = clean_clone(raw_report, commands=clean_clone_commands)
    if publish:
        io.seal("SUBSTRATE_V5_INDEPENDENT_VERIFICATION.json", verification)
        io.seal("SUBSTRATE_V5_MUTATION_REPORT.json", mutation)
        io.seal("SUBSTRATE_V5_CLEAN_CLONE.json", clone)
    final = finalize(
        raw_report,
        verification,
        mutation,
        clone,
        publish=publish,
        terminal_refusal=terminal_refusal,
    )
    result = {
        "raw": {key: value for key, value in raw_report.items() if key not in {"receipts", "checkpoints"}},
        "verification": verification,
        "mutation": mutation,
        "clean_clone": clone,
        "final": final,
        "activation": False,
    }
    result["all_pass"] = _terminal_verification_passed(result)
    return result


def _terminal_verification_passed(result: Mapping[str, Any]) -> bool:
    classification = result["final"]["classification"]["classification"]
    review_required = _RANK[classification] >= _RANK["persistent_embodied_proto_nous_candidate"]
    common = (
        result["raw"]["all_pass"]
        and result["verification"]["all_pass"]
        and result["mutation"]["zero_survived"]
        and result["clean_clone"]["all_pass"]
        and classification in C.CLASSIFICATIONS
        and result["final"]["classification"]["unqualified_nous"] is False
        and result.get("activation") is False
    )
    all_splits_positive = all(result["verification"]["splits"][split]["all_pass"] for split in P.SPLIT_SEEDS)
    positive_terminal = (
        all_splits_positive
        and _RANK[classification] >= _RANK["multimodal_cognitive_substrate"]
        and (not review_required or result["final"]["final_state"]["review_package_complete"])
    )
    refusal = result["final"].get("terminal_refusal", {})
    refusal_terminal = refusal.get("terminal_refusal") is True and refusal.get("independently_verified") is True and int(refusal.get("null_count", 0)) > 0
    return common and (positive_terminal or refusal_terminal)


__all__ = [
    "MUTATION_CLASSES",
    "Refused",
    "classification_gates",
    "classify",
    "clean_clone",
    "finalize",
    "mutations",
    "raw",
    "recompute",
    "review_package",
    "run_all",
    "terminal_refusal_authority",
]
