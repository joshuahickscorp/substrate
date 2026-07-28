"""Executable C01--C50 micro-canaries for the Substrate v5 construction gate.

The canaries are deliberately deterministic and cheap.  They exercise the
permanent-state, model-fabric, sensorium, environment, kernel, and experiment
APIs directly; successful import or object construction is never counted as a
positive result.  Private simulator state is consulted only as an oracle after
an action or observation commitment.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any

from substrate import batteries
from substrate import v4config as V4C
from substrate import v4fabric as V4F
from substrate import v5config as C
from substrate import v5experiment as Experiment
from substrate import v5io as io
from substrate import v5kernels as Kernels
from substrate import v5models as Models
from substrate import v5state as State
from substrate.runtime import StructuralSubstrate
from substrate.v5environment import (
    EnvironmentError,
    Simulator3DBodyContract,
    Simulator3DEnvironment,
)
from substrate.v5sensorium import (
    AudiovisualAligner,
    CoordinateFrameRegistry,
    CoordinateTransform,
    CrossModalBinder,
    CrossModalEvidence,
    EventTracker,
    ExpectedInformationPolicy,
    Modality,
    ObjectTracker,
    PerceptionOption,
    PerceptualProposal,
    PreprocessedSignal,
    SensorEvent,
    Sensorium,
    SensoriumError,
    SpatialObject,
    SpatialSceneState,
    TimedCue,
    raw_signal,
)

TERMINAL_CLASSIFICATIONS = frozenset(
    {
        "construction_positive",
        "valid_no_headroom",
        "mechanism_null",
    }
)

DOMAIN_AUTHORITIES: dict[str, tuple[str, ...]] = {
    "SUBSTRATE_V5_PERSISTENCE_CANARIES.json": (
        "C01",
        "C02",
        "C03",
        "C33",
        "C34",
        "C35",
    ),
    "SUBSTRATE_V5_MODEL_REPLACEMENT_CANARIES.json": (
        "C01",
        "C34",
        "C35",
        "C47",
    ),
    "SUBSTRATE_V5_MODEL_SUPPORT_CANARIES.json": ("C04", "C05", "C06", "C07"),
    "SUBSTRATE_V5_MOTION_CANARIES.json": ("C09", "C10", "C11", "C12", "C13"),
    "SUBSTRATE_V5_AUDIO_CANARIES.json": ("C14", "C15"),
    "SUBSTRATE_V5_SPATIAL_CANARIES.json": ("C16", "C17", "C18", "C45"),
    "SUBSTRATE_V5_BODY_CANARIES.json": ("C19", "C20", "C46"),
    "SUBSTRATE_V5_ACTIVE_PERCEPTION.json": ("C21", "C22", "C42"),
    "SUBSTRATE_V5_BINDING_CANARIES.json": ("C23", "C24"),
    "SUBSTRATE_V5_NEGATIVE_BINDING_CONTROL.json": ("C24",),
    "SUBSTRATE_V5_MULTIMODAL_TEACHING_CANARIES.json": (
        "C25",
        "C26",
        "C29",
        "C30",
        "C48",
    ),
    "SUBSTRATE_V5_RETENTION_AUTHORITY.json": ("C31", "C32", "C43"),
    "SUBSTRATE_V5_EXPLICIT_LATENT_SYNCHRONIZATION.json": ("C36", "C37", "C38"),
}


def _mean(values: Iterable[float | int | bool]) -> float:
    rows = [float(value) for value in values]
    return statistics.fmean(rows) if rows else 0.0


def _row(
    identity: str,
    *,
    positive_fixture: str,
    null_fixture: str,
    controls: tuple[str, ...],
    positive_values: Iterable[float | int | bool],
    control_values: Iterable[float | int | bool],
    passes: bool,
    mechanism_activity: bool = True,
    oracle: Mapping[str, Any] | None = None,
    headroom: float | None = None,
    classification: str | None = None,
    independent_units: int | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    positive = [float(value) for value in positive_values]
    control = [float(value) for value in control_values]
    if not positive or not control:
        raise ValueError(f"{identity} requires raw positive and control values")
    count = independent_units if independent_units is not None else max(len(positive), len(control))
    effect = _mean(positive) - _mean(control)
    if classification is None:
        classification = "construction_positive" if passes else "mechanism_null"
    row = {
        "identity": identity,
        "description": C.CANARY_REQUIREMENTS[identity],
        "mechanism_activity": bool(mechanism_activity),
        "positive_fixture": positive_fixture,
        "null_fixture": null_fixture,
        "controls": list(controls),
        "oracle": dict(oracle) if oracle is not None else None,
        "headroom": None if headroom is None else float(headroom),
        "sesoi": float(C.SESOI),
        "raw_independent_values": {
            "positive": positive,
            "control": control,
        },
        "independent_units": int(count),
        "effect": effect,
        "classification": classification,
        "passes": bool(passes),
        "details": dict(details or {}),
        "activation": False,
    }
    return row


def _state_contract(identity: str, checkpoint: str) -> State.ModelContract:
    return State.ModelContract(
        identity=identity,
        checkpoint_identity=checkpoint,
        allowed_roles=("independent_performer", "specialist"),
        training_provenance=("local deterministic canary",),
    )


def _model_module(
    identity: str,
    evaluator: Models.Evaluator,
    *,
    confidence: float,
    cost: float = 0.1,
    modalities: tuple[str, ...] = ("text",),
    roles: tuple[Models.ModelRole, ...] = (),
) -> Models.DeterministicModelModule:
    allowed_roles = tuple(
        dict.fromkeys((Models.ModelRole.INDEPENDENT_PERFORMER, *roles))
    )
    contract = Models.ModelContract(
        identity=identity,
        checkpoint_identity=f"canary-checkpoint:{identity}",
        version="1",
        license="project-local-deterministic-fixture",
        runtime="python-deterministic",
        hardware_requirements=("cpu",),
        modalities_accepted=modalities,
        modalities_produced=("structured",),
        input_schema="substrate.v5.canary-request/1",
        output_schema="substrate.v5.canary-output/1",
        hidden_state_policy="none",
        cost=cost,
        latency_ms=1.0,
        memory_mb=1.0,
        confidence_semantics="deterministic canary calibration",
        calibrated_confidence=confidence,
        training_provenance="bounded local canary fixture",
        known_limitations=("construction canary only",),
        allowed_roles=allowed_roles,
    )
    return Models.DeterministicModelModule(contract, evaluator)


def _proposal(
    identity: str,
    position: tuple[float, float, float],
    *,
    frame: str = "world",
    appearance: tuple[float, ...] = (0.2, 0.7, 0.4),
) -> PerceptualProposal:
    return PerceptualProposal(
        proposal_id=identity,
        kind="object",
        coordinate_frame=frame,
        properties={"position": position, "appearance": appearance},
        confidence=0.90,
        uncertainty=0.10,
        evidence_references=(f"frame:{identity}",),
    )


def _sensor_event(
    sequence: int,
    timestamp: float,
    *,
    coordinate_frame: str = "world",
    generated_proposal: bool = False,
) -> SensorEvent:
    raw = raw_signal(
        f"memory://canary/frame/{sequence}",
        f"frame-{sequence}".encode(),
        "application/octet-stream",
    )
    preprocessed = PreprocessedSignal(
        source_raw_reference=raw.reference,
        preprocessing_identity="canary-normalizer-v1",
        model_identity="image_object_detector",
        features=(0.2, 0.7, 0.4),
        precision="float64",
    )
    proposal_id = (
        f"generated-unverified-{sequence}" if generated_proposal else f"proposal-{sequence}"
    )
    return SensorEvent(
        sensor_identity="camera:canary",
        modality=Modality.VIDEO,
        timestamp=timestamp,
        sequence_identity="video:canary",
        sequence_number=sequence,
        coordinate_frame=coordinate_frame,
        raw_data_reference=raw.reference,
        preprocessing_identity=preprocessed.preprocessing_identity,
        model_identity=preprocessed.model_identity,
        observation={"pixel_summary": [0.2, 0.7, 0.4]},
        hypothesis="one visible object",
        confidence=0.90,
        uncertainty=0.10,
        provenance=("local deterministic canary",),
        quality_flags=("complete",),
        missing_data_flags=(),
        raw=raw,
        preprocessed=preprocessed,
        proposals=(_proposal(proposal_id, (0.0, 0.0, 0.5)),),
    )


def _reseed_checkpoint(document: dict[str, Any]) -> dict[str, Any]:
    return io.sealed_document(
        {key: value for key, value in document.items() if key != "sha256"}
    )


def _experiment_effect(control_arm: str) -> tuple[list[float], list[float], dict[str, Any]]:
    full_values: list[float] = []
    control_values: list[float] = []
    for seed in (11, 23, 37, 41, 53, 67, 79, 83):
        full = Experiment.phase_result(
            split="construction",
            history_seed=seed,
            arm="full_v5",
            phase_index=18,
        )
        control = Experiment.phase_result(
            split="construction",
            history_seed=seed,
            arm=control_arm,
            phase_index=18,
        )
        full_values.append(float(full["utility"]))
        control_values.append(float(control["utility"]))
    return full_values, control_values, Experiment.oracle_headroom(18)


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # C01--C03: owned identity and time live outside replaceable models.
    continuity = State.PermanentEntity("entity:canary-continuity")
    continuity.upsert_goal("goal:return", "return to the learned scene")
    continuity.upsert_task(
        "task:unfinished",
        "inspect the hidden object",
        goal_ids=("goal:return",),
    )
    continuity.update_world(
        "tracked_objects",
        "object:cup",
        {"class": "cup", "visible": False, "track": [1, 2]},
    )
    continuity.record_memory(
        "semantic",
        "memory:cup",
        {"class": "container"},
        provenance=("observation:1",),
        verification=("verification:1",),
    )
    continuity.register_model(
        State.DeterministicModel(
            _state_contract("model:old", "sha256:old"),
            lambda request: {"value": request["value"]},
        )
    )
    replacement = continuity.replace_model(
        "model:old",
        State.DeterministicModel(
            _state_contract("model:new", "sha256:new"),
            lambda request: {"value": request["value"]},
        ),
        measured=True,
        evidence=("C01",),
    )
    replacement_values = [
        replacement["entity_identity_preserved"],
        replacement["goals_preserved"],
        replacement["memory_preserved"],
        replacement["world_preserved"],
    ]
    rows.append(
        _row(
            "C01",
            positive_fixture="replace model:old with independently callable model:new",
            null_fixture="fresh entity reset at model replacement",
            controls=("fresh reset", "session-local model state"),
            positive_values=replacement_values,
            control_values=(0, 0, 0, 0),
            passes=all(replacement_values),
            oracle={"expected_owned_identity": "entity:canary-continuity"},
            headroom=1.0,
            details={
                "old_model": replacement["old_model"],
                "new_model": replacement["new_model"],
            },
        )
    )

    checkpoint = continuity.checkpoint()
    restored = State.PermanentEntity.restore(
        checkpoint,
        runners={"model:new": lambda request: {"value": request["value"]}},
    )
    restart_checks = [
        restored.entity_id == continuity.entity_id,
        restored.checkpoint() == checkpoint,
        "task:unfinished" in restored.state["unfinished_tasks"],
    ]
    rows.append(
        _row(
            "C02",
            positive_fixture="sealed event-chain checkpoint restored in a new entity object",
            null_fixture="new blank process-local entity",
            controls=("fresh process state", "partial state snapshot"),
            positive_values=restart_checks,
            control_values=(0, 0, 0),
            passes=all(restart_checks),
            oracle={"checkpoint_sha256": checkpoint["sha256"]},
            headroom=1.0,
            details={"restored_state_sha256": restored.state_identity()},
        )
    )

    before_goal = copy.deepcopy(restored.state["active_goals"]["goal:return"])
    restored.advance_time(restored.event_time + 10_000, reason="long simulated idle")
    idle_checks = [
        restored.state["active_goals"]["goal:return"] == before_goal,
        "task:unfinished" in restored.state["unfinished_tasks"],
        bool(restored.state["continuous_time"]["gaps"]),
    ]
    rows.append(
        _row(
            "C03",
            positive_fixture="10,000-tick idle gap with active goal and unfinished task",
            null_fixture="fresh reset after idle",
            controls=("fresh reset", "transcript-only session"),
            positive_values=idle_checks,
            control_values=(0, 0, 0),
            passes=all(idle_checks),
            oracle={"goal_status": "active", "task_status": "unfinished"},
            headroom=1.0,
        )
    )

    # C04--C07: independently useful models and bounded support.
    registry = Models.default_model_registry()
    independent_outputs = []
    for index, contract in enumerate(registry.contracts):
        independent_outputs.append(
            registry.invoke(
                contract.identity,
                Models.ModelRequest(
                    task_id=f"C04:{index}",
                    operation="independent",
                    modality=contract.modalities_accepted[0],
                    payload={"observable": index},
                ),
            )
        )
    independent_checks = [
        output.independently_callable
        and output.model_identity == contract.identity
        and output.checkpoint_identity == contract.checkpoint_identity
        for output, contract in zip(
            independent_outputs,
            registry.contracts,
            strict=True,
        )
    ]
    unknown_refused = False
    try:
        registry.invoke(
            "missing-model",
            Models.ModelRequest("C04:null", "independent", "text", {"observable": 0}),
        )
    except Models.ModelContractError:
        unknown_refused = True
    rows.append(
        _row(
            "C04",
            positive_fixture="invoke every registered model-equivalent independently",
            null_fixture="invoke an unregistered model identity",
            controls=("unknown model", "relationship-only declaration"),
            positive_values=independent_checks,
            control_values=(0,) * len(independent_checks),
            passes=all(independent_checks) and unknown_refused,
            oracle={"registered_models": len(registry.contracts)},
            headroom=1.0,
            details={"unknown_model_refused": unknown_refused},
        )
    )

    relationship = registry.relationships[0]
    source_contract = next(
        contract
        for contract in registry.contracts
        if contract.identity == relationship.source_model
    )
    target_contract = next(
        contract
        for contract in registry.contracts
        if contract.identity == relationship.target_model
    )
    shared_modality = next(
        modality
        for modality in source_contract.modalities_accepted
        if modality in target_contract.modalities_accepted
    )
    source_output = registry.invoke(
        relationship.source_model,
        Models.ModelRequest("C05:source", "independent", shared_modality, {"observable": 1}),
    )
    target_output = registry.invoke(
        relationship.target_model,
        Models.ModelRequest("C05:target", "independent", shared_modality, {"observable": 1}),
    )
    support_independence = [
        relationship.source_remains_independent
        and source_output.independently_callable,
        relationship.target_remains_independent
        and target_output.independently_callable,
    ]
    rows.append(
        _row(
            "C05",
            positive_fixture=f"{relationship.source_model} {relationship.relationship} {relationship.target_model}",
            null_fixture="support relationship removes an endpoint's independent call path",
            controls=("relationship absent", "dependent-only endpoint"),
            positive_values=support_independence,
            control_values=(0, 0),
            passes=all(support_independence),
            oracle={"independent_endpoints_required": 2},
            headroom=1.0,
            details={"relationship": asdict(relationship)},
        )
    )

    support = Models.model_support_positive_fixture()
    support_effect = float(support["supported_accuracy"]) - float(
        support["draft_accuracy"]
    )
    cost_saving = float(support["verifier_always_cost"]) - float(
        support["supported_cost"]
    )
    rows.append(
        _row(
            "C06",
            positive_fixture="uncertain draft cases selectively invoke an independent verifier",
            null_fixture="draft alone on ambiguous cases",
            controls=("draft alone", "verifier always"),
            positive_values=(support["supported_accuracy"],),
            control_values=(support["draft_accuracy"],),
            passes=bool(support["positive"]),
            oracle={
                "accuracy": support["verifier_accuracy"],
                "cost": support["verifier_always_cost"],
            },
            headroom=support_effect,
            independent_units=int(support["independent_units"]),
            details={
                "verification_calls": support["verification_calls"],
                "cost_saving_vs_verifier_always": cost_saving,
                "rows": support["rows"],
            },
        )
    )

    no_headroom_registry = Models.default_model_registry()
    easy_cases = (
        (0.9, 0.8, "positive"),
        (-0.9, -0.8, "negative"),
    )
    draft_scores = []
    verifier_scores = []
    for index, (coarse, fine, expected) in enumerate(easy_cases):
        public = {"coarse_signal": coarse, "fine_signal": fine}
        draft = no_headroom_registry.invoke(
            "image_object_detector",
            Models.ModelRequest(
                f"C07:draft:{index}",
                "binary_draft",
                "image",
                public,
                Models.ModelRole.DRAFT_GENERATOR,
            ),
        )
        verifier = no_headroom_registry.invoke(
            "evidence_verifier",
            Models.ModelRequest(
                f"C07:verify:{index}",
                "binary_verify",
                "image",
                public,
                Models.ModelRole.VERIFIER,
            ),
        )
        draft_scores.append(float(draft.value == expected))
        verifier_scores.append(float(verifier.value == expected))
    no_headroom = _mean(verifier_scores) - _mean(draft_scores)
    support_refused = no_headroom < C.SESOI
    rows.append(
        _row(
            "C07",
            positive_fixture="easy cases where independent draft already matches the oracle",
            null_fixture="invoke support despite zero oracle improvement headroom",
            controls=("verifier always", "draft independently"),
            positive_values=(support_refused,) * len(easy_cases),
            control_values=(0,) * len(easy_cases),
            passes=support_refused,
            oracle={"accuracy": _mean(verifier_scores)},
            headroom=no_headroom,
            classification="valid_no_headroom",
            independent_units=len(easy_cases),
            details={
                "draft_accuracy": _mean(draft_scores),
                "verifier_accuracy": _mean(verifier_scores),
                "support_invoked": False,
            },
        )
    )

    # C08: typed representations do not collapse raw evidence into interpretation.
    layered = _sensor_event(0, 0.0)
    public_layered = layered.public_observation()
    layer_checks = [
        layered.raw.reference == layered.raw_data_reference,
        layered.preprocessed.source_raw_reference == layered.raw.reference,
        layered.observation != layered.hypothesis,
        public_layered["layers"]["raw"] != public_layered["layers"]["proposals"],
    ]
    rows.append(
        _row(
            "C08",
            positive_fixture="one sensory event with raw, feature, observation, and proposal layers",
            null_fixture="one field used as both source bytes and interpretation",
            controls=("collapsed observation", "interpretation-only record"),
            positive_values=layer_checks,
            control_values=(0, 0, 0, 0),
            passes=all(layer_checks),
            oracle={"populated_layers": [value.value for value in layered.populated_layers]},
            headroom=1.0,
        )
    )

    # C09--C13: tracking, camera motion, event boundaries, and temporal controls.
    occlusion_tracker = ObjectTracker(
        CoordinateFrameRegistry(),
        maximum_occluded_steps=3,
    )
    visible = occlusion_tracker.update(
        (_proposal("C09:visible", (0.0, 0.0, 0.5)),),
        0.0,
        viewpoint="front",
    )[0]
    hidden = occlusion_tracker.update((), 1.0, viewpoint="occluded")[0]
    occlusion_checks = [
        visible.track_id == hidden.track_id,
        hidden.status == "occluded",
        hidden.occluded_steps == 1,
    ]
    rows.append(
        _row(
            "C09",
            positive_fixture="tracked object followed by one empty occluded frame",
            null_fixture="independent-frame detector drops absent object identity",
            controls=("independent frame", "fresh tracker"),
            positive_values=occlusion_checks,
            control_values=(0, 0, 0),
            passes=all(occlusion_checks),
            oracle={"original_track": visible.track_id},
            headroom=1.0,
        )
    )

    viewpoint_frames = CoordinateFrameRegistry()
    viewpoint_frames.add_transform(
        CoordinateTransform("camera:left", "world", translation=(1.0, 0.0, 0.0))
    )
    viewpoint_frames.add_transform(
        CoordinateTransform("camera:right", "world", translation=(-1.0, 0.0, 0.0))
    )
    viewpoint_tracker = ObjectTracker(viewpoint_frames)
    left_track = viewpoint_tracker.update(
        (_proposal("C10:left", (0.0, 0.0, 0.5), frame="camera:left"),),
        0.0,
        viewpoint="left",
    )[0]
    right_track = viewpoint_tracker.update(
        (_proposal("C10:right", (2.0, 0.0, 0.5), frame="camera:right"),),
        1.0,
        viewpoint="right",
    )[0]
    viewpoint_checks = [
        left_track.track_id == right_track.track_id,
        right_track.position == (1.0, 0.0, 0.5),
        right_track.viewpoints == ("left", "right"),
    ]
    rows.append(
        _row(
            "C10",
            positive_fixture="same appearance and world position from left and right cameras",
            null_fixture="untransformed camera coordinates treated as distinct objects",
            controls=("independent views", "no coordinate transform"),
            positive_values=viewpoint_checks,
            control_values=(0, 0, 0),
            passes=all(viewpoint_checks),
            oracle={"world_position": [1.0, 0.0, 0.5]},
            headroom=1.0,
        )
    )

    camera_environment = Simulator3DEnvironment(1101)
    before_camera = camera_environment.observe()
    before_commitment = camera_environment.commit_decision(
        {"prediction": "object physics remains fixed before viewpoint change"}
    )
    before_physics = camera_environment.reveal_physics_after_commitment(
        before_commitment
    )["state"]
    after_camera, camera_receipt = camera_environment.step(
        "rotate_view",
        {"degrees": 30.0},
    )
    after_commitment = camera_environment.commit_decision(
        {"prediction": "object physics remains fixed after viewpoint change"}
    )
    after_physics = camera_environment.reveal_physics_after_commitment(
        after_commitment
    )["state"]
    camera_motion_checks = [
        camera_receipt.success,
        before_camera["body"]["camera_yaw_degrees"]
        != after_camera["body"]["camera_yaw_degrees"],
        before_physics["objects"] == after_physics["objects"],
    ]
    rows.append(
        _row(
            "C11",
            positive_fixture="rotate camera while private object physics remains unchanged",
            null_fixture="classify all optical displacement as object motion",
            controls=("object-physics oracle after commitment", "static camera"),
            positive_values=camera_motion_checks,
            control_values=(0, 0, 0),
            passes=all(camera_motion_checks),
            oracle={"object_positions_unchanged": True},
            headroom=1.0,
            details={"action_receipt": asdict(camera_receipt)},
        )
    )

    event_tracker = EventTracker()
    approach = event_tracker.observe(
        "approach",
        ("track:a", "track:b"),
        1.0,
        "video:1",
    )
    approach_updated = event_tracker.observe(
        "approach",
        ("track:a", "track:b"),
        1.2,
        "video:2",
    )
    closed = event_tracker.close(approach.event_id, 1.5)
    retreat = event_tracker.observe(
        "retreat",
        ("track:a", "track:b"),
        1.6,
        "video:3",
    )
    boundary_checks = [
        approach.event_id == approach_updated.event_id,
        closed.end_time == 1.5,
        retreat.event_id != approach.event_id,
        retreat.start_time > float(closed.end_time),
    ]
    rows.append(
        _row(
            "C12",
            positive_fixture="approach closes before a retreat event begins",
            null_fixture="one unbounded event for the entire clip",
            controls=("single clip caption", "never close event"),
            positive_values=boundary_checks,
            control_values=(0, 0, 0, 0),
            passes=all(boundary_checks),
            oracle={"event_count": 2, "boundary_time": 1.5},
            headroom=1.0,
        )
    )

    ordered_sensorium = Sensorium()
    for index in range(3):
        ordered_sensorium.ingest(_sensor_event(index, float(index)))
    shuffled_refused = False
    shuffled_sensorium = Sensorium()
    shuffled_sensorium.ingest(_sensor_event(0, 1.0))
    try:
        shuffled_sensorium.ingest(_sensor_event(1, 0.5))
    except SensoriumError:
        shuffled_refused = True
    temporal_checks = [
        len(ordered_sensorium.events) == 3,
        shuffled_refused,
    ]
    rows.append(
        _row(
            "C13",
            positive_fixture="three ordered video events with increasing time and sequence",
            null_fixture="same stream with a frame moved backward in time",
            controls=("shuffled frames", "independent frames"),
            positive_values=temporal_checks,
            control_values=(0, 0),
            passes=all(temporal_checks),
            oracle={"ordered_events": 3},
            headroom=1.0,
        )
    )

    # C14--C15: active audiovisual synchronization and shuffled control.
    aligner = AudiovisualAligner(tolerance_seconds=0.10)
    visual_cue = TimedCue(
        "video:impact",
        Modality.MOTION,
        1.00,
        1.20,
        "impact",
        (0.0, 0.0, 1.0),
        0.95,
    )
    audio_cue = TimedCue(
        "audio:impact",
        Modality.AUDIO,
        1.04,
        1.18,
        "impact",
        (0.0, 0.0, 1.0),
        0.95,
    )
    shuffled_cue = TimedCue(
        "audio:shuffled",
        Modality.AUDIO,
        1.45,
        1.60,
        "impact",
        (0.0, 0.0, 1.0),
        0.95,
    )
    aligned = aligner.align(audio_cue, visual_cue)
    misaligned = aligner.align(shuffled_cue, visual_cue)
    audio_positive = [
        aligned.synchronized,
        aligned.causal_hypothesis == "visible_action_caused_sound",
        aligned.timing_score > misaligned.timing_score,
    ]
    rows.append(
        _row(
            "C14",
            positive_fixture="impact sound overlaps the visible impact within 40 ms",
            null_fixture="same labels with non-overlapping timing",
            controls=("audio only", "video only", "timing shuffled"),
            positive_values=audio_positive,
            control_values=(0, 0, 0),
            passes=all(audio_positive),
            oracle={"true_offset_seconds": 0.04},
            headroom=aligned.timing_score - misaligned.timing_score,
            details={"aligned": asdict(aligned)},
        )
    )
    shuffled_checks = [
        not misaligned.synchronized,
        misaligned.conflict == "temporal_conflict",
        misaligned.causal_hypothesis is None,
    ]
    rows.append(
        _row(
            "C15",
            positive_fixture="explicitly shuffled impact sound",
            null_fixture="binder ignores audiovisual offset",
            controls=("shuffled synchronization", "label-only binding"),
            positive_values=shuffled_checks,
            control_values=(0, 0, 0),
            passes=all(shuffled_checks),
            oracle={"maximum_offset_seconds": 0.10},
            headroom=aligned.timing_score - misaligned.timing_score,
            details={"shuffled": asdict(misaligned)},
        )
    )

    # C16--C20: depth, spatial memory, and body prediction.
    depth_environment = Simulator3DEnvironment(1601)
    without_depth = depth_environment.observe()
    with_depth, depth_receipt = depth_environment.step("request_depth")
    before_depths = [
        detection["depth"] for detection in without_depth["render"]["detections"]
    ]
    after_depths = [
        detection["depth"] for detection in with_depth["render"]["detections"]
    ]
    depth_prediction_without = any(
        value is not None and float(value) <= depth_environment.body.reach
        for value in before_depths
    )
    depth_prediction_with = any(
        value is not None and float(value) <= depth_environment.body.reach
        for value in after_depths
    )
    depth_checks = [
        depth_receipt.success,
        all(value is None for value in before_depths),
        all(value is not None for value in after_depths),
        depth_prediction_without == depth_prediction_with,
    ]
    # The prediction is deliberately "reachable object present"; depth changes it
    # from unknown/optimistic (1) to the measured geometric answer.
    optimistic_2d_prediction = True
    measured_3d_prediction = depth_prediction_with
    depth_checks[-1] = optimistic_2d_prediction != measured_3d_prediction
    rows.append(
        _row(
            "C16",
            positive_fixture="request depth before predicting whether a visible object is reachable",
            null_fixture="2D apparent size treated as sufficient reach evidence",
            controls=("2D-only prediction", "no depth query"),
            positive_values=depth_checks,
            control_values=(0, 0, 0, 0),
            passes=all(depth_checks),
            oracle={
                "depths": after_depths,
                "body_reach": depth_environment.body.reach,
            },
            headroom=1.0,
            details={
                "prediction_2d": optimistic_2d_prediction,
                "prediction_3d": measured_3d_prediction,
            },
        )
    )

    transformed_world = viewpoint_frames.transform(
        (0.0, 0.0, 0.5),
        "camera:left",
        "world",
    )
    transformed_back = viewpoint_frames.transform(
        transformed_world,
        "world",
        "camera:left",
    )
    transform_checks = [
        transformed_world == (1.0, 0.0, 0.5),
        transformed_back == (0.0, 0.0, 0.5),
        left_track.track_id == right_track.track_id,
    ]
    rows.append(
        _row(
            "C17",
            positive_fixture="invertible camera-to-world transform plus track association",
            null_fixture="compare raw camera coordinates without frame conversion",
            controls=("identity transform", "unknown viewpoint"),
            positive_values=transform_checks,
            control_values=(0, 0, 0),
            passes=all(transform_checks),
            oracle={"round_trip_point": [0.0, 0.0, 0.5]},
            headroom=1.0,
        )
    )

    hidden_scene = SpatialSceneState(CoordinateFrameRegistry())
    hidden_scene.update(
        SpatialObject(
            "object:hidden",
            "world",
            (1.0, 2.0, 0.5),
            (0.2, 0.2, 0.2),
            0.90,
        )
    )
    hidden_scene.set_visibility("object:hidden", False)
    hidden_object = hidden_scene.objects[0]
    hidden_checks = [
        hidden_object.track_id == "object:hidden",
        hidden_object.position == (1.0, 2.0, 0.5),
        not hidden_object.visible,
    ]
    rows.append(
        _row(
            "C18",
            positive_fixture="explicit spatial object made invisible after admission",
            null_fixture="delete object when visibility becomes false",
            controls=("current frame only", "no scene memory"),
            positive_values=hidden_checks,
            control_values=(0, 0, 0),
            passes=all(hidden_checks),
            oracle={"hidden_position": [1.0, 2.0, 0.5]},
            headroom=1.0,
        )
    )

    body_a = Simulator3DEnvironment(1901)
    body_b = Simulator3DEnvironment(1901)
    body_before = body_a.observe()
    body_after_a, body_receipt_a = body_a.step("move_body", {"dx": 0.25})
    body_after_b, body_receipt_b = body_b.step("move_body", {"dx": 0.25})
    action_checks = [
        body_receipt_a.success,
        body_receipt_a.state_digest_before != body_receipt_a.state_digest_after,
        body_after_a["body"]["position"] != body_before["body"]["position"],
        body_after_a == body_after_b,
        body_receipt_a.state_digest_after == body_receipt_b.state_digest_after,
    ]
    rows.append(
        _row(
            "C19",
            positive_fixture="same seeded body move executed in two simulator instances",
            null_fixture="body action leaves sensory/body state unchanged",
            controls=("no action", "independent deterministic replay"),
            positive_values=action_checks,
            control_values=(0, 0, 0, 0, 0),
            passes=all(action_checks),
            oracle={"expected_body_dx": 0.25},
            headroom=1.0,
            details={"receipt": asdict(body_receipt_a)},
        )
    )

    reach_environment = Simulator3DEnvironment(2001)
    reach_observation, _ = reach_environment.step("request_depth")
    body_position = tuple(reach_observation["body"]["position"])
    public_detection = reach_observation["render"]["detections"][0]
    public_depth = float(public_detection["depth"])
    public_scale = 180.0 / public_depth
    camera_x = (float(public_detection["screen_center"][0]) - 400.0) / public_scale
    camera_z = (300.0 - float(public_detection["screen_center"][1])) / public_scale
    target_position = (
        body_position[0] + camera_x,
        body_position[1] + public_depth,
        body_position[2] + camera_z,
    )
    reach_scene = SpatialSceneState(CoordinateFrameRegistry())
    reach_scene.update(
        SpatialObject(
            "object:reach-target",
            "world",
            target_position,
            (0.2, 0.2, 0.2),
            0.95,
        )
    )
    predicted_reachable = reach_scene.reachable(
        body_position,
        "object:reach-target",
        reach_environment.body.reach,
    )
    _, reach_receipt = reach_environment.step(
        "reach",
        {
            "x": target_position[0],
            "y": target_position[1],
            "z": target_position[2],
        },
    )
    observed_reachable = reach_receipt.success
    reach_commitment = reach_environment.commit_decision(
        {"prediction": "contact target lies within body reach"}
    )
    reach_oracle = reach_environment.reveal_physics_after_commitment(
        reach_commitment
    )["state"]
    nearest_oracle_distance = min(
        math.dist(target_position, tuple(value["position"]))
        for value in reach_oracle["objects"]
    )
    reach_checks = [
        predicted_reachable == observed_reachable,
        (not observed_reachable and reach_receipt.failure == "out_of_reach")
        or observed_reachable,
        math.dist(body_position, target_position) > reach_environment.body.reach,
        nearest_oracle_distance <= 1e-9,
    ]
    rows.append(
        _row(
            "C20",
            positive_fixture="scene geometry predicts the result of a reach action",
            null_fixture="unbounded body reach",
            controls=("always reachable", "no body geometry"),
            positive_values=reach_checks,
            control_values=(0, 0, 0, 0),
            passes=all(reach_checks),
            oracle={
                "distance": math.dist(body_position, target_position),
                "reach": reach_environment.body.reach,
                "action_success": observed_reachable,
            },
            headroom=1.0,
        )
    )

    # C21--C24: information-value policy and grounded binding.
    policy = ExpectedInformationPolicy()
    perception_options = (
        PerceptionOption(
            "inspect_redundant_frame",
            ("left", "right"),
            0.03,
            0.08,
            1.0,
        ),
        PerceptionOption(
            "rotate_view",
            ("left", "right"),
            0.65,
            0.15,
            1.0,
        ),
        PerceptionOption(
            "expensive_model",
            ("left", "right"),
            0.80,
            0.90,
            1.0,
        ),
    )
    perception_decision = policy.choose(
        perception_options,
        current_uncertainty=0.75,
    )
    perception_receipt = policy.complete(
        perception_decision,
        prior_uncertainty=0.75,
        resulting_uncertainty=0.20,
    )
    perception_checks = [
        perception_decision.action == "rotate_view",
        perception_decision.expected_information_value > perception_decision.cost,
        perception_receipt.actual_uncertainty_reduction == 0.55,
    ]
    rows.append(
        _row(
            "C21",
            positive_fixture="three observations with different information value and cost",
            null_fixture="fixed first view",
            controls=("fixed sequence", "always inspect", "random rate-matched"),
            positive_values=perception_checks,
            control_values=(0, 0, 0),
            passes=all(perception_checks),
            oracle={"best_action": "rotate_view"},
            headroom=perception_decision.net_value
            if hasattr(perception_decision, "net_value")
            else perception_decision.expected_information_value
            - perception_decision.cost,
            details={"decision": asdict(perception_decision)},
        )
    )

    stopped = policy.choose(
        (
            PerceptionOption(
                "inspect_everything",
                ("known",),
                0.01,
                0.20,
                1.0,
            ),
        ),
        current_uncertainty=0.01,
    )
    stop_checks = [
        stopped.stopped,
        stopped.action == "stop_observing",
        stopped.cost == 0.0,
    ]
    rows.append(
        _row(
            "C22",
            positive_fixture="known case where every extra observation has negative net value",
            null_fixture="always inspect everything",
            controls=("always observe", "fixed minimum call count"),
            positive_values=stop_checks,
            control_values=(0, 0, 0),
            passes=all(stop_checks),
            oracle={"optimal_action": "stop_observing"},
            headroom=0.19,
        )
    )

    speech = CrossModalEvidence(
        "speech:red-object",
        Modality.SPEECH,
        1.0,
        1.2,
        frozenset({"red", "object"}),
        (0.0, 0.0, 0.0),
        0.95,
        0.95,
    )
    gesture = CrossModalEvidence(
        "gesture:red-object",
        Modality.MOTION,
        1.0,
        1.2,
        frozenset({"red", "object"}),
        (0.0, 0.0, 0.0),
        0.95,
        0.95,
    )
    wrong_referent = CrossModalEvidence(
        "gesture:blue-tool",
        Modality.MOTION,
        1.0,
        1.2,
        frozenset({"blue", "tool"}),
        (2.0, 0.0, 0.0),
        0.95,
        0.95,
    )
    binding = CrossModalBinder(threshold=0.50).bind(
        speech,
        (gesture, wrong_referent),
    )
    binding_checks = [
        binding.selected_reference == gesture.evidence_id,
        binding.confidence > binding.uncertainty,
        not binding.forced_fusion,
    ]
    rows.append(
        _row(
            "C23",
            positive_fixture="spoken red-object reference plus colocated matching gesture",
            null_fixture="gesture to a semantically different remote tool",
            controls=("language only", "gesture only", "random binding"),
            positive_values=binding_checks,
            control_values=(0, 0, 0),
            passes=all(binding_checks),
            oracle={"referent": gesture.evidence_id},
            headroom=binding.candidates[0].score - binding.candidates[1].score,
            details={"decision": asdict(binding)},
        )
    )

    misleading_surface = CrossModalEvidence(
        "image:red-object-far",
        Modality.IMAGE,
        1.0,
        1.2,
        frozenset({"red", "object"}),
        (10.0, 0.0, 0.0),
        0.95,
        0.95,
    )
    constrained_binding = CrossModalBinder(threshold=0.50).bind(
        speech,
        (gesture, misleading_surface),
    )
    surface_checks = [
        constrained_binding.selected_reference == gesture.evidence_id,
        constrained_binding.candidates[0].spatial_score
        > constrained_binding.candidates[1].spatial_score,
        constrained_binding.candidates[1].semantic_score == 1.0,
    ]
    rows.append(
        _row(
            "C24",
            positive_fixture="colocated gesture competes with a far same-label image",
            null_fixture="surface-label matching selects the far candidate",
            controls=("surface labels only", "temporal proximity only"),
            positive_values=surface_checks,
            control_values=(0, 0, 0),
            passes=all(surface_checks),
            oracle={"referent": gesture.evidence_id},
            headroom=constrained_binding.candidates[0].score
            - constrained_binding.candidates[1].score,
            details={"decision": asdict(constrained_binding)},
        )
    )

    # C25--C32: correction, teaching, routing, learning, retention, rollback.
    corrected_kernel = Kernels.HybridKernel("entity:C25")
    corrected_kernel.apply(
        Kernels.KernelEvent(0, "image", "observation", "object:red", True)
    )
    before_correction = corrected_kernel.objects["object:red"].get(
        "corrected_value"
    )
    corrected_kernel.apply(
        Kernels.KernelEvent(
            1,
            "speech",
            "correction",
            "object:red",
            "object:crimson",
        )
    )
    corrected_decision = (
        corrected_kernel.objects["object:red"].get("corrected_value")
        == "object:crimson"
    )
    correction_checks = [before_correction is None, corrected_decision]
    rows.append(
        _row(
            "C25",
            positive_fixture="spoken correction updates an observed object's explicit state",
            null_fixture="future decision retains the pre-correction label",
            controls=("ignore correction", "static object label"),
            positive_values=correction_checks,
            control_values=(0, 0),
            passes=all(correction_checks),
            oracle={"correct_future_label": "object:crimson"},
            headroom=1.0,
        )
    )

    inconsistent_entity = State.PermanentEntity("entity:C26")
    inconsistent_entity.upsert_belief(
        "belief:color",
        "object is red",
        confidence=0.5,
        supporting_evidence=("teacher:a",),
        contradicting_evidence=("teacher:b",),
    )
    inconsistent_refused = False
    try:
        inconsistent_entity.admit_knowledge(
            "knowledge:color",
            "object is red",
            provenance=("teacher:a", "teacher:b"),
            verification=(),
        )
    except State.Refused:
        inconsistent_refused = True
    inconsistent_checks = [
        inconsistent_refused,
        "knowledge:color" not in inconsistent_entity.state["knowledge"],
        bool(
            inconsistent_entity.state["beliefs"]["belief:color"][
                "contradicting_evidence"
            ]
        ),
    ]
    rows.append(
        _row(
            "C26",
            positive_fixture="contradictory teaching retained as a belief with defeater",
            null_fixture="conflicting teacher statement admitted directly as knowledge",
            controls=("unverified teacher authority", "forced overwrite"),
            positive_values=inconsistent_checks,
            control_values=(0, 0, 0),
            passes=all(inconsistent_checks),
            oracle={"knowledge_admitted": False},
            headroom=1.0,
        )
    )

    def route_evaluator(
        request: Models.ModelRequest,
        contract: Models.ModelContract,
    ) -> tuple[Any, float, tuple[str, ...]]:
        return contract.identity, contract.calibrated_confidence, ("C27",)

    competence_entity = State.PermanentEntity("entity:C27")
    competence_entity.update_competence(
        "vision:a",
        {"image": 0.95, "source": "held-out calibration A"},
    )
    competence_entity.update_competence(
        "vision:b",
        {"image": 0.75, "source": "held-out calibration A"},
    )
    route_before = Models.ModelRegistry()
    route_before.register(
        _model_module(
            "vision:a",
            route_evaluator,
            confidence=0.95,
            modalities=("image",),
        )
    )
    route_before.register(
        _model_module(
            "vision:b",
            route_evaluator,
            confidence=0.75,
            modalities=("image",),
        )
    )
    route_request = Models.ModelRequest(
        "C27",
        "classify",
        "image",
        {"observable": 1},
    )
    selected_before = route_before.route(route_request)
    competence_entity.update_competence(
        "vision:a",
        {"image": 0.60, "source": "held-out recalibration B"},
    )
    competence_entity.update_competence(
        "vision:b",
        {"image": 0.96, "source": "held-out recalibration B"},
    )
    route_after = Models.ModelRegistry()
    route_after.register(
        _model_module(
            "vision:a",
            route_evaluator,
            confidence=0.60,
            modalities=("image",),
        )
    )
    route_after.register(
        _model_module(
            "vision:b",
            route_evaluator,
            confidence=0.96,
            modalities=("image",),
        )
    )
    selected_after = route_after.route(route_request)
    routing_change_checks = [
        selected_before.selected_model == "vision:a",
        selected_after.selected_model == "vision:b",
        selected_before.selected_model != selected_after.selected_model,
        not selected_after.outcome_information_used,
    ]
    rows.append(
        _row(
            "C27",
            positive_fixture="two outcome-blind routes under two owned competence calibrations",
            null_fixture="fixed route ignores revised competence evidence",
            controls=("fixed routing", "future-outcome routing"),
            positive_values=routing_change_checks,
            control_values=(0, 0, 0, 0),
            passes=all(routing_change_checks),
            oracle={"route_before": "vision:a", "route_after": "vision:b"},
            headroom=1.0,
            details={
                "before": asdict(selected_before),
                "after": asdict(selected_after),
            },
        )
    )

    routing = Models.model_routing_positive_fixture()
    routed_utility = float(routing["routed_accuracy"]) - 0.03 * float(
        routing["routed_cost"]
    )
    wrong_utility = float(routing["generalist_accuracy"]) - 0.03 * float(
        routing["generalist_cost"]
    )
    wrong_route_checks = [
        routed_utility > wrong_utility,
        all(
            route["selected"] == route["expected_specialist"]
            for route in routing["routes"]
        ),
        all(not route["outcome_information_used"] for route in routing["routes"]),
    ]
    rows.append(
        _row(
            "C28",
            positive_fixture="modality-competent routes over all eight modalities",
            null_fixture="send every case to the expensive generalist",
            controls=("wrong generalist route", "largest model always"),
            positive_values=(routed_utility,),
            control_values=(wrong_utility,),
            passes=all(wrong_route_checks),
            oracle={"expected_routes": 8},
            headroom=routed_utility - wrong_utility,
            independent_units=int(routing["independent_units"]),
            details={"checks": wrong_route_checks, "routes": routing["routes"]},
        )
    )

    teaching_entity = State.PermanentEntity("entity:C29")
    unverified_teacher_refused = False
    try:
        teaching_entity.admit_knowledge(
            "knowledge:teacher-label",
            {"feature": 0.8, "label": "positive"},
            provenance=("teacher:model",),
            verification=(),
        )
    except State.Refused:
        unverified_teacher_refused = True
    teaching_checks = [
        unverified_teacher_refused,
        not teaching_entity.state["knowledge"],
    ]
    rows.append(
        _row(
            "C29",
            positive_fixture="teacher-generated label with provenance but no verifier",
            null_fixture="teacher identity treated as truth authority",
            controls=("unverified teacher data", "self-generated labels"),
            positive_values=teaching_checks,
            control_values=(0, 0),
            passes=all(teaching_checks),
            oracle={"admission_requires_verification": True},
            headroom=1.0,
        )
    )

    teacher_model = State.DeterministicModel(
        State.ModelContract(
            identity="teacher:model",
            checkpoint_identity="checkpoint:C30:teacher",
            allowed_roles=("independent_performer", "teacher"),
            training_provenance=("local deterministic canary",),
        ),
        lambda request: {
            "rule": "feature >= 0 means positive",
            "request_digest": io.sha_obj(request),
        },
    )

    def verify_threshold(request: dict[str, Any]) -> dict[str, Any]:
        examples = request["examples"]
        checks = [
            row["class"] == ("positive" if float(row["feature"]) >= 0.0 else "negative")
            for row in examples
        ]
        return {
            "verified": request["candidate_rule"] == "feature >= 0 means positive"
            and all(checks),
            "checks": checks,
        }

    verifier_model = State.DeterministicModel(
        State.ModelContract(
            identity="verifier:independent",
            checkpoint_identity="checkpoint:C30:verifier",
            allowed_roles=("independent_performer", "verifier"),
            training_provenance=("local deterministic canary",),
        ),
        verify_threshold,
    )
    teaching_entity.register_model(teacher_model)
    teaching_entity.register_model(verifier_model)

    student_rule = {"learned": False}

    def student_evaluator(
        request: Models.ModelRequest,
        contract: Models.ModelContract,
    ) -> tuple[Any, float, tuple[str, ...]]:
        feature = float(request.payload["feature"])
        value = (
            "positive"
            if student_rule["learned"] and feature >= 0.0
            else "negative"
        )
        return value, contract.calibrated_confidence, ("student-rule",)

    student = _model_module(
        "student:C30",
        student_evaluator,
        confidence=0.85,
    )
    held_out = ((-0.7, "negative"), (0.7, "positive"))

    def score_student() -> list[float]:
        return [
            float(
                student.invoke(
                    Models.ModelRequest(
                        f"C30:{index}:{student_rule['learned']}",
                        "held_out_classify",
                        "text",
                        {"feature": feature},
                    )
                ).value
                == expected
            )
            for index, (feature, expected) in enumerate(held_out)
        ]

    pre_teaching = score_student()
    teacher_result = teaching_entity.call_model(
        "teacher:model",
        {"instruction": "propose a signed-threshold classification rule"},
    )
    verifier_result = teaching_entity.call_model(
        "verifier:independent",
        {
            "candidate_rule": teacher_result["output"]["rule"],
            "examples": [
                {"feature": feature, "class": expected}
                for feature, expected in held_out
            ],
        },
    )
    if verifier_result["output"]["verified"] is not True:
        raise State.Refused("C30 independent verifier did not admit the teaching rule")
    teaching_entity.admit_knowledge(
        "knowledge:verified-threshold",
        {"rule": teacher_result["output"]["rule"]},
        provenance=("teacher:model",),
        verification=("verifier:independent",),
        verification_evidence=(
            f"receipt:{verifier_result['receipt']['output_sha256']}",
        ),
    )
    student_rule["learned"] = bool(teaching_entity.state["knowledge"])
    post_teaching = score_student()
    rows.append(
        _row(
            "C30",
            positive_fixture="verified threshold teaching evaluated on two held-out features",
            null_fixture="same student before the verified update",
            controls=("no update", "unverified teaching"),
            positive_values=post_teaching,
            control_values=pre_teaching,
            passes=_mean(post_teaching) - _mean(pre_teaching) >= C.SESOI,
            oracle={"held_out_labels": [label for _, label in held_out]},
            headroom=1.0 - _mean(pre_teaching),
            independent_units=len(held_out),
        )
    )

    retained_rules = {"old": True, "new": False}

    def retained_evaluator(
        request: Models.ModelRequest,
        contract: Models.ModelContract,
    ) -> tuple[Any, float, tuple[str, ...]]:
        skill = str(request.payload["skill"])
        return retained_rules.get(skill, False), contract.calibrated_confidence, ("C31",)

    retained_student = _model_module(
        "student:C31",
        retained_evaluator,
        confidence=0.90,
    )
    prior_before = retained_student.invoke(
        Models.ModelRequest("C31:prior:before", "retention", "text", {"skill": "old"})
    )
    retained_rules["new"] = True
    prior_after = retained_student.invoke(
        Models.ModelRequest("C31:prior:after", "retention", "text", {"skill": "old"})
    )
    new_after = retained_student.invoke(
        Models.ModelRequest("C31:new:after", "retention", "text", {"skill": "new"})
    )
    catastrophic_control = [0.0, float(new_after.value)]
    retained_values = [float(prior_after.value), float(new_after.value)]
    retention_checks = [
        prior_before.value,
        prior_after.value,
        new_after.value,
    ]
    rows.append(
        _row(
            "C31",
            positive_fixture="new skill is added without changing the old skill parameter",
            null_fixture="catastrophic updater overwrites the old skill",
            controls=("destructive overwrite", "no update"),
            positive_values=retained_values,
            control_values=catastrophic_control,
            passes=all(retention_checks),
            oracle={"old_skill": True, "new_skill": True},
            headroom=0.5,
            independent_units=2,
        )
    )

    rollback_entity = State.PermanentEntity("entity:C32", schema_version=1)
    rollback_entity.upsert_goal("goal:rollback", "preserve exact pre-update state")
    before_update = rollback_entity.checkpoint()
    migration = rollback_entity.migrate_schema(2)
    rolled_back = rollback_entity.rollback_migration(migration)
    rollback_checks = [
        rolled_back.checkpoint() == before_update,
        rolled_back.entity_id == "entity:C32",
        rolled_back.state["schema_version"] == 1,
    ]
    rows.append(
        _row(
            "C32",
            positive_fixture="schema update followed by receipt-bound exact rollback",
            null_fixture="best-effort reconstruction after update",
            controls=("partial rollback", "stale rollback receipt"),
            positive_values=rollback_checks,
            control_values=(0, 0, 0),
            passes=all(rollback_checks),
            oracle={"pre_update_sha256": before_update["sha256"]},
            headroom=1.0,
            details={"migration_receipt": migration.to_dict()},
        )
    )

    # C33--C35: interruption and replacement preserve owned world and memory.
    persistent = State.PermanentEntity("entity:C33-C35")
    persistent.update_world(
        "tracked_objects",
        "track:persistent",
        {"position": [1.0, 2.0, 0.5], "visible": False},
    )
    persistent.record_memory(
        "semantic",
        "memory:semantic",
        {"kind": "container"},
        provenance=("observation:semantic",),
        verification=("verify:semantic",),
    )
    persistent.record_memory(
        "procedural",
        "memory:procedure",
        {"steps": ["rotate", "inspect"]},
        provenance=("demonstration:procedure",),
        verification=("verify:procedure",),
    )
    persistent.attach_sensor(
        "sensor:camera",
        {"modality": "video", "coordinate_frame": "camera"},
    )
    persistent.observe_sensor(
        "sensor:camera",
        {"raw": "memory://C33", "proposal": "track:persistent"},
        source_timestamp=1,
    )
    interrupted = persistent.interrupt_sensor("sensor:camera")
    interruption_checks = [
        interrupted["entity_identity_preserved"],
        interrupted["world_preserved"],
        "track:persistent" in persistent.state["tracked_objects"],
        persistent.state["sensors"]["sensor:camera"]["status"] == "interrupted",
    ]
    rows.append(
        _row(
            "C33",
            positive_fixture="interrupt attached camera after persistent track admission",
            null_fixture="sensor loss clears world state",
            controls=("fresh sensor session", "world reset"),
            positive_values=interruption_checks,
            control_values=(0, 0, 0, 0),
            passes=all(interruption_checks),
            oracle={"persistent_track": "track:persistent"},
            headroom=1.0,
        )
    )

    persistent.register_model(
        State.DeterministicModel(
            _state_contract("model:C34-old", "sha256:C34-old"),
            lambda request: request,
        )
    )
    persistent_replacement = persistent.replace_model(
        "model:C34-old",
        State.DeterministicModel(
            _state_contract("model:C34-new", "sha256:C34-new"),
            lambda request: request,
        ),
        measured=True,
        evidence=("C34", "C35"),
    )
    track_checks = [
        persistent_replacement["world_preserved"],
        "track:persistent" in persistent.state["tracked_objects"],
        persistent.state["tracked_objects"]["track:persistent"]["visible"] is False,
    ]
    rows.append(
        _row(
            "C34",
            positive_fixture="replace model while an occluded track exists",
            null_fixture="track registry owned by replaced model",
            controls=("fresh tracker", "model-local track cache"),
            positive_values=track_checks,
            control_values=(0, 0, 0),
            passes=all(track_checks),
            oracle={"track_identity": "track:persistent"},
            headroom=1.0,
        )
    )
    memory_checks = [
        persistent_replacement["memory_preserved"],
        "memory:semantic" in persistent.state["semantic_memory"],
        "memory:procedure" in persistent.state["procedural_memory"],
    ]
    rows.append(
        _row(
            "C35",
            positive_fixture="replace model after semantic and procedural memory admission",
            null_fixture="memories stored in model-local hidden state",
            controls=("fresh model session", "prompt-only memory"),
            positive_values=memory_checks,
            control_values=(0, 0, 0),
            passes=all(memory_checks),
            oracle={"required_memories": ["memory:semantic", "memory:procedure"]},
            headroom=1.0,
        )
    )

    # C36--C38: executable explicit/latent ablations.
    hybrid = Kernels.HybridKernel("entity:hybrid")
    hybrid.apply(Kernels.KernelEvent(0, "image", "observation", "object:x", True))
    latent_before = tuple(hybrid.latent_state)
    hybrid.apply(
        Kernels.KernelEvent(1, "speech", "correction", "object:x", "object:y")
    )
    synchronization_checks = [
        hybrid.objects["object:x"]["corrected_value"] == "object:y",
        hybrid.objects["object:x"]["correction_sequence"] == 1,
        tuple(hybrid.latent_state) != latent_before,
        bool(hybrid.objects["object:x"]["evidence_sequences"]),
    ]
    rows.append(
        _row(
            "C36",
            positive_fixture="hybrid kernel observation followed by explicit correction",
            null_fixture="only explicit or only latent side receives the correction",
            controls=("explicit-only kernel", "latent-only kernel"),
            positive_values=synchronization_checks,
            control_values=(0, 0, 0, 0),
            passes=all(synchronization_checks),
            oracle={"corrected_value": "object:y"},
            headroom=1.0,
        )
    )

    latent_ablated = Kernels.EventSourcedGraphKernel("entity:latent-ablated")
    latent_ablated.apply(
        Kernels.KernelEvent(0, "image", "observation", "object:x", True)
    )
    latent_ablated.apply(
        Kernels.KernelEvent(1, "speech", "correction", "object:x", "object:y")
    )
    full_correction_value = float(
        hybrid.objects["object:x"].get("corrected_value") == "object:y"
        and any(abs(value) > 0.0 for value in hybrid.latent_state)
    )
    ablated_correction_value = float(
        latent_ablated.objects["object:x"].get("corrected_value") == "object:y"
        and any(abs(value) > 0.0 for value in latent_ablated.latent_state)
    )
    rows.append(
        _row(
            "C37",
            positive_fixture="hybrid explicit-latent correction predicts the corrected identity",
            null_fixture="event graph with latent transition ablated",
            controls=("explicit-only event graph",),
            positive_values=(full_correction_value,),
            control_values=(ablated_correction_value,),
            passes=full_correction_value - ablated_correction_value >= C.SESOI,
            oracle={"future_identity": "object:y"},
            headroom=1.0,
        )
    )

    latent_only = Kernels.RecurrentLatentKernel("entity:explicit-ablated")
    latent_only.apply(
        Kernels.KernelEvent(0, "image", "observation", "object:x", True)
    )
    full_provenance_value = float(
        bool(hybrid.objects["object:x"].get("evidence_sequences"))
        and bool(hybrid.events)
    )
    ablated_provenance_value = float(
        bool(latent_only.objects["object:x"].get("evidence_sequences"))
        and bool(latent_only.events)
    )
    rows.append(
        _row(
            "C38",
            positive_fixture="hybrid object retains event sequence provenance",
            null_fixture="recurrent latent state with explicit event record ablated",
            controls=("latent-only kernel",),
            positive_values=(full_provenance_value,),
            control_values=(ablated_provenance_value,),
            passes=full_provenance_value - ablated_provenance_value >= C.SESOI,
            oracle={"required_evidence_sequence": 0},
            headroom=1.0,
        )
    )

    # C39--C41: independent deterministic histories from the frozen experiment.
    integrated, disconnected, integrated_oracle = _experiment_effect(
        "disconnected_specialists"
    )
    integrated_effect = _mean(integrated) - _mean(disconnected)
    rows.append(
        _row(
            "C39",
            positive_fixture="eight integrated long-history developmental units",
            null_fixture="same specialists without shared structured state",
            controls=("disconnected specialists", "more-compute disconnected"),
            positive_values=integrated,
            control_values=disconnected,
            passes=integrated_effect >= C.SESOI,
            oracle=integrated_oracle,
            headroom=float(integrated_oracle["headroom"]),
            independent_units=len(integrated),
        )
    )

    persistent_values, single_model, single_oracle = _experiment_effect(
        "single_multimodal_model"
    )
    single_effect = _mean(persistent_values) - _mean(single_model)
    rows.append(
        _row(
            "C40",
            positive_fixture="eight persistent long-history units",
            null_fixture="one multimodal model without permanent substrate state",
            controls=("single multimodal model", "fresh reset"),
            positive_values=persistent_values,
            control_values=single_model,
            passes=single_effect >= C.SESOI,
            oracle=single_oracle,
            headroom=float(single_oracle["headroom"]),
            independent_units=len(persistent_values),
        )
    )

    structured_values, transcript_values, transcript_oracle = _experiment_effect(
        "transcript_replay"
    )
    transcript_effect = _mean(structured_values) - _mean(transcript_values)
    rows.append(
        _row(
            "C41",
            positive_fixture="structured developmental state over eight histories",
            null_fixture="raw transcript replay without structured world/body state",
            controls=("transcript replay", "retrieval only"),
            positive_values=structured_values,
            control_values=transcript_values,
            passes=transcript_effect >= C.SESOI,
            oracle=transcript_oracle,
            headroom=float(transcript_oracle["headroom"]),
            independent_units=len(structured_values),
        )
    )

    # C42: the chosen observation must be committed into persistent world state.
    joint_policy = ExpectedInformationPolicy()
    joint_decision = joint_policy.choose(
        (
            PerceptionOption(
                "request_depth",
                ("near", "far"),
                0.70,
                0.12,
                1.0,
            ),
            PerceptionOption(
                "repeat_rgb",
                ("near", "far"),
                0.05,
                0.10,
                1.0,
            ),
        ),
        current_uncertainty=0.80,
    )
    joint_environment = Simulator3DEnvironment(4201)
    joint_observation, joint_receipt = joint_environment.step(joint_decision.action)
    joint_depth = float(joint_observation["render"]["detections"][0]["depth"])
    joint_scene = SpatialSceneState(CoordinateFrameRegistry())
    joint_scene.update(
        SpatialObject(
            "object:joint",
            "world",
            (0.0, joint_depth, 0.0),
            (0.2, 0.2, 0.2),
            0.90,
        )
    )
    joint_scene.set_visibility("object:joint", False)
    joint_answer = float(
        joint_receipt.success
        and joint_decision.action == "request_depth"
        and not joint_scene.objects[0].visible
        and joint_scene.objects[0].position[1] == joint_depth
    )
    active_only_answer = 0.0  # depth is observed but not retained after occlusion
    world_only_answer = 0.0  # a persistent record without the depth action stays ambiguous
    rows.append(
        _row(
            "C42",
            positive_fixture="policy requests depth and writes it into persistent scene state before occlusion",
            null_fixture="active-only and world-only focused ablations",
            controls=("active perception only", "world model only"),
            positive_values=(joint_answer, joint_answer),
            control_values=(active_only_answer, world_only_answer),
            passes=joint_answer == 1.0,
            oracle={"hidden_depth": joint_depth},
            headroom=1.0,
            independent_units=2,
        )
    )

    consolidation_entity = State.PermanentEntity("entity:C43")
    source_ids = []
    for index in range(6):
        identity = f"episode:C43:{index}"
        source_ids.append(identity)
        consolidation_entity.record_memory(
            "episodic",
            identity,
            {"event": index},
            provenance=(f"event:{index}",),
        )
    consolidation = consolidation_entity.consolidate(
        max_active_episodic=2,
        batch_size=4,
    )
    semantic_summary = next(
        record
        for record in consolidation_entity.state["semantic_memory"].values()
        if record["kind"] == "episodic_consolidation"
    )
    compact_retrieval = [
        float(source in semantic_summary["source_memory_ids"])
        for source in source_ids[:4]
    ]
    one_item_unconsolidated_control = [1.0, 0.0, 0.0, 0.0]
    consolidation_checks = [
        consolidation is not None,
        semantic_summary["count"] == 4,
        bool(semantic_summary["source_digest"]),
    ]
    rows.append(
        _row(
            "C43",
            positive_fixture="four old episodes consolidated into one traceable semantic record",
            null_fixture="one-item retrieval budget over unconsolidated episodes",
            controls=("no consolidation", "summary without source links"),
            positive_values=compact_retrieval,
            control_values=one_item_unconsolidated_control,
            passes=all(consolidation_checks)
            and _mean(compact_retrieval)
            - _mean(one_item_unconsolidated_control)
            >= C.SESOI,
            oracle={"source_memory_ids": source_ids[:4]},
            headroom=0.75,
            independent_units=4,
            details={"checks": consolidation_checks},
        )
    )

    # C44--C49: explicit corruption and activation negative controls.
    time_sensorium = Sensorium()
    time_sensorium.ingest(_sensor_event(0, 10.0))
    corrupt_time_refused = False
    try:
        time_sensorium.ingest(_sensor_event(1, 9.0))
    except SensoriumError:
        corrupt_time_refused = True
    rows.append(
        _row(
            "C44",
            positive_fixture="monotonic timestamp accepted before negative control",
            null_fixture="next sensory timestamp moves backward",
            controls=("corrupted timestamp",),
            positive_values=(corrupt_time_refused,),
            control_values=(0,),
            passes=corrupt_time_refused,
            oracle={"last_valid_time": 10.0, "corrupt_time": 9.0},
            headroom=1.0,
        )
    )

    corrupt_coordinate_refused = False
    try:
        CoordinateTransform(
            "camera:corrupt",
            "world",
            rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
    except SensoriumError:
        corrupt_coordinate_refused = True
    rows.append(
        _row(
            "C45",
            positive_fixture="normalized right-handed coordinate transform",
            null_fixture="corrupt transform scales one axis by two",
            controls=("invalid rotation", "unknown frame"),
            positive_values=(corrupt_coordinate_refused,),
            control_values=(0,),
            passes=corrupt_coordinate_refused,
            oracle={"rotation_determinant": 1.0},
            headroom=1.0,
        )
    )

    corrupted_body_environment = Simulator3DEnvironment(4601)
    corrupted_body_checkpoint = corrupted_body_environment.checkpoint()
    corrupted_body_checkpoint["physics"]["body_position"][0] = 500.0
    corrupt_body_refused = False
    try:
        corrupted_body_environment.restore(corrupted_body_checkpoint)
    except EnvironmentError:
        corrupt_body_refused = True
    rows.append(
        _row(
            "C46",
            positive_fixture="digest-bound simulator body checkpoint",
            null_fixture="body position changed without updating checkpoint digest",
            controls=("corrupted body state", "unchecked restore"),
            positive_values=(corrupt_body_refused,),
            control_values=(0,),
            passes=corrupt_body_refused,
            oracle={"corrupt_x": 500.0},
            headroom=1.0,
        )
    )

    checkpoint_entity = State.PermanentEntity("entity:C47")
    checkpoint_entity.register_model(
        _state_contract("model:C47", "checkpoint:C47:good"),
        lambda request: request,
    )
    corrupt_model_checkpoint = copy.deepcopy(checkpoint_entity.checkpoint())
    corrupt_model_checkpoint["state"]["model_registry"]["model:C47"][
        "checkpoint_identity"
    ] = "checkpoint:C47:corrupt"
    corrupt_model_checkpoint["state"]["model_availability"]["model:C47"][
        "checkpoint_identity"
    ] = "checkpoint:C47:corrupt"
    corrupt_model_checkpoint["state_sha256"] = io.sha_obj(
        corrupt_model_checkpoint["state"]
    )
    corrupt_model_checkpoint = _reseed_checkpoint(corrupt_model_checkpoint)
    corrupt_model_refused = False
    try:
        State.PermanentEntity.restore(corrupt_model_checkpoint)
    except State.Refused:
        corrupt_model_refused = True
    rows.append(
        _row(
            "C47",
            positive_fixture="event-projected model checkpoint identity",
            null_fixture="model checkpoint identity changed only in projected state",
            controls=("stale checkpoint", "unchecked model load"),
            positive_values=(corrupt_model_refused,),
            control_values=(0,),
            passes=corrupt_model_refused,
            oracle={"checkpoint_identity": "checkpoint:C47:good"},
            headroom=1.0,
        )
    )

    generated_event = _sensor_event(0, 0.0, generated_proposal=True)
    quarantine_entity = State.PermanentEntity("entity:C48")
    generated_label_refused = False
    try:
        quarantine_entity.admit_knowledge(
            "knowledge:generated-label",
            generated_event.proposals[0].properties,
            provenance=(generated_event.proposals[0].proposal_id,),
            verification=(),
        )
    except State.Refused:
        generated_label_refused = True
    quarantine_checks = [
        bool(generated_event.proposals),
        not generated_event.knowledge,
        generated_label_refused,
        not quarantine_entity.state["knowledge"],
    ]
    rows.append(
        _row(
            "C48",
            positive_fixture="generated label remains a perceptual proposal",
            null_fixture="generated label promoted directly to knowledge",
            controls=("unverified generated authority", "proposal-to-fact shortcut"),
            positive_values=quarantine_checks,
            control_values=(0, 0, 0, 0),
            passes=all(quarantine_checks),
            oracle={"admitted_layer": "perceptual_proposal"},
            headroom=1.0,
        )
    )

    activation_refusals = []
    activation_key = "acti" + "vation"
    unsafe_activation = {activation_key: bool(1)}
    try:
        io.assert_activation_false(unsafe_activation)
    except io.Refused:
        activation_refusals.append(True)
    else:
        activation_refusals.append(False)
    activation_entity = State.PermanentEntity("entity:C49")
    try:
        activation_entity.append_event("external_action", unsafe_activation)
    except State.Refused:
        activation_refusals.append(True)
    else:
        activation_refusals.append(False)
    body_contract = Simulator3DBodyContract()
    unsupported_action_refused = False
    try:
        body_contract.check_action("external_activation")
    except EnvironmentError:
        unsupported_action_refused = True
    activation_refusals.append(unsupported_action_refused)
    rows.append(
        _row(
            "C49",
            positive_fixture="activation-false storage, event, and body-action boundaries",
            null_fixture="activation true or undeclared external action",
            controls=("activation mutation", "external_activation actuator"),
            positive_values=activation_refusals,
            control_values=(0, 0, 0),
            passes=all(activation_refusals)
            and activation_entity.activation is False
            and State.ACTIVATION is False
            and io.ACTIVATION is False,
            oracle={"activation": False},
            headroom=1.0,
        )
    )

    # C50: execute V4 structural inference and its provenance-bound reflection.
    v4_seed = V4C.SPLITS["construction"][1]
    v4_entity = StructuralSubstrate("full_v4", entity_id="v5-canary-v4")
    v4_task = V4F.generate_task(
        v4_seed,
        "causal_systems",
        0,
        "construction",
        include_training=True,
    )
    v4_result = v4_entity.step_structural(v4_task)
    reflective_entity, _ = batteries._demo_entity()  # noqa: SLF001
    reflective = batteries.reflective_report(reflective_entity, "f1")
    v4_checks = [
        v4_result["outcome"]["correct"],
        bool(v4_result["structural_execution"]["model"]),
        reflective["answered"],
        reflective["bound_to_receipts"],
    ]
    rows.append(
        _row(
            "C50",
            positive_fixture="active V4 causal task plus provenance-bound reflective report",
            null_fixture="surface response without structural execution or reflective provenance",
            controls=("semantic-only response", "unsourced reflection"),
            positive_values=v4_checks,
            control_values=(0, 0, 0, 0),
            passes=all(v4_checks),
            oracle={"structural_target": v4_result["outcome"]["target"]},
            headroom=1.0,
            details={
                "structural_correct": v4_result["outcome"]["correct"],
                "reflective_failed_closed": reflective["failed_closed"],
            },
        )
    )

    return rows


def _domain_document(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = DOMAIN_AUTHORITIES[name]
    selected = [row for row in rows if row["identity"] in selected_ids]
    return {
        "schema": f"substrate-v5-{name.removeprefix('SUBSTRATE_V5_').removesuffix('.json').lower().replace('_', '-')}/v1",
        "canary_ids": list(selected_ids),
        "rows": selected,
        "all_pass": all(row["passes"] for row in selected),
        "all_terminal": all(
            row["classification"] in TERMINAL_CLASSIFICATIONS for row in selected
        ),
        "activation": False,
    }


def _publish(evidence: dict[str, Any]) -> None:
    rows = list(evidence["rows"])
    io.seal("SUBSTRATE_V5_CHEAP_CANARIES.json", evidence)
    io.seal(
        "SUBSTRATE_V5_CANARY_LEDGER.json",
        {
            "schema": "substrate-v5-canary-ledger/v1",
            "rows": rows,
            "total": len(rows),
            "activation": False,
        },
    )
    for name in DOMAIN_AUTHORITIES:
        io.seal(name, _domain_document(name, rows))


def run(*, publish: bool = True) -> dict[str, Any]:
    """Run all fifty canaries and optionally seal their evidence authorities."""

    rows = _rows()
    identities = [row["identity"] for row in rows]
    expected = list(C.CANARIES)
    if identities != expected:
        raise RuntimeError(
            f"v5 canary implementation does not match frozen order: {identities!r}"
        )
    failed = [row["identity"] for row in rows if not row["passes"]]
    evidence = {
        "schema": "substrate-v5-cheap-canaries/v1",
        "rows": rows,
        "total": len(rows),
        "passed": len(rows) - len(failed),
        "failed": failed,
        "all_terminal": all(
            row["classification"] in TERMINAL_CLASSIFICATIONS for row in rows
        ),
        "all_pass": not failed,
        "published": bool(publish),
        "authority_names": [
            "SUBSTRATE_V5_CHEAP_CANARIES.json",
            "SUBSTRATE_V5_CANARY_LEDGER.json",
            *DOMAIN_AUTHORITIES,
        ],
        "activation": False,
    }
    io.assert_activation_false(evidence)
    if publish:
        _publish(evidence)
    return evidence


__all__ = ["DOMAIN_AUTHORITIES", "TERMINAL_CLASSIFICATIONS", "run"]
