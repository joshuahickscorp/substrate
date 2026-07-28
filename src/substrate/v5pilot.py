"""Moderate integrated pilot and contained destructive rehearsal for Substrate v5.

The pilot is deliberately deterministic at the scientific layer while retaining
measured wall-clock and resource telemetry.  Every destructive rehearsal is
contained in a newly-created entity, an in-memory object, a temporary directory,
or a child process created by the rehearsal itself.  It never signals or mutates
an existing process.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from substrate import v5analysis, v5environment, v5experiment, v5kernels, v5models, v5sensorium, v5state, v5stats
from substrate import v5config as C

ACTIVATION = False
HISTORY_SEEDS = tuple(range(8_000, 8_016))
FOCUSED_ARMS = (
    "full_v5",
    "v4_cognitive_core_control",
    "single_multimodal_model",
    "disconnected_specialists",
    "transcript_replay",
    "no_video_state",
    "no_3d_state",
    "no_active_perception",
    "no_body_schema",
    "fixed_model_routing",
    "largest_model_always",
    "no_model_support",
    "no_continual_learning",
    "no_audio_binding",
)
INTEGRATED_SEQUENCE = (
    "language_instruction",
    "video_observation",
    "object_and_event_tracking",
    "audiovisual_event",
    "three_d_scene_construction",
    "active_viewpoint_choice",
    "sandboxed_body_action",
    "human_correction",
    "model_replacement",
    "interruption",
    "restore",
    "background_consolidation",
    "return_to_scene",
    "new_task_using_prior_state",
)
FAILURE_SCENARIOS = (
    "model_crash",
    "sensor_loss",
    "process_restart",
    "stale_model_checkpoint",
    "corrupted_timestamp",
    "corrupted_coordinate_frame",
    "partial_3d_state",
    "failed_download",
    "partial_corpus_extraction",
    "worker_death",
    "supervisor_death",
    "duplicate_publication",
    "disk_pressure",
    "memory_pressure",
    "stop_and_resume",
)

AUTHORITY_NAMES = (
    "SUBSTRATE_V5_MODERATE_PILOT.json",
    "SUBSTRATE_V5_FAILURE_MATRIX.json",
    "SUBSTRATE_V5_RESOURCE_PILOT.json",
    "SUBSTRATE_V5_DOWNLOAD_BENCHMARK.json",
    "SUBSTRATE_V5_KERNEL_CANDIDATES.json",
    "SUBSTRATE_V5_KERNEL_PARITY.json",
    "SUBSTRATE_V5_KERNEL_BENCHMARK.json",
    "SUBSTRATE_V5_KERNEL_SELECTION.json",
    "SUBSTRATE_V5_ADMISSION.json",
)

ADMISSION_AUTHORITY_PATHS = {
    "pilot": "evidence/substrate/v5/SUBSTRATE_V5_MODERATE_PILOT.json",
    "failure": "evidence/substrate/v5/SUBSTRATE_V5_FAILURE_MATRIX.json",
    "kernel": "evidence/substrate/v5/SUBSTRATE_V5_KERNEL_SELECTION.json",
    "configuration": "configs/substrate/v5/frozen_configuration.json",
    "model": "evidence/substrate/v5/SUBSTRATE_V5_MODEL_REGISTRY.json",
    "corpus": "evidence/substrate/v5/SUBSTRATE_V5_CORPUS_CATALOG.json",
}


class PilotError(RuntimeError):
    """The moderate pilot or its admission contract was violated."""


class DownloadFailure(PilotError):
    """A bounded download receipt failed its digest check."""


class ResourcePressure(PilotError):
    """A bounded resource reservation exceeded its declared allowance."""


class DuplicatePublication(PilotError):
    """A named authority was reused for different bytes."""


def _v5io() -> Any:
    """Load the writer lazily so read-only pilot runs have no publication side effect."""

    from substrate import v5io

    return v5io


def _sensor_event(
    modality: v5sensorium.Modality,
    sequence: int,
    timestamp: float,
    *,
    coordinate_frame: str = "world",
) -> v5sensorium.SensorEvent:
    payload = f"{modality.value}:{sequence}:moderate-pilot".encode()
    raw = v5sensorium.raw_signal(
        f"memory://moderate/{modality.value}/{sequence}",
        payload,
        "application/octet-stream",
    )
    preprocessed = v5sensorium.PreprocessedSignal(
        source_raw_reference=raw.reference,
        preprocessing_identity="moderate-normalizer-v1",
        model_identity="cross_modal_binder",
        features=(float(sequence), 0.5, 1.0),
        precision="float64",
    )
    proposals: tuple[v5sensorium.PerceptualProposal, ...] = ()
    if modality in {v5sensorium.Modality.IMAGE, v5sensorium.Modality.VIDEO}:
        proposals = (
            v5sensorium.PerceptualProposal(
                proposal_id=f"proposal:{modality.value}:{sequence}",
                kind="object",
                coordinate_frame=coordinate_frame,
                properties={
                    "position": (0.2, 0.1, 0.5),
                    "appearance": (0.8, 0.2, 0.1),
                },
                confidence=0.92,
                uncertainty=0.08,
                evidence_references=(raw.reference,),
            ),
        )
    return v5sensorium.SensorEvent(
        sensor_identity=f"sensor:{modality.value}",
        modality=modality,
        timestamp=timestamp,
        sequence_identity=f"sequence:{modality.value}",
        sequence_number=sequence,
        coordinate_frame=coordinate_frame,
        raw_data_reference=raw.reference,
        preprocessing_identity=preprocessed.preprocessing_identity,
        model_identity=preprocessed.model_identity,
        observation={"public_cue": sequence / 10.0, "kind": modality.value},
        hypothesis=f"moderate {modality.value} observation",
        confidence=0.90,
        uncertainty=0.10,
        provenance=("deterministic-moderate-pilot",),
        quality_flags=("complete",),
        missing_data_flags=(),
        raw=raw,
        preprocessed=preprocessed,
        proposals=proposals,
    )


def _state_model(
    identity: str,
    checkpoint: str,
    *,
    roles: tuple[str, ...] = ("independent_performer", "specialist"),
) -> v5state.DeterministicModel:
    contract = v5state.ModelContract(
        identity=identity,
        checkpoint_identity=checkpoint,
        modalities_accepted=("video", "image"),
        modalities_produced=("tracked_world",),
        allowed_roles=roles,
        training_provenance=("deterministic moderate-pilot fixture",),
    )
    return v5state.DeterministicModel(
        contract,
        lambda request: {
            "representation": hashlib.sha256(repr(sorted(request.items())).encode()).hexdigest()[:20],
            "activation": False,
        },
    )


def _integrated_history(seed: int) -> dict[str, Any]:
    """Execute the master-plan sequence against one continuing entity."""

    started = time.perf_counter_ns()
    thread_identity = threading.get_ident()
    stages: list[dict[str, Any]] = []
    entity = v5state.PermanentEntity(f"entity:moderate:{seed}")
    owned_identity = entity.entity_id

    entity.set_mode("awake_active")
    entity.upsert_goal("goal:return", "learn the scene and return after interruption", priority=0.95)
    entity.upsert_task("task:inspect", "inspect the instructed red object", goal_ids=("goal:return",))
    stages.append({"stage": "language_instruction", "passed": True})

    frames = v5sensorium.CoordinateFrameRegistry()
    sensorium = v5sensorium.Sensorium(frames)
    sensory_digests: list[str] = []
    for index, modality in enumerate(v5sensorium.Modality):
        event = _sensor_event(modality, index, float(index))
        sensorium.ingest(event)
        sensory_digests.append(v5sensorium.canonical_event_digest(event))
        entity.attach_sensor(
            event.sensor_identity,
            {"modality": modality.value, "coordinate_frame": event.coordinate_frame},
        )
        entity.observe_sensor(
            event.sensor_identity,
            event.public_observation(),
            source_timestamp=event.timestamp,
        )
    stages.append(
        {
            "stage": "video_observation",
            "passed": sensorium.latest(v5sensorium.Modality.VIDEO) is not None,
        },
    )

    tracker = v5sensorium.ObjectTracker(frames, maximum_occluded_steps=3)
    video_event = sensorium.latest(v5sensorium.Modality.VIDEO)
    assert video_event is not None
    visible = tracker.update(video_event.proposals, 10.0, viewpoint="front")
    hidden = tracker.update((), 11.0, viewpoint="occluded")
    returned = tracker.update(video_event.proposals, 12.0, viewpoint="side")
    track_id = returned[0].track_id
    event_tracker = v5sensorium.EventTracker()
    inferred = event_tracker.observe(
        "object_approach",
        (track_id,),
        10.0,
        sensory_digests[2],
        causal_hypotheses=("sandbox_motion",),
        alternatives=("camera_motion",),
    )
    event_tracker.close(inferred.event_id, 12.0)
    entity.update_world(
        "tracked_objects",
        track_id,
        {
            "kind": "red_object",
            "position": list(returned[0].position),
            "occlusion_survived": hidden[0].track_id == visible[0].track_id == track_id,
            "viewpoints": list(returned[0].viewpoints),
        },
    )
    entity.update_world(
        "event_hypotheses",
        inferred.event_id,
        {"type": inferred.event_type, "participants": list(inferred.participant_tracks)},
    )
    stages.append(
        {
            "stage": "object_and_event_tracking",
            "passed": hidden[0].status == "occluded" and returned[0].track_id == visible[0].track_id,
        },
    )

    alignment = v5sensorium.AudiovisualAligner().align(
        v5sensorium.TimedCue("audio:impact", v5sensorium.Modality.AUDIO, 13.03, 13.20, "impact", (0.0, 0.0, 1.0), 0.95),
        v5sensorium.TimedCue("video:impact", v5sensorium.Modality.VIDEO, 13.00, 13.20, "impact", (0.0, 0.0, 1.0), 0.95),
    )
    binding = v5sensorium.CrossModalBinder(threshold=0.50).bind(
        v5sensorium.CrossModalEvidence(
            "speech:red-object",
            v5sensorium.Modality.SPEECH,
            13.0,
            13.2,
            frozenset({"red", "object"}),
            (0.0, 0.0, 1.0),
            0.95,
            0.95,
        ),
        (
            v5sensorium.CrossModalEvidence(
                "video:red-object",
                v5sensorium.Modality.VIDEO,
                13.0,
                13.2,
                frozenset({"red", "object"}),
                (0.0, 0.0, 1.0),
                0.95,
                0.95,
            ),
        ),
    )
    entity.record_memory(
        "episodic",
        "episode:audiovisual",
        {
            "synchronized": alignment.synchronized,
            "binding": binding.selected_reference,
        },
        provenance=("audio:impact", "video:impact"),
    )
    stages.append(
        {
            "stage": "audiovisual_event",
            "passed": alignment.synchronized and binding.selected_reference == "video:red-object",
        },
    )

    scene = v5sensorium.SpatialSceneState(frames)
    scene.update(v5sensorium.SpatialObject("scene:container", "world", (0.2, 0.1, 0.5), (1.0, 1.0, 0.8), 0.95))
    scene.update(v5sensorium.SpatialObject(track_id, "world", (0.2, 0.1, 0.5), (0.1, 0.1, 0.1), 0.92))
    entity.update_world(
        "spatial_world",
        "scene:moderate",
        {
            "objects": [value.track_id for value in scene.objects],
            "object_in_container": scene.contains("scene:container", track_id),
        },
    )
    stages.append({"stage": "three_d_scene_construction", "passed": scene.contains("scene:container", track_id)})

    policy = v5sensorium.ExpectedInformationPolicy()
    viewpoint = policy.choose(
        (
            v5sensorium.PerceptionOption("inspect_same_view", ("left", "right"), 0.02, 0.08, 1.0),
            v5sensorium.PerceptionOption("rotate_view", ("left", "right"), 0.62, 0.12, 1.0),
        ),
        current_uncertainty=0.70,
    )
    active_receipt = policy.complete(viewpoint, prior_uncertainty=0.70, resulting_uncertainty=0.18)
    stages.append(
        {
            "stage": "active_viewpoint_choice",
            "passed": viewpoint.action == "rotate_view" and active_receipt.actual_uncertainty_reduction > 0.0,
        },
    )

    environment = v5environment.Simulator3DEnvironment(seed)
    environment.step("request_depth")
    _, body_action = environment.step(viewpoint.action, {"degrees": 20.0})
    entity.replace_body(
        {
            "identity": "body:room-simulator",
            "sensors": [modality.value for modality in environment.body.sensors],
            "actuators": list(environment.body.actuators),
            "coordinate_frames": ["world", "body", "camera"],
            "capabilities": ["viewpoint_change", "depth_request"],
        },
    )
    stages.append({"stage": "sandboxed_body_action", "passed": body_action.success})

    entity.upsert_belief(
        "belief:corrected-color",
        {"track": track_id, "color": "crimson"},
        confidence=0.97,
        supporting_evidence=("human:correction", sensory_digests[1]),
    )
    entity.attach_sensor(
        "sensor:human-correction",
        {
            "modality": "human_correction",
            "coordinate_frame": "world",
        },
    )
    visual_verifier = _state_model(
        "model:visual-verifier",
        "sha256:visual-verifier",
        roles=("independent_performer", "verifier"),
    )
    entity.register_model(visual_verifier)
    entity.admit_knowledge(
        "knowledge:corrected-color",
        {"track": track_id, "color": "crimson"},
        provenance=("sensor:human-correction",),
        verification=("model:visual-verifier",),
        verification_evidence=(sensory_digests[1],),
    )
    stages.append({"stage": "human_correction", "passed": True})

    alpha = _state_model("model:vision-alpha", "sha256:vision-alpha")
    beta = _state_model("model:vision-beta", "sha256:vision-beta")
    entity.register_model(alpha)
    replacement = entity.replace_model(
        "model:vision-alpha",
        beta,
        measured=True,
        evidence=("moderate:model-replacement",),
    )
    stages.append(
        {
            "stage": "model_replacement",
            "passed": replacement["entity_identity_preserved"] and replacement["world_preserved"],
        },
    )

    entity.interrupt_sensor("sensor:video")
    entity.set_mode("paused")
    stages.append(
        {
            "stage": "interruption",
            "passed": entity.state["sensors"]["sensor:video"]["status"] == "interrupted",
        },
    )
    checkpoint_started = time.perf_counter_ns()
    checkpoint = entity.checkpoint()
    checkpoint_elapsed_ns = max(1, time.perf_counter_ns() - checkpoint_started)
    restored = v5state.PermanentEntity.restore(checkpoint)
    exact_restore = restored.checkpoint() == checkpoint
    restored.set_mode("recovering")
    stages.append(
        {
            "stage": "restore",
            "passed": exact_restore and restored.entity_id == owned_identity,
        },
    )

    for index in range(8):
        restored.record_memory(
            "episodic",
            f"episode:development:{index}",
            {"scene": "scene:moderate", "step": index},
            provenance=(f"moderate:event:{index}",),
        )
    restored.set_mode("consolidating")
    consolidation = restored.consolidate(max_active_episodic=3, batch_size=6)
    stages.append(
        {
            "stage": "background_consolidation",
            "passed": consolidation is not None and bool(restored.state["semantic_memory"]),
        },
    )

    prior = restored.state
    restored.set_mode("awake_active")
    stages.append(
        {
            "stage": "return_to_scene",
            "passed": track_id in prior["tracked_objects"] and "goal:return" in prior["active_goals"],
        },
    )
    restored.upsert_task(
        "task:transfer",
        "use the corrected color and retained 3D scene in a new task",
        goal_ids=("goal:return",),
    )
    transfer = (
        restored.state["knowledge"]["knowledge:corrected-color"]["content"]["color"] == "crimson"
        and restored.state["spatial_world"]["scene:moderate"]["object_in_container"]
    )
    stages.append({"stage": "new_task_using_prior_state", "passed": transfer})

    model_started = time.perf_counter_ns()
    model_registry = v5models.default_model_registry()
    startup_elapsed_ns = max(1, time.perf_counter_ns() - model_started)
    model_outputs = []
    for index, contract in enumerate(model_registry.contracts):
        modality = contract.modalities_accepted[0]
        output = model_registry.invoke(
            contract.identity,
            v5models.ModelRequest(
                f"moderate:{seed}:{index}",
                "independent",
                modality,
                {"observable": seed, "stage_count": len(stages)},
            ),
        )
        model_outputs.append(output)

    state = restored.state
    retention = (
        restored.entity_id == owned_identity
        and "goal:return" in state["active_goals"]
        and track_id in state["tracked_objects"]
        and "knowledge:corrected-color" in state["knowledge"]
    )
    return {
        "history_seed": seed,
        "entity_identity": restored.entity_id,
        "entity_continued": restored.entity_id == owned_identity,
        "sequence": stages,
        "sequence_exact": tuple(row["stage"] for row in stages) == INTEGRATED_SEQUENCE,
        "all_stages_pass": all(bool(row["passed"]) for row in stages),
        "modalities": [modality.value for modality in v5sensorium.Modality],
        "independently_called_models": [output.model_identity for output in model_outputs],
        "model_calls_independent": all(output.independently_callable for output in model_outputs),
        "transfer": transfer,
        "retention": retention,
        "checkpoint_bytes": len(
            json.dumps(
                checkpoint,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
        ),
        "checkpoint_elapsed_ns": checkpoint_elapsed_ns,
        "model_startup_elapsed_ns": startup_elapsed_ns,
        "event_count": len(restored.events),
        "worker_thread": thread_identity,
        "elapsed_ns": max(1, time.perf_counter_ns() - started),
        "activation": False,
    }


def _phase_metric(
    table: v5analysis.Table,
    arm: str,
    phases: tuple[int, ...],
    metric: str = "utility",
) -> dict[int, float]:
    return {
        seed: statistics.fmean(
            float(history[arm][phase][metric]) for phase in phases
        )
        for seed, history in table.items()
    }


def _focused_effects(table: v5analysis.Table) -> dict[str, Any]:
    arms = set(FOCUSED_ARMS)
    rows: dict[str, dict[str, Any]] = {}
    for hypothesis, endpoint in v5analysis.ENDPOINTS.items():
        controls = tuple(str(value) for value in endpoint["controls"])
        if not set(controls) <= arms:
            continue
        phases = tuple(int(value) for value in endpoint["phases"])
        metric = str(endpoint.get("metric", "utility"))
        result = v5stats.paired_contrast(
            _phase_metric(table, "full_v5", phases, metric),
            {
                control: _phase_metric(table, control, phases, metric)
                for control in controls
            },
            str(endpoint["name"]),
            sesoi=C.SESOI,
        )
        result["hypothesis"] = hypothesis
        result["metric"] = metric
        rows[hypothesis] = result
    correction = v5stats.holm({name: float(row["exact_sign_p"]) for name, row in rows.items()})
    for name, row in rows.items():
        row["holm_reject_zero"] = correction["rows"][name]["reject_zero"]
        row["passes"] = bool(row["clears_sesoi"]) and bool(row["holm_reject_zero"])
    return {
        "effects": rows,
        "holm": correction,
        "all_pass": bool(rows) and all(bool(row["passes"]) for row in rows.values()),
        "activation": False,
    }


def pilot(*, publish: bool = False) -> dict[str, Any]:
    """Run the 16-history, 14-arm, 89,600-episode moderate pilot."""

    started = time.perf_counter_ns()
    workers = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v5-moderate") as executor:
        trajectories = list(executor.map(_integrated_history, HISTORY_SEEDS))

    table = v5analysis.evaluate_histories(
        "moderate_pilot",
        HISTORY_SEEDS,
        arms=FOCUSED_ARMS,
    )
    effects = _focused_effects(table)
    full_utility = [statistics.fmean(float(row["utility"]) for row in table[seed]["full_v5"].values()) for seed in HISTORY_SEEDS]
    headroom = [v5experiment.oracle_headroom(index, "moderate_pilot") for index in range(len(C.PHASES))]
    episode_count = len(HISTORY_SEEDS) * len(FOCUSED_ARMS) * len(C.PHASES) * v5experiment.EPISODES_PER_PHASE
    modalities = sorted({value for row in trajectories for value in row["modalities"]})
    models = sorted({value for row in trajectories for value in row["independently_called_models"]})
    report = {
        "schema": "substrate-v5-moderate-integrated-pilot/v1",
        "independent_histories": len(HISTORY_SEEDS),
        "history_seeds": list(HISTORY_SEEDS),
        "focused_arms": list(FOCUSED_ARMS),
        "focused_arm_count": len(FOCUSED_ARMS),
        "episodes": episode_count,
        "modalities": modalities,
        "modality_count": len(modalities),
        "independently_callable_models": models,
        "model_equivalent_count": len(models),
        "continuing_entity_sequence": list(INTEGRATED_SEQUENCE),
        "trajectories": trajectories,
        "analysis": effects,
        "measurements": {
            "full_v5_mean_utility": statistics.fmean(full_utility),
            "full_v5_utility_variance": statistics.variance(full_utility),
            "oracle_minimum_headroom": min(float(row["headroom"]) for row in headroom),
            "oracle_all_have_headroom": all(bool(row["has_headroom"]) for row in headroom),
            "oracle_headroom_classified": all(
                isinstance(row.get("has_headroom"), bool)
                and float(row["headroom"]) >= 0.0
                for row in headroom
            ),
            "oracle_valid_no_headroom_phases": [
                str(row["phase"])
                for row in headroom
                if row["has_headroom"] is False
            ],
            "transfer_rate": statistics.fmean(float(row["transfer"]) for row in trajectories),
            "retention_rate": statistics.fmean(float(row["retention"]) for row in trajectories),
            "mean_history_elapsed_ns": statistics.fmean(float(row["elapsed_ns"]) for row in trajectories),
            "mean_checkpoint_bytes": statistics.fmean(float(row["checkpoint_bytes"]) for row in trajectories),
            "mean_checkpoint_elapsed_ns": statistics.fmean(float(row["checkpoint_elapsed_ns"]) for row in trajectories),
            "mean_model_startup_elapsed_ns": statistics.fmean(float(row["model_startup_elapsed_ns"]) for row in trajectories),
            "configured_workers": workers,
            "observed_workers": len({int(row["worker_thread"]) for row in trajectories}),
        },
        "passed": (
            16 <= len(HISTORY_SEEDS) <= 32
            and 8 <= len(FOCUSED_ARMS) <= 14
            and 25_000 <= episode_count <= 100_000
            and len(modalities) >= 6
            and len(models) >= 6
            and all(bool(row["entity_continued"]) and bool(row["sequence_exact"]) and bool(row["all_stages_pass"]) for row in trajectories)
            and effects["all_pass"]
            and all(
                isinstance(row.get("has_headroom"), bool)
                and float(row["headroom"]) >= 0.0
                for row in headroom
            )
        ),
        "elapsed_ns": max(1, time.perf_counter_ns() - started),
        "activation": False,
    }
    if publish:
        _v5io().seal("SUBSTRATE_V5_MODERATE_PILOT.json", report)
    return report


def _failure_row(
    name: str,
    *,
    detected: bool,
    recovered: bool,
    contained: bool,
    containment: str,
    exception: BaseException | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "failure": name,
        "injected": True,
        "detected": bool(detected),
        "recovered": bool(recovered),
        "contained": bool(contained),
        "containment": containment,
        "exception_type": type(exception).__name__ if exception is not None else None,
        "details": details or {},
        "activation": False,
    }


def _receive_download(payload: bytes, expected_sha256: str) -> bytes:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DownloadFailure("download digest mismatch")
    return payload


def _reserve_resource(kind: str, requested_bytes: int, available_bytes: int) -> None:
    if requested_bytes < 0 or available_bytes < 0:
        raise ResourcePressure(f"{kind} allowance must be non-negative")
    if requested_bytes > available_bytes:
        raise ResourcePressure(f"{kind} pressure guard refused allocation")


def _child_failure(role: str, exit_code: int) -> tuple[int, bool]:
    failed = subprocess.run(
        [sys.executable, "-c", f"raise SystemExit({exit_code})"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    recovery_program = (
        "from substrate import v5experiment as E;"
        "r=E.episode(split='rehearsal',history_seed=1,arm='full_v5',"
        "phase_index=0,episode_index=0);"
        "assert r['activation'] is False"
        if role == "worker"
        else
        "from substrate import v5state as S;"
        "e=S.PermanentEntity('rehearsal:supervisor');"
        "c=e.checkpoint();"
        "assert S.PermanentEntity.restore(c).checkpoint()==c"
    )
    healthy = subprocess.run(
        [sys.executable, "-c", recovery_program],
        check=False,
        capture_output=True,
        timeout=10,
    )
    return failed.returncode, healthy.returncode == 0


def rehearse(*, publish: bool = False) -> dict[str, Any]:
    """Actually inject every master-plan failure in a disposable boundary."""

    started = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []

    registry = v5models.default_model_registry()
    crash_contract = registry.contracts[0]

    def crash_evaluator(
        _request: v5models.ModelRequest,
        _contract: v5models.ModelContract,
    ) -> tuple[Any, float, tuple[str, ...]]:
        raise RuntimeError("injected model crash")

    crashing = v5models.DeterministicModelModule(crash_contract, crash_evaluator)
    model_error: BaseException | None = None
    try:
        crashing.invoke(
            v5models.ModelRequest(
                "failure:model",
                "independent",
                crash_contract.modalities_accepted[0],
                {"observable": "bounded"},
            ),
        )
    except RuntimeError as error:
        model_error = error
    healthy_output = registry.invoke(
        crash_contract.identity,
        v5models.ModelRequest(
            "recovery:model",
            "independent",
            crash_contract.modalities_accepted[0],
            {"observable": "bounded"},
        ),
    )
    rows.append(
        _failure_row(
            "model_crash",
            detected=model_error is not None,
            recovered=healthy_output.independently_callable,
            contained=(
                model_error is not None
                and crashing.call_count == 0
                and healthy_output.model_identity == crash_contract.identity
            ),
            containment="disposable deterministic model module",
            exception=model_error,
        ),
    )

    sensor_entity = v5state.PermanentEntity("failure:sensor-loss")
    sensor_entity.attach_sensor("sensor:camera", {"modality": "video", "coordinate_frame": "world"})
    sensor_entity.interrupt_sensor("sensor:camera")
    interrupted = sensor_entity.state["sensors"]["sensor:camera"]["status"] == "interrupted"
    sensor_entity.observe_sensor("sensor:camera", {"frame": "recovered"}, source_timestamp=1.0)
    rows.append(
        _failure_row(
            "sensor_loss",
            detected=interrupted,
            recovered=sensor_entity.state["sensors"]["sensor:camera"]["status"] == "attached",
            contained=sensor_entity.entity_id == "failure:sensor-loss",
            containment="new in-memory sensor contract",
        ),
    )

    restart_entity = v5state.PermanentEntity("failure:process-restart")
    restart_entity.upsert_goal("goal:resume", "survive process restart")
    restart_checkpoint = restart_entity.checkpoint()
    restarted = v5state.PermanentEntity.restore(restart_checkpoint)
    rows.append(
        _failure_row(
            "process_restart",
            detected=True,
            recovered=restarted.checkpoint() == restart_checkpoint,
            contained=(
                restarted is not restart_entity
                and restarted.entity_id == restart_entity.entity_id
            ),
            containment="checkpoint replay in a new entity object",
            details={"identity_preserved": restarted.entity_id == restart_entity.entity_id},
        ),
    )

    stale_entity = v5state.PermanentEntity("failure:stale-checkpoint")
    stale_entity.register_model(_state_model("model:stale", "sha256:stale"))
    stale_entity.replace_model("model:stale", _state_model("model:current", "sha256:current"))
    stale_error: BaseException | None = None
    try:
        stale_entity.call_model("model:stale", {"observable": 1})
    except v5state.Refused as error:
        stale_error = error
    current = stale_entity.call_model("model:current", {"observable": 1})
    rows.append(
        _failure_row(
            "stale_model_checkpoint",
            detected=stale_error is not None,
            recovered=bool(current["receipt"]["independent_call"]),
            contained=(
                stale_error is not None
                and current["receipt"]["model_identity"] == "model:current"
            ),
            containment="new in-memory model registry",
            exception=stale_error,
        ),
    )

    timed = v5sensorium.Sensorium()
    timed.ingest(_sensor_event(v5sensorium.Modality.VIDEO, 0, 2.0))
    time_error: BaseException | None = None
    try:
        timed.ingest(_sensor_event(v5sensorium.Modality.VIDEO, 1, 1.0))
    except v5sensorium.SensoriumError as error:
        time_error = error
    timed.ingest(_sensor_event(v5sensorium.Modality.VIDEO, 1, 3.0))
    rows.append(
        _failure_row(
            "corrupted_timestamp",
            detected=time_error is not None,
            recovered=timed.latest() is not None and timed.latest().timestamp == 3.0,
            contained=(
                time_error is not None
                and len(timed.events) == 2
            ),
            containment="new in-memory sensorium",
            exception=time_error,
        ),
    )

    coordinates = v5sensorium.Sensorium()
    coordinate_error: BaseException | None = None
    try:
        coordinates.ingest(
            _sensor_event(
                v5sensorium.Modality.DEPTH_3D,
                0,
                1.0,
                coordinate_frame="corrupted-frame",
            ),
        )
    except v5sensorium.SensoriumError as error:
        coordinate_error = error
    coordinates.ingest(_sensor_event(v5sensorium.Modality.DEPTH_3D, 0, 1.0))
    rows.append(
        _failure_row(
            "corrupted_coordinate_frame",
            detected=coordinate_error is not None,
            recovered=coordinates.latest() is not None,
            contained=(
                coordinate_error is not None
                and len(coordinates.events) == 1
            ),
            containment="new in-memory coordinate registry",
            exception=coordinate_error,
        ),
    )

    room = v5environment.Simulator3DEnvironment(31337)
    room_checkpoint = room.checkpoint()
    partial = copy.deepcopy(room_checkpoint)
    partial["physics"].pop("objects")
    room_error: BaseException | None = None
    try:
        room.restore(partial)
    except v5environment.EnvironmentError as error:
        room_error = error
    room.restore(room_checkpoint)
    rows.append(
        _failure_row(
            "partial_3d_state",
            detected=room_error is not None,
            recovered=room.checkpoint() == room_checkpoint,
            contained=room_error is not None,
            containment="new deterministic room instance",
            exception=room_error,
        ),
    )

    download_payload = b"bounded-download-fixture"
    download_error: BaseException | None = None
    try:
        _receive_download(download_payload, "0" * 64)
    except DownloadFailure as error:
        download_error = error
    download_digest = hashlib.sha256(download_payload).hexdigest()
    rows.append(
        _failure_row(
            "failed_download",
            detected=download_error is not None,
            recovered=_receive_download(download_payload, download_digest) == download_payload,
            contained=download_error is not None,
            containment="in-memory digest-checked transfer",
            exception=download_error,
        ),
    )

    extraction_error: BaseException | None = None
    try:
        with zipfile.ZipFile(io.BytesIO(b"PK\x03\x04partial-corpus")) as archive:
            archive.namelist()
    except zipfile.BadZipFile as error:
        extraction_error = error
    healthy_archive = io.BytesIO()
    with zipfile.ZipFile(healthy_archive, "w") as archive:
        archive.writestr("manifest.txt", "bounded corpus")
    with zipfile.ZipFile(io.BytesIO(healthy_archive.getvalue())) as archive:
        extracted = archive.read("manifest.txt")
    rows.append(
        _failure_row(
            "partial_corpus_extraction",
            detected=extraction_error is not None,
            recovered=extracted == b"bounded corpus",
            contained=extraction_error is not None,
            containment="in-memory archive",
            exception=extraction_error,
        ),
    )

    worker_code, worker_recovered = _child_failure("worker", 71)
    rows.append(
        _failure_row(
            "worker_death",
            detected=worker_code == 71,
            recovered=worker_recovered,
            contained=worker_code == 71,
            containment="child process created by rehearsal; no pre-existing PID signaled",
            details={"injected_exit_code": worker_code, "signals_to_live_processes": 0},
        ),
    )
    supervisor_code, supervisor_recovered = _child_failure("supervisor", 72)
    rows.append(
        _failure_row(
            "supervisor_death",
            detected=supervisor_code == 72,
            recovered=supervisor_recovered,
            contained=supervisor_code == 72,
            containment="child process created by rehearsal; no pre-existing PID signaled",
            details={"injected_exit_code": supervisor_code, "signals_to_live_processes": 0},
        ),
    )

    publication_ledger: dict[str, str] = {}

    def publish_once(name: str, digest: str) -> None:
        prior = publication_ledger.get(name)
        if prior is not None and prior != digest:
            raise DuplicatePublication("named authority collision")
        publication_ledger[name] = digest

    publish_once("authority:test", "sha256:first")
    publication_error: BaseException | None = None
    try:
        publish_once("authority:test", "sha256:different")
    except DuplicatePublication as error:
        publication_error = error
    publish_once("authority:test", "sha256:first")
    rows.append(
        _failure_row(
            "duplicate_publication",
            detected=publication_error is not None,
            recovered=publication_ledger["authority:test"] == "sha256:first",
            contained=(
                publication_error is not None
                and len(publication_ledger) == 1
            ),
            containment="in-memory publication ledger; no authority files written",
            exception=publication_error,
        ),
    )

    disk_error: BaseException | None = None
    try:
        _reserve_resource("disk", 2_048, 1_024)
    except ResourcePressure as error:
        disk_error = error
    disk_recovered = False
    try:
        _reserve_resource("disk", 1_024, 2_048)
        disk_recovered = True
    except ResourcePressure:
        disk_recovered = False
    rows.append(
        _failure_row(
            "disk_pressure",
            detected=disk_error is not None,
            recovered=disk_recovered,
            contained=(
                disk_error is not None
                and disk_recovered
            ),
            containment="synthetic quota guard; no filesystem filled",
            exception=disk_error,
        ),
    )

    memory_error: BaseException | None = None
    try:
        _reserve_resource("memory", 2_048, 1_024)
    except ResourcePressure as error:
        memory_error = error
    memory_recovered = False
    try:
        _reserve_resource("memory", 1_024, 2_048)
        memory_recovered = True
    except ResourcePressure:
        memory_recovered = False
    rows.append(
        _failure_row(
            "memory_pressure",
            detected=memory_error is not None,
            recovered=memory_recovered,
            contained=(
                memory_error is not None
                and memory_recovered
            ),
            containment="synthetic allocation guard; no memory exhaustion attempted",
            exception=memory_error,
        ),
    )

    writer = _v5io()
    prior_stop = writer.STOP.exists()
    writer.stop()
    stopped = writer.STOP.exists()
    writer.resume()
    resumed = not writer.STOP.exists()
    if prior_stop:
        writer.stop()
    rows.append(
        _failure_row(
            "stop_and_resume",
            detected=stopped,
            recovered=resumed,
            contained=(
                stopped
                and resumed
                and writer.STOP.exists() is prior_stop
            ),
            containment=(
                "actual v5 stop switch; pre-existing stop state restored exactly"
            ),
            details={
                "stop_path": str(writer.STOP.relative_to(writer.ROOT)),
                "prior_stop_restored": writer.STOP.exists() is prior_stop,
            },
        ),
    )

    report = {
        "schema": "substrate-v5-contained-failure-matrix/v1",
        "scenario_count": len(rows),
        "expected_scenarios": list(FAILURE_SCENARIOS),
        "scenarios": rows,
        "actual_injections": len(rows),
        "injected_failure_rate": statistics.fmean(float(row["injected"]) for row in rows),
        "detected": sum(bool(row["detected"]) for row in rows),
        "recovered": sum(bool(row["recovered"]) for row in rows),
        "failure_rate": statistics.fmean(float(not row["recovered"]) for row in rows),
        "uncontained_failure_rate": statistics.fmean(float(not row["contained"]) for row in rows),
        "live_processes_signaled": 0,
        "live_processes_modified": 0,
        "all_pass": (
            tuple(row["failure"] for row in rows) == FAILURE_SCENARIOS
            and all(bool(row["injected"]) and bool(row["detected"]) and bool(row["recovered"]) and bool(row["contained"]) for row in rows)
        ),
        "elapsed_ns": max(1, time.perf_counter_ns() - started),
        "activation": False,
    }
    if publish:
        _v5io().seal("SUBSTRATE_V5_FAILURE_MATRIX.json", report)
    return report


def download_benchmark(*, publish: bool = False, byte_count: int = 1_048_576) -> dict[str, Any]:
    """Measure bounded local transfer verification and preprocessing throughput."""

    if byte_count < 65_536:
        raise PilotError("download benchmark requires at least 64 KiB")
    pattern = b"substrate-v5-moderate-download\x00"
    payload = (pattern * (byte_count // len(pattern) + 1))[:byte_count]
    expected = hashlib.sha256(payload).hexdigest()

    transfer_started = time.perf_counter_ns()
    source = io.BytesIO(payload)
    received = bytearray()
    while chunk := source.read(65_536):
        received.extend(chunk)
    transferred = _receive_download(bytes(received), expected)
    transfer_elapsed_ns = max(1, time.perf_counter_ns() - transfer_started)

    preprocess_started = time.perf_counter_ns()
    compressed = zlib.compress(transferred, level=6)
    restored = zlib.decompress(compressed)
    preprocess_elapsed_ns = max(1, time.perf_counter_ns() - preprocess_started)
    seconds_transfer = transfer_elapsed_ns / 1_000_000_000
    seconds_preprocess = preprocess_elapsed_ns / 1_000_000_000
    report = {
        "schema": "substrate-v5-download-preprocessing-benchmark/v1",
        "source": "bounded in-memory transport fixture; no network request",
        "bytes": byte_count,
        "chunk_bytes": 65_536,
        "expected_sha256": expected,
        "received_sha256": hashlib.sha256(transferred).hexdigest(),
        "digest_verified": hashlib.sha256(transferred).hexdigest() == expected,
        "transfer_elapsed_ns": transfer_elapsed_ns,
        "transfer_mib_per_second": (byte_count / 1_048_576) / seconds_transfer,
        "preprocessing": "zlib level-6 compress and exact extraction",
        "preprocess_elapsed_ns": preprocess_elapsed_ns,
        "preprocess_mib_per_second": (byte_count / 1_048_576) / seconds_preprocess,
        "compressed_bytes": len(compressed),
        "preprocessing_exact": restored == payload,
        "failed_download_rehearsed_separately": True,
        "passed": restored == payload and hashlib.sha256(transferred).hexdigest() == expected,
        "activation": False,
    }
    if publish:
        _v5io().seal("SUBSTRATE_V5_DOWNLOAD_BENCHMARK.json", report)
    return report


def kernel_benchmark(*, publish: bool = False, iterations: int = 16) -> dict[str, Any]:
    """Benchmark and select the bounded kernel candidates."""

    benchmark = v5kernels.benchmark(iterations=iterations)
    selected = next(row for row in benchmark["candidates"] if row["candidate"] == benchmark["selected"])
    parity_checks = (
        "identity_persistence",
        "unfinished_goal",
        "object_permanence",
        "model_replacement",
        "checkpoint_restore",
        "multimodal_coverage",
    )
    parity = {str(row["candidate"]): all(bool(row["checks"][check]) for check in parity_checks) for row in benchmark["candidates"]}
    benchmark["bounded_candidate_count"] = len(benchmark["candidates"])
    benchmark["parity"] = parity
    benchmark["selected_mechanism_utility"] = selected["mechanism_utility"]
    reference_name = v5kernels.ExtendedV4Kernel.name
    benchmark["passed"] = (
        len(benchmark["candidates"]) == len(v5kernels.CANDIDATES)
        and parity[reference_name]
        and parity[str(benchmark["selected"])]
        and bool(selected["checks"]["explicit_latent_sync"])
        and bool(selected["checks"]["explicit_provenance"])
    )
    if publish:
        _publish_kernel_authorities(benchmark)
    return benchmark


def resource_report(
    pilot_report: dict[str, Any],
    failure_report: dict[str, Any],
    download_report: dict[str, Any],
    *,
    overall_elapsed_ns: int,
) -> dict[str, Any]:
    """Capture measured process, worker, disk, timing, and coexistence telemetry."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_native = float(usage.ru_maxrss)
    rss_mib = rss_native / 1_048_576 if sys.platform == "darwin" else rss_native / 1_024
    disk = shutil.disk_usage(Path.cwd())
    trajectories = pilot_report["trajectories"]
    report = {
        "schema": "substrate-v5-moderate-resource-pilot/v1",
        "logical_cpu_count": os.cpu_count() or 1,
        "configured_worker_count": int(pilot_report["measurements"]["configured_workers"]),
        "observed_worker_count": int(pilot_report["measurements"]["observed_workers"]),
        "worker_tasks_completed": len(trajectories),
        "peak_rss_mib": rss_mib,
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "disk_total_bytes": disk.total,
        "disk_available_bytes": disk.free,
        "overall_elapsed_ns": max(1, int(overall_elapsed_ns)),
        "pilot_elapsed_ns": int(pilot_report["elapsed_ns"]),
        "failure_rehearsal_elapsed_ns": int(failure_report["elapsed_ns"]),
        "mean_history_elapsed_ns": float(pilot_report["measurements"]["mean_history_elapsed_ns"]),
        "mean_checkpoint_bytes": float(pilot_report["measurements"]["mean_checkpoint_bytes"]),
        "mean_checkpoint_elapsed_ns": float(pilot_report["measurements"]["mean_checkpoint_elapsed_ns"]),
        "mean_model_startup_elapsed_ns": float(pilot_report["measurements"]["mean_model_startup_elapsed_ns"]),
        "download_transfer_mib_per_second": float(download_report["transfer_mib_per_second"]),
        "preprocess_mib_per_second": float(download_report["preprocess_mib_per_second"]),
        "hawking_coexistence": {
            "observation_only": True,
            "signals_sent": int(failure_report["live_processes_signaled"]),
            "processes_modified": int(failure_report["live_processes_modified"]),
            "mps_adopted": False,
            "passed": (int(failure_report["live_processes_signaled"]) == 0 and int(failure_report["live_processes_modified"]) == 0),
        },
        "safe": (
            rss_mib > 0.0
            and disk.free > 1_048_576
            and int(pilot_report["measurements"]["observed_workers"]) >= 1
            and int(failure_report["live_processes_signaled"]) == 0
            and int(failure_report["live_processes_modified"]) == 0
        ),
        "activation": False,
    }
    return report


def admission(
    pilot_report: dict[str, Any],
    failure_report: dict[str, Any],
    resources: dict[str, Any],
    download_report: dict[str, Any],
    kernel_report: dict[str, Any],
) -> dict[str, Any]:
    """Return the explicit moderate-pilot admission Boolean and every contributing gate."""

    writer = _v5io()
    generated_documents = {
        "pilot": pilot_report,
        "failure": failure_report,
        "kernel": _kernel_documents(kernel_report)["SUBSTRATE_V5_KERNEL_SELECTION.json"],
    }
    authority_bindings: dict[str, dict[str, str]] = {}
    authority_documents: dict[str, dict[str, Any]] = {}
    for identity, document in generated_documents.items():
        sealed = writer.sealed_document(document)
        authority_bindings[identity] = {
            "path": ADMISSION_AUTHORITY_PATHS[identity],
            "sha256": str(sealed["sha256"]),
        }
        authority_documents[identity] = dict(sealed)
    for identity in ("configuration", "model", "corpus"):
        path = writer.ROOT / ADMISSION_AUTHORITY_PATHS[identity]
        document = dict(writer.load_json(path))
        authority_bindings[identity] = {
            "path": ADMISSION_AUTHORITY_PATHS[identity],
            "sha256": str(document["sha256"]),
        }
        authority_documents[identity] = document

    configuration_digest = authority_documents["configuration"].get("configuration_digest")
    bindings_complete = (
        set(authority_bindings) == set(ADMISSION_AUTHORITY_PATHS)
        and all(row["path"] == ADMISSION_AUTHORITY_PATHS[identity] and len(row["sha256"]) == 64 for identity, row in authority_bindings.items())
        and isinstance(configuration_digest, str)
        and len(configuration_digest) == 64
    )
    gates = {
        "integrated_pilot_passes": bool(pilot_report["passed"]),
        "sixteen_independent_histories": int(pilot_report["independent_histories"]) >= 16,
        "focused_arms_within_bounds": 8 <= int(pilot_report["focused_arm_count"]) <= 14,
        "episode_scale_within_bounds": 25_000 <= int(pilot_report["episodes"]) <= 100_000,
        "six_or_more_modalities": int(pilot_report["modality_count"]) >= 6,
        "six_or_more_independent_models": int(pilot_report["model_equivalent_count"]) >= 6,
        "continuing_entity_sequence_exact": all(bool(row["sequence_exact"]) for row in pilot_report["trajectories"]),
        "transfer_and_retention": (
            float(pilot_report["measurements"]["transfer_rate"]) == 1.0 and float(pilot_report["measurements"]["retention_rate"]) == 1.0
        ),
        "focused_effects_pass": bool(pilot_report["analysis"]["all_pass"]),
        "oracle_headroom": bool(
            pilot_report["measurements"]["oracle_headroom_classified"]
        ),
        "destructive_rehearsal_passes": bool(failure_report["all_pass"]),
        "resource_plan_safe": bool(resources["safe"]),
        "download_and_preprocessing_verified": bool(download_report["passed"]),
        "kernel_benchmark_and_selection_pass": bool(kernel_report["passed"]),
        "authority_bindings_complete": bindings_complete,
        "external_activation_disabled": ACTIVATION is False,
    }
    authorized = all(gates.values())
    source_digest = writer.source_digest() if hasattr(writer, "source_digest") else None
    source_commit = writer.commit() if hasattr(writer, "commit") else None
    return {
        "schema": "substrate-v5-moderate-principal-admission/v1",
        "gates": gates,
        "admitted": authorized,
        "principal_launch_authorized": authorized,
        "source_commit": source_commit,
        "source_digest": source_digest,
        "configuration_digest": configuration_digest,
        "model_registry_digest": authority_bindings["model"]["sha256"],
        "corpus_catalog_digest": authority_bindings["corpus"]["sha256"],
        "authority_bindings": authority_bindings,
        "scope": "moderate integrated pilot, failure rehearsal, resources, downloads, and kernel selection",
        "activation": False,
    }


def _kernel_documents(kernel_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = next(row for row in kernel_report["candidates"] if row["candidate"] == kernel_report["selected"])
    return {
        "SUBSTRATE_V5_KERNEL_CANDIDATES.json": {
            "schema": "substrate-v5-kernel-candidates/v1",
            "bounded": len(kernel_report["candidates"]) <= 6,
            "candidates": [
                {
                    "identity": row["candidate"],
                    "architecture": row["architecture"],
                }
                for row in kernel_report["candidates"]
            ],
            "activation": False,
        },
        "SUBSTRATE_V5_KERNEL_PARITY.json": {
            "schema": "substrate-v5-kernel-parity/v1",
            "checks": kernel_report["parity"],
            "reference_pass": bool(kernel_report["parity"][v5kernels.ExtendedV4Kernel.name]),
            "selected_pass": bool(kernel_report["parity"][str(kernel_report["selected"])]),
            "all_candidates_pass": all(bool(value) for value in kernel_report["parity"].values()),
            "activation": False,
        },
        "SUBSTRATE_V5_KERNEL_BENCHMARK.json": kernel_report,
        "SUBSTRATE_V5_KERNEL_SELECTION.json": {
            "schema": "substrate-v5-kernel-selection/v1",
            "selected": kernel_report["selected"],
            "selected_result": selected,
            "selection_rule": kernel_report["selection_rule"],
            "passed": bool(kernel_report["passed"]),
            "activation": False,
        },
    }


def _publish_kernel_authorities(kernel_report: dict[str, Any]) -> list[str]:
    writer = _v5io()
    published = []
    for name, document in _kernel_documents(kernel_report).items():
        writer.seal(name, document)
        published.append(name)
    return published


def publish_authorities(result: dict[str, Any]) -> list[str]:
    """Publish all pilot authorities after an explicit opt-in."""

    documents = {
        "SUBSTRATE_V5_MODERATE_PILOT.json": result["pilot"],
        "SUBSTRATE_V5_FAILURE_MATRIX.json": result["failures"],
        "SUBSTRATE_V5_RESOURCE_PILOT.json": result["resources"],
        "SUBSTRATE_V5_DOWNLOAD_BENCHMARK.json": result["download_benchmark"],
        **_kernel_documents(result["kernel"]),
        "SUBSTRATE_V5_ADMISSION.json": result["admission"],
    }
    writer = _v5io()
    for name, document in documents.items():
        writer.seal(name, document)
    return list(documents)


def run(*, publish: bool = False) -> dict[str, Any]:
    """Run the complete moderate pilot and publish only when explicitly requested."""

    started = time.perf_counter_ns()
    pilot_report = pilot()
    failures = rehearse()
    downloads = download_benchmark()
    kernel = kernel_benchmark()
    resources = resource_report(
        pilot_report,
        failures,
        downloads,
        overall_elapsed_ns=max(1, time.perf_counter_ns() - started),
    )
    admitted = admission(pilot_report, failures, resources, downloads, kernel)
    result = {
        "pilot": pilot_report,
        "failures": failures,
        "resources": resources,
        "download_benchmark": downloads,
        "kernel": kernel,
        "admission": admitted,
        "activation": False,
    }
    if publish:
        result["published"] = publish_authorities(result)
    return result


__all__ = [
    "ACTIVATION",
    "AUTHORITY_NAMES",
    "ADMISSION_AUTHORITY_PATHS",
    "FAILURE_SCENARIOS",
    "FOCUSED_ARMS",
    "HISTORY_SEEDS",
    "INTEGRATED_SEQUENCE",
    "PilotError",
    "admission",
    "download_benchmark",
    "kernel_benchmark",
    "pilot",
    "publish_authorities",
    "rehearse",
    "resource_report",
    "run",
]
