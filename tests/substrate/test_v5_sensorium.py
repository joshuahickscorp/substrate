from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from substrate.v5environment import (
    DesktopBodyContract,
    DesktopEnvironment,
    EnvironmentError,
    Simulator3DBodyContract,
    Simulator3DEnvironment,
    deterministic_environment_fixture,
)
from substrate.v5models import (
    ALL_MODALITIES,
    ModelContractError,
    ModelRequest,
    ModelRole,
    default_model_registry,
    model_routing_positive_fixture,
    model_support_positive_fixture,
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
    RepresentationLayer,
    SensorEvent,
    Sensorium,
    SensoriumError,
    SpatialObject,
    SpatialSceneState,
    TimedCue,
    canonical_event_digest,
    raw_signal,
)


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


def _event(sequence_number: int, timestamp: float, *, observation: dict | None = None) -> SensorEvent:
    raw = raw_signal(f"memory://frame/{sequence_number}", f"frame-{sequence_number}".encode(), "application/octet-stream")
    preprocessed = PreprocessedSignal(
        source_raw_reference=raw.reference,
        preprocessing_identity="fixture-normalizer-v1",
        model_identity="image_object_detector",
        features=(0.2, 0.7, 0.4),
        precision="float64",
    )
    proposal = _proposal(f"proposal-{sequence_number}", (float(sequence_number), 0.0, 0.5))
    return SensorEvent(
        sensor_identity="camera-primary",
        modality=Modality.IMAGE,
        timestamp=timestamp,
        sequence_identity="camera-sequence",
        sequence_number=sequence_number,
        coordinate_frame="world",
        raw_data_reference=raw.reference,
        preprocessing_identity=preprocessed.preprocessing_identity,
        model_identity=preprocessed.model_identity,
        observation=observation or {"pixel_summary": [0.2, 0.7, 0.4]},
        hypothesis="one visible object",
        confidence=0.90,
        uncertainty=0.10,
        provenance=("local-deterministic-fixture",),
        quality_flags=("complete",),
        missing_data_flags=(),
        raw=raw,
        preprocessed=preprocessed,
        proposals=(proposal,),
    )


def test_sensor_event_has_eight_modalities_and_noncollapsed_typed_layers() -> None:
    assert {modality.value for modality in Modality} == set(ALL_MODALITIES)
    event = _event(0, 0.0)
    assert event.populated_layers == (
        RepresentationLayer.RAW_SIGNAL,
        RepresentationLayer.PREPROCESSED_SIGNAL,
        RepresentationLayer.PERCEPTUAL_PROPOSAL,
    )
    assert event.raw is not event.preprocessed
    assert event.preprocessed is not event.proposals[0]
    public = event.public_observation()
    assert set(public["layers"]) == {
        "raw",
        "preprocessed",
        "proposals",
        "tracked",
        "inferred_events",
        "verified",
        "structural",
        "knowledge",
    }
    assert public["layers"]["raw"] == dataclasses.asdict(event.raw)
    assert public["layers"]["preprocessed"] == dataclasses.asdict(event.preprocessed)
    assert canonical_event_digest(event) == canonical_event_digest(event)
    for forbidden_key in ("target", "answer", "outcome", "target_id"):
        with pytest.raises(SensoriumError, match="hidden target"):
            _event(
                1,
                1.0,
                observation={"nested": [{"authority": {forbidden_key: "leaked"}}]},
            )


def test_sensorium_rechecks_all_public_layers_for_recursive_outcome_leakage() -> None:
    event = _event(0, 0.0)
    object.__setattr__(
        event.proposals[0],
        "properties",
        {"nested": [{"answer": "mutated-after-construction"}]},
    )
    with pytest.raises(SensoriumError, match="hidden target"):
        event.public_observation()
    with pytest.raises(SensoriumError, match="hidden target"):
        Sensorium(CoordinateFrameRegistry()).ingest(event)


def test_sensorium_rejects_corrupted_time_sequence_and_coordinate_frames() -> None:
    frames = CoordinateFrameRegistry()
    sensorium = Sensorium(frames)
    sensorium.ingest(_event(0, 1.0))
    with pytest.raises(SensoriumError, match="time"):
        sensorium.ingest(_event(1, 0.5))
    with pytest.raises(SensoriumError, match="sequence"):
        sensorium.ingest(_event(0, 1.5))
    unknown_frame = copy.deepcopy(_event(2, 2.0))
    object.__setattr__(unknown_frame, "coordinate_frame", "unregistered-camera")
    with pytest.raises(SensoriumError, match="unknown coordinate"):
        sensorium.ingest(unknown_frame)


def test_sensorium_ingest_and_digest_reuses_the_validated_public_observation() -> None:
    event = _event(0, 0.0)
    sensorium = Sensorium()

    digest = sensorium.ingest_and_digest(event)

    assert digest == canonical_event_digest(event)
    assert sensorium.events == (event,)


def test_tracking_preserves_object_through_occlusion_and_viewpoint_change() -> None:
    frames = CoordinateFrameRegistry()
    frames.add_transform(CoordinateTransform("camera-left", "world", translation=(1.0, 0.0, 0.0)))
    frames.add_transform(CoordinateTransform("camera-right", "world", translation=(-1.0, 0.0, 0.0)))
    tracker = ObjectTracker(frames, maximum_occluded_steps=3)
    first = tracker.update(
        (_proposal("left-view-proposal", (0.0, 0.0, 0.5), frame="camera-left"),),
        0.0,
        viewpoint="left",
    )
    identity = first[0].track_id
    hidden = tracker.update((), 1.0, viewpoint="occluded")
    assert hidden[0].track_id == identity
    assert hidden[0].status == "occluded"
    returned = tracker.update(
        (_proposal("right-view-proposal", (2.0, 0.0, 0.5), frame="camera-right"),),
        2.0,
        viewpoint="right",
    )
    assert len(returned) == 1
    assert returned[0].track_id == identity
    assert returned[0].status == "visible"
    assert returned[0].viewpoints == ("left", "right")


def test_event_tracking_keeps_evidence_and_unresolved_alternatives() -> None:
    events = EventTracker()
    opened = events.observe(
        "approach",
        ("track-1", "track-2"),
        1.0,
        "video:10",
        alternatives=("near_miss",),
    )
    updated = events.observe(
        "approach",
        ("track-1", "track-2"),
        1.1,
        "motion:11",
        causal_hypotheses=("intentional_motion",),
    )
    closed = events.close(opened.event_id, 1.5)
    assert updated.event_id == opened.event_id
    assert updated.evidence_references == ("video:10", "motion:11")
    assert updated.unresolved_alternatives == ("near_miss",)
    assert closed.end_time == 1.5


def test_audiovisual_timing_is_active_and_shuffling_breaks_alignment() -> None:
    aligner = AudiovisualAligner(tolerance_seconds=0.10)
    visual = TimedCue("video-impact", Modality.MOTION, 1.00, 1.20, "impact", (0.0, 0.0, 1.0), 0.95)
    audio = TimedCue("audio-impact", Modality.AUDIO, 1.04, 1.18, "impact", (0.0, 0.0, 1.0), 0.95)
    shuffled = TimedCue("audio-shuffled", Modality.AUDIO, 1.45, 1.60, "impact", (0.0, 0.0, 1.0), 0.95)
    aligned = aligner.align(audio, visual)
    misaligned = aligner.align(shuffled, visual)
    assert aligned.synchronized
    assert aligned.causal_hypothesis == "visible_action_caused_sound"
    assert not misaligned.synchronized
    assert misaligned.conflict == "temporal_conflict"
    assert aligned.timing_score > misaligned.timing_score


def test_3d_scene_transforms_geometry_and_detects_bad_coordinates() -> None:
    frames = CoordinateFrameRegistry()
    frames.add_transform(CoordinateTransform("camera", "world", translation=(1.0, 2.0, 0.0)))
    scene = SpatialSceneState(frames)
    scene.update(SpatialObject("container", "world", (1.0, 2.0, 1.0), (2.0, 2.0, 1.0), 0.95))
    scene.update(SpatialObject("object", "camera", (0.0, 0.0, 1.0), (0.2, 0.2, 0.2), 0.90))
    assert scene.relative_position("object", "container") == (0.0, 0.0, 0.0)
    assert scene.contains("container", "object")
    assert scene.collides("container", "object")
    scene.set_visibility("object", False)
    assert next(row for row in scene.objects if row.track_id == "object").position == (1.0, 2.0, 1.0)
    with pytest.raises(SensoriumError, match="unknown coordinate"):
        scene.update(SpatialObject("bad", "corrupted", (0.0, 0.0, 0.0), (0.1, 0.1, 0.1), 0.8))
    with pytest.raises(SensoriumError, match="rotation"):
        CoordinateTransform("bad-frame", "world", rotation=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_cross_modal_binding_preserves_conflict_and_avoids_forced_fusion() -> None:
    anchor = CrossModalEvidence("speech", Modality.SPEECH, 1.0, 1.2, frozenset({"red", "object"}), (0.0, 0.0, 0.0), 0.95, 0.95)
    gesture = CrossModalEvidence("gesture", Modality.MOTION, 1.0, 1.2, frozenset({"red", "object"}), (0.0, 0.0, 0.0), 0.95, 0.95)
    conflicting = CrossModalEvidence("audio-label", Modality.AUDIO, 1.0, 1.2, frozenset({"blue", "tool"}), (0.0, 0.0, 0.0), 0.95, 0.95)
    decision = CrossModalBinder(threshold=0.50).bind(anchor, (gesture, conflicting))
    assert decision.selected_reference == "gesture"
    assert any(conflict.conflict_type == "semantic_conflict" and conflict.preserved for conflict in decision.conflicts)
    assert not decision.forced_fusion
    assert {candidate.evidence_reference for candidate in decision.candidates} == {"gesture", "audio-label"}


def test_expected_information_policy_selects_discriminating_view_and_stops_when_wasteful() -> None:
    policy = ExpectedInformationPolicy()
    decision = policy.choose(
        (
            PerceptionOption("inspect_redundant_frame", ("left", "right"), 0.03, 0.08, 1.0),
            PerceptionOption("rotate_view", ("left", "right"), 0.65, 0.15, 1.0),
            PerceptionOption("expensive_model", ("left", "right"), 0.80, 0.90, 1.0),
        ),
        current_uncertainty=0.75,
    )
    assert decision.action == "rotate_view"
    assert decision.expected_information_value > decision.cost
    receipt = policy.complete(decision, prior_uncertainty=0.75, resulting_uncertainty=0.20)
    assert receipt.actual_uncertainty_reduction == pytest.approx(0.55)
    stopped = policy.choose(
        (PerceptionOption("inspect_everything", ("known",), 0.01, 0.20, 1.0),),
        current_uncertainty=0.01,
    )
    assert stopped.stopped
    assert stopped.action == "stop_observing"


def test_body_and_seeded_environment_contracts_are_deterministic_and_identity_safe() -> None:
    desktop, room = DesktopBodyContract(), Simulator3DBodyContract()
    assert {Modality.TEXT, Modality.IMAGE, Modality.BODY_TOOL} <= set(desktop.sensors)
    assert {Modality.VIDEO, Modality.MOTION, Modality.DEPTH_3D, Modality.BODY_TOOL} <= set(room.sensors)
    fixture = deterministic_environment_fixture()
    assert all(fixture.values())
    environment = Simulator3DEnvironment(71)
    initial = environment.observe()
    assert "physical_id" not in json.dumps(initial, sort_keys=True)
    checkpoint = environment.checkpoint()
    environment.step("rotate_view", {"degrees": 90.0})
    environment.restore(checkpoint)
    assert environment.checkpoint() == checkpoint
    corrupted = copy.deepcopy(checkpoint)
    corrupted["physics"]["body_position"][0] = 500.0
    with pytest.raises(EnvironmentError, match="digest"):
        environment.restore(corrupted)
    desktop_environment = DesktopEnvironment(71)
    assert desktop_environment.contract.render_identity != desktop_environment.contract.physics_identity


def test_environment_oracle_requires_an_issued_single_use_prior_commitment() -> None:
    environment = Simulator3DEnvironment(71)
    with pytest.raises(EnvironmentError, match="prior commitment"):
        environment.reveal_physics_after_commitment()
    with pytest.raises(EnvironmentError, match="nonempty decision"):
        environment.commit_decision({})
    with pytest.raises(EnvironmentError, match="hidden target"):
        environment.commit_decision({"nested": {"outcome": "peek"}})

    commitment = environment.commit_decision({"prediction": "three objects remain"})
    environment.step("rotate_view", {"degrees": 15.0})
    oracle = environment.reveal_physics_after_commitment(commitment)
    assert oracle["revealed_after_commitment"] is True
    assert oracle["commitment"]["committed_at_tick"] < environment.checkpoint()["tick"]
    assert oracle["commitment"]["token_sha256"] == commitment.token_sha256
    with pytest.raises(EnvironmentError, match="consumed"):
        environment.reveal_physics_after_commitment(commitment)

    foreign_environment = Simulator3DEnvironment(72)
    foreign_commitment = foreign_environment.commit_decision({"prediction": "foreign"})
    with pytest.raises(EnvironmentError, match="foreign"):
        environment.reveal_physics_after_commitment(foreign_commitment)

    desktop_environment = DesktopEnvironment(71)
    desktop_commitment = desktop_environment.commit_decision({"prediction": "desktop state"})
    desktop_environment.reset()
    with pytest.raises(EnvironmentError, match="invalid"):
        desktop_environment.reveal_physics_after_commitment(desktop_commitment)


def test_at_least_ten_model_equivalents_are_independently_callable_and_auditable() -> None:
    registry = default_model_registry()
    assert len(registry.contracts) >= 10
    assert len({contract.identity for contract in registry.contracts}) == len(registry.contracts)
    for index, contract in enumerate(registry.contracts):
        modality = contract.modalities_accepted[0]
        output = registry.invoke(
            contract.identity,
            ModelRequest(
                f"independent-{index}",
                "independent",
                modality,
                {"observable": index},
            ),
        )
        assert output.independently_callable
        assert output.model_identity == contract.identity
        assert output.checkpoint_identity == contract.checkpoint_identity
        assert output.cost == contract.cost
        assert output.latency_ms == contract.latency_ms
        assert output.confidence > 0.0
        assert ModelRole.INDEPENDENT_PERFORMER in contract.allowed_roles
        assert contract.confidence_semantics
    assert registry.relationships
    for forbidden_key in ("target", "answer", "outcome", "private_target"):
        with pytest.raises(ModelContractError, match="leaked"):
            ModelRequest(
                "bad",
                "independent",
                "text",
                {"nested": [{"authority": {forbidden_key: "secret"}}]},
            )


def test_model_support_and_outcome_blind_routing_have_positive_fixtures() -> None:
    support = model_support_positive_fixture()
    assert support["positive"]
    assert support["supported_accuracy"] > support["draft_accuracy"]
    assert support["supported_accuracy"] == support["verifier_accuracy"]
    assert support["supported_cost"] < support["verifier_always_cost"]
    routing = model_routing_positive_fixture()
    assert routing["positive"]
    assert set(routing["modalities"]) == set(ALL_MODALITIES)
    assert routing["routed_accuracy"] >= routing["generalist_accuracy"]
    assert routing["routed_cost"] < routing["generalist_cost"]
    assert all(not row["outcome_information_used"] for row in routing["routes"])
    assert all(row["selected"] == row["expected_specialist"] for row in routing["routes"])
