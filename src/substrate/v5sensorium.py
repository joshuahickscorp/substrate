"""Typed deterministic multimodal sensorium and persistent scene mechanisms.

This module keeps signal, feature, proposal, tracking, event, verification, and
structural layers as different types.  It contains no decoder or remote model;
those can be attached through the contracts in :mod:`substrate.v5models`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class SensoriumError(ValueError):
    """A sensory record violates time, coordinate, or layer authority."""


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    MOTION = "motion"
    AUDIO = "audio"
    SPEECH = "speech"
    DEPTH_3D = "depth_3d"
    BODY_TOOL = "body_tool"


class RepresentationLayer(StrEnum):
    RAW_SIGNAL = "raw_signal"
    PREPROCESSED_SIGNAL = "preprocessed_signal"
    PERCEPTUAL_PROPOSAL = "perceptual_proposal"
    TRACKED_WORLD = "tracked_world"
    INFERRED_EVENT = "inferred_event"
    VERIFIED_RELATION = "verified_relation"
    STRUCTURAL_BELIEF = "structural_belief"
    KNOWLEDGE = "knowledge"


Vec3 = tuple[float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]
IDENTITY_ROTATION: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _finite_vector(value: Iterable[float], *, length: int = 3) -> tuple[float, ...]:
    vector = tuple(float(part) for part in value)
    if len(vector) != length or not all(math.isfinite(part) for part in vector):
        raise SensoriumError(f"expected {length} finite coordinates")
    return vector


def _distance(left: Iterable[float], right: Iterable[float]) -> float:
    a = tuple(float(value) for value in left)
    b = tuple(float(value) for value in right)
    if len(a) != len(b):
        raise SensoriumError("distance operands have different dimensions")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _confidence(confidence: float, uncertainty: float) -> None:
    if not 0.0 <= confidence <= 1.0 or not 0.0 <= uncertainty <= 1.0:
        raise SensoriumError("confidence and uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class RawSignal:
    reference: str
    encoding: str
    byte_length: int
    content_digest: str

    def __post_init__(self) -> None:
        if not self.reference or not self.encoding or self.byte_length < 0:
            raise SensoriumError("raw signal requires a reference, encoding, and non-negative length")
        if len(self.content_digest) < 16:
            raise SensoriumError("raw signal digest is too short to bind source bytes")


@dataclass(frozen=True)
class PreprocessedSignal:
    source_raw_reference: str
    preprocessing_identity: str
    model_identity: str | None
    features: tuple[float, ...]
    precision: str

    def __post_init__(self) -> None:
        if not self.source_raw_reference or not self.preprocessing_identity or not self.precision:
            raise SensoriumError("preprocessed signal is missing provenance")
        if not all(math.isfinite(value) for value in self.features):
            raise SensoriumError("preprocessed features must be finite")


@dataclass(frozen=True)
class PerceptualProposal:
    proposal_id: str
    kind: str
    coordinate_frame: str
    properties: Mapping[str, Any]
    confidence: float
    uncertainty: float
    evidence_references: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.kind or not self.coordinate_frame:
            raise SensoriumError("proposal identity, kind, and coordinate frame are required")
        _confidence(self.confidence, self.uncertainty)


@dataclass(frozen=True)
class TrackedEntity:
    track_id: str
    kind: str
    coordinate_frame: str
    position: Vec3
    velocity: Vec3
    appearance: tuple[float, ...]
    status: str
    confidence: float
    uncertainty: float
    first_seen: float
    last_seen: float
    proposal_references: tuple[str, ...]
    viewpoints: tuple[str, ...]
    occluded_steps: int = 0

    def __post_init__(self) -> None:
        _finite_vector(self.position)
        _finite_vector(self.velocity)
        _confidence(self.confidence, self.uncertainty)
        if self.status not in {"visible", "occluded", "lost"}:
            raise SensoriumError("track status must be visible, occluded, or lost")


@dataclass(frozen=True)
class InferredEvent:
    event_id: str
    event_type: str
    participant_tracks: tuple[str, ...]
    roles: Mapping[str, str]
    start_time: float
    end_time: float | None
    temporal_relations: tuple[str, ...]
    spatial_relations: tuple[str, ...]
    causal_hypotheses: tuple[str, ...]
    evidence_references: tuple[str, ...]
    unresolved_alternatives: tuple[str, ...]
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        if self.end_time is not None and self.end_time < self.start_time:
            raise SensoriumError("event end precedes event start")
        _confidence(self.confidence, self.uncertainty)


@dataclass(frozen=True)
class VerifiedRelation:
    relation_id: str
    relation: str
    arguments: tuple[str, ...]
    verification_method: str
    evidence_references: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        _confidence(self.confidence, self.uncertainty)
        if not self.verification_method:
            raise SensoriumError("verified relation must identify its verification method")


@dataclass(frozen=True)
class StructuralBelief:
    belief_id: str
    statement: str
    supporting_relations: tuple[str, ...]
    defeaters: tuple[str, ...]
    verification_state: str
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        _confidence(self.confidence, self.uncertainty)


@dataclass(frozen=True)
class KnowledgeRecord:
    knowledge_id: str
    statement: str
    source_beliefs: tuple[str, ...]
    admitted_by: str
    confidence: float
    uncertainty: float

    def __post_init__(self) -> None:
        _confidence(self.confidence, self.uncertainty)
        if not self.admitted_by:
            raise SensoriumError("knowledge requires an explicit admission authority")


_HIDDEN_ID_KEYS = frozenset(
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


def _hidden_mapping_keys(value: object) -> set[str]:
    """Return forbidden key names found anywhere in a JSON-shaped value."""

    # Keep one accumulator while walking the JSON-shaped tree.  The prior
    # recursive form rebuilt and unioned every key even though callers only
    # need the hidden-target intersection.
    keys: set[str] = set()
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key in current:
                normalized = str(key).lower()
                if normalized in _HIDDEN_ID_KEYS:
                    keys.add(normalized)
            pending.extend(current.values())
        elif isinstance(current, (tuple, list)):
            pending.extend(current)
    return keys


@dataclass(frozen=True)
class SensorEvent:
    """One complete, typed sensory receipt with non-collapsed representations."""

    sensor_identity: str
    modality: Modality
    timestamp: float
    sequence_identity: str
    sequence_number: int
    coordinate_frame: str
    raw_data_reference: str
    preprocessing_identity: str
    model_identity: str | None
    observation: Mapping[str, Any]
    hypothesis: str | None
    confidence: float
    uncertainty: float
    provenance: tuple[str, ...]
    quality_flags: tuple[str, ...]
    missing_data_flags: tuple[str, ...]
    raw: RawSignal
    preprocessed: PreprocessedSignal
    proposals: tuple[PerceptualProposal, ...] = ()
    tracked_entities: tuple[TrackedEntity, ...] = ()
    inferred_events: tuple[InferredEvent, ...] = ()
    verified_relations: tuple[VerifiedRelation, ...] = ()
    structural_beliefs: tuple[StructuralBelief, ...] = ()
    knowledge: tuple[KnowledgeRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.sensor_identity or not self.sequence_identity or not self.coordinate_frame:
            raise SensoriumError("sensor, sequence, and coordinate-frame identities are required")
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0 or self.sequence_number < 0:
            raise SensoriumError("sensor time and sequence number must be non-negative")
        _confidence(self.confidence, self.uncertainty)
        if self.raw.reference != self.raw_data_reference:
            raise SensoriumError("raw layer is not bound to the event raw-data reference")
        if self.preprocessed.source_raw_reference != self.raw.reference:
            raise SensoriumError("preprocessed layer is not bound to the raw layer")
        if self.preprocessed.preprocessing_identity != self.preprocessing_identity:
            raise SensoriumError("preprocessing identity mismatch")
        if self.preprocessed.model_identity != self.model_identity:
            raise SensoriumError("model identity mismatch between event and feature layer")
        public_content = {
            "observation": self.observation,
            "proposals": [dataclasses.asdict(value) for value in self.proposals],
            "tracked": [dataclasses.asdict(value) for value in self.tracked_entities],
            "inferred_events": [dataclasses.asdict(value) for value in self.inferred_events],
            "verified": [dataclasses.asdict(value) for value in self.verified_relations],
            "structural": [dataclasses.asdict(value) for value in self.structural_beliefs],
            "knowledge": [dataclasses.asdict(value) for value in self.knowledge],
        }
        leaked = _hidden_mapping_keys(public_content)
        if leaked:
            raise SensoriumError(f"hidden target authority in public sensor event: {sorted(leaked)}")

    @property
    def populated_layers(self) -> tuple[RepresentationLayer, ...]:
        layers = [RepresentationLayer.RAW_SIGNAL, RepresentationLayer.PREPROCESSED_SIGNAL]
        for values, layer in (
            (self.proposals, RepresentationLayer.PERCEPTUAL_PROPOSAL),
            (self.tracked_entities, RepresentationLayer.TRACKED_WORLD),
            (self.inferred_events, RepresentationLayer.INFERRED_EVENT),
            (self.verified_relations, RepresentationLayer.VERIFIED_RELATION),
            (self.structural_beliefs, RepresentationLayer.STRUCTURAL_BELIEF),
            (self.knowledge, RepresentationLayer.KNOWLEDGE),
        ):
            if values:
                layers.append(layer)
        return tuple(layers)

    def public_observation(self) -> dict[str, Any]:
        body = {
            "sensor_identity": self.sensor_identity,
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "sequence_identity": self.sequence_identity,
            "sequence_number": self.sequence_number,
            "coordinate_frame": self.coordinate_frame,
            "raw_data_reference": self.raw_data_reference,
            "preprocessing_identity": self.preprocessing_identity,
            "model_identity": self.model_identity,
            "observation": dict(self.observation),
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "quality_flags": self.quality_flags,
            "missing_data_flags": self.missing_data_flags,
            "layers": {
                "raw": dataclasses.asdict(self.raw),
                "preprocessed": dataclasses.asdict(self.preprocessed),
                "proposals": [dataclasses.asdict(value) for value in self.proposals],
                "tracked": [dataclasses.asdict(value) for value in self.tracked_entities],
                "inferred_events": [dataclasses.asdict(value) for value in self.inferred_events],
                "verified": [dataclasses.asdict(value) for value in self.verified_relations],
                "structural": [dataclasses.asdict(value) for value in self.structural_beliefs],
                "knowledge": [dataclasses.asdict(value) for value in self.knowledge],
            },
        }
        leaked = _hidden_mapping_keys(body)
        if leaked:
            raise SensoriumError(f"hidden target authority leaked during serialization: {sorted(leaked)}")
        return body


class Sensorium:
    """Append-only in-memory sensory authority with time and sequence checks."""

    def __init__(self, coordinate_frames: CoordinateFrameRegistry | None = None) -> None:
        self.coordinate_frames = coordinate_frames or CoordinateFrameRegistry()
        self._events: list[SensorEvent] = []
        self._last_timestamp: dict[str, float] = {}
        self._last_sequence: dict[str, int] = {}

    @property
    def events(self) -> tuple[SensorEvent, ...]:
        return tuple(self._events)

    def ingest(self, event: SensorEvent) -> None:
        event.public_observation()
        self._append_validated(event)

    def ingest_and_digest(self, event: SensorEvent) -> str:
        """Validate, append, and digest one event without rebuilding its body."""

        public = event.public_observation()
        self._append_validated(event)
        return _canonical_observation_digest(public)

    def _append_validated(self, event: SensorEvent) -> None:
        if not self.coordinate_frames.has_frame(event.coordinate_frame):
            raise SensoriumError(f"unknown coordinate frame {event.coordinate_frame!r}")
        previous_time = self._last_timestamp.get(event.sensor_identity)
        previous_sequence = self._last_sequence.get(event.sequence_identity)
        if previous_time is not None and event.timestamp < previous_time:
            raise SensoriumError("corrupted sensory time: timestamp moved backwards")
        if previous_sequence is not None and event.sequence_number <= previous_sequence:
            raise SensoriumError("corrupted sequence: sequence number did not increase")
        self._events.append(event)
        self._last_timestamp[event.sensor_identity] = event.timestamp
        self._last_sequence[event.sequence_identity] = event.sequence_number

    def latest(self, modality: Modality | None = None) -> SensorEvent | None:
        for event in reversed(self._events):
            if modality is None or event.modality == modality:
                return event
        return None


@dataclass(frozen=True)
class CoordinateTransform:
    source_frame: str
    target_frame: str
    rotation: Matrix3 = IDENTITY_ROTATION
    translation: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.source_frame or not self.target_frame or self.source_frame == self.target_frame:
            raise SensoriumError("coordinate transform needs two distinct named frames")
        rows = tuple(_finite_vector(row) for row in self.rotation)
        translation = _finite_vector(self.translation)
        for row in rows:
            if not math.isclose(sum(value * value for value in row), 1.0, abs_tol=1e-6):
                raise SensoriumError("coordinate rotation is not normalized")
        for left in range(3):
            for right in range(left + 1, 3):
                if not math.isclose(sum(rows[left][i] * rows[right][i] for i in range(3)), 0.0, abs_tol=1e-6):
                    raise SensoriumError("coordinate rotation is not orthogonal")
        determinant = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        if not math.isclose(determinant, 1.0, abs_tol=1e-6):
            raise SensoriumError("coordinate rotation changes handedness or scale")
        object.__setattr__(self, "rotation", rows)
        object.__setattr__(self, "translation", translation)

    def apply(self, point: Vec3) -> Vec3:
        source = _finite_vector(point)
        return tuple(
            sum(self.rotation[row][column] * source[column] for column in range(3)) + self.translation[row]
            for row in range(3)
        )  # type: ignore[return-value]

    def inverse(self) -> CoordinateTransform:
        transpose: Matrix3 = tuple(tuple(self.rotation[column][row] for column in range(3)) for row in range(3))  # type: ignore[assignment]
        inverse_translation = tuple(
            -sum(transpose[row][column] * self.translation[column] for column in range(3))
            for row in range(3)
        )
        return CoordinateTransform(self.target_frame, self.source_frame, transpose, inverse_translation)  # type: ignore[arg-type]


class CoordinateFrameRegistry:
    """A checked transform graph; unregistered and disconnected frames are refused."""

    def __init__(self, root_frame: str = "world") -> None:
        self.root_frame = root_frame
        self._frames = {root_frame}
        self._edges: dict[str, list[CoordinateTransform]] = {root_frame: []}

    def has_frame(self, frame: str) -> bool:
        return frame in self._frames

    def add_transform(self, transform: CoordinateTransform) -> None:
        if transform.source_frame in self._frames:
            raise SensoriumError(f"coordinate frame {transform.source_frame!r} already registered")
        if transform.target_frame not in self._frames:
            raise SensoriumError("coordinate parent must already be registered")
        self._frames.add(transform.source_frame)
        self._edges.setdefault(transform.source_frame, []).append(transform)
        self._edges.setdefault(transform.target_frame, []).append(transform.inverse())

    def transform(self, point: Vec3, source_frame: str, target_frame: str = "world") -> Vec3:
        if source_frame not in self._frames or target_frame not in self._frames:
            raise SensoriumError("unknown coordinate frame")
        if source_frame == target_frame:
            return _finite_vector(point)  # type: ignore[return-value]
        queue: list[tuple[str, Vec3]] = [(source_frame, _finite_vector(point))]  # type: ignore[list-item]
        visited = {source_frame}
        while queue:
            frame, current = queue.pop(0)
            for edge in self._edges.get(frame, ()):
                if edge.target_frame in visited:
                    continue
                transformed = edge.apply(current)
                if edge.target_frame == target_frame:
                    return transformed
                visited.add(edge.target_frame)
                queue.append((edge.target_frame, transformed))
        raise SensoriumError(f"coordinate frames {source_frame!r} and {target_frame!r} are disconnected")


class ObjectTracker:
    """Deterministic kinematic/appearance tracker with explicit occlusion state."""

    def __init__(
        self,
        coordinate_frames: CoordinateFrameRegistry,
        *,
        association_distance: float = 2.5,
        appearance_distance: float = 0.75,
        maximum_occluded_steps: int = 4,
    ) -> None:
        self.coordinate_frames = coordinate_frames
        self.association_distance = association_distance
        self.appearance_distance = appearance_distance
        self.maximum_occluded_steps = maximum_occluded_steps
        self._tracks: dict[str, TrackedEntity] = {}
        self._counter = 0
        self._last_update: float | None = None

    @property
    def tracks(self) -> tuple[TrackedEntity, ...]:
        return tuple(self._tracks[identity] for identity in sorted(self._tracks))

    def update(self, proposals: Iterable[PerceptualProposal], timestamp: float, *, viewpoint: str) -> tuple[TrackedEntity, ...]:
        if self._last_update is not None and timestamp < self._last_update:
            raise SensoriumError("tracker time moved backwards")
        proposal_rows = []
        for proposal in proposals:
            if proposal.kind not in {"object", "agent", "tool"}:
                continue
            if "position" not in proposal.properties:
                raise SensoriumError("trackable proposal lacks a 3D position")
            position = self.coordinate_frames.transform(
                _finite_vector(proposal.properties["position"]),  # type: ignore[arg-type]
                proposal.coordinate_frame,
                self.coordinate_frames.root_frame,
            )
            appearance = tuple(float(value) for value in proposal.properties.get("appearance", ()))
            if not appearance or not all(math.isfinite(value) for value in appearance):
                raise SensoriumError("trackable proposal lacks finite appearance features")
            proposal_rows.append((proposal, position, appearance))
        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self._tracks.items():
            if track.status == "lost":
                continue
            previous_time = self._last_update if track.status == "occluded" and self._last_update is not None else track.last_seen
            dt = max(0.0, timestamp - previous_time)
            predicted = tuple(track.position[index] + track.velocity[index] * dt for index in range(3))
            for index, (_, position, appearance) in enumerate(proposal_rows):
                spatial = _distance(predicted, position)
                visual = _distance(track.appearance, appearance) if len(track.appearance) == len(appearance) else math.inf
                if spatial <= self.association_distance and visual <= self.appearance_distance:
                    candidates.append((spatial + 0.5 * visual, track_id, index))
        matched_tracks: set[str] = set()
        matched_proposals: set[int] = set()
        for _, track_id, proposal_index in sorted(candidates):
            if track_id in matched_tracks or proposal_index in matched_proposals:
                continue
            track = self._tracks[track_id]
            proposal, position, appearance = proposal_rows[proposal_index]
            previous_time = self._last_update if track.status == "occluded" and self._last_update is not None else track.last_seen
            dt = timestamp - previous_time
            velocity = track.velocity if dt <= 0.0 else tuple((position[index] - track.position[index]) / dt for index in range(3))
            self._tracks[track_id] = replace(
                track,
                position=position,
                velocity=velocity,  # type: ignore[arg-type]
                appearance=tuple(
                    (old + new) / 2.0 for old, new in zip(track.appearance, appearance, strict=True)
                ),
                status="visible",
                confidence=min(1.0, (track.confidence + proposal.confidence) / 2.0 + 0.05),
                uncertainty=max(0.0, (track.uncertainty + proposal.uncertainty) / 2.0 - 0.05),
                last_seen=timestamp,
                proposal_references=track.proposal_references + (proposal.proposal_id,),
                viewpoints=tuple(dict.fromkeys((*track.viewpoints, viewpoint))),
                occluded_steps=0,
            )
            matched_tracks.add(track_id)
            matched_proposals.add(proposal_index)
        for index, (proposal, position, appearance) in enumerate(proposal_rows):
            if index in matched_proposals:
                continue
            self._counter += 1
            track_id = f"track-{self._counter:06d}"
            self._tracks[track_id] = TrackedEntity(
                track_id=track_id,
                kind=proposal.kind,
                coordinate_frame=self.coordinate_frames.root_frame,
                position=position,
                velocity=(0.0, 0.0, 0.0),
                appearance=appearance,
                status="visible",
                confidence=proposal.confidence,
                uncertainty=proposal.uncertainty,
                first_seen=timestamp,
                last_seen=timestamp,
                proposal_references=(proposal.proposal_id,),
                viewpoints=(viewpoint,),
            )
            matched_tracks.add(track_id)
        for track_id, track in tuple(self._tracks.items()):
            if track_id in matched_tracks or track.status == "lost":
                continue
            steps = track.occluded_steps + 1
            status = "lost" if steps > self.maximum_occluded_steps else "occluded"
            previous_time = self._last_update if self._last_update is not None else track.last_seen
            dt = max(0.0, timestamp - previous_time)
            predicted = tuple(track.position[index] + track.velocity[index] * dt for index in range(3))
            self._tracks[track_id] = replace(
                track,
                position=predicted,  # type: ignore[arg-type]
                status=status,
                confidence=max(0.0, track.confidence - 0.08),
                uncertainty=min(1.0, track.uncertainty + 0.08),
                occluded_steps=steps,
            )
        self._last_update = timestamp
        return self.tracks


class EventTracker:
    """Persistent event hypotheses with delayed closure and alternatives."""

    def __init__(self) -> None:
        self._events: dict[str, InferredEvent] = {}
        self._counter = 0

    @property
    def events(self) -> tuple[InferredEvent, ...]:
        return tuple(self._events[key] for key in sorted(self._events))

    def observe(
        self,
        event_type: str,
        participants: tuple[str, ...],
        timestamp: float,
        evidence_reference: str,
        *,
        causal_hypotheses: tuple[str, ...] = (),
        alternatives: tuple[str, ...] = (),
    ) -> InferredEvent:
        active = next(
            (
                event
                for event in self._events.values()
                if event.end_time is None and event.event_type == event_type and event.participant_tracks == participants
            ),
            None,
        )
        if active is None:
            self._counter += 1
            active = InferredEvent(
                event_id=f"event-{self._counter:06d}",
                event_type=event_type,
                participant_tracks=participants,
                roles={f"participant_{index}": identity for index, identity in enumerate(participants)},
                start_time=timestamp,
                end_time=None,
                temporal_relations=(),
                spatial_relations=(),
                causal_hypotheses=causal_hypotheses,
                evidence_references=(evidence_reference,),
                unresolved_alternatives=alternatives,
                confidence=0.65,
                uncertainty=0.35,
            )
        else:
            active = replace(
                active,
                evidence_references=active.evidence_references + (evidence_reference,),
                causal_hypotheses=tuple(dict.fromkeys((*active.causal_hypotheses, *causal_hypotheses))),
                unresolved_alternatives=tuple(dict.fromkeys((*active.unresolved_alternatives, *alternatives))),
                confidence=min(0.98, active.confidence + 0.08),
                uncertainty=max(0.02, active.uncertainty - 0.08),
            )
        self._events[active.event_id] = active
        return active

    def close(self, event_id: str, timestamp: float) -> InferredEvent:
        try:
            event = self._events[event_id]
        except KeyError as exc:
            raise SensoriumError(f"unknown event {event_id!r}") from exc
        closed = replace(event, end_time=timestamp)
        self._events[event_id] = closed
        return closed


@dataclass(frozen=True)
class TimedCue:
    cue_id: str
    modality: Modality
    onset: float
    offset: float
    content: str
    position: Vec3 | None
    confidence: float

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise SensoriumError("cue offset precedes onset")
        if not 0.0 <= self.confidence <= 1.0:
            raise SensoriumError("cue confidence must be in [0, 1]")
        if self.position is not None:
            _finite_vector(self.position)


@dataclass(frozen=True)
class AudiovisualAlignment:
    audio_reference: str
    visual_reference: str
    offset_seconds: float
    overlap_seconds: float
    timing_score: float
    synchronized: bool
    causal_hypothesis: str | None
    conflict: str | None


class AudiovisualAligner:
    def __init__(self, tolerance_seconds: float = 0.12) -> None:
        if tolerance_seconds <= 0.0:
            raise SensoriumError("audiovisual tolerance must be positive")
        self.tolerance_seconds = tolerance_seconds

    def align(self, audio: TimedCue, visual: TimedCue) -> AudiovisualAlignment:
        if audio.modality not in {Modality.AUDIO, Modality.SPEECH}:
            raise SensoriumError("audio cue has a non-audio modality")
        if visual.modality not in {Modality.VIDEO, Modality.MOTION, Modality.IMAGE}:
            raise SensoriumError("visual cue has a non-visual modality")
        offset = audio.onset - visual.onset
        overlap = max(0.0, min(audio.offset, visual.offset) - max(audio.onset, visual.onset))
        timing_score = math.exp(-abs(offset) / self.tolerance_seconds)
        synchronized = abs(offset) <= self.tolerance_seconds and overlap > 0.0
        semantic_conflict = audio.content != visual.content
        conflict = None
        if not synchronized:
            conflict = "temporal_conflict"
        elif semantic_conflict:
            conflict = "semantic_conflict"
        causal = "visible_action_caused_sound" if synchronized and not semantic_conflict else None
        return AudiovisualAlignment(
            audio_reference=audio.cue_id,
            visual_reference=visual.cue_id,
            offset_seconds=offset,
            overlap_seconds=overlap,
            timing_score=timing_score,
            synchronized=synchronized,
            causal_hypothesis=causal,
            conflict=conflict,
        )


@dataclass(frozen=True)
class CrossModalEvidence:
    evidence_id: str
    modality: Modality
    onset: float
    offset: float
    semantics: frozenset[str]
    position: Vec3 | None
    confidence: float
    source_reliability: float

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise SensoriumError("cross-modal evidence offset precedes onset")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.source_reliability <= 1.0:
            raise SensoriumError("evidence confidence and reliability must be in [0, 1]")


@dataclass(frozen=True)
class ModalConflict:
    conflict_type: str
    evidence_references: tuple[str, ...]
    description: str
    preserved: bool = True


@dataclass(frozen=True)
class BindingCandidate:
    evidence_reference: str
    score: float
    temporal_score: float
    spatial_score: float
    semantic_score: float


@dataclass(frozen=True)
class BindingDecision:
    anchor_reference: str
    selected_reference: str | None
    candidates: tuple[BindingCandidate, ...]
    conflicts: tuple[ModalConflict, ...]
    confidence: float
    uncertainty: float
    forced_fusion: bool = False


class CrossModalBinder:
    """Bind by temporal, spatial, and semantic constraints while retaining conflict."""

    def __init__(self, *, threshold: float = 0.58, ambiguity_margin: float = 0.08, temporal_scale: float = 0.5) -> None:
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.temporal_scale = temporal_scale

    def bind(self, anchor: CrossModalEvidence, candidates: Iterable[CrossModalEvidence]) -> BindingDecision:
        rows: list[BindingCandidate] = []
        conflicts: list[ModalConflict] = []
        anchor_midpoint = (anchor.onset + anchor.offset) / 2.0
        for candidate in candidates:
            if candidate.evidence_id == anchor.evidence_id:
                continue
            midpoint = (candidate.onset + candidate.offset) / 2.0
            temporal = math.exp(-abs(midpoint - anchor_midpoint) / self.temporal_scale)
            if anchor.position is None or candidate.position is None:
                spatial = 0.5
            else:
                spatial = math.exp(-_distance(anchor.position, candidate.position))
            union = anchor.semantics | candidate.semantics
            semantic = len(anchor.semantics & candidate.semantics) / len(union) if union else 0.5
            reliability = math.sqrt(
                anchor.confidence
                * anchor.source_reliability
                * candidate.confidence
                * candidate.source_reliability,
            )
            score = reliability * (0.38 * temporal + 0.32 * spatial + 0.30 * semantic)
            rows.append(BindingCandidate(candidate.evidence_id, score, temporal, spatial, semantic))
            if temporal >= 0.75 and semantic == 0.0:
                conflicts.append(
                    ModalConflict(
                        "semantic_conflict",
                        (anchor.evidence_id, candidate.evidence_id),
                        "temporally compatible sources assert disjoint semantics",
                    ),
                )
            if temporal >= 0.75 and spatial < 0.15:
                conflicts.append(
                    ModalConflict(
                        "spatial_conflict",
                        (anchor.evidence_id, candidate.evidence_id),
                        "temporally compatible sources disagree spatially",
                    ),
                )
        ranked = tuple(sorted(rows, key=lambda row: (-row.score, row.evidence_reference)))
        selected = None
        confidence = 0.0
        if ranked:
            runner_up = ranked[1].score if len(ranked) > 1 else 0.0
            if ranked[0].score >= self.threshold and ranked[0].score - runner_up >= self.ambiguity_margin:
                selected = ranked[0].evidence_reference
                confidence = ranked[0].score
            elif ranked[0].score >= self.threshold:
                conflicts.append(
                    ModalConflict(
                        "identity_conflict",
                        tuple(row.evidence_reference for row in ranked[:2]),
                        "multiple cross-modal bindings remain plausible",
                    ),
                )
                confidence = ranked[0].score
        return BindingDecision(
            anchor_reference=anchor.evidence_id,
            selected_reference=selected,
            candidates=ranked,
            conflicts=tuple(conflicts),
            confidence=confidence,
            uncertainty=1.0 - confidence,
        )


@dataclass(frozen=True)
class SpatialObject:
    track_id: str
    coordinate_frame: str
    position: Vec3
    half_extents: Vec3
    confidence: float
    visible: bool = True

    def __post_init__(self) -> None:
        _finite_vector(self.position)
        extents = _finite_vector(self.half_extents)
        if any(value <= 0.0 for value in extents):
            raise SensoriumError("spatial object extents must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise SensoriumError("spatial confidence must be in [0, 1]")


class SpatialSceneState:
    """Persistent explicit 3D scene state independent of current visibility."""

    def __init__(self, coordinate_frames: CoordinateFrameRegistry) -> None:
        self.coordinate_frames = coordinate_frames
        self._objects: dict[str, SpatialObject] = {}

    @property
    def objects(self) -> tuple[SpatialObject, ...]:
        return tuple(self._objects[key] for key in sorted(self._objects))

    def update(self, spatial_object: SpatialObject) -> SpatialObject:
        world_position = self.coordinate_frames.transform(
            spatial_object.position,
            spatial_object.coordinate_frame,
            self.coordinate_frames.root_frame,
        )
        admitted = replace(
            spatial_object,
            coordinate_frame=self.coordinate_frames.root_frame,
            position=world_position,
        )
        self._objects[admitted.track_id] = admitted
        return admitted

    def set_visibility(self, track_id: str, visible: bool) -> None:
        try:
            self._objects[track_id] = replace(self._objects[track_id], visible=visible)
        except KeyError as exc:
            raise SensoriumError(f"unknown spatial object {track_id!r}") from exc

    def relative_position(self, subject: str, reference: str) -> Vec3:
        left = self._objects[subject].position
        right = self._objects[reference].position
        return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]

    def collides(self, left_id: str, right_id: str) -> bool:
        left = self._objects[left_id]
        right = self._objects[right_id]
        return all(
            abs(left.position[index] - right.position[index]) <= left.half_extents[index] + right.half_extents[index]
            for index in range(3)
        )

    def contains(self, container_id: str, object_id: str) -> bool:
        container = self._objects[container_id]
        child = self._objects[object_id]
        return all(
            abs(child.position[index] - container.position[index]) + child.half_extents[index] <= container.half_extents[index]
            for index in range(3)
        )

    def supported_by(self, object_id: str, support_id: str, *, tolerance: float = 1e-3) -> bool:
        child = self._objects[object_id]
        support = self._objects[support_id]
        child_bottom = child.position[2] - child.half_extents[2]
        support_top = support.position[2] + support.half_extents[2]
        horizontal_overlap = all(
            abs(child.position[index] - support.position[index]) <= child.half_extents[index] + support.half_extents[index]
            for index in (0, 1)
        )
        return horizontal_overlap and math.isclose(child_bottom, support_top, abs_tol=tolerance)

    def reachable(self, body_position: Vec3, track_id: str, reach: float) -> bool:
        if reach < 0.0:
            raise SensoriumError("body reach must be non-negative")
        return _distance(body_position, self._objects[track_id].position) <= reach


@dataclass(frozen=True)
class PerceptionOption:
    action: str
    remaining_hypotheses: tuple[str, ...]
    expected_uncertainty_reduction: float
    cost: float
    downstream_decision_value: float
    sensor_reliability: float = 1.0

    def __post_init__(self) -> None:
        if self.cost < 0.0 or self.downstream_decision_value < 0.0:
            raise SensoriumError("perception cost and decision value must be non-negative")
        if not 0.0 <= self.expected_uncertainty_reduction <= 1.0 or not 0.0 <= self.sensor_reliability <= 1.0:
            raise SensoriumError("perception reductions and reliability must be in [0, 1]")

    @property
    def expected_information_value(self) -> float:
        return self.expected_uncertainty_reduction * self.sensor_reliability * self.downstream_decision_value

    @property
    def net_value(self) -> float:
        return self.expected_information_value - self.cost


@dataclass(frozen=True)
class ActivePerceptionDecision:
    remaining_hypotheses: tuple[str, ...]
    action: str
    expected_information_value: float
    cost: float
    predicted_uncertainty_reduction: float
    downstream_decision_value: float
    stopped: bool


@dataclass(frozen=True)
class ActivePerceptionReceipt:
    decision: ActivePerceptionDecision
    prior_uncertainty: float
    resulting_uncertainty: float
    actual_uncertainty_reduction: float


class ExpectedInformationPolicy:
    """Select the highest positive information value, or stop when none is useful."""

    def choose(
        self,
        options: Iterable[PerceptionOption],
        *,
        current_uncertainty: float,
        stop_action: str = "stop_observing",
    ) -> ActivePerceptionDecision:
        if not 0.0 <= current_uncertainty <= 1.0:
            raise SensoriumError("current uncertainty must be in [0, 1]")
        rows = tuple(options)
        if not rows:
            return ActivePerceptionDecision((), stop_action, 0.0, 0.0, 0.0, 0.0, True)
        best = min(rows, key=lambda option: (-option.net_value, option.cost, option.action))
        if best.net_value <= 0.0 or current_uncertainty == 0.0:
            hypotheses = tuple(dict.fromkeys(value for option in rows for value in option.remaining_hypotheses))
            return ActivePerceptionDecision(hypotheses, stop_action, 0.0, 0.0, 0.0, 0.0, True)
        predicted = min(current_uncertainty, best.expected_uncertainty_reduction * best.sensor_reliability)
        return ActivePerceptionDecision(
            remaining_hypotheses=best.remaining_hypotheses,
            action=best.action,
            expected_information_value=best.expected_information_value,
            cost=best.cost,
            predicted_uncertainty_reduction=predicted,
            downstream_decision_value=best.downstream_decision_value,
            stopped=False,
        )

    def complete(
        self,
        decision: ActivePerceptionDecision,
        *,
        prior_uncertainty: float,
        resulting_uncertainty: float,
    ) -> ActivePerceptionReceipt:
        if not 0.0 <= prior_uncertainty <= 1.0 or not 0.0 <= resulting_uncertainty <= 1.0:
            raise SensoriumError("actual uncertainty values must be in [0, 1]")
        return ActivePerceptionReceipt(
            decision=decision,
            prior_uncertainty=prior_uncertainty,
            resulting_uncertainty=resulting_uncertainty,
            actual_uncertainty_reduction=prior_uncertainty - resulting_uncertainty,
        )


def raw_signal(reference: str, payload: bytes, encoding: str) -> RawSignal:
    """Construct a content-bound raw layer without retaining bytes in world state."""

    return RawSignal(reference, encoding, len(payload), hashlib.sha256(payload).hexdigest())


def canonical_event_digest(event: SensorEvent) -> str:
    """Stable receipt identity suitable for deterministic replay comparisons."""

    return _canonical_observation_digest(event.public_observation())


def _canonical_observation_digest(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ActivePerceptionDecision",
    "ActivePerceptionReceipt",
    "AudiovisualAligner",
    "AudiovisualAlignment",
    "BindingCandidate",
    "BindingDecision",
    "CoordinateFrameRegistry",
    "CoordinateTransform",
    "CrossModalBinder",
    "CrossModalEvidence",
    "EventTracker",
    "ExpectedInformationPolicy",
    "IDENTITY_ROTATION",
    "InferredEvent",
    "KnowledgeRecord",
    "ModalConflict",
    "Modality",
    "ObjectTracker",
    "PerceptionOption",
    "PerceptualProposal",
    "PreprocessedSignal",
    "RawSignal",
    "RepresentationLayer",
    "SensorEvent",
    "Sensorium",
    "SensoriumError",
    "SpatialObject",
    "SpatialSceneState",
    "StructuralBelief",
    "TimedCue",
    "TrackedEntity",
    "VerifiedRelation",
    "canonical_event_digest",
    "raw_signal",
]
