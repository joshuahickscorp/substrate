from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mop.evidence import canonical_sha256

SCHEMA = "mop-starss23-clip-contract/v1"

SAMPLE_RATE_HZ = 24_000
FRAME_MS = 100
N_CHANNELS = 4
SAMPLES_PER_FRAME = SAMPLE_RATE_HZ * FRAME_MS // 1000  # 2400 samples per 100 ms label frame
COLLAR_FRAMES = 2
N_CLASSES = 13

_SHA256_HEX = frozenset("0123456789abcdef")


class SchemaRefusal(ValueError):
    pass


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _SHA256_HEX for ch in value):
        raise SchemaRefusal(f"{label} must be a lowercase sha256 digest")
    return value


def _require_finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SchemaRefusal(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class OnsetEvent:
    frame: int
    class_id: int
    azimuth: float
    elevation: float
    distance: float

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise SchemaRefusal("OnsetEvent.frame must be a nonnegative integer")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise SchemaRefusal("OnsetEvent.class_id must be an integer")
        if not 0 <= self.class_id < N_CLASSES:
            raise SchemaRefusal(f"OnsetEvent.class_id must be in [0, {N_CLASSES})")
        az = _require_finite(self.azimuth, "OnsetEvent.azimuth")
        el = _require_finite(self.elevation, "OnsetEvent.elevation")
        dist = _require_finite(self.distance, "OnsetEvent.distance")
        if not -180.0 <= az <= 180.0:
            raise SchemaRefusal("OnsetEvent.azimuth must lie in [-180, 180] degrees")
        if not -90.0 <= el <= 90.0:
            raise SchemaRefusal("OnsetEvent.elevation must lie in [-90, 90] degrees")
        if dist <= 0.0:
            raise SchemaRefusal("OnsetEvent.distance must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "class_id": self.class_id,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OnsetEvent:
        return cls(
            frame=int(payload["frame"]),
            class_id=int(payload["class_id"]),
            azimuth=float(payload["azimuth"]),
            elevation=float(payload["elevation"]),
            distance=float(payload["distance"]),
        )


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    room_id: str
    n_frames: int
    audio_sha256: str
    onsets: tuple[OnsetEvent, ...]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise SchemaRefusal(f"unsupported clip schema {self.schema!r}")
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise SchemaRefusal("Clip.clip_id must be a nonempty string")
        if not isinstance(self.room_id, str) or not self.room_id.strip():
            raise SchemaRefusal("Clip.room_id must be a nonempty string")
        if isinstance(self.n_frames, bool) or not isinstance(self.n_frames, int) or self.n_frames <= 0:
            raise SchemaRefusal("Clip.n_frames must be a positive integer")
        _require_sha256(self.audio_sha256, "Clip.audio_sha256")
        seen: set[int] = set()
        previous = -1
        for onset in self.onsets:
            if not isinstance(onset, OnsetEvent):
                raise SchemaRefusal("Clip.onsets must contain OnsetEvent values")
            if onset.frame >= self.n_frames:
                raise SchemaRefusal("Clip onset frame is out of range for the clip length")
            if onset.frame < previous:
                raise SchemaRefusal("Clip.onsets must be sorted by nondecreasing frame")
            previous = onset.frame
            if onset.frame in seen:
                raise SchemaRefusal("Clip.onsets must carry at most one event per frame")
            seen.add(onset.frame)

    @property
    def onset_frames(self) -> tuple[int, ...]:
        return tuple(onset.frame for onset in self.onsets)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "clip_id": self.clip_id,
            "room_id": self.room_id,
            "n_frames": self.n_frames,
            "audio_sha256": self.audio_sha256,
            "onsets": [onset.payload() for onset in self.onsets],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Clip:
        return cls(
            clip_id=str(payload["clip_id"]),
            room_id=str(payload["room_id"]),
            n_frames=int(payload["n_frames"]),
            audio_sha256=str(payload["audio_sha256"]),
            onsets=tuple(OnsetEvent.from_payload(row) for row in payload["onsets"]),
        )


@dataclass(frozen=True, slots=True)
class ClipSplit:
    train: tuple[Clip, ...]
    val: tuple[Clip, ...]
    test: tuple[Clip, ...]
    schema: str = SCHEMA
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise SchemaRefusal(f"unsupported split schema {self.schema!r}")
        if not self.train or not self.val or not self.test:
            raise SchemaRefusal("ClipSplit train, val, and test partitions must each be nonempty")
        partitions = {"train": self.train, "val": self.val, "test": self.test}
        clip_owner: dict[str, str] = {}
        room_owner: dict[str, str] = {}
        for name, clips in partitions.items():
            for clip in clips:
                if not isinstance(clip, Clip):
                    raise SchemaRefusal("ClipSplit partitions must contain Clip values")
                if clip.clip_id in clip_owner:
                    raise SchemaRefusal(
                        f"clip {clip.clip_id} appears in both {clip_owner[clip.clip_id]} and {name}"
                    )
                clip_owner[clip.clip_id] = name
                prior_room = room_owner.get(clip.room_id)
                if prior_room is not None and prior_room != name:
                    raise SchemaRefusal(f"room {clip.room_id} is not disjoint: in {prior_room} and {name}")
                room_owner[clip.room_id] = name

    @property
    def rooms(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": tuple(sorted({clip.room_id for clip in self.train})),
            "val": tuple(sorted({clip.room_id for clip in self.val})),
            "test": tuple(sorted({clip.room_id for clip in self.test})),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "train": [clip.payload() for clip in self.train],
            "val": [clip.payload() for clip in self.val],
            "test": [clip.payload() for clip in self.test],
            "rooms": {name: list(rooms) for name, rooms in self.rooms.items()},
            "detail": dict(self.detail),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def room_disjoint_split(clips: Sequence[Clip], *, n_train_rooms: int, n_val_rooms: int) -> ClipSplit:

    if n_train_rooms <= 0 or n_val_rooms <= 0:
        raise SchemaRefusal("room split needs at least one train room and one val room")
    by_room: dict[str, list[Clip]] = {}
    for clip in clips:
        by_room.setdefault(clip.room_id, []).append(clip)
    ordered_rooms = sorted(by_room)
    if len(ordered_rooms) < n_train_rooms + n_val_rooms + 1:
        raise SchemaRefusal(
            "room-disjoint split needs at least "
            f"{n_train_rooms + n_val_rooms + 1} rooms, saw {len(ordered_rooms)}"
        )
    train_rooms = ordered_rooms[:n_train_rooms]
    val_rooms = ordered_rooms[n_train_rooms : n_train_rooms + n_val_rooms]
    test_rooms = ordered_rooms[n_train_rooms + n_val_rooms :]

    def _collect(room_ids: Sequence[str]) -> tuple[Clip, ...]:
        gathered: list[Clip] = []
        for room_id in room_ids:
            gathered.extend(sorted(by_room[room_id], key=lambda clip: clip.clip_id))
        return tuple(gathered)

    return ClipSplit(
        train=_collect(train_rooms),
        val=_collect(val_rooms),
        test=_collect(test_rooms),
        detail={
            "train_rooms": list(train_rooms),
            "val_rooms": list(val_rooms),
            "test_rooms": list(test_rooms),
        },
    )


def clip_partition(split: ClipSplit) -> dict[str, tuple[str, ...]]:

    return {
        "train": tuple(clip.clip_id for clip in split.train),
        "val": tuple(clip.clip_id for clip in split.val),
        "test": tuple(clip.clip_id for clip in split.test),
    }
