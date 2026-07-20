from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAMPLE_RATE_HZ = 24_000
FRAME_MS = 100
N_CHANNELS = 4
SAMPLES_PER_FRAME = SAMPLE_RATE_HZ * FRAME_MS // 1000
N_CLASSES = 13
_SHA256_HEX = frozenset("0123456789abcdef")


class SchemaRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    room_id: str
    n_frames: int
    audio_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise SchemaRefusal("Clip.clip_id must be a nonempty string")
        if not isinstance(self.room_id, str) or not self.room_id.strip():
            raise SchemaRefusal("Clip.room_id must be a nonempty string")
        if isinstance(self.n_frames, bool) or not isinstance(self.n_frames, int) or self.n_frames <= 0:
            raise SchemaRefusal("Clip.n_frames must be a positive integer")
        digest = self.audio_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _SHA256_HEX for ch in digest):
            raise SchemaRefusal("Clip.audio_sha256 must be a lowercase sha256 digest")


@dataclass(frozen=True, slots=True)
class ClipSplit:
    train: tuple[Clip, ...]
    val: tuple[Clip, ...]
    test: tuple[Clip, ...]
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train or not self.val or not self.test:
            raise SchemaRefusal("ClipSplit train, val, and test partitions must each be nonempty")
        clip_owner: dict[str, str] = {}
        room_owner: dict[str, str] = {}
        for name, clips in {"train": self.train, "val": self.val, "test": self.test}.items():
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
