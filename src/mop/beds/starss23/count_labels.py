from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.evidence import canonical_sha256

from .adapter import MetadataRow, RealStarssAdapter, parse_starss23_metadata

COUNT_LABELS_SCHEMA = "mop-starss23-count-labels/v1"

COUNT_CEILING = 16


class CountLabelRefusal(ValueError):
    pass


def _require_n_frames(n_frames: int) -> int:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise CountLabelRefusal("n_frames must be a positive integer")
    return n_frames


def count_track_from_rows(rows: Sequence[MetadataRow], n_frames: int) -> np.ndarray:

    n_frames = _require_n_frames(n_frames)
    per_frame_tracks: list[set[tuple[int, int]]] = [set() for _ in range(n_frames)]
    for row in rows:
        if not isinstance(row, MetadataRow):
            raise CountLabelRefusal("rows must be MetadataRow values")
        if 0 <= row.frame < n_frames:
            per_frame_tracks[row.frame].add((row.class_id, row.source_id))
    track = np.array([len(tracks) for tracks in per_frame_tracks], dtype=np.int64)
    if track.size and int(track.max()) > COUNT_CEILING:
        raise CountLabelRefusal(
            f"derived concurrent count {int(track.max())} exceeds COUNT_CEILING {COUNT_CEILING}"
        )
    return track


def count_track_from_metadata_text(text: str, n_frames: int) -> np.ndarray:

    return count_track_from_rows(parse_starss23_metadata(text), n_frames)


@dataclass(frozen=True, slots=True)
class CountClip:
    clip_id: str
    room_id: str
    n_frames: int
    audio_sha256: str
    count_track: tuple[int, ...]
    schema: str = COUNT_LABELS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise CountLabelRefusal("CountClip.clip_id must be a nonempty string")
        if not isinstance(self.room_id, str) or not self.room_id.strip():
            raise CountLabelRefusal("CountClip.room_id must be a nonempty string")
        _require_n_frames(self.n_frames)
        if len(self.count_track) != self.n_frames:
            raise CountLabelRefusal("CountClip.count_track length must equal n_frames")
        for value in self.count_track:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CountLabelRefusal("CountClip.count_track must hold nonnegative integers")
            if value > COUNT_CEILING:
                raise CountLabelRefusal("CountClip.count_track value exceeds COUNT_CEILING")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "clip_id": self.clip_id,
            "room_id": self.room_id,
            "n_frames": self.n_frames,
            "audio_sha256": self.audio_sha256,
            "count_track": list(self.count_track),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    @property
    def n_changes(self) -> int:

        track = self.count_track
        return sum(1 for t in range(1, len(track)) if track[t] != track[t - 1])


def _metadata_index(metadata_root: str | Path) -> dict[str, Path]:

    root = Path(metadata_root)
    if not root.is_dir():
        raise CountLabelRefusal(f"metadata root {root} is not a directory")
    index: dict[str, Path] = {}
    for meta_path in sorted(root.rglob("*.csv")):
        index.setdefault(meta_path.stem, meta_path)
    return index


def build_count_clips(adapter: RealStarssAdapter, metadata_root: str | Path) -> dict[str, CountClip]:

    index = _metadata_index(metadata_root)
    out: dict[str, CountClip] = {}
    for clip in adapter.clips():
        meta_path = index.get(clip.clip_id)
        if meta_path is None:
            raise CountLabelRefusal(f"clip {clip.clip_id} has no matching metadata under {metadata_root}")
        track = count_track_from_metadata_text(meta_path.read_text(encoding="utf-8"), clip.n_frames)
        out[clip.clip_id] = CountClip(
            clip_id=clip.clip_id,
            room_id=clip.room_id,
            n_frames=clip.n_frames,
            audio_sha256=clip.audio_sha256,
            count_track=tuple(int(v) for v in track.tolist()),
        )
    return out


def change_density(clips: Iterable[CountClip]) -> float:

    total_changes = 0
    total_frames = 0
    for clip in clips:
        total_changes += clip.n_changes
        total_frames += clip.n_frames
    return total_changes / total_frames if total_frames > 0 else 0.0


def coast_from_zero_mae(clips: Iterable[CountClip]) -> float:

    abs_sum = 0
    total_frames = 0
    for clip in clips:
        abs_sum += sum(abs(int(v)) for v in clip.count_track)
        total_frames += clip.n_frames
    return abs_sum / total_frames if total_frames > 0 else 0.0
