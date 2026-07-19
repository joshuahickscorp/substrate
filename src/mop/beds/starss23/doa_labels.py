
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mop.substrate.events import canonical_sha256

from .adapter import MetadataRow, RealStarssAdapter, parse_starss23_metadata

DOA_LABELS_SCHEMA = "mop-starss23-doa-labels/v1"

DOA_CHANGE_THRESHOLD_DEG = 5.0

DOA_VOC_WINDOW = 1


class DoaLabelRefusal(ValueError):
    pass


def _require_n_frames(n_frames: int) -> int:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise DoaLabelRefusal("n_frames must be a positive integer")
    return n_frames


def dominant_track_at_frame(rows: Sequence[MetadataRow]) -> MetadataRow:

    if not rows:
        raise DoaLabelRefusal("dominant_track_at_frame needs at least one row")
    for row in rows:
        if not isinstance(row, MetadataRow):
            raise DoaLabelRefusal("dominant_track_at_frame rows must be MetadataRow values")
        if not row.has_distance:
            raise DoaLabelRefusal(
                "a polyphonic frame requires STARSS23 distance on every row to break the dominance tie"
            )
    return min(rows, key=lambda row: (row.distance, row.class_id, row.azimuth, row.elevation))


def doa_track_from_rows(
    rows: Sequence[MetadataRow], n_frames: int
) -> tuple[tuple[float, float] | None, ...]:

    n_frames = _require_n_frames(n_frames)
    by_frame: dict[int, list[MetadataRow]] = {}
    for row in rows:
        if not isinstance(row, MetadataRow):
            raise DoaLabelRefusal("rows must be MetadataRow values")
        if 0 <= row.frame < n_frames:
            by_frame.setdefault(row.frame, []).append(row)
    track: list[tuple[float, float] | None] = []
    for t in range(n_frames):
        rows_t = by_frame.get(t)
        if not rows_t:
            track.append(None)
        elif len(rows_t) == 1:
            row = rows_t[0]
            track.append((float(row.azimuth), float(row.elevation)))
        else:
            dominant = dominant_track_at_frame(rows_t)
            track.append((float(dominant.azimuth), float(dominant.elevation)))
    return tuple(track)


def doa_track_from_metadata_text(text: str, n_frames: int) -> tuple[tuple[float, float] | None, ...]:

    return doa_track_from_rows(parse_starss23_metadata(text), n_frames)


def direction_to_unit_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:

    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def great_circle_degrees(az1_deg: float, el1_deg: float, az2_deg: float, el2_deg: float) -> float:

    v1 = direction_to_unit_vector(az1_deg, el1_deg)
    v2 = direction_to_unit_vector(az2_deg, el2_deg)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def great_circle_degrees_batch(directions_a: np.ndarray, directions_b: np.ndarray) -> np.ndarray:

    a = np.asarray(directions_a, dtype=np.float64)
    b = np.asarray(directions_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise DoaLabelRefusal("directions_a must be shape (N, 2)")
    if b.shape != a.shape:
        raise DoaLabelRefusal("directions_a and directions_b must share the same shape")
    az1, el1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    az2, el2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    cos_el1, cos_el2 = np.cos(el1), np.cos(el2)
    v1 = np.stack([cos_el1 * np.cos(az1), cos_el1 * np.sin(az1), np.sin(el1)], axis=1)
    v2 = np.stack([cos_el2 * np.cos(az2), cos_el2 * np.sin(az2), np.sin(el2)], axis=1)
    dot = np.clip((v1 * v2).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _change_frames_for_track(
    track: Sequence[tuple[float, float] | None], angle_threshold_deg: float = DOA_CHANGE_THRESHOLD_DEG
) -> tuple[int, ...]:

    changes: list[int] = []
    previous: tuple[float, float] | None = None
    for t, current in enumerate(track):
        if current is None:
            previous = None
            continue
        if previous is None:
            changes.append(t)
        else:
            jump = great_circle_degrees(previous[0], previous[1], current[0], current[1])
            if jump >= angle_threshold_deg:
                changes.append(t)
        previous = current
    return tuple(changes)


@dataclass(frozen=True, slots=True)
class DoaClip:

    clip_id: str
    room_id: str
    n_frames: int
    audio_sha256: str
    doa_track: tuple[tuple[float, float] | None, ...]
    schema: str = DOA_LABELS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise DoaLabelRefusal("DoaClip.clip_id must be a nonempty string")
        if not isinstance(self.room_id, str) or not self.room_id.strip():
            raise DoaLabelRefusal("DoaClip.room_id must be a nonempty string")
        _require_n_frames(self.n_frames)
        if len(self.doa_track) != self.n_frames:
            raise DoaLabelRefusal("DoaClip.doa_track length must equal n_frames")
        for entry in self.doa_track:
            if entry is None:
                continue
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise DoaLabelRefusal(
                    "DoaClip.doa_track entries must be None or an (azimuth, elevation) pair"
                )
            az, el = entry
            if isinstance(az, bool) or isinstance(el, bool):
                raise DoaLabelRefusal("DoaClip.doa_track entries must hold real numbers")
            if not -180.0 <= float(az) <= 180.0:
                raise DoaLabelRefusal("DoaClip.doa_track azimuth must lie in [-180, 180] degrees")
            if not -90.0 <= float(el) <= 90.0:
                raise DoaLabelRefusal("DoaClip.doa_track elevation must lie in [-90, 90] degrees")

    @property
    def active_frames(self) -> tuple[int, ...]:
        return tuple(t for t, entry in enumerate(self.doa_track) if entry is not None)

    @property
    def n_active_frames(self) -> int:
        return len(self.active_frames)

    @property
    def change_frames(self) -> tuple[int, ...]:
        return _change_frames_for_track(self.doa_track)

    @property
    def n_changes(self) -> int:
        return len(self.change_frames)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "clip_id": self.clip_id,
            "room_id": self.room_id,
            "n_frames": self.n_frames,
            "audio_sha256": self.audio_sha256,
            "doa_track": [None if e is None else [float(e[0]), float(e[1])] for e in self.doa_track],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _metadata_index(metadata_root: str | Path) -> dict[str, Path]:

    root = Path(metadata_root)
    if not root.is_dir():
        raise DoaLabelRefusal(f"metadata root {root} is not a directory")
    index: dict[str, Path] = {}
    for meta_path in sorted(root.rglob("*.csv")):
        index.setdefault(meta_path.stem, meta_path)
    return index


def build_doa_clips(adapter: RealStarssAdapter, metadata_root: str | Path) -> dict[str, DoaClip]:

    index = _metadata_index(metadata_root)
    out: dict[str, DoaClip] = {}
    for clip in adapter.clips():
        meta_path = index.get(clip.clip_id)
        if meta_path is None:
            raise DoaLabelRefusal(f"clip {clip.clip_id} has no matching metadata under {metadata_root}")
        track = doa_track_from_metadata_text(meta_path.read_text(encoding="utf-8"), clip.n_frames)
        out[clip.clip_id] = DoaClip(
            clip_id=clip.clip_id,
            room_id=clip.room_id,
            n_frames=clip.n_frames,
            audio_sha256=clip.audio_sha256,
            doa_track=track,
        )
    return out


def to_arrays(clip: DoaClip) -> tuple[np.ndarray, np.ndarray]:

    active_mask = np.zeros(clip.n_frames, dtype=bool)
    directions = np.zeros((clip.n_frames, 2), dtype=np.float64)
    for t, entry in enumerate(clip.doa_track):
        if entry is not None:
            active_mask[t] = True
            directions[t, 0] = entry[0]
            directions[t, 1] = entry[1]
    return active_mask, directions


def change_density(clips: Iterable[DoaClip]) -> float:

    total_changes = 0
    total_active = 0
    for clip in clips:
        total_changes += clip.n_changes
        total_active += clip.n_active_frames
    return total_changes / total_active if total_active > 0 else 0.0


def mean_change_jump_deg(clips: Iterable[DoaClip]) -> float:

    total = 0.0
    count = 0
    for clip in clips:
        track = clip.doa_track
        change_set = set(clip.change_frames)
        previous_active: tuple[float, float] | None = None
        for t, current in enumerate(track):
            if current is None:
                continue
            if t in change_set and previous_active is not None:
                total += great_circle_degrees(previous_active[0], previous_active[1], current[0], current[1])
                count += 1
            previous_active = current
    return total / count if count > 0 else 0.0


def doa_voc_targets_from_track(
    doa_track: Sequence[tuple[float, float] | None],
    *,
    angle_threshold_deg: float = DOA_CHANGE_THRESHOLD_DEG,
    window: int = DOA_VOC_WINDOW,
) -> np.ndarray:

    track = list(doa_track)
    n_frames = len(track)
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise DoaLabelRefusal("window must be a nonnegative integer")
    changes = _change_frames_for_track(track, angle_threshold_deg)
    targets = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        if any(abs(t - c) <= window for c in changes):
            targets[t] = 1.0
    return targets
