"""Direction-of-arrival bed, component 1: ground-truth per-frame DoA, geometry, and VoC targets.

This is a net-new, additive component for the STARSS23 direction-of-arrival (DoA) re-estimation bed. It
sits entirely beside the sealed onset-localization path and the sealed counting-bed modules (nothing
under referee.py stats.py controls.py harness.py gate.py prereg.py or count_*.py is edited). It reuses
the adapter's metadata parse path by import and adds a per-frame ground-truth direction track on top of
the frozen ``Clip`` identity, room, and audio digest.

Derivation
----------
STARSS23 native metadata carries a mandatory ``(azimuth, elevation)`` on every active-frame row
(``MetadataRow``), unlike ``distance`` which has an explicit absent sentinel. So, unlike source counting
or onset presence, ground-truth direction of arrival is available directly from the corpus at the full
100 ms frame rate, for the entire duration each source is active:

    A(t) == {}      (silence): no ground-truth direction is defined at t (excluded from all scoring).
    |A(t)| == 1: ground truth is that row's (azimuth, elevation).
    |A(t)| > 1  (polyphonic overlap): ground truth is the DOMINANT track's direction, using the identical
                tie-break the sealed adapter already applies to simultaneous onsets (nearest by distance,
                then class_id, then azimuth, then elevation, ascending), generalized from onset frames to
                every active frame. A polyphonic frame with an absent distance is refused, mirroring the
                adapter's existing refusal when distance is missing at a co-onset.

Geometry (``direction_to_unit_vector``, ``great_circle_degrees``, ``great_circle_degrees_batch``) is owned
here because it is label and ground-truth geometry, reused by the referee and the gate's value-of-
computation target. The great-circle distance is the standard DCASE SELD-style DOA error: a wraparound-
free spherical distance with no naive-difference artifacts at the +/-180 degree seam or near the poles,
for the same reason ``featurizer_spatial_doa.py`` already encodes direction cosines instead of raw
azimuth and elevation.

House style: no em dashes and no en dashes.
"""

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

# Preregistered DSP-prior-style constant, not label-tuned: the minimum great-circle jump between two
# consecutive active frames that counts as a direction "change" for the label-derived change track.
DOA_CHANGE_THRESHOLD_DEG = 5.0

# Value-of-computation training-target window: a re-estimation is valuable within +/- this many frames of
# a labeled direction change. Mirrors count_gate.COUNT_VOC_WINDOW's window-dilation discipline exactly.
DOA_VOC_WINDOW = 1


class DoaLabelRefusal(ValueError):
    """Raised when a DoA track cannot be derived or would violate the DoA label contract."""


def _require_n_frames(n_frames: int) -> int:
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames <= 0:
        raise DoaLabelRefusal("n_frames must be a positive integer")
    return n_frames


# ---------------------------------------------------------------------------
# Dominant-track tie-break and per-frame direction derivation.
# ---------------------------------------------------------------------------


def dominant_track_at_frame(rows: Sequence[MetadataRow]) -> MetadataRow:
    """Return the dominant row at one polyphonic frame: nearest by distance, then class_id, az, el.

    Mirrors ``adapter._prefer_onset``'s tie-break tuple ``(distance, class_id, azimuth, elevation)``
    ascending, generalized from co-onset frames to every polyphonic active frame. A polyphonic frame
    requires every row to carry STARSS23 distance, exactly as the sealed onset derivation requires it
    before it will break a co-onset tie; a row missing distance refuses the whole frame.
    """

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
    """Return the length-``n_frames`` ground-truth direction track: ``(azimuth_deg, elevation_deg)`` or
    ``None`` at a silent frame. Rows with ``frame >= n_frames`` are dropped, mirroring the whole-frame
    truncation provenance the real adapter already records.
    """

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
    """Parse native STARSS23 metadata text (reused ``parse_starss23_metadata``) into a DoA track."""

    return doa_track_from_rows(parse_starss23_metadata(text), n_frames)


# ---------------------------------------------------------------------------
# Geometry primitives: wraparound-free great-circle distance on the unit sphere.
# ---------------------------------------------------------------------------


def direction_to_unit_vector(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Convert an (azimuth, elevation) direction in degrees to a unit vector on the sphere."""

    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def great_circle_degrees(az1_deg: float, el1_deg: float, az2_deg: float, el2_deg: float) -> float:
    """Great-circle angular distance in degrees between two directions. Symmetric; 0 identical, 180 antipodal."""

    v1 = direction_to_unit_vector(az1_deg, el1_deg)
    v2 = direction_to_unit_vector(az2_deg, el2_deg)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def great_circle_degrees_batch(directions_a: np.ndarray, directions_b: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance. Each input is ``(N, 2)`` of ``(azimuth_deg, elevation_deg)``."""

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


# ---------------------------------------------------------------------------
# Change-frame rule, shared by DoaClip.change_frames and the VoC target derivation.
# ---------------------------------------------------------------------------


def _change_frames_for_track(
    track: Sequence[tuple[float, float] | None], angle_threshold_deg: float = DOA_CHANGE_THRESHOLD_DEG
) -> tuple[int, ...]:
    """Frame t is a change iff (a) it is the first active frame after silence or the clip start, or
    (b) t-1 and t are both active and their great-circle jump is at least ``angle_threshold_deg``."""

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


# ---------------------------------------------------------------------------
# The clip container.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoaClip:
    """One clip's ground-truth DoA track, keyed to the frozen ``Clip`` identity and audio digest."""

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
                raise DoaLabelRefusal("DoaClip.doa_track entries must be None or an (azimuth, elevation) pair")
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
    """Index metadata CSVs by clip stem with the exact ``rglob`` the adapter uses. Label-only."""

    root = Path(metadata_root)
    if not root.is_dir():
        raise DoaLabelRefusal(f"metadata root {root} is not a directory")
    index: dict[str, Path] = {}
    for meta_path in sorted(root.rglob("*.csv")):
        index.setdefault(meta_path.stem, meta_path)
    return index


def build_doa_clips(adapter: RealStarssAdapter, metadata_root: str | Path) -> dict[str, DoaClip]:
    """Derive a ``DoaClip`` for every clip the adapter serves. Mirrors ``count_labels.build_count_clips``."""

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
    """Return ``(active_mask, directions)``: a ``(n_frames,)`` bool mask and a ``(n_frames, 2)`` float64
    array of ``(azimuth_deg, elevation_deg)``, garbage (zero) where the mask is False."""

    active_mask = np.zeros(clip.n_frames, dtype=bool)
    directions = np.zeros((clip.n_frames, 2), dtype=np.float64)
    for t, entry in enumerate(clip.doa_track):
        if entry is not None:
            active_mask[t] = True
            directions[t, 0] = entry[0]
            directions[t, 1] = entry[1]
    return active_mask, directions


def change_density(clips: Iterable[DoaClip]) -> float:
    """Pooled ``n_changes / n_active_frames`` across clips. Label-only, mirrors count_labels.change_density."""

    total_changes = 0
    total_active = 0
    for clip in clips:
        total_changes += clip.n_changes
        total_active += clip.n_active_frames
    return total_changes / total_active if total_active > 0 else 0.0


def mean_change_jump_deg(clips: Iterable[DoaClip]) -> float:
    """Pooled mean great-circle jump between the direction just before and just after each change frame.

    A change frame caused by a silence gap contributes the jump from the nearest surrounding active frame
    (the last active direction seen before the gap); a clip-start change with no prior active frame in the
    clip is excluded, since there is no "jump size" for a first observation. Label-only, no arm score read.
    """

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
    """Value-of-computation targets: 1 within +/- ``window`` frames of a change_frames entry, else 0.

    Used only on TRAIN rooms, mirrors ``count_gate.voc_targets_from_count_track``'s window-dilation
    exactly. A pure function of ground truth; no arm score is read.
    """

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
